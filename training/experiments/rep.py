import copy
import json
import os
import re
from contextlib import nullcontext

import monai
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.utils.tensorboard
import wandb.sdk.wandb_run
from monai.visualize import plot_2d_or_3d_image
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.cuda import amp
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import wandb
from training.config.train import TrainConfig
from training.data.glioma_public import PublicGliomaDataset
from training.data.lmdb import CelebAlmdb, LMDBImageWriter
from training.data.mri import extract_slices_from_volume
from training.data.transforms import make_transform
from training.dist import get_rank, get_world_size
from training.loss.info_nce import info_nce_loss
from training.metrics.fid import evaluate_fid
from training.metrics.lpips import evaluate_lpips
from training.mode.optim import OptimizerType
from training.mode.train import TrainMode
from training.models.simsiam import SimSiam
from training.models.unet_autoenc import BeatGANsAutoencModel
from training.vis.renderer import render_condition, render_uncondition


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

            monai.utils.misc.set_determinism(seed=conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf
        self.model_conf = conf.make_model_conf()
        self.model = self.model_conf.make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print("Model params: %.2f M" % (model_size / 1024 / 1024))

        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # this is shared for both model and latent
        self.T_sampler = conf.make_T_sampler()

        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf().make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf().make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # initial variables for consistent sampling
        self.register_buffer("x_T", torch.randn(conf.sample_size, self.conf.in_channels, *(conf.img_size,) * conf.dims))

        if conf.pretrain is not None:
            print(f"loading pretrain ... {conf.pretrain.name}")
            state = torch.load(conf.pretrain.path, map_location="cpu")
            print("step:", state["global_step"])
            self.load_state_dict(state["state_dict"], strict=False)

        if self.conf.train_mode == TrainMode.simsiam:
            self.criterion = torch.nn.CosineSimilarity(dim=1)

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(self.device)
        return cond

    def sample(self, N, device, T=None, T_latent=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

        noise = torch.randn(N, 3, self.conf.img_size, self.conf.img_size, device=device)
        pred_img = render_uncondition(
            self.conf,
            self.ema_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=self.conds_mean,
            conds_std=self.conds_std,
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def render(self, noise, cond=None, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        if cond is not None:
            pred_img = render_condition(self.conf, self.ema_model, noise, sampler=sampler, cond=cond)
        else:
            pred_img = render_uncondition(self.conf, self.ema_model, noise, sampler=sampler, latent_sampler=None)
        return pred_img

    def encode(self, x):
        assert self.conf.model_type.has_autoenc()
        cond = self.ema_model.encoder.forward(x)
        return cond

    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(self.ema_model, x, model_kwargs={"cond": cond})
        return out["sample"]

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with amp.autocast(False):
            if ema_model:
                model = self.ema_model
            else:
                model = self.model
            gen = self.eval_sampler.sample(model=model, noise=noise, x_start=x_start)
            return gen

    def prepare_data(self):
        self.train_data = self.conf.make_dataset(split="train")
        self.val_data = self.conf.make_dataset(split="val")

        print("train data:", len(self.train_data))
        print("val data:", len(self.val_data))

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("local seed:", seed)
        ##############################################

        if self.conf.train_mode.require_dataset_infer():
            if self.conf.latent_infer_path is not None:
                print("loading latent stats ...")
                state = torch.load(self.conf.latent_infer_path)
                self.conds = state["conds"]
                self.register_buffer("conds_mean", state["conds_mean"][None, :])
                self.register_buffer("conds_std", state["conds_std"][None, :])
            else:
                self.conds_mean = None
                self.conds_std = None

                # usually we load self.conds from a file
                # so we do not need to do this again!
                self.conds = self.infer_whole_dataset()
                # need to use float32! unless the mean & std will be off!
                # (1, c)
                self.conds_mean = self.conds.float().mean(dim=0, keepdim=True)
                self.conds_std = self.conds.float().std(dim=0, keepdim=True)
                print("mean:", self.conds_mean.mean(), "std:", self.conds_std.mean())

    def train_dataloader(self):
        """
        return the dataloader, if diffusion mode => return image dataset
        if latent mode => return the inferred latent dataset
        """
        return self._shared_loader("train")

    def val_dataloader(self):
        """
        return the dataloader, if diffusion mode => return image dataset
        if latent mode => return the inferred latent dataset
        """
        return self._shared_loader("val")

    def _shared_loader(self, mode: str):
        print("on train dataloader start ...")
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        if self.conf.train_mode.require_dataset_infer():
            # return the dataset with pre-calculated conds
            loader_kwargs = dict(dataset=TensorDataset(self.conds), shuffle=True)
        else:
            loader_kwargs = dict(dataset=getattr(self, f"{mode}_data"), shuffle=mode == "train", drop_last=mode != "train")
        dataloader = conf.make_loader(batch_size=conf.batch_size, mode=mode, **loader_kwargs)

        return dataloader

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop?
        used with gradient_accum > 1 and to see if the optimizer will perform 'step' in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def infer_whole_dataset(self, split="train", with_render=False, T_render=None, render_save_path=None):
        """
        predicting the latents given images using the encoder

        Args:
            both_flips: include both original and flipped images; no need, it's not an improvement
            with_render: whether to also render the images corresponding to that latent
            render_save_path: lmdb output for the rendered images
        """
        data = self.conf.make_dataset(split=split)
        if isinstance(data, CelebAlmdb) and data.crop_d2c:
            # special case where we need the d2c crop
            data.transform = make_transform(self.conf.img_size, flip_prob=0, crop_d2c=True)
        elif isinstance(data, PublicGliomaDataset):
            data.view_transform = None

        else:
            data.transform = make_transform(self.conf.img_size, flip_prob=0)

        # data = SubsetDataset(data, 21)

        loader = self.conf.make_loader(
            data,
            shuffle=False,
            drop_last=False,
            batch_size=self.conf.batch_size_eval,
            parallel=True,
        )
        model = self.ema_model
        model.eval()
        conds = []

        if with_render:
            sampler = self.conf._make_diffusion_conf(T=T_render or self.conf.T_eval).make_sampler()

            if self.global_rank == 0:
                writer = LMDBImageWriter(render_save_path, format="webp", quality=100)
            else:
                writer = nullcontext()
        else:
            writer = nullcontext()

        with writer:
            for batch in tqdm(loader, total=len(loader), desc="infer"):
                with torch.no_grad():
                    # (n, c)
                    # print('idx:', batch['index'])
                    cond = model.encoder(batch["img"].to(self.device))

                    # used for reordering to match the original dataset
                    idx = batch["index"]
                    idx = self.all_gather(idx)
                    if idx.dim() == 2:
                        idx = idx.flatten(0, 1)
                    argsort = idx.argsort()

                    if with_render:
                        noise = torch.randn_like(batch["img"], device=self.device)
                        render = sampler.sample(model, noise=noise, cond=cond)
                        render = (render + 1) / 2
                        # print('render:', render.shape)
                        # (k, n, c, h, w)
                        render = self.all_gather(render)
                        if render.dim() == 5:
                            # (k*n, c)
                            render = render.flatten(0, 1)

                        if self.global_rank == 0:
                            writer.put_images(render[argsort])

                    # (k, n, c)
                    cond = self.all_gather(cond)

                    if cond.dim() == 3:
                        # (k*n, c)
                        cond = cond.flatten(0, 1)

                    conds.append(cond[argsort].cpu().detach())
                # break
        model.train()
        # (N, c) cpu

        conds = torch.cat(conds).float()
        return conds

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._shared_step(batch, batch_idx, "val")

    def _shared_step(self, batch, batch_idx, step_mode: str):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with amp.autocast(False):
            # batch size here is local!
            # forward
            losses = {}

            if self.conf.train_mode == TrainMode.diffusion:
                """
                main training mode!!!
                """
                imgs, idxs = batch["img"], batch["index"]
                x_start = imgs
                # with numpy seed we have the problem that the sample t's are related!
                t, weight = self.T_sampler.sample(len(x_start), x_start.device)

                model_kwargs = {}
                if "img_aug" in batch:
                    model_kwargs = dict(x_start_aug=batch["img_aug"])

                losses = self.sampler.training_losses(model=self.model, x_start=x_start, t=t, model_kwargs=model_kwargs)
            elif self.conf.train_mode.is_latent_diffusion():
                """
                training the latent variables!
                """
                cond = batch[0]
                if self.conf.latent_znormalize:
                    cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)
                # diffusion on the latent
                t, weight = self.T_sampler.sample(len(cond), cond.device)
                latent_losses = self.latent_sampler.training_losses(model=self.model.latent_net, x_start=cond, t=t)
                # train only do the latent diffusion
                losses = {"latent": latent_losses["loss"], "loss": latent_losses["loss"]}
            elif self.conf.train_mode == TrainMode.simclr:
                images = torch.cat([b["img"] for b in batch], dim=0)

                features_list = []
                batch_size_gpu = self.conf.sample_size

                for img_chunk in torch.split(images, batch_size_gpu):
                    cur_feats = self.model(img_chunk)
                    features_list.append(cur_feats)

                features = torch.cat(features_list, dim=0)

                loss, accuracy = info_nce_loss(features)
                losses = {}
                losses["info_nce"] = loss.detach()
                losses["loss"] = loss
                losses["sim_accuracy"] = accuracy.detach()
            elif self.conf.train_mode == TrainMode.simsiam:
                self.model: SimSiam
                images = torch.stack([b["img"] for b in batch], dim=0)

                # compute output and loss
                p1, p2, z1, z2 = self.model(x1=images[0], x2=images[1])
                loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
                losses["loss"] = loss

            else:
                raise NotImplementedError()

            loss = losses["loss"].mean()
            losses = {k: v.detach() if k != "loss" else v for k, v in losses.items()}

            loss_keys = [*filter(lambda k: k in losses, ["loss", "vae", "latent", "mmd", "chamfer", "arg_cnt", "info_nce", "sim_accuracy"])]
            # divide by accum batches to make the accumulated gradient exact!
            for key in loss_keys:
                losses[key] = self.all_gather(losses[key]).mean()

            for key in loss_keys:
                self.log(f"loss/{step_mode}_{key}", losses[key].item(), rank_zero_only=True)
            # if isinstance(batch, dict):
            #    cur_class_labels = batch.get("og_cls_labels", batch["cls_labels"])
            # elif isinstance(batch, list):
            #    cur_class_labels = batch[0].get("og_cls_labels", batch[0]["cls_labels"])

            # get the histogram of the labels in the batch
            # class_range = torch.arange(getattr(self, f"{step_mode}_data").num_classes).to(cur_class_labels)
            # all_labels_hist = (class_range == cur_class_labels[:, None]).sum(dim=0)
            # bins = torch.arange(0, getattr(self, f"{step_mode}_data").num_classes + 1) - 0.5
            # log the histogram
            # self.log_histogram("labels_distribution", (all_labels_hist, bins), step=self.global_step)

        return losses

    def on_train_start(self):
        super().on_train_start()
        early_stopping = next(c for c in self.trainer.callbacks if isinstance(c, EarlyStopping))
        early_stopping.patience = self.conf.clf_early_stopping_patience

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            if self.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(self.model.latent_net, self.ema_model.latent_net, self.conf.ema_decay)
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

            # logging
            if self.conf.train_mode.require_dataset_infer():
                imgs = None
            elif isinstance(batch, list):
                imgs = torch.cat([b["img"] for b in batch], dim=0)
            else:
                imgs = batch["img"]

            if not self.trainer.fast_dev_run:
                if self.conf.train_mode.is_diffusion():
                    self.log_sample(x_start=imgs)
                # self.evaluate_scores()

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [p for group in optimizer.param_groups for p in group["params"]]
            # print('before:', grads_norm(iter_opt_params(optimizer)))
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.conf.grad_clip)
            # print('after:', grads_norm(iter_opt_params(optimizer)))

    def log_sample(self, x_start):
        """
        put images to the tensorboard
        """

        def do(model, postfix, use_xstart, save_real=False, no_latent_diff=False, interpolate=False):
            model.eval()
            with torch.no_grad():
                all_x_T = self.split_tensor(self.x_T)
                batch_size = min(len(all_x_T), self.conf.batch_size_eval)
                # allow for superlarge models
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen = []
                for x_T in loader:
                    if use_xstart:
                        _xstart = x_start[: len(x_T)]
                    else:
                        _xstart = None

                    if self.conf.train_mode.is_latent_diffusion() and not use_xstart:
                        # diffusion of the latent first
                        gen = render_uncondition(
                            conf=self.conf,
                            model=model,
                            x_T=x_T,
                            sampler=self.eval_sampler,
                            latent_sampler=self.eval_latent_sampler,
                            conds_mean=self.conds_mean,
                            conds_std=self.conds_std,
                        )
                    else:
                        if not use_xstart and self.conf.model_type.has_noise_to_cond():
                            model: BeatGANsAutoencModel
                            # special case, it may not be stochastic, yet can sample
                            cond = torch.randn(len(x_T), self.conf.style_ch, device=self.device)
                            cond = model.noise_to_cond(cond)
                        else:
                            if interpolate:
                                with amp.autocast(self.conf.fp16):
                                    cond = model.encoder(_xstart)
                                    i = torch.randperm(len(cond))
                                    cond = (cond + cond[i]) / 2
                            else:
                                cond = None
                        gen = self.eval_sampler.sample(model=model, noise=x_T, cond=cond, x_start=_xstart)
                    Gen.append(gen)

                gen = torch.cat(Gen)
                gen = self.all_gather(gen)
                if (gen.dim() - self.conf.dims) == 3:
                    # collect tensors from different workers
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)
                if self.conf.dims == 3:
                    if self.global_rank == 0:
                        # TODO: convert gif to mp4 to save it to wandb
                        # if logged to tensorboard
                        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
                            plot_2d_or_3d_image(gen, self.global_step, self.logger.experiment, tag=f"sample{postfix}/fake", frame_dim=-1)
                        elif isinstance(self.logger, pl_loggers.WandbLogger):
                            ...

                    gen = extract_slices_from_volume(gen)

                sample_dir = os.path.join(self.conf.logdir, f"sample{postfix}")
                if self.global_rank == 0 and not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)
                grid_params = lambda t: dict(
                    tensor=t, nrow=3 if self.conf.dims == 3 else int(np.sqrt(t.size(0))), normalize=True, padding=0, value_range=(-1, 1)
                )
                if save_real and use_xstart:
                    # save the original images to the tensorboard
                    real = self.all_gather(_xstart)
                    if (real.dim() - self.conf.dims) == 3:
                        real = real.flatten(0, 1)

                    if self.conf.dims == 3:
                        # visualize volume using MONAI
                        if self.global_rank == 0:
                            if isinstance(self.logger, pl_loggers.TensorBoardLogger):
                                plot_2d_or_3d_image(
                                    real, self.global_step, self.logger.experiment, tag=f"sample{postfix}/real", frame_dim=-1
                                )
                            elif isinstance(self.logger, pl_loggers.WandbLogger):
                                # log as 3d object
                                # TODO: add rendering as mesh
                                ...

                        # extract 2d slice from different sequences
                        real = extract_slices_from_volume(real)
                    if self.global_rank == 0:
                        real_grid = make_grid(**grid_params(real))
                        self.log_image(f"sample{postfix}/real", real_grid, self.global_step)

                        path = os.path.join(sample_dir, "real.png")
                        save_image(real_grid, path)

                if self.global_rank == 0:
                    # save samples to the tensorboard

                    gen_grid = make_grid(**grid_params(gen))
                    path = os.path.join(sample_dir, f"{self.global_step}.png")
                    save_image(gen_grid, path)
                    self.log_image(f"sample{postfix}/fake", gen_grid, self.global_step)
            model.train()

        if self.conf.sample_every_samples > 0 and is_time(self.num_samples, self.conf.sample_every_samples, self.conf.batch_size_effective):
            if self.conf.train_mode.require_dataset_infer():
                do(self.model, "", use_xstart=False)
                do(self.ema_model, "_ema", use_xstart=False)
            else:
                if self.conf.model_type.has_autoenc() and self.conf.model_type.can_sample():
                    do(self.model, "", use_xstart=False)
                    do(self.ema_model, "_ema", use_xstart=False)
                    # autoencoding mode
                    do(self.model, "_enc", use_xstart=True, save_real=True)
                    do(self.ema_model, "_enc_ema", use_xstart=True, save_real=True)
                elif self.conf.train_mode.use_latent_net():
                    do(self.model, "", use_xstart=False)
                    do(self.ema_model, "_ema", use_xstart=False)
                    # autoencoding mode
                    do(self.model, "_enc", use_xstart=True, save_real=True)
                    do(self.model, "_enc_nodiff", use_xstart=True, save_real=True, no_latent_diff=True)
                    do(self.ema_model, "_enc_ema", use_xstart=True, save_real=True)
                else:
                    do(self.model, "", use_xstart=True, save_real=True)
                    do(self.ema_model, "_ema", use_xstart=True, save_real=True)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        if self.global_rank == 0:
            experiment = self.logger.experiment
            if isinstance(experiment, torch.utils.tensorboard.SummaryWriter):
                experiment.add_image(tag, image, step)
            elif isinstance(experiment, wandb.sdk.wandb_run.Run):
                experiment.log(
                    {tag: [wandb.Image(image.cpu())]},
                    # step=step,
                )

    def log_histogram(self, tag: str, np_histogram: torch.Tensor, step: int = None) -> None:
        if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            np_histogram = tuple([h.cpu().numpy() for h in np_histogram])
            np_histogram = wandb.Histogram(np_histogram=np_histogram, num_bins=53)
            self.logger.experiment.log({tag: np_histogram}, step=step)

    def evaluate_scores(self):
        """
        evaluate FID and other scores during training (put to the tensorboard)
        For, FID. It is a fast version with 5k images (gold standard is 50k).
        Don't use its results in the paper!
        """

        def fid(model, postfix):
            score = evaluate_fid(
                self.eval_sampler,
                model,
                self.conf,
                device=self.device,
                train_data=self.train_data,
                val_data=self.val_data,
                latent_sampler=self.eval_latent_sampler,
                conds_mean=self.conds_mean,
                conds_std=self.conds_std,
            )
            if self.global_rank == 0:
                self.log(f"FID{postfix}", score)
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, "eval.txt"), "a") as f:
                    metrics = {
                        f"FID{postfix}": score,
                        "num_samples": self.num_samples,
                        "step": self.global_step,
                    }
                    f.write(json.dumps(metrics) + "\n")

        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc() and self.conf.train_mode.is_autoenc():
                # {'lpips', 'ssim', 'mse'}
                score = evaluate_lpips(
                    self.eval_sampler, model, self.conf, device=self.device, val_data=self.val_data, latent_sampler=self.eval_latent_sampler
                )

                if self.global_rank == 0:
                    for key, val in score.items():
                        self.log(f"{key}{postfix}", val.item())

        if (
            self.conf.eval_every_samples > 0
            and self.num_samples > 0
            and is_time(self.num_samples, self.conf.eval_every_samples, self.conf.batch_size_effective)
        ):
            print(f"eval fid @ {self.num_samples}")
            lpips(self.model, "")
            if self.conf.dims == 2:
                fid(self.model, "")

        if (
            self.conf.eval_ema_every_samples > 0
            and self.num_samples > 0
            and is_time(self.num_samples, self.conf.eval_ema_every_samples, self.conf.batch_size_effective)
        ):
            if self.conf.dims == 2:
                print(f"eval fid ema @ {self.num_samples}")
                fid(self.ema_model, "_ema")
            # it's too slow
            # lpips(self.ema_model, '_ema')

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)

        elif self.conf.optimizer == OptimizerType.sgd:
            init_lr = self.conf.lr * self.conf.batch_size / 256

            if self.conf.fix_pred_lr:
                optim_params = [
                    {"params": self.model.module.encoder.parameters(), "fix_lr": False},
                    {"params": self.model.module.predictor.parameters(), "fix_lr": True},
                ]
            else:
                optim_params = self.model.parameters()
            optim = [torch.optim.SGD(o, init_lr, momentum=self.conf.momentum, weight_decay=self.conf.weight_decay) for o in optim_params]

        else:
            raise NotImplementedError()
        out["optimizer"] = optim

        lr_sched_name = self.model_conf.lr_sched_name
        if self.conf.warmup > 0:
            out["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=WarmupLR(self.conf.warmup)),
                "interval": "step",
            }
        elif lr_sched_name == "cosine":
            out["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim[1] if isinstance(optim, list) else optim,
                    T_max=self.model_conf.T_max,
                    eta_min=self.model_conf.eta_min,
                    last_epoch=self.model_conf.last_epoch,
                ),
                "interval": "epoch",
                "frequency": 10,
            }

        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding 'worker' in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank : (rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        for the 'eval' mode.
        We first select what to do according to the 'conf.eval_programs'.
        test_step will only run for 'one iteration' (it's a hack!).

        We just want the multi-gpu support.
        """
        # make sure you seed each worker differently!
        self.setup()

        # it will run only one step!
        print("global step:", self.global_step)
        """
        'infer' = predict the latent variables using the encoder on the whole dataset
        """
        if "infer" in self.conf.eval_programs:
            print("infer ...")
            conds = self.infer_whole_dataset(split="train").float()
            # NOTE: always use this path for the latent.pkl files
            save_path = f'{self.conf.logdir.replace("_debug","")}/latent.pkl'

            if self.global_rank == 0:
                conds_mean = conds.mean(dim=0)
                conds_std = conds.std(dim=0)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(
                    {
                        "conds": conds,
                        "conds_mean": conds_mean,
                        "conds_std": conds_std,
                    },
                    save_path,
                )
        """
        'infer+render' = predict the latent variables using the encoder on the whole dataset
        THIS ALSO GENERATE CORRESPONDING IMAGES
        """
        # infer + reconstruction quality of the input
        for each in self.conf.eval_programs:
            if each.startswith("infer+render"):
                m = re.match(r"infer\+render([0-9]+)", each)
                if m is not None:
                    T = int(m[1])
                    self.setup()
                    print(f"infer + reconstruction T{T} ...")
                    conds = self.infer_whole_dataset(
                        with_render=True,
                        T_render=T,
                        render_save_path=f"latent_infer_render{T}/{self.conf.name}.lmdb",
                    )
                    save_path = f"latent_infer_render{T}/{self.conf.name}.pkl"
                    conds_mean = conds.mean(dim=0)
                    conds_std = conds.std(dim=0)
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    torch.save(
                        {
                            "conds": conds,
                            "conds_mean": conds_mean,
                            "conds_std": conds_std,
                        },
                        save_path,
                    )

        # evals those 'fidXX'
        """
        'fid<T>' = unconditional generation (conf.train_mode = diffusion).
            Note:   Diff. autoenc will still receive real images in this mode.
        'fid<T>,<T_latent>' = unconditional generation for latent models (conf.train_mode = latent_diffusion).
            Note:   Diff. autoenc will still NOT receive real images in this made.
                    but you need to make sure that the train_mode is latent_diffusion.
        """
        for each in self.conf.eval_programs:
            if each.startswith("fid"):
                m = re.match(r"fid\(([0-9]+),([0-9]+)\)", each)
                clip_latent_noise = False
                if m is not None:
                    # eval(T1,T2)
                    T = int(m[1])
                    T_latent = int(m[2])
                    print(f"evaluating FID T = {T}... latent T = {T_latent}")
                else:
                    m = re.match(r"fidclip\(([0-9]+),([0-9]+)\)", each)
                    if m is not None:
                        # fidclip(T1,T2)
                        T = int(m[1])
                        T_latent = int(m[2])
                        clip_latent_noise = True
                        print(f"evaluating FID (clip latent noise) T = {T}... latent T = {T_latent}")
                    else:
                        # evalT
                        _, T = each.split("fid")
                        T = int(T)
                        T_latent = None
                        print(f"evaluating FID T = {T}...")

                self.train_dataloader()
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
                if T_latent is not None:
                    latent_sampler = self.conf._make_latent_diffusion_conf(T=T_latent).make_sampler()
                else:
                    latent_sampler = None

                conf = self.conf.clone()
                conf.eval_num_images = 50_000
                score = evaluate_fid(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    train_data=self.train_data,
                    val_data=self.val_data,
                    latent_sampler=latent_sampler,
                    conds_mean=self.conds_mean,
                    conds_std=self.conds_std,
                    remove_cache=False,
                    clip_latent_noise=clip_latent_noise,
                )
                if T_latent is None:
                    self.log(f"fid_ema_T{T}", score.item())
                else:
                    name = "fid"
                    if clip_latent_noise:
                        name += "_clip"
                    name += f"_ema_T{T}_Tlatent{T_latent}"
                    self.log(name, score.item())
        """
        'recon<T>' = reconstruction & autoencoding (without noise inversion)
        """
        for each in self.conf.eval_programs:
            if each.startswith("recon"):
                self.model: BeatGANsAutoencModel
                _, T = each.split("recon")
                T = int(T)
                print(f"evaluating reconstruction T = {T}...")

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = evaluate_lpips(
                    sampler,
                    self.model,
                    #    self.ema_model,
                    conf,
                    device=self.device,
                    val_data=self.val_data,
                    latent_sampler=None,
                )
                for k, v in score.items():
                    self.log(f"{k}_ema_T{T}", v.item())
        """
        'inv<T>' = reconstruction with noise inversion
        """
        for each in self.conf.eval_programs:
            if each.startswith("inv"):
                self.model: BeatGANsAutoencModel
                _, T = each.split("inv")
                T = int(T)
                print(f"evaluating reconstruction with noise inversion T = {T}...")

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = evaluate_lpips(
                    sampler, self.ema_model, conf, device=self.device, val_data=self.val_data, latent_sampler=None, use_inverted_noise=True
                )
                for k, v in score.items():
                    self.log(f"{k}_inv_ema_T{T}", v.item())


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay), non_blocking=True)


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


def train(conf: TrainConfig, gpus, nodes=1, mode: str = "train"):
    print("conf:", conf.name, "version:", conf.version)

    # assert not (conf.fp16 and conf.grad_clip > 0
    #             ), 'pytorch lightning has bug with amp + gradient clipping'
    model = LitModel(conf)
    # model = torch.compile(model)  # , mode="reduce-overhead")

    if not os.path.exists(conf.logdir) and get_rank() == 0:
        os.makedirs(conf.logdir)

    monitor_str = "loss/val_loss"

    checkpoint = ModelCheckpoint(
        monitor=monitor_str,
        mode="min",
        dirpath=f"{conf.logdir}",
        save_last=True,
        save_top_k=1,
        verbose=True,
        auto_insert_metric_name=True,
        every_n_train_steps=conf.save_every_samples // conf.batch_size_effective,
    )

    early_stopping = EarlyStopping(monitor=monitor_str, mode="min", verbose=True, patience=conf.clf_early_stopping_patience)

    checkpoint_path = f"{conf.logdir}/last.ckpt"
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print(f"Resuming from {resume}")
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path

        else:
            resume = None

    wandb_resume_kwargs = {}
    wandb_id_fp = f"{conf.logdir}/wandb_id.txt"
    if os.path.exists(wandb_id_fp):
        # read wandb id
        wandb_resume_kwargs = {"resume": "must"}
        # load wandb id
        with open(wandb_id_fp, "r") as f:
            conf.wandb_id = f.read().strip()
    else:
        # save wandb id to checkpoints dir
        with open(wandb_id_fp, "w") as f:
            f.write(conf.wandb_id)

    wandb_logger = pl_loggers.WandbLogger(
        project=conf.wandb_project,
        config=conf.as_dict_jsonable(),
        save_code=False,
        log_model=False,
        job_type="rep",
        id=conf.wandb_id,
        **wandb_resume_kwargs,
    )

    plugins = []
    if len(gpus) == 0:
        accelerator = "auto"
    elif len(gpus) == 1 and nodes == 1:
        accelerator = "cuda"
    else:
        accelerator = "ddp"
        from pytorch_lightning.plugins import DDPPlugin

        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))
    n_overfit_batches = 1 if conf.overfit else 0.0
    # TODO: make depdendent on dataset
    log_every_n_steps = 1 if conf.overfit else 3000 // conf.batch_size_effective

    # profiler = SimpleProfiler(dirpath='.', filename='perf_logs_simple')
    # profiler = AdvancedProfiler(dirpath='.', filename='perf_logs')
    fast_dev_run = "debug" in conf.name
    if not fast_dev_run:
        _ = wandb_logger.experiment  # init wandb

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        devices=gpus,
        num_nodes=nodes,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            early_stopping,
            # LearningRateMonitor(),
        ],
        # clip in the model instead
        # gradient_clip_val=conf.grad_clip,
        # replace_sampler_ddp=True,
        logger=wandb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
        log_every_n_steps=log_every_n_steps,
        overfit_batches=n_overfit_batches,
        fast_dev_run=fast_dev_run,
        # benchmark=True,
        # profiler=profiler
    )

    if mode == "train":
        trainer.fit(model, ckpt_path=resume)
    elif mode == "eval":
        # load the latest checkpoint
        # perform lpips
        # dummy loader to allow calling 'test_step'
        dummy = DataLoader(TensorDataset(torch.tensor([0.0] * conf.batch_size)), batch_size=conf.batch_size)
        eval_path = conf.eval_path or checkpoint_path
        # conf.eval_num_images = 50
        print("loading from:", eval_path)
        state = torch.load(eval_path, map_location="cpu")
        print("step:", state["global_step"])
        model.load_state_dict(state["state_dict"])
        # trainer.fit(model)
        out = trainer.test(model, dataloaders=dummy)
        if len(out) == 0:
            # no results where returned
            return
        # first (and only) loader
        out = out[0]
        print(out)

        if get_rank() == 0:
            # save to tensorboard
            for k, v in out.items():
                model.log(k, v)

            # # save to file
            # # make it a dict of list
            # for k, v in out.items():
            #     out[k] = [v]
            tgt = f"evals/{conf.name}.txt"
            dirname = os.path.dirname(tgt)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(tgt, "a") as f:
                f.write(json.dumps(out) + "\n")
            # pd.DataFrame(out).to_csv(tgt)
    else:
        raise NotImplementedError()
