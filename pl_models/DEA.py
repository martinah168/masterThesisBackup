import copy
import json
import os
from pathlib import Path
import random
import re
from contextlib import nullcontext
import sys
from typing import Literal
from diffusion.beta_schedule import ScheduleSampler

from diffusion.ddim_sampler import get_sampler
from diffusion.renderer import render_condition, render_uncondition
from models import Model
from utils.enums_model import OptimizerType
from utils.mri import extract_slices_from_volume
from utils.hessian_penalty_pytorch import hessian_penalty
from config import *

from utils.metrics import *

sys.path.append("..")
import monai
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.utils.tensorboard
from monai.visualize import plot_2d_or_3d_image
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from .pl_utils.dist import get_rank, get_world_size
from monai.utils.misc import set_determinism
from models.model_factory import get_model
from dataloader.dataset_factory import get_dataset, get_data_loader
from utils.arguments import DAE_Option

from utils.enums import TrainMode
from tensorboard.plugins import projector
config = projector.ProjectorConfig()

# embedding = config.embeddings.add()
# embedding.tensor_name = h.name

# # Use the same LOG_DIR where you stored your checkpoint.
# summary_writer = tf.summary.FileWriter(LOG_DIR)

# projector.visualize_embeddings(summary_writer, config)

class DAE_LitModel(pl.LightningModule):
    ###### INIT PROCESS ######
    def __init__(self, conf: DAE_Option):
        super().__init__()
        assert conf.train_mode.value != TrainMode.manipulate.value
        if conf.seed is not None:
            pl.seed_everything(conf.seed)
            set_determinism(seed=conf.seed)

        self.save_hyperparameters()

        self.conf = conf
        self.model: Model = get_model(conf)
        self.ema_model: Model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        self.last_100_loss = deque()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print("Model params: %.2f M" % (model_size / 1024 / 1024))

        self.sampler = get_sampler(conf, eval=False)
        self.eval_sampler = get_sampler(conf, eval=True)

        # this is shared for both model and latent
        self.T_sampler = ScheduleSampler(conf.num_timesteps)

        if conf.train_mode.use_latent_net():
            raise NotImplementedError("use_latent_net")
            self.latent_sampler = conf.make_latent_diffusion_conf().make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf().make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # initial variables for consistent sampling
        self.register_buffer("x_T", torch.randn(conf.sample_size, self.conf.in_channels, *conf.shape))

        if conf.pretrain is not None:
            raise NotImplementedError("pretrained")
            print(f"loading pretrain ... {conf.pretrain.name}")
            state = torch.load(conf.pretrain.path, map_location="cpu")
            print("step:", state["global_step"])
            self.load_state_dict(state["state_dict"], strict=False)

        if self.conf.train_mode == TrainMode.simsiam:
            self.criterion = torch.nn.CosineSimilarity(dim=1)
        # Pytorch Lightning calls the following things.
        # self.prepare_data()
        # self.setup(stage)
        # self.train_dataloader()
        # self.val_dataloader()
        # self.test_dataloader()
        # self.predict_dataloader()

    def prepare_data(self):
        self.train_data = get_dataset(self.conf, split="train")
        self.val_data = get_dataset(self.conf, split="val")

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
            raise NotImplementedError("latent_infer_path")
            if self.conf.latent_infer_path is not None:
                print("loading latent stats ...")
                state = torch.load(self.conf.latent_infer_path)
                self.conds = state["conds"]
                self.register_buffer("conds_mean", state["conds_mean"][None, :])
                self.register_buffer("conds_std", state["conds_std"][None, :])
            else:
                # usually we load self.conds from a file
                # so we do not need to do this again!
                self.conds = self.infer_whole_dataset()
                # need to use float32! unless the mean & std will be off!
                # (1, c)
                self.conds_mean = self.conds.float().mean(dim=0, keepdim=True)
                self.conds_std = self.conds.float().std(dim=0, keepdim=True)
                print("mean:", self.conds_mean.mean(), "std:", self.conds_std.mean())

    ####### DATA LOADER #######
    def train_dataloader(self):
        return self._shared_loader("train")

    def val_dataloader(self):
        return self._shared_loader("val")

    def _shared_loader(self, mode: Literal["train", "val"], super_res=False):
        opt = self.conf
        print(f"on {mode} dataloader start ...")
        if self.conf.train_mode.require_dataset_infer():
            raise NotImplementedError("latent_infer_path")
            # return the dataset with pre-calculated conds
            loader_kwargs = dict(dataset=TensorDataset(self.conds), shuffle=True)
            return get_data_loader()  # TODO
        else:
            train = mode == "train"
            return get_data_loader(opt, self.train_data if train else self.val_data, shuffle=train, drop_last=not train)

    ###### Training #####
    def forward(self, noise=None, x_start=None, ema_model: bool = False, **qargs):
        with autocast(False):
            model = self.ema_model if ema_model else self.model
            return self.eval_sampler.sample(model=model, noise=noise, x_start=x_start)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        self.last_100_loss.append(loss["loss"].detach().cpu().numpy())
        if len(self.last_100_loss) == 101:
            self.last_100_loss.popleft()
        self.log("train/avg_loss", value=np.mean(np.array(self.last_100_loss)).item(), prog_bar=True)
        
#        self.log_image(tag="img_log", image=batch["img"],step=self.global_step)
        #plot_2d_or_3d_image(data=batch["img"], step=self.global_step, writer=self.logger.experiment, frame_dim=-1, tag="3Dimage")
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss =  self._shared_step(batch, batch_idx, "val")
        #print(loss)
        #TODO: log val loss
        #loss = self._shared_step(batch, batch_idx, "val")
        #self.log("loss/val_loss", loss["loss"], prog_bar=True)  # Log validation loss
        # Log any other validation metrics if needed
        return loss

    def _shared_step(self, batch, batch_idx, step_mode: str):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with autocast(False):
            losses = {}
            if self.conf.train_mode == TrainMode.diffusion:
                """
                main training mode!!!
                """
                
                imgs = batch["img"]
                x_start = imgs
                # with numpy seed we have the problem that the sample t's are related!
                t = self.T_sampler.sample(len(x_start), x_start.device)
                model_kwargs = {}
                if "img_aug" in batch:
                    model_kwargs = dict(x_start_aug=batch["img_aug"])
                losses = self.sampler.training_losses(model=self.model, x_start=x_start, t=t, model_kwargs=model_kwargs)

                # hessian_penalty
                if self.conf.hessian_penalty != 0:
                    # https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510579.pdf
                    assert "cond_emb" in losses
                    hl = hessian_penalty(self.model.encoder, x_start, G_z=losses["cond_emb"])
                    losses["hessian_penalty"] = hl.detach()
                    losses["loss"] = losses["loss"] + hl

            elif self.conf.train_mode.is_latent_diffusion():
                raise NotImplementedError("latent_infer_path")
                """
                training the latent variables!
                """
                cond = batch[0]
                if self.conf.latent_znormalize:
                    cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)
                # diffusion on the latent
                t = self.T_sampler.sample(len(cond), cond.device)
                latent_losses = self.latent_sampler.training_losses(model=self.model.latent_net, x_start=cond, t=t)
                # train only do the latent diffusion
                losses = {"latent": latent_losses["loss"], "loss": latent_losses["loss"]}
            else:
                raise NotImplementedError()
            losses = {k: v.detach() if k != "loss" else v for k, v in losses.items()}
            loss_keys = [
                *filter(
                    lambda k: k in losses,
                    ["loss", "vae", "latent", "mmd", "chamfer", "arg_cnt", "info_nce", "sim_accuracy", "hessian_penalty"],
                )
            ]
            # divide by accum batches to make the accumulated gradient exact!
            #self.evaluate_scores()
            #it = losses["cond_emb"]
            #self.logger.experiment.add_embedding(it, global_step=self.global_step)
            #self.logger.experiment.add_scalar("loss/{step_mode}_{key}", losses["loss"], global_step = self.global_step)
            for key in loss_keys:
                losses[key] = self.all_gather(losses[key]).mean()  # type: ignore
            for key in loss_keys:
                self.log(f"loss/{step_mode}_{key}", losses[key].item(), rank_zero_only=True)
                
        return losses

    def on_train_start(self):
        super().on_train_start()
        early_stopping = next(c for c in self.trainer.callbacks if isinstance(c, EarlyStopping))  # type: ignore
        early_stopping.patience = self.conf.early_stopping_patience

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop?
        used with gradient_accum > 1 and to see if the optimizer will perform 'step' in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
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
            imgs_lr = None
            # logging
            if self.conf.train_mode.require_dataset_infer():
                imgs = None
            elif isinstance(batch, list):
                imgs = torch.cat([b["img"] for b in batch], dim=0)
            elif "img_lr" in batch:
                imgs_lr = batch["img_lr"]
                imgs = batch["img"]
            else:
                imgs = batch["img"]

            if not self.trainer.fast_dev_run:  # type: ignore
                if self.conf.train_mode.is_diffusion():
                    self.log_sample(x_start=imgs, x_start_lr=imgs_lr)
            #self.evaluate_scores()

    #### Optimizer ####
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        grad_clip = self.conf.grad_clip
        if grad_clip > 0:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            clip_grad_norm_(params, max_norm=grad_clip)

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamW:
            optim = torch.optim.AdamW(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out["optimizer"] = optim

        # lr_sched_name = self.model_conf.lr_sched_name
        # if self.conf.warmup > 0:
        #    out["lr_scheduler"] = {
        #        "scheduler": torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=WarmupLR(self.conf.warmup)),
        #        "interval": "step",
        #    }
        # elif lr_sched_name == "cosine":
        #    out["lr_scheduler"] = {
        #        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
        #            optim[1] if isinstance(optim, list) else optim,
        #            T_max=self.model_conf.T_max,
        #            eta_min=self.model_conf.eta_min,
        #            last_epoch=self.model_conf.last_epoch,
        #        ),
        #        "interval": "epoch",
        #        "frequency": 10,
        #    }

        return out

    ###### Utils ######

    # def normalize(self, cond):
    #    cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)
    #    return cond

    # def denormalize(self, cond):
    #    cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(self.device)
    #    return cond

    # def sample(self, N, device, T=None, T_latent=None):
    #    if T is None:
    #        sampler = self.eval_sampler
    #        latent_sampler = self.latent_sampler
    #    else:
    #        sampler = self.conf._make_diffusion_conf(T).make_sampler()
    #        latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

    #    noise = torch.randn(N, 3, self.conf.img_size, self.conf.img_size, device=device)
    #    pred_img = render_uncondition(
    #        self.conf,
    #        self.ema_model,
    #        noise,
    #        sampler=sampler,
    #        latent_sampler=latent_sampler,
    #        conds_mean=self.conds_mean,
    #        conds_std=self.conds_std,
    #    )
    #    pred_img = (pred_img + 1) / 2
    #    return pred_img

    def render(self, noise, cond=None, T=None, x_start=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = get_sampler(self.conf, eval=False, T=T)

        if cond is not None:
            pred_img = render_condition(self.conf, self.ema_model, noise, sampler=sampler, cond=cond, x_start=x_start)
        else:
            pred_img = render_uncondition(self.conf, self.ema_model, noise, sampler=sampler, latent_sampler=None)
        return pred_img

    def encode(self, x):
        assert self.conf.model_type.has_autoenc()
        cond = self.model.encoder.forward(x)
        return cond

    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = get_sampler(self.conf, eval=False, T=T)

        out = sampler.ddim_reverse_sample_loop(self.ema_model, x, model_kwargs={"cond": cond})
        return out["sample"]

    # @property
    # def batch_size(self):
    #    """
    #    local batch size for each worker
    #    """
    #    ws = get_world_size()
    #    assert self.conf.batch_size % ws == 0
    #    return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    # def infer_whole_dataset(self, split="train", with_render=False, T_render=None, render_save_path=None):
    #    """
    #    predicting the latents given images using the encoder

    #    Args:
    #        both_flips: include both original and flipped images; no need, it's not an improvement
    #        with_render: whether to also render the images corresponding to that latent
    #        render_save_path: lmdb output for the rendered images
    #    """
    #    data = self.conf.make_dataset(split=split)
    #    if isinstance(data, CelebAlmdb) and data.crop_d2c:
    #        # special case where we need the d2c crop
    #        data.transform = make_transform(self.conf.img_size, flip_prob=0, crop_d2c=True)
    #    elif isinstance(data, PublicGliomaDataset):
    #        data.view_transform = None

    #    else:
    #        data.transform = make_transform(self.conf.img_size, flip_prob=0)

    #    # data = SubsetDataset(data, 21)

    #    loader = self.conf.make_loader(
    #        data,
    #        shuffle=False,
    #        drop_last=False,
    #        batch_size=self.conf.batch_size_eval,
    #        parallel=True,
    #    )
    #    model = self.ema_model
    #    model.eval()
    #    conds = []

    #    if with_render:
    #        sampler = self.conf._make_diffusion_conf(T=T_render or self.conf.T_eval).make_sampler()

    #        if self.global_rank == 0:
    #            writer = LMDBImageWriter(render_save_path, format="webp", quality=100)
    #        else:
    #            writer = nullcontext()
    #    else:
    #        writer = nullcontext()

    #    with writer:
    #        for batch in tqdm(loader, total=len(loader), desc="infer"):
    #            with torch.no_grad():
    #                # (n, c)
    #                # print('idx:', batch['index'])
    #                cond = model.encoder(batch["img"].to(self.device))

    #                # used for reordering to match the original dataset
    #                idx = batch["index"]
    #                idx = self.all_gather(idx)
    #                if idx.dim() == 2:
    #                    idx = idx.flatten(0, 1)
    #                argsort = idx.argsort()

    #                if with_render:
    #                    noise = torch.randn_like(batch["img"], device=self.device)
    #                    render = sampler.sample(model, noise=noise, cond=cond)
    #                    render = (render + 1) / 2
    #                    # print('render:', render.shape)
    #                    # (k, n, c, h, w)
    #                    render = self.all_gather(render)
    #                    if render.dim() == 5:
    #                        # (k*n, c)
    #                        render = render.flatten(0, 1)

    #                    if self.global_rank == 0:
    #                        writer.put_images(render[argsort])

    #                # (k, n, c)
    #                cond = self.all_gather(cond)

    #                if cond.dim() == 3:
    #                    # (k*n, c)
    #                    cond = cond.flatten(0, 1)

    #                conds.append(cond[argsort].cpu().detach())
    #            # break
    #    model.train()
    #    # (N, c) cpu

    #    conds = torch.cat(conds).float()
    #    return conds
    def log_sample(self, x_start, x_start_lr):
        """
        put images to the tensorboard
        """
        if self.conf.sample_every_samples > 0 and is_time(self.num_samples, self.conf.sample_every_samples, self.conf.batch_size_effective):
            if self.conf.train_mode.require_dataset_infer():
                _log_sample(self, x_start, x_start_lr, self.model, "", use_xstart=False)
                # _log_sample(self, x_start, self.ema_model, "_ema", use_xstart=False)
            else:
                if self.conf.model_type.has_autoenc() and self.conf.model_type.can_sample():
                    _log_sample(self, x_start, x_start_lr, self.model, "", use_xstart=False)
                    # _log_sample(self, x_start, self.ema_model, "_ema", use_xstart=False)
                    # autoencoding mode
                    _log_sample(self, x_start, x_start_lr, self.model, "_enc", use_xstart=True, save_real=True)
                    # _log_sample(self, x_start, self.ema_model, "_enc_ema", use_xstart=True, save_real=True)
                elif self.conf.train_mode.use_latent_net():
                    _log_sample(self, x_start, x_start_lr, self.model, "", use_xstart=False)
                    # _log_sample(self, x_start, self.ema_model, "_ema", use_xstart=False)
                    # autoencoding mode
                    _log_sample(self, x_start, x_start_lr, self.model, "_enc", use_xstart=True, save_real=True)
                    # _log_sample(self, x_start, self.ema_model, "_enc_ema", use_xstart=True, save_real=True)
                else:
                    _log_sample(self, x_start, x_start_lr, self.model, "", use_xstart=True, save_real=True)
                    # _log_sample(self, x_start, self.ema_model, "_ema", use_xstart=True, save_real=True)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        if self.global_rank == 0:
            experiment: SummaryWriter = self.logger.experiment  # type: ignore
            if isinstance(experiment, SummaryWriter):
                #experiment.add_images(tag, image, step,dataformats='NCHW')
                experiment.add_image('Original', image)#, dataformats='NCHW')
            # elif isinstance(experiment, wandb.sdk.wandb_run.Run):
            #    experiment.log(
            #        {tag: [wandb.Image(image.cpu())]},
            #        # step=step,
            #    )
            else:
                raise NotImplementedError()

    # def log_histogram(self, tag: str, np_histogram: torch.Tensor, step: int = None) -> None:
    #    if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
    #        np_histogram = tuple([h.cpu().numpy() for h in np_histogram])
    #        np_histogram = wandb.Histogram(np_histogram=np_histogram, num_bins=53)
    #        self.logger.experiment.log({tag: np_histogram}, step=step)

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
               latent_sampler=self.eval_latent_sampler#,
               #conds_mean=self.conds_mean,
               #conds_std=self.conds_std,
           )
           print("score:",score)
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
        
    #    if (
    #        self.num_samples > 0
    #        and is_time(self.conf.batch_size_effective)
    #    ):
    #        print(f"eval fid @ {self.num_samples}")
    #        lpips(self.model, "")
       if self.conf.dims == 2:
           fid(self.model, "")

    #    if (
    #        self.conf.eval_ema_every_samples > 0
    #        and self.num_samples > 0
    #        and is_time(self.num_samples, self.conf.eval_ema_every_samples, self.conf.batch_size_effective)
    #    ):
    #        if self.conf.dims == 2:
    #            print(f"eval fid ema @ {self.num_samples}")
    #            fid(self.ema_model, "_ema")
           #it's too slow
           #lpips(self.ema_model, '_ema')

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

    # def test_step(self, batch, *args, **kwargs):
    #    """
    #    for the 'eval' mode.
    #    We first select what to do according to the 'conf.eval_programs'.
    #    test_step will only run for 'one iteration' (it's a hack!).

    #    We just want the multi-gpu support.
    #    """
    #    # make sure you seed each worker differently!
    #    self.setup()

    #    # it will run only one step!
    #    print("global step:", self.global_step)
    #    """
    #    'infer' = predict the latent variables using the encoder on the whole dataset
    #    """
    #    if "infer" in self.conf.eval_programs:
    #        print("infer ...")
    #        conds = self.infer_whole_dataset(split="train").float()
    #        # NOTE: always use this path for the latent.pkl files
    #        save_path = f'{self.conf.logdir.replace("_debug","")}/latent.pkl'

    #        if self.global_rank == 0:
    #            conds_mean = conds.mean(dim=0)
    #            conds_std = conds.std(dim=0)
    #            if not os.path.exists(os.path.dirname(save_path)):
    #                os.makedirs(os.path.dirname(save_path))
    #            torch.save(
    #                {
    #                    "conds": conds,
    #                    "conds_mean": conds_mean,
    #                    "conds_std": conds_std,
    #                },
    #                save_path,
    #            )
    #    """
    #    'infer+render' = predict the latent variables using the encoder on the whole dataset
    #    THIS ALSO GENERATE CORRESPONDING IMAGES
    #    """
    #    # infer + reconstruction quality of the input
    #    for each in self.conf.eval_programs:
    #        if each.startswith("infer+render"):
    #            m = re.match(r"infer\+render([0-9]+)", each)
    #            if m is not None:
    #                T = int(m[1])
    #                self.setup()
    #                print(f"infer + reconstruction T{T} ...")
    #                conds = self.infer_whole_dataset(
    #                    with_render=True,
    #                    T_render=T,
    #                    render_save_path=f"latent_infer_render{T}/{self.conf.name}.lmdb",
    #                )
    #                save_path = f"latent_infer_render{T}/{self.conf.name}.pkl"
    #                conds_mean = conds.mean(dim=0)
    #                conds_std = conds.std(dim=0)
    #                if not os.path.exists(os.path.dirname(save_path)):
    #                    os.makedirs(os.path.dirname(save_path))
    #                torch.save(
    #                    {
    #                        "conds": conds,
    #                        "conds_mean": conds_mean,
    #                        "conds_std": conds_std,
    #                    },
    #                    save_path,
    #                )

    #    # evals those 'fidXX'
    #    """
    #    'fid<T>' = unconditional generation (conf.train_mode = diffusion).
    #        Note:   Diff. autoenc will still receive real images in this mode.
    #    'fid<T>,<T_latent>' = unconditional generation for latent models (conf.train_mode = latent_diffusion).
    #        Note:   Diff. autoenc will still NOT receive real images in this made.
    #                but you need to make sure that the train_mode is latent_diffusion.
    #    """
    #    for each in self.conf.eval_programs:
    #        if each.startswith("fid"):
    #            m = re.match(r"fid\(([0-9]+),([0-9]+)\)", each)
    #            clip_latent_noise = False
    #            if m is not None:
    #                # eval(T1,T2)
    #                T = int(m[1])
    #                T_latent = int(m[2])
    #                print(f"evaluating FID T = {T}... latent T = {T_latent}")
    #            else:
    #                m = re.match(r"fidclip\(([0-9]+),([0-9]+)\)", each)
    #                if m is not None:
    #                    # fidclip(T1,T2)
    #                    T = int(m[1])
    #                    T_latent = int(m[2])
    #                    clip_latent_noise = True
    #                    print(f"evaluating FID (clip latent noise) T = {T}... latent T = {T_latent}")
    #                else:
    #                    # evalT
    #                    _, T = each.split("fid")
    #                    T = int(T)
    #                    T_latent = None
    #                    print(f"evaluating FID T = {T}...")

    #            self.train_dataloader()
    #            sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
    #            if T_latent is not None:
    #                latent_sampler = self.conf._make_latent_diffusion_conf(T=T_latent).make_sampler()
    #            else:
    #                latent_sampler = None

    #            conf = self.conf.clone()
    #            conf.eval_num_images = 50_000
    #            score = evaluate_fid(
    #                sampler,
    #                self.ema_model,
    #                conf,
    #                device=self.device,
    #                train_data=self.train_data,
    #                val_data=self.val_data,
    #                latent_sampler=latent_sampler,
    #                conds_mean=self.conds_mean,
    #                conds_std=self.conds_std,
    #                remove_cache=False,
    #                clip_latent_noise=clip_latent_noise,
    #            )
    #            if T_latent is None:
    #                self.log(f"fid_ema_T{T}", score.item())
    #            else:
    #                name = "fid"
    #                if clip_latent_noise:
    #                    name += "_clip"
    #                name += f"_ema_T{T}_Tlatent{T_latent}"
    #                self.log(name, score.item())
    #    """
    #    'recon<T>' = reconstruction & autoencoding (without noise inversion)
    #    """
    #    for each in self.conf.eval_programs:
    #        if each.startswith("recon"):
    #            self.model: BeatGANsAutoencModel
    #            _, T = each.split("recon")
    #            T = int(T)
    #            print(f"evaluating reconstruction T = {T}...")

    #            sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

    #            conf = self.conf.clone()
    #            # eval whole val dataset
    #            conf.eval_num_images = len(self.val_data)
    #            # {'lpips', 'mse', 'ssim'}
    #            score = evaluate_lpips(
    #                sampler,
    #                self.model,
    #                #    self.ema_model,
    #                conf,
    #                device=self.device,
    #                val_data=self.val_data,
    #                latent_sampler=None,
    #            )
    #            for k, v in score.items():
    #                self.log(f"{k}_ema_T{T}", v.item())
    #    """
    #    'inv<T>' = reconstruction with noise inversion
    #    """
    #    for each in self.conf.eval_programs:
    #        if each.startswith("inv"):
    #            self.model: BeatGANsAutoencModel
    #            _, T = each.split("inv")
    #            T = int(T)
    #            print(f"evaluating reconstruction with noise inversion T = {T}...")

    #            sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

    #            conf = self.conf.clone()
    #            # eval whole val dataset
    #            conf.eval_num_images = len(self.val_data)
    #            # {'lpips', 'mse', 'ssim'}
    #            score = evaluate_lpips(
    #                sampler, self.ema_model, conf, device=self.device, val_data=self.val_data, latent_sampler=None, use_inverted_noise=True
    #            )
    #            for k, v in score.items():
    #                self.log(f"{k}_inv_ema_T{T}", v.item())


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


from collections import deque


@torch.no_grad()
def _log_sample(self: DAE_LitModel, x_start, x_start_lr, model, postfix, use_xstart, save_real=False, interpolate=False):
    global buffer_image_names
    model.eval()
    all_x_T = self.split_tensor(self.x_T)
    batch_size = min(len(all_x_T), self.conf.batch_size_eval)
    ## allow for super large models
    loader = DataLoader(all_x_T, batch_size=batch_size)  # type: ignore
    Gen = []
    for x_T in loader:  # tqdm(loader, desc="img", total=len(loader)):
        if use_xstart:
            if x_start_lr is None:
                _xstart = x_start[: len(x_T)]

            else:
                _xstart = x_start_lr[: len(x_T)]
                _xstart_hr = x_start[: len(x_T)]
        else:
            _xstart = None
        if self.conf.train_mode.is_latent_diffusion() and not use_xstart:
            raise NotImplementedError("render_uncondition")
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
                raise NotImplementedError("render_uncondition")
                model: BeatGANsAutoencModel
                # special case, it may not be stochastic, yet can sample
                cond = torch.randn(len(x_T), self.conf.style_ch, device=self.device)
                cond = model.noise_to_cond(cond)
            else:
                if interpolate:
                    with autocast(not self.conf.fp32):
                        cond = model.encoder(_xstart)
                        i = torch.randperm(len(cond))
                        cond = (cond + cond[i]) / 2
                else:
                    cond = None
            gen = self.eval_sampler.sample(model=model, noise=x_T, cond=cond, x_start=_xstart)
        Gen.append(gen)
    gen: torch.Tensor = torch.cat(Gen)
    gen = self.all_gather(gen)  # type: ignore
    if (gen.dim() - self.conf.dims) == 3:
        # collect tensors from different workers
        # (n, c, h, w)
        gen = gen.flatten(0, 1)
    if self.conf.dims == 3:
        raise NotImplementedError()
        if self.global_rank == 0:
            # TODO: convert gif to mp4 to save it to wandb
            # if logged to tensorboard
            if isinstance(self.logger, pl_loggers.TensorBoardLogger):
                plot_2d_or_3d_image(gen, self.global_step, self.logger.experiment, tag=f"sample{postfix}/fake", frame_dim=-1)
            elif isinstance(self.logger, pl_loggers.WandbLogger):
                ...
        gen = extract_slices_from_volume(gen)
    sample_dir = os.path.join(self.conf.log_dir, self.conf.experiment_name, f"sample{postfix}")
    if self.global_rank == 0 and not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    grid_params = lambda t: dict(
        tensor=t, nrow=3 if self.conf.dims == 3 else int(np.sqrt(t.size(0))), normalize=True, padding=0, value_range=(-1, 1)
    )
    log_img = []
    if save_real and use_xstart and x_start_lr is not None:
        a = _save_image(self, _xstart_hr, grid_params, postfix, sample_dir)
        log_img.append(a)
    if save_real and use_xstart:
        a = _save_image(self, _xstart, grid_params, postfix, sample_dir)
        log_img.append(a)
    if self.global_rank == 0:
        # save samples to the tensorboard
        gen_grid = make_grid(**grid_params(gen))
        path = os.path.join(sample_dir, f"{self.global_step}.png")
        remove_old_jpgs(path)
        save_image(gen_grid, path)
        log_img.append(gen_grid)
        print("log")
        #self.log_image(f"sample{postfix}/fake", torch.concat(log_img, dim=-1), self.global_step)
    model.train()
    # x_start_lr


def _save_image(self: DAE_LitModel, _xstart, grid_params, postfix, sample_dir):
    # save the original images to the tensorboard
    real: torch.Tensor = self.all_gather(_xstart)  # type: ignore
    if (real.dim() - self.conf.dims) == 3:
        real = real.flatten(0, 1)
    if self.conf.dims == 3:
        raise NotImplementedError()
        # visualize volume using MONAI
        if self.global_rank == 0:
            if isinstance(self.logger, pl_loggers.TensorBoardLogger):
                plot_2d_or_3d_image(real, self.global_step, self.logger.experiment, tag=f"sample{postfix}/real", frame_dim=-1)
            elif isinstance(self.logger, pl_loggers.WandbLogger):
                # log as 3d object
                # TODO: add rendering as mesh
                ...
        # extract 2d slice from different sequences
        real = extract_slices_from_volume(real)
    if self.global_rank == 0:
        real_grid = make_grid(**grid_params(real))
        # self.log_image(f"sample{postfix}/real", real_grid, self.global_step)
        path = os.path.join(sample_dir, "real.png")
        remove_old_jpgs(path)
        save_image(real_grid, path)
        return real_grid


buffer_image_names = deque()
images_keept_anyway = 0
images_removed = -100


def remove_old_jpgs(path):
    global images_removed
    global images_keept_anyway
    buffer_image_names.append(path)
    if len(buffer_image_names) >= 100:
        old_path = buffer_image_names.popleft()
        p = random.random() + images_removed / (images_keept_anyway * images_keept_anyway * 100 + 1)
        if p < 1:
            Path(old_path).unlink()
            images_removed += 1
        else:
            images_keept_anyway += 1
