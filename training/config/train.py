import os
import time
from multiprocessing import get_context
from typing import Literal

from torch import distributed
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

import wandb
from training.config.base import BaseConfig
from training.data.cache import use_cached_dataset_path
from training.data.glioma_public import PublicGliomaDataset
from training.data.lmdb import CelebAlmdb, FFHQlmdb
from training.data.mri import MriCrop
from training.data.nako import NAKO_JPG
from training.data.transforms import ContrastiveLearningViewGenerator, get_simclr_transform_mri
from training.mode.clf import ClfMode
from training.mode.loss import LossType, ManipulateLossType
from training.mode.manipulate import ManipulateMode
from training.mode.model import GenerativeType, ModelMeanType, ModelName, ModelType, ModelVarType
from training.mode.optim import OptimizerType
from training.mode.train import TrainMode
from training.models import *
from training.models.diffusion import *
from training.models.diffusion.base import get_named_beta_schedule
from training.models.diffusion.diffusion import space_timesteps
from training.models.diffusion.resample import UniformSampler
from training.models.latentnet import *
from training.models.simclr import SimCLRConfig
from training.models.simsiam import SimSiamConfig
from training.models.unet import ScaleAt
from training.sampler.stratified import UniformStratifiedSampler, WeightedStratifiedSampler

data_paths = {
    "ffhqlmdb256": os.path.expanduser("datasets/ffhq256.lmdb"),
    # used for training a classifier
    "celeba": os.path.expanduser("datasets/celeba"),
    # used for training DPM models
    "celebalmdb": os.path.expanduser("datasets/celeba.lmdb"),
    "celebahq": os.path.expanduser("datasets/celebahq256.lmdb"),
    "horse256": os.path.expanduser("datasets/horse256.lmdb"),
    "bedroom256": os.path.expanduser("datasets/bedroom256.lmdb"),
    "celeba_anno": os.path.expanduser("datasets/celeba_anno/list_attr_celeba.txt"),
    "celebahq_anno": os.path.expanduser("datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt"),
    "celeba_relight": os.path.expanduser("datasets/celeba_hq_light/celeba_light.txt"),
    "gliomapublic": os.path.expanduser("~/datasets/glioma_public"),
    # os.path.expanduser('/mnt/Drive4/daniel/datasets/glioma_public'),
    "gliomapublic_64": os.path.expanduser("~/datasets/glioma_public_64"),
    "nako": "/media/data/robert/code/nako_embedding/dataset/train.csv"
    # os.path.expanduser('/mnt/Drive4/daniel/datasets/glioma_public'),
}


@dataclass
class PretrainConfig(BaseConfig):
    _name: str
    _path: str

    @property
    def name(self):
        # remove debug from the name
        return self._name.replace("_debug", "")

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def path(self):
        return self._path.replace("_debug", "")

    @path.setter
    def path(self, value):
        # remove debug from the name
        self._path = value


@dataclass
class TrainConfig(BaseConfig):
    # random seed
    wandb_project: str = ""
    wandb_id: str = None  # used for resuming
    wandb_id_pretrain: str = None  # stores id of pretrain run
    seed: int = 0
    train_mode: TrainMode = TrainMode.diffusion
    train_cond0_prob: float = 0
    train_pred_xstart_detach: bool = True
    train_interpolate_prob: float = 0
    train_interpolate_img: bool = False
    manipulate_mode: ManipulateMode = ManipulateMode.celebahq_all
    manipulate_cls: tuple = tuple([])
    manipulate_shots: int = None
    manipulate_loss: ManipulateLossType = ManipulateLossType.bce
    manipulate_znormalize: bool = False
    manipulate_seed: int = 0
    clf_mode: ClfMode = ClfMode.one_vs_all
    clf_arch: Literal[
        "resnet18",
        "densenet",
        "resnet50",
    ] = "resnet18"
    # early stopping patience: number of epochs with no improvement after which training will be stopped
    clf_early_stopping_patience: int = 10
    split_ratio: float = 0.9
    split_mode: Literal["mixed", "study"] = "study"  # default: study, for sanity check: mixed
    ckpt_state: Literal["best", "last"] = "best"  # default: best, for sanity check: last

    stratified_sampling_mode: Literal["weighted", "uniform"] = "uniform"

    accum_batches: int = 1
    autoenc_mid_attn: bool = True
    batch_size: int = 16
    batch_size_eval: int = None
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.mse
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False
    latent_infer_path: str = None
    latent_znormalize: bool = False
    latent_gen_type: GenerativeType = GenerativeType.ddim
    latent_loss_type: LossType = LossType.mse
    latent_model_mean_type: ModelMeanType = ModelMeanType.eps
    latent_model_var_type: ModelVarType = ModelVarType.fixed_large
    latent_rescale_timesteps: bool = False
    latent_T_eval: int = 1_000
    latent_clip_sample: bool = False
    latent_beta_scheduler: str = "linear"
    beta_scheduler: str = "linear"
    data_name: str = ""
    data_val_name: str = None
    diffusion_type: str = None
    dropout: float = 0.1
    ema_decay: float = 0.9999
    eval_num_images: int = 5_000
    eval_every_samples: int = 200_000
    eval_ema_every_samples: int = 200_000
    fid_use_torch: bool = True
    fp16: bool = False
    grad_clip: float = 1.0
    img_size: int = 64
    dims: int = 2  # whether to use 2D or 3D diffusion
    in_channels: int = 3  # number of input channels
    model_out_channels: int = 3  # number of output channels
    lr: float = 0.0001
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0
    model_conf: ModelConfig = None
    model_name: ModelName = None
    model_type: ModelType = None
    net_attn: tuple[int] = None
    net_beatgans_attn_head: int = 1
    # not necessarily the same as the the number of style channels
    embed_channels: int = 512
    net_resblock_updown: bool = True
    net_enc_use_time: bool = False
    net_enc_pool: str = "adaptivenonzero"
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    net_beatgans_resnet_cond_channels: int = None
    net_ch_mult: tuple[int, ...] = None
    net_ch: int = 64
    net_enc_attn: tuple[int, ...] = None
    net_enc_k: int = None
    # number of resblocks for the encoder (half-unet)
    net_enc_num_res_blocks: int = 2
    net_enc_channel_mult: tuple[int, ...] = None
    net_enc_grad_checkpoint: bool = False
    net_autoenc_stochastic: bool = False
    net_latent_activation: str = "silu"
    net_latent_channel_mult: tuple[int, ...] = (1, 2, 4)
    net_latent_condition_bias: float = 0
    net_latent_dropout: float = 0
    net_latent_layers: int = None
    net_latent_net_last_act: Union[str, None] = None
    net_latent_net_type: LatentNetType = LatentNetType.none
    net_latent_num_hid_channels: int = 1024
    net_latent_num_time_layers: int = 2
    net_latent_skip_layers: tuple[int] = None
    net_latent_time_emb_channels: int = 64
    net_latent_use_norm: bool = False
    net_latent_time_last_act: bool = False
    net_num_res_blocks: int = 2
    # number of resblocks for the UNET
    net_num_input_res_blocks: int = None
    net_enc_num_cls: int = None
    num_workers: int = 4
    parallel: bool = False
    postfix: str = ""
    sample_size: int = 64
    sample_every_samples: int = 20_000
    save_every_samples: int = 100_000
    style_ch: int = 512
    T_eval: int = 1_000
    T_sampler: str = "uniform"
    T: int = 1_000
    total_samples: int = 10_000_000
    warmup: int = 0

    # classification
    with_data_aug: bool = False
    aug_encoder: bool = False

    pretrain: PretrainConfig = None
    continue_from: PretrainConfig = None
    eval_programs: tuple[str] = None
    # if present load the checkpoint from this path instead
    eval_path: str = None
    base_dir: str = "checkpoints"
    version: str = "2"

    # pretrain path, possibility to load autoencoder weights from different run than classifier
    pretrain_path: str = ""
    full_fine_tuning: bool = False

    use_cache_dataset: bool = False
    data_cache_dir: str = os.path.expanduser("~/cache")
    work_cache_dir: str = os.path.expanduser("~/mycache")
    # to be overridden
    name: str = ""
    # current timestamp in the format YYYYMMDD_HHMMSS
    timestamp: str = time.strftime("%Y%m%d_%H%M%S")

    # whether to overfit to a single batch
    overfit: bool = False
    # whether to only test the model or also train it
    test_only: bool = False

    # mri specific
    mri_sequences: tuple[str, ...] = ()
    mri_crop: MriCrop = None  # whether to crop the mri images

    # dataset specific
    use_healthy: bool = False
    data_aug_prob: float = 0.5

    # simclr config
    num_views: int = 2

    def __post_init__(self):
        self.batch_size_eval = self.batch_size_eval or self.batch_size
        self.data_val_name = self.data_val_name or self.data_name

    def scale_up_gpus(self, num_gpus, num_nodes=1):
        self.eval_ema_every_samples *= num_gpus * num_nodes
        self.eval_every_samples *= num_gpus * num_nodes
        self.sample_every_samples *= num_gpus * num_nodes
        self.batch_size *= num_gpus * num_nodes
        self.batch_size_eval *= num_gpus * num_nodes
        return self

    def update_with_args(self, args=None):
        if args is None:
            return
        for k, v in args.__dict__.items():
            if v is not None:
                if hasattr(self, k):
                    if (v_type := type(getattr(self, k))) is not None:
                        # cast to the correct type
                        v = v_type(v)
                    setattr(self, k, v)
                else:
                    print(f"Warning: {k} is not a valid config attribute")

    def update_sweep(self, sweep_config: wandb.Config):
        if sweep_config is None:
            return
        for k, v in sweep_config.items():
            setattr(self, k, v)

    @property
    def batch_size_effective(self):
        return self.batch_size * self.accum_batches

    @property
    def fid_cache(self):
        # we try to use the local dirs to reduce the load over network drives
        # hopefully, this would reduce the disconnection problems with sshfs
        return f"{self.work_cache_dir}/eval_images/{self.data_name}_size{self.img_size}_{self.eval_num_images}"

    @property
    def data_path(self):
        # may use the cache dir
        path = data_paths[self.data_name]
        if self.use_cache_dataset and path is not None:
            path = use_cached_dataset_path(path, f"{self.data_cache_dir}/{self.data_name}")
        return path

    @property
    def logdir(self):
        return os.path.join(
            self.base_dir,
            self.name,
            f"version_{self.version}",
        )

    @property
    def generate_dir(self):
        # we try to use the local dirs to reduce the load over network drives
        # hopefully, this would reduce the disconnection problems with sshfs
        return f"{self.work_cache_dir}/gen_images/{self.name}"

    def _make_diffusion_conf(self, T=None):
        if self.diffusion_type == "beatgans":
            # can use T < self.T for evaluation
            # follows the guided-diffusion repo conventions
            # t's are evenly spaced
            if self.beatgans_gen_type == GenerativeType.ddpm:
                section_counts = [T]
            elif self.beatgans_gen_type == GenerativeType.ddim:
                section_counts = f"ddim{T}"
            else:
                raise NotImplementedError()

            return SpacedDiffusionBeatGansConfig(
                gen_type=self.beatgans_gen_type,
                model_type=self.model_type,
                betas=get_named_beta_schedule(self.beta_scheduler, self.T),
                model_mean_type=self.beatgans_model_mean_type,
                model_var_type=self.beatgans_model_var_type,
                loss_type=self.beatgans_loss_type,
                rescale_timesteps=self.beatgans_rescale_timesteps,
                use_timesteps=space_timesteps(num_timesteps=self.T, section_counts=section_counts),
                fp16=self.fp16,
            )
        else:
            raise NotImplementedError()

    def _make_latent_diffusion_conf(self, T=None):
        # can use T < self.T for evaluation
        # follows the guided-diffusion repo conventions
        # t's are evenly spaced
        if self.latent_gen_type == GenerativeType.ddpm:
            section_counts = [T]
        elif self.latent_gen_type == GenerativeType.ddim:
            section_counts = f"ddim{T}"
        else:
            raise NotImplementedError()

        return SpacedDiffusionBeatGansConfig(
            train_pred_xstart_detach=self.train_pred_xstart_detach,
            gen_type=self.latent_gen_type,
            # latent's model is always ddpm
            model_type=ModelType.ddpm,
            # latent shares the beta scheduler and full T
            betas=get_named_beta_schedule(self.latent_beta_scheduler, self.T),
            model_mean_type=self.latent_model_mean_type,
            model_var_type=self.latent_model_var_type,
            loss_type=self.latent_loss_type,
            rescale_timesteps=self.latent_rescale_timesteps,
            use_timesteps=space_timesteps(num_timesteps=self.T, section_counts=section_counts),
            fp16=self.fp16,
        )

    def make_T_sampler(self):
        if self.T_sampler == "uniform":
            return UniformSampler(self.T)
        else:
            raise NotImplementedError()

    def make_diffusion_conf(self):
        return self._make_diffusion_conf(self.T)

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T_eval)

    def make_latent_diffusion_conf(self):
        return self._make_latent_diffusion_conf(T=self.T)

    def make_latent_eval_diffusion_conf(self):
        # latent can have different eval T
        return self._make_latent_diffusion_conf(T=self.latent_T_eval)

    def make_dataset(self, path=None, **kwargs):
        if self.data_name == "ffhqlmdb256":
            return FFHQlmdb(path=path or self.data_path, image_size=self.img_size, **kwargs)
        elif self.data_name == "celebalmdb":
            # always use d2c crop
            return CelebAlmdb(path=path or self.data_path, image_size=self.img_size, original_resolution=None, crop_d2c=True, **kwargs)
        elif "gliomapublic" in self.data_name:
            if self.model_name in [ModelName.simclr, ModelName.simsiam]:
                kwargs["view_transform"] = ContrastiveLearningViewGenerator(
                    get_simclr_transform_mri(
                        self.img_size, keys=["img", PublicGliomaDataset.BRAINMASK_NAME, PublicGliomaDataset.SEG_LABEL_NAME], with_crop=True
                    ),
                    self.num_views,
                )
            return PublicGliomaDataset(
                data_dir=path or self.data_path,
                img_size=self.img_size,
                mri_sequences=self.mri_sequences,
                mri_crop=self.mri_crop,
                train_mode=self.train_mode,
                split_ratio=self.split_ratio,
                use_healthy=self.use_healthy,
                with_data_aug=self.with_data_aug,
                aug_encoder=self.aug_encoder,
                split_mode=self.split_mode,
                data_aug_prob=self.data_aug_prob,
                **kwargs,
            )
        elif "nako" in self.data_name:
            # if self.model_name in [ModelName.simclr, ModelName.simsiam]:
            #    kwargs["view_transform"] = ContrastiveLearningViewGenerator(
            #        get_simclr_transform_mri(
            #            self.img_size, keys=["img", PublicGliomaDataset.BRAINMASK_NAME, PublicGliomaDataset.SEG_LABEL_NAME], with_crop=True
            #        ),
            #        self.num_views,
            #    )
            return NAKO_JPG(
                path=path or self.data_path,
                image_size=self.img_size,
                mri_sequences=self.mri_sequences,
                mri_crop=self.mri_crop,
                train_mode=self.train_mode,
                split_ratio=self.split_ratio,
                with_data_aug=self.with_data_aug,
                aug_encoder=self.aug_encoder,
                split_mode=self.split_mode,
                data_aug_prob=self.data_aug_prob,
                **kwargs,
            )
        else:
            raise NotImplementedError(self.data_name)

    def make_loader(
        self,
        dataset,
        shuffle: bool,
        mode: str,
        num_workers: int = None,
        drop_last: bool = True,
        batch_size: int = None,
        parallel: bool = False,
    ):
        use_stratified_sampler = False
        if parallel and distributed.is_initialized():
            # drop last to make sure that there is no added special indexes
            sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
        elif use_stratified_sampler and mode == "train":
            print("using stratified sampler")
            # stratified sampler for imbalanced data
            labels = [dataset.cls_labels[dataset._make_patient_id(subject)] for subject in dataset.subject_dirs]
            if self.stratified_sampling_mode == "uniform":
                # uniform stratified sampler
                sampler = UniformStratifiedSampler(batch_size, labels)
            elif self.stratified_sampling_mode == "weighted":
                sampler = WeightedStratifiedSampler(batch_size, labels)
            else:
                raise NotImplementedError(f"unknown stratified sampler mode: {self.stratified_sampling_mode}")
        elif hasattr(dataset, "sample_weights") and mode == "train" and self.train_mode.is_manipulate():
            print("using weighted sampler")
            # weighted sampler for imbalanced data
            sampler = WeightedRandomSampler(dataset.sample_weights(), len(dataset))
        else:
            sampler = None

        num_workers = num_workers or self.num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            sampler=sampler,
            # with sampler, use the sample instead of this option
            shuffle=False if sampler else shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            multiprocessing_context=get_context("fork") if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    def make_model_conf(self, args=None):
        if self.model_name == ModelName.beatgans_ddpm:
            self.model_type = ModelType.ddpm
            self.model_conf = BeatGANsUNetConfig(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=self.dims,
                dropout=self.dropout,
                embed_channels=self.embed_channels,
                image_size=self.img_size,
                in_channels=self.in_channels,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.net_beatgans_resnet_use_zero_module,
            )
        elif self.model_name in [
            ModelName.beatgans_autoenc,
        ]:
            cls = BeatGANsAutoencConfig
            # supports both autoenc and vaeddpm
            if self.model_name == ModelName.beatgans_autoenc:
                self.model_type = ModelType.autoencoder
            else:
                raise NotImplementedError()

            if self.net_latent_net_type == LatentNetType.none:
                latent_net_conf = None
            elif self.net_latent_net_type == LatentNetType.skip:
                latent_net_conf = MLPSkipNetConfig(
                    num_channels=self.style_ch,
                    skip_layers=self.net_latent_skip_layers,
                    num_hid_channels=self.net_latent_num_hid_channels,
                    num_layers=self.net_latent_layers,
                    num_time_emb_channels=self.net_latent_time_emb_channels,
                    activation=self.net_latent_activation,
                    use_norm=self.net_latent_use_norm,
                    condition_bias=self.net_latent_condition_bias,
                    dropout=self.net_latent_dropout,
                    last_act=self.net_latent_net_last_act,
                    num_time_layers=self.net_latent_num_time_layers,
                    time_last_act=self.net_latent_time_last_act,
                )
            else:
                raise NotImplementedError()

            self.model_conf = cls(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=self.dims,
                dropout=self.dropout,
                embed_channels=self.embed_channels,
                enc_out_channels=self.style_ch,
                enc_pool=self.net_enc_pool,
                enc_num_res_block=self.net_enc_num_res_blocks,
                enc_channel_mult=self.net_enc_channel_mult,
                enc_grad_checkpoint=self.net_enc_grad_checkpoint,
                enc_attn_resolutions=self.net_enc_attn,
                image_size=self.img_size,
                in_channels=self.in_channels,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.net_beatgans_resnet_use_zero_module,
                latent_net_conf=latent_net_conf,
                resnet_cond_channels=self.net_beatgans_resnet_cond_channels,
            )
        elif self.model_name == ModelName.simclr:
            self.model_conf = SimCLRConfig(out_dim=self.embed_channels)
        elif self.model_name == ModelName.simsiam:
            self.model_conf = SimSiamConfig(out_dim=self.embed_channels)
        else:
            raise NotImplementedError(self.model_name)

        return self.model_conf
