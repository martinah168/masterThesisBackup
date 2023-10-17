from collections.abc import Callable, Mapping
from typing import Any, Literal
from attr import dataclass
import torch
from utils.auto_arguments import Option_to_Dataclass
from dataclasses import Field, asdict, field
from dataloader import transforms as T
from pathlib import Path
from .enums import TrainMode
from diffusion.beta_schedule import Beta_Schedule
from utils.enums_model import GenerativeType, LatentNetType, ModelMeanType, ModelType, ModelVarType, LossType, OptimizerType, ModelName


@dataclass
class Train_Option(Option_to_Dataclass):
    experiment_name: str = "NAKO_256"
    lr: float = 0.0001
    batch_size: int = 64
    batch_size_eval: int = 64
    debug: bool = True
    new: bool = False
    gpus: list[int] | None = None
    num_cpu: int = 16
    # Logging
    log_dir: str = "lightning_logs"
    log_every_n_steps = 1#3000
    fast_dev_run: bool = False
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0.0


@dataclass
class DataSet_Option:
    dataset: str = "/media/DATA/martina_ma/datasets/test_csv.csv"#"/media/data/robert/code/nako_embedding/dataset/train.csv"
    ds_type: str = "csv_2D"  # Literal["csv_2D"]
    transforms: list[T.Transforms_Enum] | None = None
    in_channels: int = 1  # Channel of the Noised input
    img_size: int | list[int] = 128#256  # TODO Diffusion_Autoencoder_Model can't deal with list[int]
    dims: int = 2#3

    @property
    def shape(self):
        if isinstance(self.img_size, int):
            return (self.img_size,) * self.dims
        elif len(self.img_size) == 1:
            self.img_size = self.img_size[0]
            return self.shape
        if len(self.img_size) != self.dims:
            raise ValueError(
                f"dims ({self.dims}) is different length than image_size ({self.img_size}). Use same length for image_size or an int or update dims"
            )
        return tuple(self.img_size)


@dataclass
class DAE_Model_Option:
    attention_resolutions: list[int] = field(default_factory=lambda: [16])
    net_ch_mult: tuple[int, ...] = field(default_factory=lambda: (1, 1, 2, 2))
    dropout: float = 0.1
    embed_channels: int = 512
    enc_out_channels: int = 512
    net_enc_pool: str = "adaptivenonzero"
    enc_num_res_blocks: int = 2
    enc_channel_mult: tuple[int, ...] = field(default_factory=lambda: (1, 1, 2, 4, 4))
    enc_grad_checkpoint = False
    enc_attn = None
    model_channels: int = 64
    net_beatgans_attn_head = 1
    net_num_res_blocks = 2
    net_num_input_res_blocks = None
    net_resblock_updown = True
    net_beatgans_gradient_checkpoint = False
    net_beatgans_resnet_two_cond = True
    net_beatgans_resnet_use_zero_module = True
    net_beatgans_resnet_cond_channels = None


@dataclass
class DAE_Option(Train_Option, DAE_Model_Option, DataSet_Option):
    seed: int | None = 0
    # Train
    total_samples: int = 100000000
    # Validation
    early_stopping_patience: int = 100
    save_every_samples: int = 25000
    sample_every_samples = 5000
    fp32: bool = False

    @property
    def batch_size_effective(self):
        return self.batch_size * self.accum_batches

    @property
    def model_out_channels(self):
        return self.in_channels

    # DIFFUSION
    beta_schedule: Beta_Schedule = Beta_Schedule.linear
    num_timesteps: int = 1000
    num_timesteps_ddim: int = 20  # | str | list[int]
    generative_type: GenerativeType = GenerativeType.ddim
    model_type: ModelType = ModelType.autoencoder
    model_mean_type: ModelMeanType = ModelMeanType.eps
    model_var_type: ModelVarType = ModelVarType.fixed_large
    loss_type: LossType = LossType.mse
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    rescale_timesteps: bool = False
    # Embedding
    hessian_penalty: float = 0
    # Debugging
    overfit: bool = False
    train_mode: TrainMode = TrainMode.diffusion
    pretrain = None
    # Model
    model_name: ModelName = ModelName.autoencoder  # TODO BEATSGAN???
    net_latent_net_type = LatentNetType.none

    @property
    def target_batch_size(self):
        if hasattr(self, "_target_batch_size"):
            return self._target_batch_size
        gpu_name = torch.cuda.get_device_name(0)
        _target_batch_size = self.batch_size
        if self.dims == 2:
            if "NVIDIA GeForce RTX 3090" == gpu_name:
                max_batch_size_that_fits_in_memory = 16
            else:
                max_batch_size_that_fits_in_memory = 16
            if self.img_size == 128:
                max_batch_size_that_fits_in_memory *= 4
        else:
            gpu_name = torch.cuda.get_device_name(0)
            if "A40" in gpu_name:
                max_batch_size_that_fits_in_memory = 4
            else:
                max_batch_size_that_fits_in_memory = 1

            _target_batch_size = 32

            _target_batch_size = _target_batch_size if not self.overfit else max_batch_size_that_fits_in_memory

        self._target_batch_size = _target_batch_size
        self.batch_size = min(max_batch_size_that_fits_in_memory, _target_batch_size)
        return self._target_batch_size

    @property
    def accum_batches(self):
        return self.target_batch_size // self.batch_size

    @property
    def sample_size(self):
        return self.batch_size


#    # Training
#    experiment_name: str = "NAME"
#    lr: float = 1e-4
#    batch_size: int = 64
#    batch_size_val: int = 1
#    num_epochs: int = 150
#    num_cpu: int = 16
#    target_patch_shape: int | list[int] = 256
#    gpus: list[int] | None = None
#    new: bool = False
#    prevent_nan: bool = False
#    channels: int = 64
#    cpu: bool = False
#    start_epoch: int = 0
#    log_dir: str = str(Path(__file__).parent.parent.parent.absolute()) + "/models/age_prediction"
#    model_name: str = "unet"  # No Other implemented
#    # auto_lr_find = False
#    dataset: str = ""
#    transforms: list[T.Transforms_Enum] | None = None
#
#
# @dataclass
# class Age_Prediction_Option(Train_Option):
#    model_name: model_factory.Model_Enum = model_factory.Model_Enum.DensNN121
#    weight_decay: float = 0.01
#    pass
#
#
# @dataclass
# class SVM_Option(Option_to_Dataclass):
#    # Training
#    experiment_name: str = "SVM"
#    batch_size: int = 64
#    batch_size_val: int = 1
#    num_cpu: int = 16
#    new: bool = False
#    log_dir: str = str(Path(__file__).parent.parent.parent.absolute()) + "/models/age_prediction"
#    dataset: str = ""
#    limit: int = 1000
#    key: str = "age"


def get_latest_Checkpoint(opt: Train_Option, version="*", log_dir_name="lightning_logs", best=False, verbose=True) -> str | None:
    import glob
    import os

    ckpt = "*"
    if best:
        ckpt = "*best*"
    print() if verbose else None
    checkpoints = None

    if isinstance(opt, str) or not opt.new:
        if isinstance(opt, str):
            checkpoints = sorted(glob.glob(f"{log_dir_name}/{opt}/version_{version}/checkpoints/{ckpt}.ckpt"), key=os.path.getmtime)
        else:
            checkpoints = sorted(
                glob.glob(f"{log_dir_name}/{opt.experiment_name}/version_{version}/checkpoints/{ckpt}.ckpt"),
                key=os.path.getmtime,
            )

        if len(checkpoints) == 0:
            checkpoints = None
        else:
            checkpoints = checkpoints[-1]
        print("Reload recent Checkpoint and continue training:", checkpoints) if verbose else None
    else:
        return None

    return checkpoints
