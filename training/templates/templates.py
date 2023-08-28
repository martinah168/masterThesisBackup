import torch

import wandb
from training.config.train import PretrainConfig, TrainConfig
from training.data.mri import MriCrop
from training.mode.model import GenerativeType, ModelName
from training.mode.train import TrainMode
from training.models.simclr import SimCLRConfig


def ddpm():
    """
    base configuration for all DDIM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = "linear"
    conf.data_name = "ffhq"
    conf.diffusion_type = "beatgans"
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn = (16,)
    conf.net_beatgans_attn_head = 1
    conf.embed_channels = 512
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = "linear"
    conf.data_name = "ffhq"
    conf.diffusion_type = "beatgans"
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16,)
    conf.net_beatgans_attn_head = 1
    conf.embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)

    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_ch = 64

    conf.net_enc_pool = "adaptivenonzero"
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.dims = 2

    # give a unique id to this run, can be overwritten to resume training
    conf.wandb_id = wandb.util.generate_id()

    conf.make_model_conf()
    return conf


def autoenc_base_3d():
    """
    base configuration for all Diff-AE models.
    """
    conf = autoenc_base()
    conf.dims = 3
    conf.make_model_conf()
    return conf


def ffhq256_autoenc():
    conf = autoenc_base()
    conf.data_name = "ffhqlmdb256"
    conf.scale_up_gpus(4)

    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    _orig_batch_size = 64
    conf.batch_size = 32
    # conf.sample_size = 32
    conf.accum_batches = _orig_batch_size // conf.batch_size
    conf.make_model_conf()
    conf.name = "ffhq256_autoenc"
    return conf


def gliomapublic_autoenc(is_debugging: bool = False, args=None):
    conf = autoenc_base_3d()

    conf.data_name = "gliomapublic"
    conf.wandb_project = "gliomapublic"

    # (is_debugging and args.pretrain_path is None)
    if args.low_res:
        # debug on low res
        conf.data_name += "_64"

    conf.num_workers = 8
    conf.img_size = 96  # crop size
    if "64" in conf.data_name:
        conf.img_size = 24

    # train val split
    conf.split_ratio = 0.9

    conf.net_ch = 64  # 64 best
    conf.net_attn = (16,)  # 16 best

    # final resolution = 8x8, diffusion resolution
    conf.net_ch_mult = (1, 1, 2, 2)
    if "64" in conf.data_name:
        conf.net_ch_mult = (1, 2)
    assert conf.img_size // (2 ** len(conf.net_ch_mult)) in [6, 8]
    # final resolution = 4x4, encoder network final feat map resolution
    conf.net_enc_channel_mult = (1, 1, 2, 4, 4)
    if "64" in conf.data_name:
        conf.net_enc_channel_mult = (1, 1, 2)
    assert conf.img_size // (2 ** len(conf.net_enc_channel_mult)) in [3, 4]

    conf.overfit = False

    gpu_name = torch.cuda.get_device_name(0)
    if "A40" in gpu_name:
        if args.low_res:
            max_batch_size_that_fits_in_memory = 32
        else:
            max_batch_size_that_fits_in_memory = 4
    else:
        max_batch_size_that_fits_in_memory = 1

    if conf.model_name in [ModelName.simsiam, ModelName.simclr] or conf.train_mode in [TrainMode.supervised]:
        # because batchnorm needs at least batch size two
        max_batch_size_that_fits_in_memory = max(2, max_batch_size_that_fits_in_memory)
    if conf.model_name in [ModelName.simsiam, ModelName.simclr]:
        max_batch_size_that_fits_in_memory *= 4
        _target_batch_size = 32
    else:
        _target_batch_size = 32

    _target_batch_size = _target_batch_size if not conf.overfit else max_batch_size_that_fits_in_memory

    conf.batch_size = min(max_batch_size_that_fits_in_memory, _target_batch_size)
    conf.accum_batches = _target_batch_size // conf.batch_size
    conf.batch_size_eval = conf.batch_size

    if is_debugging:
        conf.batch_size = max_batch_size_that_fits_in_memory
        conf.accum_batches = 1
        conf.num_workers = 0

    conf.sample_size = conf.batch_size
    if conf.train_mode == TrainMode.simclr:
        conf.batch_size = SimCLRConfig.batch_size

    # mri specific
    conf_mri_sequences(conf)
    conf.mri_crop = MriCrop.TUMOR
    conf.use_healthy = True  # sample healthy and unhealthy patients

    # logging
    conf.sample_every_samples = 5_000
    conf.save_every_samples = 25_000
    conf.eval_num_images = 200
    conf.ema_decay = 0.999

    # train
    conf.clf_early_stopping_patience = 100
    conf.total_samples = 100_000_000

    conf.name = "_".join([conf.data_name, make_kv_name("seq", ".".join(conf.mri_sequences) if len(conf.mri_sequences) != 4 else "all")])

    if args is not None:
        conf.update_with_args(args)

    if conf.model_name in [ModelName.simsiam, ModelName.simclr]:
        conf.train_mode = TrainMode(conf.model_name.value)

    conf.batch_size_eval = conf.batch_size

    if conf.overfit:
        conf.name += "_overfit"
    if is_debugging:
        conf.name += "_debug"

    conf.make_model_conf(args=args)
    return conf


def make_kv_name(key, val):
    return f"{key}-{val}"


def conf_mri_sequences(conf: TrainConfig):
    conf.mri_sequences = ("t1", "t1c", "t2", "flair")
    # conf.mri_sequences = ('flair', )
    conf.model_out_channels = conf.in_channels = len(conf.mri_sequences)


def pretrain_ffhq256_autoenc():
    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(
        _name="90M",
        _path=f"checkpoints/{ffhq256_autoenc().name}/last.ckpt",
    )
    conf.latent_infer_path = f"checkpoints/{ffhq256_autoenc().name}/latent.pkl"
    return conf


def pretrain_gliomapublic_autoenc(args=None):
    conf = gliomapublic_autoenc(args=args)

    conf.pretrain = PretrainConfig(
        gliomapublic_autoenc().name,
        f"{conf.logdir}/last.ckpt",
    )
    conf.latent_infer_path = f'{conf.logdir.replace("_debug","")}/latent.pkl'
    return conf
