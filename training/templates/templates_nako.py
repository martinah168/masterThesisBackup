import torch

import wandb
from training.config.train import PretrainConfig, TrainConfig
from training.data.mri import MriCrop
from training.mode.model import GenerativeType, ModelName
from training.mode.train import TrainMode
from training.models.simclr import SimCLRConfig

# beatgans_loss_type=<LossType.mse: 'mse'>, TODO L1?
# latent_beta_scheduler='linear', #TODO other scheduler???

"""
TrainConfig(wandb_project='nako'                                                      'gliomapublic',
            wandb_id='vjznzvyp',
            wandb_id_pretrain=None,
            seed=0,
            train_mode=<TrainMode.diffusion: 'diffusion'>,
            train_cond0_prob=0,
            train_pred_xstart_detach=True,
            train_interpolate_prob=0,
            train_interpolate_img=False,
            manipulate_mode=<ManipulateMode.celebahq_all: 'celebahq_all'>,
            manipulate_cls=(),
            manipulate_shots=None,
            manipulate_loss=<ManipulateLossType.bce: 'bce'>,
            manipulate_znormalize=False,
            manipulate_seed=0,
            clf_mode=<ClfMode.one_vs_all: 'one_vs_all'>,
            clf_arch='resnet18',
            clf_early_stopping_patience=100,
            split_ratio=0.9, TODO
            split_mode='study',
            ckpt_state='best',
            stratified_sampling_mode='uniform',
            accum_batches=32,
            autoenc_mid_attn=True,
            batch_size=1,
            batch_size_eval=1,
            beatgans_gen_type=<GenerativeType.ddim: 'ddim'>,
            beatgans_loss_type=<LossType.mse: 'mse'>, TODO L1?
            beatgans_model_mean_type=<ModelMeanType.eps: 'eps'>,
            beatgans_model_var_type=<ModelVarType.fixed_large: 'fixed_large'>,
            beatgans_rescale_timesteps=False,
            latent_infer_path=None,
            latent_znormalize=False,
            latent_gen_type=<GenerativeType.ddim: 'ddim'>,
            latent_loss_type=<LossType.mse: 'mse'>, TODO L1?
            latent_model_mean_type=<ModelMeanType.eps: 'eps'>,
            latent_model_var_type=<ModelVarType.fixed_large: 'fixed_large'>,
            latent_rescale_timesteps=False,
            latent_T_eval=1000,
            latent_clip_sample=False,
            latent_beta_scheduler='linear', #TODO other scheduler???
            beta_scheduler='linear', #TODO other scheduler???
            data_name='nako'                                                             'gliomapublic',
            data_val_name='',
            diffusion_type='beatgans',
            dropout=0.1, TODO impact?
            ema_decay=0.999,
            eval_num_images=200,
            eval_every_samples=200000,
            eval_ema_every_samples=200000,
            fid_use_torch=True,
            fp16=True,
            grad_clip=1.0,
            img_size=128, TODO SIZE!!!
            dims=3,
            in_channels=1                                                                 4,
            model_out_channels=1                                                          4,
            lr=0.0001,
            optimizer=<OptimizerType.adam: 'adam'>,
            weight_decay=0, TODO test if relevant?
            model_conf=BeatGANsAutoencConfig(image_size=--PROPAGATED--,
                                             in_channels=--PROPAGATED--,
                                             model_channels=64,
                                             out_channels=--PROPAGATED--,
                                             num_res_blocks=2,
                                             num_input_res_blocks=None,
                                             embed_channels=512,
                                             attention_resolutions=(16,),
                                             time_embed_channels=None,
                                             dropout=0.1,
                                             channel_mult=(1, 1, 2, 2),
                                             input_channel_mult=None,
                                             conv_resample=True,
                                             dims=3,
                                             num_classes=None,
                                             use_checkpoint=False,
                                             num_heads=1,
                                             num_head_channels=-1,
                                             num_heads_upsample=-1,
                                             resblock_updown=True,
                                             use_new_attention_order=False,
                                             resnet_two_cond=True,
                                             resnet_cond_channels=None,
                                             resnet_use_zero_module=True,
                                             attn_checkpoint=False,
                                             enc_out_channels=512,
                                             enc_attn_resolutions=None,
                                             enc_pool='adaptivenonzero',
                                             enc_num_res_block=2,
                                             enc_channel_mult=(1, 1, 2, 4, 4),
                                             enc_grad_checkpoint=False,
                                             latent_net_conf=None,
                                             lr_sched_name='lambda'),
            model_name=<ModelName.beatgans_autoenc: 'beatgans_autoenc'>,
            model_type=<ModelType.autoencoder: 'autoencoder'>,
            net_attn=(16,),
            net_beatgans_attn_head=1,
            embed_channels=512,
            net_resblock_updown=True,
            net_enc_use_time=False,
            net_enc_pool='adaptivenonzero',
            net_beatgans_gradient_checkpoint=False,
            net_beatgans_resnet_two_cond=True,
            net_beatgans_resnet_use_zero_module=True,
            net_beatgans_resnet_scale_at=<ScaleAt.after_norm: 'afternorm'>,
            net_beatgans_resnet_cond_channels=None,
            net_ch_mult=(1, 1, 2, 2),
            net_ch=64,
            net_enc_attn=None,
            net_enc_k=None,
            net_enc_num_res_blocks=2,
            net_enc_channel_mult=(1, 1, 2, 4, 4),
            net_enc_grad_checkpoint=False,
            net_autoenc_stochastic=False,
            net_latent_activation='silu',
            net_latent_channel_mult=(1, 2, 4),
            net_latent_condition_bias=0,
            net_latent_dropout=0,
            net_latent_layers=None,
            net_latent_net_last_act=None,
            net_latent_net_type=<LatentNetType.none: 'none'>,
            net_latent_num_hid_channels=1024,
            net_latent_num_time_layers=2,
            net_latent_skip_layers=None,
            net_latent_time_emb_channels=64,
            net_latent_use_norm=False,
            net_latent_time_last_act=False,
            net_num_res_blocks=2,
            net_num_input_res_blocks=None,
            net_enc_num_cls=None,
            num_workers=8,
            parallel=False,
            postfix='',
            sample_size=1,
            sample_every_samples=5000,
            save_every_samples=25000,
            style_ch=512,
            T_eval=20,
            T_sampler='uniform',
            T=1000,
            total_samples=100000000,
            warmup=0,
            with_data_aug=False,
            aug_encoder=False,
            pretrain=None,
            continue_from=None,
            eval_programs=None,
            eval_path=None,
            base_dir='checkpoints',
            version='2',
            pretrain_path='',
            full_fine_tuning=False,
            use_cache_dataset=False,
            data_cache_dir='/home/robert/cache',
            work_cache_dir='/home/robert/mycache',
            name='--PROPAGATED--_seq-all',
            timestamp='20230816_103121',
            overfit=False,
            test_only=False,
            mri_sequences=("T2w")                         ('t1', 't1c', 't2', 'flair'),
            mri_crop=TODO
            data_aug_prob=0.5,
            num_views=2)
"""


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
    raise NotImplementedError()
    conf = autoenc_base()
    conf.dims = 2
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


def nako_autoenc(is_debugging: bool = False, args=None, name="nako", dim=2, img_size=256):
    if dim == 2:
        conf = autoenc_base()
        conf.img_size = img_size  # crop size
        gpu_name = torch.cuda.get_device_name(0)
        _target_batch_size = 128
        if "NVIDIA GeForce RTX 3090" == gpu_name:
            max_batch_size_that_fits_in_memory = 16
        else:
            max_batch_size_that_fits_in_memory = 32
        if conf.img_size == 128:
            max_batch_size_that_fits_in_memory *= 4
    else:
        assert dim == 3
        conf = autoenc_base_3d()
        conf.img_size = 96  # crop size
        gpu_name = torch.cuda.get_device_name(0)
        if "A40" in gpu_name:
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

    conf.data_name = name
    conf.wandb_project = name + "_" + str(img_size)

    conf.num_workers = 16
    # train val split
    conf.net_ch = 64  # 64 best
    conf.net_attn = (16,)  # 16 best

    # final resolution = 8x8, diffusion resolution
    conf.net_ch_mult = (1, 1, 2, 2)
    assert conf.img_size // (2 ** len(conf.net_ch_mult)) in [6, 8, 16], conf.img_size // (2 ** len(conf.net_ch_mult))
    # final resolution = 4x4, encoder network final feat map resolution
    conf.net_enc_channel_mult = (1, 1, 2, 4, 4)
    assert conf.img_size // (2 ** len(conf.net_enc_channel_mult)) in [3, 4, 8], conf.img_size // (2 ** len(conf.net_enc_channel_mult))

    conf.overfit = False

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
    conf.mri_crop = MriCrop.CENTER
    # logging
    conf.sample_every_samples = 5_000
    conf.save_every_samples = 25_000
    conf.eval_num_images = 200
    conf.ema_decay = 0.999

    # train
    conf.clf_early_stopping_patience = 100
    conf.total_samples = 100_000_000

    conf.name = "_".join(
        [conf.data_name, make_kv_name("seq", ".".join(conf.mri_sequences) if len(conf.mri_sequences) != 4 else "all"), str(conf.img_size)]
    )

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
    conf.mri_sequences = ("T2w",)  # ("t1", "t1c", "t2", "flair")
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
