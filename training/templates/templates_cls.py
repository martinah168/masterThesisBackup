from pathlib import Path
from training.config.train import PretrainConfig
from training.mode.clf import ClfMode
from training.mode.manipulate import ManipulateMode
from training.mode.model import ModelName
from training.mode.train import TrainMode
from training.templates.templates import ffhq256_autoenc, gliomapublic_autoenc


def ffhq256_autoenc_cls():
    '''We first train the encoder on FFHQ dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
    conf = ffhq256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'  # we train on Celeb dataset, not FFHQ
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.name = 'ffhq256_autoenc_cls'
    return conf


def gliomapublic_autoenc_cls(is_debugging, args, *p_args, **kwargs):
    conf = gliomapublic_autoenc(is_debugging, args, *p_args, **kwargs)

    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.gliomapublic

    # hyperparameters for experiments
    conf.clf_mode = ClfMode.one_vs_all
    conf.manipulate_cls = (0,)
    conf.manipulate_znormalize = conf.model_name == ModelName.beatgans_autoenc
    # end of hyperparameters for experiments

    if conf.pretrain_path:
        conf.latent_infer_path = conf.pretrain_path.replace(
            'last.ckpt', 'latent.pkl')
    else:
        conf.latent_infer_path = f'{conf.logdir.replace("_debug","")}/latent.pkl'

    # override autoencoder config
    _target_batch_size = max(32, conf.batch_size)
    conf.accum_batches = _target_batch_size // conf.batch_size  # ==> yields batch_size of _target_batch_size

    if is_debugging:
        _target_batch_size = 2
        conf.accum_batches = max(
            1, _target_batch_size //
            conf.batch_size)  # ==> yields batch_size of _target_batch_size
        conf.num_workers = 0

    conf.lr = 1e-3
    conf.total_samples = 2_000_000
    conf.clf_early_stopping_patience = 50

    conf.with_data_aug = True

    if not conf.pretrain_path:
        conf.pretrain_path = f'{conf.logdir}/last.ckpt'
    else:
        # get wandb id from pretrain path
        pretrain_dir = Path(conf.pretrain_path).parent
        with open(pretrain_dir / 'wandb_id.txt', 'r') as f:
            conf.wandb_id_pretrain = f.read().strip()

    conf.pretrain = PretrainConfig(
        conf.name,
        conf.pretrain_path,
    )

    if conf.clf_arch == 'vit':
        conf.weight_decay = 0.0001
        conf.warmup = 10  # in epochs
        conf.grad_clip = 1.
    else:
        conf.grad_clip = None
        conf.weight_decay = 0

    conf.overfit = False
    if conf.overfit:
        conf.accum_batches = 1

    conf.update_with_args(args)

    conf.name += '_cls'
    if conf.overfit:
        conf.name += '_overfit'
    # add classification mode and manipulate class to name
    if conf.use_healthy:
        conf.name += '_healthy'
    else:
        conf.name += f'_{conf.clf_mode.name}'

        if conf.clf_mode == ClfMode.one_vs_all:
            conf.name += f'_pos{conf.manipulate_cls[0]}'

        if conf.clf_mode == ClfMode.one_vs_one:
            conf.name += f'_{conf.manipulate_cls[0]}vs{conf.manipulate_cls[1]}'
    return conf
