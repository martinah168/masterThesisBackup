import argparse

import torch.cuda
import torch.multiprocessing

import training.args as args
import training.experiments.rep as rep
from training.config.train import TrainConfig
from training.experiments.cls import run_cls
from training.templates.templates import gliomapublic_autoenc
from training.templates.templates_cls import gliomapublic_autoenc_cls
from training.templates.templates_latent import gliomapublic_autoenc_latent


def main(cli_args: argparse.Namespace = None):
    is_sweep = cli_args is not None
    if not is_sweep:
        cli_args: argparse.Namespace = args.add()

    mode: str = cli_args.mode  # 'latent'  # "autoenc", "cls" or "latent"
    if is_sweep:
        # default to low res images for sweeps
        cli_args.low_res = True
        cli_args.debug = False

    assert mode in ["rep", "cls", "latent"], f"unknown mode {mode}"
    print(f"mode: {mode}".upper())

    if torch.cuda.is_available():
        gpus = [*range(torch.cuda.device_count())]
    else:
        gpus = []
    nodes = 1

    n_gpus: int = torch.cuda.device_count()
    if n_gpus > 0:
        print(f"using gpus ({n_gpus}): { [torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
    else:
        print("using cpu")

    conf = {"rep": gliomapublic_autoenc, "cls": gliomapublic_autoenc_cls, "latent": gliomapublic_autoenc}[mode](
        is_debugging=cli_args.debug, args=cli_args
    )
    conf: TrainConfig

    if mode == "latent":
        # update config for latent space diffusion training
        conf.from_dict(
            {
                "eval_programs": ["infer"],
                "eval_path": f'{conf.logdir.replace("_debug","")}/last.ckpt',
                "batch_size": conf.batch_size * 2,
            }
        )

    if is_sweep:
        # conf.update_sweep(fixed_config)
        # conf.update_sweep(sweep_config)
        conf.wandb_id = cli_args.wandb_id
        conf.version = cli_args.wandb_id

    if is_sweep and mode != "cls":
        raise NotImplementedError("Sweeping is only supported for classification")

    train_fn = {
        "rep": lambda: rep.train(conf, gpus=gpus, nodes=nodes),
        "cls": lambda: run_cls(conf, gpus=gpus),
        "latent": lambda: rep.train(conf, gpus=gpus, mode="eval"),
    }[mode]
    score = train_fn()

    if mode == "latent":
        conf = gliomapublic_autoenc_latent(args=cli_args)
        rep.train(conf, gpus=gpus)

    return score


if __name__ == "__main__":
    main()
