import argparse

import torch.cuda
import torch.multiprocessing

import training.args as args
import training.experiments.rep as rep
from training.config.train import TrainConfig
from training.experiments.cls import run_cls
from training.templates.templates import gliomapublic_autoenc
from training.templates.templates_nako import nako_autoenc
from training.templates.templates_cls import gliomapublic_autoenc_cls
from training.templates.templates_latent import gliomapublic_autoenc_latent
from dataclasses import dataclass


@dataclass()
class ARGS:
    low_res = False


def main(debug=False, nako=True):
    cli_args = ARGS()
    mode: str = "rep"
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
    if nako:
        conf = nako_autoenc(is_debugging=debug, args=cli_args)
    else:
        conf = gliomapublic_autoenc(is_debugging=debug, args=cli_args)
    import pprint

    # pprint.pprint(conf)
    score = rep.train(conf, gpus=gpus, nodes=nodes)
    return score


if __name__ == "__main__":
    print("score", main())
