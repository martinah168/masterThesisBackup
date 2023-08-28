import argparse
from typing import Union

from training.config.train import TrainConfig


def add():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--mode',
        type=str,
        default='rep',
    )
    parser.add_argument('--low_res',
                        action='store_true',
                        help='use low resolution images')

    # add sub parser for config options
    subparsers = parser.add_subparsers()
    config_parser(subparsers)

    args = parser.parse_args()
    return args


def config_parser(subparsers: Union[argparse._SubParsersAction, None] = None):
    if subparsers is None:
        # create new parser
        config_parser = argparse.ArgumentParser()
    else:
        config_parser = subparsers.add_parser('config')

    for k, v in TrainConfig().__dict__.items():
        if k.startswith('_') or callable(v):
            continue
        if v != None:
            v_type = type(v)
            if v_type == bool:
                v_type = str2bool
        else:
            v_type = None
        config_parser.add_argument(f'--{k}', type=v_type, default=None)

    return config_parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
