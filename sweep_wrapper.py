# Import the W&B Python Library and log into W&B
import wandb
from run import main as run_glioma_public
from training import args
from training.mode.clf import ClfMode
from training.mode.train import TrainMode

wandb.login()
WANDB_PROJECT = 'glioma-sup-baseline-sweeps'


def main():
    wandb.init(project=WANDB_PROJECT)
    parser = args.config_parser()
    sweep_args = parser.parse_args()
    sweep_args.wandb_project = WANDB_PROJECT

    # replacement for CLI args

    fixed_config = {
        # 'mode': 'cls',
        # 'debug': False,
        'train_mode': TrainMode.supervised,
        'use_healthy': False,
        'clf_mode': ClfMode.multi_class,
        'wandb_project': WANDB_PROJECT,
        'wandb_id': wandb.run.id,
        'mode': 'cls',
    }
    for k, v in fixed_config.items():
        setattr(sweep_args, k, v)

    score = run_glioma_public(sweep_args)

    wandb.log({'score': score})


if __name__ == '__main__':
    main()
