"""Initialize a sweep for the glioma-sup-baseline project as a python script. """
import json
import os

import yaml

# Import the W&B Python Library and log into W&B
import wandb


def main():
    wandb.login()
    WANDB_PROJECT = 'glioma-sup-baseline-sweeps'

    # 2: Define the search space
    sweep_configuration = {
        'method': 'random',
        'metric': {
            'goal': 'minimize',
            'name': 'score'
        },
        'parameters': {
            'batch_size': {
                'value': 64
            },
            'lr': {
                'max': 0.01,
                'min': 1e-5
            },
            'stratified_sampling_mode': {
                'value': 'weighted'
            },
            'clf_arch': {
                'values': [
                    'resnet18',
                    'densenet',
                    'resnet50',
                ]
            },
            'data_aug_prob': {
                'min': 0.3,
                'max': 0.99,
            }
        },
        'program': 'sweep_wrapper.py',
        'run_cap': 100,
        'description': 'sweep to determine best data augmentation probability.'
    }

    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=WANDB_PROJECT)

    os.makedirs(f'sweeps/', exist_ok=True)

    # store sweep configuration

    with open(f'sweeps/sweep_{sweep_id}_config.yaml', 'w') as f:
        yaml.dump(sweep_configuration, f)

    print(json.dumps(sweep_configuration, indent=2))
    print(f'sweep id: {sweep_id}')


if __name__ == '__main__':
    main()
