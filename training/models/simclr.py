from dataclasses import dataclass

import monai.networks.nets
import torch.nn as nn

from training.config.base import BaseConfig


class SimCLRModel(nn.Module):

    def __init__(self, model_config: BaseConfig):
        super(SimCLRModel, self).__init__()
        out_dim = model_config.out_dim
        backbone_name = model_config.backbone_name

        # TODO: check if this works
        self.backbone = getattr(monai.networks.nets,
                                backbone_name)(num_classes=model_config.out_dim,
                                               n_input_channels=4)

        assert self.backbone.fc is not None, 'backbone must have fc layer'

        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            self.backbone.fc,
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # additional layer in projection head to deal with smaller batch sizes (https://arxiv.org/pdf/2011.02803.pdf)
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.backbone(x)


@dataclass
class SimCLRConfig(BaseConfig):
    model_cls = SimCLRModel  # make_model() will use this

    # model parameters
    backbone_name: str = 'resnet50'
    out_dim: int = 512

    # training parameters
    batch_size: int = 8  # should ideally be 512
    learning_rate: float = 0.0003
    weight_decay: float = 1e-4

    # scheduler parameters
    lr_sched_name: str = 'cosine'
    T_max: int = 100
    eta_min: float = 0.
    last_epoch: int = -1
