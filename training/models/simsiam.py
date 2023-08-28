from dataclasses import dataclass

import monai.networks.nets
import torch.nn as nn

from training.config.base import BaseConfig


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, model_config: BaseConfig):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        dim = model_config.interm_dim
        pred_dim = model_config.out_dim

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = getattr(monai.networks.nets,
                               model_config.backbone_name)(num_classes=dim,
                                                           n_input_channels=4)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.fc[
            6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()


@dataclass
class SimSiamConfig(BaseConfig):
    # model params
    model_cls = SimSiam  # make_model() will use this
    backbone_name: str = 'resnet50'

    out_dim: int = 512
    interm_dim: int = 2048

    # optimizer parameters
    momentum: float = 0.9
    weight_decay: float = 1e-4
    fix_pred_lr: bool = True

    # scheduler parameters
    lr_sched_name: str = 'cosine'
    T_max: int = 100
    eta_min: float = 0.
    last_epoch: int = -1
