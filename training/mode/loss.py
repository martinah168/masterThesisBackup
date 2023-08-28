from enum import Enum


class ManipulateLossType(str, Enum):
    bce = 'bce'
    mse = 'mse'


class LossType(str, Enum):
    mse = 'mse'  # use raw MSE loss (and KL when learning variances)
    l1 = 'l1'
