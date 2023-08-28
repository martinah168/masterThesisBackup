from enum import Enum


class OptimizerType(str, Enum):
    adam = 'adam'
    adamw = 'adamw'
    sgd = 'sgd'
