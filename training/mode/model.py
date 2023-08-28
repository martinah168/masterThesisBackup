from enum import Enum


class ModelVarType(str, Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    # posterior beta_t
    fixed_small = 'fixed_small'
    # beta_t
    fixed_large = 'fixed_large'


class ModelMeanType(str, Enum):
    """
    Which type of output the model predicts.
    """

    eps = 'eps'


class ModelName(str, Enum):
    """
    List of all supported model classes
    """

    beatgans_ddpm = 'beatgans_ddpm'
    beatgans_autoenc = 'beatgans_autoenc'
    simclr = 'simclr'
    simsiam = 'simsiam'


class ModelType(str, Enum):
    """
    Kinds of the backbone models
    """

    # unconditional ddpm
    ddpm = 'ddpm'
    # autoencoding ddpm cannot do unconditional generation
    autoencoder = 'autoencoder'

    def has_autoenc(self):
        return self in [
            ModelType.autoencoder,
        ]

    def can_sample(self):
        return self in [ModelType.ddpm]


class GenerativeType(str, Enum):
    """
    How's a sample generated
    """

    ddpm = 'ddpm'
    ddim = 'ddim'
