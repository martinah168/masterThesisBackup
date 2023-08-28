from typing import Union

from .latentnet import MLPSkipNet, MLPSkipNetConfig

from .unet import BeatGANsUNetConfig, BeatGANsUNetModel
from .unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel

Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel, MLPSkipNet]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig, MLPSkipNetConfig]
