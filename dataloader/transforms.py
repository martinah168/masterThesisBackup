from torchvision import transforms
from enum import Enum, auto
from math import floor, ceil
from torch.nn import functional as F
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from utils.arguments import DataSet_Option
else:
    DataSet_Option = ()


class Transforms_Enum(Enum):
    random_crop = auto()
    resize = auto()
    RandomHorizontalFlip = auto()
    RandomVerticalFlip = auto()
    pad = auto()
    CenterCrop = auto()
    CenterCrop256 = auto()
    to_RGB = auto()


def get_transforms2D(opt: DataSet_Option, split):
    size = opt.img_size  # type: ignore
    if size is None:
        size: tuple[int, int] = (128, 128)
    return get_transforms(size, opt.transforms, split == "train")


def get_transforms(size: tuple[int, int], tf: list[Transforms_Enum] | None, train=False):
    if tf is None:
        tf = [Transforms_Enum.pad, Transforms_Enum.random_crop, Transforms_Enum.RandomHorizontalFlip]
    out: list = [transforms.ToTensor()]
    for t in tf:
        if isinstance(t, str):
            t = Transforms_Enum[t]
        if isinstance(t, int):
            t = Transforms_Enum(t)
        if t.value == Transforms_Enum.random_crop.value:
            out.append(transforms.RandomCrop(size)) if train else out.append(transforms.CenterCrop(size))
        elif t.value == Transforms_Enum.CenterCrop.value:
            out.append(transforms.CenterCrop(size))
        elif t.value == Transforms_Enum.CenterCrop256.value:
            out.append(transforms.CenterCrop(256))
        elif t.value == Transforms_Enum.resize.value:
            out.append(transforms.Resize(size))
        elif t.value == Transforms_Enum.RandomHorizontalFlip.value:
            out.append(transforms.RandomHorizontalFlip())
        elif t.value == Transforms_Enum.RandomVerticalFlip.value:
            out.append(transforms.RandomVerticalFlip())
        elif t.value == Transforms_Enum.pad.value:
            out.append(Pad(size))
        elif t.value == Transforms_Enum.to_RGB.value:
            out.append(to_RGB())
        else:
            raise NotImplementedError(t.name)
    out.append(transforms.Normalize(0.5, 0.5))
    return transforms.Compose(out)


class Pad:
    def __init__(self, size: tuple[int, int] | int) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        pass

    def __call__(self, image):
        w, h = image.shape[-2], image.shape[-1]
        max_w, max_h = self.size
        hp = max((max_w - w) / 2, 0)
        vp = max((max_h - h) / 2, 0)
        padding = (int(floor(vp)), int(ceil(vp)), int(floor(hp)), int(ceil(hp)))
        # print(padding,w,h)
        x = F.pad(image, padding, value=0, mode="constant")
        # print(x.shape)
        return x


class to_RGB:
    def __call__(self, image: torch.Tensor):
        if len(image) == 2:
            image = image.unsqueeze(0)
        if image.shape[-3] == 3:
            return image
        return torch.cat([image, image, image], dim=-3)
