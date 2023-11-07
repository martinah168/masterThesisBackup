from torchvision import transforms
from enum import Enum, auto
from math import floor, ceil
from torch.nn import functional as F
from typing import TYPE_CHECKING
import torch
import numpy as np

if TYPE_CHECKING:
    from utils.arguments import DataSet_Option
else:
    DataSet_Option = ()


class Transforms_Enum(Enum):
    random_crop = auto()
    resize = auto()
    #RandomHorizontalFlip = auto()
    #RandomVerticalFlip = auto()
    pad = auto()
    CenterCrop = auto()
    Flip_wLabels = auto()
    #CenterCrop256 = auto()
    #to_RGB = auto()

def np_flip(x, dim):
    dim = x.ndim + dim if dim < 0 else dim
    return x[
        tuple(
            slice(None, None) if i != dim else np.arange(x.shape[i] - 1, -1, -1)
            for i in range(x.ndim)
        )
    ]
def np_rhsubf(image: np.ndarray, dim: int, tmp=60):
    assert (
        np.max(image) >= 1
    ), f"RHF without a image1 doesnt work! {np.min(image)}, {np.max(image)}"
    image_f = np.flip(image, dim)  # np_flip(image, dim=dim)
    for pair in [(43, 44), (45, 46), (47, 48)]:  # costalis, superior, inferior
        image_f[image_f == pair[0]] = tmp
        image_f[image_f == pair[1]] = pair[0]
        image_f[image_f == tmp] = pair[1]
    return image_f

def get_transforms2D(opt: DataSet_Option, split):
    size = opt.img_size  # type: ignore
    if size is None:
        size: tuple[int, int] = (128, 128)
    return get_transforms(size, opt.transforms, split == "train")


def get_transforms(size: tuple[int, int], tf: list[Transforms_Enum] | None, train=False):
    if tf is None:
        tf = [Transforms_Enum.pad, Transforms_Enum.random_crop]#, Transforms_Enum.RandomHorizontalFlip]
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
       # elif t.value == Transforms_Enum.CenterCrop256.value:
        #    out.append(transforms.CenterCrop(256))
        elif t.value == Transforms_Enum.resize.value:
            out.append(transforms.Resize(size))
       # elif t.value == Transforms_Enum.RandomHorizontalFlip.value:
        #    out.append(transforms.RandomHorizontalFlip())
        #elif t.value == Transforms_Enum.RandomVerticalFlip.value:
        #    out.append(transforms.RandomVerticalFlip())
        elif t.value == Transforms_Enum.pad.value:
            out.append(Pad(size))
        #elif t.value == Transforms_Enum.to_RGB.value:
         #   out.append(to_RGB())
        elif t.value == Transforms_Enum.Flip_wLabels.value:
            out.append(Flip_wLabels)
        else:
            raise NotImplementedError(t.name)
    #out.append(transforms.Normalize(0.5, 0.5))
    return transforms.Compose(out)

class Flip_wLabels:
    def __call__(self, image: np.ndarray, dim: int, tmp=60):
        return np_rhsubf(image=image, dim=0)

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
