from torchvision import transforms
from enum import Enum, auto
from math import floor, ceil
from torch.nn import functional as F
from typing import TYPE_CHECKING
import torch
import numpy as np
from monai.transforms import RandAffine, RandSpatialCrop, Pad
import random
from math import ceil, floor
import numpy as np
from torch import Tensor

if TYPE_CHECKING:
    from utils.arguments import DataSet_Option
else:
    DataSet_Option = ()


class Transforms_Enum(Enum):
    #random_crop = auto()
    #resize = auto()
    #RandomHorizontalFlip = auto()
    #RandomVerticalFlip = auto()
    #pad = auto()
    # CenterCrop = auto()
    Flip_wLabels = auto()
    RandAffineTest = auto()
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
    size = (144, 96, 144)#opt.img_size  # type: ignore
    if size is None:
        size: tuple[int, int,int] = (114, 84, 114)#(112,82,112)# (144, 96, 144)
    return get_transforms( opt.transforms, opt.corpus , split == "train")
#mode=GridSampleMode.BILINEAR

def get_transforms( tf: list[Transforms_Enum] | None, corpus = False, train=False):#size: tuple[int, int, int],
    #if tf is None:
        #tf = [Transforms_Enum.pad, Transforms_Enum.random_crop]#, Transforms_Enum.RandomHorizontalFlip]
    size: tuple[int, int,int] = (144, 96, 144)
    out: list = [transforms.ToTensor()]
    
    if train: 
        #out.append(Pad(size=size))
        #out.append(RandSpatialCrop())#size))
        out.append(RandAffine(mode="nearest",translate_range=(5, 5, 5),rotate_range=(np.pi / 18, np.pi / 18, np.pi / 9),  # 20 degrees in radians
    scale_range=(0.8, 1.2),  # 20% scaling
    padding_mode="zeros", cache_grid = False))
        if not corpus:
            out.append(Flip_wLabels())
    return transforms.Compose(out)

#     for t in tf:
#         if isinstance(t, str):
#             t = Transforms_Enum[t]
#         if isinstance(t, int):
#             t = Transforms_Enum(t)
#         if t.value == Transforms_Enum.random_crop.value:
#             out.append(transforms.RandomCrop(size)) if train else out.append(transforms.CenterCrop(size))
#         # elif t.value == Transforms_Enum.CenterCrop.value:
#         #     out.append(transforms.CenterCrop(size))
#     #    elif t.value == Transforms_Enum.CenterCrop256.value:
#     #        out.append(transforms.CenterCrop(256))
#         # elif t.value == Transforms_Enum.resize.value:
#         #     out.append(transforms.Resize(size))
#         elif t.value == Transforms_Enum.RandAffineTest.value:
#             out.append(RandAffine(spatial_size=(144, 96, 144),mode="nearest",translate_range=(5, 5, 5),rotate_range=(np.pi / 18, np.pi / 18, np.pi / 9),  # 20 degrees in radians
#     scale_range=(0.8, 1.2),  # 20% scaling
#     padding_mode="zeros")) if train else None
# #        elif t.value == Transforms_Enum.RandomHorizontalFlip.value:
# #            out.append(transforms.RandomHorizontalFlip())
#         # elif t.value == Transforms_Enum.RandomVerticalFlip.value:
#         #    out.append(transforms.RandomVerticalFlip())
#         elif t.value == Transforms_Enum.pad.value:
#             out.append(Pad(size))
#         # elif t.value == Transforms_Enum.to_RGB.value:
#         #    out.append(to_RGB())
#         elif t.value == Transforms_Enum.Flip_wLabels.value:
#             out.append(Flip_wLabels()) if train else None
#         else:
#             raise NotImplementedError(t.name)
    #out.append(transforms.Normalize(0.5, 0.5))
    

class Flip_wLabels:
    def __call__(self, image: np.ndarray, dim = 2, tmp=60):
        return np_rhsubf(image=image, dim=2)

# class Pad:
#     def __init__(self, size: tuple[int, int] | int) -> None:
#         if isinstance(size, int):
#             size = (size, size)
#         self.size = size
#         pass

#     def __call__(self, image):
#         w, h = image.shape[-2], image.shape[-1]
#         max_w, max_h = self.size
#         hp = max((max_w - w) / 2, 0)
#         vp = max((max_h - h) / 2, 0)
#         padding = (int(floor(vp)), int(ceil(vp)), int(floor(hp)), int(ceil(hp)))
#         # print(padding,w,h)
#         x = F.pad(image, padding, value=0, mode="constant")
#         # print(x.shape)
#         return x



# rand_affine = RandAffine(
#     spatial_size=(144, 96, 144),
#     translate_range=(5, 5, 5),
#     rotate_range=(np.pi / 18, np.pi / 18, np.pi / 9),  # 20 degrees in radians
#     scale_range=(0.8, 1.2),  # 20% scaling
#     padding_mode="zeros",
# )
# rand_affine.set_random_state(seed=123)
    


_Shape = list[int] | torch.Size | tuple[int, ...]

mask_keys = ["msk", "seg"]
ignore_keys = [*mask_keys, "linspace"]

non_mri_keys = [*ignore_keys, "ct"]



def pad(x, mod: int):
    padding = []
    for dim in reversed(x.shape[1:]):
        padding.extend([0, (mod - dim % mod) % mod])
    x = F.pad(x, padding)
    return x


def pad_size(x: Tensor, target_shape, mode="constant"):
    while 1.0 * target_shape[-1] / x.shape[-1] > 2:
        x = pad_size(x, target_shape=[min(2 * a, b) for a, b in zip(x.shape[-len(target_shape) :], target_shape, strict=True)])
    while 1.0 * target_shape[-2] / x.shape[-2] > 2:
        x = pad_size(x, target_shape=[min(2 * a, b) for a, b in zip(x.shape[-len(target_shape) :], target_shape, strict=True)])
    padding = []
    for in_size, out_size in zip(reversed(x.shape[-2:]), reversed(target_shape), strict=True):
        to_pad_size = max(0, out_size - in_size) / 2.0
        padding.extend([ceil(to_pad_size), floor(to_pad_size)])
    x_ = (
        F.pad(x.unsqueeze(0).unsqueeze(0), padding, mode=mode).squeeze(0).squeeze(0)
    )  # mode - 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
    return x_


# def random_crop(target_shape: tuple[int, int], *arrs: torch.Tensor):
#     sli = [slice(None), slice(None)]
#     for i in range(2):
#         z = max(0, arrs[0].shape[-i] - target_shape[-i])
#         if z != 0:
#             r = random.randint(0, z)
#             r2 = r + target_shape[-i]
#             sli[-i] = slice(r, r2 if r2 != arrs[0].shape[-i] else None)

#     return tuple(a[..., sli[0], sli[1]] for a in arrs)


def calc_random_crop3D(target_shape: tuple[int, ...], shape: _Shape) -> tuple[slice, slice, slice]:
    sli = [slice(None), slice(None), slice(None)]
    for i in range(3):
        z = max(0, shape[-i] - target_shape[-i])
        if z != 0:
            r = random.randint(0, z)
            r2 = r + target_shape[-i]
            sli[-i] = slice(r, r2 if r2 != shape[-i] else None)

    return tuple(sli)  # type: ignore


def apply_random_crop3D(sli: tuple[slice, slice, slice], *arrs: torch.Tensor | None) -> tuple[Tensor, ...]:
    return tuple(a[..., sli[0], sli[1], sli[2]] if a is not None else None for a in arrs)  # type: ignore


# class Pad:
#     def __init__(self, size: list[int]):
#         self.mod = size
#         # else:
#         #    self.mod = 2**n_downsampling

#     def __call__(self, input_tensor2: dict[str, torch.Tensor]):
#         for key, x in input_tensor2.items():
#             padding = []
#             for dim, mod in zip(reversed(x.shape[-len(self.mod) :]), reversed(self.mod), strict=True):
#                 a = max((mod - dim), 0) / 2
#                 padding.extend([floor(a), ceil(a)])
#             input_tensor2[key] = F.pad(x, padding, mode="reflect")
#         return input_tensor2

#     def pad(self, x, n_downsampling: int = 1):
#         mod = 2**n_downsampling
#         padding = []
#         for dim in reversed(x.shape[1:]):
#             padding.extend([0, (mod - dim % mod) % mod])
#         x = F.pad(x, padding)
#         return x


class Crop3D:
    def __init__(self, size):
        self.size = size

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        example = next(iter(input_tensor2.values()))
        crop = calc_random_crop3D(self.size, example.shape[-3:])

        for key, x in input_tensor2.items():
            input_tensor2[key] = apply_random_crop3D(crop, x)[0]
        return input_tensor2

    def pad(self, x, n_downsampling: int = 1):
        mod = 2**n_downsampling
        padding = []
        for dim in reversed(x.shape[1:]):
            padding.extend([0, (mod - dim % mod) % mod])
        x = F.pad(x, padding)
        return x

