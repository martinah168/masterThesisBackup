import logging
from math import ceil
import random
from typing import Literal
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import torch.nn.functional as F
from resize.pytorch import resize
from degrade.degrade import fwhm_units_to_voxel_space, fwhm_needed
from degrade.degrade import select_kernel
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils.arguments import DataSet_Option


def get_patch(img_rot, patch_center, patch_size, return_idx=False):
    """
    img_rot: np.array, the HR in-plane image at a single rotation
    patch_center: tuple of ints, center position of the patch
    patch_size: tuple of ints, the patch size in 3D. For 2D patches, supply (X, Y, 1).
    """

    # Get random rotation and center
    sts = [c - p // 2 if p != 1 else c for c, p in zip(patch_center, patch_size)]
    ens = [st + p for st, p in zip(sts, patch_size)]
    idx = tuple(slice(st, en) for st, en in zip(sts, ens))

    if return_idx:
        return idx

    return img_rot[idx].squeeze()


from tqdm import tqdm


def get_random_centers(imgs_rot, patch_size, n_patches, weighted=True):
    rot_choices = np.random.randint(0, len(imgs_rot), size=n_patches)
    centers = []

    for i, img_rot in tqdm(enumerate(imgs_rot), total=len(imgs_rot), desc="random_centers"):
        n_choices = int(np.sum(rot_choices == i))

        if weighted:
            smooth = gaussian_filter(img_rot, 1.0)
            grads = np.gradient(smooth)
            grad_mag = np.sum([np.sqrt(np.abs(grad)) for grad in grads], axis=0)

            # Set probability to zero at edges
            for p, axis in zip(patch_size, range(grad_mag.ndim)):
                if p > 1:
                    grad_mag = np.swapaxes(grad_mag, 0, axis)
                    grad_mag[: p // 2 + 1] = 0.0
                    grad_mag[-p // 2 - 1 :] = 0.0
                    grad_mag = np.swapaxes(grad_mag, axis, 0)

            # Normalize gradient magnitude to create probabilities
            grad_probs = grad_mag / grad_mag.sum()
            grad_probs = [
                grad_probs.sum(axis=tuple(k for k in range(grad_probs.ndim) if k != axis)) for axis in range(len(grad_probs.shape))
            ]
            # Re-normalize per axis to ensure probabilities sum to 1
            for axis in range(len(grad_probs)):
                grad_probs[axis] = grad_probs[axis] / grad_probs[axis].sum()

        else:
            grad_probs = [None for _ in img_rot.shape]

        # Generate random patch centers for each dimension
        random_indices = [
            np.random.choice(
                np.arange(0, img_dim),
                size=n_choices,
                p=grad_probs[axis],
            )
            for axis, img_dim in enumerate(img_rot.shape)
        ]
        # Combine random indices to form multi-dimensional patch centers
        centers.extend((i, tuple(coord)) for coord in zip(*random_indices))
    np.random.shuffle(centers)
    return centers


def get_pads(target_dim, d):
    if target_dim <= d:
        return 0, 0
    p = (target_dim - d) // 2
    if (p * 2 + d) % 2 != 0:
        return p, p + 1
    return p, p


def target_pad(img, target_dims, mode="reflect") -> tuple[torch.Tensor, tuple]:
    pads = tuple(get_pads(t, d) for t, d in zip(target_dims, img.shape))
    return np.pad(img, pads, mode=mode), pads  # type: ignore


def parse_kernel(blur_kernel_file, blur_kernel_type, blur_fwhm):
    if blur_kernel_file is not None:
        blur_kernel: np.ndarray = np.load(blur_kernel_file)
    else:
        window_size = int(2 * round(blur_fwhm) + 1)
        blur_kernel = select_kernel(window_size, blur_kernel_type, fwhm=blur_fwhm)  # type: ignore
    blur_kernel /= blur_kernel.sum()
    blur_kernel = blur_kernel.squeeze()[None, None, :, None]
    blur_kernel_t = torch.from_numpy(blur_kernel).float()

    return blur_kernel_t


def calc_extended_patch_size(blur_kernel, patch_size):
    """
    Calculate the extended patch size. This is necessary to remove all boundary
    effects which could occur when we apply the blur kernel. We will pull a patch
    which is the specified patch size plus half the size of the blur kernel. Then we later
    blur at test time, crop off this extended patch size, then downsample.
    """

    L = blur_kernel.shape[0]

    ext_patch_size = [p + 2 * ceil(L / 2) if p != 1 else p for p in patch_size]
    ext_patch_crop = [(e - p) // 2 for e, p in zip(ext_patch_size, patch_size)]
    ext_patch_crop = tuple([slice(d, -d) for d in ext_patch_crop if d != 0])

    return ext_patch_size, ext_patch_crop


class Dataset_CSV_super(Dataset):
    """Argumentation like ins SMORE"""

    def __init__(
        self,
        opt: DataSet_Option,
        path,
        transform,
        split: None | Literal["train", "val", "test"] = None,
        col="file_path",
        resolutions=[
            (0.857, 5),
            (0.857, 5.2),
            (0.857, 6.2),
            (0.857, 6),
            (0.857, 5.8),
            (0.857, 7.6),
            (0.857, 7.2),
            (0.857, 8),
            (0.857, 7),
            (0.857, 10),
        ],
    ):
        print(path)
        self.opt = opt
        dataset = pd.read_csv(path)
        assert col in dataset, dataset
        assert not isinstance(transform, tuple)
        self.transform = transform
        self.dataset = dataset.loc[dataset["Split"] == split]
        self.dataset.reset_index()
        self.col = col
        self.hflip = False
        self.vflip = False
        self.kernel = []
        for hr, lr in resolutions:
            # Model the blurring-effect of a LR slice (lr x lr) compared to (hr,hr)
            # Axial images 0.5 mm x 0.5 mm x 5-6 mm
            # Sagittal (1.5 T) images 1.1 mm x 1.1 mm x 2-4 mm
            # Sagittal (3 T) images 0.95-0.8 mm x 0.95-0.8 mm x 2-4 mm
            blur_fwhm = fwhm_units_to_voxel_space(fwhm_needed(hr, lr), hr)
            slice_separation = float(lr / hr)
            self.kernel.append((parse_kernel(None, "rf-pulse-slr", blur_fwhm), slice_separation))
        self.patch_size = (self.opt.img_size, self.opt.img_size)

    def __len__(self):
        return len(self.dataset)

    def get_img(self, index, c=None):
        row = self.dataset.iloc[index]
        patch_hr = Image.open(row[self.col])
        patch_hr = np.array(patch_hr.convert("L")).astype(np.float32) / 255
        # patch_hr = self.transform(patch_hr)
        if c is None:
            blur_kernel, slice_separation = self.kernel[random.randint(0, len(self.kernel) - 1)]
        else:
            blur_kernel, slice_separation = self.kernel[c]
        ext_patch_size, ext_patch_crop = calc_extended_patch_size(blur_kernel, self.patch_size)

        # apply the pad
        patch_hr, pads = target_pad(patch_hr, ext_patch_size, mode="reflect")
        patch_hr = torch.from_numpy(patch_hr)
        patch_hr = patch_hr.unsqueeze(0).unsqueeze(1)
        patch_lr = F.conv2d(patch_hr, blur_kernel, padding="same")
        assert patch_hr.max() <= 1.0, patch_hr.max()
        assert patch_hr.min() >= 0.0, patch_hr.min()
        ext_patch_crop = (slice(None, None), slice(None, None), *ext_patch_crop)
        patch_hr = patch_hr[ext_patch_crop]
        patch_lr = patch_lr[ext_patch_crop]
        # patch_lr2 = patch_lr.clone()
        patch_lr: torch.Tensor = resize(patch_lr, (slice_separation, 1), order=3)  # type: ignore
        patch_lr: torch.Tensor = resize(patch_lr, (1 / slice_separation, 1), order=3)  # type: ignore

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(patch_hr, output_size=self.patch_size)
        patch_hr = TF.crop(patch_hr, i, j, h, w)
        patch_lr = TF.crop(patch_lr, i, j, h, w)
        # patch_lr2 = TF.crop(patch_lr2, i, j, h, w)

        # Random horizontal flipping
        if self.hflip and random.random() > 0.5:
            patch_hr = TF.hflip(patch_hr)
            patch_lr = TF.hflip(patch_lr)

        # Random vertical flipping
        if self.vflip and random.random() > 0.5:
            patch_hr = TF.vflip(patch_hr)
            patch_lr = TF.vflip(patch_lr)

        # Normalize to -1, 1
        patch_hr = patch_hr * 2 - 1
        patch_lr = patch_lr * 2 - 1
        # patch_lr2 = patch_lr2 * 2 - 1

        patch_hr = patch_hr.squeeze(0)
        patch_lr = patch_lr.squeeze(0)
        # patch_lr2 = patch_lr2.squeeze(0)
        return {"img": patch_hr, "img_lr": patch_lr}  # , "index": target, "cls_labels": target}

    def __getitem__(self, index):
        return self.get_img(index)

    def get_extended_info(self, index):
        return self.dataset.iloc[index], *self.__getitem__(index)
