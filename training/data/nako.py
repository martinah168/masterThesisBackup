import logging
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Literal, Optional
from PIL import Image
from torchvision import transforms
import monai.transforms as mon_transforms
import numpy as np
import torch
from monai.transforms import LoadImaged, Padd, ScaleIntensity, SpatialPad, ToTensord
from torch.utils.data import Dataset
from torchvision.utils import save_image

from training.data.csv import read_csv_labels
from training.data.mri import MriCrop, calc_center_of_mass, extract_slices_from_volume
from training.data.transforms import CenterOfMassTumorCropd, RandHealthyOnlyCropd, TwoStageCenterOfMassTumorCropd, get_mri_aug
from training.mode.train import TrainMode
import pandas as pd

nib_logger = logging.getLogger("nibabel")
from math import floor, ceil


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
        x = torch.nn.functional.pad(image, padding, value=0, mode="constant")
        # print(x.shape)
        return x


class NAKO_JPG(Dataset):
    def __init__(
        self,
        path="",
        image_size=256,
        # original_resolution=256,
        split: None | Literal["train", "test"] = None,
        as_tensor: bool = True,
        do_augment: bool = True,
        do_normalize: bool = True,
        mri_sequences=("T2w"),
        mri_crop=MriCrop.CENTER,
        train_mode=TrainMode.diffusion,
        split_ratio=0.9,
        with_data_aug=False,
        aug_encoder=False,
        split_mode="study",
        data_aug_prob=0.5,
        target_label=["PatientAge"],
        classification=False,
        **kwargs,
    ):
        # def __init__(
        #    self,
        #    transform,
        #    dataset: pd.DataFrame,
        #    target_label: list[str] = ["PatientSex_t", "PatientAge_t", "PatientWeight_t", "PatientSize_t"],
        #    classification=True,
        #    split="train",
        #    names=[True, False],
        # ):
        print(path)
        train = split == "train"
        dataset = pd.read_csv(path)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                Pad(image_size),
                transforms.RandomCrop(image_size) if train else transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(0.5, 0.5),
            ]
        )
        assert "file_path" in dataset
        assert not isinstance(transform, tuple)
        self.transform = transform
        self.target_label = target_label
        # self.source_paths = sorted(make_dataset("/media/data/pohl/GAN/middle_folder"))
        self.dataset = dataset.loc[dataset["Split"] == split]
        self.dataset.reset_index()
        for t in target_label:
            assert t in dataset, (t, self.dataset.head())

        self.classification = classification
        if classification:
            self.names = target_label
            self.labels = [len(dataset[i].unique()) for i in target_label]
            if split == "train":
                print("Different values of the several Features: \n", self.labels)
                print(dataset[target_label[0]].value_counts())
        else:
            self.min_v = [dataset[i].min() for i in target_label]
            self.max_v = [dataset[i].max() for i in target_label]
            if split == "train":
                print("Different (min,max) of the several Features: \n", list(zip(self.min_v, self.max_v)))
                # print(dataset[target_label[0]].hist())
            # mapping for all keywords

            pass
        # TODO multiclass support
        self.num_classes = self.max_v[0] - self.min_v[0]

    def __len__(self):
        return len(self.dataset)

    def convert_back(self, value, id):
        return (value + 1) * 0.5 * (self.max_v[id] - self.min_v[id]) + self.min_v[id]

    def __getitem__(self, index):
        row = self.dataset.iloc[index]

        from_im = Image.open(row["file_path"])
        from_im = from_im.convert("L")
        from_im = np.asarray(from_im).copy()

        # from_im = transforms.ToTensor()(from_im)

        if self.transform:
            from_im = self.transform(from_im)
        target = []
        if type(self.target_label) == str:
            target = [row[self.target_label]]

        if type(self.target_label) == list:
            target = list()
            for i, label in enumerate(self.target_label):
                target.append(row[label])
        if not self.classification:
            for i, (min_v, max_v) in enumerate(zip(self.min_v, self.max_v)):
                target[i] = ((target[i] - min_v) / (max_v - min_v) * 2) - 1
            if len(target) > 1:
                target = np.concatenate(target)
            else:
                target = target[0].reshape(-1)
            target = target.astype(np.float32)
        # TODO revert to classification
        target = np.round((target + 1) / 2 * self.num_classes)
        return {"img": from_im, "index": target, "cls_labels": target}

    def get_extended_info(self, index):
        return self.dataset.iloc[index], *self.__getitem__(index)
