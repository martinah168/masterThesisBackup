import os
import sys
from pathlib import Path
from typing import Literal
import pandas as pd
import torch

sys.path.append("..")
from utils.arguments import DataSet_Option, DAE_Option
import pandas as pd
from torchvision.datasets import MNIST
from dataloader.transforms import get_transforms2D
from .datasets.dataset_csv import Dataset_CSV
from multiprocessing import get_context


def get_num_channels(opt: DataSet_Option):
    return 1


SPLIT = Literal["train", "val", "test"]


def get_dataset(opt: DataSet_Option, split: SPLIT = "train"):
    dataset = opt.dataset
    ds_type = opt.ds_type
    if ds_type == "csv_2D":
        assert dataset.endswith(".csv"), dataset + " is not a csv"
        transf1 = get_transforms2D(opt, split)
        return Dataset_CSV(path=dataset, transform=transf1, split=split)
    else:
        raise NotImplementedError(ds_type)


def get_data_loader(
    opt: DAE_Option,
    dataset,
    shuffle: bool,
    drop_last: bool = True,
    parallel=False,  # FIXME always False?
    split: SPLIT = "train",
):
    from torch import distributed
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if parallel and distributed.is_initialized():
        # drop last to make sure that there is no added special indexes
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
    elif hasattr(dataset, "sample_weights") and split == "train" and opt.train_mode.is_manipulate():
        print("using weighted sampler for imbalanced data")
        sampler = WeightedRandomSampler(dataset.sample_weights(), len(dataset))
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
        # with sampler, use the sample instead of this option
        shuffle=False if sampler else shuffle,
        num_workers=opt.num_cpu,
        pin_memory=True,
        drop_last=drop_last,
        multiprocessing_context=get_context("fork"),
        persistent_workers=True,
    )
