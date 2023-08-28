import logging
from typing import Literal
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class Dataset_CSV(Dataset):
    def __init__(self, path, transform, split: None | Literal["train", "val", "test"] = None, col="file_path"):
        print(path)
        dataset = pd.read_csv(path)
        assert col in dataset
        assert not isinstance(transform, tuple)
        self.transform = transform
        self.dataset = dataset.loc[dataset["Split"] == split]
        self.dataset.reset_index()
        self.col = col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        from_im = Image.open(row[self.col])
        from_im = from_im.convert("L")
        from_im = self.transform(from_im)
        return {"img": from_im}  # , "index": target, "cls_labels": target}

    def get_extended_info(self, index):
        return self.dataset.iloc[index], *self.__getitem__(index)
