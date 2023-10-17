import logging
from typing import Literal
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib


class Dataset_CSV(Dataset):
    def __init__(self, path, transform, split: None | Literal["train", "val", "test"] = None, col="file_path"):
        print(path)
        dataset = pd.read_csv(path, sep=";")
        assert col in dataset, dataset
        assert not isinstance(transform, tuple)
        self.transform = transform
        self.dataset = dataset.loc[dataset["Split"] == split]
        self.dataset.reset_index()
        self.col = col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        img_nifti = nib.load(row[self.col])

        #reorient,rescale
        print(img_nifti.header.get_data_shape())
        from_im = img_nifti.get_fdata()#.unsqueeze(1).float()
   #     from_im = read_image(row[self.col])#Image.open(row[self.col])
    #    from_im = from_im.convert("L")
        # add dimension for 3d conv
# Flatten the 3D image into a 2D image
        from_im = from_im.reshape(-1, from_im.shape[-1])

        

        from_im = self.transform(from_im)
        from_im = from_im.to(torch.float32) 
        # Convert to a 2D tensor
       # from_im = torch.from_numpy(from_im).float()
        #single_image_arrary = np.expand_dims(from_im, axis=0)
        
        #single_image_arrary = single_image_arrary.astype(np.float32)
        # to tensor
        #from_im = torch.from_numpy(single_image_arrary)
        return {"img": from_im}  # , "index": target, "cls_labels": target}

    def get_extended_info(self, index):
        return self.dataset.iloc[index], *self.__getitem__(index)
