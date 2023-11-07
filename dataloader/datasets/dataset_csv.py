import logging
from typing import Literal
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
#from BIDS import NII
from BIDS import NII
from BIDS.core.np_utils import np_map_labels, Label_Map

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

    #2D 
    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        img_nifti = nib.load(row[self.col])
        #Extract a specific 2D slice from the 3D volume (e.g., the middle slice along the z-axis)
        #print("img shape:", img_nifti.shape)
        z_slice = img_nifti.shape[2] // 2  # Choose the middle slice or adjust as needed
        from_im = img_nifti.get_fdata()[:, :, z_slice]  # Extract the 2D slice
        
        #reorient,rescale
        #print(img_nifti.header.get_data_shape())
        #from_im = img_nifti.get_fdata()
        #print(np.unique( from_im))
        from_im = np_map_labels(arr=from_im,label_map={50: 49})
        #from_im = self.map_labels(from_im)
        #print(np.unique( from_im))
        from_im = self.transform(from_im)
        from_im = from_im/49
        #print(np.unique( from_im))
        from_im = from_im.to(torch.float32) 
        return {"img": from_im}  

    #3D
    # def __getitem__(self, index):
    #         row = self.dataset.iloc[index]
    #         nii = NII.load(row[self.col], True)
    #         from_im = nii.get_seg_array()
    #         from_im = np_map_labels(arr=from_im,label_map={50: 49})
    #         #img_nifti = nib.load(row[self.col])

    #         #reorient,rescale
    #         #print(img_nifti.header.get_data_shape())
    #         #from_im = img_nifti.get_fdata()
    #         #from_im = from_im.reshape(-1, from_im.shape[-1])

            

    #         #from_im = self.transform(from_im)
    #         #print("from_im shape:", from_im.shape)
    #         from_im = from_im/49
    #         #print("from_im shape:", from_im.shape)
    #         from_im = torch.from_numpy(from_im).unsqueeze(0)
            
    #         from_im = from_im.to(torch.float32)
           
    #         #print("tesnor", from_im.shape)
    #         return {"img": from_im}  # , "index": target, "cls_labels": target}

    def get_extended_info(self, index):
        return self.dataset.iloc[index], *self.__getitem__(index)
    
    def map_labels(self, seg_arr):
        seg_arr = np_map_labels(arr=seg_arr,label_map={49: 9})
        seg_arr = np_map_labels(arr=seg_arr,label_map={48: 8})
        seg_arr = np_map_labels(arr=seg_arr,label_map={47: 7})
        seg_arr = np_map_labels(arr=seg_arr,label_map={46: 6})
        seg_arr = np_map_labels(arr=seg_arr,label_map={45: 5})
        seg_arr = np_map_labels(arr=seg_arr,label_map={44: 4})
        seg_arr = np_map_labels(arr=seg_arr,label_map={43: 3})
        seg_arr = np_map_labels(arr=seg_arr,label_map={42: 2})
        seg_arr = np_map_labels(arr=seg_arr,label_map={41: 1})
        return seg_arr

