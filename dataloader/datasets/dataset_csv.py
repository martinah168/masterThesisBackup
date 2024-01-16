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
import re
import os

outliers = ["verse553", "verse526", "verse534","verse617"]

class Dataset_CSV(Dataset):
    def __init__(self, path, transform, split: None | Literal["train", "val", "test"] = None, col="file_path"):
        print(path)
        dataset = pd.read_csv(path, sep=",")
        #dataset = dataset[~dataset['file_path'].str.contains('ctfu')]
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
        folder , label = extract_label(row.file_path)
        nii = NII.load(row[self.col], True)
        from_im = nii.get_seg_array()
        #print("re_y:",from_im[143,:,143])
        if from_im.ndim == 2:
             item = self.get_item_2D(from_im)
        elif from_im.ndim == 3:
             item = self.get_item_3D(from_im, folder, label)
        else:
             raise NotImplementedError()
        # prepared_item = item["img"]
        # re = prepared_item.squeeze(0).numpy()
        # print(np.unique(re))
        # re = re*9 
        # re = np.round(re)
        # re = re + 40
        # Print indexes where the value is not zero
        #nonzero_indexes = np.nonzero(re)
       # print("Nonzero indexes:", nonzero_indexes)
        # Write nonzero indexes to a file
        # with open("nonzero_indexes.txt", "a") as file:
        #     file.write(f"Index {index} - Nonzero indexes: {nonzero_indexes}\n")
        # For better readability, you can print the corresponding values
        #nonzero_values = re[nonzero_indexes]
        #print("Corresponding nonzero values:", nonzero_values)
        # re = re*9 
        # re = np.round(re)
        # re = re + 40
        # re = np_map_labels(re, {40: 0})
        # print(np.unique(re))
       # print("re_all_y:",re[0,:,0])
        #print("re_all_x:",re[:,0,0])
        #print("re_all_z:",re[0,0,:])
        #print(re)
        # if index == 373:
        #     nii.set_array_(re).save("prepared_item.nii.gz") 
        #print(type(item))
        return item
             

   # 2D 
    def get_item_2D(self, from_im):
        #row = self.dataset.iloc[index]
        #img_nifti = NII.load(row[self.col], True)
        z_slice = from_im.shape[2] // 2  # Choose the middle slice or adjust as needed
        from_im = from_im[:, :, z_slice] 
        from_im = from_im.astype(float)
        from_im = np_map_labels(arr=from_im,label_map={50: 49})
        #reorient,rescale
        from_im = self.transform(from_im)
        from_im = from_im.numpy()
        from_im = np_map_labels(arr=from_im,label_map={50: 49})
        from_im = self.segmentation_map_to_model_map(from_im)
        #from_im = from_im/49
        #print(np.unique( from_im))
        #print(type(from_im))
        #from_im = from_im.to(torch.float32) 
        from_im = torch.from_numpy(from_im)
        #from_im = from_im.unsqueeze(0)
        #print(from_im.shape)
        from_im = from_im.to(torch.float32) 
        return {"img": from_im}  
    

    #3D
    def get_item_3D(self, from_im, folder, vert):
            from_im = np_map_labels(arr=from_im,label_map={50: 49})
            from_im = from_im.astype(float)
            from_im = self.transform(from_im)
            #print("from_im shape:", from_im.shape)
            from_im = from_im.numpy()
            #from_im = np_map_labels(arr=from_im,label_map={50: 49})
            from_im = segmentation_map_to_model_map(from_im)# self.segmentation_map_to_model_map(from_im)
            #print("from_im shape:", from_im.shape)
            #from_im = (from_im - 0.5) / 0.5
            from_im = torch.from_numpy(from_im).unsqueeze(0)
            
            from_im = from_im.to(torch.float32)
           
            #print("tensor", from_im.shape)
            return {"img": from_im, "label": vert , "subject": folder}  # , "index": target, "cls_labels": target}

    def get_extended_info(self, index):
        return self.dataset.iloc[index], *self.__getitem__(index)
    

    
    def map_to_binary(self,y):
        #y = y.astype(float)
        labelmap = {i: 1 for i in range(41, 50)} 
        #print(labelmap)
        return np_map_labels(y, labelmap)

def segmentation_map_to_model_map(y):
    #y = y.astype(float)
    labelmap = {i: round((i - 40)/9, ndigits=5) for i in range(41, 50)} #  labelmap = {i: round((i - 40)/9, ndigits=2) for i in range(41, 50)}
    #print(labelmap)
    y =  np_map_labels(y, labelmap)
    y = (y-0.5)/0.5
    return y


def model_map_to_segmentation_map(x):
    x = (x+1)/2
    x *= 9
    x = np.round(x+40)
    x = np_map_labels(x, {40: 0})
    return x

def range_print(x, lb, ub):
    # Find values and indices within the range
    condition = (x >= lb) & (x <= ub)
    values_in_range = x[condition]
    indices_in_range = torch.nonzero(condition)

    # Print the values and indices
    print("Values in the range:", values_in_range)
    print("Indices in the range:", indices_in_range)

def extract_label(file_path):
    parent_folder = os.path.basename(os.path.dirname(file_path))
    pattern = r"_([0-9]+)_"
    # Use re.search to find the match in the file path
    match = re.search(pattern, file_path)
    extracted_number = None
    # Check if a match is found
    if match:
        # Extract the matched number
        extracted_number = match.group(1)
        #print("Extracted Number:", extracted_number)
    else:
        print("No match found.")
    #result = 
    return parent_folder , extracted_number 