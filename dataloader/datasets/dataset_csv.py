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
        dataset = pd.read_csv(path, sep=",", index_col=0)
        #dataset = dataset[~dataset['file_path'].str.contains('ctfu')]
        assert col in dataset, dataset
        assert not isinstance(transform, tuple)
        self.transform = transform
        self.dataset = dataset.loc[dataset["Split"] == split]
        self.dataset.reset_index()
        self.col = col
    
    def sample_weights(self):#sample_weights
        class_counts = self.dataset.fracture_flag.value_counts()
        class_weights = 1 / class_counts
        sample_weights = [class_weights[i] for i in self.dataset.fracture_flag.values]
        #print(class_weights)
        return sample_weights

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        folder , label = extract_label(row.file_path)
        corpus = "corpus" in row.file_path
        split = row["Split"]
        nii = NII.load(row[self.col], True)
        from_im = nii.get_seg_array()
        if np.max(from_im) < 1:
            print(row.filepath,from_im)
        #print("re_y:",from_im[143,:,143])
        if from_im.ndim == 2:
             item = self.get_item_2D(from_im)
        elif from_im.ndim == 3:
             item = self.get_item_3D(from_im, folder, label, row.file_path,split,row.fracture_flag,corpus)#, row.fracture_flag)
        else:
             raise NotImplementedError()
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
    def get_item_3D(self, from_im, folder, vert, filepath, split,frac, corpus ):#,frac):
        if corpus:
            from_im = corpus_to_model_map(from_im)
        else:
            from_im = np_map_labels(arr=from_im,label_map={50: 49})
        from_im = from_im.astype(float)
        x = np.unique(from_im)
        from_im = self.transform(from_im)
        #nii.set_array_(from_im).save("Check_transform_folder_vert.nii.gz")
        #print("from_im shape:", from_im.shape)
        if split == 'val':
            from_im = from_im.numpy()
        elif corpus:
            from_im = from_im.numpy()
        #from_im = np_map_labels(arr=from_im,label_map={50: 49})
        if not corpus: 
            from_im = segmentation_map_to_model_map(from_im)# self.segmentation_map_to_model_map(from_im)
            #print("from_im shape:", from_im.shape)
            #from_im = (from_im - 0.5) / 0.5
        from_im = torch.from_numpy(from_im).unsqueeze(0)
        
        from_im = from_im.to(torch.float32)
        
        #print("tensor", from_im.shape)
        return {"img": from_im, "label": vert , "subject": folder, "filepath": filepath, "fracture":frac}  # , "index": target, "cls_labels": target}

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

def corpus_to_model_map(y):
    y = np_map_labels(arr=y,label_map={49: 1})
    y = (y-0.5)/0.5
    return y

def model_map_to_corpus(x):
    x = (x+1)/2
    x = np.round(x)
    x = np_map_labels(x, {1: 49})
    return x

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
    if parent_folder == '':
        parent_folder = file_path.split('_')[0]
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
        #print("No match found.")
        # Input string
        input_string = file_path#'fxclass0410_ 9_subreg_cropped_cleaned.nii.gz'

        # Regular expression pattern to match a single digit
        pattern = r'\s(\d+)_'

        # Find all matches in the input string
        match = re.search(pattern, input_string)

        if match:
        # Extract the matched number
            extracted_number = match.group(1)
        #print("Extracted Number:", extracted_number)
        else:
            print("No match found.")
    #result = 
    return parent_folder , extracted_number 