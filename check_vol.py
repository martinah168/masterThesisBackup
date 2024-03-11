from pathlib import Path
from pl_models.DEA import DAE_LitModel
import matplotlib.pyplot as plt
import nibabel as nib
import torch
from torchvision import transforms
from BIDS import NII
from BIDS.core.np_utils import np_dice
import numpy as np
from BIDS.core.np_utils import np_map_labels, Label_Map
from sklearn.manifold import TSNE
import torchvision
from dataloader.dataset_factory import get_data_loader, get_dataset
from torch.utils.data import DataLoader
from utils import arguments
import tqdm
import re
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import re
from BIDS.core import vert_constants
from BIDS.core.np_utils import np_volume
from dataloader.datasets.dataset_csv import model_map_to_segmentation_map, segmentation_map_to_model_map 
#from utils.metadata import make_anomaly_dict, add_fxclass_fracture_anomaly, parse_string
import utils.metadata as m
from BIDS.core.vert_constants import v_name2idx, v_idx2name





def calc_volume(seg_arr):
    seg = model_map_to_segmentation_map(seg_arr)
    labels = np.unique(seg_arr)
    labels = model_map_to_segmentation_map(labels)[1:]
    vol_dict = np_volume(seg, labels)
    total_volume = sum(vol_dict.values())
    return total_volume

def extract_embeddings(train_dataloader):#, val_dataloader):
    data_list = []
    i = 0
    with torch.no_grad():
        for x in train_dataloader:
            i = i +1
            print(i)
            img = x['img'].to(device)
            label = x['label']
            subject = x['subject']
            file = x['filepath']
            #output = model.encode(img)

            
            label_str = str(label)
            subject_str = str(subject)
            #print(subject_str)
            # if (int(extract_nr(subject_str)) == 526 and int(extract_nr(label)) == 25):# or (subject_str == "['verse534']" and label == "['025']") or (subject_str == "['verse617']" and label == "['024']"):
            #     print("skipped")
            #     continue 
            #region = get_region(int(extract_nr(label_str)))
            volume = calc_volume(img.cpu().numpy())

            data_item = {
                #"embeddings": output,  
                "label": label_str,
                "subject": subject_str,
                "file_path": file,
                "volume": volume
            }
            i = i +1
            print(i)

            data_list.append(data_item)
        # for x in val_dataloader:
        #     img = x['img'].to(device)
        #     label = x['label']
        #     subject = x['subject']
        #     file = x['filepath']
        #     #output = model.encode(img)

            
        #     label_str = str(label)
        #     subject_str = str(subject)
        #     #print(subject_str)
        #     # if (int(extract_nr(subject_str)) == 526 and int(extract_nr(label)) == 25):# or (subject_str == "['verse534']" and label == "['025']") or (subject_str == "['verse617']" and label == "['024']"):
        #     #     print("skipped")
        #     #     continue 
        #    # region = get_region(int(extract_nr(label_str)))
        #     volume = calc_volume(img.cpu().numpy())

        #     data_item = {
        #         #"embeddings": output,  
        #         "label": label_str,
        #         "subject": subject_str,
        #         "file_path": file,
        #         "volume": volume
        #     }
        #     i = i +1
        #     print(i)
        #     data_list.append(data_item)


    # Convert the list of dictionaries to a DataFrame
    dataframe = pd.DataFrame(data_list)

    return dataframe

if __name__ == '__main__':
    # Example usage:
    opt = arguments.DAE_Option().get_opt(None, None)
    arguments.DataSet_Option.dataset = "/media/DATA/martina_ma/dae/test_set_cleaned_and_not_filtered.csv"
    dataset = get_dataset(arguments.DataSet_Option,"test")
    #dataset_val = get_dataset(arguments.DataSet_Option,"val")
    part_tr = dataset #torch.utils.data.Subset(dataset, subset_indices)
    #datset = dataset.
    # train_dataloader = DataLoader(
    #         part_tr,
    #         batch_size=1,#opt.batch_size,
    #         #sampler=sampler,
    #         # with sampler, use the sample instead of this option
    #         shuffle=False,# if sampler else shuffle,
    #         num_workers=16,#opt.num_cpu,
    #         pin_memory=True,
    #         #drop_last=drop_last,
    #         #multiprocessing_context=get_context("fork"),
    #         persistent_workers=True,
    #     )
    # val_dataloader = DataLoader(
    #         dataset_val,
    #         batch_size=1,#opt.batch_size,
    #         #sampler=sampler,
    #         # with sampler, use the sample instead of this option
    #         shuffle=False,# if sampler else shuffle,
    #         num_workers=16,#opt.num_cpu,
    #         pin_memory=True,
    #         #drop_last=drop_last,
    #         #multiprocessing_context=get_context("fork"),
    #         persistent_workers=True,
    #     )
    test_dataloader = DataLoader(
            dataset,
            batch_size=1,#opt.batch_size,
            #sampler=sampler,
            # with sampler, use the sample instead of this option
            shuffle=False,# if sampler else shuffle,
            num_workers=16,#opt.num_cpu,
            pin_memory=True,
            #drop_last=drop_last,
            #multiprocessing_context=get_context("fork"),
            persistent_workers=True,
        )
    device = "cuda:0"  # or "cpu" if you want to use CPU
    embeddings_dataframe = extract_embeddings(test_dataloader)#train_dataloader, val_dataloader)
    torch.save(embeddings_dataframe,'/media/DATA/martina_ma/test_vol.pt')
#embeddings_dataframe.to_csv('emb_test.csv', index=False)