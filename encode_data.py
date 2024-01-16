from scipy import ndimage
from BIDS import BIDS_Global_info, NII
from BIDS.core.np_utils import np_calc_crop_around_centerpoint, np_bbox_nd
#from BIDS import NII
import nibabel as nib
from pathlib import Path
#from BIDS.core.poi import load_poi, VertebraCentroids
from IPython.display import Image
import csv
import os
import re
import torch
from pl_models.DEA import DAE_LitModel
import matplotlib.pyplot as plt
import nibabel as nib
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


# Input string match = re.search(r'\b\d+\b', string)
def extract_nr(pattern_string):
    # Use re.search to find the match in the string
    match = re.search(r'\b\d+\b', pattern_string)
    # If no match is found, try a modified pattern
    if not match:
        match = re.search(r'\b\d+\b', re.sub(r'[^\d]+', '', pattern_string))
    # Check if a match is found
    if match:
        # Extract the matched number
        extracted_number = match.group(0)
        #print("Extracted Number:", extracted_number)
    else:
        print("No match found.")
    return extracted_number

v_idx2name = {
     1: "C1",     2: "C2",     3: "C3",     4: "C4",     5: "C5",     6: "C6",     7: "C7", 
     8: "T1",     9: "T2",    10: "T3",    11: "T4",    12: "T5",    13: "T6",    14: "T7",    15: "T8",    16: "T9",    17: "T10",   18: "T11",   19: "T12", 28: "T13",
    20: "L1",    21: "L2",    22: "L3",    23: "L4",    24: "L5",    25: "L6",    
    26: "S1",    29: "S2",    30: "S3",    31: "S4",    32: "S5",    33: "S6",
    27: "Cocc"
}


def get_region(label):
    if label in [1,2,3,4,5,6,7]:#range(1, 8):
        return 'C'
    elif label in [8,9,10,11,12,13,14,15,16,17,18,19,28]:#range(8, 20):
        return 'T'
    elif label in range(20, 26):
        return 'L'
    elif label in [26,29,30,31,32,33]:#range(26, 34):
        return 'S'
    elif label == 27:
        return 'Cocc'
    #elif label in vert_constants.subreg_idx2name:
     #   return 'subreg'
    else:
        return 'unknown'


def extract_embeddings(train_dataloader, val_dataloader, ctfu_dataloader, model):
    data_list = []

    with torch.no_grad():

        for x in ctfu_dataloader:
            img = x['img'].to(device)
            label = x['label']
            subject = x['subject']
            
            output = model.encode(img)

            
            label_str = str(label)
            subject_str = str(subject)
            #print(subject_str)
            # if (int(extract_nr(subject_str)) == 526 and int(extract_nr(label)) == 25):# or (subject_str == "['verse534']" and label == "['025']") or (subject_str == "['verse617']" and label == "['024']"):
            #     print("skipped")
            #     continue 
            region = get_region(int(extract_nr(label_str)))
            volume = calc_volume(img.cpu().numpy())

            data_item = {
                "embeddings": output,  
                "label": label_str,
                "subject": subject_str,
                "region": region,
                "volume": volume
            }

            data_list.append(data_item)

    # Convert the list of dictionaries to a DataFrame
    dataframe = pd.DataFrame(data_list)

    return dataframe


if __name__ == '__main__':
   
    opt = arguments.DAE_Option().get_opt(None, None)


    dataset = get_dataset(arguments.DataSet_Option,"train")
    dataset_val = get_dataset(arguments.DataSet_Option,"val")

    #part_tr = torch.utils.data.random_split(dataset, [0, len(dataset)-10])[0]
    #subset_indices = range(100)#[0,1,2,3,4,5,6,7,8,9]
    part_tr = dataset #torch.utils.data.Subset(dataset, subset_indices)
    #datset = dataset.
    train_dataloader = DataLoader(
            part_tr,
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
    val_dataloader = DataLoader(
            dataset_val,
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
    arg_ctfu = arguments.DataSet_Option()
    arg_ctfu.dataset  = "/media/DATA/martina_ma/cutout/ctfu_to_encode.csv"

    dataset_ctfu = get_dataset(arg_ctfu,"train")

    ctfu_dataloader = DataLoader(
            dataset_ctfu,
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
    checkpoint_path ="/media/DATA/martina_ma/dae/lightning_logs/DAE_3D_95_old_verse_w_norm/version_30/checkpoints/epoch=10-step=46717d_score=0.9912_d_score_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_133/checkpoints/epoch=3-step=310_latest.ckpt"#/media/DATA/martina_ma/dae/lightning_logs/DAE_3D_95_old_verse_w_norm/version_10/checkpoints/epoch=0-step=1488_d_score_latest copy.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_232/checkpoints/epoch=283-step=27190_latest.ckpt"# "/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_133/checkpoints/epoch=3-step=310_latest.ckpt"
    device = "cuda:0"  # or "cpu" if you want to use CPU
    assert Path(checkpoint_path).exists()
    model = DAE_LitModel.load_from_checkpoint(checkpoint_path)
    model.ema_model.eval()
    model.ema_model.to(device)
    embeddings_dataframe = extract_embeddings(train_dataloader, val_dataloader, ctfu_dataloader, model)
    torch.save(embeddings_dataframe,'/media/DATA/martina_ma/emb_dict_ctfu.pt')
   