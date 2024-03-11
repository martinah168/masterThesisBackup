import sys
from torchmetrics.functional import dice
# print the original sys.path
#print('Original sys.path:', sys.path)

# append a new directory to sys.path
sys.path.append('/media/DATA/martina_ma/dae')
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
from dataloader.datasets.dataset_csv import model_map_to_segmentation_map, segmentation_map_to_model_map, model_map_to_corpus 
#from utils.metadata import make_anomaly_dict, add_fxclass_fracture_anomaly, parse_string
import utils.metadata as m
from BIDS.core.vert_constants import v_name2idx, v_idx2name
from itertools import product

if __name__ == '__main__':
    arguments.DataSet_Option.corpus = True
    arguments.DataSet_Option.dataset = "/media/DATA/martina_ma/dae/test_set_corpus_filtered_cleaned.csv"#"/media/DATA/martina_ma/dae/Experiments/test_subset_corpus_fracture3_VS_healthy.csv"
    dataset = get_dataset(arguments.DataSet_Option,"test")
    print(len(dataset))
    checkpoint_path ="/media/DATA/martina_ma/dae/lightning_logs/DAE_3D_corpus/version_20/checkpoints/epoch=15-step=121840d_score=1.0000_d_score_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_3D_cleaned_balanced/version_11/checkpoints/epoch=1-step=15230d_score=0.9726_d_score_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_3D_cleaned_balanced/version_11/checkpoints/epoch=0-step=7615d_score=0.9726_d_score_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_3D_cleaned_balanced/version_11/checkpoints/epoch=2-step=22845d_score=0.9726_d_score_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_3D_cleaned_balanced/version_11/checkpoints/epoch=57-step=440775_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_3D_cleaned/version_5/checkpoints/epoch=11-step=113760d_score=0.9303_d_score_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_3D_95_old_verse_w_norm/version_11/checkpoints/epoch=0-step=1488_d_score_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_232/checkpoints/epoch=283-step=27190_latest.ckpt"# "/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_133/checkpoints/epoch=3-step=310_latest.ckpt"
    #"/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_135/checkpoints/epoch=27-step=2775_latest.ckpt"#/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_133/checkpoints/epoch=3-step=310_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_118/checkpoints/epoch=80-step=243_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_116/checkpoints/epoch=81-step=246_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_108/checkpoints/epoch=47-step=144_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_105/checkpoints/epoch=69-step=210_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_63/checkpoints/epoch=79-step=240_latest.ckpt"#"/media/DATA/martina_ma/dae/lightning_logs/DAE_NAKO_256/version_60/checkpoints/epoch=71-step=216_latest.ckpt"#lightning_logs/DAE_NAKO_256/version_26/checkpoints/epoch=404-step=405_latest.ckpt"
    device = "cuda:0"
    assert Path(checkpoint_path).exists()
    model = DAE_LitModel.load_from_checkpoint(checkpoint_path)
    model.ema_model.eval()
    model.ema_model.to(device)

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
    dice_scores_test = []
    T_Enc = [100]
    T_Dec = [100]
    r = []
    for t_enc, t_dec in product(T_Enc, T_Dec):
        dice_scores_test = []
        for t in test_dataloader:
            img = t['img'].to(device)
            cond = model.encode(img)
            stoch = model.encode_stochastic(img, cond, T=t_enc)
            xT = model.render(stoch, cond, T=t_dec)
            if arguments.DataSet_Option.corpus:
                #xT_mapped = (xT[0][0].cpu().numpy()+1)/2
                #i = (imgs[0][0].cpu().numpy()+1)/2
                xT_mapped = model_map_to_corpus(xT[0][0].cpu().numpy())
                i = model_map_to_corpus(img[0][0].cpu().numpy())
                l_x = np.unique(xT_mapped)
                l_i = np.unique(i)
                d = dice(torch.tensor(xT_mapped,dtype=int),torch.tensor(i,dtype=int),ignore_index= 0)#, multiclass=True)
                print(d)
            else:
                xT_mapped = model_map_to_segmentation_map(xT[0][0].cpu().numpy())
                xT_mapped = xT_mapped-40
                xT_mapped = np_map_labels(xT_mapped, {-40: 0})
                i = model_map_to_segmentation_map(img[0][0].cpu().numpy())
                i = i-40
                i = np_map_labels(i, {-40: 0})
                d = dice(torch.tensor(xT_mapped,dtype=int),torch.tensor(i,dtype=int), average = 'macro', ignore_index= 0, num_classes=10)#, multiclass=True)
                print(d)
            dice_scores_test.append(d)

        combined_tensor = torch.stack(dice_scores_test)
        print(len(combined_tensor))
        # Compute the mean of the combined tensor
        average_value = torch.mean(combined_tensor)

        # Print the average value
        print(average_value.item())
        reconstruction = {'TimestepsEnc':t_enc, 'TimestepsDec':t_dec, "AvgReconstDice": average_value}
        r.append(reconstruction)


    dataframe = pd.DataFrame(r)

    torch.save(dataframe,'all_100_100_corpus_test.pt')
    dataframe.to_csv('all_100_100_corpus_test.csv')