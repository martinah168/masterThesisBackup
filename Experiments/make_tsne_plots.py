import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import sys
import os

# print the original sys.path
print('Original sys.path:', sys.path)

# append a new directory to sys.path
sys.path.append('/media/DATA/martina_ma/dae')

# print the updated sys.path
print('Updated sys.path:', sys.path)


import utils.metadata as meta
import torch
from BIDS.core.vert_constants import v_name2idx, v_idx2name

def make_figs(emb_df,folder):
    attributes = ['sex','label','volume',"subject","region","Implant","fracture_flag","dataset","cancerous","CT scanner","age"]
    emb_df = emb_df[emb_df['volume'] > 8000]
    emb_df_copy = emb_df.copy()
    for a in attributes:
        if a == 'sex':
             emb_df = emb_df[emb_df['sex'] != 'U']
        if a == 'age':
             emb_df = emb_df[emb_df['age'] != -1]
             emb_df = emb_df[emb_df['age'] < 110]
             emb_df = emb_df[emb_df['age'] > 20]
        if a == 'fracture_flag':
            emb_df = emb_df[emb_df['fracture_flag'] != 'U']
        if a == 'fracture_grade':
            emb_df = emb_df[emb_df['fracture_grading'] != -1]
            emb_df = emb_df[emb_df['fracture_grading'] != 4]
        if a == "CT scanner":
             emb_df = emb_df[emb_df["CT scanner"] != -1]
        if a == 'Implant':
             emb_df = emb_df[emb_df['Implant'] != 'U']
        emb_df_x = emb_df['tsne_1']
        emb_df_y = emb_df['tsne_2']
        fig = px.scatter(emb_df, x='tsne_1', y='tsne_2', color=a)
        fig.update_layout(autosize=False,
        width=1200,
        height=600,)#legend_orientation="h")
        fig.update_traces(marker={'size': 5})
        fig.write_image(folder+'{}.svg'.format(a))
        emb_df = emb_df_copy.copy()


if __name__ == '__main__':
    path = "/media/DATA/martina_ma/emb_df_cleaned_epoch10_tsne.pt"#'/media/DATA/martina_ma/emb_dict_3D_cleaned_balanced_tsne.pt'#'/media/DATA/martina_ma/emb_df_cleaned_cleaned_epoch10_ver5_tsne.pt'#'/media/DATA/martina_ma/emb_dict_3D_less_blocks_epoch5_ver12_tsne.pt'#'/media/DATA/martina_ma/emb_dict_3D_corpus_tsne_epoch15.pt'#
    experiment = path.split('/')[-1].replace('.pt','')
    emb_df = torch.load(path)
    folder = "/media/DATA/martina_ma/tsne_plots/{}/".format(experiment)
    if not os.path.exists(folder):
            os.makedirs(folder)
    make_figs(emb_df, folder)
    

