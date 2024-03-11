import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

import sys

# print the original sys.path
#print('Original sys.path:', sys.path)

# append a new directory to sys.path
sys.path.append('/media/DATA/martina_ma/dae')

# print the updated sys.path
#print('Updated sys.path:', sys.path)

import utils.metadata as meta
import torch
from BIDS.core.vert_constants import v_name2idx, v_idx2name


def clean_string(s):
    return s.strip("[]").replace("'", "")

if __name__ == '__main__':
    df = torch.load("/media/DATA/martina_ma/emb_tsne_cleaned.pt") #/media/DATA/martina_ma/emb_dict_3D_cleaned_balanced.pt
    df['label'] = df['label'].apply(clean_string)
    df['label'] = (df['label'].apply(int))
    df['subject'] = df['subject'].apply(clean_string)
    print(df.info())

    extended_df = meta.add_labels(df)
    extended_df.info()
    extended_df.describe()


    # # Assuming `features_array` is your embeddings array with shape (num_samples, num_features)
    # features_ex = extended_df["embeddings"].tolist()
    # features_tensor_ex = torch.cat(features_ex, dim=0)
    # features_array_ex = features_tensor_ex.cpu().numpy()

    # # Apply t-SNE
    # tsne = TSNE(n_components=2, perplexity=30, n_iter=1500)
    # tsne_result = tsne.fit_transform(features_array_ex)#tsne.fit_transform(features_array)

    # # # Add t-SNE results to the DataFrame
    # extended_df["tsne_1"] = tsne_result[:, 0]
    # extended_df["tsne_2"] = tsne_result[:, 1]

    extended_df['dataset'] = None
    for index, row in extended_df.iterrows():
        if "verse" in row['subject']:
            extended_df.at[index, 'dataset'] = "verse"
        elif "rsna" in row['subject']:
            extended_df.at[index, 'dataset'] = "rsna"
        elif "fxclass" in row['subject']:
            extended_df.at[index, 'dataset'] = "fxclass"
        elif "tri" in row['subject']:
            extended_df.at[index, 'dataset'] = "tri"
        elif "ctfu" in row['subject']:
            extended_df.at[index, 'dataset'] = "ctfu"

    torch.save(extended_df,'/media/DATA/martina_ma/emb_df_cleaned_epoch10_tsne.pt')#/media/DATA/martina_ma/emb_dict_3D_cleaned_balanced_epoch57.pt
#/media/DATA/martina_ma/emb_tsne_cleaned.pt