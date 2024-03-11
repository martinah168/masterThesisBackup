import pandas as pd

import os
import torch
import pandas as pd
import numpy as np

from BIDS import NII
from BIDS.core.np_utils import np_volume
from BIDS.core.np_utils import np_map_labels
import sys

# print the original sys.path
print('Original sys.path:', sys.path)

# append a new directory to sys.path
sys.path.append('/media/DATA/martina_ma/dae')

# print the updated sys.path
print('Updated sys.path:', sys.path)
from dataloader.datasets.dataset_csv import extract_label
import utils.metadata as meta


def clean_string(s):
    return s.strip("[]").replace("'", "")

def prepare_data(df):
    ex_df = pd.DataFrame()
    i = 0
    for index, row in df.iterrows():
        sub, label = extract_label(row.file_path)
        ex_df.at[i, 'subject'] = sub
        ex_df.at[i, 'label'] = label
        ex_df.at[i, 'file_path'] = row.file_path
        ex_df.at[i,'Split'] = row.Split
        i = i + 1
    return ex_df

if __name__ == '__main__':
    path = '/media/DATA/martina_ma/dae/test_set_corpus_filtered_cleaned.csv'
    df = pd.read_csv(path, sep=",")#, index_col=0)
    ex_df = prepare_data(df)
    extended_df = meta.add_labels(ex_df)
    print(len(extended_df))
    torch.save(extended_df,'test_set_corpus_filtered_cleaned_extended.pt')
    # Assuming df is your original DataFrame
    # Filter rows with fracture_grade values 0 and 3
    df_subset_0 = extended_df[extended_df['fracture_grading'] == 0].sample(n=5, random_state=42)
    print(extended_df[extended_df['fracture_grading'] == 3])
    df_subset_3 = extended_df[extended_df['fracture_grading'] == 3].sample(n=4, random_state=42)

    # Concatenate the two subsets into one DataFrame
    df_subset = pd.concat([df_subset_0, df_subset_3])

    # Reset index of the subset DataFrame
    df_subset.reset_index(drop=True, inplace=True)
    print(len(df_subset))
    torch.save(df_subset,'test_subset_corpus_fracture3_VS_healthy.pt')
    df_subset.to_csv('test_subset_corpus_fracture3_VS_healthy.csv')
    
