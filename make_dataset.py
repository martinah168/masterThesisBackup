import os
import torch
import pandas as pd
import numpy as np
from dataloader.datasets.dataset_csv import extract_label
from BIDS import NII
from BIDS.core.np_utils import np_volume
from BIDS.core.np_utils import np_map_labels

data_dir = "/media/DATA/martina_ma/cutout_clean"
csv_file_path = "train_val_cleaned.csv"  # Replace with the desired output CSV file path /media/DATA/martina_ma/cutout/

def calc_volume(seg_arr):
    #seg = model_map_to_segmentation_map(seg_arr)
    seg_arr = np_map_labels(arr=seg_arr,label_map={50: 49})
    labels = np.unique(seg_arr)[1:]
    #labels = labels[:9]
    #labels = model_map_to_segmentation_map(labels)[1:]
    vol_dict = np_volume(seg_arr, labels)
    total_volume = sum(vol_dict.values())
    return total_volume

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
    
# List of NIfTI files to exclude
files_to_exclude = ["verse814_19_subreg_cropped_cleaned.nii.gz", "verse814_18_subreg_cropped_cleaned.nii.gz","verse814_20_subreg_cropped_cleaned.nii.gz",
"verse642_19_subreg_cropped_cleaned.nii.gz", "verse642_18_subreg_cropped_cleaned.nii.gz","verse642_20_subreg_cropped_cleaned.nii.gz",
"verse636_19_subreg_cropped_cleaned.nii.gz", "verse636_18_subreg_cropped_cleaned.nii.gz","verse636_20_subreg_cropped_cleaned.nii.gz",
"verse626_19_subreg_cropped_cleaned.nii.gz", "verse626_18_subreg_cropped_cleaned.nii.gz","verse626_20_subreg_cropped_cleaned.nii.gz",
"verse606_19_subreg_cropped_cleaned.nii.gz", "verse606_18_subreg_cropped_cleaned.nii.gz","verse606_20_subreg_cropped_cleaned.nii.gz",
"verse581_19_subreg_cropped_cleaned.nii.gz", "verse581_18_subreg_cropped_cleaned.nii.gz","verse581_20_subreg_cropped_cleaned.nii.gz",
"verse559_19_subreg_cropped_cleaned.nii.gz", "verse559_18_subreg_cropped_cleaned.nii.gz","verse559_20_subreg_cropped_cleaned.nii.gz",
"verse544_19_subreg_cropped_cleaned.nii.gz", "verse544_18_subreg_cropped_cleaned.nii.gz","verse544_20_subreg_cropped_cleaned.nii.gz",
"verse535_25_subreg_cropped_cleaned.nii.gz", "verse599_25_subreg_cropped_cleaned.nii.gz","verse555_25_subreg_cropped_cleaned.nii.gz",
"verse591_24_subreg_cropped_cleaned.nii.gz", "verse617_24_subreg_cropped_cleaned.nii.gz","verse526_25_subreg_cropped_cleaned.nii.gz",
"verse534_25_subreg_cropped_cleaned.nii.gz", "ctfu00974_12_subreg_cropped_cleaned.nii.gz",
"verse024_25_subreg_cropped_cleaned.nii.gz",
"verse584_24_subreg_cropped_cleaned.nii.gz",
"verse609_25_subreg_cropped_cleaned.nii.gz",
"fxclass0339_ 9_subreg_cropped_cleaned.nii.gz",
"ctfu00155_21_subreg_cropped_cleaned.nii.gz",
"fxclass0056_19_subreg_cropped_cleaned.nii.gz",
"verse587_25_subreg_cropped_cleaned.nii.gz",
"verse581_25_subreg_cropped_cleaned.nii.gz",
"verse088_25_subreg_cropped_cleaned.nii.gz",
"verse626_25_subreg_cropped_cleaned.nii.gz",
"verse606_25_subreg_cropped_cleaned.nii.gz",
"fxclass0290_14_subreg_cropped_cleaned.nii.gz",
"fxclass0352_ 9_subreg_cropped_cleaned.nii.gz",
"verse616_25_subreg_cropped_cleaned.nii.gz",
"verse590_25_subreg_cropped_cleaned.nii.gz",
"ctfu00074_14_subreg_cropped_cleaned.nii.gz",
"verse510_25_subreg_cropped_cleaned.nii.gz",
"verse592_25_subreg_cropped_cleaned.nii.gz",
"verse573_24_subreg_cropped_cleaned.nii.gz",
"fxclass0178_12_subreg_cropped_cleaned.nii.gz",
"fxclass0056_19_subreg_cropped_cleaned.nii.gz",
"fxclass0056_21_subreg_cropped_cleaned.nii.gz",
"fxclass0290_14_subreg_cropped_cleaned.nii.gz",
"fxclass0152_ 8_subreg_cropped_cleaned.nii.gz"]#to do check

# List of subejcts for test split
selected_folders =["/media/DATA/martina_ma/cutout_clean/verse009",
"/media/DATA/martina_ma/cutout_clean/verse073",
"/media/DATA/martina_ma/cutout_clean/verse269",
"/media/DATA/martina_ma/cutout_clean/verse750",
"/media/DATA/martina_ma/cutout_clean/verse712",
"/media/DATA/martina_ma/cutout_clean/verse594",
"/media/DATA/martina_ma/cutout_clean/verse236",
"/media/DATA/martina_ma/cutout_clean/verse757",
"/media/DATA/martina_ma/cutout_clean/verse650",
"/media/DATA/martina_ma/cutout_clean/verse068",
"/media/DATA/martina_ma/cutout_clean/verse080",
"/media/DATA/martina_ma/cutout_clean/verse759",
"/media/DATA/martina_ma/cutout_clean/verse511",
"/media/DATA/martina_ma/cutout_clean/verse260",
"/media/DATA/martina_ma/cutout_clean/verse081",
"/media/DATA/martina_ma/cutout_clean/verse514",
"/media/DATA/martina_ma/cutout_clean/fxclass0326",
"/media/DATA/martina_ma/cutout_clean/fxclass0037",
"/media/DATA/martina_ma/cutout_clean/fxclass0271",
"/media/DATA/martina_ma/cutout_clean/fxclass0309",
"/media/DATA/martina_ma/cutout_clean/fxclass0019",
"/media/DATA/martina_ma/cutout_clean/fxclass0334",
"/media/DATA/martina_ma/cutout_clean/fxclass0120",
"/media/DATA/martina_ma/cutout_clean/fxclass0384",
"/media/DATA/martina_ma/cutout_clean/fxclass0205",
"/media/DATA/martina_ma/cutout_clean/fxclass0159",
"/media/DATA/martina_ma/cutout_clean/fxclass0321",
"/media/DATA/martina_ma/cutout_clean/fxclass0218",
"/media/DATA/martina_ma/cutout_clean/fxclass0185",
"/media/DATA/martina_ma/cutout_clean/fxclass0314",
"/media/DATA/martina_ma/cutout_clean/fxclass0059",
"/media/DATA/martina_ma/cutout_clean/fxclass0103",
"/media/DATA/martina_ma/cutout_clean/fxclass0018",
"/media/DATA/martina_ma/cutout_clean/fxclass0277",
"/media/DATA/martina_ma/cutout_clean/fxclass0110",
"/media/DATA/martina_ma/cutout_clean/tri018",
"/media/DATA/martina_ma/cutout_clean/ctfu00006",
"/media/DATA/martina_ma/cutout_clean/ctfu00492",
"/media/DATA/martina_ma/cutout_clean/ctfu00289",
"/media/DATA/martina_ma/cutout_clean/ctfu00521",
"/media/DATA/martina_ma/cutout_clean/verse553"
 #broken segmentation
]

if __name__ == '__main__':
    file_paths = []
    to_exclude_files = []
    first_last_list = []
    
    def extract_number(filename):
        return int(filename.split('_')[1])

    for root, dirs, files in os.walk(data_dir):
        if root not in selected_folders:
            # Sort the list of filenames using the custom key function
            sorted_filenames = sorted(files, key=extract_number)
            for file in files:
                first = sorted_filenames[0]
                last = sorted_filenames[-1]
                if file.endswith(".nii.gz") and file not in files_to_exclude:
                # if 'ctfu' in root: #if "verse" in root and "new" in file:
                        #check volume if its smaller or C then dont add
                        if file == first or file == last:
                            first_last_list.append(first)
                            first_last_list.append(last)
                            continue

                        folder , label = extract_label(file)
                        
                        region = get_region(int(label))
                        if region == 'C':
                            print('skipped')
                            continue
                        if "fxclass0352" in root:
                            print(folder)
                        nii = NII.load(os.path.join(root, file), True)
                        from_im = nii.get_seg_array()
                        volume = calc_volume(from_im)
                        if volume < 8000 and int(label) !=  28:
                            print("brokenfile",file)
                            to_exclude_files.append(file)
                            continue
                        if np.max(from_im) < 1:
                            print("brokenfile",file)
                            to_exclude_files.append(file)
                            continue
                        file_paths.append(os.path.join(root, file))

    torch.save(to_exclude_files,'/media/DATA/martina_ma/to_exclude_files.pt')
    torch.save(first_last_list,'/media/DATA/martina_ma/first_last_list.pt')
    torch.save(file_paths,'/media/DATA/martina_ma/file_paths.pt')