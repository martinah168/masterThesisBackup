
from BIDS import NII
import numpy as np
import pandas as pd
import os
from dataloader.datasets.dataset_csv import extract_label
from scipy import ndimage
from BIDS import BIDS_Global_info, NII
from BIDS.core.np_utils import np_calc_crop_around_centerpoint, np_bbox_nd
#from BIDS import NII
import nibabel as nib
from pathlib import Path
#from BIDS.core.poi import load_poi, VertebraCentroids
from IPython.display import Image
import numpy as  np
import pandas as pd
import csv
import os
import re
#3D
def load_nifti(nii):
        
    
    nii = NII.load(
    "/media/DATA/martina_ma/cutout/fxclass0149/fxclass0149_012_subreg_cropped.nii.gz",#"" "/media/DATA/martina_ma/cutout/fxclass0288/fxclass0288_019_subreg_cropped.nii.gz",
        True,
    )
    return nii, nii.get_seg_array()

# du kannst NII.get_segmentation_connected_components machen
# oder falls du es zu dem zeitpunkt als array hast, unter BIDS.core.np_utils dieselbe funktion np_connected_components
# daraus erhälst du ein
#import cc3d as c
def clean(nii_seg_arr, nii, labels):
  # labels =  [0,41,42,43,44,45,46,47,48,49,50]
   subreg_cc, subreg_cc_stats = nii.get_segmentation_connected_components( labels, connectivity = 3)#np_connected_components(nii_seg_arr) # (die funktion von oben)
   cc_size_threshold: int = 50 #(check das vllt. bei dem frakturierten und drüber, was hier ne gute zahl)
   result_arr = nii_seg_arr.copy() #(kopier dein input, sobalds ein arr ist)
   #labels =  [0,41,42,43,44,45,46,47,48,49,50]#(deine möglichen subregion labels, wie sie halt grad vorliegen
   for l in labels:
      cc_indices = [i for i, v in enumerate(subreg_cc_stats[l]["voxel_counts"]) if v < cc_size_threshold] #yields cc indices of cc volume below threshold
      for cc_idx in cc_indices:
         mask_cc = subreg_cc[l]
         mask_cc_l = mask_cc.copy()
         mask_cc_l[mask_cc_l != cc_idx+1] = 0
         result_arr[mask_cc_l != 0] = 0#!= 0] = 0
   return result_arr


if __name__ == '__main__':
    path = "/media/DATA/martina_ma/train_val_cleaned_seg_fails.csv"
    dataset = pd.read_csv(path, sep=",")
    for index in range(len(dataset)):
        row = dataset.iloc[index]
        sub , label = extract_label(row.file_path)
        nii = NII.load(row['file_path'], True)
        seg_arr = nii.get_seg_array()
        #nii, nii_seg_arr = load_nifti('nii')
        #print(np.unique(nii_seg_arr))
        labels = np.unique(seg_arr)
        result_arr = clean(seg_arr,nii, labels)
        
        folder = "/media/DATA/martina_ma/cutout_clean/{}/".format(sub)
        if not os.path.exists(folder):
                os.makedirs(folder)
        #nii.set_array_(result_arr).save("Cleaned_fxclass0149_012_30.nii.gz")

        nii.set_array_(result_arr).save("/media/DATA/martina_ma/cutout_clean/{}/{}_{:2d}_subreg_cropped_cleaned.nii.gz".format(sub, sub, int(label)))#.format(subject,subject, label))

