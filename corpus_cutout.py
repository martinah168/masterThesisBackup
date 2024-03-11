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
from dataloader.datasets.dataset_csv import extract_label
from BIDS.core.np_utils import np_map_labels, Label_Map


def erhoehe_auf_groesste_2er_potenz(tupel):
   max_wert = max(tupel)  # Finde den größten Wert im Tupel
   nächste_potenz = 2
   while nächste_potenz < max_wert:
       nächste_potenz *= 2


   neue_werte = (nächste_potenz,) * 3  # Alle Werte im Tupel auf die gleiche 2er-Potenz setzen
   return neue_werte


def make_max_tuple(data):
   # Initialize variables to store the maximum values
   max_values = (float('-inf'), float('-inf'), float('-inf'))
   # Iterate through the list of tuples
   for tup in data:
       for i in range(3):
           max_values = tuple(max(max_values[i], tup[i]) for i in range(3))
   return max_values




def make_max_slice(data):
   # Initialize variables to store the maximum values
   print(type(data[0]))
   max_start = -1
   max_stop = -1
   # Iterate through the list of slices
   for slc in data:
       start, stop, step = slc.indices
       if start > max_start:
           max_start = start
       if stop > max_stop:
           max_stop = stop
   return [max_start, max_stop]



def find_crop_slices(vert_nii, subreg_nii, label):
   vert_l1 = vert_nii.extract_label(label)  # 20 is L1, after extract, mask is only binary!
   vert_arr_l1 = vert_l1.get_seg_array()
   subreg_arr = subreg_nii.get_seg_array()
   subreg_arr_l1 = subreg_arr.copy()
   subreg_arr_l1[vert_arr_l1 == 0] = 0


   ex_slice = vert_l1.compute_crop_slice()
   x = ndimage.center_of_mass(vert_arr_l1)
   return ex_slice




# Function to extract start and stop values from a slice
def extract_slice(slice_obj):
   if isinstance(slice_obj, slice):
       return slice_obj.start, slice_obj.stop
   return None, None


def find_cutout(corpus_arr, label):
   bbox = np_bbox_nd(corpus_arr)
   t = tuple(extract_slice(slice_obj) for slice_obj in bbox)
   c = tuple(b-a for (a,b) in t)
   print(c, label)
   return c

def write_tuple_to_csv(data, csv_filename):
    with open(csv_filename, "w",newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)



#crop(center, vert_arr, vert_, subreg_nii, label, subject, max_cutout_size)
def crop(center, vert_arr, nii_vert, nii_subreg, label, subject_name, cut_s):
   cropped_arr, cutout_coord_slices, padding = np_calc_crop_around_centerpoint(center,vert_arr,cut_s)
   nii_vert.set_array_(cropped_arr)
   cropped_arr_subreg, cutout_coord_slices, padding = np_calc_crop_around_centerpoint(center, vert_arr, cut_s)
   nii_subreg_s = nii_subreg.set_array(cropped_arr_subreg)
   #nii_subreg = nii_subreg.apply_crop_slice(cutout_coord_slices)
   nii_subreg_s = nii_subreg.pad_to(cut_s)
   #print("subreg",nii_subreg.shape)
   nii_vert.rescale_()
   nii_vert.reorient_()
   nii_subreg.rescale_()
   nii_subreg.reorient_()
  # nii_vert.save("{}_{:03d}_vert_cropped.nii.gz".format(subject_name, label))
   nii_subreg_s.save("{}/{}_{:03d}_subreg_cropped.nii.gz".format(subject_name,subject_name, label))

def make_cutout(dataset, max_cutout_size):
   #TODO move this to cutout making method
                    # folder = "/media/DATA/martina_ma/cutout_corpus/{}/".format(sub)
                    # if not os.path.exists(folder):
                    #         os.makedirs(folder)
                    #nii.set_array_(result_arr).save("Cleaned_fxclass0149_012_30.nii.gz")

                    #nii.set_array_(result_arr).save("/media/DATA/martina_ma/cutout_clean/{}/{}_{:2d}_subreg_cropped_cleaned.nii.gz".format(sub, sub, int(label)))#.format(subject,subject, label))
    for index in range(len(dataset)):
        row = dataset.iloc[index]

        sub , label = extract_label(row.file_path)
        nii = NII.load(row['file_path'], True)
        seg_arr = nii.get_seg_array()
        from_im = np_map_labels(arr=seg_arr,label_map={50: 49})
        labelmap = {i: 0 for i in range(41, 49)}
        corpus_arr =  np_map_labels(from_im, labelmap)

        if np.max(corpus_arr) < 1 or int(label)in range(1,8):
            print("brokenfile",sub,label)
            continue
            
        center_subreg = ndimage.center_of_mass(corpus_arr)
        cropped_arr_subreg, cutout_coord_slices, padding = np_calc_crop_around_centerpoint(center_subreg, corpus_arr, max_cutout_size)
    #    nii_subreg_s = subreg_nii.set_array(cropped_arr_subreg)
        nii_subreg_s = nii.copy()
        nii_subreg_s.set_array_(cropped_arr_subreg)
        nii_subreg_s = nii_subreg_s.pad_to(max_cutout_size)
        nii_subreg_s.rescale_()
        nii_subreg_s.reorient_()
        #vert_label_nii.save("{}_{:03d}_vert_cropped.nii.gz".format(subject, label))
        folder = "/media/DATA/martina_ma/cutout_corpus_test/{}/".format(sub)
        if not os.path.exists(folder):
                os.makedirs(folder)
        nii_subreg_s.save("/media/DATA/martina_ma/cutout_corpus_test/{}/{}_{:2d}_subreg_corpus.nii.gz".format(sub,sub, int(label)))#.format(subject,subject, label))




def extract_substring(input_string):
    # Define a regular expression pattern to match the desired substring
    pattern = re.compile(r'^(.*?_ses-\d+)_.*$')

    # Use search to find the first match in the input string
    match = pattern.search(input_string)

    # Return the matched substring or None if no match is found
    return match.group(1) if match else None


if __name__ == '__main__':
    max_x = 0
    max_y = 0
    max_z = 0
    cut = True

    path = "/media/DATA/martina_ma/dae/test_set_filtered_cleaned_with_cervicals.csv"#"corpus_train_val_set.csv"
    size =  (128, 96, 128)#(144, 96, 144) 75 60 87
    dataset = pd.read_csv(path, sep=",")
    if cut:
        make_cutout(dataset, size)
    else:

        csv_filename = '/media/DATA/martina_ma/cutout/cutout_corpus.csv'
        with open(csv_filename, "w") as file:
                writer = csv.writer(file, lineterminator='\n')
                writer.writerow(['P', 'I', 'R','Label','subject'])
                for index in range(len(dataset)):
                    row = dataset.iloc[index]
                    
                    sub , label = extract_label(row.file_path)
                    nii = NII.load(row['file_path'], True)
                    seg_arr = nii.get_seg_array()
                    from_im = np_map_labels(arr=seg_arr,label_map={50: 49})
                    labelmap = {i: 0 for i in range(41, 49)}
                    corpus_arr =  np_map_labels(from_im, labelmap)
                    
                    
                    
                    cut_size = find_cutout(corpus_arr, label)
                    if cut_size[0] > max_x:
                        max_x = cut_size[0]
                    if cut_size[1] > max_y:
                        max_y = cut_size[1]
                    if cut_size[2] > max_z:
                        max_z = cut_size[2]
                    row = cut_size+(label,sub,)
                    print(row)
                    writer.writerow(row)

                print(max_x,max_y,max_z)
                file.close()
                with open('/media/DATA/martina_ma/cutout/max_courpus.txt', 'w') as the_file:
                    the_file.write(str(max_x))
                    the_file.write('\n')
                    the_file.write(str(max_y))
                    the_file.write('\n')
                    the_file.write(str(max_z))
                    the_file.write('\n')
                print(max_x,max_y,max_z)