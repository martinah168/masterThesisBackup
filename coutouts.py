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




def get_data(root='/media/DATA/martina_ma/data/dataset-verse'):
   bids_global_object = BIDS_Global_info([root], ["derivatives_new"],
                                         additional_key=["sequ", "seg", "ovl","ses"], verbose=True, )
   bids_family_dict = {}
#bids_global_object = BIDS_Global_info(['/media/data/robert/datasets/spinegan_T2w/raw/'],['rawdata',"rawdata_dixon","derivatives"],additional_key = ["sequ", "seg", "ovl"], verbose=True,)
   x = bids_global_object.enumerate_subjects(sort=True)
#derivatives_spine_r/
   for subject_name, subject_container in bids_global_object.enumerate_subjects(sort=True):
       query = subject_container.new_query(flatten=False)
       query.filter('seg', 'vert')
       query.filter('seg', 'subreg')
       if subject_name == "unsorted":
        print("unsorted")
        continue

       for bids_family in query.loop_dict(sort=True):
           # finally we get a bids_family
           if ["msk_seg-vert", "msk_seg-subreg"] not in bids_family:
            #maybe print bids_family.family_id oder so
             print(bids_family.family_id)
             continue
           bids_family_dict[query.subject.name] = bids_family.get_bids_files_as_dict(
               ['msk_seg-vert', 'msk_seg-subreg'])
   return bids_family_dict




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


def find_cutout(nii_vert, label, vert_seg):
   vert_L1 = nii_vert.extract_label(label)
   vert_arr_L1 = vert_L1.get_seg_array()
   if np.count_nonzero(vert_arr_L1) == 0:
       with open('/media/DATA/martina_ma/cutout/fehler.txt', 'w') as the_file:
        the_file.write(vert_seg)
        the_file.write('\n')
        return
   bbox = np_bbox_nd(vert_arr_L1)
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

def make_cutout(bids_family_dict, max_cutout_size):
   for subject in bids_family_dict:
       match = re.search(r'\d+', subject)

       if match:
            number = int(match.group())
            print(number)
       else:
            print("No number found")
    #    if subject == "Alex10" or "ctfu" in subject or number <= 411:
    #        print("already cutout done - skip")
    #        continue
       
       vert_seg = bids_family_dict[subject]['msk_seg-vert'][0]
       subreg = bids_family_dict[subject]['msk_seg-subreg'][0]
       #ctd = bids_family_dict[subject]['ctd_seg-subreg'][0]


       #c = load_poi(ctd)
       vert_nii = vert_seg.open_nii()
       subreg_nii = subreg.open_nii()


       #c.zoom = vert_nii.zoom
       #c.rescale_()
       vert_nii.rescale_()
       vert_nii.reorient_()
       subreg_nii.rescale_()
       subreg_nii.reorient_()


       labels = vert_nii.unique()
       for label in labels:
           if label > 26 and label < 28:
                print("label")
                continue
           vert_label_nii = vert_nii.extract_label(label)
           vert_arr = vert_label_nii.get_seg_array()
           subreg_arr = subreg_nii.get_seg_array()#vert_arr_l1 = vert_L1.get_seg_array()
           subreg_arr_L1 = subreg_arr.copy()
           subreg_arr_L1[vert_arr == 0] = 0
           #subreg_nii.set_array_(subreg_arr_L1)
           #center = ndimage.center_of_mass(vert_arr)#,c.centroids[label]
           #todo seg araay
           #cropped_arr, cutout_coord_slices, padding = np_calc_crop_around_centerpoint(center, vert_arr, max_cutout_size)
           #vert_label_nii.set_array_(cropped_arr)
           center_subreg = ndimage.center_of_mass(subreg_arr_L1)
           cropped_arr_subreg, cutout_coord_slices, padding = np_calc_crop_around_centerpoint(center_subreg, subreg_arr_L1, max_cutout_size)
       #    nii_subreg_s = subreg_nii.set_array(cropped_arr_subreg)
           nii_subreg_s = subreg_nii.copy()
           nii_subreg_s.set_array_(cropped_arr_subreg)
           nii_subreg_s = nii_subreg_s.pad_to(max_cutout_size)
           # print("subreg",nii_subreg.shape)
           #vert_label_nii.rescale_()
           #vert_label_nii.reorient_()
           nii_subreg_s.rescale_()
           nii_subreg_s.reorient_()
           #vert_label_nii.save("{}_{:03d}_vert_cropped.nii.gz".format(subject, label))
           folder = "/media/DATA/martina_ma/cutout/{}/".format(subject)
           if not os.path.exists(folder):
                  os.makedirs(folder)
           nii_subreg_s.save("/media/DATA/martina_ma/cutout/{}/{}_{:03d}_new_subreg_cropped.nii.gz".format(subject,subject, label))


if __name__ == '__main__':
   max_x = 0
   max_y = 0
   max_z = 0
   cut = True
   bids_family_dict = get_data()
   if cut:
       make_cutout(bids_family_dict, (144, 96, 144))
   else:
    
    csv_filename = '/media/DATA/martina_ma/cutout/cutout_verse_new.csv'
    with open(csv_filename, "w") as file:
            writer = csv.writer(file, lineterminator='\n')
    
            bids_family_dict = get_data()
            #max_cutout_size = tuple()
            #cutout_size = []
            
            writer.writerow(['P', 'I', 'R','Label','subject'])
            #find biggest cutout size
            for subject in bids_family_dict:
                print(subject)
                match = re.search(r'\d+', subject)

                if match:
                    number = int(match.group())
                    print(number)
                else:
                    print("No number found")
                #writer.writerow(subject)
                vert_seg = bids_family_dict[subject]['msk_seg-vert'][0]
                subreg = bids_family_dict[subject]['msk_seg-subreg'][0]


                vert_nii = vert_seg.open_nii()
                subreg_nii = vert_seg.open_nii()
                #if vert_nii.orientation != subreg_nii.orientation:
                vert_nii.reorient_()
                subreg_nii.reorient_()
                
                vert_nii.rescale_()
                subreg_nii.rescale_()


                labels = vert_nii.unique()
                for label in labels:
                    cut_size = find_cutout(vert_nii, label, vert_seg)
                    if cut_size[0] > max_x:
                        max_x = cut_size[0]
                    if cut_size[1] > max_y:
                        max_y = cut_size[1]
                    if cut_size[2] > max_z:
                        max_z = cut_size[2]
                    row = cut_size+(label,number,)
                    print(row)
                    writer.writerow(row)
                    
                #cutout_size.append(cut_size)

        
            #c_max = make_max_tuple(cutout_size)
            # type(c_max)
            # type(max_cutout_size)
            #if max_cutout_size < c_max:
            #    max_cutout_size = c_max
                # print(max_cutout_size)
            # break
            #
            print(max_x,max_y,max_z)
            file.close()
            with open('/media/DATA/martina_ma/cutout/max_c_test_tru.txt', 'w') as the_file:
                the_file.write(str(max_x))
                the_file.write('\n')
                the_file.write(str(max_y))
                the_file.write('\n')
                the_file.write(str(max_z))
                the_file.write('\n')
            print(max_x,max_y,max_z)
   #max_cutout_size = erhoehe_auf_groesste_2er_potenz(max_cutout_size)
   #print(max_cutout_size)


#    for subject in bids_family_dict:
#        vert_seg = bids_family_dict[subject]['msk_seg-vert'][0]
#        subreg = bids_family_dict[subject]['msk_seg-subreg'][0]
#        #ctd = bids_family_dict[subject]['ctd_seg-subreg'][0]


#        #c = load_poi(ctd)
#        vert_nii = vert_seg.open_nii()
#        subreg_nii = subreg.open_nii()


#        #c.zoom = vert_nii.zoom
#        #c.rescale_()
#        vert_nii.rescale_()
#        vert_nii.reorient_()
#        subreg_nii.rescale_()
#        subreg_nii.reorient_()


#        labels = vert_nii.unique()
#        for label in labels:
#            vert_label_nii = vert_nii.extract_label(label)
#            vert_arr = vert_label_nii.get_seg_array()
#            subreg_arr = subreg_nii.get_seg_array()#vert_arr_l1 = vert_L1.get_seg_array()
#            subreg_arr_L1 = subreg_arr.copy()
#            subreg_arr_L1[vert_arr == 0] = 0
#            #subreg_nii.set_array_(subreg_arr_L1)
#            center = ndimage.center_of_mass(vert_arr)#,c.centroids[label]
#            #todo seg araay
#            cropped_arr, cutout_coord_slices, padding = np_calc_crop_around_centerpoint(center, vert_arr, max_cutout_size)
#            vert_label_nii.set_array_(cropped_arr)
#            center_subreg = ndimage.center_of_mass(subreg_arr_L1)
#            cropped_arr_subreg, cutout_coord_slices, padding = np_calc_crop_around_centerpoint(center_subreg, subreg_arr_L1, max_cutout_size)
#        #    nii_subreg_s = subreg_nii.set_array(cropped_arr_subreg)
#            nii_subreg_s = subreg_nii.copy()
#            nii_subreg_s.set_array_(cropped_arr_subreg)
#            nii_subreg_s = nii_subreg_s.pad_to(max_cutout_size)
#            # print("subreg",nii_subreg.shape)
#            vert_label_nii.rescale_()
#            vert_label_nii.reorient_()
#            nii_subreg_s.rescale_()
#            nii_subreg_s.reorient_()
#            #vert_label_nii.save("{}_{:03d}_vert_cropped.nii.gz".format(subject, label))
#            nii_subreg_s.save("{}/{}_{:03d}_subreg_cropped.nii.gz".format(subject,subject, label))

#           # nii_subreg_s.save("{}_{:03d}_subreg_cropped.nii.gz".format(subject, label))
#           # crop(center, vert_arr, vert_label_nii, subreg_nii, label, subject, max_cutout_size)

