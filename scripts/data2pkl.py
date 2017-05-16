#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:59:12 2017

@author: txzhao
"""

from PIL import Image
import pickle
from glob import glob
import numpy as np
import os


def dir_to_dataset(glob_files, new_width = 200, new_height = 200):
    print("To be processed:\n\t %s"%glob_files)
    dataset = []
    delete_file = []
    
    for file_count, file_name in enumerate(sorted(glob(glob_files), key = len)):
        print("\t %s files processed"%(file_count + 1))
        
        # resize images and return a sequence object (flattened) with pixel values
        img = Image.open(file_name)
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        pixels = list(img.getdata())
        
        # avoid potential binary images
        if type(pixels[0]) == int:
#            os.remove(file_name)
            delete_file.append(file_name)
            continue
        
        dataset.append(pixels)

    return np.array(dataset), delete_file


# define read directory of a dataset
file_dir = '/Users/MMGF2/Desktop/dl_pro/datasets/MIT_Scene_Parsing/ADEChallengeData2016/'
filename_x_1 = 'images/bedroom_tr1//*.jpg'
filename_x_2 = 'images/bedroom_tr2//*.jpg'
filename_y_1 = 'annotations/bedroom_tr_label1//*.png'
filename_y_2 = 'annotations/bedroom_tr_label2//*.png'
tr_file_r_dir = os.path.join(file_dir, filename_x_2)

# read in image data
Data, delete_file = dir_to_dataset(tr_file_r_dir)

# regroup data
train_set_x = Data

# define write directory of .pkl files
file_dir = '/Users/MMGF2/Desktop/dl_pro/datasets/'
filename_x_1 = 'tr_x_batch_1.pkl'
filename_x_2 = 'tr_x_batch_2.pkl'
filename_y_1 = 'tr_y_batch_1.pkl'
filename_y_2 = 'tr_y_batch_2.pkl'
tr_file_w_dir = os.path.join(file_dir, filename_x_2)

# write data into .pkl files
f = open(tr_file_w_dir, 'wb')
pickle.dump(train_set_x, f)
f.close()

