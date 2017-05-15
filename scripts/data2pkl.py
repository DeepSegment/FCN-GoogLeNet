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


def dir_to_dataset(glob_files, loc_train_labels=""):
    print("To be processed:\n\t %s"%glob_files)
    dataset = []
    delete_file = []
    
    for file_count, file_name in enumerate(sorted(glob(glob_files), key = len)):
        print("\t %s files processed"%(file_count + 1))
        
        # return a sequence object (flattened) with pixel values
        img = Image.open(file_name)
        pixels = list(img.getdata())
        
        # avoid potential binary images
        if type(pixels[0]) == int:
#            os.remove(file_name)
            delete_file.append(file_name)
            continue
        
        dataset.append(pixels)

    return np.array(dataset), delete_file


# define the directory of a dataset
file_dir = "/Users/MMGF2/Desktop/dl_pro/datasets/MIT_Scene_Parsing/ADEChallengeData2016/images/bedroom_tr//*.jpg"
Data, delete_file = dir_to_dataset(file_dir, "")

# regroup data
train_set_x = Data

# write data into a .pkl file
f = open('/Users/MMGF2/Desktop/dl_pro/datasets/tr_x_batch_1.pkl','wb')
pickle.dump(train_set_x, f)
f.close()

