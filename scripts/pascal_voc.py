#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:31:04 2017

@author: txzhao
"""

import numpy as np
import pickle
import os


# ======================  parameter settings  ======================  
# directory of dataset
data_path = "/Users/MMGF2/Desktop/dl_pro/datasets/MIT_Scene_Parsing/"

# Width and height of each image
img_size = 200

# Number of channels in each image, 3 channels: Red, Green, Blue
x_num_channels = 3
y_num_channels = 1

# Length of an image when flattened to a 1-dim array
x_img_size_flat = img_size * img_size * x_num_channels

# Number of files for the training-set
_num_files_train = 2

# Number of images for each batch-file in the training-set
_images_per_file = 500

# Total number of images in the training-set
_num_images_train = _num_files_train * _images_per_file


# ======================  define function  ======================
def _get_file_path(filename = ""):
    """
    Return the full path of a data-file for the data-set
    If filename == "" then return the directory of the files
    """
    
    # may change folder name to reuse this code
    return os.path.join(data_path, "200x200_pkl/", filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data
    Note that the appropriate dir-name is prepended the filename
    """

    # Create full path for the file
    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)

    with open(file_path, mode = 'rb') as file:
        data = pickle.load(file)

    return data


def _convert_images(raw, num_channels):
    """
    Convert images from the PASCAL VOC dataset and
    return a 4-dim array with shape: [image_number, height, width, channel]
    scale pixel values to floats between 0.0 and 1.0
    """

    # Convert the raw images from the data-files to floating-points and scale
    raw_float = np.array(raw, dtype = float)/255.0

    # Reshape the array to 4-dimensions
    images = raw_float.reshape([-1, img_size, img_size, num_channels])

    return images


def _load_data(filename, num_channels):
    """
    Load a pickled data-file from the PASCAL VOC data-set
    and return the converted images (see above) and the class-number
    for each image
    """

    # Load the pickled data-file
    data = _unpickle(filename)

    # Get the raw images
    raw_images = data

    # Convert the images
    images = _convert_images(raw_images, num_channels)

    return images


def load_training_data():
    """
    Load all the training-data for the PASCAL VOC data-set 
    and return images and annotations
    """

    # Pre-allocate the arrays for the images and annotations for efficiency
    images_x = np.zeros(shape = [_num_images_train, img_size, img_size, x_num_channels], dtype = float)
    images_y = np.zeros(shape = [_num_images_train, img_size, img_size, y_num_channels], dtype = float)

    begin = 0
    for i in range(_num_files_train):
        # Load the images and annotations from .pkl files
        images_x_batch = _load_data(filename = "tr_x_batch_" + str(i + 1) + ".pkl",
                                    num_channels = x_num_channels)
        images_y_batch = _load_data(filename = "tr_y_batch_" + str(i + 1) + ".pkl", 
                                    num_channels = y_num_channels)

        num_images = len(images_x_batch)
        end = begin + num_images

        # Store the images and annotations into the array
        images_x[begin:end, :] = images_x_batch
        images_y[begin:end, :] = images_y_batch

        begin = end

    return images_x, images_y


# TODO LIST
def load_test_data():
    """
    Load all the test-data for the PASCAL VOC data-set
    Returns the images and annotations
    """

    images = _load_data(filename = "test_batch")

    return images

