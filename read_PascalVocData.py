import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

DATA_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'

train_list = '/VOC2012/ImageSets/Segmentation/train.txt'
val_list = '/VOC2012/ImageSets/Segmentation/val.txt'
test_list = '/VOC2012/ImageSets/Segmentation/test.txt'
img_dir = '/VOC2012/JPEGImages'
annotation_dir = '/VOC2012/SegmentationClass'

def read_dataset(data_dir):
    pickle_filename = "PascalVoc.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_tarfile=True)
        PascalVoc_folder = "VOCdevkit"
        result = create_image_lists(os.path.join(data_dir, PascalVoc_folder))
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None

    with open(image_dir + train_list) as f:
        train_data = f.readlines()
    f.close()

    with open(image_dir + val_list) as f:
        val_data = f.readlines()
    f.close()

    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []

        if directory == 'training':
        	f_data = train_data
        else:
        	f_data = val_data

        for f in f_data:
            filename = (os.path.splitext(f.split("/")[-1])[0]).replace('\n', '')
            image_file = os.path.join(image_dir + img_dir, filename + '.jpg')
            annotation_file = os.path.join(image_dir + annotation_dir, filename + '.png')
        
            if os.path.exists(annotation_file):
                record = {'image': image_file, 'annotation': annotation_file, 'filename': filename}
                image_list[directory].append(record)
            else:
                print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list
