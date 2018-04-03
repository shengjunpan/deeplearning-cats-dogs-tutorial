#!/usr/bin/python3

'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

# Assuming current working directory is deeplearning-cats-dogs-tutorial
train_lmdb = 'model_data/input/train_lmdb'
validation_lmdb = 'model_data/input/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)


train_data = [img for img in glob.glob("model_data/input/train/*jpg")]
test_data = [img for img in glob.glob("model_data/input/test1/*jpg")]

#Shuffle train_data
random.shuffle(train_data)

print('Creating train_lmdb and validation_lmdb ..')

num_train = 0
num_test = 0
in_db_train = lmdb.open(train_lmdb, map_size=int(1e12))
in_db_test = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db_train.begin(write=True) as in_txn_train:
    with in_db_test.begin(write=True) as in_txn_test:
        for in_idx, img_path in enumerate(train_data):
            if in_idx %  6 == 0:
                in_txn = in_txn_test
                num_test += 1
            else:
                in_txn = in_txn_train
                num_train += 1

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            if 'cat' in img_path:
                label = 0
            else:
                label = 1
            datum = make_datum(img, label)
            key = '{:0>5d}'.format(in_idx)
            value = datum.SerializeToString()
            in_txn.put(key.encode(), value)
            print(key + ':' + img_path)
in_db_train.close()
in_db_test.close()

print('\nFinished processing all images: {} train, {} test'.format(num_train, num_test))
