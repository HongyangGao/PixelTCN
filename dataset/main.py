#from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
from  Data_Reader import DataReader
from utils import decode_labels

BATCH_SIZE = 5
DATA_DIRECTORY = '/tempspace/hyuan/DrSleep/VOC2012/VOCdevkit/VOC2012'
DATA_LIST_PATH = './dataset/train1.txt'
INPUT_SIZE = '320,320'
RANDOM_SCALE = True
SAVE_DIR = './images_val/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted masks.")
    return parser.parse_args()


args = get_arguments()
h, w = map(int, args.input_size.split(','))
input_size = (h, w)

# Load reader.
with tf.name_scope("create_inputs"):
    reader = DataReader(
        args.data_dir,
        args.data_list,
        input_size,
        RANDOM_SCALE,
        coord)
    image_batch, label_batch = reader.next_batch(args.batch_size)


#print args.batch_size
#image_batch, label_batch = reader.next_batch(args.batch_size)

#print image_batch.get_shape()
#label_batch = tf.squeeze(label_batch, squeeze_dims=[0])

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
label_batch= sess.run(label_batch)
# label_batch= label_batch*255
# for i in range(21):
#     cv2.imwrite(str(i)+'messigray.png',label_batch[:,:,i])
# img = decode_labels(label_batch[0, :, :,1])
# im = Image.fromarray(img)
# im.save(args.save_dir + 'test' + '.png')

coord.request_stop()
coord.join(threads)
# sess = tf.Session()
# sess.run(image_batch)


        
