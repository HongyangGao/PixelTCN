import os
import time
import tensorflow as tf
import numpy as np
from network import DilatedPixelCNN
from tensorflow.examples.tutorials.mnist import input_data


def configure():
    flags = tf.app.flags
    # training
    flags.DEFINE_float("max_epoch", 100000, "# of step in an epoch")
    flags.DEFINE_float("test_step", 100, "# of step to test a model")
    flags.DEFINE_float("save_step", 1000, "# of step to save a model")
    flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
    # data
    flags.DEFINE_string("dataset", "mnist", "Name of dataset [mnist, cifar]")
    flags.DEFINE_string("data_dir", "data", "Name of data directory")
    flags.DEFINE_string("sample_dir", "samples", "Sample directory")
    # Debug
    flags.DEFINE_boolean("is_train", True, "Training or testing")
    flags.DEFINE_string("log_level", "INFO", "Log level")
    flags.DEFINE_integer("random_seed", int(time.time()), "random seed")
    # network
    flags.DEFINE_integer('network_depth', 5, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 21, 'output class number')
    flags.DEFINE_integer('start_channel_num', 64, 'start number of outputs')
    flags.DEFINE_boolean('use_gpu', False, 'use GPU or not')

    return flags.FLAGS


def main():
    conf = configure()
    sess = tf.Session()
    model = DilatedPixelCNN(sess, conf, 3, 16, 16, 3)
    inputs = np.ones((3,16,16,3))
    model.train(inputs)
    writer = tf.summary.FileWriter('./my_graph', model.sess.graph)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()
