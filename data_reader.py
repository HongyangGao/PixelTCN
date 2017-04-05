import os
import glob
import h5py
import random
import tensorflow as tf
from img_utils import get_images


class FileDataReader(object):

    def __init__(self, data_dir, input_height, input_width, height, width,
                 batch_size):
        self.data_dir = data_dir
        self.input_height, self.input_width = input_height, input_width
        self.height, self.width = height, width
        self.batch_size = batch_size
        self.image_files = glob.glob(data_dir+'*')

    def next_batch(self, batch_size):
        sample_files = np.random.choice(self.image_files, batch_size)
        images = get_images(
            sample_files, self.input_height, self.input_width,
            self.height, self.width)
        return images


class H5DataLoader(object):

    def __init__(self, data_path, is_train=True):
        self.is_train = is_train
        data_file = h5py.File(data_path, 'r')
        self.images, self.labels = data_file['X'][()], data_file['Y'][()]
        self.cur_index = 0

    def next_batch(self, batch_size):
        if self.is_train:
            indexes = random.sample(range(self.images.shape[0]), batch_size)
        else:
            indexes = range(self.cur_index, self.cur_index+batch_size)
            self.cur_index += len(indexes)
        return self.images[indexes], self.labels[indexes]


class QueueDataReader(object):

    def __init__(self, sess, data_dir, data_list, input_size, class_num,
                 name, data_format):
        self.sess = sess
        self.scope = name + '/data_reader'
        self.class_num = class_num
        self.channel_axis = 3
        images, labels = self.read_data(data_dir, data_list)
        images = tf.convert_to_tensor(images, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.string)
        queue = tf.train.slice_input_producer(
            [images, labels], shuffle=True, name=self.scope+'/slice')
        self.image, self.label = self.read_dataset(
            queue, input_size, data_format)

    def next_batch(self, batch_size):
        image_batch, label_batch = tf.train.shuffle_batch(
            [self.image, self.label], batch_size=batch_size,
            num_threads=4, capacity=50000, min_after_dequeue=10000,
            name=self.scope+'/batch')
        return image_batch, label_batch

    def read_dataset(self, queue, input_size, data_format):
        image = tf.image.decode_jpeg(
            tf.read_file(queue[0]), channels=3, name=self.scope+'/image')
        label = tf.image.decode_png(
            tf.read_file(queue[1]), channels=1, name=self.scope+'/label')
        image = tf.image.resize_images(image, input_size)
        label = tf.image.resize_images(label, input_size, 1)
        if data_format == 'NCHW':
            self.channel_axis = 1
            image = tf.transpose(image, [2, 0, 1])
            label = tf.transpose(label, [2, 0, 1])
        image -= tf.reduce_mean(tf.cast(image, dtype=tf.float32),
                                (0, 1), name=self.scope+'/mean')
        return image, label

    def read_data(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            images, labels = [], []
            for line in f:
                image, label = line.strip('\n').split(' ')
                images.append(data_dir + image)
                labels.append(data_dir + label)
        return images, labels

    def start(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(
            coord=self.coord, sess=self.sess)

    def close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
