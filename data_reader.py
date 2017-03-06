import tensorflow as tf


class BatchDataReader(object):

    def __init__(self, sess, data_dir, data_list, input_size):
        self.sess = sess
        self.scope = 'data_reader'
        images, labels = self.read_data(data_dir, data_list)
        images = tf.convert_to_tensor(images, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.string)
        queue = tf.train.slice_input_producer([images, labels], name=self.scope+'/slice')
        self.image, self.label = self.read_dataset(queue, input_size)

    def next_batch(self, batchsize):
        image_batch, label_batch = tf.train.batch([self.image, self.label], batchsize, name=self.scope+'/batch')
        label_batch = tf.one_hot(tf.squeeze(label_batch, axis=[3], name=self.scope+'/squeeze'),
            depth=21, name=self.scope+'/one_hot')
        return image_batch, label_batch

    def read_dataset(self, queue, input_size):
        image = tf.image.decode_jpeg(tf.read_file(queue[0]), channels=3, name=self.scope+'/image')
        label = tf.image.decode_png(tf.read_file(queue[1]), channels= 1, name=self.scope+'/label')
        image = tf.image.resize_images(image, input_size)
        label = tf.image.resize_images(label, input_size, 1)
        image -= tf.reduce_mean(tf.cast(image, dtype= tf.float32), (0, 1), name=self.scope+'/mean')
        return image, label

    def read_data(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            images, labels = [], []
            for line in f:
                image, label= line.strip('\n').split(' ')
                images.append(data_dir + image)
                labels.append(data_dir + label)
        return images, labels

    def start(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    def close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
