import tensorflow as tf


class BatchDataReader(object):

    def __init__(self, sess, data_dir, data_list, input_size):
        self.sess = sess
        images, labels = self.read_data(data_dir, data_list)
        images = tf.convert_to_tensor(images, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.string)
        queue = tf.train.slice_input_producer([images, labels])
        self.images, self.labels = self.read_dataset(queue, input_size)

    def next_batch(self, batchsize):
        image_batch, label_batch = tf.train.batch([self.images, self.labels], batchsize)
        label_batch = tf.squeeze(label_batch, axis=[3])
        import ipdb; ipdb.set_trace()
        label_batch = tf.contrib.layers.one_hot_encoding(labels=label_batch, num_classes=21)
        label_batch = tf.one_hot(label_batch, depth=21)
        #label_batch = tf.reshape(label_batch, [-1, 21])
        return image_batch, label_batch

    def read_dataset(self, queue, input_size):
        images = tf.image.decode_jpeg(tf.read_file(queue[0]), channels=3)
        labels = tf.image.decode_png(tf.read_file(queue[1]), channels= 1)
        images = tf.image.resize_images(images, input_size)
        labels = tf.image.resize_images(labels, input_size)
        images -= tf.reduce_mean(tf.cast(images, dtype= tf.float32), (0, 1))
        return images, labels

    def read_data(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            images, labels = [], []
            for line in f:
                image, label= line.strip('\n').split(' ')
                images.append(data_dir + image)
                labels.append(data_dir + label)
        return images, labels

    def __enter__(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
        return self

    def __exit__(self, type, value, traceback):
        print('---->batchdatareader exit')
        self.coord.request_stop()
        self.coord.join(self.threads)
