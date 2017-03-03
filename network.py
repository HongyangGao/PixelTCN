import tensorflow as tf
from data_reader import BatchData
from ops import conv2d, pool2d, dilated_conv


class DilatedPixelCNN(object):

    def __init__(self, sess, conf, batch, height, width, channel):
        self.sess = sess
        self.conf = conf
        self.height, self.width, self.channel, self.batch = height, width, channel, batch
        self.conv_kernel_size = (3, 3)
        self.pool_kernel_size = (2, 2)
        if conf.use_gpu:
            self.data_format = 'NCHW'
            self.axis, self.channel_axis = (2, 3), 1
            self.input_shape = [self.batch, self.channel, self.height, self.width]
            self.output_shape = [self.batch, self.conf.class_num, self.height, self.width]
        else:
            self.data_format = 'NHWC'
            self.axis, self.channel_axis = (1, 2), 3
            self.input_shape = [self.batch, self.height, self.width, self.channel]
            self.output_shape = [self.batch, self.height, self.width, self.conf.class_num]
        self.inputs = tf.placeholder(tf.float32, self.input_shape, 'inputs')
        self.annotations = tf.placeholder(tf.float32, self.output_shape, 'annotations')
        self.build_network()
        tf.set_random_seed(conf.random_seed)
        self.sess.run(tf.global_variables_initializer())

    def build_network(self):
        outputs = self.inputs
        down_outputs = []
        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            outputs = self.construct_down_block(outputs, name, down_outputs, first=is_first)
        outputs = self.construct_bottom_block(outputs, 'bottom')
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_final = True if layer_index==0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            outputs = self.construct_up_block(outputs, down_inputs, name, final=is_final)
        optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        self.prediction = outputs
        self.loss_op = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(self.annotations, outputs))
        self.train_op = optimizer.minimize(loss, 'train_op')

    def construct_down_block(self, inputs, name, down_outputs, first=False):
        num_outputs = self.conf.start_channel_num if first else 2*inputs.shape[channel_axis].value
        conv1 = conv2d(inputs, num_outputs, self.conv_kernel_size, name+'/conv1', self.data_format)
        conv2 = conv2d(conv1, num_outputs, self.conv_kernel_size, name+'/conv2', self.data_format)
        down_outputs.append(conv2)
        pool = pool2d(conv2, self.pool_kernel_size, name+'/pool', self.data_format)
        return pool

    def construct_bottom_block(self, inputs, name):
        num_outputs = inputs.shape[channel_axis].value
        conv1 = conv2d(inputs, 2*num_outputs, self.conv_kernel_size, name+'/conv1', self.data_format)
        conv2 = conv2d(conv1, num_outputs, self.conv_kernel_size, name+'/conv2', self.data_format)
        return conv2

    def construct_up_block(self, inputs, down_inputs, name, final=False):
        num_outputs = inputs.shape[channel_axis].value
        conv1 = dilated_conv(inputs, num_outputs, self.conv_kernel_size, name+'/conv1',
            self.axis, self.data_format)
        conv1 = tf.concat([conv1, down_inputs], channel_axis, name=name+'/concat')
        conv2 = conv2d(conv1, num_outputs, self.conv_kernel_size, name+'/conv2', self.data_format)
        num_outputs = self.conf.class_num if final else num_outputs/2
        conv3 = conv2d(conv2, num_outputs, self.conv_kernel_size, name+'/conv3', self.data_format)
        return conv3

    def train(self, data):
        train_images = []
        data_reader = BatchData(train_images)
        for iter_num in range(self.conf.max_epoch):
            images, annotations = data_reader.next_batch(self.batch)
            feed_dict = {self.inputs: images, self.annotations: annotations}
            loss, _ = self.sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

            if iter_num % self.conf.save_step:
                self.save()
            if iter_num % self.conf.test_step:
                self.test()

    def test(self):
        print('---->testing')
        pass

    def predict(self):
        pass

    def save(self):
        print('---->saving')
        pass

    def load(self):
        pass