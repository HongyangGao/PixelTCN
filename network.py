import tensorflow as tf
from enum import Enum
from ops import conv2d, pool2d, dilated_conv


class Axis(Enum):
    BATCH = 0
    CHANNEL = 3
    HEIGHT = 1
    WIDTH = 2


class DilatedPixelCNN(object):

    def __init__(self, sess, conf, height, width, channel):
        self.sess = sess
        self.conf = conf
        self.height, self.width, self.channel = height, width, channel
        self.start_channel_num = conf.start_channel_num
        self.class_num = conf.class_num
        self.network_depth = conf.network_depth
        self.conv_kernel_size = (3, 3)
        self.pool_kernel_size = (2, 2)
        if conf.use_gpu:
            self.data_format = 'NCHW'
            self.axis = (2, 3)
            self.input_shap = [None, self.channel, self.height, self.width]
        else:
            self.data_format = 'NHWC'
            self.axis = (1, 2)
            self.input_shap = [None, self.height, self.width, self.channel]

    def build_network(self, inputs):
        down_outputs = []
        for layer_index in range(self.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            inputs = self.construct_down_block(inputs, name, down_outputs, first=is_first)
        inputs = self.construct_bottom_block(inputs, 'bottom')
        for layer_index in range(self.network_depth-2, -1, -1):
            is_final = True if layer_index==0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            inputs = self.construct_up_block(inputs, down_inputs, name, final=is_final)
        return inputs

    def construct_down_block(self, inputs, name, down_outputs, first=False):
        num_outputs = self.start_channel_num if first else 2*inputs.shape[Axis.CHANNEL.value]
        conv1 = conv2d(inputs, num_outputs, self.conv_kernel_size, name+'/conv1', self.data_format)
        conv2 = conv2d(conv1, num_outputs, self.conv_kernel_size, name+'/conv2', self.data_format)
        down_outputs.append(conv2)
        pool = pool2d(conv2, self.pool_kernel_size, name+'/pool', self.data_format)
        return pool

    def construct_bottom_block(self, inputs, name):
        num_outputs = inputs.shape[Axis.CHANNEL.value]
        conv1 = conv2d(inputs, 2*num_outputs, self.conv_kernel_size, name+'/conv1', self.data_format)
        conv2 = conv2d(conv1, num_outputs, self.conv_kernel_size, name+'/conv2', self.data_format)
        return conv2

    def construct_up_block(self, inputs, down_inputs, name, final=False):
        num_outputs = inputs.shape[Axis.CHANNEL.value]
        conv1 = dilated_conv(inputs, num_outputs, self.conv_kernel_size, name+'/conv1',
            self.axis, self.data_format)
        conv1 = tf.concat([conv1, down_inputs], Axis.CHANNEL.value)
        conv2 = conv2d(conv1, num_outputs, self.conv_kernel_size, name+'/conv2', self.data_format)
        num_outputs = self.class_num if final else num_outputs/2
        conv3 = conv2d(conv2, num_outputs, self.conv_kernel_size, name+'/conv3', self.data_format)
        return conv3
        
    def train(self):
        pass
        # sess.run(tf.global_variables_initializer())

    def test(self):
        pass

    def generate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass