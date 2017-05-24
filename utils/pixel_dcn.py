import tensorflow as tf
import numpy as np


"""
This module realizes the three methods proposed in paper
[Pixel Deconvolutional Networks] (https://arxiv.org/abs/1705.06820)

pixel_dcl: realizes Pixel Deconvolutional Layer
ipixel_dcl: realizes Input Pixel Deconvolutional Layer
ipixel_dcl: realizes Input Pixel Convolutional Layer
"""

def pixel_dcl(inputs, out_num, kernel_size, scope, activation_fn=tf.nn.relu,
              d_format='NHWC'):
    """
    inputs: input tensor
    out_num: output channel number
    kernel_size: convolutional kernel size
    scope: operation scope
    activation_fn: activation function, could be None if needed
    """
    axis = (d_format.index('H'), d_format.index('W'))
    conv0 = conv2d(inputs, out_num, kernel_size, scope+'/conv0')
    conv1 = conv2d(conv0, out_num, kernel_size, scope+'/conv1')
    dilated_conv0 = dilate_tensor(conv0, axis, 0, 0, scope+'/dialte_conv0')
    dilated_conv1 = dilate_tensor(conv1, axis, 1, 1, scope+'/dialte_conv1')
    conv1 = tf.add(dilated_conv0, dilated_conv1, scope+'/add1')
    with tf.variable_scope(scope+'/conv2'):
        shape = list(kernel_size) + [out_num, out_num]
        weights = tf.get_variable(
            'weights', shape, initializer=tf.truncated_normal_initializer())
        weights = tf.multiply(weights, get_mask(shape, scope))
        strides = [1, 1, 1, 1]
        conv2 = tf.nn.conv2d(conv1, weights, strides, padding='SAME',
                             data_format=d_format)
    outputs = tf.add(conv1, conv2, name=scope+'/add2')
    if activation_fn:
        outputs = activation_fn(outputs)
    return outputs


def ipixel_dcl(inputs, out_num, kernel_size, scope, activation_fn=tf.nn.relu,
               d_format='NHWC'):
    """
    inputs: input tensor
    out_num: output channel number
    kernel_size: convolutional kernel size
    scope: operation scope
    activation_fn: activation function, could be None if needed
    """
    axis = (d_format.index('H'), d_format.index('W'))
    channel_axis = d_format.index('C')
    conv1 = conv2d(inputs, out_num, kernel_size, scope+'/conv1')
    conv1_concat = tf.concat(
        [inputs, conv1], channel_axis, name=scope+'/concat1')
    conv2 = conv2d(conv1_concat, out_num, kernel_size, scope+'/conv2')
    conv2_concat = tf.concat(
        [conv1_concat, conv2], channel_axis, name=scope+'/concat2')
    conv3 = conv2d(conv2_concat, 2*out_num, kernel_size, scope+'/conv3')
    conv4, conv5 = tf.split(conv3, 2, channel_axis, name=scope+'/split')
    dialte1 = dilate_tensor(conv1, axis, 0, 0, scope+'/dialte1')
    dialte2 = dilate_tensor(conv2, axis, 1, 1, scope+'/dialte2')
    dialte3 = dilate_tensor(conv4, axis, 1, 0, scope+'/dialte3')
    dialte4 = dilate_tensor(conv5, axis, 0, 1, scope+'/dialte4')
    outputs = tf.add_n([dialte1, dialte2, dialte3, dialte4], scope+'/add')
    if activation_fn:
        outputs = activation_fn(outputs)
    return outputs


def ipixel_cl(inputs, out_num, kernel_size, scope, activation_fn=tf.nn.relu,
              d_format='NHWC'):
    """
    inputs: input tensor
    out_num: output channel number
    kernel_size: convolutional kernel size
    scope: operation scope
    activation_fn: activation function, could be None if needed
    """
    axis = (d_format.index('H'), d_format.index('W'))
    channel_axis = d_format.index('C')
    conv1 = tf.contrib.layers.conv2d(
        inputs, out_num, kernel_size, stride=2, scope=scope+'/conv1',
        data_format=d_format, activation_fn=None, biases_initializer=None)
    dialte1 = dilate_tensor(conv1, axis, 0, 0, scope+'/dialte1')
    shifted_inputs = shift_tensor(inputs, axis, 1, 1, scope+'/shift1')
    conv1_concat = tf.concat(
        [shifted_inputs, dialte1], channel_axis, name=scope+'/concat1')
    conv2 = tf.contrib.layers.conv2d(
        conv1_concat, out_num, kernel_size, stride=2, scope=scope+'/conv2',
        data_format=d_format, activation_fn=None, biases_initializer=None)
    dialte2 = dilate_tensor(conv2, axis, 1, 1, scope+'/dialte2')
    conv3 = tf.add_n([dialte1, dialte2], scope+'/add')
    shifted_inputs = shift_tensor(inputs, axis, 1, 0, scope+'/shift2')
    conv2_concat = tf.concat(
        [shifted_inputs, conv3], channel_axis, name=scope+'/concat2')
    conv4 = tf.contrib.layers.conv2d(
        conv2_concat, out_num, kernel_size, stride=2, scope=scope+'/conv4',
        data_format=d_format, activation_fn=None, biases_initializer=None)
    dialte3 = dilate_tensor(conv4, axis, 1, 0, scope+'/dialte3')
    shifted_inputs = shift_tensor(inputs, axis, 0, 1, scope+'/shift3')
    conv2_concat = tf.concat(
        [shifted_inputs, conv3], channel_axis, name=scope+'/concat3')
    conv5 = tf.contrib.layers.conv2d(
        conv2_concat, out_num, kernel_size, stride=2, scope=scope+'/conv5',
        data_format=d_format, activation_fn=None, biases_initializer=None)
    dialte4 = dilate_tensor(conv5, axis, 0, 1, scope+'/dialte4')
    outputs = tf.add_n([dialte1, dialte2, dialte3, dialte4], scope+'/add')
    if activation_fn:
        outputs = activation_fn(outputs)
    return outputs


def conv2d(inputs, num_outputs, kernel_size, scope, d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope,
        data_format=d_format, activation_fn=None, biases_initializer=None)
    return outputs


def get_mask(shape, scope):
    new_shape = (shape[0]*shape[1], shape[2], shape[3])
    mask = np.ones(new_shape, dtype=np.float32)
    for i in range(0, new_shape[0], 2):
        mask[i, :, :] = 0
    mask = np.reshape(mask, shape, 'F')
    return tf.constant(mask, dtype=tf.float32, name=scope+'/mask')


def shift_tensor(inputs, axis, row_shift, column_shift, scope):
    if row_shift:
        rows = tf.unstack(inputs, axis=axis[0], name=scope+'/rowsunstack')
        row_zeros = tf.zeros_like(
            rows[0], dtype=tf.float32, name=scope+'/rowzeros')
        rows = rows[row_shift:] + [row_zeros]*row_shift
        inputs = tf.stack(rows, axis=axis[0], name=scope+'/rowsstack')
    if column_shift:
        columns = tf.unstack(
            inputs, axis=axis[1], name=scope+'/columnsunstack')
        columns_zeros = tf.zeros_like(
            columns[0], dtype=tf.float32, name=scope+'/columnzeros')
        columns = columns[column_shift:] + [columns_zeros]*column_shift
        inputs = tf.stack(columns, axis=axis[1], name=scope+'/columnsstack')
    return inputs


def dilate_tensor(inputs, axis, row_shift, column_shift, scope):
    rows = tf.unstack(inputs, axis=axis[0], name=scope+'/rowsunstack')
    row_zeros = tf.zeros_like(
        rows[0], dtype=tf.float32, name=scope+'/rowzeros')
    for index in range(len(rows), 0, -1):
        rows.insert(index-row_shift, row_zeros)
    inputs = tf.stack(rows, axis=axis[0], name=scope+'/rowsstack')
    columns = tf.unstack(
        inputs, axis=axis[1], name=scope+'/columnsunstack')
    columns_zeros = tf.zeros_like(
        columns[0], dtype=tf.float32, name=scope+'/columnzeros')
    for index in range(len(columns), 0, -1):
        columns.insert(index-column_shift, columns_zeros)
    inputs = tf.stack(
        columns, axis=axis[1], name=scope+'/columnsstack')
    return inputs
