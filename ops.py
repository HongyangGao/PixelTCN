import tensorflow as tf
import numpy as np


def conv2d(inputs, num_outputs, kernel_size, scope, data_format):
    return tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope,
        data_format=data_format)


def pool2d(inputs, kernel_size, scope, data_format):
    return tf.contrib.layers.max_pool2d(
        inputs, kernel_size, scope=scope, padding='SAME',
        data_format=data_format)


def dilated_conv(inputs, num_outputs, kernel_size, scope, axis, data_format):
    conv1 = conv2d(inputs, num_outputs, kernel_size, scope+'/conv1', data_format)
    dilated_inputs = dilate_tensor(inputs, axis, 0, scope+'/dialteinputs')
    dilated_conv1 = dilate_tensor(conv1, axis, 1, scope+'/dialteconv1')
    conv1 = tf.add(dilated_inputs, dilated_conv1, scope+'/add1')
    with tf.variable_scope(scope+'/conv2'):
        shape = list(kernel_size) + [num_outputs, num_outputs]
        weights = tf.get_variable(
            'weights', shape, initializer=tf.truncated_normal_initializer())
        weights = tf.multiply(weights, get_mask(shape, scope))
        strides = [1, 1, 1, 1]
        conv2 = tf.nn.conv2d(conv1, weights, strides, padding='SAME',
            data_format=data_format)
    return tf.nn.relu(tf.add(conv1, conv2, name=scope+'/add2'), name=scope+'/relu')


def get_mask(shape, scope):
    new_shape = (shape[0]*shape[1], shape[2], shape[3])
    mask = np.ones(new_shape, dtype=np.float32)
    for i in range(0, new_shape[0], 2):
        mask[i,:,:] = 0
    return tf.constant(np.reshape(mask, shape, 'F'), dtype=tf.float32, name=scope+'/mask')


def dilate_tensor(inputs, axis, shift, scope):
    rows = tf.unstack(inputs, axis=axis[0], name=scope+'/rowsunstack')
    row_zeros = tf.zeros(rows[0].shape, dtype=tf.float32, name=scope+'/rowzeros')
    for index in range(len(rows), 0, -1):
        rows.insert(index-shift, row_zeros)
    row_outputs = tf.stack(rows, axis=axis[0], name=scope+'/rowsstack')
    columns =  tf.unstack(row_outputs, axis=axis[1], name=scope+'/columnsunstack')
    columns_zeros = tf.zeros(columns[0].shape, dtype=tf.float32, name=scope+'/columnzeros')
    for index in range(len(columns), 0, -1):
        columns.insert(index-shift, columns_zeros)
    column_outputs = tf.stack(columns, axis=axis[1], name=scope+'/columnsstack')
    return column_outputs
