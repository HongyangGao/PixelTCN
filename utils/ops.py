import tensorflow as tf
import numpy as np
from . import pixel_dcn


"""
This module provides some short functions to reduce code volume
"""


<<<<<<< HEAD
def pixel_dcl(inputs, out_num, kernel_size, scope, data_type='2D'):
    if data_type == '2D':
        outputs = pixel_dcn.pixel_dcl(inputs, out_num, kernel_size, scope, None)
    else:
        outputs = pixel_dcn.pixel_dcl3d(inputs, out_num, kernel_size, scope, None)
=======
def conv2d(inputs, num_outputs, kernel_size, scope, d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope,
        data_format=d_format, activation_fn=None, biases_initializer=None)
>>>>>>> d15aee2bbbe7072c83fac644959ca096a3e25a1c
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')


<<<<<<< HEAD
def ipixel_cl(inputs, out_num, kernel_size, scope, data_type='2D'):
    # only support 2d
    outputs = pixel_dcn.ipixel_cl(inputs, out_num, kernel_size, scope, None)
=======
def deconv(inputs, out_num, kernel_size, scope, d_format='NHWC'):
    kernel_size = [i*2 for i in kernel_size]
    outputs = tf.contrib.layers.conv2d_transpose(
        inputs, out_num, kernel_size, scope=scope, stride=[2, 2],
        data_format=d_format, activation_fn=None, biases_initializer=None)
>>>>>>> d15aee2bbbe7072c83fac644959ca096a3e25a1c
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')


def ipixel_dcl(inputs, out_num, kernel_size, scope, data_type='2D'):
    # only support 2d
    if data_type == '2D':
        outputs = pixel_dcn.ipixel_dcl(inputs, out_num, kernel_size, scope, None)
    else:
        outputs = pixel_dcn.ipixel_dcl3d(inputs, out_num, kernel_size, scope, None)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')


def conv(inputs, out_num, kernel_size, scope, data_type='2D'):
    if data_type == '2D':
        outputs = tf.layers.conv2d(
            inputs, out_num, kernel_size, padding='same', name=scope+'/conv',
            kernel_initializer=tf.truncated_normal_initializer)
    else:
        shape = list(kernel_size) + [inputs.shape[-1].value, out_num]
        weights = tf.get_variable(
            scope+'/conv/weights', shape, initializer=tf.truncated_normal_initializer())
        outputs = tf.nn.conv3d(
            inputs, weights, (1, 1, 1, 1, 1), padding='SAME', name=scope+'/conv')
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')


def deconv(inputs, out_num, kernel_size, scope, data_type='2D'):
    if data_type == '2D':
        outputs = tf.layers.conv2d_transpose(
            inputs, out_num, kernel_size, (2, 2), padding='same', name=scope,
            kernel_initializer=tf.truncated_normal_initializer)
    else:
        shape = list(kernel_size) + [out_num, out_num]
        input_shape = inputs.shape.as_list()
        out_shape = [input_shape[0]] + \
            list(map(lambda x: x*2, input_shape[1:-1])) + [out_num]
        weights = tf.get_variable(
            scope+'/deconv/weights', shape, initializer=tf.truncated_normal_initializer())
        outputs = tf.nn.conv3d_transpose(
            inputs, weights, out_shape, (1, 2, 2, 2, 1), name=scope+'/deconv')
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
        updates_collections=None, scope=scope+'/batch_norm')


def pool(inputs, kernel_size, scope, data_type='2D'):
    if data_type == '2D':
        return tf.layers.max_pooling2d(inputs, kernel_size, (2, 2), name=scope)
    return tf.layers.max_pooling3d(inputs, kernel_size, (2, 2, 2), name=scope)
