import tensorflow as tf
import numpy as np
from . import pixel_dcn


"""
This module provides some short functions to reduce code volume
"""


def conv3d(inputs, num_outputs, kernel_size, scope, d_format='NHWC'):
    inputs = tf.expand_dims(inputs, axis=-1, name=scope+'/expand_dims')
    shape = [list(kernel_size)[0]]*3 + [1, 1]
    weights = tf.get_variable(
        scope+'/conv/weights', shape, initializer=tf.truncated_normal_initializer())
    outputs = tf.nn.conv3d(
        inputs, weights, (1, 1, 1, 1, 1), padding='SAME', name=scope+'/conv')
    outputs = tf.squeeze(outputs, axis=[-1], name=scope+'/squeeze')
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
        updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
        data_format=d_format)


def conv2d(inputs, num_outputs, kernel_size, scope, d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope,
        data_format=d_format, activation_fn=None, biases_initializer=None)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
        updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
        data_format=d_format)


def deconv(inputs, out_num, kernel_size, scope, d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d_transpose(
        inputs, out_num, kernel_size, scope=scope, stride=[2, 2],
        data_format=d_format, activation_fn=None, biases_initializer=None)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)


def ipixel_cl(inputs, out_num, kernel_size, scope, d_format='NHWC'):
    outputs = pixel_dcn.ipixel_cl(
        inputs, out_num, kernel_size, scope, None, d_format)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)


def pixel_dcl(inputs, out_num, kernel_size, scope, d_format='NHWC'):
    outputs = pixel_dcn.pixel_dcl(
        inputs, out_num, kernel_size, scope, None, d_format)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)


def ipixel_dcl(inputs, out_num, kernel_size, scope, d_format='NHWC'):
    outputs = pixel_dcn.ipixel_dcl(
        inputs, out_num, kernel_size, scope, None, d_format)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)


def pool2d(inputs, kernel_size, scope, data_format='NHWC'):
    return tf.contrib.layers.max_pool2d(
        inputs, kernel_size, scope=scope, padding='SAME',
        data_format=data_format)
