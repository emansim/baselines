import time

import gym
import numpy as np
import tensorflow as tf


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def mean(value):
    if value == []:
        return 0.
    else:
        return np.mean(np.asarray(value))

def std(value):
    if value == []:
        return 1.
    else:
        return np.std(np.asarray(value))
