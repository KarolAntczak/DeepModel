import numpy as np
import tensorflow as tf


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def l2_norm(weights):
    weights_flat = []
    for weight in weights:
        weights_flat = tf.concat_v2([weights_flat, tf.reshape(weight, [-1])], axis=0)
    return tf.reduce_mean(tf.square(weights_flat))
