import tensorflow as tf
import numpy as np


class Convolution2D:
    """ Single layer of 2D convolution
    """

    def __init__(self, filter_height, filter_width, in_channels, out_channels, strides=None, padding='SAME',
                 flatten=False):
        """Inits new convolutional layer.

        Args:
            filter_height: Width of input filter.
            filter_width: Height of input filter.
            in_channels: Number of input channels. for greyscaled images is typically 1, for RGB - 3, etc.
            out_channels: Number of output channels.
            strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input.
                    By default it is [1, 1, 1, 1].
            padding: Padding type. Available values: 'VALID' or 'SAME'.
            flatten: whether to flatten output tensor
        """
        self.h1 = tf.Variable(tf.truncated_normal([filter_height, filter_width, in_channels, out_channels]))
        self.b1 = tf.Variable(tf.zeros([out_channels]))
        self.strides = strides or [1, 1, 1, 1]
        self.padding = padding
        self.flatten = flatten

    def __str__(self):
        return "h1%s b1%s" % (self.h1.eval().flatten(), self.b1.eval().flatten())

    @property
    def weights(self):
        return [self.h1, self.b1]

    def predict(self, x):
        """Processes given input vector

        Args:
            x: Input vector to process.

        Returns:
            Processed vector
        """
        conv = tf.nn.conv2d(x, self.h1, self.strides, self.padding) + self.b1

        if self.flatten:
            shape = conv.get_shape().as_list()
            out_size = np.prod(shape[1:])
            conv = tf.reshape(conv, [shape[0], out_size])

        return conv

