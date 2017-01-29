import tensorflow as tf
import numpy as np


class MaxPooling2D:
    """ Maximum pooling function.
    """

    def __init__(self, ksizes=None, strides=None, padding='SAME', flatten=False):
        """Inits function.

        Args:
            ksizes: A list of ints. 1-D of length 4.  The size of the window for each dimension of the input tensor.
                    Default is [1, 2, 2, 1].
            strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input.
                    Default is [1, 2, 2, 1].
            padding: Padding type. Available values: 'VALID' or 'SAME'.
            flatten: Whether to flatten output tensor.
        """
        self.ksizes = ksizes or [1, 2, 2, 1]
        self.strides = strides or [1, 2, 2, 1]
        self.padding = padding
        self.flatten = flatten

    @property
    def weights(self):
        return []

    def predict(self, x):
        """Processes given input vector.

        Args:
            x: Input vector to process.

        Returns:
            Processed vector
        """
        max_pool = tf.nn.max_pool(x, self.ksizes, self.strides, padding=self.padding)

        if self.flatten:
            shape = max_pool.get_shape().as_list()
            out_size = np.prod(shape[1:])
            max_pool = tf.reshape(max_pool, [shape[0], out_size])
        return max_pool
