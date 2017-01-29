import tensorflow as tf


class Sigmoid:
    """ Sigmoidal function.
    """

    def __init__(self):
        """Inits Sigmoid.
        """

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
        return tf.nn.sigmoid(x)
