import tensorflow as tf


class ReLU:
    """ Rectified linear function ( f(x) = max(0,x) ).
    """

    def __init__(self,):
        """Inits function.
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
        return tf.nn.relu(x)
