import tensorflow as tf


class Perceptron:
    """ Single layer perceptron network with sigmoidal activation.
    """

    def __init__(self, input_size, output_size):
        """Inits new perceptron.

        Args:
            input_size: Number of neurons in input layer.
            output_size: Number of neurons in output layer.
        """

        self.h1 = tf.Variable(tf.truncated_normal([input_size, output_size]))
        self.b1 = tf.Variable(tf.zeros([output_size]))

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
        return tf.nn.sigmoid(tf.matmul(x, self.h1) + self.b1)
