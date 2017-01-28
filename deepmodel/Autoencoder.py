import tensorflow as tf


class Autoencoder:
    """ Shallow autoencoding network with sigmoidal encoder and decoder.
    """

    def __init__(self, input_size, hidden_size):
        """Inits new autoencoder.

        Args:
            input_size: Number of neurons in input layer.
            hidden_size: Number of neurons in hidden layer.
        """

        self.h1 = tf.Variable(tf.truncated_normal([input_size, hidden_size]))
        self.b1 = tf.Variable(tf.zeros([hidden_size]))
        self.h2 = tf.Variable(tf.truncated_normal([hidden_size, input_size]))
        self.b2 = tf.Variable(tf.zeros([input_size]))

    def __str__(self):
        return "h1%s b1%s h2%s b2%s" % (self.h1.eval().flatten(), self.b1.eval().flatten(),
                                        self.h2.eval().flatten(), self.b2.eval().flatten())

    @property
    def weights(self):
        return [self.h1, self.b1, self.h2, self.b2]

    def encode(self, x):
        """Encodes given input.

        Args:
            x: Input vector to encode.

        Returns:
            Encoded vector.
        """
        return tf.nn.sigmoid(tf.matmul(x, self.h1) + self.b1)

    def predict(self, x):
        """Encodes and then decodes given input vector

        Args:
            x: Input vector to recover.

        Returns:
            Recovered vector, i.e. encoded and decoded.
        """
        encoded = self.encode(x)
        return tf.matmul(encoded, self.h2) + self.b2
