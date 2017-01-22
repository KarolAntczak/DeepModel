import tensorflow as tf

from deepmodel import Perceptron


class Autoencoder(Perceptron):
    """ Shallow autoencoding network with sigmoidal encoder and decoder.
    """

    def __init__(self, input_size, hidden_size):
        """Inits new autoencoder.

        Args:
            input_size: Number of neurons in input layer.
            hidden_size: Number of neurons in hidden layer.
        """

        super(Autoencoder, self).__init__(input_size, hidden_size)

        self.h2 = tf.Variable(tf.random_normal([hidden_size, input_size]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.random_normal([input_size]), dtype=tf.float32)

        tf.variables_initializer([self.h2, self.b2]).run()

    def encode(self, x):
        """Encodes given input.

        Args:
            x: Input vector to encode.

        Returns:
            Encoded vector.
        """
        return tf.nn.sigmoid(tf.add(tf.matmul(x, self.h1), self.b1))

    def recover(self, x):
        """Encodes and then decodes given input vector

        Args:
            x: Input vector to recover.

        Returns:
            Recovered vector, i.e. encoded and decoded.
        """
        encoded = self.encode(x)
        return tf.add(tf.matmul(encoded, self.h2), self.b2)

    @staticmethod
    def get_next_batch(train_dataset, step, batch_size):
        offset = 0 if batch_size == 1 else (step * batch_size) % (len(train_dataset) - batch_size)
        return train_dataset[offset:offset + batch_size]

    def train(self, train_dataset, steps=1000, batch_size=128, learning_rate=0.05, momentum=0.99, l2=40):
        """Train model.

        Args:
            train_dataset: Train dataset of unlabeled data.
            steps: Maximum number of training steps.
            batch_size: number of examples from dataset used during each step.
            learning_rate: learning rate of optimizer.
            momentum: momentum parameter of optimizer.
            l2: L2 regularization parameter (weights size penalty)

        Yields:
            Loss value in current training step.
        """

        for l in super(Autoencoder, self).train(train_dataset, train_dataset, steps, batch_size, learning_rate,
                                                momentum, l2):
            yield l
