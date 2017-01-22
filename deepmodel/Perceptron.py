import tensorflow as tf


class Perceptron:
    """ Single layer perceptron network with sigmoidal activation.
    """

    def __init__(self, input_size, output_size):
        """Inits new autoencoder.

        Args:
            input_size: Number of neurons in input layer.
            output_size: Number of neurons in output layer.
        """

        self.h1 = tf.Variable(tf.random_normal([input_size, output_size]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.random_normal([output_size]), dtype=tf.float32)

        tf.variables_initializer([self.h1, self.b1]).run()

    def encode(self, x):
        """Encodes given input.

        Args:
            x: Input vector to encode.

        Returns:
            Encoded vector.
        """
        return tf.nn.sigmoid(tf.add(tf.matmul(x, self.h1), self.b1))

    def decode(self, y):
        """Decodes given encoded vector.

        Args:
            y: Vector to decode.

        Returns:
            Decoded vector.
        """
        return y

    def recover(self, x):
        """Encodes and then decodes given input vector

        Args:
            x: Input vector to recover.

        Returns:
            Recovered vector, i.e. encoded and decoded.
        """
        encoded = self.encode(x)
        return self.decode(encoded)

    @staticmethod
    def get_next_batch(train_dataset, step, batch_size):
        offset = 0 if batch_size == 1 else (step * batch_size) % (len(train_dataset) - batch_size)
        return train_dataset[offset:offset + batch_size]

    def train(self, train_dataset, train_labels, steps=1000, batch_size=128, learning_rate=0.05, momentum=0.99, l2=40):
        """Train model.

        Args:
            train_dataset: Train dataset.
            train_labels: Train dataset labels.
            steps: Maximum number of training steps.
            batch_size: number of examples from dataset used during each step.
            learning_rate: learning rate of optimizer.
            momentum: momentum parameter of optimizer.
            l2: L2 regularization parameter (weights size penalty)

        Yields:
            Loss value in current training step.
        """

        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, train_dataset.shape[1]))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, train_labels.shape[1]))
        recovered = self.recover(tf_train_dataset)

        loss = tf.reduce_mean(tf.square(tf.sub(tf_train_labels, recovered)))

        weights_penalty = l2 * (tf.reduce_mean(tf.square(self.h1)) + tf.reduce_mean(tf.square(self.b1)))
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss + weights_penalty)

        momentum_initializer = [var for var in tf.global_variables() if 'Momentum' in var.name]

        tf.variables_initializer(momentum_initializer).run()

        for step in range(steps):
            batch_data = self.get_next_batch(train_dataset, step, batch_size)
            batch_labels = self.get_next_batch(train_labels, step, batch_size)
            _, l = tf.get_default_session().run((optimizer, loss), feed_dict={tf_train_dataset: batch_data,
                                                                              tf_train_labels: batch_labels})
            yield l

    def print(self):
        print("h1%s b1%s" % (self.h1.eval().flatten(), self.b1.eval().flatten()))
