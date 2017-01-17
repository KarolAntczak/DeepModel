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
        self.h1 = tf.Variable(tf.random_normal([input_size, hidden_size]), dtype=tf.float32)
        self.h2 = tf.Variable(tf.random_normal([hidden_size, input_size]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.random_normal([hidden_size]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.random_normal([input_size]), dtype=tf.float32)

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
        return tf.add(tf.matmul(y, self.h2), self.b2)

    def recover(self, x):
        """Encodes and then decodes given input vector

        Args:
            x: Input vector to recover.

        Returns:
            Recovered vector, i.e. encoded and decoded.
        """
        encoded = self.encode(x)
        return self.decode(encoded)

    @property
    def _vars(self):
        """Get all variables used by this model"""
        return [self.h1, self.h2, self.b1, self.b2]

    def train(self, train_dataset, steps=1000, batch_size=256, learning_rate=0.05, momentum=0.99):
        """Train model.

        Args:
            train_dataset: Train dataset of unlabeled data.
            steps: Maximum number of training steps.
            batch_size: number of examples from dataset used during each step.
            learning_rate: learning rate of optimizer.
            momentum: momentum parameter of optimizer.

        Yields:
            Loss value in current training step.
        """

        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, train_dataset.shape[1]))

        recovered = self.recover(tf_train_dataset)
        loss = tf.reduce_mean(tf.square(tf.sub(tf_train_dataset, recovered)))
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, var_list=self._vars)

        momentum_initializers = [var for var in tf.global_variables() if 'Momentum' in var.name]

        tf.variables_initializer(momentum_initializers + self._vars).run()

        for step in range(steps):
            offset = 0 if batch_size == 1 else (step * batch_size) % (len(train_dataset) - batch_size)
            batch_data = train_dataset[offset:offset+batch_size]
            _, l = tf.get_default_session().run((optimizer, loss), feed_dict={tf_train_dataset: batch_data})
            yield l
