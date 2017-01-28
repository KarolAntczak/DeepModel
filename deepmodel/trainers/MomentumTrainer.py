import tensorflow as tf
from deepmodel.trainers.Batch import *


class MomentumTrainer:

    def __init__(self, model, train_dataset, train_labels, batch_size=128, learning_rate=0.04, momentum=0.99,
                 weight_penalty=0):
        """Inits trainer.

        Args:
            model: model to train
            train_dataset: Train dataset.
            train_labels: Train dataset labels.
            batch_size: number of examples from dataset used during each step.
            weight_penalty: penalty factor for weights size
        """
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, train_dataset.shape[1]))
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, train_labels.shape[1]))
        tr_train_predicted = model.predict(self.tf_train_dataset)
        weight_loss = weight_penalty*self.l2_norm(model.weights)
        self.loss = tf.reduce_mean(tf.square(tf.sub(self.tf_train_labels, tr_train_predicted))) + weight_loss

        self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.loss)
        self.model = model
        self.batch_size = batch_size

    def train(self, steps=1000):
        """Train model.

        Args:
            steps: Maximum number of training steps.

        Yields:
            Loss value in current training step.
        """

        variables = [var for var in tf.global_variables() if 'Momentum' in var.name]

        tf.variables_initializer(variables).run()

        for step in range(steps):
            batch_data = get_next_batch(self.train_dataset, step, self.batch_size)
            batch_labels = get_next_batch(self.train_labels, step, self.batch_size)
            _, l = tf.get_default_session().run((self.optimizer, self.loss), feed_dict={
                                                                              self.tf_train_dataset: batch_data,
                                                                              self.tf_train_labels: batch_labels})
            yield l

    @staticmethod
    def l2_norm(weights):
        weights_flat = []
        for weight in weights:
            weights_flat = tf.concat_v2([weights_flat, tf.reshape(weight, [-1])], axis=0)
        return tf.reduce_mean(tf.square(weights_flat))
