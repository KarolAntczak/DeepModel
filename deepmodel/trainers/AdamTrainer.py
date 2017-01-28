import tensorflow as tf
from deepmodel.trainers.Batch import *


class AdamTrainer:

    def __init__(self, model, train_dataset, train_labels, batch_size=128):
        """Inits trainer.

        Args:
            model: model to train
            train_dataset: Train dataset.
            train_labels: Train dataset labels.
            batch_size: number of examples from dataset used during each step.
        """
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, train_dataset.shape[1]))
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, train_labels.shape[1]))
        tr_train_predicted = model.predict(self.tf_train_dataset)
        self.loss = tf.reduce_mean(tf.square(tf.sub(self.tf_train_labels, tr_train_predicted)))

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.model = model
        self.batch_size = batch_size

    def train(self, steps=1000):
        """Train model.

        Args:
            steps: Maximum number of training steps.

        Yields:
            Loss value in current training step.
        """

        variables = [var for var in tf.global_variables() if 'Adam' or 'beta1_power' in var.name]

        tf.variables_initializer(variables).run()

        for step in range(steps):
            batch_data = get_next_batch(self.train_dataset, step, self.batch_size)
            batch_labels = get_next_batch(self.train_labels, step, self.batch_size)
            _, l = tf.get_default_session().run((self.optimizer, self.loss), feed_dict={
                                                                              self.tf_train_dataset: batch_data,
                                                                              self.tf_train_labels: batch_labels})
            yield l
