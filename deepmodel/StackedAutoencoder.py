from deepmodel import Perceptron


class StackedAutoencoder:
    """ Stacked autoencoding network
    """

    def __init__(self):
        self.layers = []

    def weights_norm(self):
        norm = 0
        for layer in self.layers:
            norm += layer.weights_norm()
        return norm

    def add_layer(self, layer):
        self.layers.append(layer)

    def encode(self, x):
        encoded = x
        for autoencoder in self.layers:
            encoded = autoencoder.encode(encoded)

        return encoded

    def recover(self, x):
        encoded = x
        for autoencoder in self.layers[:-1]:
            encoded = autoencoder.encode(encoded)

        return self.layers[-1].recover(encoded)

    def train(self, train_dataset, steps=1000, batch_size=256, learning_rate=0.04, momentum=0.99,l2=0):
        for l in Perceptron.train(self, train_dataset, train_dataset, steps, batch_size, learning_rate, momentum, l2):
            yield l

    def print(self):
        for i, autoencoder in enumerate(self.layers):
            print("   Layer %i: h1%s b1%s" % (i, autoencoder.h1.eval().flatten(), autoencoder.b1.eval().flatten()))
