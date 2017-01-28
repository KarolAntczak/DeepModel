from deepmodel import Multilayer


class StackedAutoencoder(Multilayer):
    """ Stacked autoencoding network
    """

    def __init__(self, layers=None):
        super(StackedAutoencoder, self).__init__(layers)

    def encode(self, x):
        encoded = x
        for autoencoder in self.layers:
            encoded = autoencoder.encode(encoded)

        return encoded

    def predict(self, x):
        encoded = x
        for autoencoder in self.layers[:-1]:
            encoded = autoencoder.encode(encoded)

        return self.layers[-1].predict(encoded)
