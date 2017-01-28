from deepmodel import Multilayer


class StackedAutoencoder(Multilayer):
    """ Stacked autoencoding network
    """

    def __init__(self, layers=None):
        super(StackedAutoencoder, self).__init__(layers)

    def predict(self, x):
        encoded = x
        for autoencoder in self.layers[:-1]:
            encoded = autoencoder.encode(encoded)

        return self.layers[-1].predict(encoded)
