from deepmodel import Perceptron


class Multilayer:
    """ Multi-layered network
    """

    def __init__(self):
        self.layers = []

    def weights_norm(self):
        norm = 0
        for layer in self.layers:
            norm += layer.weights_norm()
        return norm

    def predict(self, x):
        predicted = x
        for layer in self.layers:
            predicted = layer.predict(predicted)
        return predicted

    def train(self, train_dataset, train_labels, steps=1000, batch_size=256, learning_rate=0.04, momentum=0.99,l2=0):
        for l in Perceptron.train(self, train_dataset, train_labels, steps, batch_size, learning_rate, momentum, l2):
            yield l