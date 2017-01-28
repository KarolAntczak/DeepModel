class Multilayer:
    """ Multi-layered network
    """

    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = layers

    def __str__(self):
        string = ""
        for i, layer in enumerate(self.layers):
            string += "   Layer %i: h1%s b1%s\n" % (i, layer.h1.eval().flatten(), layer.b1.eval().flatten())
        return string

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights

    def predict(self, x):
        predicted = x
        for layer in self.layers:
            predicted = layer.predict(predicted)
        return predicted
