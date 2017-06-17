import numpy as np


class Model(object):
    weights = [np.random.randn(8, 16), np.random.randn(16, 16), np.random.randn(16, 2)]
    def predict(self, inp):
        out = inp
        for layer in self.weights:
            out = np.dot(out, layer)
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
