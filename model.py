import numpy as np


class Model(object):

    def __init__(self):
        self.weights = [np.random.randn(8, 500), np.random.randn(500, 2), np.random.randn(1, 500)]

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        out = np.dot(out, self.weights[0]) + self.weights[-1]
        out = np.dot(out, self.weights[1])
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
