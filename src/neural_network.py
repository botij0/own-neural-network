import numpy as np
from neuron import Neuron


class OwnNeuralNetwork:
    """
    A neural network with:
      - 2 inputs
      - 1 hidden layer with 2 neurons (h1, h2)
      - 1 output layer with 1 neuron (o1)

    Each neuron has the same weights and bias:
      - weights = [0, 1]
      - bias = 0
    """

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feed_forward(self, x: np.ndarray) -> float:
        out_h1 = self.h1.feed_forward(x)
        out_h2 = self.h2.feed_forward(x)

        out_o1 = self.o1.feed_forward(np.array([out_h1, out_h2]))
        return out_o1
