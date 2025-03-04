from typing import List
import numpy as np


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class Neuron:
    def __init__(self, weights: List[int], bias: int):
        self.weights = weights
        self.bias = bias

    def feed_forward(self, inputs: List[int]) -> float:
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
