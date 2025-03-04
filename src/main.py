import numpy as np
from neuron import Neuron
from neural_network import OwnNeuralNetwork


def main():
    # basic_neuron()
    network = OwnNeuralNetwork()
    x = np.array([2, 3])
    print(network.feed_forward(x))


def basic_neuron():
    weights = np.array([0, 1])
    bias = 4
    n = Neuron(weights, bias)

    x = np.array([2, 3])
    print(n.feed_forward(x))


if __name__ == "__main__":
    main()
