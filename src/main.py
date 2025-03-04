import numpy as np
from neuron import Neuron


def main():
    weights = np.array([0, 1])
    bias = 4
    n = Neuron(weights, bias)

    x = np.array([2, 3])
    print(n.feed_forward(x))


if __name__ == "__main__":
    main()
