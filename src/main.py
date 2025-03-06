import numpy as np
from neuron import Neuron
from neural_network import OwnNeuralNetwork


def main():
    # Define dataset
    data = np.array(
        [
            [-2, -1],  # Alice
            [25, 6],  # Bob
            [17, 4],  # Charlie
            [-15, -6],  # Diana
        ]
    )
    all_y_trues = np.array(
        [
            1,  # Alice
            0,  # Bob
            0,  # Charlie
            1,  # Diana
        ]
    )

    # Train our neural network!
    network = OwnNeuralNetwork()
    network.train(data, all_y_trues)

    # Make some predictions
    emily = np.array([-7, -3])  # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches
    print("Emily: %.3f" % network.feed_forward(emily))  # 0.951 - F
    print("Frank: %.3f" % network.feed_forward(frank))  # 0.039 - M


def basic_neuron():
    weights = np.array([0, 1])
    bias = 4
    n = Neuron(weights, bias)

    x = np.array([2, 3])
    print(n.feed_forward(x))


if __name__ == "__main__":
    main()
