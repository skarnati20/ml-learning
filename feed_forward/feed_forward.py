import math
import numpy as np
import random
from typing import List


class FFNN:
    """
    A feed-forward neural network implementation that supports multiple layers
    with sigmoid activation functions. The network handles weight initialization,
    forward propagation, and backpropagation for training.

    Attributes
    __________
    weights : List[np.ndarray]
        List of weight matrices for each layer
    bias : List[np.ndarray]
        List of bias vectors for each layer
    node_values : List[np.ndarray]
        List of activation values for each layer after sigmoid
    pre_activation_values : List[np.ndarray]
        List of values for each layer before sigmoid activation
    """

    def __init__(self):
        """
        Initializes an empty feed-forward neural network with no layers or weights.
        """
        self.weights = []
        self.bias = []
        self.node_values = []
        self.pre_activation_values = []

    def initialize_empty_weights(self, layers_spec: List[int]):
        """
        Initializes the network with zero weights based on the layer specifications.

        Parameters
        __________
        layers_spec : List[int]
            List of integers specifying the number of nodes in each layer
        """
        new_weights = []
        for i in range(1, len(layers_spec)):
            inLayer = layers_spec[i - 1]
            outLayer = layers_spec[i]

            if inLayer < 1 or outLayer < 1:
                print(
                    "Error - initialize_empty_weights: layer must have positive specification"
                )
                return
            new_weights.append(np.zeros((outLayer, inLayer)))
        self.weights = new_weights

    def initialize_random_weights(self, layers_spec: List[int]):
        """
        Initializes the network with random weights and biases using He initialization.
        Weights are drawn from a normal distribution with scale based on layer sizes.

        Parameters
        __________
        layers_spec : List[int]
            List of integers specifying the number of nodes in each layer
        """
        new_weights = []
        new_bias = []
        for i in range(1, len(layers_spec)):
            inLayer = layers_spec[i - 1]
            outLayer = layers_spec[i]

            if inLayer < 1 or outLayer < 1:
                print(
                    "Error - initialize_random_weights: layer must have positive specification"
                )
                return

            stdev = math.sqrt(2 / (inLayer + outLayer))
            new_weights.append(
                np.random.normal(loc=0, scale=stdev, size=(outLayer, inLayer))
            )
            new_bias.append(np.random.normal(loc=0, scale=stdev, size=(outLayer, 1)))

        self.weights = new_weights
        self.bias = new_bias


def create_ffnn(layers_spec: List[int]) -> FFNN:
    """
    Creates a new feed-forward neural network with random weights based on
    the specified layer configuration.

    Parameters
    __________
    layers_spec : List[int]
        List of integers specifying the number of nodes in each layer

    Returns
    __________
    FFNN
        A new neural network instance with initialized weights
    """
    if len(layers_spec) == 0:
        print("Error - create_ffn: layers specification is empty")
        return FFNN()

    nn = FFNN()
    nn.initialize_random_weights(layers_spec)
    return nn


def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid activation function element-wise to the input array.

    Parameters
    __________
    x : np.ndarray
        Input array to apply sigmoid function to

    Returns
    __________
    np.ndarray
        Array with sigmoid function applied element-wise
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_function_derivative(x: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the sigmoid function element-wise.

    Parameters
    __________
    x : np.ndarray
        Input array to compute sigmoid derivative for

    Returns
    __________
    np.ndarray
        Array with sigmoid derivative computed element-wise
    """
    return np.multiply(x, 1 - x)


def forward_propogate(nn: FFNN, input: np.ndarray):
    """
    Performs forward propagation through the network, computing and storing
    both pre-activation and post-activation values for each layer.

    Parameters
    __________
    nn : FFNN
        The neural network to perform forward propagation on
    input : np.ndarray
        Input data to propagate through the network

    Returns
    __________
    np.ndarray
        The output of the final layer after activation
    """
    try:
        nn.node_values.clear()
        nn.pre_activation_values.clear()

        curr = input.reshape(-1, 1)
        nn.node_values.append(curr)
        nn.pre_activation_values.append(curr)

        for i in range(len(nn.weights)):
            weights = nn.weights[i]
            bias = nn.bias[i]
            pre_activation = np.matmul(weights, curr) + bias
            nn.pre_activation_values.append(pre_activation)
            curr = sigmoid_function(pre_activation)
            nn.node_values.append(curr)

        return curr
    except Exception as error:
        print("Error - forward_propogate: Error multiplying matrices")
        print(error)


def back_propogate(nn: FFNN, learning_rate: float, true_output: np.ndarray):
    """
    Performs backpropagation through the network to compute gradients and
    update weights and biases. Uses the stored node values from forward
    propagation to compute the gradients.

    Parameters
    __________
    nn : FFNN
        The neural network to perform backpropagation on
    learning_rate : float
        The learning rate to use for weight updates
    true_output : np.ndarray
        The target output values for computing the error
    """
    if len(nn.node_values) == 0:
        print("Error - back_propogate: No node layers found")
        return
    if len(nn.node_values) - len(nn.weights) != 1:
        print("Error - back_propogate: Improper number of layers or weights.")
        return
    if len(nn.node_values) == 1:
        print("back_propogate: Only one node layer found. No weights to change.")
        return

    error_weight_gradients = []
    error_bias_gradients = []

    try:
        layer_ptr = len(nn.node_values) - 1

        # Initialize the first set of values based on output layers
        error_func_deriv = nn.node_values[layer_ptr] - true_output.reshape(-1, 1)
        sigmoid_deriv = sigmoid_function_derivative(
            sigmoid_function(nn.pre_activation_values[layer_ptr])
        )
        curr_delta = np.multiply(error_func_deriv, sigmoid_deriv)

        while layer_ptr > 0:
            gradient = np.matmul(curr_delta, nn.node_values[layer_ptr - 1].T)
            error_weight_gradients.insert(0, gradient)
            # Sum up delta values to determine bias gradients
            error_bias_gradients.insert(0, np.sum(curr_delta, axis=1, keepdims=True))

            if layer_ptr > 1:
                # The derivative determines how much the preactivation values
                # affect the activation value
                sigmoid_deriv = sigmoid_function_derivative(
                    sigmoid_function(nn.pre_activation_values[layer_ptr - 1])
                )
                # Use the previous weights and delta to keep the chain rule
                # connection back to the original error
                curr_delta = np.multiply(
                    np.matmul(nn.weights[layer_ptr - 1].T, curr_delta), sigmoid_deriv
                )

            layer_ptr -= 1
    except Exception as error:
        print("Error - back_propogate: Error while generating gradients")
        print(error)

    try:
        for i in range(len(error_weight_gradients)):
            nn.weights[i] -= learning_rate * error_weight_gradients[i]
            nn.bias[i] -= learning_rate * error_bias_gradients[i]
    except Exception as error:
        print("Error - back_propogate: Error updating weights and biases")
        print(error)


def train(
    nn: FFNN,
    learning_rate: float,
    epochs: int,
    training_input: np.ndarray,
    training_output: np.ndarray,
):
    """
    Trains the neural network using stochastic gradient descent.
    For each epoch, shuffles the training data and performs forward
    and backward propagation for each training example.

    Parameters
    __________
    nn : FFNN
        The neural network to train
    learning_rate : float
        The learning rate to use for weight updates
    epochs : int
        Number of complete passes through the training data
    training_input : np.ndarray
        Array of training input data
    training_output : np.ndarray
        Array of corresponding target outputs
    """
    if epochs < 1:
        print("Error - train: Must have a positive integer for number of epochs")
        return
    if len(training_input) != len(training_output):
        print("Error - train: training input and output have different sizes")
        return

    training_set_copy = list(zip(training_input, training_output))
    for _ in range(epochs):
        # Shuffle the training set to make neural net more robust
        random.shuffle(training_set_copy)
        for input_data, output_data in training_set_copy:
            forward_propogate(nn, input_data.reshape(-1, 1))
            back_propogate(nn, learning_rate, output_data.reshape(-1, 1))


def process(nn: FFNN, input: np.ndarray) -> np.ndarray:
    """
    Processes a single input through the trained neural network.

    Parameters
    __________
    nn : FFNN
        The neural network to use for processing
    input : np.ndarray
        The input data to process

    Returns
    __________
    np.ndarray
        The network's output for the given input
    """
    return forward_propogate(nn, input.reshape(-1, 1))
