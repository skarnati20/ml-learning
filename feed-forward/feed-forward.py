import math
import numpy as np
import random
from typing import List


class FFNN:
    def __init__(self):
        self.weights = []
        self.bias = []
        self.node_values = []

    def initialize_empty_weights(self, layers_spec: List[int]):
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
        # Glorot initialization
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

            arr = np.zeros((outLayer, inLayer))
            stdev = math.sqrt(2 / (inLayer + outLayer))
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    arr[i, j] = np.random.normal(loc=0, scale=stdev)
            new_weights.append(arr)
            new_bias.append(np.random.normal(loc=0, scale=stdev))
        self.weights = new_weights
        self.bias = new_bias


def create_ffnn(layers_spec: List[int]) -> FFNN:
    if len(layers_spec) == 0:
        print("Error - create_ffn: layers specification is empty")
        return FFNN()
    nn = FFNN()
    nn.initialize_random_weights(layers_spec)
    return nn


def sigmoid_function(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_function_derivative(input: np.ndarray) -> np.ndarray:
    return np.multiply(input, 1 - input)


def forward_propogate(nn: FFNN, input: np.ndarray):
    try:
        nn.node_values.clear()
        curr = input
        nn.node_values.append(curr)
        for i in range(len(nn.weights)):
            weights = nn.weights[i]
            bias = nn.bias[i]
            weights_product = np.matmul(weights, curr)
            weights_product += bias
            for i in range(weights_product.shape[0]):
                weights_product[i, 0] = sigmoid_function(weights_product[i, 0])
            curr = weights_product
            nn.node_values.append(curr)
        return curr
    except Exception as error:
        print("Error - forward_propogate: Error multiplying matrices")
        print(error)


def back_propogate(nn: FFNN, learning_rate: float, true_output: np.ndarray):
    if len(nn.node_values) == 0:
        print("Error - back_propogate: No node layers found")
        return
    if len(nn.node_values) - len(nn.weights) != 1:
        print("Error - back_propogate: Imporpert number of layers or weights.")
        return
    if len(nn.node_values) == 1:
        print("back_propogate: Only one node layer found. No weights to change.")
        return

    # Find error gradients
    error_weight_gradients = []
    try:
        layer_ptr = len(nn.node_values) - 1

        error_func_deriv = nn.node_values[layer_ptr] - true_output
        last_layer_input = np.matmul(
            nn.weights[layer_ptr - 1], nn.node_values[layer_ptr - 1]
        )
        sigmoid_deriv = sigmoid_function_derivative(last_layer_input)
        curr_sigma = np.multiply(error_func_deriv, sigmoid_deriv)

        while layer_ptr > 0:
            gradient = np.matmul(curr_sigma, nn.node_values[layer_ptr - 1].T)
            error_weight_gradients.insert(0, gradient)

            if layer_ptr > 1:
                last_layer_input = np.matmul(
                    nn.weights[layer_ptr - 2], nn.node_values[layer_ptr - 2]
                )
                sigmoid_deriv = sigmoid_function_derivative(last_layer_input)
                curr_sigma = np.multiply(
                    np.matmul(nn.weights[layer_ptr - 1].T, curr_sigma), sigmoid_deriv
                )
            layer_ptr -= 1
    except Exception as error:
        print("Error - backward_propogate: Error while generating gradients")
        print(error)

    # Update weights based of gradient and learning rate
    try:
        for i in range(len(error_weight_gradients)):
            nn.weights[i] -= learning_rate * error_weight_gradients[i]
    except Exception as error:
        print("Error - backward_propogate: Error updating weights")
        print(error)


def train(nn: FFNN, learning_rate: float, epochs: int, training_set: List[tuple[np.ndarray, np.ndarray]]):
    if epochs < 1:
        print("Error - train: Must have a positive integer for number of epochs")
        return

    training_set_copy = training_set.copy()
    for _ in range(epochs):
        training_set_copy.shuffle()
        for (input, output) in training_set_copy:
            forward_propogate(nn, input)
            back_propogate(nn, learning_rate, output)
            
def process(nn: FFNN, input: np.ndarray) -> np.ndarray:
    return forward_propogate(nn, input)
