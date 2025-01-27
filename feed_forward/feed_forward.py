import math
import numpy as np
import random
from typing import List


class FFNN:
    def __init__(self):
        self.weights = []
        self.bias = []
        self.node_values = []  # Sigmoid activated values
        self.pre_activation_values = []  # Values before sigmoid activation

    def initialize_empty_weights(self, layers_spec: List[int]):
        new_weights = []
        for i in range(1, len(layers_spec)):
            inLayer = layers_spec[i - 1]
            outLayer = layers_spec[i]

            if inLayer < 1 or outLayer < 1:
                print("Error - initialize_empty_weights: layer must have positive specification")
                return
            new_weights.append(np.zeros((outLayer, inLayer)))
        self.weights = new_weights

    def initialize_random_weights(self, layers_spec: List[int]):
        new_weights = []
        new_bias = []
        for i in range(1, len(layers_spec)):
            inLayer = layers_spec[i - 1]
            outLayer = layers_spec[i]

            if inLayer < 1 or outLayer < 1:
                print("Error - initialize_random_weights: layer must have positive specification")
                return

            stdev = math.sqrt(2 / (inLayer + outLayer))
            new_weights.append(np.random.normal(loc=0, scale=stdev, size=(outLayer, inLayer)))
            new_bias.append(np.random.normal(loc=0, scale=stdev, size=(outLayer, 1)))
        
        self.weights = new_weights
        self.bias = new_bias


def create_ffnn(layers_spec: List[int]) -> FFNN:
    if len(layers_spec) == 0:
        print("Error - create_ffn: layers specification is empty")
        return FFNN()
    
    nn = FFNN()
    nn.initialize_random_weights(layers_spec)
    return nn


def sigmoid_function(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_function_derivative(x: np.ndarray) -> np.ndarray:
    return np.multiply(x, 1 - x)


def forward_propogate(nn: FFNN, input: np.ndarray):
    try:
        nn.node_values.clear()
        nn.pre_activation_values.clear()
        curr = input.reshape(-1, 1)  # Ensure column vector format
        nn.node_values.append(curr)
        nn.pre_activation_values.append(curr)  # Input layer has no activation

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

        error_func_deriv = nn.node_values[layer_ptr] - true_output.reshape(-1, 1)
        # Calculate sigmoid derivative using pre-activation values
        sigmoid_deriv = sigmoid_function_derivative(sigmoid_function(nn.pre_activation_values[layer_ptr]))
        curr_sigma = np.multiply(error_func_deriv, sigmoid_deriv)

        while layer_ptr > 0:
            gradient = np.matmul(curr_sigma, nn.node_values[layer_ptr - 1].T)
            error_weight_gradients.insert(0, gradient)
            error_bias_gradients.insert(0, np.sum(curr_sigma, axis=1, keepdims=True))

            if layer_ptr > 1:
                # Calculate sigmoid derivative using pre-activation values
                sigmoid_deriv = sigmoid_function_derivative(sigmoid_function(nn.pre_activation_values[layer_ptr - 1]))
                curr_sigma = np.multiply(np.matmul(nn.weights[layer_ptr - 1].T, curr_sigma), sigmoid_deriv)
            
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


def train(nn: FFNN, learning_rate: float, epochs: int, training_input: np.ndarray, training_output: np.ndarray):
    if epochs < 1:
        print("Error - train: Must have a positive integer for number of epochs")
        return
    if len(training_input) != len(training_output):
        print("Error - train: training input and output have different sizes")
        return
    
    training_set_copy = list(zip(training_input, training_output))
    for _ in range(epochs):
        random.shuffle(training_set_copy)
        for input_data, output_data in training_set_copy:
            forward_propogate(nn, input_data.reshape(-1, 1))
            back_propogate(nn, learning_rate, output_data.reshape(-1, 1))


def process(nn: FFNN, input: np.ndarray) -> np.ndarray:
    return forward_propogate(nn, input.reshape(-1, 1))