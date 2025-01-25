import math
import numpy as np
from typing import List

class FFNN:
    def __init__(self):
        self.weights = []
        self.bias = []

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

def forward_propogate(nn: FFNN, input: np.ndarray) -> np.ndarray:
    try:
        curr = input
        for i in range(len(nn.weights)):
            weights = nn.weights[i]
            bias = nn.bias[i]
            weights_product = np.matmul(weights, curr)
            weights_product += bias
            for i in range(weights_product.shape[0]):
                weights_product[i, 0] = sigmoid_function(weights_product[i, 0])
            curr = weights_product
        return curr
    except:
        print("Error - forward_propogate: Error multiplying matrices")
        return np.array([])

nn = create_ffnn([4, 3, 4])
print(forward_propogate(nn, np.array([[1], [2], [3], [4]])))
