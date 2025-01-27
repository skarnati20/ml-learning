import numpy as np
from tools.mnist_loader import MnistDataloader
from feed_forward.feed_forward import FFNN, create_ffnn, train, process
from os.path  import join

input_path = 'mnist'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data(10000)

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # Subtracting max for numerical stability
    return e_x / e_x.sum(axis=0)

nn = create_ffnn([784, 100, 10])
train(nn, 0.2, 4, x_train, y_train)
print(softmax(process(nn, x_test[0])))
print(y_test[0])