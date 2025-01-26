import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath, num_images):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())    
                
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images[:num_images], labels[:num_images]
    
    def convert_to_result_arr(self, input: int) -> np.ndarray:
        if input < 0 or input > 9:
            return np.array([])
        res = np.zeros(10)
        res[input] = 1
        return res
        
            
    def load_data(self, num_images):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath, num_images)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath, num_images)
        
        # Make images in expected form
        x_train = np.array([np.matrix(np.concatenate(x)).T / 255 for x in x_train])
        x_test = np.array([np.matrix(np.concatenate(x)).T / 255 for x in x_test])
        
        # Make output in expected form
        y_train = np.array([np.matrix(self.convert_to_result_arr(i)).T for i in y_train])
        y_test = np.array([np.matrix(self.convert_to_result_arr(i)).T for i in y_test])
        
        return (x_train, y_train),(x_test, y_test)