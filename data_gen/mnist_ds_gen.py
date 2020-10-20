"""
Generates the MNIST dataset for the downstream task of digit classification
Requires:
torchvision==0.4.0 (to be compatible with torch==1.2.0)
"""
#MNIST prep
from torchvision import datasets
import torch
import random
import numpy as np
import sys

'''
Transforms an MNIST image into a 64x64 
containing the digit and the rest is 
filled with black pixels. The position of 
the digit is random
'''
def digit_to_64x64(digit_array):
    padded_digit = np.zeros((64, 64))
    x_pos = random.randint(0, 36)
    y_pos = random.randint(0, 36)
    padded_digit[x_pos:x_pos+28, y_pos:y_pos+28] = digit_array
    return padded_digit

'''
Transforms the entire MNIST using digit_to_64x64
includes the channel dimension
'''
def mnist_to_64x64(train_set_array, test_set_array, seed=1):
    random.seed(seed)
    num_train_samples = train_set_array.shape[0]
    padded_train_set_array = np.zeros((num_train_samples, 1, 64, 64))
    for i in range(num_train_samples):
        padded_train_set_array[i, 0] = digit_to_64x64(train_set_array[i])
    
    num_test_samples = test_set_array.shape[0]
    padded_test_set_array = np.zeros((num_test_samples, 1, 64, 64))
    for i in range(num_test_samples):
        padded_test_set_array[i, 0] = digit_to_64x64(test_set_array[i])
    
    return padded_train_set_array, padded_test_set_array
  
'''
Main:
Args:
sys.argv[1] = path to where MNIST dataset will be put
sys.argv[2] = path to where the padded MNIST npz file will be put
'''
mnist_path = sys.argv[1]
padded_mnist_path = sys.argv[2]

dataset1 = datasets.MNIST(mnist_path, train=True, download=True)
dataset2 = datasets.MNIST(mnist_path, train=False)

train_set_array = dataset1.data.numpy()
train_targets_array = dataset1.targets.numpy()
test_set_array = dataset2.data.numpy()
test_targets_array = dataset2.targets.numpy()

padded_train_set_array, padded_test_set_array = mnist_to_64x64(train_set_array, test_set_array, seed=1)
np.savez(padded_mnist_path, padded_train_set_array, train_targets_array,  padded_test_set_array, test_targets_array)

'''
arr_0 = padded_train_set_array
arr_1 = train_targets_array
arr_2 = padded_test_set_array
arr_3 = test_targets_array
'''