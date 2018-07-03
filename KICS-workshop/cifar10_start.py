import tensorflow as tf
import numpy as np
from keras.datasets.cifar10 import load_data

from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = load_data()
plt.imshow(x_train[0])
plt.show()


'''
print("x_train.shape :", x_train.shape)
print("y_train.shape :", y_train.shape)
print(x_train[0])
'''

#========================================================
# cifar10 dataset load를 위한 라이브러리
#========================================================



