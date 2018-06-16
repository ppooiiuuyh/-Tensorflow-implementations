import tensorflow as tf
import numpy as np
import math

if __name__ == "__main__":

    with tf.Session() as sess :

        a = 3
        b = tf.Variable(3)
        print(b.eval())