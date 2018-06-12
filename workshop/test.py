import tensorflow as tf
import numpy as np
from keras.datasets.cifar10 import load_data

if __name__ == "__main__":
# ==========================================================
# 1. CIFAR-10 데이터 다운로드 및 데이터 로드
# ==========================================================
    (x_train, y_train), (x_test, y_test) = load_data()

    # scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
    y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
    y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

    print(tf.one_hot(y_train,10))

# ==========================================================
# 2. 인풋, 아웃풋데이터를 입력받기 위한 플레이스홀더 정의
# ==========================================================
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 10])
