import tensorflow as tf
import numpy as np
from keras.datasets.cifar10 import load_data


if __name__ == "__main__" :
#==========================================================
# 1. CIFAR-10 데이터 다운로드 및 데이터 로드
#==========================================================
    (x_train, y_train), (x_test, y_test) = load_data()

    #one hot encoding
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]

    y_train_onehot = np.squeeze(y_train_onehot) # (50000,1,10) -> (50000,10)
    y_test_onehot = np.squeeze(y_test_onehot) # (10000,1,10) -> (10000,10)

#==========================================================
# 2. 인풋, 레이블을 입력받기 위한 플레이스홀더 정의
#==========================================================
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 10])


#==========================================================
# 3. 모델 정의
#==========================================================
    x_image = x
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[3,3,3, 32], stddev=0.01))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.sigmoid(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = tf.Variable(tf.truncated_normal(shape=[3,3,32, 64], stddev=0.01))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.sigmoid(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_conv2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 64, 384], stddev=0.01))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=0.01))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    y_pred = tf.nn.softmax(logits)


#==========================================================
# 4. 비용함수 정의
#==========================================================
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 정확도를 계산하는 연산.
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#==========================================================
# 5. 학습
#==========================================================
    with tf.Session() as sess:
    #----------------------------------
    # 5.1 세션, 변수 초기화
    #----------------------------------
        sess.run(tf.global_variables_initializer())

    #---------------------------------
    # 5.2 Training loop
    #---------------------------------
        total_epoch = 300
        for e in range(total_epoch):
        #..........................
        # 5.2.1 학습
        #.........................
            total_size = x_train.shape[0]
            batch_size = 128

            loss_list = []
            train_accuracy_list = []
            for i in range(int(total_size / batch_size)):
                #== batch load
                batch_x = x_train[i*batch_size:(i+1)*batch_size]
                batch_y = y_train_onehot[i*batch_size:(i+1)*batch_size]

                #== train
                sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

                #== logging
                train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
                loss_print = loss.eval(feed_dict={x: batch_x, y: batch_y})
                train_accuracy_list.append(train_accuracy)
                loss_list.append(loss_print)

            print("반복(Epoch):", e, "트레이닝 데이터 정확도:", np.mean(train_accuracy_list), "손실 함수(loss):",np.mean(loss_list))

        # ..........................
        # 5.2.2 평가
        # .........................
        # 매epoch 마다 test 데이터셋에 대한 정확도와 loss를 출력.
            test_total_size = x_test.shape[0]
            test_batch_size = 128

            test_accuracy_list = []
            for i in range(int(test_total_size / test_batch_size)):
                #== test batch load
                test_batch_x = x_test[i*test_batch_size:(i+1)*test_batch_size]
                test_batch_y = y_test_onehot[i*test_batch_size:(i+1)*test_batch_size]

                #== logging
                test_accuracy = accuracy.eval(feed_dict={x: test_batch_x, y: test_batch_y})
                test_accuracy_list.append(test_accuracy)
            print("테스트 데이터 정확도:",np.mean(test_accuracy_list))
            print()
