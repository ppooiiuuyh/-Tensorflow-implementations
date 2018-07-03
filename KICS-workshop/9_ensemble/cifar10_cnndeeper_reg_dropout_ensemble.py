import tensorflow as tf
import numpy as np
from keras.datasets.cifar10 import load_data
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.contrib.layers import xavier_initializer


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
# 2. 인풋, 아웃풋데이터를 입력받기 위한 플레이스홀더 정의
#==========================================================
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    #batch norm 을위한 train pahse 정의
    trainphase = tf.placeholder(tf.bool)
    #dropout 을 위한 keepprob 정의
    keepprob = tf.placeholder(tf.float32)

#==========================================================
# 3. 모델 정의
#==========================================================
    x_image = x


    def conv(X, in_ch, out_ch, name):
        with tf.variable_scope(name) as scope:
            W_conv = tf.get_variable(name='weights', shape=[3, 3, in_ch, out_ch], initializer=xavier_initializer_conv2d())
            h_bn = tf.layers.batch_normalization(tf.nn.conv2d(X, W_conv, strides=[1, 1, 1, 1], padding='SAME'), training =trainphase)
            h_conv = tf.nn.relu(h_bn)
        return h_conv

    with tf.variable_scope("block1") as scope:
        h_conv1 = conv(x_image,3,64,"Conv1")
        h_conv2 = conv(h_conv1,64,64,"Conv2")
        h_conv2_pool = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("block2") as scope:
        h_conv3 = conv(h_conv2_pool,64,128,"Conv3")
        h_conv4 = conv(h_conv3, 128,128 , "Conv4")
        h_conv4_pool = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("block3") as scope:
        h_conv5 = conv(h_conv4_pool, 128, 256, "Conv5")
        h_conv6 = conv(h_conv5, 256, 256, "Conv6")
        h_conv6_pool = tf.nn.max_pool(h_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("block4") as scope:
        h_conv7 = conv(h_conv6_pool, 256, 512, "Conv7")
        h_conv8 = conv(h_conv7, 512, 512, "Conv8")
        h_conv8_pool = tf.nn.max_pool(h_conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("block5") as scope:
        h_conv9 = conv(h_conv8_pool, 512, 512, "Conv9")
        h_conv10 = conv(h_conv9, 512, 512, "Conv10")
        h_conv10_pool = tf.nn.max_pool(h_conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("fclayer") as scope:
        h_conv10_pool_flat = tf.reshape(h_conv10_pool, [-1, 1 * 1 * 512])
        h_conv10_pool_flat_dropout = tf.nn.dropout(h_conv10_pool_flat , keep_prob=keepprob)
        W_fc = tf.get_variable(name='weights', shape=[512, 10], initializer=xavier_initializer())
        b_fc = tf.Variable(tf.constant(0.1, shape=[10]))
        logits = tf.matmul(h_conv10_pool_flat_dropout, W_fc) + b_fc
        y_pred = tf.nn.softmax(logits)



#==========================================================
# 4. 비용함수 정의
#==========================================================
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.0005

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)) + lossL2
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    # 정확도를 계산하는 연산.
    output_pred = tf.placeholder(tf.float32)
    correct_prediction = tf.equal(tf.argmax(output_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#==========================================================
# 5. 학습
#==========================================================
#----------------------------------
# 5.1 세션, 변수 초기화
#----------------------------------
    num_ensembles = 5
    sess_list  = [ tf.Session() for i in range(num_ensembles)]
    for i in range(num_ensembles) : sess_list[i].run(tf.global_variables_initializer())

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

        for i in range(int(total_size / batch_size)):
            #== batch load
            batch_x = x_train[i*batch_size:(i+1)*batch_size]
            batch_y = y_train_onehot[i*batch_size:(i+1)*batch_size]

            #== train
            for ens in range(num_ensembles):
                sess_list[ens].run(train_step, feed_dict={x: batch_x, y: batch_y, trainphase : True , keepprob:0.7})


    # ..........................
    # 5.2.2 평가
    # .........................
    # 매epoch 마다 test 데이터셋에 대한 정확도와 loss를 출력.
        test_total_size = x_test.shape[0]
        test_batch_size = 128

        test_accuracy_list = []
        for i in range(int(test_total_size / test_batch_size)):
            # == test batch load
            test_batch_x = x_test[i * test_batch_size:(i + 1) * test_batch_size]
            test_batch_y = y_test_onehot[i * test_batch_size:(i + 1) * test_batch_size]

            # == logging
            predictions = []
            test_accuracy = []
            for ens in range(num_ensembles):
                predictions.append(sess_list[ens].run(y_pred, feed_dict={x: test_batch_x, y: test_batch_y, trainphase : True , keepprob:0.7}))
                test_accuracy.append(sess_list[ens].run(accuracy,
                                                        feed_dict={output_pred: np.mean(predictions[0:ens + 1], axis=0),
                                                                   y: test_batch_y, trainphase : True , keepprob:0.7}))
            test_accuracy_list.append(test_accuracy)

        print("반복[epoch] : ",e)
        for ens in range(num_ensembles) :
           print("[Ens:",ens,"] 테스트 데이터 정확도:",np.mean(test_accuracy_list[ens]))
        print()
