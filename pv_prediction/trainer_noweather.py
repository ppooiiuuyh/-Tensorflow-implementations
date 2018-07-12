import tensorflow as tf
import numpy as np
from dataset_loader_noweahter import Dataset_loader
from model import Model

class Trainer:
    def __init__(self):
        #parameters
        self.totalEpoch = 10000
        self.batchSize = 128
        self.batchSize_test = 128

        #dataset
        self.dataset =  Dataset_loader(
                            pvdir = "./data/pv_2016_processed.csv",
                            raindir= "./data/rain_2016_processed.csv",
                            skydir="./data/sky_2016_processed.csv")
        self.trainset = self.dataset.trainset
        self.testset = self.dataset.testset


        #model tensor :
        self.numClasses = 149
        self.model = Model(input_dim = self.dataset.duration*6,output_dim = self.numClasses)
        self.Y = tf.placeholder(tf.float32, shape=[None, self.numClasses])




    def train(self):
    # ===============================================
    # 1. declare writers for tensorboard
    # ===============================================


    # ==========================================================
    # 2. define cost function and train step
    # ==========================================================
        with tf.name_scope("trainer") as scope:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.model.logits))
            train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

        with tf.name_scope("evaluation") as scope:
            correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.model.logits, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # ===============================================
    # 3. train
    # ===============================================
    # ===================== main =======================
    # Option set config
    # ==================================================
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #===============================================
        # 3.1 session init and graph load
        #===============================================
            sess.run(tf.global_variables_initializer())


        #===============================================
        # 3.2. train loop
        #===============================================
            for e in range(self.totalEpoch):
            # ..........................
            # 3.2.1 학습
            # .........................
                loss_list = []
                train_accuracy_list = []
                for i in range(int(len(self.trainset) / self.batchSize)):
                    # == batch load
                    batch_x = np.array([self.trainset[b].getLinearShapeInput()  for b in range(i*self.batchSize,(i+1)*self.batchSize)])
                    batch_y = np.array([self.trainset[b].getOnehotLabel(self.numClasses)  for b in range(i*self.batchSize,(i+1)*self.batchSize)])


                    # == train
                    sess.run(train_step, feed_dict={self.model.X: batch_x, self.Y: batch_y, self.model.trainphase: True})


                    # == logging
                    train_accuracy = accuracy.eval(feed_dict={self.model.X: batch_x, self.Y: batch_y, self.model.trainphase: True})
                    loss_print = loss.eval(feed_dict={self.model.X: batch_x, self.Y: batch_y, self.model.trainphase: True})
                    train_accuracy_list.append(train_accuracy)
                    loss_list.append(loss_print)

                print("반복(Epoch):", e, "트레이닝 데이터 정확도:", np.mean(train_accuracy_list), "손실 함수(loss):",
                      np.mean(loss_list))

            # ..........................
            # 3.2.2 평가
            # .........................
            # 매epoch 마다 test 데이터셋에 대한 정확도와 loss를 출력.
                p = np.random.permutation(len(self.trainset))
                self.trainset = np.array(self.trainset)[p]

                #p = np.random.permutation(len(self.testset))
                #self.testset = np.array(self.testset)[p]

                test_accuracy_list = []
                for i in range(int(len(self.testset) / self.batchSize_test)):
                    # == test batch load
                    test_batch_x = np.array([self.testset[b].getLinearShapeInput()  for b in range(i*self.batchSize_test,(i+1)*self.batchSize_test)])
                    test_batch_y = np.array([self.testset[b].getOnehotLabel(self.numClasses)  for b in range(i*self.batchSize_test,(i+1)*self.batchSize_test)])

                    # == logging
                    test_accuracy = accuracy.eval(
                        feed_dict={self.model.X: test_batch_x, self.Y: test_batch_y, self.model.trainphase: True})
                    test_accuracy_list.append(test_accuracy)

                    for o in range(10):
                        output = self.model.logits_softmax.eval(feed_dict={self.model.X: test_batch_x, self.Y: test_batch_y, self.model.trainphase: True})
                        output_scalar = np.argmax(output[o])
                        print( "정답: ", np.argmax(test_batch_y[o]),"출력: ",output_scalar,"손실 함수(loss):",
                      np.mean(loss_list))

                print("테스트 데이터 정확도:", np.mean(test_accuracy_list))
                print()

def test():
    t = Trainer()
    t.train()
if __name__ == "__main__":
    test()
    pass
