import tensorflow as tf
import numpy as np
from dataset_loader_noweahter import Dataset_loader
from model_rnn_regression import Model

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
        self.numClasses = 1
        self.model = Model(input_dim = self.dataset.duration*6,output_dim = self.numClasses,duration=self.dataset.duration)
        self.Y = tf.placeholder(tf.float32, shape=[None,1])
        self.lr = tf.placeholder(tf.float32)




    def train(self):
    # ===============================================
    # 1. declare writers for tensorboard
    # ===============================================




    # ==========================================================
    # 2. define cost function and train step
    # ==========================================================
        with tf.name_scope("l2_loss") as scope:
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.0005
        with tf.name_scope("tendancy_loss") as scope:
            y_t = tf.slice(self.Y,[0,0],[self.batchSize-1,1]) - tf.slice(self.Y,[1,0],[self.batchSize-1,1])
            ypred_t = tf.slice(self.model.logits, [0, 0], [self.batchSize - 1, 1]) - tf.slice(self.model.logits, [1, 0],
                                                                               [self.batchSize - 1, 1])

            tloss = 0.05*tf.reduce_mean(tf.square(y_t-ypred_t))

        with tf.name_scope("trainer") as scope:
            loss = tf.reduce_mean(tf.losses.mean_squared_error(labels= self.Y,predictions=self.model.logits))+0*tloss#+lossL2
            #loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.Y- self.model.logits_relu),reduction_indices=1))+lossL2
            train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

        with tf.name_scope("evaluation") as scope:
            correct_prediction = tf.reduce_sum((self.Y)-(self.model.logits),reduction_indices=1)
            correct_prediction_square = tf.reduce_sum(tf.square((self.Y) - (self.model.logits)), reduction_indices=1)

            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # ===============================================
    # 3. train
    # ===============================================
    # ===================== main =======================
    # Option set config
    # ==================================================
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.10)
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #===============================================
        # 3.1 session init and graph load
        #===============================================
            sess.run(tf.global_variables_initializer())

        # ===============================================
        # 1. declare writers for tensorboard
        # ==============================================
            accuracy_hist = tf.placeholder(tf.float32)
            accuracy_hist_summary = tf.summary.scalar('acc_hist', accuracy_hist)

            accuracy_s_hist = tf.placeholder(tf.float32)
            accuracy_s_hist_summary = tf.summary.scalar('acc_square_hist', accuracy_s_hist)

            loss_hist_summary = tf.summary.scalar('training_loss_hist', loss)
            merged = tf.summary.merge_all()
            writer_acc_loss = tf.summary.FileWriter("./board_rrnw_tl/acc_loss", sess.graph)

            prediction_hist = tf.placeholder(tf.float32)
            prediction_hist_summary = tf.summary.scalar('pred_hist', prediction_hist)
            prediction_hist_merged = tf.summary.merge([prediction_hist_summary])

            writer_pred = tf.summary.FileWriter("./board_rrnw_tl/pred", sess.graph)
            writer_pred_label = tf.summary.FileWriter("./board_rrnw_tl/pred_label", sess.graph)


        #===============================================
        # 3.2. train loop
        #===============================================
            for e in range(self.totalEpoch):
            # ..........................
            # 3.2.1 학습
            # .........................
                p = np.random.permutation(len(self.trainset))
                self.trainset = np.array(self.trainset)[p]

                #p = np.random.permutation(len(self.testset))
                #self.testset = np.array(self.testset)[p]

                loss_list = []
                tloss_list = []
                train_accuracy_list = []
                for i in range(int(len(self.trainset) / self.batchSize)):
                    # == batch load
                    batch_x = np.array([np.squeeze(self.trainset[b].get2DShapeInput(),axis=-1)  for b in range(i*self.batchSize,(i+1)*self.batchSize)])
                    batch_y = np.array([self.trainset[b].pv_label  for b in range(i*self.batchSize,(i+1)*self.batchSize)]).reshape(self.batchSize,-1)

                    # == train
                    #if (e < 5000):lr_value = 0.0001
                    #elif (5000 < e and e < 10000):           lr_value = 0.00001
                    lr_value = 0.001 * pow((1 - min(e / self.totalEpoch, 1)), 0.9)

                    sess.run(train_step, feed_dict={self.model.X: batch_x, self.Y: batch_y,  self.lr: lr_value})


                    # == logging
                    loss_print,tloss_print = sess.run([loss,tloss],feed_dict={self.model.X: batch_x, self.Y: batch_y, })
                    loss_list.append(loss_print)
                    tloss_list.append(tloss_print)
                print("반복(Epoch):", e, "트레이닝 데이터 정확도:", np.mean(train_accuracy_list), "손실 함수(loss):",
                      np.mean(loss_list),"tloss:",np.mean(tloss_list))

            # ..........................
            # 3.2.2 평가
            # .........................
            # 매epoch 마다 test 데이터셋에 대한 정확도와 loss를 출력.
                test_accuracy_list = []
                test_accuracy_s_list = []

                loss_list = []
                tloss_list = []
                histloginterval = 2000
                if (e % histloginterval == 0):
                    writer_pred = tf.summary.FileWriter("./board_rrnw_tl/pred" + str(e), sess.graph)
                    writer_pred_label = tf.summary.FileWriter("./board_rrnw_tl/pred_label" + str(e), sess.graph)

                for i in range(int(len(self.testset) / self.batchSize_test)):
                    # == test batch load
                    test_batch_x = np.array([np.squeeze(self.testset[b].get2DShapeInput(),axis=-1)  for b in range(i*self.batchSize_test,(i+1)*self.batchSize_test)])
                    test_batch_y = np.array([self.testset[b].pv_label  for b in range(i*self.batchSize_test,(i+1)*self.batchSize_test)]).reshape(self.batchSize_test,-1)

                    # == logging
                    test_accuracy, test_accuracy_square = sess.run([correct_prediction, correct_prediction_square],
                                                                   feed_dict={self.model.X: test_batch_x,
                                                                              self.Y: test_batch_y})
                    test_accuracy_list.append(test_accuracy)
                    test_accuracy_s_list.append(test_accuracy_square)

                    loss_print, tloss_print = sess.run([loss, tloss],
                                                       feed_dict={self.model.X: test_batch_x,
                                                                  self.Y: test_batch_y,})
                    loss_list.append(loss_print)
                    tloss_list.append(tloss_print)



                    if (e % histloginterval == 0):
                        for o in range(len(test_batch_y)):
                            output = self.model.logits_relu.eval(
                                feed_dict={self.model.X: test_batch_x, self.Y: test_batch_y})
                            output_scalar = (output[o])
                            print("정답: ", (test_batch_y[o]), "출력: ", output_scalar, "step: ",
                                  o + i * self.batchSize_test)
                            writer_pred.add_summary(
                                prediction_hist_merged.eval(feed_dict={prediction_hist: float(output_scalar)}),
                                global_step=o + i * self.batchSize_test)
                            # writer_pred.flush()
                            writer_pred_label.add_summary(
                                prediction_hist_merged.eval(feed_dict={prediction_hist: float(test_batch_y[o])}),
                                global_step=o + i * self.batchSize_test)
                            # writer_pred_label.flush()

                summary = merged.eval(
                    feed_dict={self.model.X: batch_x, self.Y: batch_y, accuracy_hist: np.mean(test_accuracy_list),
                               accuracy_s_hist: np.mean(test_accuracy_s_list)})
                writer_acc_loss.add_summary(summary, global_step=e)
                writer_acc_loss.flush()
                print("테스트 데이터 정확도:", np.mean(test_accuracy_list), "손실 함수(loss):", np.mean(loss_list),"tloss:", np.mean(tloss_list))
                print()


def test():
    t = Trainer()
    t.train()
if __name__ == "__main__":
    test()
    pass
