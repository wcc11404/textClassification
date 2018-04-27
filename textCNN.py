import tensorflow as tf
import os
import time
import datetime
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
# from zhihu_sample_dataset import dataset
from zhihu_dataset import dataset
import math

#申请权重向量
def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))

#申请bias向量
def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

class TextCNN(object):
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.data=dataset()             #数据集

        self.sequence_length,self.num_classes,self.vocab_size,self.embedding_size=self.data.get_param()

        self.filter_sizes=[3,4,5]       #cnn    filter大小,kernel为[filter,embedding_size]
        self.num_filters=128            #cnn    filter数量
        self.l2_reg_lambda = 0.0001     #l2范数的学习率

        self.num_checkpoints=100       #存模型的频率
        self.num_test=2000
        self.dropout=0.5               #dropout比例
        self.learningrate=1e-5          #学习率
        # self.learning_rate = tf.Variable(1e-4, trainable=False, name="learning_rate")  # ADD learning_rate
        self.batch_size=64
        self.Model_dir = "./TextCNN"  # 模型参数默认保存位置

        self.test= tf.placeholder(tf.bool)
        ###########
        self.iter = tf.placeholder(tf.int32)  # training iteration
        self.tst = tf.placeholder(tf.bool)
        self.b1 = tf.Variable(tf.ones([self.embedding_size]) / 10)
        # print(self.b1)
        #################

        self.buildModel()

    # check:https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.1_batchnorm_five_layers_relu.py#L89
    def batchnorm(self, Ylogits, is_test, iteration, offset,convolutional=False):
        # adding the iteration prevents from averaging across non-existing iterations
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,iteration)
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def batch_norm(self,x, train, eps=1e-05, decay=0.9, affine=True, name=None):
        from tensorflow.python.training.moving_averages import assign_moving_average
        with tf.variable_scope(name, default_name='BatchNorm2d'):
            params_shape = x.shape[-1:]
            moving_mean = tf.get_variable('mean', params_shape, initializer=tf.zeros_initializer,trainable=False)  # moving两个变量永久保存，供测试时候使用
            moving_variance = tf.get_variable('variance', params_shape, initializer=tf.ones_initializer,trainable=False)

            def mean_var_with_update():

                mean, variance = tf.nn.moments(x, [i for i in range(len(x.shape)-1)], name='moments')  # 求当前batch的平均值和标准差
                # control_dependencies通常与with一起用，当with代码块中语句为tensorflow op操作时，确保先执行con里边的语句，如果不是op操作，只是单纯的tensor取值，则不执行
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),assign_moving_average(moving_variance, variance,decay)]):  # 求滑动平均值，赋值给moving两个变量
                    return tf.identity(mean), tf.identity(variance)  # 将当前batch的平均值和标准差返回，当需要y=x这种赋值操作时，使用y=tf.identity(x)

            mean, variance = tf.cond(train, mean_var_with_update,lambda: (moving_mean, moving_variance))  # if else ，train为真时，返回第二个参数，假时返回第三个参数
            if affine:
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)  #
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
                tf.contrib
            return x

    def buildModel(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        init_value=tf.random_normal_initializer(stddev=0.1)
        # init_value=tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0)
        # init_value=self.data.vector
        with tf.name_scope("embedding"):
            # W = tf.get_variable(initializer=init_value,shape=[self.vocab_size, self.embedding_size],name="Embedding")  #随机产生vocab_size个embedding_size维的字典
            W = tf.Variable(self.data.load_vocabulary())
            embedded_chars = tf.nn.embedding_lookup(W, self.input_x)                                        #通过input_x查找对应字典的随机数

        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)#将[none,56,128]后边加一维，变成[none,56,128,1]

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W%d" % i)
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b%d" % i)

                conv = tf.nn.conv2d(embedded_chars_expanded, W,strides=[1, 1, 1, 1],padding="VALID")
                # conv, self.update_ema = self.batchnorm(conv, self.tst, self.iter, self.b1)
                conv = self.batch_norm(conv,self.test)

                h = tf.nn.relu(tf.nn.bias_add(conv, b))

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h,ksize=[1, self.sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool%d" % i)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",shape=[num_filters_total, self.num_classes],initializer=tf.random_normal_initializer(stddev=0.1))#tf.contrib.layers.xavier_initializer()
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")               #全链接
            self.out = tf.nn.sigmoid(self.scores)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            l2_loss =tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])#
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses = tf.reduce_sum(losses,axis=1)
            losses=tf.reduce_mean(losses)
            self.loss=losses + self.l2_reg_lambda * l2_loss


        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # self.decay_steps = 10000
        # self.decay_rate = 0.562
        # learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
        #                                            self.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(self.learningrate)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        self.buildSummaries(grads_and_vars)

        # 初始化变量
        self.sess.run(tf.initialize_all_variables())

    def buildSummaries(self,grads_and_vars):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 创建记录文件
        #timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        # acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # 训练集的记录
        self.train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])    #归并记录
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)   #类似打开文件操作

        # 校验集的记录
        self.dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

        # # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        # checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        # self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        #
        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)
        #
        # self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

    def trainModel(self,num_epoch=10,last_batch=None):
        if last_batch==None:
            batches = self.data.train_batch_iter(self.batch_size, num_epoch)  # batch迭代器
        else:
            batches = self.data.train_batch_iter(self.batch_size, num_epoch,True,last_batch)  # batch迭代器

        for x_batch,y_batch,epochnum,epochmax,batchnum,batchmax in batches:                                 #通过迭代器取出batch数据
            step=self.sess.run(self.global_step)
            feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: self.dropout,self.test:False}
            _, summaries, loss = self.sess.run([self.train_op, self.train_summary_op, self.loss], feed_dict=feed_dict)
            # current_step = tf.train.global_step(self.sess, self.global_step)    #无卵用的操作，等于sess.run(global_step),得到训练次数
            self.train_summary_writer.add_summary(summaries, step)  # 对记录文件添加上边run出的记录和step数

            if(batchnum%self.num_checkpoints==0):
                print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                print('epoch:%d/%d\tbatch:%d/%d' % (epochnum, epochmax, batchnum, batchmax))
                self.saveModel(batchnum)

            if(batchnum and batchnum%self.num_test==0):
                self.testModel()



    def testModel(self):
        dev_iter = self.data.dev_batch_iter()

        eval_counter, eval_f1_score, eval_p, eval_r = 0, 0.0, 0.0, 0.0
        for x,y in dev_iter:
            step=self.sess.run(self.global_step)
            feed_dict = {self.input_x: x, self.input_y: y, self.dropout_keep_prob: 1.0,self.test:True}
            summaries, loss, out = self.sess.run([self.dev_summary_op, self.loss, self.out],feed_dict=feed_dict)

            # temp = np.around(out)
            # f1 = precision_recall_fscore_support(y_dev, temp, average='micro')
            for i in range(len(out)):
                label_list_top5 = self.get_label_using_logits(out[i])
                eval_y_short = self.get_target_label_short(y[i])
                f1_score, p, r = self.compute_f1_score(list(label_list_top5), eval_y_short)
                eval_counter, eval_f1_score, eval_p, eval_r = eval_counter + 1, eval_f1_score + f1_score, eval_p + p, eval_r + r

        # self.dev_summary_writer.add_summary(summaries, step)

        # print(f1)
        print("Evaluation: step {}, loss {:g}, precision {:g}, recall {:g}, f1 {:g}".format(step, loss, eval_p / float(eval_counter),
                                                                                eval_r / float(eval_counter), eval_f1_score / float(eval_counter)))
        print("")

    def Looptrain(self):
            num=self.loadModel()

            self.trainModel(last_batch=num)

    def saveModel(self, batchnum=0, dir=None):
        if (dir == None):
            dir = self.Model_dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        tf.train.Saver().save(self.sess, dir + "/model.ckpt")
        f=open(dir+'/info.txt','w')
        f.write('%d\n' % batchnum)
        f.close()

    def loadModel(self,dir=None):
        if (dir == None):
            dir = self.Model_dir
        tf.train.Saver().restore(self.sess, dir+"/model.ckpt")
        f = open(dir + '/info.txt', 'r')
        line=f.readline()
        f.close()
        return int(line)

    def compute_f1_score(self,label_list_top5, eval_y_short):
        num_correct_label = 0
        p_5=0.0
        for i,label_predict in enumerate(label_list_top5):
            if label_predict in eval_y_short:
                num_correct_label = num_correct_label + 1
                p_5+=1.0/math.log(2+i)

        all_real_labels = len(eval_y_short)
        r_5 = num_correct_label / all_real_labels

        f1_score = p_5 * r_5 / (p_5 + r_5 + 0.000001)

        return f1_score, p_5, r_5

    def get_target_label_short(self,eval_y):
        eval_y_short = []  # will be like:[22,642,1391]
        for index, label in enumerate(eval_y):
            if label > 0:
                eval_y_short.append(index)
        return eval_y_short

    # get top5 predicted labels
    def get_label_using_logits(self,logits, top_number=5):
        index_list = np.argsort(logits)[-top_number:]
        index_list = index_list[::-1]
        return index_list

def main():
    cnn=TextCNN()
    cnn.Looptrain()

if __name__ == '__main__':
    main()