import tensorflow as tf
import tensorflow.contrib
import os
import time
import datetime
import numpy as np
# from zhihu_sample_dataset import dataset
from zhihu_dataset import dataset
import math

class TextRNN(object):
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.data=dataset()             #数据集

        self.sequence_length,self.num_classes,self.vocab_size,self.embedding_size=self.data.get_param()

        self.hidden_size=self.embedding_size
        self.is_train = tf.placeholder(tf.bool) #batchnormalization用

        self.decay_steps = 3000
        self.decay_rate = 0.5  # 62
        self.l2_reg_lambda = 0.0001     #l2范数的学习率
        self.num_checkpoints=100       #存模型的频率
        self.num_test=1000
        self.dropout=1.0               #dropout比例
        self.learning_rate = tf.Variable(1e-2, trainable=False, name="learning_rate")  # ADD learning_rate
        self.batch_size=64
        self.Model_dir = "./TextRNN"  # 模型参数默认保存位置

        self.buildModel()

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
            return x

    def buildModel(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        init_value=tf.truncated_normal_initializer(stddev=0.1)
        # init_value=tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0)
        with tf.name_scope("embedding"):
            # W = tf.get_variable(initializer=init_value,shape=[self.vocab_size, self.embedding_size],name="Embedding")  #随机产生vocab_size个embedding_size维的字典
            W = tf.Variable(self.data.load_vocabulary(),name="embedding_w")
            embedded_chars = tf.nn.embedding_lookup(W, self.input_x)                                        #通过input_x查找对应字典的随机数

        with tf.name_scope("RNN"):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)  # forward direction cell
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)  # backward direction cell
            # if self.dropout_keep_prob is not None:
            #     lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            #     lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

            outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,embedded_chars,dtype=tf.float32,scope="LSTM_1") #[batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
            output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]


            # lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size*2)  # forward direction cell
            # lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size*2)  # backward direction cell
            # # if self.dropout_keep_prob is not None:
            # #     lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            # #     lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
            #
            # outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,output_rnn,dtype=tf.float32,scope="LSTM_2") #[batch_size,sequence_length,hidden_size*2] #creates a dynamic bidirectional recurrent neural network
            # output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*4]

        output_rnn=tf.expand_dims(output_rnn,-1)
        pooled = tf.nn.max_pool(output_rnn, ksize=[1, self.sequence_length, 1, 1], strides=[1, 1, 1, 1],padding='VALID', name="pool")
        pooled=tf.reshape(pooled,[-1,self.hidden_size*2])

        with tf.name_scope("output"):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            w_projection = tf.get_variable("w_projection", shape=[self.hidden_size * 2, self.num_classes],initializer=init_value)  # [embed_size,label_size]
            b_projection = tf.get_variable("bias_projection", shape=[self.num_classes])  # [label_size]
            logits = tf.matmul(pooled, w_projection) + b_projection  # [batch_size,num_classes]
            self.out = tf.nn.sigmoid(logits)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            l2_loss =tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])#bias偏置变量不参与L2范数计算
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            losses = tf.reduce_sum(losses,axis=1)
            losses = tf.reduce_mean(losses)
            self.loss=losses + self.l2_reg_lambda * l2_loss

        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        var_expect_embedding = [v for v in tf.trainable_variables() if 'embedding_w' not in v.name]
        grads_and_vars = optimizer.compute_gradients(self.loss, var_list=var_expect_embedding)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # optimizer1 = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars1 = optimizer.compute_gradients(self.loss)
        self.train_op1 = optimizer.apply_gradients(grads_and_vars1, global_step=self.global_step)

        # var_expect_embedding = [v for v in tf.trainable_variables() if 'embedding_w' not in v.name]
        # train_op_array=[]
        # learning_rate_temp=1e-3
        # for i in range(10):
        #     train_op_array.append(tf.train.AdamOptimizer(learning_rate_temp).minimize(self.loss,global_step=self.global_step,var_list=var_expect_embedding))
        #     learning_rate_temp/=2
        #
        # var_embedding=[v for v in tf.trainable_variables() if 'embedding_w' in v.name]
        # train_embedding_op=tf.train.AdamOptimizer(2e-4).minimize(self.loss,var_list=var_embedding)

        self.buildSummaries(grads_and_vars)

        # 初始化变量
        self.sess.run(tf.initialize_all_variables())
        self.Saver = tf.train.Saver()

    def buildSummaries(self,grads_and_vars):
        # Keep track of gradient values and sparsity (optional)
        # grad_summaries = []
        # for g, v in grads_and_vars:
        #     if g is not None:
        #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        #         grad_summaries.append(grad_hist_summary)
        #         grad_summaries.append(sparsity_summary)
        # grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 创建记录文件
        #timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        list = os.listdir(out_dir)
        make_dir=False
        for i in range(0, len(list)):
            if os.path.exists(out_dir + '\summaries_%d' % i):
                pass
            else:
                out_dir=out_dir + '\summaries_%d' % i
                os.makedirs(out_dir)
                make_dir=True
                break
        if not make_dir:
            out_dir = out_dir + '\summaries_%d' % len(list)
            os.makedirs(out_dir)
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        # acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # 训练集的记录
        self.train_summary_op = tf.summary.merge([loss_summary])    #归并记录, grad_summaries_merged
        train_summary_dir = os.path.join(out_dir, "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)   #类似打开文件操作

        # 校验集的记录
        self.dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

    def trainModel(self,num_epoch=10,last_batch=None):
        train_op_chioce=self.train_op
        train_num=0
        f1_max=0.0
        for epochnum in range(num_epoch):
            if last_batch==None:
                batches = self.data.train_batch_iter(self.batch_size, num_epoch)  # batch迭代器
            else:
                if last_batch!=-1:
                    batches = self.data.train_batch_iter(self.batch_size, num_epoch,True,last_batch)  # batch迭代器
                last_batch=None

            if epochnum>0:
                train_op_chioce=self.train_op1

            for x_batch,y_batch,batchnum,batchmax in batches:                                 #通过迭代器取出batch数据
                self.sess.graph.finalize()
                feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: self.dropout,self.is_train:True}
                _, summaries, loss ,step= self.sess.run([train_op_chioce, self.train_summary_op, self.loss,self.global_step], feed_dict=feed_dict)
                self.train_summary_writer.add_summary(summaries, step)  # 对记录文件添加上边run出的记录和step数

                if ((step - 1) % self.num_checkpoints == 0):
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    print('epoch:%d/%d\tbatch:%d/%d' % (epochnum, num_epoch, batchnum, batchmax))

                if ((step - 1) and (step - 1) % self.num_test == 0):
                    f1=self.testModel()
                    if f1>f1_max:
                        f1_max=f1
                        self.saveModel(batchnum)
                        print("saved")

            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print('epoch %d finish' % epochnum+1)

            # f1 = self.testModel()
            # if f1 > f1_max:
            #     f1_max = f1
            #     self.saveModel(-1)
            # else:
            #     self.loadModel()
            #     train_num+=1

    def testModel(self):
        self.data.init_evalution()
        dev_iter = self.data.dev_batch_iter()

        for x, y in dev_iter:
            feed_dict = {self.input_x: x, self.input_y: y, self.dropout_keep_prob: 1.0, self.is_train: False}
            summaries, loss, out, step = self.sess.run([self.dev_summary_op, self.loss, self.out, self.global_step],
                                                       feed_dict=feed_dict)

            for i in range(len(out)):
                self.data.evalution(out[i], y[i])

        p_5, r_5, f1 = self.data.get_evalution_result()

        print("Evaluation: step {}, loss {:g}, precision {:g}, recall {:g}, f1 {:g}".format(step, loss, p_5, r_5, f1))

        return f1 

    def Looptrain(self):
            num=self.loadModel()

            self.trainModel(last_batch=num)

    def saveModel(self, batchnum=0, dir=None):
        if (dir == None):
            dir = self.Model_dir
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.Saver.save(self.sess, dir + "/model.ckpt")
        f=open(dir+'/info.txt','w')
        f.write('%d\n' % batchnum)
        f.close()

    def loadModel(self,dir=None):
        if (dir == None):
            dir = self.Model_dir
        self.Saver.restore(self.sess, dir+"/model.ckpt")
        f = open(dir + '/info.txt', 'r')
        line=f.readline()
        f.close()
        return int(line)

def main():
    rnn=TextRNN()
    rnn.trainModel()

if __name__ == '__main__':
    main()