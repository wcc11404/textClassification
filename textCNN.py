import tensorflow as tf
import os
import time
import datetime
from textCNN_dataset import dataset

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

        self.sequence_length=56         #最长句子的长度
        self.num_classes=2              #分类结果的标签数
        self.vocab_size=18759           #字典数
        self.embedding_size=128         #词embedding后的维数
        self.filter_sizes=[3,4,5]       #cnn    filter大小,kernel为[filter,embedding_size]
        self.num_filters=128            #cnn    filter数量
        self.l2_reg_lambda = 0.0        #l2范数的学习率

        self.num_checkpoints=100        #存模型的频率
        self.dropout=0.3               #dropout比例
        self.learningrate=1e-3          #学习率
        self.batch_size=64
        self.num_epochs=100             #训练数据重复训练遍数
        self.test_epochs=100            #每testepochs次后测试一次校验集结果

        self.buildModel()

    def buildModel(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W")  #随机产生vocab_size个embedding_size维的字典
            embedded_chars = tf.nn.embedding_lookup(W, self.input_x)                                        #通过input_x查找对应字典的随机数
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)                                    #将[none,56,128]后边加一维，变成[none,56,128,1]

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W%d" % i)
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b%d" % i)
                conv = tf.nn.conv2d(embedded_chars_expanded, W,strides=[1, 1, 1, 1],padding="VALID")
                h = tf.nn.relu(tf.nn.bias_add(conv, b))

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h,ksize=[1, self.sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool%d" % i)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, len(self.filter_sizes))
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",shape=[num_filters_total, self.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)                                               #计算W,b的二范数，让他们尽可能小
            self.scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")               #全链接
            self.predictions = tf.argmax(self.scores, 1, name="predictions")        #取最大值所在下标

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))    #判断最大值下标是否一样
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy") #tf.cast强制转换数据类型

        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learningrate)
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
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # 训练集的记录
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])    #归并记录
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)   #类似打开文件操作

        # 校验集的记录
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

    def train_step(self,x_batch, y_batch):
        feed_dict = {self.input_x: x_batch,self.input_y: y_batch,self.dropout_keep_prob: self.dropout}
        _, step, summaries, loss, accuracy = self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy],feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()

        #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        self.train_summary_writer.add_summary(summaries, step)      #对记录文件添加上边run出的记录和step数

    def dev_step(self,x_batch, y_batch):
        feed_dict = {self.input_x: x_batch,self.input_y: y_batch,self.dropout_keep_prob: 1.0}
        step, summaries, loss, accuracy = self.sess.run([self.global_step, self.dev_summary_op, self.loss, self.accuracy],feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()

        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        self.dev_summary_writer.add_summary(summaries, step)

    def trainModel(self):
        x_dev,y_dev=self.data.dev_dataset()
        batches=self.data.batch_iter(self.batch_size,self.num_epochs)     #batch迭代器

        for x_batch,y_batch in batches:                                 #通过迭代器取出batch数据
            self.train_step(x_batch, y_batch)                           #训练
            current_step = tf.train.global_step(self.sess, self.global_step)    #无卵用的操作，等于sess.run(global_step),得到训练次数

            if current_step % self.test_epochs == 0:
                print("\nEvaluation:")
                self.dev_step(x_dev, y_dev)
                print("")

            if current_step % self.num_checkpoints == 0:
                path = self.saver.save(self.sess, self.checkpoint_prefix)#self.checkpoint_prefix是文件地址，破名字
                #print("Saved model checkpoint to {}\n".format(path))
