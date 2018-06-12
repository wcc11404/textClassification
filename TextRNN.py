import tensorflow as tf
import datetime
from bioasq_dataset import dataset
# from zhihu_dataset import dataset
from tensorflow.contrib import rnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TextRNN(object):
    def __init__(self,mode=2):
        self.sess = tf.InteractiveSession()
        self.mode=mode
        self.data=dataset(self.mode)             #数据集

        self.sequence_length,self.num_classes,self.vocab_size,self.embedding_size=self.data.get_param()

        # self.decay_steps = 3000
        # self.decay_rate = 0.5  # 62
        #self.learning_rate = tf.Variable(1e-2, trainable=False, name="learning_rate")  # ADD learning_rate

        self.l2_reg_lambda = 0.0001     #l2范数的学习率
        self.num_checkpoints = 100  # 打印的频率
        self.dropout=1.0               #dropout比例
        self.mode_learning_rate = 5e-4
        self.embed_learning_rate = 2e-4
        self.batch_size=50
        self.num_epochs = 10            #总的训练次数
        self.Model_dir = "TextRNN"  # 模型参数默认保存位置

        self.hidden_size = self.embedding_size
        self.is_train = tf.placeholder(tf.bool)  # batchnormalization用

        self.buildModel()

    def buildModel(self):
        if self.vocab_size!=0:
            self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        else :
            self.input_x = tf.placeholder(tf.float32, [None, self.sequence_length,self.embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        init_value=tf.truncated_normal_initializer(stddev=0.1)
        if self.vocab_size!=0:
            with tf.name_scope("embedding"):
                W = tf.Variable(self.data.load_vocabulary(),name="embedding_w")
                embedded_chars = tf.nn.embedding_lookup(W, self.input_x)                                        #通过input_x查找对应字典的随机数
        else:
            embedded_chars=self.input_x

        with tf.name_scope("RNN"):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)  # backward direction cell
            mlstm_fw_cell = rnn.MultiRNNCell([lstm_fw_cell]*2,state_is_tuple=True)
            mlstm_bw_cell = rnn.MultiRNNCell([lstm_bw_cell]*2,state_is_tuple=True)
            # if self.dropout_keep_prob is not None:
            #     lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            #     lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

            outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,embedded_chars,dtype=tf.float32,scope="LSTM_1") #[batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
            output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]

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
        ##########################################  简版训练op  #################################################
        # if self.mode==1:
        #     learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
        #                                                self.decay_rate, staircase=True)
        #     optimizer = tf.train.AdamOptimizer(learning_rate)
        #
        #     var_expect_embedding=[v for v in tf.trainable_variables() if 'embedding_w' not in v.name]
        #     grads_and_vars = optimizer.compute_gradients(self.loss,var_list=var_expect_embedding)
        #     self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        #
        #     # optimizer1 = tf.train.AdamOptimizer(learning_rate)
        #     grads_and_vars1=optimizer.compute_gradients(self.loss)
        #     self.train_op1 = optimizer.apply_gradients(grads_and_vars1, global_step=self.global_step)
        # ##########################################################################################################
        # elif self.mode==2:
        var_expect_embedding = [v for v in tf.trainable_variables() if 'embedding_w' not in v.name]
        train_adamop_array = []
        learning_rate_temp = self.mode_learning_rate
        for i in range(self.num_epochs):
            train_adamop_array.append(tf.train.AdamOptimizer(
                learning_rate_temp))  # .minimize(self.loss,global_step=self.global_step,var_list=var_expect_embedding))
            learning_rate_temp /= 2.0
        var_embedding = [v for v in tf.trainable_variables() if 'embedding_w' in v.name]
        train_embedding_adamop = tf.train.AdamOptimizer(self.embed_learning_rate)  # .minimize(self.loss,var_list=var_embedding)

        grads = tf.gradients(self.loss, var_expect_embedding + var_embedding)
        grads1 = grads[:len(var_expect_embedding)]
        grads2 = grads[len(var_expect_embedding):]

        self.train_op_array = []
        for i in range(self.num_epochs):
            self.train_op_array.append(
                train_adamop_array[i].apply_gradients(zip(grads1, var_expect_embedding), global_step=self.global_step))
        if self.vocab_size != 0:
            self.train_embedding_op = train_embedding_adamop.apply_gradients(zip(grads2, var_embedding))

        self.buildSummaries()

        # 初始化变量
        self.sess.run(tf.global_variables_initializer())
        self.Saver = tf.train.Saver()

    def buildSummaries(self):
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

    def trainModel(self,num_epoch=10):
        # if self.mode==1:
        #     self.trainModel1(num_epoch)
        # elif self.mode==2:
        self.trainModel2()

    def trainModel1(self,num_epoch=10):
        train_op_chioce=self.train_op
        f1_max=0.0

        print("start training")
        self.starttime = datetime.datetime.now()

        for epochnum in range(num_epoch):
            batches = self.data.train_batch_iter(self.batch_size, num_epoch)  # batch迭代器

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
                    p,r,f1=self.testModel()
                    if f1>f1_max:
                        f1_max=f1
                        self.saveModel()
                        f = open("./" + self.Model_dir + '/info.txt', 'w')
                        time = datetime.datetime.now()
                        str = '第%d轮训练用时%ds\n' % (epochnum + 1, (time - self.starttime).seconds)
                        str += 'p_5 : %f , r_5 : %f , f1 : %f\n' % (p, r, f1)
                        f.write(str)
                        f.close()
                        print("saved")

            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print('epoch %d finish' % (epochnum+1))

    def trainModel2(self):
        train_num = 0
        train_op_chioce = self.train_op_array[train_num]
        f1_max = 0.0

        print("start training")
        self.starttime = datetime.datetime.now()
        self.write_log_infomation("start time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True)

        for epochnum in range(self.num_epochs):
            batches = self.data.train_batch_iter(self.batch_size)  # batch迭代器

            for x_batch, y_batch, batchnum, batchmax in batches:  # 通过迭代器取出batch数据
                # self.sess.graph.finalize()

                feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: self.dropout,
                             self.is_train: True}
                if self.vocab_size!=0 and epochnum>=1:
                    _,_, summaries, loss, step = self.sess.run([self.train_embedding_op,train_op_chioce, self.train_summary_op, self.loss, self.global_step], feed_dict=feed_dict)
                else:
                    _, summaries, loss, step = self.sess.run(
                        [train_op_chioce, self.train_summary_op, self.loss, self.global_step], feed_dict=feed_dict)
                self.train_summary_writer.add_summary(summaries, step)  # 对记录文件添加上边run出的记录和step数

                if (batchnum % self.num_checkpoints == 0):
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    print('epoch:%d/%d\tbatch:%d/%d' % (epochnum, self.num_epochs, batchnum, batchmax))

                if batchnum%20001==0 or (epochnum == self.num_epochs-1 and batchnum == batchmax // 2):
                    p, r, f1 = self.testModel()
                    if f1 > f1_max:
                        f1_max = f1
                        self.saveModel()
                        print("saved")
                    str = "\n第%d轮训练一半\n时间 : " % (epochnum + 1)
                    str += datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    str += '\np : %f , r : %f , f1 : %f\n' % (p, r, f1)
                    self.write_log_infomation(str)

            #结束一轮训练后，测试
            p,r,f1 = self.testModel()
            if f1 > f1_max:
                f1_max = f1
                self.saveModel()
                print("saved")
            # else:
            #     self.loadModel()

            str = "\n第%d轮训练结束\n时间 : " % (epochnum + 1)
            str += datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            str += '\np : %f , r : %f , f1 : %f\n' % (p, r, f1)
            self.write_log_infomation(str)
            # train_num += 1

            if train_num < self.num_epochs:
                train_op_chioce = self.train_op_array[train_num]
            else:
                break

            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print('epoch %d finish' % (epochnum + 1))

        self.write_log_infomation('\n最大F值为 : %f' % f1_max)

    def testModel(self):
        self.data.init_evalution()
        dev_iter = self.data.dev_batch_iter()

        for x,y in dev_iter:
            feed_dict = {self.input_x: x, self.input_y: y, self.dropout_keep_prob: 1.0,self.is_train:False}
            summaries, loss, out ,step= self.sess.run([self.dev_summary_op, self.loss, self.out,self.global_step],feed_dict=feed_dict)

            for i in range(len(out)):
                self.data.evalution(out[i],y[i])

        p, r, f1 = self.data.get_evalution_result()

        print("Evaluation: precision {:g}, recall {:g}, f1 {:g}".format(p, r, f1))

        return p, r, f1

    def saveModel(self, dir=None):
        if (dir == None):
            dir = self.Model_dir+'_'+self.data.name
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.Saver.save(self.sess, dir + "/model.ckpt")

    def loadModel(self,dir=None):
        if (dir == None):
            dir = self.Model_dir+'_'+self.data.name
        self.Saver.restore(self.sess, "./"+dir+"/model.ckpt")

    def write_log_infomation(self,str,init=False):
        if init:
            if not os.path.exists(self.Model_dir + '_' + self.data.name):
                os.makedirs(self.Model_dir + '_' + self.data.name)
            f = open("./" + self.Model_dir + '_' + self.data.name + '/info.txt', 'w')
        else:
            f = open("./" + self.Model_dir + '_' + self.data.name + '/info.txt', 'a')
        f.write(str)
        f.close()

def main():
    rnn=TextRNN()
    rnn.trainModel()

if __name__ == '__main__':
    main()