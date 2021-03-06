import tensorflow as tf
import datetime
from bioasq_dataset import dataset
# from zhihu_dataset import dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Inception(object):
    def __init__(self, mode=2):
        self.sess = tf.InteractiveSession()
        self.mode=mode
        self.data=dataset(self.mode)             #数据集

        self.sequence_length,self.num_classes,self.vocab_size,self.embedding_size=self.data.get_param()

        self.mode_learning_rate=1e-3
        self.embed_learning_rate = 2e-4
        self.num_checkpoints = 100      # 打印的频率
        self.l2_reg_lambda = 0.0001     #l2范数的学习率
        self.dropout=0.5               #dropout比例
        self.batch_size=100
        self.num_optimizer = 10
        self.Model_dir = "Inception"  # 模型参数默认保存位置

        self.is_train= tf.placeholder(tf.bool)

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

    def convolution(self,input,shape,scope,strides=[1,1,1,1],padding="SAME",init=tf.truncated_normal_initializer(stddev=0.1),is_bn=True):
        with tf.variable_scope(scope):
            w=tf.get_variable("weight_conv",shape,tf.float32,init)
            b=tf.get_variable("bias_conv",[shape[3]],tf.float32,tf.constant_initializer(0.1))
            conv=tf.nn.conv2d(input,w,strides,padding)
            #tf.nn.convolution(dilation_rate)
            conv=tf.nn.bias_add(conv, b)
            if is_bn:
                conv=self.batch_norm(conv,self.is_train,name='bn')
            conv=tf.nn.relu(conv)
            return conv

    def liner(self,input,outputshape,scope,init=tf.truncated_normal_initializer(stddev=0.1),is_bn=True):
        shape=int(input.shape[1])
        with tf.variable_scope(scope):
            w = tf.get_variable("weight_line", [shape, outputshape], tf.float32, init)
            b = tf.get_variable("bias_line", [outputshape], tf.float32, tf.constant_initializer(0.1))
            line = tf.matmul(input, w) + b
            if is_bn:
                line = self.batch_norm(line, self.is_train, name='bn')
            line = tf.nn.relu(line)
            return line

    def reduction(self,input,scope="reduction"):
        shape_in = int(input.shape[3])
        with tf.variable_scope(scope):
            branch_1 = self.convolution(input, [3, 3, shape_in, shape_in], 'conv', [1, 2, 2, 1])
            branch_2 = tf.nn.max_pool(input, [1, 3, 3, 1], [1, 2, 2, 1], "SAME", name='max_pool')
            out=tf.concat([branch_1,branch_2],3)
        return out

    def inceptionA(self,input,output_size=-1,output_proportion=[64,128,32,32],max_size=3,scope='inceptionA'):#[2,4,1,1]
        if len(output_proportion)!=4:
            print('error')
            return None
        shape_in=int(input.shape[3])
        if output_size==-1:
            output_size=shape_in
        sum=0
        for i in output_proportion:
            sum+=i
        shape_out_1 = output_size * (output_proportion[0] / sum)
        shape_out_2 = output_size * (output_proportion[1] / sum)
        shape_out_3 = output_size * (output_proportion[2] / sum)
        shape_out_4 = output_size - shape_out_1 - shape_out_2 - shape_out_3
        with tf.variable_scope(scope):
            branch_1=self.convolution(input,[1,1,shape_in,shape_out_1],'conv1')

            branch_2 = self.convolution(input, [1, 1, shape_in, shape_out_2 // 2], 'conv2_1')
            branch_2 = self.convolution(branch_2, [max_size, max_size, shape_out_2 // 2, shape_out_2], 'conv2_2')

            branch_3 = self.convolution(input, [1, 1, shape_in, shape_out_3 // 2], 'conv3_1')
            branch_3 = self.convolution(branch_3, [max_size, max_size, shape_out_3 // 2, shape_out_3 // 2], 'conv3_2')
            branch_3 = self.convolution(branch_3, [max_size, max_size, shape_out_3 // 2, shape_out_3], 'conv3_3')

            branch_4 = tf.nn.avg_pool(input, [1, 3, 3, 1], [1, 1, 1, 1], "SAME", name='avg_pool')
            branch_4 = self.convolution(branch_4, [1, 1, shape_in, shape_out_4], 'conv4')

            out=tf.concat([branch_1,branch_2,branch_3,branch_4],3)
        return out

    def inceptionB(self,input,output_size=-1,output_proportion=[64,128,32,32],max_size=3,scope='inceptionB'):#[2,4,1,1]
        if len(output_proportion)!=4:
            print('error')
            return None
        shape_in=int(input.shape[3])
        if output_size==-1:
            output_size=shape_in
        sum=0
        for i in output_proportion:
            sum+=i
        shape_out_1 = output_size * (output_proportion[0] / sum)
        shape_out_2 = output_size * (output_proportion[1] / sum)
        shape_out_3 = output_size * (output_proportion[2] / sum)
        shape_out_4 = output_size - shape_out_1 - shape_out_2 - shape_out_3
        with tf.variable_scope(scope):
            branch_1=self.convolution(input,[1,1,shape_in,shape_out_1],'conv1')

            branch_2 = self.convolution(input, [1, 1, shape_in, shape_out_2 // 2], 'conv2_1')
            branch_2 = self.convolution(branch_2, [1, max_size, shape_out_2 // 2, shape_out_2 // 2], 'conv2_2')
            branch_2 = self.convolution(branch_2, [max_size, 1, shape_out_2 // 2, shape_out_2], 'conv2_3')

            branch_3 = self.convolution(input, [1, 1, shape_in, shape_out_3 // 2], 'conv3_1')
            branch_3 = self.convolution(branch_3, [1, max_size, shape_out_3 // 2, shape_out_3 // 2], 'conv3_2')
            branch_3 = self.convolution(branch_3, [max_size, 1, shape_out_3 // 2, shape_out_3], 'conv3_3')
            branch_3 = self.convolution(branch_3, [1, max_size, shape_out_3, shape_out_3], 'conv3_4')
            branch_3 = self.convolution(branch_3, [max_size, 1, shape_out_3, shape_out_3], 'conv3_5')

            branch_4 = tf.nn.avg_pool(input, [1, 3, 3, 1], [1, 1, 1, 1], "SAME", name='avg_pool')
            branch_4 = self.convolution(branch_4, [1, 1, shape_in, shape_out_4], 'conv4')

            out=tf.concat([branch_1,branch_2,branch_3,branch_4],3)
        return out

    def buildModel(self):
        if self.vocab_size!=0:
            self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        else :
            self.input_x = tf.placeholder(tf.float32, [None, self.sequence_length,self.embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope('embedding'):
            init_value=tf.random_normal_initializer(stddev=0.1)
            if self.vocab_size!=0:
                with tf.name_scope("embedding"):
                    W = tf.Variable(self.data.load_vocabulary(),name="embedding_w")
                    embedded_chars = tf.nn.embedding_lookup(W, self.input_x)                                        #通过input_x查找对应字典的随机数
            else:
                embedded_chars=self.input_x

            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)#将[none,56,128]后边加一维，变成[none,56,128,1]

        with tf.name_scope('Inception'):
            conv = self.convolution(embedded_chars_expanded, [3, 3, 1, 32], 'conv111',strides=[1, 2, 2, 1])
            conv = self.reduction(conv, 'reduction1')
            conv = self.reduction(conv, 'reduction2')
            conv = self.reduction(conv, 'reduction3')
            out = self.inceptionB(conv,output_proportion=[1,1,1,1])

        with tf.name_scope('pooling'):
            out=tf.nn.max_pool(out,ksize=[1, int(out.shape[1]), 10, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool")
            temp=int(out.shape[1])*int(out.shape[2])*int(out.shape[3])
            out=tf.reshape(out,[-1,temp])

        # with tf.name_scope("dropout"):
        #     out = tf.nn.dropout(out, self.dropout_keep_prob)

        with tf.name_scope("liner"):
            w2 = tf.Variable(tf.truncated_normal([temp, self.num_classes], stddev=0.1), name='weight_line_2')
            b2 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='bias_liner_2')
            liner_out = tf.matmul(out, w2) + b2

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.out = tf.nn.sigmoid(liner_out)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            l2_loss =tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=liner_out, labels=self.input_y)
            losses = tf.reduce_sum(losses,axis=1)
            losses = tf.reduce_mean(losses)
            self.loss=losses + self.l2_reg_lambda * l2_loss

        # Define Training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        var_expect_embedding = [v for v in tf.trainable_variables() if 'embedding_w' not in v.name]
        train_adamop_array=[]
        learning_rate_temp=self.mode_learning_rate
        for i in range(self.num_optimizer):
            train_adamop_array.append(tf.train.AdamOptimizer(learning_rate_temp))#.minimize(self.loss,global_step=self.global_step,var_list=var_expect_embedding))
            learning_rate_temp/=2.0
        var_embedding=[v for v in tf.trainable_variables() if 'embedding_w' in v.name]
        train_embedding_adamop=tf.train.AdamOptimizer(self.embed_learning_rate)#.minimize(self.loss,var_list=var_embedding)

        grads=tf.gradients(self.loss,var_expect_embedding+var_embedding)
        grads1=grads[:len(var_expect_embedding)]
        grads2=grads[len(var_expect_embedding):]

        self.train_op_array=[]
        for i in range(self.num_optimizer):
            self.train_op_array.append(train_adamop_array[i].apply_gradients(zip(grads1, var_expect_embedding), global_step=self.global_step))
        if self.vocab_size != 0:
            self.train_embedding_op = train_embedding_adamop.apply_gradients(zip(grads2, var_embedding))

        self.buildSummaries()

        # 初始化变量
        self.sess.run(tf.global_variables_initializer())
        self.Saver = tf.train.Saver()

    def buildSummaries(self,):
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

    def trainModel(self):
        train_num = 0
        same_num=0
        train_op_chioce = self.train_op_array[train_num]
        f1_max = 0.0

        print("start training")
        self.starttime = datetime.datetime.now()
        self.write_log_infomation("start time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True)

        max_epochs = self.num_optimizer * 4
        for epochnum in range(max_epochs):
            batches = self.data.train_batch_iter(self.batch_size)  # batch迭代器

            for x_batch, y_batch, batchnum, batchmax in batches:  # 通过迭代器取出batch数据
                self.sess.graph.finalize()

                feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: self.dropout,
                             self.is_train: True}
                if self.vocab_size!=0 and epochnum>=1:
                    _,_, summaries, loss, step = self.sess.run(
                        [self.train_embedding_op,train_op_chioce, self.train_summary_op, self.loss, self.global_step], feed_dict=feed_dict)
                else:
                    _, summaries, loss, step = self.sess.run(
                        [train_op_chioce, self.train_summary_op, self.loss, self.global_step], feed_dict=feed_dict)
                self.train_summary_writer.add_summary(summaries, step)  # 对记录文件添加上边run出的记录和step数

                if (batchnum % self.num_checkpoints == 0):
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    print('epoch:%d/%d\tbatch:%d/%d\tloss:%f' % (epochnum, max_epochs, batchnum, batchmax, loss))

                if batchnum % 15001==0 or (epochnum == max_epochs-1 and batchnum == batchmax // 2):
                    p, r, f1 = self.testModel()
                    # if f1 > f1_max:
                    #     f1_max = f1
                    #     self.saveModel()
                    #     print("saved")
                    str = "\n第%d轮训练一半\n时间 : " % (epochnum + 1)
                    str += datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    str += '\np : %f , r : %f , f1 : %f\n' % (p, r, f1)
                    self.write_log_infomation(str)

            # 结束一轮训练后，测试
            p, r, f1 = self.testModel()
            if f1 > f1_max:
                f1_max = f1
                self.saveModel()
                print("saved")
                same_num += 1
            else:
                self.loadModel()
                train_num += 1
                same_num = 0

            #同样的优化器最多用5次
            if same_num == 5:
                train_num += 1

            str = "\n第%d轮训练结束\n时间 : " % (epochnum + 1)
            str += datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            str += '\np : %f , r : %f , f1 : %f\n' % (p, r, f1)
            self.write_log_infomation(str)

            if train_num < self.num_optimizer:
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
    inception=Inception()
    inception.trainModel()
    # inception.loadModel()
    # inception.testModel()

if __name__ == '__main__':
    main()

    def buildModel(self):
        if self.vocab_size!=0:
            self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        else :
            self.input_x = tf.placeholder(tf.float32, [None, self.sequence_length,self.embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        init_value=tf.random_normal_initializer(stddev=0.1)
        if self.vocab_size!=0:
            with tf.name_scope("embedding"):
                W = tf.Variable(self.data.load_vocabulary(),name="embedding_w")
                embedded_chars = tf.nn.embedding_lookup(W, self.input_x)                                        #通过input_x查找对应字典的随机数
        else:
            embedded_chars=self.input_x

        # cell = [rnn.LSTMCell(self.hidden_size) for i in range(2)]
        # mutli_layer = rnn.MultiRNNCell(cell)
        # attention_mechanism = LuongAttention(num_units=self.hidden_size,memory=embedded_chars)
        # att_wrapper = AttentionWrapper(cell=mutli_layer,attention_mechanism=attention_mechanism,
        #                                attention_layer_size=self.hidden_size,
        #                                cell_input_fn=lambda input, attention: input)
        # states = att_wrapper.zero_state(self.batch_size, tf.float32)
        # print(states)

        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)  # forward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)  # backward direction cell
        # mlstm_fw_cell = rnn.MultiRNNCell([lstm_fw_cell] * 1, state_is_tuple=True)
        # mlstm_bw_cell = rnn.MultiRNNCell([lstm_bw_cell] * 1, state_is_tuple=True)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_chars, dtype=tf.float32,
                                                     scope="LSTM_1")  # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
        out=self.attention(output_rnn,self.hidden_size*2)
        num_filters_total=int(out.shape[1])
        # Add dropout
        # with tf.name_scope("dropout"):
        #     h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope("liner"):
            w2 = tf.Variable(tf.truncated_normal([num_filters_total, self.num_classes], stddev=0.1),
                             name='weight_line_2')
            b2 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='bias_liner_2')
            liner_out2 = tf.matmul(out, w2) + b2

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.out = tf.nn.sigmoid(liner_out2)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            l2_loss =tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=liner_out2, labels=self.input_y)
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
        train_adamop_array=[]
        learning_rate_temp = self.mode_learning_rate
        for i in range(self.num_epochs):
            train_adamop_array.append(tf.train.AdamOptimizer(learning_rate_temp))#.minimize(self.loss,global_step=self.global_step,var_list=var_expect_embedding))
            learning_rate_temp/=2.0
        var_embedding=[v for v in tf.trainable_variables() if 'embedding_w' in v.name]
        train_embedding_adamop=tf.train.AdamOptimizer(self.embed_learning_rate)#.minimize(self.loss,var_list=var_embedding)

        grads=tf.gradients(self.loss,var_expect_embedding+var_embedding)
        grads1=grads[:len(var_expect_embedding)]
        grads2=grads[len(var_expect_embedding):]

        self.train_op_array=[]
        for i in range(self.num_epochs):
            self.train_op_array.append(train_adamop_array[i].apply_gradients(zip(grads1,var_expect_embedding),global_step=self.global_step))
        if self.vocab_size!=0:
            self.train_embedding_op=train_embedding_adamop.apply_gradients(zip(grads2,var_embedding))

        self.buildSummaries()

        # 初始化变量
        self.sess.run(tf.global_variables_initializer())

        self.Saver = tf.train.Saver()