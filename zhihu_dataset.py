import numpy as np
import os
from collections import Counter
import pickle


class dataset(object):
    def __init__(self,init=False):
        self.data_dir = "./dataset/zhihudataset/temp2/"
        self.char_array_dir = self.data_dir + "char_array.pik"
        self.char_model_dir = self.data_dir + "char_model.pik"
        self.label_array_dir = self.data_dir + "label_array.pik"
        self.label_model_dir = self.data_dir + "label_model.pik"

        #加载模型参数
        with open(self.data_dir+"info.txt",'r') as f:
            line=f.readline()
            line=line.strip().split(' ')
            self.vx_num=int(line[0])
            self.vx_size=int(line[2])
            line=f.readline()
            self.vy_num=int(line.strip())
            line=f.readline()
            line=line.strip().split(' ')
            self.max_sentence_size=int(line[0])
            self.train_num=int(line[1])
            self.test_num=int(line[2])

    def get_param(self):
        #return 最长句子长度，输出维数，输入字典维数,输入字典内容维数(embedding_size)
        return self.max_sentence_size,self.vy_num,self.vx_num,self.vx_size

    def load_vocabulary(self):
        with open(self.char_array_dir, 'rb') as data_f:
            v=pickle.load(data_f)
        return v

    def load_data_and_labels(self,i):
        f = open(self.data_dir+'question_temp_%d.txt' % i, 'r', errors='ignore')
        X = []
        Y = []
        for line in f.readlines():
            line = line.strip().split("__label__")
            input_list = line[0].strip().split(" ")
            label_list = line[1:]
            x = [word for word in input_list if word != '']
            num=len(x)
            for i in range(self.max_sentence_size - num):
                x.append(0)
            y = [int(label.strip()) for label in label_list if label!='']
            y=self.transform_multilabel_as_multihot(y,self.vy_num)
            X.append(x)
            Y.append(y)

        return X, Y

    def transform_multilabel_as_multihot(self,label_list, label_size):
        """
        convert to multi-hot style
        :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
        :param label_size: e.g.199
        :return:e.g.[1,1,0,1,0,0,........]
        """
        # result = np.zeros(label_size)
        result = [0 for i in range(label_size)]
        # set those location as 1, all else place as 0.
        # result[label_list] = 1
        for temp in label_list:
            result[temp]=1
        return result

    def get_information_from_line(self,line):
        line = line.strip().split("__label__")
        input_list = line[0].strip().split(" ")
        label_list = line[1:]
        x = [word for word in input_list if word != '']
        num = len(x)
        for i in range(self.max_sentence_size - num):
            x.append(0)
        y = [int(label.strip()) for label in label_list if label != '']
        y = self.transform_multilabel_as_multihot(y, self.vy_num)
        return x,y

    # for i in range(self.textfilenum):
    #     x_train,y_train=self.load_data_and_labels(i)
    #     data = np.array(list(zip(x_train, y_train)))  # 转换成[x,y]键值对的形式数组
    #     data_size = len(data)
    #     num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    #
    #     if shuffle:
    #         shuffle_indices = np.random.permutation(np.arange(data_size))
    #         shuffled_data = data[shuffle_indices]  # 乱序数组
    #     else:
    #         shuffled_data = data  # 正序数组
    #     for batch_num in range(num_batches_per_epoch):
    #         start_index = batch_num * batch_size
    #         end_index = min((batch_num + 1) * batch_size, data_size)
    #         x,y=zip(*shuffled_data[start_index:end_index])
    #         yield x,y,epoch,num_epochs,i,self.textfilenum,batch_num,num_batches_per_epoch

    def train_batch_iter(self, batch_size, num_epochs,load=False,load_batch_num=0):
        for epoch in range(num_epochs):
            f = open(self.data_dir + 'question_temp.txt', 'r', errors='ignore')
            min_num=0

            if load:
                load=False
                min_num=load_batch_num+1
                for i in range(min_num*batch_size):
                    line=f.readline()

            tempnum=(self.train_num-1)//batch_size+1
            for k in range(min_num,tempnum):
                X = []
                Y = []
                for j in range(batch_size):
                    line=f.readline()
                    if line=='':
                        break

                    x,y=self.get_information_from_line(line)
                    X.append(x)
                    Y.append(y)
                yield X,Y,epoch,num_epochs,k,tempnum

            f.close()



    def dev_batch_iter(self,batch_size=64):
        f = open(self.data_dir + 'question_temp_test.txt', 'r', errors='ignore')
        for k in range((self.test_num-1)//batch_size+1):
            X = []
            Y = []
            for j in range(batch_size):
                line = f.readline()
                if line == '':
                    break

                x, y = self.get_information_from_line(line)
                X.append(x)
                Y.append(y)
            yield X, Y
        f.close()

import datetime

def main():
    data=dataset()

    starttime = datetime.datetime.now()

    i=0
    iter=data.train_batch_iter(64,1)
    for x,y,_,_,_,_ in iter:
        i+=1

        if(i%1000==0):
            print(i)

    endtime = datetime.datetime.now()
    print(endtime - starttime).seconds

if __name__ == '__main__':
  main()