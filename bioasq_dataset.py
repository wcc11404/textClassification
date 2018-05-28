import numpy as np
import pickle
import datetime

class dataset(object):
    def __init__(self,mode=1):
        self.data_dir = "D:/wang/"
        self.train_data_x_dir = self.data_dir + "out/train_data_x/data_"
        self.train_data_y_dir = self.data_dir + "out/train_data_y/data_"

        self.max_sentence_size=360
        self.vy_num=28340
        self.vx_num=0
        self.vx_size=200
        self.max_file_num=45
        self.max_train_file_num=self.max_file_num-1 #0-43号文件作为训练文件

        self.mode=mode
        self.name='bioasq'

        self.init_evalution()

    def get_param(self):
        #return 最长句子长度，输出维数，输入字典维数,输入字典内容维数(embedding_size)
        return self.max_sentence_size,self.vy_num,self.vx_num,self.vx_size

    def load_vocabulary(self):
        return None

    def process_X(self,x):
        temp_x=[]
        for i in range(len(x)):
            temp_x.append(list(float(w) for w in x[i].split(' ')))
        return temp_x

    def process_Y(self,y):
        result=[0 for i in range(self.vy_num)]
        for i in y:
            result[i] = 1
        return result

    def train_batch_iter(self, batch_size, num_epochs=0):
        # for epoch in range(num_epochs):
        for i in range(self.max_train_file_num+1):
            with open(self.train_data_x_dir + '%d' % i, 'rb') as f:
                temp_data_x=pickle.load(f)
            with open(self.train_data_y_dir + '%d' % i, 'rb') as f:
                temp_data_y=pickle.load(f)

            train_num=len(temp_data_x)
            tempnum=(train_num-1)//batch_size+1
            for k in range(tempnum):
                X = []
                Y = []
                min_num=k*batch_size
                max_num=min((k+1)*batch_size,train_num)
                for j in range(min_num,max_num):
                    # x = self.process_X(temp_data_x[j])
                    x=temp_data_x[i]
                    y = self.process_Y(temp_data_y[j])
                    X.append(x)
                    Y.append(y)
                yield X,Y,k+i*tempnum+1,tempnum*(self.max_file_num+1)+1

    def dev_batch_iter(self,batch_size=50):
        with open(self.train_data_x_dir + '%d' % self.max_train_file_num, 'rb') as f:
            temp_data_x=pickle.load(f)
        with open(self.train_data_y_dir + '%d' % self.max_train_file_num, 'rb') as f:
            temp_data_y=pickle.load(f)

        train_num=len(temp_data_x)
        tempnum=(train_num-1)//batch_size+1
        for k in range(200):
            X = []
            Y = []
            min_num=k*batch_size
            max_num=min((k+1)*batch_size,train_num)
            for j in range(min_num,max_num):
                # x = self.process_X(temp_data_x[j])
                x=temp_data_x[j]
                y = self.process_Y(temp_data_y[j])
                X.append(x)
                Y.append(y)
            yield X,Y,k,tempnum

    def init_evalution(self):
        self.fenzi = 0.0
        self.p_fenmu = 0.0
        self.r_fenmu = 0.0
        self.f1 = 0.0

    def get_evalution_result(self):
        p = self.fenzi / self.p_fenmu
        r = self.fenzi / self.r_fenmu
        f1 = 2 * p * r / (p + r + 0.000001)
        return p, r, f1

    def evalution(self,logits, label):      #计算微平均
        label_list = self.get_label_using_logits(logits)
        eval_y_short = self.get_target_label_short(label)
        num_correct_label = 0

        for i,label_predict in enumerate(label_list):
            if label_predict in eval_y_short:
                num_correct_label += 1

        self.fenzi +=num_correct_label
        self.p_fenmu+=len(label_list)
        self.r_fenmu+=len(eval_y_short)

    def get_target_label_short(self,eval_y):
        eval_y_short = []  # will be like:[22,642,1391]
        for index, label in enumerate(eval_y):
            if label > 0:
                eval_y_short.append(index)
        return eval_y_short

    # get top5 predicted labels
    def get_label_using_logits(self,logits, top_number=-1,maximin=0.5,minimax=1):
        if top_number>0:
            index_list = np.argsort(logits)[-top_number:]
            index_list = index_list[::-1]
        else:
            logits=np.array(logits)
            index_list = np.where((logits > maximin) & (logits <= minimax))[0]
        return index_list

def main():
    data = dataset()

    # starttime = datetime.datetime.now()
    #
    # iter=data.train_batch_iter(100)
    # for x,y,a,b in iter:
    #     if a%100==0:
    #         print("%d/%d" %(a,b))
    #
    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)
    x=[0.6,0.7,0.8,0.9]
    y=[1,1,1,1]
    data.evalution(x,y)
    print(data.get_evalution_result())

if __name__ == '__main__':
    main()