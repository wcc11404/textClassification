import textCNN
import sentiment_dataset
import numpy as np
import tensorflow as tf
import codecs
import os
import pickle
from collections import Counter
import pickle
import random

save_dir='./dataset/zhihudataset/temp1/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
###########    读取问题id和topicid的对应关系     ######
topic_map={}
f=open('./dataset/zhihudataset/ieee_zhihu_cup/question_topic_train_set.txt','r')
for line in f.readlines():
    line=line.strip().split('\t')
    topic_map[line[0]]=line[1]
f.close()
print('load topic_map')

############    读取字符word2vec模型，并处理成{}和[]        ######
def readModel(dir):
    map={}
    array=[]
    f=open(dir,'r')
    line=f.readline()
    line=line.strip().split(' ')
    num=int(line[0])
    embedding_size=int(line[1])
    line=f.readline()
    i=0
    array.append([0 for i in range(embedding_size)])
    array.append([random.uniform(-0.25,0.25) for i in range(embedding_size)])   #未出现的词统一随机初始化
    while line!='':
        line=line.strip().split(' ')
        map[line[0]]=i+2
        i+=1
        array.append([float(l) for l in line[1:]])
        line=f.readline()
    return map,array,num,embedding_size

char_model,char_array,char_num,char_size=readModel('./dataset/zhihudataset/ieee_zhihu_cup/char_embedding.txt')
with open(save_dir+'char_model.pik', 'wb') as data_f:
    pickle.dump(char_model,data_f)
with open(save_dir+'char_array.pik', 'wb') as data_f:
    pickle.dump(char_array,data_f)
# word_model,word_num,_,word_size=readModel('./dataset/zhihudataset/ieee_zhihu_cup/word_embedding.txt')
print('load char_model')

###########     统计所有label，并处理成{}和[]     ###########
f=open('./dataset/zhihudataset/ieee_zhihu_cup/question_train_set.txt','r')
line=f.readline()
count=Counter()
while line!='':
    line = line.strip().split('\t')
    topic = topic_map[line[0]]
    topic = topic.strip().split(',')
    for i in topic:
        count.update([int(i)])

    line = f.readline()

label_map={}
label_array=[]
for j,tuplee in enumerate(count.most_common()):
    label,_=tuplee
    label_map[label]=j
    label_array.append(label)
with open(save_dir+'label_model.pik', 'wb') as data_f:
    pickle.dump(label_map,data_f)
with open(save_dir+'label_array.pik', 'wb') as data_f:
    pickle.dump(label_array,data_f)
print('load label_map')

#########       处理训练数据，将所有字符转换成对应id，label转换成对应id，存储在n个文件中       #########
max_example=2999967
title_size_maxlength=85
text_size_maxlength=300
num=0
max_sentence_size=0
# train_max=2960000
train_proportion=0.95
train_num=0
test_num=0

f=open('./dataset/zhihudataset/ieee_zhihu_cup/question_train_set.txt','r')
f1=open(save_dir+'question_temp.txt','w')
f2=open(save_dir+'question_temp_test.txt','w')
line=f.readline()
while line!='':
    try:
        line = line.strip().split('\t')
        t = line[1].strip().split(',')

        temptitle=''
        temptext=''
        title_size=0
        text_size=0

        if len(line)>=2:
            for ch in t:
                try:
                    n=char_model[ch]
                except:
                    n=1
                temptitle=temptitle+ '%d' % n +' '
                title_size+=1
                if title_size==title_size_maxlength:
                    break

        if len(line)>=4:
            t = line[3].strip().split(',')
            for ch in t:
                try:
                    n = char_model[ch]
                except:
                    n = 1
                temptext = temptext + '%d' % n + ' '
                text_size += 1
                if text_size==text_size_maxlength:
                    break

        tempnum = title_size_maxlength - title_size
        for i in range(tempnum):
            temptitle = temptitle + '0 '
            title_size += 1
        tempnum = text_size_maxlength - text_size
        for i in range(tempnum):
            temptext = temptext + '0 '
            text_size += 1
        temp=temptitle+temptext
        sentence_size=title_size+text_size

        max_sentence_size=max(max_sentence_size,sentence_size)

        topic=topic_map[line[0]]
        topic=topic.strip().split(',')
        for i in topic:
            temp=temp+' __label__%d' % label_map[int(i)]
        temp+= '\n'

        # f1.write(temp)
        if random.random()<train_proportion:
            f1.write(temp)
            train_num+=1
        else:
            f2.write(temp)
            test_num+=1
    except:
        line = f.readline()
        continue

    line = f.readline()

    num+=1
    if(num%10000==0):
        print('finish:\t%f%%' % (num/max_example*100))
    # if num==int(max_example*train_proportion):
    #     train_num=num
    #     f1.close()
    #     f1=open(save_dir+'question_temp_test.txt','w')
f.close()
f1.close()
f2.close()
# test_num=num-train_num

#######         保存所有重要模型参数            ########
f=open(save_dir+'info.txt', 'w')
f.write('%d %d %d\n' % (len(char_array),char_num,char_size))
f.write('%d\n' % (len(label_array)))
f.write('%d %d %d\n' % (max_sentence_size,train_num,test_num))
f.close()