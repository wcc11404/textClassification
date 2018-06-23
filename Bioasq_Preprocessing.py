import os
from collections import Counter
import pickle
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import shutil
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import numpy as np
import datetime

max_abstract_length=300
max_title_length=25
embedding_num=1701041
embedding_size=200
max_data=13486072
test_data_interval=400
label_num=28340
max_abstract_perfile=40000
desktop=2
if desktop==1:
    f_in_name="D:/wang/"
    f_out_name=f_in_name+"out/"
elif desktop==2:
    f_in_name="D:/bioasq2018/"
    f_out_name="E:/D盘数据备份/out/"

def process_line(line):
    line=line[11:-1]
    dict = {}

    line=line.split("\",\"meshMajor\":[\"")
    dict['journal']=line[0]

    line=line[1].split("\"],\"year\":")
    dict['meshMajor']=list(line[0].split("\",\""))

    if line[1][0]=='\"':
        line=line[1].split('\",\"abstractText\":\"')
        dict['year'] = line[0][1:]
    elif line[1][0]=='n':
        line = line[1].split(',\"abstractText\":\"')
        dict['year'] = line[0]

    line=line[1].split('\",\"pmid\":\"')
    dict['abstractText']=line[0]

    line=line[1].split('\",\"title\":')
    dict['pmid']=line[0]

    if line[1][0]=='\"':
        dict['title']=line[1][1:-1]
    else:
        dict['title']=line[1]

    return dict

def getLineIter():
    with open(f_in_name + 'allMeSH_2018.json', 'r', encoding="utf-8", errors='ignore') as f:
        str = f.readline()
        str = f.readline()
        line_num=0
        while str != '':
            str = str.strip()[1:-1]
            try:
                dict = process_line(str)
                # print(dict)
            except:
                print('shit ' + str)
            line_num+=1

            yield dict,line_num
            str = f.readline()

def getLineIter2():
    f = open(f_in_name + 'allMeSH_2018_process_abstract.txt', 'r', encoding="utf-8")
    f1 = open(f_in_name + 'allMeSH_2018_process_title.txt', 'r', encoding="utf-8")
    str = f.readline()
    str1 = f1.readline()
    line_num = 0
    while str != '':
        line_num += 1
        yield str, str1, line_num
        str = f.readline()
        str1 = f1.readline()
    f.close()
    f1.close()

def readEmbedding():
    if not os.path.exists(f_in_name + 'model'):
        os.makedirs(f_in_name + 'model')

    map = {}
    with open(f_in_name + "word2vecTools/types.txt", 'r', encoding='utf-8') as f:
        with open(f_in_name + "word2vecTools/vectors.txt", 'r', errors='ignore') as vec:
            str = f.readline()
            vector = vec.readline()
            while str != "":
                map[str.strip()] = vector
                str = f.readline()
                vector = vec.readline()

    # print(len(map))#1701041
    map['zero_embed'] = " ".join(['0' for _ in range(embedding_size)])
    map['random_embed'] = " ".join(['%f' % random.uniform(-0.25,0.25) for _ in range(embedding_size)])
    with open(f_in_name + 'model/embedding.pik', 'wb') as data_f:
        pickle.dump(map, data_f)

def all_abstract_word():
    tt=[':',',','.','-','(',')',';','<','>','?','[',']','+','{','}','&','*','/','=','\'s','%','>=','<=']
    ttt=['.',',',':','(',')','?','\'s']
    list_stopwords=list(set(stopwords.words('english')))
    count=Counter()
    iter = getLineIter()
    title_len=[]
    abstract_len=[]
    f=open(f_in_name+'allMeSH_2018_process_abstract.txt',mode='w',encoding='utf-8')
    f1=open(f_in_name+'allMeSH_2018_process_title.txt',mode='w',encoding='utf-8')
    for line, num in iter:
        title = line['title']
        if title=='null':
            title=''
        title_temp=word_tokenize(title.lower())
        title_temp=[word for word in title_temp if word not in list_stopwords and word not in ttt]

        abstract = line['abstractText']
        abstract_temp = word_tokenize(abstract.lower())
        abstract_temp=[word for word in abstract_temp if word not in list_stopwords and word not in tt]

        title_len.append(len(title_temp))
        abstract_len.append(len(abstract_temp))

        count.update(title_temp)
        count.update(abstract_temp)

        str = ' '.join(abstract_temp)
        str += '\n'
        f.write(str)
        str = ' '.join(title_temp)
        str += '\n'
        f1.write(str)

        if num%5000==0:
            print('finished %f%%' % (num / max_data * 100))

    f.close()
    f1.close()

    with open(f_in_name + 'model/all_abstract_word.pik', 'wb') as data_f:
        pickle.dump(count, data_f)
    with open(f_in_name + 'model/title_len.pik', 'wb') as data_f:
        pickle.dump(title_len, data_f)
    with open(f_in_name + 'model/abstract_len.pik', 'wb') as data_f:
        pickle.dump(abstract_len, data_f)

def process_title_abstract(abstract,title,map,cache_embed):
    array = []

    #处理标题
    for i,word in enumerate(title):
        if word in cache_embed:
            n = cache_embed[word]
        elif word in map:
            n = list(float(w) for w in map[word].split(' '))
        else:
            n = cache_embed['random_embed']

        array.append(n)

        if i == max_title_length - 1:  # 截断字符
            break

    length = max_title_length - len(array)
    for i in range(length):
        n = cache_embed['zero_embed']
        array.append(n)  # 补齐字符

    #处理摘要
    for i,word in enumerate(abstract):
        if word in cache_embed:
            n = cache_embed[word]
        elif word in map:
            n = list(float(w) for w in map[word].split(' '))
        else:
            n = cache_embed['random_embed']

        array.append(n)

        if i == max_abstract_length - 1:  # 截断字符
            break

    length = max_title_length + max_abstract_length - len(array)
    # length = max_abstract_length - len(array)
    for i in range(length):
        n = cache_embed['zero_embed']
        array.append(n)  # 补齐字符

    return array

def process_abstract_main():
    with open(f_in_name + 'model/embedding.pik', 'rb') as data_f:
        map=pickle.load(data_f)
    with open(f_in_name + 'model/all_abstract_word.pik', 'rb') as f:
        count=pickle.load(f)

    if not os.path.exists(f_out_name + 'train_data_x'):
        os.makedirs(f_out_name + 'train_data_x')
    else:
        shutil.rmtree(f_out_name + 'train_data_x')
        os.makedirs(f_out_name + 'train_data_x')

    if not os.path.exists(f_out_name + 'test_data_x'):
        os.makedirs(f_out_name + 'test_data_x')
    else:
        shutil.rmtree(f_out_name + 'test_data_x')
        os.makedirs(f_out_name + 'test_data_x')

    #建立embed快速缓存
    cache_embed={}
    random_embed = list(float(w) for w in map['random_embed'].split(' '))
    zero_embed = list(float(w) for w in map['zero_embed'].split(' '))
    cache_embed['random_embed']=random_embed
    cache_embed['zero_embed']=zero_embed
    i=0
    for tuplee in count.most_common(2000000):
        word, _ = tuplee
        if word in map:
            cache_embed[word]=list(float(w) for w in map[word].split(' '))
            i+=1
        # else:
        #     print(word)
        if i==500000:
            break
    print('finished cache_embed')
    del count

    iter=getLineIter2()
    train_array = []
    test_array = []
    File_num = 0
    for line_abstract, line_title, num in iter:
        line_abstract = line_abstract[:-1].strip().split(' ')   #-1是为了去除最后一个\n字符
        line_title = line_title[:-1].strip().split(' ')
        title_abstract = process_title_abstract(line_abstract, line_title, map, cache_embed)

        if num % test_data_interval == 0:
            test_array.append(title_abstract)
        else:
            train_array.append(title_abstract)

        if len(train_array) == max_abstract_perfile:
            with open(f_out_name + 'train_data_x/data_%d' % File_num, 'wb') as data_f:
                pickle.dump(train_array,data_f)
            print("saved file %d/%d" % (File_num + 1,
                        (max_data - max_data // test_data_interval) // max_abstract_perfile + 1))  # 测试数据被刨除出去
            File_num += 1
            del train_array
            train_array = []

    if len(train_array)!=0:
        with open(f_out_name + 'train_data_x/data_%d' % File_num, 'wb') as data_f:
            pickle.dump(train_array, data_f)
        print("saved file %d/%d" % (File_num + 1,
                        (max_data - max_data // test_data_interval) // max_abstract_perfile + 1))  # 测试数据被刨除出去

    with open(f_out_name + 'test_data_x/data_0', 'wb') as data_f:
        pickle.dump(test_array, data_f)

def one_hot(x):
    # result=[0 for _ in range(label_num)]
    # for i in x:
    #     result[i]=1
    # return result

# t=np.array([0,0,0])
# a=np.array([1,3,5])
# b=np.array([1,1,1])
# c=sparse.csr_matrix((b,(t,a)),shape=(1,6)).toarray().reshape((6))
# print(c)

    # t=np.array([0 for i in x])
    # a=np.array(x)
    # b=np.array([1 for i in x])
    # result=[]
    # result.append(t)
    # result.append(a)
    # result.append(b)
    result=x
    # result=np.array(x)
    return result

def process_meshMajor():
    with open(f_in_name + 'model/label_map.pik', 'rb') as data_f:
        label_map=pickle.load(data_f)

    if not os.path.exists(f_out_name + 'train_data_y'):
        os.makedirs(f_out_name + 'train_data_y')
    else:
        shutil.rmtree(f_out_name + 'train_data_y')
        os.makedirs(f_out_name + 'train_data_y')

    if not os.path.exists(f_out_name + 'test_data_y'):
        os.makedirs(f_out_name + 'test_data_y')
    else:
        shutil.rmtree(f_out_name + 'test_data_y')
        os.makedirs(f_out_name + 'test_data_y')
    # print(len(label_map))#28340

    iter = getLineIter()
    File_num = 0
    train_array=[]
    test_array=[]
    for line,num in iter:
        meshMajor = line['meshMajor']
        temparray=[]
        for label in meshMajor:
            try:
                temparray.append(label_map[label])
            except:
                print('loss '+ label)
        temparray=one_hot(temparray)

        if num % test_data_interval == 0:
            test_array.append(temparray)
        else:
            train_array.append(temparray)

        if len(train_array) == max_abstract_perfile:
            with open(f_out_name + 'train_data_y/data_%d' % File_num, 'wb') as data_f:
                pickle.dump(train_array,data_f)
            print("saved file %d/%d" % (File_num + 1,
                        (max_data - max_data // test_data_interval) // max_abstract_perfile + 1))  # 测试数据被刨除出去
            File_num += 1
            del train_array
            train_array=[]

    if len(train_array)!=0:
        with open(f_out_name + 'train_data_y/data_%d' % File_num, 'wb') as data_f:
            pickle.dump(train_array, data_f)
        print("saved file %d/%d" % (File_num + 1,
                        (max_data - max_data // test_data_interval) // max_abstract_perfile + 1))  # 测试数据被刨除出去

    with open(f_out_name + 'test_data_y/data_0', 'wb') as data_f:
        pickle.dump(test_array, data_f)

def process_meshMajor_main():
    count=Counter()
    if not os.path.exists(f_in_name + 'model'):
        os.makedirs(f_in_name + 'model')

    iter=getLineIter()
    for line,num in iter:
        meshMajor = line['meshMajor']
        count.update(meshMajor)

        if num%500000==0:
            print('finished %f%%' % (num / max_data * 100))

    # print(count.most_common())
    label_map = {}
    label_array = []
    for j, tuplee in enumerate(count.most_common()):
        label, _ = tuplee
        label_map[label] = j
        label_array.append(label)
    with open(f_in_name + 'model/label_map.pik', 'wb') as data_f:
        pickle.dump(label_map, data_f)
    with open(f_in_name + 'model/label_array.pik', 'wb') as data_f:
        pickle.dump(label_array, data_f)

def load_xydata0():
    starttime = datetime.datetime.now()
    with open(f_out_name+'train_data_y/data_0', 'rb') as f:
        temp_data_x = pickle.load(f)
    with open(f_out_name+'train_data_y/data_0', 'rb') as f:
        temp_data_y = pickle.load(f)

    tt=[]
    for i in temp_data_y:
        a=np.array([0 for j in i])
        b=np.array([1 for j in i])
        tt.append(sparse.csr_matrix((b,(a,i)),shape=(1,label_num)).toarray().reshape((label_num)))
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)

    del temp_data_y

    while True:
        pass

def hist_title_abstract():
    with open(f_in_name + 'model/title_len.pik', 'rb') as data_f:
        title_len=pickle.load(data_f)
    with open(f_in_name + 'model/abstract_len.pik', 'rb') as data_f:
        abstract_len=pickle.load(data_f)

    temp=abstract_len
    max_len=0
    min_len=500
    sum=0
    num=0
    for i in temp:
        if i>max_len:
            max_len=i
        if i<min_len:
            min_len=i
        if i>300:
            num+=1
        sum+=i
    print(max_len,min_len,sum/len(temp),num,len(temp))
    plt.hist(temp,bins=500)
    plt.show()

def hist_MeshMajor():
    arr=[]
    sum=0
    for i in range(75):
        with open(f_out_name+'train_data_y/data_%d' % i, 'rb') as f:
            temp_data_y = pickle.load(f)
        for label in temp_data_y:
            arr.append(len(label))
            sum+=len(label)
    print(sum/len(arr)) #75:13  338:12.68
    plt.hist(arr,bins=500)
    plt.show()

def main():
    # process_meshMajor_main() #预统计label信息，存储label编码模型
    process_meshMajor()      #替换所有label为其编码，并分批存储成pickle

    # readEmbedding()          #预统计embedding信息，存储embedding模型，map['word']='str'形式
    # all_abstract_word()        #预统计所有title和abstract的单词信息，分词，去除停用词，标点符号，存储到文件，并用counter统计，存储处理后的word
    process_abstract_main()  #处理上一步处理后的word，将所有word转换成embedding（float形式），分批存储成pickle

    # load_xydata0()           #测试读取xy数据后内存占用大小
    # hist_title_abstract()       #观察title和abstract长度直方图
    # hist_MeshMajor()            #观察标签数量直方图

if __name__ == '__main__':
    main()