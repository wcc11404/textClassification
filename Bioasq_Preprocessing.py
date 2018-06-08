import os
from collections import Counter
import pickle
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import shutil
import matplotlib.pyplot as plt

max_length=360
embedding_num=1701041
embedding_size=200
max_train_data=13486072
label_num=28340
max_abstract_perfile=20000
f_in_name="D:/wang/"
f_out_name=f_in_name+"out1/"

def process_line(line):
    line=line.split("\":")
    dict={}

    # dict['journal']=line[1][1:-12]

    t=line[2][2:-8]
    dict['meshMajor']=list(t.split("\",\""))

    # if line[3][0]=='\"':
    #     dict['year'] = line[3][1:-15]
    # else:
    #     dict['year'] = line[3][0:-15]

    dict['abstractText']=line[4][1:-7]

    # dict['pmid']=line[5][1:-8]

    dict['title']=line[6][1:-1]

    return dict

def getLineIter():
    with open(f_in_name + 'allMeSH_2018.json', 'r', encoding="utf-8", errors='ignore') as f:
        str = f.readline()
        str = f.readline()
        line_num=0
        while str != '':
            str = str.strip()[1:-2]
            try:
                dict = process_line(str)
                # print(dict)
            except:
                print('shit ' + str)
            line_num+=1

            yield dict,line_num
            str = f.readline()

def getLineIter2():
    with open(f_in_name + 'allMeSH_2018_process.txt', 'r', encoding="utf-8") as f:
        str = f.readline()
        line_num = 0
        while str != '':
            line_num += 1

            yield str, line_num
            str = f.readline()

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
    tt=[':',',','.','-','(',')',';','<','>','?','[',']','+','{','}','&','*','/','=','\'s','%'
        ,'0.1','0.01','0.001','0.0001','>=','<=']
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
        title_temp=word_tokenize(title.lower())
        title_temp=[word for word in title_temp if word not in list_stopwords and word not in ttt]

        abstract = line['abstractText']
        abstract_temp = word_tokenize(abstract.lower())
        abstract_temp=[word for word in abstract_temp if word not in list_stopwords and word not in tt]

        title_len.append(len(title_temp))
        abstract_len.append(len(abstract_temp))

        count.update(title_temp)
        count.update(abstract_temp)

        f.write(' '.join(abstract_temp))
        f1.write(' '.join(title_temp))

        if num%5000==0:
            print('finished %f%%'% (num/max_train_data*100))

    f.close()
    f1.close()

    with open(f_in_name + 'model/all_abstact_word.pik', 'wb') as data_f:
        pickle.dump(count, data_f)
    with open(f_in_name + 'model/title_len.pik', 'wb') as data_f:
        pickle.dump(title_len, data_f)
    with open(f_in_name + 'model/abstract_len.pik', 'wb') as data_f:
        pickle.dump(abstract_len, data_f)

def process_title_abstract(title_abstract,map,cache_embed):
    array = []
    for i,word in enumerate(title_abstract):
        if word in cache_embed:
            n = cache_embed[word]
        elif word in map:
            n = list(float(w) for w in map[word].split(' '))
        else:
            n = cache_embed['random_embed']

        array.append(n)

        if i==max_length-1: #截断字符
            break

    length = max_length - len(array)
    for i in range(length):
        n = cache_embed['zero_embed']
        array.append(n)  # 补齐字符

    return array

def process_abstract_main():
    with open(f_in_name + 'model/embedding.pik', 'rb') as data_f:
        map=pickle.load(data_f)
    with open(f_in_name + 'model/all_abstact_word.pik', 'rb') as f:
        count=pickle.load(f)

    if not os.path.exists(f_out_name + 'train_data_x'):
        os.makedirs(f_out_name + 'train_data_x')
    else:
        shutil.rmtree(f_out_name + 'train_data_x')
        os.makedirs(f_out_name + 'train_data_x')

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
    abstract_array = []
    File_num = 0
    for line,num in iter:
        line=line.strip().split(' ')
        title_abstract = process_title_abstract(line, map, cache_embed)
        abstract_array.append(title_abstract)

        if num % max_abstract_perfile == 0:
            with open(f_out_name + 'train_data_x/data_%d' % File_num, 'wb') as data_f:
                pickle.dump(abstract_array,data_f)
            # bcolz.carray(abstract_array, rootdir=f_out_name + 'train_data_x/data_%d' % File_num, mode='w')
            print("saved file %d/%d" % (File_num,13486072//max_abstract_perfile+1))
            File_num += 1
            del abstract_array
            abstract_array = []

    if len(abstract_array)!=0:
        with open(f_out_name + 'train_data_x/data_%d' % File_num, 'wb') as data_f:
            pickle.dump(abstract_array, data_f)
        # bcolz.carray(abstract_array, rootdir=f_out_name + 'train_data_x/data_%d' % File_num, mode='w')
        print("saved file %d/%d" % (File_num, max_train_data // max_abstract_perfile + 1))

def one_hot(x):
    result=[0 for _ in range(label_num)]
    for i in x:
        result[i]=1
    return result

def process_meshMajor():
    with open(f_in_name + 'model/label_map.pik', 'rb') as data_f:
        label_map=pickle.load(data_f)
    if not os.path.exists(f_out_name + 'train_data_y'):
        os.makedirs(f_out_name + 'train_data_y')
    else:
        shutil.rmtree(f_out_name + 'train_data_y')
        os.makedirs(f_out_name + 'train_data_y')
    # print(len(label_map))#28340

    iter = getLineIter()
    File_num = 0
    label_array=[]
    for line,num in iter:
        meshMajor = line['meshMajor']
        temparray=[]
        for label in meshMajor:
            try:
                temparray.append(label_map[label])
            except:
                print('loss '+ label)
        temparray=one_hot(temparray)
        label_array.append(temparray)

        if num % max_abstract_perfile==0:
            with open(f_out_name + 'train_data_y/data_%d' % File_num, 'wb') as data_f:
                pickle.dump(label_array,data_f)
            # bcolz.carray(a, rootdir='D:/wang/train_data_y/data_0', mode='w')
            print("saved file %d/%d" % (File_num, max_train_data // max_abstract_perfile + 1))
            File_num += 1
            del label_array
            label_array=[]

    if len(label_array)!=0:
        with open(f_out_name + 'train_data_y/data_%d' % File_num, 'wb') as data_f:
            pickle.dump(label_array, data_f)
        print("saved file %d/%d" % (File_num, max_train_data // max_abstract_perfile + 1))

def process_meshMajor_main():
    count=Counter()
    if not os.path.exists(f_in_name + 'model'):
        os.makedirs(f_in_name + 'model')

    iter=getLineIter()
    for line,num in iter:
        meshMajor = line['meshMajor']
        count.update(meshMajor)

        if num%500000==0:
            print('finished %f%%'% (num/max_train_data*100))

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
    with open(f_out_name+'train_data_y/data_0', 'rb') as f:
        temp_data_x = pickle.load(f)
    with open(f_out_name+'train_data_y/data_0', 'rb') as f:
        temp_data_y = pickle.load(f)

    print('finished')
    while True:
        pass

def hist_title_abstract():
    with open(f_in_name + 'model/title_len.pik', 'rb') as data_f:
        title_len=pickle.load(data_f)
    with open(f_in_name + 'model/abstract_len.pik', 'rb') as data_f:
        abstract_len=pickle.load(data_f)

    plt.hist(title_len,bins=500)
    plt.hist(abstract_len,bins=500)
    plt.show()

def main():
    # process_meshMajor_main() #预统计label信息，存储label编码模型
    # process_meshMajor()      #替换所有label为其编码，并分批存储成pickle

    # readEmbedding()          #预统计embedding信息，存储embedding模型，map['word']='str'形式
    all_abstract_word()        #预统计所有title和abstract的单词信息，分词，去除停用词，标点符号，存储到文件，并用counter统计，存储处理后的word
    # process_abstract_main()  #处理上一步处理后的word，将所有word转换成embedding（float形式），分批存储成pickle

    # load_xydata0()           #测试读取xy数据后内存占用大小
    # hist_title_abstract()       #观察title和abstract长度直方图

if __name__ == '__main__':
    main()