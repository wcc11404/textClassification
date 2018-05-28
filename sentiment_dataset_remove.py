import numpy as np
import re
from tensorflow.contrib import learn
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

class dataset(object):
    def __init__(self):
        self.positive_data_file="./dataset/rt-polaritydata/rt-polarity.pos"
        self.negative_data_file="./dataset/rt-polaritydata/rt-polarity.neg"
        self.dev_sample_percentage=0.1              #测试集占总体的比例

        x,y=self.load_data_and_labels()             #加载数据集

        # Build vocabulary

        stoplist = set(', ! \\\\? \\? ? \\( \\\\( \\) \\\\)'.split(' '))
        texts = [[word for word in b.lower().split() if word not in stoplist] for b in x]
        self.max_document_length = max([len(x1) for x1 in texts])  # 获取最大语句的长度
        self.embedding_size = 100

        # model = Word2Vec.load("word.w2v")
        model=self.load_vocabulary("./dataset/word2vec/glove.6B/glove.6B.100d.txt",False)

        texts,self.vector=self.build_embedding(texts,model,self.embedding_size)

        self.vocabulary_size=len(self.vector)

        # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)#如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充
        # x = np.array(list(vocab_processor.fit_transform(x)))                #将每句话的每个单词通过词典替换成唯一编号，小于最大长度的部分用0填充

        # Randomly shuffle data
        # np.random.seed(10)
        # shuffle_indices = np.random.permutation(np.arange(len(y)))          #返回洗牌后的下标
        # x_shuffled = x[shuffle_indices]     #通过下标取值，打乱的数组
        # y_shuffled = y[shuffle_indices]
        x_shuffled=texts
        y_shuffled=y

        # Split train/test set
        dev_sample_index = -1 * int(self.dev_sample_percentage * float(len(y)))
        self.x_train, self.x_dev = np.array(x_shuffled[:dev_sample_index]), np.array(x_shuffled[dev_sample_index:])
        self.y_train, self.y_dev = np.array(y_shuffled[:dev_sample_index]), np.array(y_shuffled[dev_sample_index:])

    def load_vocabulary(self,dir="./glove.6B.200d.txt",isGoogle=False):
        if(isGoogle):
            return gensim.models.KeyedVectors.load_word2vec_format(dir, binary=True)
        embeddings_index = {}
        f = open(dir, 'r', encoding='utf-8',errors='ignore')
        for line in f:
            values = line.split()
            word = values[0]
            # if word not in word2idx:
            #     continue
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def build_embedding(self,texts,model,embedding_size):
        v=[]#v[0]=[0,0,0]
        v+=[[0 for i in range(embedding_size)]]
        i=1
        e={}#e['word']=0
        tt=[]#转换后的文本
        for sentences in texts:
            t=[]
            for word in sentences:
                try:
                    if word not in e:
                        v+=[model[word]]
                        e[word]=i
                        i+=1
                    t+=[e[word]]
                except:
                    t+=[0]
            for i in range(self.max_document_length - len(sentences)):
                t += [0]
            tt+=[t]
        return tt,v

    def clean_str(self,string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def load_data_and_labels(self):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        positive_examples = list(open(self.positive_data_file, "r",errors='ignore').readlines())    #按行读出，并排列成list
        positive_examples = [s.strip() for s in positive_examples]              #移除头尾空格
        negative_examples = list(open(self.negative_data_file, "r",errors='ignore').readlines())
        negative_examples = [s.strip() for s in negative_examples]

        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self.clean_str(sent) for sent in x_text]

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]       #生成y标签并合并
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)

        return x_text, y

    def next_batch(self,batch_size):
        """
        Generates a batch iterator for a dataset.
        """
        data_size = self.x_train.shape[0]
        s=np.random.randint(0,data_size-1,batch_size)
        return self.x_train[s],self.y_train[s]

    def batch_iter(self,batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        
        data = np.array(list(zip(self.x_train,self.y_train)))   #转换成[x,y]键值对的形式数组
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]       #乱序数组
            else:
                shuffled_data = data                        #正序数组
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield zip(*shuffled_data[start_index:end_index])

    def dev_dataset(self):
        return self.x_dev,self.y_dev

def main():
    data=dataset()

if __name__ == '__main__':
  main()