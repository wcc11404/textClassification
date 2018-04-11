import numpy as np
import re
from tensorflow.contrib import learn
import itertools
from collections import Counter

class dataset(object):
    def __init__(self):
        self.positive_data_file="./dataset/rt-polaritydata/rt-polarity.pos"
        self.negative_data_file="./dataset/rt-polaritydata/rt-polarity.neg"
        self.dev_sample_percentage=0.1

        x,y=self.load_data_and_labels()

        # Build vocabulary
        max_document_length = max([len(x1.split(" ")) for x1 in x])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        dev_sample_index = -1 * int(self.dev_sample_percentage * float(len(y)))
        self.x_train, self.x_dev = np.array(x_shuffled[:dev_sample_index]), np.array(x_shuffled[dev_sample_index:])
        self.y_train, self.y_dev = np.array(y_shuffled[:dev_sample_index]), np.array(y_shuffled[dev_sample_index:])

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
        positive_examples = list(open(self.positive_data_file, "r",errors='ignore').readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(self.negative_data_file, "r",errors='ignore').readlines())
        negative_examples = [s.strip() for s in negative_examples]

        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self.clean_str(sent) for sent in x_text]

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
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
        
        data = np.array(list(zip(self.x_train,self.y_train)))
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield zip(*shuffled_data[start_index:end_index])

    def dev_dataset(self):
        return self.x_dev,self.y_dev