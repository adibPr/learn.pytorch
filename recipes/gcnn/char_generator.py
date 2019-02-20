'''
modified from https://github.com/sherjilozair/char-rnn-tensorflow
'''
import codecs
import os
import collections
import pickle
import numpy as np
import pdb

class TextLoader(object):
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "shakespere.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        """
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        """
        self.preprocess(input_file, vocab_file, tensor_file)
        self.create_batches()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()

        # since they not removing dot ('.') assume its a stop
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)

        # get dictionary {char, id}
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)

        # construct string to its index (all of it)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)


    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        # what we want is
        #  [[seq] [seq] ... [seq]]
        # one batch conist of batch size of seq_length
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # discard the leftover
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def __call__(self):
        for x, y in zip(self.x_batches, self.y_batches):
            yield x, np.expand_dims(y, -1)

if __name__ == '__main__' :
    TL = TextLoader ('data', batch_size=5, seq_length=50)
    """
    # x shape is (batch, seq) 
        --> embed ( batch, 1, seq, embed) 
        --> CNN_1  (batch, dim_out, h_out, 1) ~ (batch, dim_out, h_out)
        --> CNN_2 (batch, dim_out, h_out, 1) ~ (batch, dim_out, h_out)
        --> CNN_1 * sigmoid CNN_2 (batch, dim_out, h_out)
        --> fc (batch, tot_vocab)
    # y shape is (batch, seq, 1)
    """
