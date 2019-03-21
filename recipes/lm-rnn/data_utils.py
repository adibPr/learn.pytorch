#!/usr/bin/env python

import torch
import os
import nltk
from collections import defaultdict

class Vocab (object):


    def __init__ (self):
        self.stoi = {}
        self.itos = {}
        self.sfreq = defaultdict (int)
        self.ctr = 0

    def fit (texts):
        if type (texts) is str:
            texts = nltk.word_tokenize (texts)
        for word in texts:
            self.add_word (word)

        # add unk
        self.add_word ('<UNK>')

    def add_word (self, word):
        if word not in self.stoi:
            self.itos[self.ctr] = word
            self.stoi[word] = self.ctr
            self.ctr += 1
        self.sfreq[word] += 1

    def limit (self, limit):
        if limit < len (self.stoi):
            sorted_sfreq = sorted (self.sfreq.items (), key=lambda kv: kv[0], reverse=True)
            negated_sfreq = sorted_sfreq[limit:]
            for (k,v) in negated_sfreq:
                self.sfreq.pop (k, None)
                idx = self.stoi.pop (k)
                self.itos.pop (idx)

        # add unk
        self.add_word ('<UNK>')

class Corpus (object):
    

    def __init__ (self):
        self.vocab = Vocab ()

    def fit (self, path, batch_size=20, limit=None):
        # build vocab
        with open (path, 'r') as f_buff:
            tot_tokens = 0
            data = f_buff.read ().splitlines ()[0]
            sentences = nltk.sent_tokenize (data)
            for sent in sentences:
                words = nltk.word_tokenize (sent) + ['<EOS>']
                for word in words:
                    self.vocab.add_word (word)
                    tot_tokens += 1

        text_tensor = torch.Tensor (tot_tokens)

        if limit:
            self.vocab.limit (limit)

        # convert to tensor
        with open (path, 'r') as f_buff:
            token_idx = 0
            data = f_buff.read ().splitlines ()[0]
            sentences = nltk.sent_tokenize (data)
            for sent in sentences:
                words = nltk.word_tokenize (sent) + ['<EOS>']
                for word in words:
                    this_word_index = self.vocab.stoi.get (word, None)
                    if this_word_index is None:
                        this_word_index = self.vocab.stoi['<UNK>']
                    text_tensor[token_idx] = this_word_index
                    token_idx += 1

        text_tensor = text_tensor.to (torch.long)
        return self.make_batch (text_tensor, batch_size)

    def make_batch (self, tensor, batch_size):
        num_batches = tensor.size(0) // batch_size # tot batches
        tensor = tensor[:num_batches*batch_size] # remove spill over
        return tensor.view(batch_size, -1) # convert into (batch size, max_seq)


if __name__ == '__main__':
    cp = Corpus ()
    tensor = cp.fit ('./data/meditations.mb.txt')
    print (len (cp.vocab.stoi))
    print (tensor.shape)
