#!/usr/bin/env python

import os
import sys
import pdb

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from char_generator import TextLoader
from GCNN import GatedCNN

parameter = {
        # training
        'batch_size' : 50,
        'epoch' : 3,
        'vocab_size' : None,

        # CNN-specific parameter
        'kernel_height' : 3,
        'embed_dim' : 300,
        'max_seq' : 50 ,
        'channel_out' : 3
    }

TL = TextLoader ('data', batch_size=parameter['batch_size'], seq_length=parameter['max_seq'])
parameter['vocab_size'] = TL.vocab_size

gcnn = GatedCNN (parameter)

criterion = nn.CrossEntropyLoss ()
learning_rate = 0.0005
optimizer = torch.optim.Adam (gcnn.parameters (), lr = learning_rate)


for e in range (parameter['epoch']):
    ctr = 0
    avg_loss = 0
    for x, y in tqdm (TL ()):
        x = torch.tensor (x)
        y = torch.tensor (y)

        gcnn.zero_grad ()

        out = gcnn (x)
        out = out.view (-1, parameter['vocab_size'])
        # out --> (batch * seq, vocab_size)
        y = y.squeeze (-1).view (-1)
        # y --> (batch * seq)

        loss = criterion (out, y)
        loss.backward ()
        optimizer.step ()

        avg_loss += loss
        ctr += 1
    print ("Avg loss - {} : {}".format (e, avg_loss/ctr))
