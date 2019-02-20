#!/usr/bin/env python

import os
import sys
import pdb
import random

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from char_generator import TextLoader
from GCNN import GatedCNN

parameter = {
        # training
        'batch_size' : 50,
        'epoch' : 100,
        'vocab_size' : None,

        # CNN-specific parameter
        'kernel_height' : 3,
        'embed_dim' : 300,
        'max_seq' : 50 ,
        'channel_out' : 3
    }

device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')
TL = TextLoader ('data', batch_size=parameter['batch_size'], seq_length=parameter['max_seq'])
parameter['vocab_size'] = TL.vocab_size

gcnn = GatedCNN (parameter)
gcnn.to (device)

criterion = nn.CrossEntropyLoss ()
learning_rate = 0.0005
optimizer = torch.optim.Adam (gcnn.parameters (), lr = learning_rate)


gcnn.train ()
past_loss = []
for e in range (parameter['epoch']):
    ctr = 0
    avg_loss = 0
    for x, y in tqdm (TL ()):
        x = torch.tensor (x).to (device)
        y = torch.tensor (y).to (device)

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

    avg_loss /= ctr
    if len (past_loss) >= 5:
        past_avg = sum (past_loss) / len (past_loss)
        past_loss.append (avg_loss)
        past_loss = past_loss[1:]
        curr_avg = sum (past_loss) / len (past_loss)

        print ("curr {:.2F}, past {:.2F}".format (curr_avg, past_avg))

        if curr_avg > past_avg:
            print ("Break point")
            break
    else : 
        past_loss.append (avg_loss)

    print ("Avg loss - {} : {}".format (e, avg_loss))

# try generate content
char = random.randint (0, parameter['vocab_size'])
print (TL.chars[char], end="")

gcnn.eval()
ctr = 0
with torch.no_grad ():
    while True:
        x = torch.tensor ([char]).unsqueeze (0).to (device)
        out = gcnn (x)
        out = out.squeeze (0).squeeze (0)
        char = torch.argmax (out)
        print (TL.chars[char], end="")
        ctr += 1

        if TL.chars[char] == '.' or ctr >= 100 :
            break
print ()
