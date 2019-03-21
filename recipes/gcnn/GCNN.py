#!/usr/bin/env python3

import os
import sys

import torch
from torch import nn
from torch.nn import functional as F

class GatedCNN (nn.Module):


    def __init__ (self, parameter):

        super (GatedCNN, self).__init__ ()
        self.parameter = parameter

        # embedding variable
        self.embedding = nn.Embedding (parameter['vocab_size'], parameter['embed_dim'])

        # 2 CNN layers
        self.cnns = nn.ModuleList ([
                nn.Conv2d (
                    1, 
                    parameter['channel_out'], 
                    (parameter['kernel_height'], parameter['embed_dim']),
                    bias=True # is it means we dont have to add a specific bias variable?
                ),
                nn.Conv2d (
                    1, 
                    parameter['channel_out'], 
                    (parameter['kernel_height'], parameter['embed_dim']),
                    bias=True
                ),
            ])

        self.do = nn.Dropout (0.2)

        # since we can have multiple channel, we use avg pool layer
        self.pool = nn.AvgPool2d ((parameter['channel_out'], parameter['max_seq']))
        self.fc = nn.Linear (parameter['channel_out'], parameter['vocab_size'])
        self.softmax = nn.Softmax (dim=2)


    def forward (self, x):
        # x --> (batch, seq)
        # pass to embedding
        embed = self.embedding (x)
        # embed --> (batch, seq, embed_dim)

        embed = embed.unsqueeze (1)
        # embed --> (batch, 1, seq, embed_dim)

        # padding first
        embed = F.pad (embed, (0, 0, self.parameter['kernel_height'] - 1, 0), 'constant')

        # pass to all CNNS
        h, g = self.cnns[0] (embed), self.cnns[1] (embed)
        # h, g --> (batch, channel_out, seq, 1)
        h = h.squeeze (-1) # h --> (batch, channel_out, seq)
        g = g.squeeze (-1) # g --> (batch, channel_out, seq)

        # dropout
        h = self.do (h)
        g = self.do (g)
        # h, g --> (batch, channel_out, seq)

        # operate on h g
        out = h * torch.sigmoid (g) # (batch, channel_out, seq)
        out = out.permute (0, 2, 1) # (batch, seq, channel_out)

        # get to fc
        out = self.fc (out)
        out = self.softmax (out)
        # out -->  (batch, seq, vocab_size)
        return out
