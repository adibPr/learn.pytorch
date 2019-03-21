#!/usr/bin/env python

import os
import sys
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
path_data = os.path.join (path_this, 'data', 'meditations.mb.txt')

from data_utils import Corpus

device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')

embed_size = 128
hidden_size = 1024
num_layers = 1
num_epoch = 20
batch_size = 20
max_seq = 30
learning_rate = 0.002
max_vocab = None

corpus = Corpus ()
tensor = corpus.fit (path_data, limit=max_vocab)
vocab_size = len (corpus.vocab.stoi)
tot_batch = tensor.shape[1] // max_seq
# not yet?
# tensor = tensor[:, :max_seq * tot_batch] # remove spilled 

# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, states):
        # Embed word ids to vectors
        x = self.embed(x)
        
        # Forward propagate LSTM
        out, states = self.lstm(x, states)
        # out is the last layer, having shape of (batch_size, max_seq, hidden_size)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, states

model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

# Train the model
for epoch in range(num_epoch):
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))

    # i can do other way (by using epoch index)
    for i in range (tot_batch):
        i_start = i*max_seq

        # Get mini-batch inputs and targets
        inputs = tensor[:, i_start:i_start+max_seq].to(device)
        # the target is the next word (might error if the data is divided by max_seq
        targets = tensor[:, (i_start+1):(i_start+1)+max_seq].to(device)

        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        # clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if i % 10 == 0:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                   .format(epoch+1, num_epoch, i, tot_batch, loss.item(), np.exp(loss.item())))
    
num_samples = 1000 # total word to generate
# Test the model
with torch.no_grad():
    with open(os.path.join (path_this, 'output.txt'), 'w') as f:
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # Select one word id randomly
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # Forward propagate RNN 
            output, state = model(input, state)

            # Sample a word id
            prob = output.exp()
            word_id = output.argmax ().item ()
            # word_id = torch.multinomial(prob, num_samples=1).item()

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)

            # File write
            word = corpus.vocab.itos[word_id]
            word = '\n' if word.lower () == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))
