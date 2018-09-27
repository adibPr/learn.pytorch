#!/usr/bin/env python

# sys module
import time 
import os
import sys

# third parties module
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# local module
path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))
from utils import progress_bar
from model import CNN

transformer = transforms.Compose ([
    transforms.Resize ((224, 224)),

    # to tensor
    transforms.ToTensor (),

    # Normalize args is (Mean ..), (std ..) for each channel
    transforms.Normalize ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# prepare data
dataset = torchvision.datasets.ImageFolder (root='./data', 
        transform=transformer)

# split into validation and train
# first, specify indices
indices = range (len (dataset)) # all indexes of dataset
validation_size = int (0.2 * len (dataset)) 
validation_idx = np.random.choice (indices, size=validation_size, replace=False)
train_idx = list (set (indices) - set (validation_idx))

# create sampler
train_sampler = torch.utils.data.SubsetRandomSampler (train_idx)
validation_sampler = torch.utils.data.SubsetRandomSampler (validation_idx)

# create loader
train_loader = torch.utils.data.DataLoader (dataset, batch_size=20,
        num_workers=2, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader (dataset, batch_size=20,
        num_workers=2, sampler=validation_sampler)


# for GPU optimization
device = torch.device ("cuda:0" if torch.cuda.is_available ()  else "cpu")
print ("Using ", device)

# create network
# net = CNN ()

# """
# use resnet34 pretrained with freeze model
net = torchvision.models.resnet34 (pretrained=True)
# freeze the network
for param in net.parameters () : 
    param.requires_grad = True

net.fc = nn.Linear (512, 80)
# """

net.to (device)

# optimization
criterion = nn.CrossEntropyLoss ()
optimizer = optim.Adam (net.fc.parameters (), lr=0.001)

# training network
print ("Training..")
for epoch in range (10) : 
    running_loss = 0
    tot_time = 0
    for i, data in enumerate (train_loader) : 
        t_start = time.time ()
        progress_bar (i+1, len (train_loader))

        inputs, labels = data
        # send to device
        inputs, labels = inputs.to (device), labels.to (device)

        # feed forward
        optimizer.zero_grad ()
        outputs = net (inputs)

        loss = criterion (outputs, labels)
        loss.backward ()
        optimizer.step ()

        tot_time += time.time () - t_start

    progress_bar (None)

    # validating
    with torch.no_grad () : 
        correct = 0
        total = 0
        for i, data in enumerate (validation_loader) : 
            progress_bar (i, len (validation_loader))
            inputs, labels = data
            inputs, labels = inputs.to (device), labels.to (device)

            predicted = net (inputs)
            _, predicted = torch.max (predicted.data, 1)

            correct += (predicted == labels).sum ().item ()
            total += labels.size (0)

    progress_bar (None)

    print ("epoch {},loss {:.2f}, acc {:.2F},{:.2F}/it".format (
        epoch + 1,
        loss.item (),
        correct/total,
        tot_time / len (train_loader)
    ))
print ('Finished Training')

# """
