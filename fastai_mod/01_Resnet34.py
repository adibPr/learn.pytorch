"""
01_Resnet34.py
To replicate result of chapter 1, minute 20:44 of fast.ai lesson 1 video.
We first try to use our own approach, then compare it with fast.ai and improve it
"""

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

"""
The general idea is to use resnet34 pretrained model, with data resized to
224x224.
"""

sz = 224 # for size
PATH = '/media/Linux/Learn/PyTorch/dataset/dogscats' # for data path

# first declare transformation
my_transformer = transforms.Compose ([
    transforms.Resize ((sz,sz)),

    # to tensor
    transforms.ToTensor (),

    # Normalize args is (Mean ..), (std ..) for each channel
    transforms.Normalize ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# load data
train_set = torchvision.datasets.ImageFolder (root=PATH+'/train', 
        transform=my_transformer)
valid_set = torchvision.datasets.ImageFolder (root=PATH+'/valid', 
        transform=my_transformer)

# make loader
train_loader = torch.utils.data.DataLoader (train_set, batch_size=100, num_workers=2)
valid_loader = torch.utils.data.DataLoader (valid_set, batch_size=100, num_workers=2)
        

# then create learner of arch resnet34 with pretrained
net = torchvision.models.resnet34 (pretrained=True)

# freeze all layers
for param in net.parameters ():
    param.require_grad  = False

# add the last one fully connected layer
net.fc = nn.Linear (512, 2)

# optimization
criterion = nn.CrossEntropyLoss ()
optimizer = optim.Adam (net.parameters (), lr=0.01)

# training network
print ("Training..")
for epoch in range (3) : 
    running_loss = 0

    # load per minibatch
    for i, data in enumerate (train_loader) : 
        inputs, labels = data
        optimizer.zero_grad ()
        outputs = net (inputs)
        loss = criterion (outputs, labels)
        loss.backward ()
        optimizer.step ()

    # check with validation
    with torch.no_grad () : 
        correct = 0
        total = 0
        for i, data in enumerate (validation_loader) : 
            inputs, labels = data
            predicted = net (inputs)
            _, predicted = torch.max (predicted.data, 1)

            correct += (predicted == labels).sum ().item ()
            total += labels.size (0)


    print ("epoch {}\tloss {:2f}\taccuracy {:2F}".format (
        epoch + 1,
        loss.item (),
        correct/total
    ))
print ('Finished Training')
