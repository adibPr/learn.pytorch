"""
01_Resnet34_LearningRate.py
Implementation of Learning Rate finder and Cyclical Learning Rate Finder of fast.ai library
The whole model is the same with 01_Resnet34.py.
"""

# sys module
import os
import sys
import time
import math

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
import pandas as pd

# local module
path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))
from utils import progress_bar

sz = 224 # for size
PATH = '/home/paperspace/data/dogscats' # for data path

# need to see the image first, we did not normalize it.
# first declare transformation
train_transformer = transforms.Compose ([
    transforms.Resize ((sz,sz)),

    # random rotation by 10 degree
    transforms.RandomRotation (10),

    # random brightness
    transforms.ColorJitter (brightness=0.05), 

    # random horizontal flip
    transforms.RandomHorizontalFlip (),

    # to tensor
    transforms.ToTensor (),

    # Normalize args is (Mean ..), (std ..) for each channel
    # source for each value : https://pytorch.org/docs/stable/torchvision/models.html
    transforms.Normalize ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])
valid_transformer =  transforms.Compose ([
    transforms.Resize ((sz,sz)),

    # to tensor
    transforms.ToTensor (),

    # Normalize args is (Mean ..), (std ..) for each channel
    # source for each value : https://pytorch.org/docs/stable/torchvision/models.html
    transforms.Normalize ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

# load data
train_set = torchvision.datasets.ImageFolder (root=PATH+'/train', 
        transform=train_transformer)
valid_set = torchvision.datasets.ImageFolder (root=PATH+'/valid', 
        transform=valid_transformer)

# make loader
train_loader = torch.utils.data.DataLoader (train_set, batch_size=64, num_workers=4, shuffle=True)
valid_loader = torch.utils.data.DataLoader (valid_set, batch_size=64, num_workers=4, shuffle=True)


# then create learner of arch resnet34 with pretrained
device = torch.device (torch.device ("cuda") if torch.cuda.is_available ()  else "cpu")
print ("Using ", device)

net = torchvision.models.resnet34 (pretrained=True)

# freeze all layers
for param in net.parameters ():
    param.requires_grad  = False 

# add the last one fully connected layer
net.fc = nn.Linear (512, 2)

# optimization
criterion = nn.CrossEntropyLoss ().to (device)
optimizer = optim.SGD (net.fc.parameters (), lr=0.001)

# to cuda
net.to (device)


"""
The first task is to detect the best learning rate (i.e lr_finder in fast.ai)
The general idea is to start with very small learning rate, and increase it
linearly, and then plot it against the error. 
The moment the error decreament start slowing down, its a good sign for a good
learning rate

Pytorch provide a scheduler for this : 
    https://pytorch.org/docs/stable/optim.html?highlight=scheduler#how-to-adjust-learning-rate
all you need to do is to define scheduler, with parameter of its optimizer, and 
a function to update the LR.

"""

min_LR = 0.001
max_LR = 0.01
# step size is the step needed to reach min_LR to max_LR and vice versa
# in other words, with code below, it will take 4 epoch to form a triangle
step_size = 2*len (train_loader)  

def triangular_cyclical (epochCounter) : 
    # based o paper https://arxiv.org/pdf/1506.01186.pdf, the original paper
    cycle = math.floor (1 + epochCounter / (2  * step_size))
    x = abs (epochCounter / step_size - 2*cycle + 1)
    local_lr = min_LR + (max_LR - min_LR) * max (0, (1-x))

    # since local_lr is our target lr, but lambdaLr only support multiplicative
    # so we return multiplication factor
    return local_lr / 0.001

# learning scheduler
scheduler = optim.lr_scheduler.LambdaLR (
        optimizer, 
        triangular_cyclical
    )


# training network
print ("Training..")
for epoch in range (3) : 
    running_loss = 0

    tot_time = 0
    # load per minibatch
    for i, data in enumerate (train_loader) : 
        t_start = time.time ()

        progress_bar (i+1, len (train_loader))
        inputs, labels = data
        inputs, labels = inputs.to (device), labels.to (device)

        optimizer.zero_grad ()
        outputs = net (inputs)

        loss = criterion (outputs, labels)

        loss.backward ()
        optimizer.step ()
        scheduler.step ()

        tot_time += time.time () - t_start

    progress_bar (None)
    print ("{:2F} s/it for {} iterations".format (tot_time/len (train_loader), len (train_loader)))

    # check with validation
    with torch.no_grad () : 
        correct = 0
        total = 0

        for i, data in enumerate (valid_loader) : 
            progress_bar (i+1, len (valid_loader))
            inputs, labels = data
            inputs, labels = inputs.to (device), labels.to (device)

            predicted = net (inputs)
            _, predicted = torch.max (predicted.data, 1)

            correct += (predicted == labels).sum ().item ()
            total += labels.size (0)

        progress_bar (None)


    print ("epoch {}\tloss {:2f}\taccuracy {:2F}".format (
        epoch + 1,
        loss.item (),
        correct/total
    ))

print ('Finished Training')

"""
The error vs LR chart can be seen on 
    https://imgur.com/a/4vJbjyv
But there is something fishy about it. No matter how I change the LR, its form
is same. Perhaps its because the pretrained model and we just use the last fc.
But it should never matter though.
Apparently, if using precompute=False, the chart is stagged.

By using 2*epoch and cyclical learning rate, on the first epoch, we got 97.5, 
the best so far, compare to augmented vanila LR=0.01.
We vary the LR between 0.001 to 0.01. This of course could be modify to 
see the optimum range value.
"""
