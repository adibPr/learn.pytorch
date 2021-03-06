"""
01_Resnet34.py
To replicate result of chapter 1, minute 20:44 of fast.ai lesson 1 video.
We first try to use our own approach, then compare it with fast.ai and improve it
"""

# sys module
import os
import sys
import time

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

"""
The general idea is to use resnet34 pretrained model, with data resized to
224x224.
"""

sz = 224 # for size
PATH = '/home/paperspace/data/dogscats' # for data path

# first declare transformation
my_transformer = transforms.Compose ([
    transforms.Resize ((sz,sz)),

    # to tensor
    transforms.ToTensor (),

    # Normalize args is (Mean ..), (std ..) for each channel
    # source for each value : https://pytorch.org/docs/stable/torchvision/models.html
    transforms.Normalize ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

# load data
train_set = torchvision.datasets.ImageFolder (root=PATH+'/train', 
        transform=my_transformer)
valid_set = torchvision.datasets.ImageFolder (root=PATH+'/valid', 
        transform=my_transformer)

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
optimizer = optim.SGD (net.fc.parameters (), lr=0.01)

# to cuda
net.to (device)

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
Up until here we got 97.1 accuracy for the first epoch, which is a little
different with fast.ai with 98.97 accuracy. But it got it with only 20 s.
We need like 5 minute on it. 

Its mainly because in fast.ai library, they use precalculated scheme that
store its temporary result from unchanged network, so only calculation of the last
layer involved.
"""
