"""
01_Resnet34_LearningRate.py
Implementation of Learning Rate finder and Cyclical Learning Rate Finder of fast.ai library
The whole model is the same with 01_Resnet34.py.
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

# learning scheduler
scheduler = optim.lr_scheduler.LambdaLR (
        optimizer, 
        lambda step : 0.01 + (step * 0.01) # starts with 0.01, and increase 0.01 per step 
    )

errors = []  # accumluate all error in one place
LR = []

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

        errors.append (loss)
        LR.append (scheduler.get_lr ())
        scheduler.step ()
        print (LR[-1])
        sys.exit ()


    progress_bar (None)
    print ("{:2F} s/it for {} iterations".format (tot_time/len (train_loader), len (train_loader)))

print ('Finished Training')

