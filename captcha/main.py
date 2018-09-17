import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from model import CNN

transformer = transforms.Compose ([
    # resize the 
    transforms.Resize ((20, 20)),
    
    # convert into grayscale 1 channel
    transforms.Grayscale (1),

    # to tensor
    transforms.ToTensor (),

    # Normalize args is (Mean ..), (std ..) for each channel
    transforms.Normalize ([0.5], [0.5])
])

# prepare data
trainset = torchvision.datasets.ImageFolder (root='./data/letter/train', 
        transform=transformer)
testset = torchvision.datasets.ImageFolder (root='./data/letter/test', 
        transform=transformer)

# create loader
trainloader = torch.utils.data.DataLoader (trainset, batch_size=20, 
        shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader (testset, batch_size=20, 
        shuffle=True, num_workers=2)

# create network
net = CNN ()

# optimization
criterion = nn.CrossEntropyLoss ()
optimizer = optim.Adam (net.parameters (), lr=0.001)

# training network
print ("Training..")
for epoch in range (3) : 
    running_loss = 0
    for i, data in enumerate (trainloader) : 
        inputs, labels = data
        optimizer.zero_grad ()
        outputs = net (inputs)
        loss = criterion (outputs, labels)
        loss.backward ()
        optimizer.step ()

        print ("epoch {}, batch {}, loss {:2f}".format (
            epoch + 1,
            i + 1, 
            loss.item ()
        ))
print ('Finished Training')

# testing network
print ("Testing")
with torch.no_grad () : 
    correct = 0
    total = 0
    for i, data in enumerate (testloader) : 
        inputs, labels = data
        predicted = net (inputs)
        _, predicted = torch.max (predicted.data, 1)

        correct += (predicted == labels).sum ().item ()
        total += labels.size (0)

    print ('Accuracy : {:.2F}'.format (100 * (correct / total) ))

