import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

class CNN (nn.Module) : 
    def __init__ (self): 
        super (CNN, self).__init__ ()
        self.conv1 = nn.Conv2d (1, 20, 5)
        self.conv2 = nn.Conv2d (20,50,5)

        self.fc1 = nn.Linear (200, 80)
        self.fc2 = nn.Linear (80, 32 )


    def forward (self, x) : 
        x = F.max_pool2d (F.relu (self.conv1 (x)), (2,2))
        x = F.max_pool2d (F.relu (self.conv2 (x)), (2,2))

        x = x.view (-1, self.tot_params (x))
        x = F.relu (self.fc1 (x))
        x = self.fc2 (x)

        return x

    def tot_params (self, x) : 
        p = 1
        for s in x.size ()[1:] : 
            p *= s
        return p

