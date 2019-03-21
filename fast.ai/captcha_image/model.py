import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

class CNN (nn.Module) : 
    def __init__ (self): 
        super (CNN, self).__init__ ()

        self.conv1 = nn.Sequential (
            nn.Conv2d (3, 16, 5, stride=1, padding=2),
            nn.ReLU (),
            nn.MaxPool2d ((2,2), stride=2, ceil_mode=True)
        )

        self.conv2 = nn.Sequential (
            nn.Conv2d (16,32, 5, stride=1, padding=2),
            nn.ReLU (),
            nn.MaxPool2d ((2,2), stride=2, ceil_mode=True)
        )

        self.conv3 = nn.Sequential (
            nn.Conv2d (32, 16, 3, stride=1, padding=1),
            nn.ReLU (),
            nn.MaxPool2d ((2,2), stride=2, ceil_mode=True)
        )

        self.fc = nn.Sequential (
            nn.Linear (9 * 9 * 16, 800),
            nn.ReLU (),
            nn.Linear (800, 300),
            nn.ReLU (),
            nn.Linear (300,  80)
        )


    def forward (self, x) : 
        out = self.conv1 (x)
        out = self.conv2 (out)
        out = self.conv3 (out)
        out = out.reshape (out.size (0), -1)
        out = self.fc (out)

        return out 

    def tot_params (self, x) : 
        p = 1
        for s in x.size ()[1:] : 
            p *= s
        return p

