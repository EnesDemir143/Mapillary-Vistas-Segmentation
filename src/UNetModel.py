import torch 
import torch.nn as nn
import  torch.nn.functional as F


class UNetEncoder(nn.modules):
    def __init__(self):
        super(UNetEncoder,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.cov2(x))
        x = self.pool(x)

        return x

