import torch 
import torch.nn as nn
import torch.nn.functional as F


class resnetStemBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(resnetStemBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=7,stride=2)
        self.batchnorm1 = nn.BatchNorm2d()
        self.pooling1 = nn.MaxPool2d(3,3,stride=2) 


    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pooling1(x)

        return x 
    

