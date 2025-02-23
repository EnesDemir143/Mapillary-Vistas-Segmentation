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
    
class resnetBottleNeckResudialBlock(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(resnetBottleNeckResudialBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0)
        self.batchnorm1 = nn.BatchNorm2d()
        
        self.conv2 = nn.Conv2d(in_channel,out_channel,stride=1,padding=1)
        self.batchnorm2 = nn.BatchNorm2d()
        
        self.conv3 = nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0)
        self.batchnorm3 = nn.BatchNorm2d()


    def forward(self,x):
        identity  =  x 

        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))

        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))

        x = self.conv3(x)
        x = F.relu(self.batchnorm3(x))

        x = F.relu(torch.cat([x, identity]))

        return x 
    
