import torch 
import torch.nn as nn
import  torch.nn.functional as F


class UNetEncoder(nn.Module):

    def __init__(self,in_channels,out_channels):
        super(UNetEncoder,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        skip = x 
        x = self.pool(x)

        return x, skip

class UNetDecoder(nn.Module):

    def __init__(self, in_channels,out_channels):
        super(UNetDecoder,self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2,out_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        

    def forward(self,x,skip):
        x = self.upconv(x)
        x = torch.concat([x, skip],dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x 
    
class UNetBottleneck(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        super(UNetBottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x 