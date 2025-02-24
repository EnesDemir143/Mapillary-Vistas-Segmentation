import torch
import torch.nn as nn
import torch.nn.functional as F


class resnetStemBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super(resnetStemBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=2, padding=3)  
        self.batchnorm1 = nn.BatchNorm2d(out_channel)
        self.pooling1 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pooling1(x)

        return x


class resnetBottleNeckResidualBlock(nn.Module):  

    def __init__(self, in_channel, out_channel,stride):
        super(resnetBottleNeckResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel // 4, kernel_size=1, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(out_channel // 4)
        
        self.conv2 = nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, stride=stride, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channel // 4)
        
        self.conv3 = nn.Conv2d(out_channel // 4, out_channel, kernel_size=1, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(out_channel)

        if  in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.downsample = None

            
    def forward(self, x):
        identity = x 

        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))

        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))

        x = self.conv3(x)
        x = self.batchnorm3(x)

        x += identity  

        x = F.relu(x)

        return x


class resnetModel(nn.Module):

    def __init__(self, in_channel,num_classes):
        super(resnetModel, self).__init__()

        # Stem block
        self.stem = resnetStemBlock(in_channel, 64)

        #Resudial blocks of resnet50
        self.block1 = self._makelayer(64, 256, 3,1)
        self.block2 = self._makelayer(256,512,4,2)
        self.block3 = self._makelayer(512,1024,6,2)
        self.block4 = self._makelayer(1024,2048,3,2)

        self.global_avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fully_connetted = nn.Linear(2048,num_classes)

    def _makelayer(self, in_channel, out_channel, block,stride):
            layer = []
            layer.append(resnetBottleNeckResidualBlock(in_channel, out_channel,stride))

            for i in range(block-1):
                layer.append(resnetBottleNeckResidualBlock(out_channel, out_channel,1))
            
            return nn.Sequential(*layer)

    def forward(self, x):
         
        x = self.stem(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4 (x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connetted(x)

        return x
