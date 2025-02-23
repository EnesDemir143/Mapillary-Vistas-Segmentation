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

    def __init__(self, in_channel, out_channel):
        super(resnetBottleNeckResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel // 4, kernel_size=1, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(out_channel // 4)
        
        self.conv2 = nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channel // 4)
        
        self.conv3 = nn.Conv2d(out_channel // 4, out_channel, kernel_size=1, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(out_channel)

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

    def __init__(self, in_channel, out_channel):
        super(resnetModel, self).__init__()

        # Stem block
        self.stem = resnetStemBlock(3, 64)


        # 3 BottleNeck Blocks (with correct channels)
        self.bottleNeck1 = resnetBottleNeckResidualBlock(64, 256)
        self.bottleNeck2 = resnetBottleNeckResidualBlock(256, 256)
        self.bottleNeck3 = resnetBottleNeckResidualBlock(256, 256)

        # 4 BottleNeck Blocks
        self.bottleNeck4 = resnetBottleNeckResidualBlock(256, 512)
        self.bottleNeck5 = resnetBottleNeckResidualBlock(512, 512)
        self.bottleNeck6 = resnetBottleNeckResidualBlock(512, 512)
        self.bottleNeck7 = resnetBottleNeckResidualBlock(512, 512)

        # 6 BottleNeck Blocks
        self.bottleNeck8 = resnetBottleNeckResidualBlock(512, 1024)
        self.bottleNeck9 = resnetBottleNeckResidualBlock(1024, 1024)
        self.bottleNeck10 = resnetBottleNeckResidualBlock(1024, 1024)
        self.bottleNeck11 = resnetBottleNeckResidualBlock(1024, 1024)
        self.bottleNeck12 = resnetBottleNeckResidualBlock(1024, 1024)

        # 3 BottleNeck Blocks
        self.bottleNeck13 = resnetBottleNeckResidualBlock(1024, 2048)
        self.bottleNeck14 = resnetBottleNeckResidualBlock(2048, 2048)
        self.bottleNeck15 = resnetBottleNeckResidualBlock(2048, 2048)

    def forward(self, x):

        x = self.stem(x)

        x = self.bottleNeck1(x)
        x = self.bottleNeck2(x)
        x = self.bottleNeck3(x)

        x = self.bottleNeck4(x)
        x = self.bottleNeck5(x)
        x = self.bottleNeck6(x)
        x = self.bottleNeck7(x)

        x = self.bottleNeck8(x)
        x = self.bottleNeck9(x)
        x = self.bottleNeck10(x)
        x = self.bottleNeck11(x)
        x = self.bottleNeck12(x)

        x = self.bottleNeck13(x)
        x = self.bottleNeck14(x)
        x = self.bottleNeck15(x)

        return x