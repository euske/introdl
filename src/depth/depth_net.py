#!/usr/bin/env python
##
##  depth_net.py - Depth Model.
##
import torch
import torch.nn as nn
import torch.nn.functional as F


##  CoarseNet
##
class CoarseNet(nn.Module):

    INPUT_SIZE = (3, 228, 304)
    OUTPUT_SIZE = (55, 74)

    def __init__(self):
        super().__init__()
        # x: (N × 3 × 228 × 304)
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=1)
        # x: (N × 96 × 55 × 74)
        self.norm1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(2)
        # x: (N × 96 × 27 × 37)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        # x: (N × 256 × 27 × 37)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)
        # x: (N × 256 × 13 × 18)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        # x: (N × 384 × 13 × 18)
        self.norm3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(384)
        self.pool4 = nn.MaxPool2d(2)
        # x: (N × 384 × 6 × 9)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        # x: (N × 256 × 6 × 9)
        self.norm5 = nn.BatchNorm2d(256)
        self.fc6 = nn.Linear(256*6*9, 4096)
        # x: (N × 4096)
        self.norm6 = nn.BatchNorm1d(4096)
        self.fc7 = nn.Linear(4096, 55*74)
        # x: (N × 55*74)
        return

    def forward(self, x):
        assert x.shape[1:] == self.INPUT_SIZE, x.shape
        # x: (N × 3 × 228 × 304)
        x = self.conv1(x)
        # x: (N × 96 × 55 × 74)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # x: (N × 96 × 27 × 37)
        x = self.conv2(x)
        # x: (N × 256 × 27 × 37)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # x: (N × 256 × 13 × 18)
        x = self.conv3(x)
        # x: (N × 384 × 13 × 18)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.pool4(x)
        # x: (N × 384 × 6 × 9)
        x = self.conv5(x)
        # x: (N × 256 × 6 × 9)
        x = x.reshape(-1, 256*6*9)
        # x: (N × 256*6*9)
        x = self.fc6(x)
        # x: (N × 4096)
        x = self.norm6(x)
        x = F.relu(x)
        x = self.fc7(x)
        # x: (N × 55*74)
        x = x.reshape(-1, 55, 74)
        # x: (N × 55 × 74)
        assert x.shape[1:] == self.OUTPUT_SIZE, x.shape
        return x


##  FineNet
##
class FineNet(nn.Module):

    INPUT_SIZE = (3, 228, 304)
    OUTPUT_SIZE = (55, 74)

    def __init__(self):
        super().__init__()
        # x: (N × 3 × 228 × 304)
        self.conv1 = nn.Conv2d(3, 63, 9, stride=2, padding=1)
        # x: (N × 63 × 111 × 149)
        self.norm1 = nn.BatchNorm2d(63)
        self.pool1 = nn.MaxPool2d(2)
        # x: (N × 64 × 55 × 74)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, 5, padding=2)
        # x: (N × 1 × 55 × 74)
        return

    def forward(self, x, coarse=None):
        assert x.shape[1:] == self.INPUT_SIZE, x.shape
        # x: (N × 3 × 228 × 304)
        x = self.conv1(x)
        # x: (N × 63 × 111 × 149)
        x = self.norm1(x)
        x = self.pool1(x)
        # x: (N × 63 × 55 × 74)
        x = F.relu(x)
        if coarse is not None:
            # coarse: (N × 1 × 55 × 74)
            assert (coarse.shape[1:] == x.shape[2:] and
                    coarse.shape[0] == x.shape[0])
            x = torch.cat([x, coarse.unsqueeze(1)], 1)
        x = self.conv2(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        # x: (N × 1 × 55 × 74)
        x = x.reshape(-1, 55, 74)
        # x: (N × 55 × 74)
        assert x.shape[1:] == self.OUTPUT_SIZE, x.shape
        return x

if __name__ == '__main__':
    import torchsummary
    net = CoarseNet().cuda()
    torchsummary.summary(net, net.INPUT_SIZE)
    net = FineNet().cuda()
    torchsummary.summary(net, net.INPUT_SIZE)
