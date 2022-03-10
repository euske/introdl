#!/usr/bin/env python
##
##  vgg16.py - VGG-16 Model.
##
import torch
import torch.nn as nn
import torch.nn.functional as F


##  VGG16
##
class VGG16(nn.Module):

    def __init__(self):
        super().__init__()
        # x: (N × 3 × 224 × 224)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # x: (N × 64 × 112 × 112)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        # x: (N × 128 × 56 × 56)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool7 = nn.MaxPool2d(2)
        # x: (N × 256 × 28 × 28)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool10 = nn.MaxPool2d(2)
        # x: (N × 512 × 14 × 14)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool13 = nn.MaxPool2d(2)
        # x: (N × 512 × 7 × 7)
        self.fc14 = nn.Linear(512*7*7, 4096)
        # x: (N × 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 1000)
        # x: (N × 1000)
        return

    def forward(self, x):
        # x: (N × 3 × 224 × 224)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # x: (N × 64 × 112 × 112)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        # x: (N × 128 × 56 × 56)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.pool7(x)
        # x: (N × 256 × 28 × 28)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)
        x = self.pool10(x)
        # x: (N × 512 × 14 × 14)
        x = self.conv11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = F.relu(x)
        x = self.conv13(x)
        x = F.relu(x)
        x = self.pool13(x)
        # x: (N × 512 × 7 × 7)
        x = x.reshape(-1, 512*7*7)
        x = self.fc14(x)
        # x: (N × 4096)
        x = F.relu(x)
        x = self.fc15(x)
        x = F.relu(x)
        x = self.fc16(x)
        # x: (N × 1000)
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == '__main__':
    import torchsummary
    net = VGG16()
    torchsummary.summary(net, (3,224,224))
