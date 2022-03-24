#!/usr/bin/env python
import sys, os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def load_mnist(path):
    import gzip
    import struct
    if path.endswith('.gz'):
        fp = gzip.open(path)
    else:
        fp = open(path, 'rb')
    magic = fp.read(4)
    if magic == b'\x00\x00\x08\x01':
        (n,) = struct.unpack('>L', fp.read(4))
        size = n
        data = np.frombuffer(fp.read(size), dtype=np.uint8)
        assert data.size == size
    elif magic == b'\x00\x00\x08\x03':
        (n,rows,cols) = struct.unpack('>LLL', fp.read(12))
        size = n * rows * cols
        data = np.frombuffer(fp.read(size), dtype=np.uint8)
        assert data.size == size
        data = data.reshape(n, rows, cols)
    else:
        raise ValueError(f'Invalid Magic number: {magic}')
    fp.close()
    return data

class MNISTDataset(Dataset):

    def __init__(self, images_path, labels_path):
        super().__init__()
        self.images = load_mnist(images_path)
        self.labels = load_mnist(labels_path)
        return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index].astype(np.float32)
        img = img[2:26,2:26]
        return (img/255, self.labels[index])

class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24*24, 100)
        self.fc2 = nn.Linear(100, 10)
        return

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

basedir = sys.argv[1]
train_dataset = MNISTDataset(
    os.path.join(basedir, 'train-images-idx3-ubyte.gz'),
    os.path.join(basedir, 'train-labels-idx1-ubyte.gz'))
test_dataset = MNISTDataset(
    os.path.join(basedir, 't10k-images-idx3-ubyte.gz'),
    os.path.join(basedir, 't10k-labels-idx1-ubyte.gz'))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset)

model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5):
    model.train()
    for (images, labels) in train_loader:
        images = images.reshape(len(images), 24*24)
        inputs = images.float()
        targets = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    correct = 0
    for (images, labels) in test_loader:
        images = images.reshape(len(images), 24*24)
        inputs = images.float()
        targets = labels.long()
        outputs = model(inputs)
        for (y,label) in zip(outputs, targets):
            i = torch.argmax(y)
            if i == label:
                correct += 1
    print(f'train: {epoch}, correct={correct}', file=sys.stderr)

def fmt(a):
    if isinstance(a, list):
        return f'[{",".join(fmt(x) for x in a)}]'
    else:
        return f'{a:.4f}'

names = ['MINIST_W1','MINIST_B1','MINIST_W2','MINIST_B2']
for (name,x) in zip(names, model.parameters()):
    x = x.detach().numpy()
    print(f'{name}={fmt(x.tolist())};')
