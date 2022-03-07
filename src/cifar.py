#!/usr/bin/env python
import numpy as np

def load_cifar(path):
    import pickle
    # ファイルを開く。
    with open(path, 'rb') as fp:
        data = pickle.load(fp, encoding='bytes')
    images = [ img.reshape(3, 32, 32) for img in data[b'data'] ]
    labels = data[b'labels']
    return (images, labels)

def preview(n=10):
    from PIL import Image
    (images, labels) = load_cifar('data_batch_1')
    for i in range(n):
        img = Image.fromarray(images[i].transpose(1,2,0))
        img.save(f'output_{i:05d}_{labels[i]}.png')
    return

#preview()
