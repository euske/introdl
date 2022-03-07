#!/usr/bin/env python
import numpy as np
from nn_numpy import Layer
from nn_numpy import SoftmaxLayer
from mnist import load_mnist
np.random.seed(1)

def main1():
    # 訓練データの画像・ラベルを読み込む (パス名は適宜変更)。
    train_images = load_mnist('./MNIST/train-images-idx3-ubyte.gz')
    train_labels = load_mnist('./MNIST/train-labels-idx1-ubyte.gz')
    # レイヤーを 2つ作成。
    layer1 = Layer(784, 100)
    layer2 = Layer(100, 10)
    n = 0
    for i in range(1):
        for (image,label) in zip(train_images, train_labels):
            # 28×28の画像をフラットな配列に変換。
            x = (image/255).reshape(784)
            # 正解部分だけが 1 になっている 10要素の配列を作成。
            ya = np.zeros(10)
            ya[label] = 1
            # 学習させる。
            y = layer1.forward(x)
            y = layer2.forward(y)
            delta = layer2.mse_loss(ya)
            delta = layer2.backward(delta)
            delta = layer1.backward(delta)
            n += 1
            if (n % 50 == 0):
                print(n, layer2.loss)
                layer1.update(0.01)
                layer2.update(0.01)
    test_images = load_mnist('./MNIST/t10k-images-idx3-ubyte.gz')
    test_labels = load_mnist('./MNIST/t10k-labels-idx1-ubyte.gz')
    correct = 0
    for (image,label) in zip(test_images, test_labels):
        x = (image/255).flatten()
        y = layer1.forward(x)
        y = layer2.forward(y)
        i = np.argmax(y)
        if i == label:
            correct += 1
    print(correct, len(test_images))

def main2():
    # 訓練データの画像・ラベルを読み込む (パス名は適宜変更)。
    train_images = load_mnist('./MNIST/train-images-idx3-ubyte.gz')
    train_labels = load_mnist('./MNIST/train-labels-idx1-ubyte.gz')
    # レイヤーを 3つ作成。
    layer1 = Layer(784, 100)
    layerx = Layer(100, 100)
    layer2 = SoftmaxLayer(100, 10)
    n = 0
    for i in range(1):
        for (image,label) in zip(train_images, train_labels):
            # 28×28の画像をフラットな配列に変換。
            x = (image/255).reshape(784)
            # 正解部分だけが 1 になっている 10要素の配列を作成。
            ya = np.zeros(10)
            ya[label] = 1
            # 学習させる。
            y = layer1.forward(x)
            y = layerx.forward(y)
            y = layer2.forward(y)
            delta = layer2.cross_entropy_loss_backward(ya)
            delta = layerx.backward(delta)
            delta = layer1.backward(delta)
            n += 1
            if (n % 50 == 0):
                print(n, layer2.loss)
                layer1.update(0.01)
                layerx.update(0.01)
                layer2.update(0.01)
    test_images = load_mnist('./MNIST/t10k-images-idx3-ubyte.gz')
    test_labels = load_mnist('./MNIST/t10k-labels-idx1-ubyte.gz')
    correct = 0
    for (image,label) in zip(test_images, test_labels):
        x = (image/255).flatten()
        y = layer1.forward(x)
        y = layerx.forward(y)
        y = layer2.forward(y)
        i = np.argmax(y)
        if i == label:
            correct += 1
    print(correct, len(test_images))

main2()
