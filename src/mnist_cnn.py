#!/usr/bin/env python
import numpy as np
from cnn_numpy import ConvolutionalLayer
from cnn_numpy import ConvolutionalLayerWithMaxPooling
from cnn_numpy import SoftmaxLayer
from mnist import load_mnist
np.random.seed(1)

def main1():
    # 訓練データの画像・ラベルを読み込む (パス名は適宜変更)。
    train_images = load_mnist('./MNIST/train-images-idx3-ubyte.gz')
    train_labels = load_mnist('./MNIST/train-labels-idx1-ubyte.gz')
    # レイヤーを 2つ作成。
    conv1 = ConvolutionalLayer(1, 5, 3)
    fc1 = SoftmaxLayer(5*26*26, 10)
    n = 0
    for i in range(1):
        for (image,label) in zip(train_images, train_labels):
            # 28×28の画像を 1×28×28 の3次元配列に変換。
            x = (image/255).reshape(1, 28, 28)
            # 正解部分だけが 1 になっている 10要素の配列を作成。
            ya = np.zeros(10)
            ya[label] = 1
            # 学習させる。
            y = conv1.forward(x)
            y = y.reshape(5*26*26)
            y = fc1.forward(y)
            delta = fc1.cross_entropy_loss_backward(ya)
            delta = conv1.backward(delta.reshape(5, 26, 26))
            n += 1
            if (n % 50 == 0):
                print(n, conv1.loss)
                conv1.update(0.01)
                fc1.update(0.01)
    test_images = load_mnist('./MNIST/t10k-images-idx3-ubyte.gz')
    test_labels = load_mnist('./MNIST/t10k-labels-idx1-ubyte.gz')
    correct = 0
    for (image,label) in zip(test_images, test_labels):
        x = (image/255).reshape(1, 28, 28)
        y = conv1.forward(x)
        y = y.reshape(5*26*26)
        y = fc1.forward(y)
        i = np.argmax(y)
        if i == label:
            correct += 1
    print(correct, len(test_images))

def main2():
    # 訓練データの画像・ラベルを読み込む (パス名は適宜変更)。
    train_images = load_mnist('./MNIST/train-images-idx3-ubyte.gz')
    train_labels = load_mnist('./MNIST/train-labels-idx1-ubyte.gz')
    # レイヤーを 3つ作成。
    conv1 = ConvolutionalLayerWithMaxPooling(1, 5, 3, 2)  # 1x28x28 -> 5x13x13
    conv2 = ConvolutionalLayerWithMaxPooling(5, 10, 3, 2) # 5x13x13 -> 10x6x6
    fc1 = SoftmaxLayer(10*6*6, 10)
    n = 0
    for i in range(1):
        for (image,label) in zip(train_images, train_labels):
            # 28×28の画像を 1×28×28 の3次元配列に変換。
            x = (image/255).reshape(1, 28, 28)
            # 正解部分だけが 1 になっている 10要素の配列を作成。
            ya = np.zeros(10)
            ya[label] = 1
            # 学習させる。
            y = conv1.forward(x)
            y = conv2.forward(y)
            y = y.reshape(10*6*6)
            y = fc1.forward(y)
            delta = fc1.cross_entropy_loss_backward(ya)
            delta = delta.reshape(10, 6, 6)
            delta = conv2.backward(delta)
            delta = conv1.backward(delta)
            n += 1
            if (n % 50 == 0):
                print(n, fc1.loss)
                conv1.update(0.01)
                conv2.update(0.01)
                fc1.update(0.01)
    test_images = load_mnist('./MNIST/t10k-images-idx3-ubyte.gz')
    test_labels = load_mnist('./MNIST/t10k-labels-idx1-ubyte.gz')
    correct = 0
    for (image,label) in zip(test_images, test_labels):
        x = (image/255).reshape(1, 28, 28)
        y = conv1.forward(x)
        y = conv2.forward(y)
        y = y.reshape(10*6*6)
        y = fc1.forward(y)
        i = np.argmax(y)
        if i == label:
            correct += 1
    print(correct, len(test_images))

main2()
