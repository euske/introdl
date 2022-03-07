#!/usr/bin/env python
import numpy as np
from cnn_numpy import ConvolutionalLayer
from cnn_numpy import ConvolutionalLayerWithMaxPooling
from cnn_numpy import SoftmaxLayer
from cifar import load_cifar
np.random.seed(1)

def main():
    # レイヤーを 3つ作成。
    conv1 = ConvolutionalLayerWithMaxPooling(3, 5, 3, 2)  # 3x32x32 -> 5x15x15
    conv2 = ConvolutionalLayerWithMaxPooling(5, 10, 3, 2) # 5x15x15 -> 10x7x7
    fc1 = SoftmaxLayer(10*7*7, 10)
    n = 0
    for i in range(1):
        # すべてのファイルに対して処理をおこなう。
        for name in ['data_batch_1', 'data_batch_2']:
            # 訓練データの画像・ラベルを読み込む (パス名は適宜変更)。
            (train_images, train_labels) = load_cifar('./CIFAR/cifar-10-batches-py/'+name)
            for (image,label) in zip(train_images, train_labels):
                x = (image/255)
                # 正解部分だけが 1 になっている 10要素の配列を作成。
                ya = np.zeros(10)
                ya[label] = 1
                # 学習させる。
                y = conv1.forward(x)
                y = conv2.forward(y)
                y = y.reshape(10*7*7)
                y = fc1.forward(y)
                delta = fc1.cross_entropy_loss_backward(ya)
                delta = delta.reshape(10, 7, 7)
                delta = conv2.backward(delta)
                delta = conv1.backward(delta)
                n += 1
                if (n % 50 == 0):
                    print(n, fc1.loss)
                    conv1.update(0.01)
                    conv2.update(0.01)
                    fc1.update(0.01)
    # テストデータの画像・ラベルを読み込む (パス名は適宜変更)。
    (test_images, test_labels) = load_cifar('./CIFAR/cifar-10-batches-py/test_batch')
    correct = 0
    for (image,label) in zip(test_images, test_labels):
        x = (image/255)
        y = conv1.forward(x)
        y = conv2.forward(y)
        y = y.reshape(10*7*7)
        y = fc1.forward(y)
        i = np.argmax(y)
        if i == label:
            correct += 1
    print(correct, len(test_images))

main()
