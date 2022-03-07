#!/usr/bin/env python
import numpy as np
from nn_numpy import Layer, sigmoid, d_sigmoid
from nn_numpy import SoftmaxLayer
from mnist import load_mnist
np.random.seed(1)

class ResidualLayer(Layer):

    def forward(self, x):
        # xは nin個の要素をもつ入力値のリスト。
        # 与えられた入力に対する各ノードの出力を計算する。
        self.x = x
        self.y = sigmoid(np.dot(self.w, x) + self.b)
        # yは nout個の要素をもつ出力値のリスト
        if len(x) < len(self.y):
            x = np.pad(x, (0, len(self.y)-len(x)))
        return self.y + x

    def backward(self, delta):
        # self.y が計算されたときのシグモイド関数の微分を求める。
        ds = d_sigmoid(self.y)
        # 各偏微分を計算する。
        self.dw += (delta * ds).reshape(self.nout, 1) * self.x
        self.db += delta * ds
        # 各入力値の微分を求める。
        dx = np.dot(delta * ds, self.w)
        if len(self.x) < len(delta):
            delta = delta[:len(self.x)]
        return dx + delta


def main():
    # 訓練データの画像・ラベルを読み込む (パス名は適宜変更)。
    train_images = load_mnist('./MNIST/train-images-idx3-ubyte.gz')
    train_labels = load_mnist('./MNIST/train-labels-idx1-ubyte.gz')
    # N個の中間レイヤーを作成。
    N = 10
    layers = [Layer(784, 100)]
    for _ in range(N):
        #layers.append(Layer(100, 100))
        layers.append(ResidualLayer(100, 100))
    softmax = SoftmaxLayer(100, 10)
    n = 0
    for i in range(1):
        for (image,label) in zip(train_images, train_labels):
            x = (image/255).reshape(784)
            ya = np.zeros(10)
            ya[label] = 1
            # 学習させる。
            for layer in layers:
                x = layer.forward(x)
            y = softmax.forward(x)
            delta = softmax.cross_entropy_loss_backward(ya)
            for layer in reversed(layers):
                delta = layer.backward(delta)
            n += 1
            if (n % 50 == 0):
                # 各レイヤーの勾配を表示。
                print(n, softmax.loss, [ np.sqrt((layer.dw**2).mean()) for layer in layers ])
                for layer in layers:
                    layer.update(0.01)
                softmax.update(0.01)

main()
