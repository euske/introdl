#!/usr/bin/env python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(y):
    return y * (1-y)

def softmax(x):
    x = np.exp(x)
    return x / np.sum(x)

# 入力 nin個、出力 nout個のレイヤーを定義する。
class Layer:
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        # 重み・バイアスを初期化する。
        self.w = np.random.random((self.nout, self.nin))-.5
        self.b = np.random.random(self.nout)-.5
        # 計算用の変数を初期化する。
        self.x = self.y = None
        self.dw = np.zeros((self.nout, self.nin))
        self.db = np.zeros(self.nout)
        self.loss = 0
        return

    def forward(self, x):
        # xは nin個の要素をもつ入力値のリスト。
        # 与えられた入力に対する各ノードの出力を計算する。
        self.x = x
        self.y = sigmoid(np.dot(self.w, x) + self.b)
        # yは nout個の要素をもつ出力値のリスト
        return self.y

    def mse_loss(self, ya):
        # 与えられた正解に対する損失を求める。
        self.loss += ((self.y - ya)**2).sum()
        # 損失関数の微分を計算する。
        delta = 2*(self.y - ya)
        return delta

    def backward(self, delta):
        # self.y が計算されたときのシグモイド関数の微分を求める。
        ds = d_sigmoid(self.y)
        # 各偏微分を計算する。
        self.dw += (delta * ds).reshape(self.nout, 1) * self.x
        self.db += delta * ds
        # 各入力値の微分を求める。
        dx = np.dot(delta * ds, self.w)
        return dx

    def update(self, alpha):
        # 現在の勾配をもとに、損失が減る方向へ重み・バイアスを変化させる。
        self.w -= alpha * self.dw
        self.b -= alpha * self.db
        # 計算用の変数をクリアしておく。
        self.dw.fill(0)
        self.db.fill(0)
        self.loss = 0
        return

# 入力 nin個、出力 nout個のレイヤーを定義する。
class SoftmaxLayer:

    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        # 重み・バイアスを初期化する。
        self.w = np.random.random((self.nout, self.nin))-.5
        self.b = np.random.random(self.nout)-.5
        # 計算用の変数を初期化する。
        self.x = self.y = None
        self.dw = np.zeros((self.nout, self.nin))
        self.db = np.zeros(self.nout)
        self.loss = 0
        return

    def forward(self, x):
        # xは nin個の要素をもつ入力値のリスト。
        # 与えられた入力に対する各ノードの出力を計算する。
        self.x = x
        self.y = softmax(np.dot(self.w, x) + self.b)
        # yは nout個の要素をもつ出力値のリスト
        return self.y

    def cross_entropy_loss_backward(self, ya):
        # 与えられた正解に対する損失を求める。
        i = np.argmax(ya)
        self.loss -= np.log(self.y[i])
        # 損失関数の微分を計算する。
        delta = (self.y - ya)
        # 各偏微分を計算する。
        self.dw += delta.reshape(self.nout, 1) * self.x
        self.db += delta
        # 各入力値の微分を求める。
        dx = np.dot(delta, self.w)
        return dx

    def update(self, alpha):
        # 現在の勾配をもとに、損失が減る方向へ重み・バイアスを変化させる。
        self.w -= alpha * self.dw
        self.b -= alpha * self.db
        # 計算用の変数をクリアしておく。
        self.dw.fill(0)
        self.db.fill(0)
        self.loss = 0
        return

# nn1.py の main2() の NumPy版。
def main2():
    # 100個分のランダムな訓練データを作成する。
    np.random.seed(0)
    data = []
    for i in range(100):
        x = np.random.random(3)
        ya = np.sqrt(np.sum(x**2) / 3)
        data.append((x, ya))

    layer1 = Layer(3, 3)
    layer2 = Layer(3, 1)
    layern = layer2
    # 1000回繰り返す。
    for i in range(5000):
        for (x,ya) in data:
            # 入力に対する出力を計算する。
            y = layer1.forward(x)
            y = layer2.forward(y)
            # 損失を計算する。
            delta = layern.mse_loss(ya)
            # 勾配を計算。
            delta = layer2.backward(delta)
            delta = layer1.backward(delta)
        # 現在の損失を表示する。
        if i % 100 == 0:
            print(i, layern.loss)
        # 重み・バイアスを学習率 0.1 で変化させる。
        layer1.update(0.1)
        layer2.update(0.1)
