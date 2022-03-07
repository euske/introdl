from math import exp, sqrt
from random import random, seed
seed(0)

def sigmoid(x):
    return 1 / (1 + exp(-x))

class Node:
    def __init__(self, nin):
        self.nin = nin
        self.w = [ random() for i in range(self.nin) ]
        self.b = random()
        self.x = self.y = None
        self.dw = [ 0 for i in range(self.nin) ]
        self.db = 0
        self.loss = 0
        return

    def forward(self, x):
        self.x = x
        self.y = sigmoid(sum( w1*x1 for (w1,x1) in zip(self.w, x) ) + self.b)
        return self.y

    def mse_loss(self, ya):
        self.loss += (self.y - ya)**2
        return 2*(self.y - ya)

    def backward(self, delta):
        ds = self.y * (1-self.y)
        for i in range(self.nin):
            self.dw[i] += delta * ds * self.x[i]
        self.db += delta*ds
        return

    def update(self, alpha):
        for i in range(self.nin):
            self.w[i] -= alpha * self.dw[i]
        self.b -= alpha * self.db
        self.dw = [ 0 for i in range(self.nin) ]
        self.db = 0
        self.loss = 0
        return

# 入力 nin個、出力 nout個のレイヤーを定義する。
class Layer:
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        # 重み・バイアスを初期化する。
        self.w = [ [ random() for j in range(self.nin) ] for i in range(self.nout) ]
        self.b = [ random() for i in range(self.nout) ]
        # 計算用の変数を初期化する。
        self.x = self.y = None
        self.dw = [ [ 0 for j in range(self.nin) ] for i in range(self.nout) ]
        self.db = [ 0 for i in range(self.nout) ]
        self.loss = 0
        return

    def forward(self, x):
        # xは nin個の要素をもつ入力値のリスト。
        # 与えられた入力に対する各ノードの出力を計算する。
        self.x = x
        self.y = [
            sigmoid(sum( w1*x1 for (w1,x1) in zip(w, x) ) + b)
            for (w,b) in zip(self.w, self.b)
        ]
        # yは nout個の要素をもつ出力値のリスト
        return self.y

    def mse_loss(self, ya):
        # 与えられた正解に対する損失を求める。
        self.loss += sum( (y1-ya1)**2 for (y1,ya1) in zip(self.y, ya) )
        # 損失関数の微分を計算する。
        delta = [ 2*(y1-ya1) for (y1,ya1) in zip(self.y, ya) ]
        return delta

    def backward(self, delta):
        # self.y が計算されたときのシグモイド関数の微分を求める。
        ds = [ y1*(1-y1) for y1 in self.y ]
        # 各偏微分を計算する。
        for i in range(self.nout):
            for j in range(self.nin):
                self.dw[i][j] += delta[i] * ds[i] * self.x[j]
        for i in range(self.nout):
            self.db[i] += delta[i] * ds[i]
        # 各入力値の微分を求める。
        dx = [
            sum( delta[j]*ds[j]*self.w[j][i] for j in range(self.nout) )
            for i in range(self.nin)
        ]
        return dx

    def update(self, alpha):
        # 現在の勾配をもとに、損失が減る方向へ重み・バイアスを変化させる。
        for i in range(self.nout):
            for j in range(self.nin):
                self.w[i][j] -= alpha * self.dw[i][j]
        for i in range(self.nout):
            self.b[i] -= alpha * self.db[i]
        # 計算用の変数をクリアしておく。
        for i in range(self.nout):
            for j in range(self.nin):
                self.dw[i][j] = 0
        for i in range(self.nout):
            self.db[i] = 0
        self.loss = 0
        return

def main1():
    n1 = Node(3)
    for i in range(100):
        y = n1.forward([0,0,0])
        delta = n1.mse_loss(1)
        n1.backward(delta)
        y = n1.forward([0,1,0])
        delta = n1.mse_loss(1)
        n1.backward(delta)
        y = n1.forward([1,0,1])
        delta = n1.mse_loss(0)
        n1.backward(delta)
        print(n1.loss)
        n1.update(0.1)
    return

def main2():
    # 100個分のランダムな訓練データを作成する。
    N = 3
    data = []
    for i in range(100):
        x = [ random() for _ in range(N) ]              # 入力
        ya = [ sqrt( sum( z*z for z in x )/len(x) ) ]  # 正解
        data.append((x, ya))

    layer1 = Layer(N, 3)
    layer2 = Layer(3, 1)
    #layer3 = Layer(3, 1)
    layern = layer2
    # 1000回繰り返す。
    for i in range(5000):
        for (x,ya) in data:
            # 入力に対する出力を計算する。
            y = layer1.forward(x)
            y = layer2.forward(y)
            #y = layer3.forward(y)
            # 損失を計算する。
            delta = layern.mse_loss(ya)
            # 勾配を計算。
            #delta = layer3.backward(delta)
            delta = layer2.backward(delta)
            delta = layer1.backward(delta)
        # 現在の損失を表示する。
        if i % 100 == 0:
            print(i, layern.loss)
        # 重み・バイアスを学習率 0.1 で変化させる。
        layer1.update(0.1)
        layer2.update(0.1)
        #layer3.update(0.1)

main2()
