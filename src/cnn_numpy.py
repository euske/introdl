#!/usr/bin/env python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(y):
    return y * (1-y)

def softmax(x):
    x = np.exp(x)
    return x / np.sum(x)

def relu(x):
    return np.maximum(0, x)
def d_relu(x):
    return (0 < x) * 1

# ksize×ksize のカーネルに対応する2次元配列 m の各部分を返す。
def enumerate2d(ksize, m):
    # カーネルの幅だけ小さくしたサイズを計算する。
    (h, w) = m.shape
    h -= ksize-1
    w -= ksize-1
    for i in range(0, h):
        for j in range(0, w):
            # 配列 m から、位置 (i,j) 大きさ ksize×ksize の部分を2次元で切り出す。
            yield (i, j, m[i:i+ksize, j:j+ksize])
    return

# Max pooling を適用した3次元配列と、最大値をとる各要素の位置を返す。
def maxpool2d(stride, m):
    (nc, h1, w1) = m.shape
    # 配列サイズをstrideで割る。このとき端数は切り上げる。
    (h2, w2) = ((h1+stride-1)//stride, (w1+stride-1)//stride)
    # プーリング結果と、各要素の位置を入れる配列。
    p = np.zeros((nc, h2, w2))
    s = np.zeros((nc, h2, w2))
    for c in range(nc):
        # 各チャンネルごとに処理。
        for i in range(0, h2):
            for j in range(0, w2):
                # 部分列を取り出し、最大値を求める。
                i0 = i*stride
                j0 = j*stride
                z = m[c, i0:i0+stride, j0:j0+stride]
                p[c,i,j] = np.max(z)
                # 最大値をとった要素の添字を記録する。
                s[c,i,j] = np.argmax(z)
    return (p, s)# 与えられた配列を、元の配列に配置しなおす。

# 与えられた配列を、元の配列に配置しなおす。
def rev_maxpool2d(stride, p, s):
    (nc, h1, w1) = p.shape
    (h2, w2) = (h1*stride, w1*stride)
    # 元の大きさをもつ配列。
    m = np.zeros((nc, h1*stride, w1*stride))
    for c in range(nc):
        # 各チャンネルごとに処理。
        for i in range(0, h1):
            for j in range(0, w1):
                # 添字から元の要素の位置を復元する。
                i0 = i*stride
                j0 = j*stride
                k = int(s[c,i,j])
                # 元の要素は、(i0,j0) より (k//stride, k%stride) だけずれた位置にある。
                m[c, i0+k//stride, j0+k%stride] = p[c,i,j]
    return m

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

# 入力 ninチャンネル、出力 noutチャンネルで (ksize×ksize) のカーネルを使った
# 畳み込みレイヤーを定義する。
class ConvolutionalLayer:

    def __init__(self, nin, nout, ksize):
        self.nin = nin
        self.nout = nout
        self.ksize = ksize
        # 重み・バイアスを初期化する。
        self.w = np.random.random((self.nout, self.nin, self.ksize, self.ksize))-.5
        self.b = np.random.random((self.nout, self.nin))-.5
        # 計算用の変数を初期化する。
        self.x = self.y = None
        self.dw = np.zeros((self.nout, self.nin, self.ksize, self.ksize))
        self.db = np.zeros((self.nout, self.nin))
        return

    def forward(self, x):
        # xは (ninチャンネル×高さ×幅) の要素をもつ3次元配列。
        self.x = x
        # 出力画像の大きさを計算する。これは入力画像よりカーネルの幅だけ小さい。
        (_,h,w) = x.shape
        h -= self.ksize-1
        w -= self.ksize-1
        # yは (noutチャンネル×高さ×幅) の要素をもつ3次元配列。
        self.y = np.zeros((self.nout, h, w))
        # 各チャンネルの出力を計算する。
        for i in range(self.nout):
            for j in range(self.nin):
                # j番目のチャンネルの各ピクセルに対して、畳み込みを計算する。
                w = self.w[i,j]
                for (p,q,z) in enumerate2d(self.ksize, x[j]):
                    # p,q は出力画像の位置、z は入力画像の一部。
                    self.y[i,p,q] += np.sum(w * z)
        # 各要素にシグモイド関数を適用する。
        self.y = sigmoid(self.y)
        return self.y

    def backward(self, delta):
        # deltaは (noutチャンネル×高さ×幅) の要素をもつ3次元配列。
        # self.y が計算されたときのシグモイド関数の微分を求める。
        ds = d_sigmoid(self.y)
        # 各偏微分を計算する。
        # self.dw += delta * ds * self.x
        # self.db += delta * ds
        for i in range(self.nout):
            for j in range(self.nin):
                dw = self.dw[i,j]
                db = self.db[i,j]
                for (p,q,z) in enumerate2d(self.ksize, self.x[j]):
                    # p,q は出力画像の位置、z は入力画像の一部。
                    d = delta[i,p,q] * ds[i,p,q]
                    dw += d * z
                    db += d
        # 各入力値の微分を求める。
        # dxは (ninチャンネル×高さ×幅) の要素をもつ3次元配列。
        # dx = np.dot(delta * ds, self.w)
        dx = np.zeros(self.x.shape)
        for i in range(self.nout):
            for j in range(self.nin):
                w = self.w[i,j]
                for (p,q,z) in enumerate2d(self.ksize, dx[j]):
                    # z はカーネルに対応する入力ピクセルの一部分。
                    # z は dx の一部を共有しているため、z を変化させることで dx の一部も変化する。
                    z += delta[i,p,q] * ds[i,p,q] * w
        return dx

    def update(self, alpha):
        # 現在の勾配をもとに、損失が減る方向へ重み・バイアスを変化させる。
        self.w -= alpha * self.dw
        self.b -= alpha * self.db
        # 計算用の変数をクリアしておく。
        self.dw.fill(0)
        self.db.fill(0)
        return

# MaxPoolingつきの畳み込みレイヤー。
class ConvolutionalLayerWithMaxPooling:

    def __init__(self, nin, nout, ksize, stride):
        self.nin = nin
        self.nout = nout
        self.ksize = ksize
        self.stride = stride
        # 重み・バイアスを初期化する。
        self.w = np.random.random((self.nout, self.nin, self.ksize, self.ksize))-.5
        self.b = np.random.random((self.nout, self.nin))-.5
        # 計算用の変数を初期化する。
        self.x = self.y = self.sp = None
        self.dw = np.zeros((self.nout, self.nin, self.ksize, self.ksize))
        self.db = np.zeros((self.nout, self.nin))
        return

    def forward(self, x):
        # xは (ninチャンネル×高さ×幅) の要素をもつ3次元配列。
        self.x = x
        # 出力画像の大きさを計算する。これは入力画像よりカーネルの幅だけ小さい。
        (_,h,w) = x.shape
        h -= self.ksize-1
        w -= self.ksize-1
        # yは (noutチャンネル×高さ×幅) の要素をもつ3次元配列。
        self.y = np.zeros((self.nout, h, w))
        # 各チャンネルの出力を計算する。
        for i in range(self.nout):
            for j in range(self.nin):
                # j番目のチャンネルの各ピクセルに対して、畳み込みを計算する。
                w = self.w[i,j]
                for (p,q,z) in enumerate2d(self.ksize, x[j]):
                    # p,q は出力画像の位置、z は入力画像の一部。
                    self.y[i,p,q] += np.sum(w * z)
        # 各要素にシグモイド関数を適用する。
        self.y = sigmoid(self.y)
        # Max poolingを適用する。このとき微分値も保存しておく。
        (y, self.sp) = maxpool2d(self.stride, self.y)
        return y

    def backward(self, delta):
        # Max poolingされたものを復元する。
        delta = rev_maxpool2d(self.stride, delta, self.sp)
        # self.y が計算されたときのシグモイド関数の微分を求める。
        ds = d_sigmoid(self.y)
        # 各偏微分を計算する。
        # self.dw += delta * ds * self.x
        # self.db += delta * ds
        for i in range(self.nout):
            for j in range(self.nin):
                dw = self.dw[i,j]
                db = self.db[i,j]
                for (p,q,z) in enumerate2d(self.ksize, self.x[j]):
                    # p,q は出力画像の位置、z は入力画像の一部。
                    d = delta[i,p,q] * ds[i,p,q]
                    dw += d * z
                    db += d
        # 各入力値の微分を求める。
        # dxは (ninチャンネル×高さ×幅) の要素をもつ3次元配列。
        # dx = np.dot(delta * ds, self.w)
        dx = np.zeros(self.x.shape)
        for i in range(self.nout):
            for j in range(self.nin):
                w = self.w[i,j]
                for (p,q,z) in enumerate2d(self.ksize, dx[j]):
                    # z はカーネルに対応する入力ピクセルの一部分。
                    # z は dx の一部を共有しているため、z を変化させることで dx の一部も変化する。
                    z += delta[i,p,q] * ds[i,p,q] * w
        return dx

    def update(self, alpha):
        # 現在の勾配をもとに、損失が減る方向へ重み・バイアスを変化させる。
        self.w -= alpha * self.dw
        self.b -= alpha * self.db
        # 計算用の変数をクリアしておく。
        self.dw.fill(0)
        self.db.fill(0)
        return
