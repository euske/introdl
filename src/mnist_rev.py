#!/usr/bin/env python
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from mnist_torch import MNISTNet

# main
def main():
    # MNISTモデルを読み込む。
    model = MNISTNet()
    params = torch.load('mnist_model.pt')
    model.load_state_dict(params)

    alpha = 0.1
    epochs = 100
    # 0 から 9 までの数値を試す。
    for i in range(10):
        t = torch.tensor([i])
        # ランダムな入力データから始める。
        x = torch.rand((1,1,28,28))
        # epochs回だけ繰り返す。
        for _ in range(epochs):
            # 入力値の勾配を有効にする。
            x.requires_grad_(True)
            # モデルを使って推論し、損失を計算する。
            y = model(x)
            loss = F.cross_entropy(y, t)
            loss.backward()
            # 勾配に応じて入力値を変化させる。
            x = x.detach() - alpha*x.grad
        print(i, loss.item())
        assert model(x).argmax() == i
        # 最終的な入力値を画像として保存する。
        x = (x.detach().reshape(28,28).numpy()*255).astype(np.uint8)
        image = Image.fromarray(x, 'L')
        image.save(f'digit_{i}.png')
