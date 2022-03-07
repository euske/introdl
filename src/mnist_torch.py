#!/usr/bin/env python
import os.path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from mnist import load_mnist


##  MNISTDataset
##  指定されたファイルから入力と正解を読み込む。
##  __len__() と __getitem__() メソッドを実装する。
##
class MNISTDataset(Dataset):

    def __init__(self, images_path, labels_path):
        super().__init__()
        self.images = load_mnist(images_path)
        self.labels = load_mnist(labels_path)
        return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return (self.images[index], self.labels[index])


##  MNISTを処理するニューラルネットワーク。
##
class MNISTNet(nn.Module):

    # 各レイヤーの初期化。
    def __init__(self):
        super().__init__()
        # 畳み込み: 入力1チャンネル、出力10チャンネル、カーネル3×3。
        self.conv1 = nn.Conv2d(1, 10, 3)
        # Max Pooling: 1/2に縮める。
        self.pool1 = nn.MaxPool2d(2)
        # 畳み込み: 入力10チャンネル、出力20チャンネル、カーネル3×3。
        self.conv2 = nn.Conv2d(10, 20, 3)
        # Max Pooling: 1/2に縮める。
        self.pool2 = nn.MaxPool2d(2)
        # 全接続 (fully connected): 入力500ノード、出力10ノード。
        self.fc1 = nn.Linear(20*5*5, 10)
        return

    # 与えらえたミニバッチ x を処理する。
    def forward(self, x):
        # x: (N × 1 × 28 × 28)
        x = self.conv1(x)
        x = F.relu(x)
        # x: (N × 10 × 26 × 26)
        x = self.pool1(x)
        # x: (N × 10 × 13 × 13)
        x = self.conv2(x)
        x = F.relu(x)
        # x: (N × 20 × 11 × 11)
        x = self.pool2(x)
        # x: (N × 20 × 5 × 5)
        x = x.reshape(len(x), 20*5*5)
        # x: (N × 500)
        x = self.fc1(x)
        # x: (N × 10)
        return x

# train: 1エポック分の訓練をおこなう。
def train(model, device, loader, optimizer, log_interval=1, dry_run=False):
    # ニューラルネットワークを訓練モードにする。
    model.train()
    # 各ミニバッチを処理する。
    for (idx, (images, labels)) in enumerate(loader):
        images = images.reshape(len(images), 1, 28, 28)
        # 入力をfloat型のテンソルに変換。
        inputs = images.float().to(device)
        # 正解をlong型のテンソルに変換。
        targets = labels.long().to(device)
        # すべての勾配(.grad)をクリアしておく。
        optimizer.zero_grad()
        # 与えられたミニバッチをニューラルネットワークに処理させる。
        outputs = model(inputs)
        # 損失を計算する。
        loss = F.cross_entropy_loss(outputs, targets)
        # 勾配を計算する。
        loss.backward()
        # 重み・バイアスを更新する。
        optimizer.step()
        # 定期的に現在の状況を表示する。
        if dry_run or ((idx+1) % log_interval) == 0:
            avg_loss = loss.item() / len(outputs)
            logging.info(f'train: batch={idx+1}/{len(loader)}, loss={avg_loss:.4f}')
        if dry_run:
            # dry_run モードの場合、1回のみで終了。
            break
    return

# test: テストをおこなう。
def test(model, device, loader):
    # ニューラルネットワークを評価モードにする。
    model.eval()
    correct = 0
    # 以下の処理ではautograd機能を使わない:
    with torch.no_grad():
        # 各ミニバッチを処理する。
        for (idx, (images, labels)) in enumerate(loader):
            images = images.reshape(len(images), 1, 28, 28)
            # 入力をfloat型のテンソルに変換。
            inputs = images.float().to(device)
            # 正解をlong型のテンソルに変換。
            targets = labels.long().to(device)
            # 与えられたミニバッチをニューラルネットワークに処理させる。
            outputs = model(inputs)
            # 正解かどうかを判定する。
            n = 0
            for (y,label) in zip(outputs, targets):
                i = torch.argmax(y)
                if i == label:
                    n += 1
            logging.debug(f'test: batch={idx+1}/{len(loader)}, correct={n}/{len(outputs)}')
            correct += n
    # 結果を表示する。
    total = len(loader.dataset)
    logging.info(f'test: total={correct}/{total} ({100*correct/total:.2f}%)')
    return

# main
def main():
    import argparse
    # コマンドライン引数を解析する。
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='enables verbose logging')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-shuffle', action='store_true', default=False,
                        help='disables dataset shuffling')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str, metavar='path', default=None,
                        help='saves model to file')
    parser.add_argument('datadir', type=str)

    args = parser.parse_args()

    # ログ出力を設定する。
    level = (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=level)

    # 乱数シードを設定する。
    torch.manual_seed(args.seed)

    # CUDA の使用・不使用を設定する。
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # バッチサイズその他のパラメータを設定する。
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': not args.no_shuffle}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 訓練データを読み込む。
    train_dataset = MNISTDataset(
        os.path.join(args.datadir, 'train-images-idx3-ubyte.gz'),
        os.path.join(args.datadir, 'train-labels-idx1-ubyte.gz'))
    train_loader = DataLoader(train_dataset, **train_kwargs)

    # テストデータを読み込む。
    test_dataset = MNISTDataset(
        os.path.join(args.datadir, 't10k-images-idx3-ubyte.gz'),
        os.path.join(args.datadir, 't10k-labels-idx1-ubyte.gz'))
    test_loader = DataLoader(test_dataset, **train_kwargs)

    # モデルを作成。
    model = MNISTNet()
    if args.save_model is not None:
        # モデルをファイルから読み込む。
        logging.info(f'Loading: {args.save_model}...')
        try:
            params = torch.load(args.save_model, map_location=device)
            model.load_state_dict(params)
        except FileNotFoundError as e:
            logging.error(f'Error: {e}')
    model = model.to(device)

    # 最適化器と学習率を定義する。
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # エポック回だけ訓練・テストを繰り返す。
    for epoch in range(args.epochs):
        logging.info(f'*** epoch={epoch+1}/{args.epochs} ***')
        train(model, device, train_loader, optimizer,
              log_interval=args.log_interval, dry_run=args.dry_run)
        test(model, device, test_loader)

    # モデルをファイルに保存する。
    if args.save_model is not None:
        logging.info(f'Saving: {args.save_model}...')
        params = model.state_dict()
        torch.save(params, args.save_model)

    return

if __name__ == '__main__': main()
