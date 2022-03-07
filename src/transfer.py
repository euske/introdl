#!/usr/bin/env python
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from yolo_utils import VOCDataset
from yolo_utils import rect_fit, rect_map, image_map, image2torch

# get_label: もっとも広範囲のアノテーションを「画像全体のラベル」とする。
def get_label(annot):
    c = {}
    best = None
    for (name,(x,y,w,h)) in annot:
        if name not in c:
            c[name] = 0
        c[name] += w*h
        if best is None or c[best] < c[name]:
            best = name
    return best

# make_batch: 画像とラベルのミニバッチを作成。
def make_batch(samples):
    inputs = []
    labels = []
    for (image, annot) in samples:
        (mapping,_) = rect_fit(INPUT_SIZE, image.size)
        image = image_map(INPUT_SIZE, mapping, image)
        inputs.append(image2torch(image))
        label = get_label(annot)
        labels.append(LABELS.index(label))
    return (np.array(inputs), np.array(labels))

LABELS = (
    'person', 'car', 'aeroplane', 'bicycle',
    'bird', 'boat', 'bottle', 'bus',
    'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor',
)

INPUT_SIZE = (224, 224)

class AdapterNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, len(LABELS))
        return

    def forward(self, x):
        x = self.fc1(x)
        return x

# train: 1エポック分の訓練をおこなう。
def train(pretrained, adapter, device, loader, optimizer, log_interval=1, dry_run=False):
    # ニューラルネットワークを訓練モードにする。
    adapter.train()
    # 各ミニバッチを処理する。
    for (idx, samples) in enumerate(loader):
        (inputs, labels) = make_batch(samples)
        inputs = torch.tensor(inputs).to(device)
        labels = torch.tensor(labels).to(device)
        # すべての勾配(.grad)をクリアしておく。
        optimizer.zero_grad()
        # 与えられたミニバッチをニューラルネットワークに処理させる。
        outputs = adapter(pretrained(inputs))
        # 損失を計算する。
        loss = F.cross_entropy(outputs, labels)
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
def test(pretrained, adapter, device, loader):
    # ニューラルネットワークを訓練モードにする。
    pretrained.eval()
    adapter.eval()
    correct = 0
    # 以下の処理ではautograd機能を使わない:
    with torch.no_grad():
        # 各ミニバッチを処理する。
        for (idx, samples) in enumerate(loader):
            (inputs, labels) = make_batch(samples)
            inputs = torch.tensor(inputs).to(device)
            # 与えられたミニバッチをニューラルネットワークに処理させる。
            outputs = adapter(pretrained(inputs))
            # 正解かどうかを判定する。
            n = 0
            for (y,label) in zip(outputs, labels):
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
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
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
    parser.add_argument('--start-random', action='store_true', default=False,
                        help='start from random weights')
    parser.add_argument('voc_data', type=str)

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
    train_dataset = VOCDataset(args.voc_data)
    train_loader = DataLoader(train_dataset, collate_fn=lambda x:x, **train_kwargs)
    val_dataset = VOCDataset(args.voc_data, split='val')
    val_loader = DataLoader(val_dataset, collate_fn=lambda x:x, **train_kwargs)

    # 訓練済みのモデル (ResNet-18) を取得する。
    pretrained = torchvision.models.resnet18(not args.start_random)
    # 重み・ウェイトの更新を禁止する。
    if not args.start_random:
        for p in pretrained.parameters():
            p.requires_grad = False
    pretrained.fc = nn.Identity()
    pretrained = pretrained.to(device)
    # モデルを作成。
    adapter = AdapterNet()
    if args.save_model is not None:
        # モデルをファイルから読み込む。
        logging.info(f'Loading: {args.save_model}...')
        try:
            params = torch.load(args.save_model, map_location=device)
            adapter.load_state_dict(params)
        except FileNotFoundError as e:
            logging.error(f'Error: {e}')
    adapter = adapter.to(device)

    # 最適化器と学習率を定義する。
    if args.start_random:
        parameters = list(pretrained.parameters()) + list(adapter.parameters())
    else:
        parameters = adapter.parameters()
    optimizer = optim.Adam(parameters, lr=args.lr)

    # エポック回だけ訓練・テストを繰り返す。
    for epoch in range(args.epochs):
        logging.info(f'*** epoch={epoch+1}/{args.epochs} ***')
        if args.start_random:
            pretrained.train()
        else:
            pretrained.eval()
        train(pretrained, adapter, device, train_loader, optimizer,
              log_interval=args.log_interval, dry_run=args.dry_run)
        test(pretrained, adapter, device, val_loader)

    # モデルをファイルに保存する。
    if args.save_model is not None:
        logging.info(f'Saving: {args.save_model}...')
        params = adapter.state_dict()
        torch.save(params, args.save_model)

    return

if __name__ == '__main__': main()
