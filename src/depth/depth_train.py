#!/usr/bin/env python
##
##  depth_train.py - DEPTH training.
##
##  usage:
##    $ ./depth_train.py --model-coarse depth_net_coarse.pt nyudepth.mat
##    $ ./depth_train.py --train-fine --model-coarse depth_net_coarse.pt --model-fine depth_net_fine.pt nyudepth.mat
##
import logging
import os.path
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from depth_net import CoarseNet, FineNet
from depth_utils import NYUDepth2Dataset, conv_image, conv_depth


# train: 1エポック分の訓練をおこなう。
def train(model_coarse, model_fine, device, loader, optimizer,
          log_interval=1, dry_run=False):
    (_,irows,icols) = model_coarse.INPUT_SIZE
    (orows,ocols) = model_coarse.OUTPUT_SIZE
    npix = orows * ocols
    # 各ミニバッチを処理する。
    for (idx, samples) in enumerate(loader):
        # すべての勾配(.grad)をクリアしておく。
        optimizer.zero_grad()
        images = []
        depths = []
        flips = (0.5 < torch.rand(len(samples)))
        for ((image,depth),flip) in zip(samples, flips):
            images.append(conv_image(image, (irows,icols), flip))
            depths.append(conv_depth(depth, (orows,ocols), flip))
        inputs = torch.tensor(np.array(images)).to(device)
        depths = torch.tensor(np.array(depths)).to(device)
        # 与えられたミニバッチをニューラルネットワークに処理させる。
        outputs = model_coarse(inputs)
        if model_fine is not None:
            outputs = model_fine(inputs, outputs.detach())
        # 損失を計算する。
        assert outputs.shape == depths.shape
        log_depths = torch.log(depths)
        loss = (F.mse_loss(outputs, log_depths) +
                0.5*(F.l1_loss(outputs, log_depths)**2)/npix)
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

# main
def main():
    import argparse
    # コマンドライン引数を解析する。
    parser = argparse.ArgumentParser(description='Depth Train')
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
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-coarse', type=str, metavar='path', default=None,
                        help='saves coarse model to file')
    parser.add_argument('--model-fine', type=str, metavar='path', default=None,
                        help='saves fine model to file')
    parser.add_argument('--train-fine', action='store_true', default=False,
                        help='train fine model')
    parser.add_argument('depth_data', type=str)

    args = parser.parse_args()

    # ログ出力を設定する。
    level = (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=level)

    # 乱数シードを設定する。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # CUDA の使用・不使用を設定する。
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # バッチサイズその他のパラメータを設定する。
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': not args.no_shuffle}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # 訓練データを読み込む。
    dataset = NYUDepth2Dataset(args.depth_data)
    loader = DataLoader(dataset, collate_fn=lambda x:x, **train_kwargs)

    # Coarseモデルを作成。
    model_coarse = CoarseNet()
    if args.model_coarse is not None:
        # モデルをファイルから読み込む。
        logging.info(f'Loading: {args.model_coarse}...')
        try:
            params = torch.load(args.model_coarse, map_location=device)
            model_coarse.load_state_dict(params)
        except FileNotFoundError as e:
            logging.error(f'Error: {e}')
    model_coarse = model_coarse.to(device)
    model_fine = None
    if args.train_fine:
        # Fineモデルを作成。
        model_fine = FineNet()
        if args.model_fine is not None:
            # モデルをファイルから読み込む。
            logging.info(f'Loading: {args.model_fine}...')
            try:
                params = torch.load(args.model_fine, map_location=device)
                model_fine.load_state_dict(params)
            except FileNotFoundError as e:
                logging.error(f'Error: {e}')
        model_fine = model_fine.to(device)

    if model_fine is None:
        # Coarseモデルの訓練:
        # ニューラルネットワークを訓練モードにする。
        model_coarse.train()
        # 最適化器と学習率を定義する。
        optimizer = optim.Adam(model_coarse.parameters(), lr=args.lr)
        # エポック回だけ訓練を繰り返す。
        for epoch in range(args.epochs):
            logging.info(f'*** epoch={epoch+1}/{args.epochs} ***')
            train(model_coarse, model_fine, device, loader, optimizer,
                  log_interval=args.log_interval, dry_run=args.dry_run)
        # モデルをファイルに保存する。
        if args.model_coarse is not None:
            logging.info(f'Saving: {args.model_coarse}...')
            params = model_coarse.state_dict()
            torch.save(params, args.model_coarse)

    else:
        # Fineモデルの訓練:
        # ニューラルネットワークを訓練モードにする。
        model_coarse.eval()
        model_fine.train()
        # 最適化器と学習率を定義する。
        optimizer = optim.Adam(model_fine.parameters(), lr=args.lr)
        # エポック回だけ訓練を繰り返す。
        for epoch in range(args.epochs):
            logging.info(f'*** epoch={epoch+1}/{args.epochs} ***')
            train(model_coarse, model_fine, device, loader, optimizer,
                  log_interval=args.log_interval, dry_run=args.dry_run)
        # モデルをファイルに保存する。
        if args.model_fine is not None:
            logging.info(f'Saving: {args.model_fine}...')
            params = model_fine.state_dict()
            torch.save(params, args.model_fine)

    return

if __name__ == '__main__': main()
