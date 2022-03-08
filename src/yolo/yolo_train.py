#!/usr/bin/env python
##
##  yolo_train.py - Mini YOLO training.
##
##  usage:
##    $ ./yolo_train.py --save-model yolo_net.pt VOC2007.zip
##
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from yolo_net import YOLONet
from yolo_utils import VOCDataset, YOLOCell, YOLOObject
from yolo_utils import rect_intersect, rect_fit, rect_map, image_map, annot_map, image2torch

# convert_grid: 入力をグリッドに区切る。
def convert_grid(dataset, samples, input_size, output_size):
    (width, height) = input_size
    (rows, cols) = output_size
    inputs = []
    targets = []
    for (image, annot) in samples:
        # 画像を入力の大きさに合わせて拡大・縮小する。
        (_,image2input) = rect_fit(image.size, input_size)
        image = image_map(input_size, image2input, image)
        annot = annot_map(input_size, image2input, annot)
        inputs.append(image2torch(image))
        # 物体の矩形を (rows×cols)のセルに区切る。
        cells = {}
        for (name, (x1,y1,w1,h1)) in annot:
            cat = dataset.CAT2INDEX.get(name)
            if cat is None: continue
            (cx,cy) = ((x1+w1/2)/width, (y1+h1/2)/height)
            cell = YOLOCell(cx, cy, w1/width, h1/height, cat=cat)
            pos = (int(cx*cols), int(cy*rows))
            if pos in cells:
                objs = cells[pos]
            else:
                objs = cells[pos] = []
            objs.append(cell)
        for objs in cells.values():
            objs.sort(key=lambda cell: cell.w*cell.h, reverse=True)
        targets.append(cells)
    assert len(inputs) == len(targets)
    return (inputs, targets)

# train: 1エポック分の訓練をおこなう。
def train(model, device, loader, optimizer, log_interval=1, dry_run=False):
    (_,height,width) = model.INPUT_SIZE
    (rows,cols,_) = model.OUTPUT_SIZE
    dataset = loader.dataset
    # 各ミニバッチを処理する。
    for (idx, samples) in enumerate(loader):
        # 正解データをセルに変換する。
        (inputs, targets) = convert_grid(dataset, samples, (width,height), (rows,cols))
        inputs = torch.tensor(np.array(inputs)).to(device)
        # すべての勾配(.grad)をクリアしておく。
        optimizer.zero_grad()
        # 与えられたミニバッチをニューラルネットワークに処理させる。
        outputs_raw = model(inputs).to('cpu')
        outputs = []
        for grid in outputs_raw.reshape(-1, rows, cols, model.NVALS):
            # 各セルの出力を YOLOCell インスタンスに変換。
            cells = {}
            for (i,row) in enumerate(grid):
                for (j,v) in enumerate(row):
                    cx = (j+torch.sigmoid(v[0]))/cols
                    cy = (i+torch.sigmoid(v[1]))/rows
                    w = torch.sigmoid(v[2])
                    h = torch.sigmoid(v[3])
                    conf = torch.sigmoid(v[4])
                    cprobs = F.log_softmax(v[5:], dim=0)
                    pos = (j,i)
                    if pos in cells:
                        a = cells[pos]
                    else:
                        a = cells[pos] = []
                    a.append(YOLOCell(cx, cy, w, h, conf=conf, cprobs=cprobs))
            outputs.append(cells)
        # 損失を計算する。
        loss = torch.tensor(0.)
        for (cells, cells_ref) in zip(outputs, targets):
            # 各セルの損失を合計する。
            for (pos, a) in cells.items():
                if pos in cells_ref:
                    for (cell, obj_ref) in zip(a, cells_ref[pos]):
                        loss += cell.get_cost_full(obj_ref)
                else:
                    for cell in a:
                        loss += cell.get_cost_noobj()
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
    parser = argparse.ArgumentParser(description='YOLO Train')
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
    parser.add_argument('voc_data', type=str)

    args = parser.parse_args()

    # ログ出力を設定する。
    level = (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=level)

    # 乱数シードを設定する。
    torch.manual_seed(args.seed)

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
    dataset = VOCDataset(args.voc_data)
    loader = DataLoader(dataset, collate_fn=lambda x:x, **train_kwargs)

    # モデルを作成。
    model = YOLONet()
    if args.save_model is not None:
        # モデルをファイルから読み込む。
        logging.info(f'Loading: {args.save_model}...')
        try:
            params = torch.load(args.save_model, map_location=device)
            model.load_state_dict(params)
        except FileNotFoundError as e:
            logging.error(f'Error: {e}')
    model = model.to(device)

    # ニューラルネットワークを訓練モードにする。
    model.train()

    # 最適化器と学習率を定義する。
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # エポック回だけ訓練を繰り返す。
    for epoch in range(args.epochs):
        logging.info(f'*** epoch={epoch+1}/{args.epochs} ***')
        train(model, device, loader, optimizer,
              log_interval=args.log_interval, dry_run=args.dry_run)

    # モデルをファイルに保存する。
    if args.save_model is not None:
        logging.info(f'Saving: {args.save_model}...')
        params = model.state_dict()
        torch.save(params, args.save_model)

    return

if __name__ == '__main__': main()
