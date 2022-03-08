#!/usr/bin/env python
##
##  yolo_eval.py - Mini YOLO testing.
##
##  Usage:
##    $ ./yolo_eval.py --voc-data VOC2007.zip yolo_net.pt
##    $ ./yolo_eval.py yolo_net.pt input.jpg ...
##    $ ./yolo_eval.py --camera yolo_net.pt
##
import os.path
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from yolo_net import YOLONet
from yolo_utils import VOCDataset, YOLOCell, YOLOObject
from yolo_utils import rect_fit, image_map, annot_map, image2torch, soft_nms, render_objs

def resize_image(input_size, image):
    (_,image2input) = rect_fit(image.size, input_size)
    return image_map(input_size, image2input, image)

# detect:
def detect(categories, model, device, images, threshold=0.20):
    (_,height,width) = model.INPUT_SIZE
    (rows,cols,_) = model.OUTPUT_SIZE
    inputs = []
    for image in images:
        assert image.size == (width,height)
        inputs.append(image2torch(image))
    inputs = torch.tensor(np.array(inputs)).to(device)
    # 以下の処理ではautograd機能を使わない:
    with torch.no_grad():
        # 与えられたミニバッチをニューラルネットワークに処理させる。
        outputs_raw = model(inputs).to('cpu')
    outputs = []
    for (i,grid) in enumerate(outputs_raw.reshape(-1, rows, cols, model.NVALS)):
        image = images[i]
        # 各セルの出力を YOLOCell インスタンスに変換し、
        # さらに YOLOObject を取得。
        found = []
        for (i,row) in enumerate(grid):
            for (j,v) in enumerate(row):
                cx = (j+torch.sigmoid(v[0]))/cols
                cy = (i+torch.sigmoid(v[1]))/rows
                w = torch.sigmoid(v[2])
                h = torch.sigmoid(v[3])
                conf = torch.sigmoid(v[4])
                cprobs = F.log_softmax(v[5:], dim=0)
                cell = YOLOCell(cx, cy, w, h, conf=conf, cprobs=cprobs)
                (cat, prob) = cell.get_cat()
                if cat != 0:
                    (x,y,w,h) = cell.get_bbox()
                    bbox = (x*width, y*height, w*width, h*height)
                    (name,_) = categories[cat]
                    obj = YOLOObject(name, cell.conf*prob, bbox)
                    found.append(obj)
        found = soft_nms(found, threshold)
        outputs.append(found)
    return outputs

# test: テストをおこなう。
def test(model, device, loader, threshold=0.20, iou=0.50):
    (_,height,width) = model.INPUT_SIZE
    input_size = (width,height)
    # 各ミニバッチを処理する。
    total = tap = 0
    dataset = loader.dataset
    for (idx, samples) in enumerate(loader):
        images = []
        annots = []
        for (image,annot) in samples:
            (_,image2input) = rect_fit(image.size, input_size)
            images.append(image_map(input_size, image2input, image))
            annots.append(annot_map(input_size, image2input, annot))
        outputs = detect(dataset.CATEGORIES, model, device, images, threshold=threshold)
        for (annot, found) in zip(annots, outputs):
            # 認識対象の種類だけに限定する。
            annot = [ (name,bbox) for (name,bbox) in annot if name in dataset.CAT2INDEX ]
            if not annot: continue
            # Average Precision を計算する。
            ap = 0.0
            prec0 = 1.0
            recl0 = 0.0
            done = set()
            correct = 0
            for (i,obj) in enumerate(found):
                for (name, bbox) in annot:
                    bbox = tuple(bbox)
                    if (name, bbox) in done: continue
                    if name == obj.name and iou <= obj.get_iou(bbox):
                        done.add((name, bbox))
                        correct += 1
                        break
                recl = correct/len(annot)
                prec = correct/(i+1)
                if prec < prec0:
                    ap += prec0 * (recl - recl0)
                    recl0 = recl
                prec0 = prec
            recl = correct/len(annot)
            ap += prec0 * (recl - recl0)
            tap += ap
            total += 1
            logging.debug(f'test: found={len(found)}, annot={len(annot)}, correct={correct}, ap={ap:.2f}')
    # 結果を表示する。
    logging.info(f'test: total={total}, mAP={tap/total}')
    return

def capture(categories, model, device, threshold=0.20):
    import cv2
    video = cv2.VideoCapture(0)
    logging.info(f'capture: video={video}')
    (_,height,width) = model.INPUT_SIZE
    while True:
        (ok, image_bgr) = video.read()
        if not ok: break
        image = np.flip(image_bgr, axis=2) # BGR -> RGB
        image = Image.fromarray(image, 'RGB')
        image = resize_image((width,height), image)
        outputs = detect(categories, model, device, [image], threshold=threshold)
        found = outputs[0]
        print('Detected:', found)
        image = render_objs(image, found)
        image = np.asarray(image)
        image_bgr = np.flip(image, axis=2) # RGB -> BGR
        cv2.imshow('YOLO', image_bgr)
        if 0 <= cv2.waitKey(1): break
    return

# main
def main():
    import argparse
    # コマンドライン引数を解析する。
    parser = argparse.ArgumentParser(description='YOLO Eval')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='enables verbose logging')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--voc-data', type=str, metavar='path', default=None,
                        help='VOC test data')
    parser.add_argument('--camera', action='store_true', default=False,
                        help='enables camera detection')
    parser.add_argument('-t', '--threshold', type=float, default=0.20,
                        help='detection threshold')
    parser.add_argument('-O', '--output-dir', type=str, metavar='path', default='.',
                        help='output directory')
    parser.add_argument('model_path', type=str)
    parser.add_argument('image_path', type=str, nargs='*')

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
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    categories = VOCDataset.CATEGORIES

    # モデルをファイルから読み込む。
    model = YOLONet()
    logging.info(f'Loading: {args.model_path}...')
    params = torch.load(args.model_path, map_location=device)
    model.load_state_dict(params)
    model = model.to(device)
    # 検出しきい値。
    threshold = args.threshold

    # ニューラルネットワークを評価モードにする。
    model.eval()

    (_,height,width) = model.INPUT_SIZE
    for path in args.image_path:
        # 指定された画像を認識する。
        image = Image.open(path)
        logging.info(f'detect: path={path}')
        image = resize_image((width,height), image)
        outputs = detect(categories, model, device, [image], threshold=threshold)
        found = outputs[0]
        print(f'found={found}')
        if args.output_dir is not None:
            # 結果を保存する。
            image = render_objs(image, found)
            (name,_) = os.path.splitext(os.path.basename(path))
            outpath = os.path.join(args.output_dir, f'output_{name}.png')
            image.save(outpath)
            logging.info(f'save: outpath={outpath}')

    if args.voc_data is not None:
        # テストデータで評価する。
        dataset = VOCDataset(
            args.voc_data, split='test.txt',
            basedir='VOCtest_06-Nov-2007/VOCdevkit/VOC2007')
        logging.info(f'test: voc_data={args.voc_data}')
        loader = DataLoader(dataset, collate_fn=lambda x:x, **test_kwargs)
        test(model, device, loader, threshold=threshold)

    if args.camera:
        # カメラからの動画を認識する。
        capture(categories, model, device, threshold=threshold)

    return

if __name__ == '__main__': main()
