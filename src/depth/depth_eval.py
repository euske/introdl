#!/usr/bin/env python
##
##  depth_eval.py
##
##  Usage:
##    $ ./depth_eval.py depth_net_coarse.pt depth_net_fine.pt input.jpg ...
##    $ ./depth_eval.py --camera depth_net_coarse.pt depth_net_fine.pt
##
import os.path
import logging
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from depth_net import CoarseNet, FineNet
from depth_utils import rect_fit, image_map, image2torch, render_depth

def resize_image(input_size, image):
    (_,image2input) = rect_fit(image.size, input_size)
    return image_map(input_size, image2input, image)

# detect:
def detect(model_coarse, model_fine, device, images):
    (_,rows,cols) = model_coarse.INPUT_SIZE
    input_size = (cols,rows)
    inputs = []
    for image in images:
        # 画像を入力の大きさに合わせて拡大・縮小する。
        assert image.size == input_size
        inputs.append(image2torch(image))
    inputs = torch.tensor(np.array(inputs)).to(device)
    # 以下の処理ではautograd機能を使わない:
    with torch.no_grad():
        # 与えられたミニバッチをニューラルネットワークに処理させる。
        outputs = model_coarse(inputs)
        if model_fine is not None:
            outputs = model_fine(inputs, outputs)
        outputs = torch.exp(outputs)
    return np.array(outputs.to('cpu'))

def capture(model_coarse, model_fine, device):
    import cv2
    video = cv2.VideoCapture(0)
    logging.info(f'capture: video={video}')
    (_,height,width) = model_coarse.INPUT_SIZE
    while True:
        (ok, image_bgr) = video.read()
        if not ok: break
        image = np.flip(image_bgr, axis=2) # BGR -> RGB
        image = Image.fromarray(image, 'RGB')
        image = resize_image((width,height), image)
        outputs = detect(model_coarse, model_fine, device, [image])
        depth = outputs[0]
        mindepth = np.min(depth)
        maxdepth = np.max(depth)
        logging.info(f'capture: depth={mindepth:.2f}-{maxdepth:.2f}')
        dimage = render_depth(depth)
        size = (width+dimage.width, max(height, dimage.height))
        dst = Image.new(image.mode, size)
        dst.paste(image, (0,0))
        dst.paste(dimage, (width,0))
        dst_bgr = np.flip(dst, axis=2) # RGB -> BGR
        cv2.imshow('DEPTH', np.asarray(dst_bgr))
        if 0 <= cv2.waitKey(1): break
    return

# main
def main():
    import argparse
    # コマンドライン引数を解析する。
    parser = argparse.ArgumentParser(description='Depth Eval')
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
    parser.add_argument('--camera', action='store_true', default=False,
                        help='enables camera detection')
    parser.add_argument('-O', '--output-dir', type=str, metavar='path', default='.',
                        help='output directory')
    parser.add_argument('model_coarse', type=str)
    parser.add_argument('model_fine', type=str)
    parser.add_argument('image_path', type=str, nargs='*')

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
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Coarseモデルをファイルから読み込む。
    logging.info(f'Loading: {args.model_coarse}...')
    params = torch.load(args.model_coarse, map_location=device)
    model_coarse = CoarseNet()
    model_coarse.load_state_dict(params)
    model_coarse = model_coarse.to(device)
    # ニューラルネットワークを評価モードにする。
    model_coarse.eval()
    # Fineモデルをファイルから読み込む。
    model_fine = None
    if args.model_fine:
        # モデルをファイルから読み込む。
        logging.info(f'Loading: {args.model_fine}...')
        try:
            params = torch.load(args.model_fine, map_location=device)
            # Fineモデルを作成。
            model_fine = FineNet()
            model_fine.load_state_dict(params)
            model_fine = model_fine.to(device)
            # ニューラルネットワークを評価モードにする。
            model_fine.eval()
        except FileNotFoundError as e:
            logging.error(f'Error: {e}')

    (_,height,width) = model_coarse.INPUT_SIZE
    for path in args.image_path:
        # 指定された画像を認識する。
        image = Image.open(path)
        logging.info(f'detect: path={path}')
        image = resize_image((width,height), image)
        outputs = detect(model_coarse, model_fine, device, [image])
        depth = outputs[0]
        print(f'depth={depth}')
        if args.output_dir is not None:
            # 結果を保存する。
            (name,_) = os.path.splitext(os.path.basename(path))
            outpath = os.path.join(args.output_dir, f'output_{name}.png')
            dimage = render_depth(depth)
            dimage.save(outpath)
            logging.info(f'save: outpath={outpath}')

    if args.camera:
        # カメラからの動画を認識する。
        capture(model_coarse, model_fine, device)

    return

if __name__ == '__main__': main()
