#!/usr/bin/env python
##
##  depth_utils.py - NYU DEPTH Utilities
##
##  usage: (save images)
##    $ ./depth_utils.py -n10 -O. nyu_depth.mat
##
import sys
import os.path
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


##  NYUDepth2Dataset
##
class NYUDepth2Dataset(Dataset):

    def __init__(self, path):
        super().__init__()
        import h5py
        self.logger = logging.getLogger()
        self.fp = h5py.File(path)
        self.images = self.fp['images']
        self.depths = self.fp['depths']
        assert self.images.shape[0] == self.depths.shape[0]
        return

    def close(self):
        if self.fp is not None:
            self.fp.close()
            self.fp = None
        return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.fp is None:
            raise IOError('file already closed')
        image = self.images[index]  # (C×W×H)
        assert len(image.shape) == 3
        # 画像: (C×W×H) -> (C×H×W)
        image = image.transpose(0,2,1)
        depth = self.depths[index]  # (W×H)
        assert len(depth.shape) == 2
        # 深さ行列: (W×H) -> (H×W)
        depth = depth.transpose(1,0)
        return (image, depth)


# conv_image: 入力画像を補正する。
def conv_image(array, size, flip=False):
    assert len(array.shape) == 3
    (rows1, cols1) = size
    array = array[:, ::2, ::(-2 if flip else 2)]
    (_, rows0, cols0) = array.shape
    assert rows1 <= rows0 and cols1 <= cols0
    array = array[:, (rows0-rows1)//2:(rows0+rows1)//2, (cols0-cols1)//2:(cols0+cols1)//2]
    # ランダムに色調を変える。
    g = np.random.rand(3,1,1) + 0.7
    array = (array/255 ** g)
    # 平均・分散を正規化する。
    a = array.astype(np.float32)
    a[0] = (a[0]-0.485)/0.229
    a[1] = (a[1]-0.456)/0.224
    a[2] = (a[2]-0.406)/0.225
    return a

# conv_depth: 深さ行列を補正する。
def conv_depth(array, size, flip=False):
    assert len(array.shape) == 2
    (rows1, cols1) = size
    array = array[::8, ::(-8 if flip else 8)]
    (rows0, cols0) = array.shape
    assert rows1 <= rows0 and cols1 <= cols0
    array = array[(rows0-rows1)//2:(rows0+rows1)//2, (cols0-cols1)//2:(cols0+cols1)//2]
    return array.astype(np.float32)

# rect_fit: size1の矩形を拡大・縮小し、
#   size0の矩形と中心を合わせるような写像 (size1->size0, size0->size1) を返す。
def rect_fit(size0, size1):
    (w0,h0) = size0
    (w1,h1) = size1
    if h1*w0 < w1*h0:
        # horizontal fit (w1->w0)
        (wd,hd) = (w0, h1*w0//w1) # < (w0,h0)
        (ws,hs) = (w1, h0*w1//w0) # > (w1,h1)
    else:
        # vertical fit (h1->h0)
        (wd,hd) = (w1*h0//h1, h0) # < (w0,h0)
        (ws,hs) = (w0*h1//h0, h1) # > (w1,h1)
    (xd,yd) = ((w0-wd)//2, (h0-hd)//2)
    (xs,ys) = ((w1-ws)//2, (h1-hs)//2)
    # (x0,y0) = (x1*wd/w1+xd, y1*hd/h1+yd)
    # (x1,y1) = (x0*ws/w0+xs, y0*hs/h0+ys)
    return ((wd,hd,w1,h1,xd,yd), (ws,hs,w0,h0,xs,ys))
assert rect_fit((100,100), (200,200)) == ((100,100,200,200,0,0), (200,200,100,100,0,0))
assert rect_fit((200,200), (100,100)) == ((200,200,100,100,0,0), (100,100,200,200,0,0))
assert rect_fit((100,100), (100,200)) == ((50,100,100,200,25,0), (200,200,100,100,-50,0))
assert rect_fit((100,200), (100,100)) == ((100,100,100,100,0,50), (100,200,100,200,0,-50))
assert rect_fit((200,100), (100,200)) == ((50,100,100,200,75,0), (400,200,200,100,-150,0))
assert rect_fit((100,200), (200,100)) == ((100,50,200,100,0,75), (200,400,100,200,0,-150))

# image_map: imageを写像した新しい画像を返す。
#
# 画像がはみ出ないように拡大:
#   (mapping,_) = rect_fit(output_size, image.size)
#   image = image_map(output_size, mapping, image)
# 画像がはみ出るように拡大:
#   (_,mapping) = rect_fit(image.size, output_size)
#   image = image_map(output_size, mapping, image)
#
def image_map(size, mapping, image):
    (wd,hd,_,_,xd,yd) = mapping
    src = image.resize((wd,hd))
    dst = Image.new(image.mode, size)
    dst.paste(src, (xd,yd))
    return dst

# image2torch: PIL画像 (H×W×C) を Torch用入力 (C×H×W) に変換する。
def image2torch(image):
    assert image.mode == 'RGB'
    a = np.array(image, dtype=np.float32)
    # [ [[R1,G1,B1], ...], [[R2,G2,B2], ...], ... ] ->
    # [ [[R1,] , [R2,], ...], [[G1,], [G2,], ...], [[B1,], [B2,], ...] ]
    a = a.transpose(2,0,1) / 255
    # 平均・分散を正規化する。
    a[0] = (a[0]-0.485)/0.229
    a[1] = (a[1]-0.456)/0.224
    a[2] = (a[2]-0.406)/0.225
    return a

# render_depth: 深さ行列 (H×W) を画像に変換。
def render_depth(array, mindepth=0.5, maxdepth=10.0):
    array = np.minimum(array, maxdepth)
    array = np.maximum(array, mindepth)
    # depthが小さいほど明るい色になり、大きいと暗い色になる。
    depth = ((maxdepth-array)*255/(maxdepth-mindepth)).astype(np.uint8)
    image = Image.fromarray(depth, 'L')
    return image

# main
def main(argv):
    import getopt
    def usage():
        print(f'usage: {argv[0]} [-v] [-O output] [-n images] depth.mat')
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'vO:n:')
    except getopt.GetoptError:
        return usage()
    level = logging.INFO
    output_dir = '.'
    num_images = 10
    for (k, v) in opts:
        if k == '-v': level = logging.DEBUG
        elif k == '-O': output_dir = v
        elif k == '-n': num_images = int(v)
    if not args: return usage()

    logging.basicConfig(level=level)

    path = args.pop(0)
    dataset = NYUDepth2Dataset(path)

    for (i,(aimg, depth)) in enumerate(dataset):
        output0 = os.path.join(output_dir, f'image_{i:04}.png')
        output1 = os.path.join(output_dir, f'depth_{i:04}.png')
        # (C×H×W) -> (H×W×C)
        aimg = aimg.transpose(1,2,0)
        image = Image.fromarray(aimg, 'RGB')
        image.save(output0)
        mindepth = np.min(depth)
        maxdepth = np.max(depth)
        image = render_depth(depth, mindepth=mindepth, maxdepth=maxdepth)
        image.save(output1)
        logging.info(f'save: output={output0},{output1}')
        if i+1 == num_images: break

    dataset.close()
    return

if __name__ == '__main__':
    sys.exit(main(sys.argv))
