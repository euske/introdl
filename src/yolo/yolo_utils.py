#!/usr/bin/env python
##
##  yolo_utils.py - YOLO Utilities
##
##  usage: (show annotations)
##    $ ./yolo_utils.py -n10 VOC2007.zip
##  usage: (save annotated images)
##    $ ./yolo_utils.py -n10 -O. VOC2007.zip
##
import sys
import os.path
import zipfile
import math
import numpy as np
import logging
import json
from torch.utils.data import Dataset
from PIL import Image
from xml.etree.ElementTree import XML


##  Utils.
##

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

# render_objs: 画像中に物体の枠を描画する。
COLORS = (
    'red', 'green', 'magenta', 'cyan',
    'yellow', 'gray', 'orange', 'brown', 'blue',
    )
def render_objs(image, objs):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    for obj in objs:
        # simplistic hashing.
        z = sum( ord(c)*(i+7) for (i,c) in enumerate(obj.name) )
        color = COLORS[z % len(COLORS)]
        (x,y,w,h) = obj.bbox
        (tw,th) = draw.textsize(obj.name)
        draw.rectangle((x,y,x+w,y+h), outline=color, width=2)
        draw.rectangle((x,y-th-2,x+tw+2,y), fill=color)
        draw.text((x+2,y-th), obj.name, fill='black')
    return image

# rect_split: returns [((x,y),(rx,ry,rw,rh)), ...]
def rect_split(rect, grid):
    (x,y,w,h) = rect
    (rows,cols) = grid
    for i in range(rows):
        for j in range(cols):
            yield ((i,j), (x+j*w//cols, y+i*h//rows, w//cols, h//rows))
    return

# rect_intersect: 2つの矩形の交差を求める。
def rect_intersect(rect0, rect1):
    (x0,y0,w0,h0) = rect0
    (x1,y1,w1,h1) = rect1
    x = max(x0, x1)
    y = max(y0, y1)
    w = min(x0+w0, x1+w1) - x
    h = min(y0+h0, y1+h1) - y
    return (x, y, w, h)

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

# rect_map: rectを写像した矩形を求める。
def rect_map(mapping, rect):
    (wd,hd,w0,h0,xd,yd) = mapping
    (x,y,w,h) = rect
    return (x*wd//w0+xd, y*hd//h0+yd, w*wd//w0, h*hd//h0)
assert rect_map((10,10,100,100,0,0), (10,10,20,20)) == (1,1,2,2)
assert rect_map((10,10,100,100,-5,-5), (50,0,50,100)) == (0,-5,5,10)

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

# annot_map: アノテーションを写像したものを返す。
def annot_map(size, mapping, annot):
    frame = (0,0,)+size
    a = []
    for (name, bbox) in annot:
        bbox = rect_map(mapping, bbox)
        (x,y,w,h) = rect_intersect(frame, bbox)
        if w <= 0 or h <= 0: continue
        a.append((name, bbox))
    return a


##  VOCDataset
##  PASCAL VOC データセットを使うためのクラス。
##
class VOCDataset(Dataset):

    # CATEGORIES: 認識する物体の種類。
    # (コメントアウトしたものは認識されない)
    CATEGORIES = [
        (None, 0),                  # 0
        ('person', 5447),           # 1
        ('car', 1644),              # 2
        #('aeroplane', 331),         # 3
        #('bicycle', 418),           # 4
        #('bird', 599),              # 5
        #('boat', 398),              # 6
        #('bottle', 634),            # 7
        #('bus', 272),               # 8
        #('cat', 389),               # 9
        #('chair', 1432),            # 10
        #('cow', 356),               # 11
        #('diningtable', 310),       # 12
        #('dog', 538),               # 13
        #('horse', 406),             # 14
        #('motorbike', 390),         # 15
        #('pottedplant', 625),       # 16
        #('sheep', 353),             # 17
        #('sofa', 425),              # 18
        #('train', 328),             # 19
        #('tvmonitor', 367)          # 20
    ]
    # CAT2INDEX: カテゴリ名を数値に変換。
    CAT2INDEX = { k:idx for (idx,(k,_)) in enumerate(CATEGORIES) }

    def __init__(self, zip_path,
                 split='train', basedir='VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'):
        super().__init__()
        self.logger = logging.getLogger('VOCDataset')
        self.image_base = f'{basedir}/JPEGImages/'
        self.annot_base = f'{basedir}/Annotations/'
        self.data_zip = zipfile.ZipFile(zip_path)
        self.logger.info(f'zip_path={zip_path}, split={split}')
        path =  f'{basedir}/ImageSets/Main/{split}.txt'
        with self.data_zip.open(path) as fp:
            data = fp.read().decode('utf-8')
            self.keys = sorted( line.strip() for line in data.splitlines() )
        self.logger.info(f'images={len(self.keys)}')
        return

    def close(self):
        if self.data_zip is not None:
            self.data_zip.close()
            self.data_zip = None
        return

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.data_zip is None:
            raise IOError('file already closed')
        k = self.keys[index]
        image_path = self.image_base + k + '.jpg'
        with self.data_zip.open(image_path) as fp:
            image = Image.open(fp)
            image.load()
            image.filename = image_path
        annot_path = self.annot_base + k + '.xml'
        with self.data_zip.open(annot_path) as fp:
            elem = XML(fp.read())
        assert elem.tag == 'annotation'
        annot = []
        for obj in elem:
            if obj.tag == 'filename':
                filename = obj.text
                assert filename == k+'.jpg'
            elif obj.tag == 'object':
                name = None
                x0 = x1 = y0 = y1 = None
                for e in obj:
                    if e.tag == 'name':
                        name = e.text
                    elif e.tag == 'bndbox':
                        for c in e:
                            if c.tag == 'xmin':
                                x0 = int(c.text)
                            elif c.tag == 'xmax':
                                x1 = int(c.text)
                            elif c.tag == 'ymin':
                                y0 = int(c.text)
                            elif c.tag == 'ymax':
                                y1 = int(c.text)
                if (name is not None and x0 is not None and x1 is not None and
                    y0 is not None and y1 is not None):
                    annot.append((name, (x0,y0,x1-x0,y1-y0)))
        return (image, annot)


##  COCODataset
##  COCO データセットを使うためのクラス。
##
class COCODataset(Dataset):

    # CATEGORIES: 認識する物体の種類。
    # (コメントアウトしたものは認識されない)
    CATEGORIES = [
        (None, 0),                  # 0
        ('person', 262465),         # 1
        ('bicycle', 7113),          # 2
        ('car', 43867),             # 3
        ('motorcycle', 8725),       # 4
        ('airplane', 5135),         # 5
        ('bus', 6069),              # 6
        ('train', 4571),            # 7
        ('truck', 9973),            # 8
        ('boat', 10759),            # 9
        ('traffic light', 12884),   # 10
        ('fire hydrant', 1865),     # 11
        ('stop sign', 1983),        # 12
        ('parking meter', 1285),    # 13
        ('bench', 9838),            # 14
        ('bird', 10806),            # 15
        ('cat', 4768),              # 16
        ('dog', 5508),              # 17
        ('horse', 6587),            # 18
        ('sheep', 9509),            # 19
        ('cow', 8147),              # 20
        ('elephant', 5513),         # 21
        ('bear', 1294),             # 22
        ('zebra', 5303),            # 23
        ('giraffe', 5131),          # 24
        ('backpack', 8720),         # 25
        ('umbrella', 11431),        # 26
        ('handbag', 12354),         # 27
        ('tie', 6496),              # 28
        ('suitcase', 6192),         # 29
        ('frisbee', 2682),          # 30
        ('skis', 6646),             # 31
        ('snowboard', 2685),        # 32
        ('sports ball', 6347),      # 33
        ('kite', 9076),             # 34
        ('baseball bat', 3276),     # 35
        ('baseball glove', 3747),   # 36
        ('skateboard', 5543),       # 37
        ('surfboard', 6126),        # 38
        ('tennis racket', 4812),    # 39
        ('bottle', 24342),          # 40
        ('wine glass', 7913),       # 41
        ('cup', 20650),             # 42
        ('fork', 5479),             # 43
        ('knife', 7770),            # 44
        ('spoon', 6165),            # 45
        ('bowl', 14358),            # 46
        ('banana', 9458),           # 47
        ('apple', 5851),            # 48
        ('sandwich', 4373),         # 49
        ('orange', 6399),           # 50
        ('broccoli', 7308),         # 51
        ('carrot', 7852),           # 52
        ('hot dog', 2918),          # 53
        ('pizza', 5821),            # 54
        ('donut', 7179),            # 55
        ('cake', 6353),             # 56
        ('chair', 38491),           # 57
        ('couch', 5779),            # 58
        ('potted plant', 8652),     # 59
        ('bed', 4192),              # 60
        ('dining table', 15714),    # 61
        ('toilet', 4157),           # 62
        ('tv', 5805),               # 63
        ('laptop', 4970),           # 64
        ('mouse', 2262),            # 65
        ('remote', 5703),           # 66
        ('keyboard', 2855),         # 67
        ('cell phone', 6434),       # 68
        ('microwave', 1673),        # 69
        ('oven', 3334),             # 70
        ('toaster', 225),           # 71
        ('sink', 5610),             # 72
        ('refrigerator', 2637),     # 73
        ('book', 24715),            # 74
        ('clock', 6334),            # 75
        ('vase', 6613),             # 76
        ('scissors', 1481),         # 77
        ('teddy bear', 4793),       # 78
        ('hair drier', 198),        # 79
        ('toothbrush', 1954),       # 80
    ]
    # CAT2INDEX: カテゴリ名を数値に変換。
    CAT2INDEX = { k:idx for (idx,(k,_)) in enumerate(CATEGORIES) }

    def __init__(self, image_path, annot_path):
        super().__init__()
        self.logger = logging.getLogger('COCODataset')
        self.image_path = image_path
        self.annot_path = annot_path
        self.logger.info(f'image_path={self.image_path}')
        self.image_zip = zipfile.ZipFile(self.image_path)
        images = {}
        for name in self.image_zip.namelist():
            if name.endswith('/'): continue
            (image_id,ext) = os.path.splitext(os.path.basename(name))
            if ext != '.jpg': continue
            images[int(image_id)] = name
        self.logger.info(f'images={len(images)}')
        annots = {}
        self.logger.info(f'annot_path={self.annot_path}')
        with open(self.annot_path) as fp:
            objs = json.load(fp)
            id2name = {}
            for obj in objs['categories']:
                cat_id = obj['id']
                cat_name = obj['name']
                assert cat_name in self.CAT2INDEX, cat_name
                id2name[cat_id] = cat_name
            for obj in objs['annotations']:
                cat_id = obj['category_id']
                if cat_id not in id2name: continue
                image_id = obj['image_id']
                bbox = obj['bbox']
                if image_id in annots:
                    a = annots[image_id]
                else:
                    a = annots[image_id] = []
                a.append((id2name[cat_id], bbox))
        self.logger.info(f'annots={len(annots)}')
        self.data = [ (images[i], annots.get(i)) for i in sorted(images.keys()) ]
        print(id2name)
        return

    def close(self):
        if self.image_zip is not None:
            self.image_zip.close()
            self.image_zip = None
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (name, annot) = self.data[index]
        with self.image_zip.open(name) as fp:
            image = Image.open(fp)
            image.load()
            image.filename = name
        return (image, annot or [])


##  YOLOCell
##
def argmax(a, key=lambda x:x):
    (imax, vmax) = (None, None)
    for (i,x) in enumerate(a):
        v = key(x)
        if vmax is None or vmax < v:
            (imax, vmax) = (i, v)
    if imax is None: raise ValueError(a)
    return (imax, vmax)

def nll(cat, cprobs):
    return -cprobs[cat]

class YOLOCell:

    L_NOOBJ = 0.5
    L_COORD = 5.0

    def __init__(self, cx, cy, w, h, cat=None, conf=None, cprobs=None):
        assert cat is not None or conf is not None
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.cat = cat
        self.conf = conf
        self.cprobs = cprobs
        return

    def __repr__(self):
        if self.cat is not None:
            return (f'<YOLOCell: cat={self.cat},'
                    f' center=({self.cx:.2f},{self.cy:.2f}), size=({self.w:.2f},{self.h:.2f})>')
        else:
            (cat, prob) = self.get_cat()
            return (f'<YOLOCell: conf={self.conf:.3f}, cat={cat}({prob:.2f}),'
                    f' center=({self.cx:.2f},{self.cy:.2f}), size=({self.w:.2f},{self.h:.2f})>')

    def get_bbox(self):
        return (float(self.cx-self.w/2), float(self.cy-self.h/2),
                float(self.w), float(self.h))

    def get_cat(self):
        if self.cat is not None: return (self.cat, 1.0)
        (cat, prob) = argmax(self.cprobs)
        return (cat, math.exp(prob))

    def get_cost_noobj(self):
        assert self.cprobs is not None
        return (self.L_NOOBJ *
                (self.conf - 0)**2 +
                nll(0, self.cprobs))

    def get_cost_full(self, obj):
        assert self.cprobs is not None
        assert obj.cat is not None
        (_,_,w,h) = rect_intersect(self.get_bbox(), obj.get_bbox())
        if w < 0 or h < 0:
            iou = 0
        else:
            iou = (w*h)/(self.w * self.h)
        return (self.L_COORD *
                ((self.cx - obj.cx)**2 +
                 (self.cy - obj.cy)**2 +
                 (self.w - obj.w)**2 +
                 (self.h - obj.h)**2) +
                (self.conf - iou)**2 +
                nll(obj.cat, self.cprobs))


##  YOLOObject
##
class YOLOObject:

    def __init__(self, name, conf, bbox):
        self.name = name
        self.conf = conf
        self.bbox = bbox
        return

    def __repr__(self):
        return (f'<YOLOObject({self.name}): conf={self.conf:.3f}, bbox={self.bbox}>')

    def get_iou(self, bbox):
        (_,_,w,h) = rect_intersect(self.bbox, bbox)
        if w <= 0 or h <= 0: return 0
        (_,_,w0,h0) = self.bbox
        return (w*h)/(w0*h0)

# soft_nms: https://arxiv.org/abs/1704.04503
def soft_nms(objs, threshold):
    result = []
    score = { obj:obj.conf for obj in objs }
    while objs:
        (i,conf) = argmax(objs, key=lambda obj:score[obj])
        if conf < threshold: break
        m = objs[i]
        result.append(m)
        del objs[i]
        for obj in objs:
            v = m.get_iou(obj.bbox)
            score[obj] = score[obj] * math.exp(-3*v*v)
    result.sort(key=lambda obj:score[obj], reverse=True)
    return result


# main
def main(argv):
    import getopt
    def usage():
        print(f'usage: {argv[0]} [-v] [-O output] [-n images] voc.zip')
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'vO:n:')
    except getopt.GetoptError:
        return usage()
    level = logging.INFO
    output_dir = None
    num_images = 10
    for (k, v) in opts:
        if k == '-v': level = logging.DEBUG
        elif k == '-O': output_dir = v
        elif k == '-n': num_images = int(v)
    if not args: return usage()

    logging.basicConfig(level=level)

    path = args.pop(0)
    dataset = VOCDataset(path)

    if output_dir is not None:
        # Save annotated images.
        n = 0
        for (image,annot) in dataset:
            (k,_) = os.path.splitext(os.path.basename(image.filename))
            output = os.path.join(output_dir, f'image_{k}.png')
            print(f'image {k}: size={image.size}, annot={len(annot)}')
            objs = []
            for (name, bbox) in annot:
                if name not in dataset.CAT2INDEX: continue
                objs.append(YOLOObject(name, 1, bbox))
            image = render_objs(image, objs)
            image.save(output)
            n += 1
            if n == num_images: break
    else:
        # Show annotations.
        n = 0
        for (image,annot) in dataset:
            (k,_) = os.path.splitext(os.path.basename(image.filename))
            print(f'image {k}: size={image.size}')
            for (i,(name,(x,y,w,h))) in enumerate(annot):
                print(f'  {i}: {name} ({x},{y}) {w}x{h}')
            print()
            n += 1
            if n == num_images: break

    return

if __name__ == '__main__':
    sys.exit(main(sys.argv))
