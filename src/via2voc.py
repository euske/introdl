#!/usr/bin/env python
#
#  via2voc.py - converts VIA json export to VOC xml format.
#
#  Usage:
#    $ python via2voc.py -O outdir output_json.json
#
import sys
import json
import os.path
from xml.etree.ElementTree import Element, tostring

def getimgsize(path):
    from PIL import Image
    img = Image.open(path)
    return img.size

def addelem(p, k, v):
    e = Element(k)
    e.text = str(v)
    p.append(e)

def via2voc(x, imgsize=None):
    annotation = Element('annotation')
    addelem(annotation, 'filename', x['filename'])
    if imgsize is not None:
        (width,height) = imgsize
        size = Element('size')
        addelem(size, 'width', width)
        addelem(size, 'height', height)
        annotation.append(size)
    for region in x['regions']:
        obj = Element('object')
        attrs = region['region_attributes']
        for (k,v) in attrs.items():
            addelem(obj, k, v)
        shape = region['shape_attributes']
        if shape['name'] == 'rect':
            bndbox = Element('bndbox')
            addelem(bndbox, 'xmin', shape['x'])
            addelem(bndbox, 'ymin', shape['y'])
            addelem(bndbox, 'xmax', shape['x']+shape['width'])
            addelem(bndbox, 'ymax', shape['y']+shape['height'])
            obj.append(bndbox)
        annotation.append(obj)
    return annotation

def main(argv):
    import getopt
    def usage():
        print(f'usage: {argv[0]} [-i imgdir] [-O outdir] [file ...]')
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'i:O:')
    except getopt.GetoptError:
        return usage()
    imgdir = None
    outdir = '.'
    for (k, v) in opts:
        if k == '-i': imgdir = v
        elif k == '-O': outdir = v
    for path in args:
        print(f'loading: {path}')
        with open(path) as fp:
            data = json.load(fp)
            for x in data.values():
                filename = x['filename']
                imgsize = None
                if imgdir is not None:
                    imgsize = getimgsize(os.path.join(imgdir, filename))
                annotation = via2voc(x, imgsize=imgsize)
                (name,_) = os.path.splitext(filename)
                outpath = os.path.join(outdir, name+'.xml')
                with open(outpath, 'wb') as out:
                    out.write(tostring(annotation))
                print(f'saved: {outpath}')
    return 0

if __name__ == '__main__': sys.exit(main(sys.argv))
