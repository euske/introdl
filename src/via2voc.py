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

def addelem(p, k, v):
    e = Element(k)
    e.text = str(v)
    p.append(e)

def via2voc(x):
    annotation = Element('annotation')
    addelem(annotation, 'filename', x['filename'])
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
        print(f'usage: {argv[0]} [-O outdir] [file ...]')
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'O:')
    except getopt.GetoptError:
        return usage()
    outdir = '.'
    for (k, v) in opts:
        if k == '-O': outdir = v
    for path in args:
        print(f'loading: {path}')
        with open(path) as fp:
            data = json.load(fp)
            for x in data.values():
                filename = x['filename']
                annotation = via2voc(x)
                (name,_) = os.path.splitext(filename)
                outpath = os.path.join(outdir, name+'.xml')
                with open(outpath, 'wb') as out:
                    out.write(tostring(annotation))
                print(f'saved: {outpath}')
    return 0

if __name__ == '__main__': sys.exit(main(sys.argv))
