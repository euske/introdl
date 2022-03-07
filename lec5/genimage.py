#!/usr/bin/env python
from PIL import Image, ImageFilter
img = Image.open('original.png').convert('RGB')
kern1 = img.filter(ImageFilter.Kernel((3,3), [1,1,1, 1,1,1, 1,1,1], 9))
kern1.save('kern1.png')
kern2 = img.filter(ImageFilter.Kernel((3,3), [-1,-1,-1, 0,0,0, 1,1,1], 1))
kern2.save('kern2.png')
kern3 = img.filter(ImageFilter.Kernel((3,3), [-1,0,1, -1,0,1, -1,0,1], 1))
kern3.save('kern3.png')
