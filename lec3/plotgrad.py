#!/usr/bin/env python
import numpy as np
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

size = 256
ls = np.linspace(-2,+2,size,False)
w1 = ls
w2 = ls.reshape(size,1)[::-1]
w3 = -2.0
b = 1.0
# compute y.
ya = sigmoid(w1*0 + w2*0 + w3*0 + b)
yb = sigmoid(w1*0 + w2*1 + w3*0 + b)
yc = sigmoid(w1*1 + w2*0 + w3*1 + b)
# compute loss.
L = (ya-1)**2 + (yb-1)**2 + (yc-0)**2
# scale to 0.0 - 1.0.
(l0, l1) = (L.min(), L.max())
L = (L-l0)/(l1-l0)
# convert to RGB.
a = np.array([L, np.zeros_like(L), 1-L])
# save image.
a = a.transpose(1,2,0)*255
img = Image.fromarray(a.astype(np.uint8), 'RGB')
img.save('grad1.png')
