#!/usr/bin/env python
import numpy as np

# load_mnist: MNISTデータベースを読み込む。
#
# train_images = load_mnist('train-images-idx3-ubyte.gz')
# train_labels = load_mnist('train-labels-idx1-ubyte.gz')
# test_images = load_mnist('t10k-images-idx3-ubyte.gz')
# test_labels = load_mnist('t10k-labels-idx1-ubyte.gz')
#
def load_mnist(path):
    import gzip
    import struct
    # ファイルを開く (gzip圧縮可)。
    if path.endswith('.gz'):
        fp = gzip.open(path)
    else:
        fp = open(path, 'rb')
    # 先頭4バイトを検査する。
    magic = fp.read(4)
    if magic == b'\x00\x00\x08\x01':
        # 正解 (ラベル) データの場合: n個の uint8が続く。
        (n,) = struct.unpack('>L', fp.read(4))
        size = n
        data = np.frombuffer(fp.read(size), dtype=np.uint8)
        assert data.size == size
    elif magic == b'\x00\x00\x08\x03':
        # 画像データの場合: n*rows*cols個の uint8が続く。
        (n,rows,cols) = struct.unpack('>LLL', fp.read(12))
        size = n * rows * cols
        data = np.frombuffer(fp.read(size), dtype=np.uint8)
        assert data.size == size
        data = data.reshape(n, rows, cols)
    else:
        # 不明なデータ。
        raise ValueError(f'Invalid Magic number: {magic}')
    # ファイルを閉じる。
    fp.close()
    return data
