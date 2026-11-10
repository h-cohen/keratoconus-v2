"""E10 step 2: hand-crafted colour-grid features (no neural encoder). For each map, the 256x256 image
is split into a 4x4 grid; each cell gives mean hue as sin/cos (hue is circular), mean saturation and
mean value (HSV). Output rethink/data/colorgrid.npy, shape (N, 4 maps, 16 cells, 4). Unsupervised.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
IMG = Path(os.environ.get('KC_SRC', '/home/hadar/Cloud/Work/projects/keratoconus-v2'), 'data/cropped_output_v2')
MAPS = ['corneal_thickness', 'curvature_front', 'elevation_front', 'elevation_back']


def cell_feats(path):
    hsv = np.asarray(Image.open(path).convert('HSV').resize((256, 256)), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0] * 2 * np.pi, hsv[..., 1], hsv[..., 2]
    out = []
    for i in range(4):
        for j in range(4):
            sl = (slice(i * 64, (i + 1) * 64), slice(j * 64, (j + 1) * 64))
            out.append([np.cos(h[sl]).mean(), np.sin(h[sl]).mean(), s[sl].mean(), v[sl].mean()])
    return out


def main():
    keys = pd.read_csv(DATA / 'private_keys.csv')
    arr = np.zeros((len(keys), 4, 16, 4), dtype=np.float32)
    for i, f in enumerate(keys.filename):
        for k, m in enumerate(MAPS):
            arr[i, k] = cell_feats(IMG / f"{f[:-4]}_{m}.jpg")
    np.save(DATA / 'colorgrid.npy', arr)
    print('saved', arr.shape)


if __name__ == '__main__':
    sys.exit(main())
