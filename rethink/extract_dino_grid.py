"""E10 step 1: spatial DINOv2-small features. Full map resized to 224 (no crop), 16x16 patch grid
pooled to 4x4 cells. Output rethink/data/dinogrid_<map>.npy, shape (N, 16 cells, 384), rows aligned
with cohort.parquet. Unsupervised: no labels read.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
IMG = Path('/home/hadar/Cloud/Work/projects/keratoconus-v2/data/cropped_output_v2')
MAPS = ['corneal_thickness', 'curvature_front', 'elevation_front', 'elevation_back']
MODEL = 'facebook/dinov2-small'


def load_encoder():
    proc = AutoImageProcessor.from_pretrained(MODEL, do_center_crop=False, size={'height': 224, 'width': 224})
    return proc, AutoModel.from_pretrained(MODEL).eval()


def grid_features(paths, proc, model, batch=32, log=None):
    """(len(paths), 16 cells, 384) DINOv2 features: full map at 224, 16x16 patches pooled to 4x4."""
    feats = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert('RGB') for p in paths[i:i + batch]]
        with torch.no_grad():
            h = model(**proc(images=ims, return_tensors='pt')).last_hidden_state[:, 1:]  # drop CLS
        h = h.reshape(len(ims), 16, 16, -1).permute(0, 3, 1, 2)
        feats.append(torch.nn.functional.adaptive_avg_pool2d(h, 4).flatten(2).permute(0, 2, 1).numpy())
        if log:
            log(i)
    return np.concatenate(feats).astype(np.float32)


def main():
    torch.set_num_threads(8)
    keys = pd.read_csv(DATA / 'private_keys.csv')
    proc, model = load_encoder()
    for m in MAPS:
        out = DATA / f'dinogrid_{m}.npy'
        if out.exists():
            continue
        paths = [IMG / f"{f[:-4]}_{m}.jpg" for f in keys.filename]
        np.save(out, grid_features(paths, proc, model, log=lambda i: print(m, i, flush=True)))
    print('done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
