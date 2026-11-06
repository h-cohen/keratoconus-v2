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


def main():
    torch.set_num_threads(8)
    keys = pd.read_csv(DATA / 'private_keys.csv')
    proc = AutoImageProcessor.from_pretrained(MODEL, do_center_crop=False, size={'height': 224, 'width': 224})
    model = AutoModel.from_pretrained(MODEL).eval()
    for m in MAPS:
        out = DATA / f'dinogrid_{m}.npy'
        if out.exists():
            continue
        feats = []
        for i in range(0, len(keys), 32):
            ims = [Image.open(IMG / f"{f[:-4]}_{m}.jpg").convert('RGB') for f in keys.filename.iloc[i:i + 32]]
            with torch.no_grad():
                h = model(**proc(images=ims, return_tensors='pt')).last_hidden_state[:, 1:]  # drop CLS
            h = h.reshape(len(ims), 16, 16, -1).permute(0, 3, 1, 2)
            pooled = torch.nn.functional.adaptive_avg_pool2d(h, 4).flatten(2).permute(0, 2, 1)  # (B,16,384)
            feats.append(pooled.numpy())
            print(m, i, flush=True)
        np.save(out, np.concatenate(feats).astype(np.float32))
    print('done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
