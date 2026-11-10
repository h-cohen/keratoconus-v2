"""E04 step 1: frozen DINOv2-small features for the four Pentacam maps of every cohort scan.

Unsupervised: no labels are read. Output rethink/data/dino_<map>.npy, rows aligned with cohort.parquet.
Per map: CLS token + mean of patch tokens (2 x 384). Image preprocessing is the model's default.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
IMG = Path(os.environ.get('KC_SRC', '/home/hadar/Cloud/Work/projects/keratoconus-v2'), 'data/cropped_output_v2')
MAPS = ['corneal_thickness', 'curvature_front', 'elevation_front', 'elevation_back']
MODEL = 'facebook/dinov2-small'


def main():
    torch.set_num_threads(8)
    keys = pd.read_csv(DATA / 'private_keys.csv')
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    for m in MAPS:
        out = DATA / f'dino_{m}.npy'
        if out.exists():
            continue
        feats = []
        for i in range(0, len(keys), 32):
            ims = [Image.open(IMG / f"{f[:-4]}_{m}.jpg").convert('RGB') for f in keys.filename.iloc[i:i + 32]]
            with torch.no_grad():
                h = model(**proc(images=ims, return_tensors='pt')).last_hidden_state
            feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy())
            print(m, i, flush=True)
        np.save(out, np.concatenate(feats).astype(np.float32))
    print('done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
