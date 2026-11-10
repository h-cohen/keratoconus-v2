"""E22 step 7: encoder D - frozen DINOv2-small on EVERY composite (design 5, baseline encoder; no labels).

Same model and input as extract_dino_grid.py (the four cropped map images, full map resized to 224, no centre
crop), but every composite is embedded, so fellow composites outside the cohort get features too, and LEFT eyes
are flipped horizontally first (nasal-positive, as decoded_hr), so own and fellow embeddings are comparable.
Embedding per composite = mean-pooled patch tokens of each map, concatenated: 4 x 384 = 1,536 dims (reduced by
in-fold PCA downstream, like every encoder block).
Output rethink/data/emb_D.npy (2,260 x 1,536), aligned with decoded_hr_meta.parquet.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).parent))
from decode_hr import files  # noqa: E402

DATA = Path(__file__).parent / 'data'
CROP = Path(os.environ.get('KC_SRC', Path(__file__).resolve().parents[1]), 'data/cropped_output_v2')
MAPS = ['corneal_thickness', 'curvature_front', 'elevation_front', 'elevation_back']
MODEL = 'facebook/dinov2-small'


def main():
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda:4')
    proc = AutoImageProcessor.from_pretrained(MODEL, do_center_crop=False, size={'height': 224, 'width': 224})
    model = AutoModel.from_pretrained(MODEL).eval().to(dev)
    meta = pd.read_parquet(DATA / 'decoded_hr_meta.parquet')
    on_disk = {p.name.lower(): p for p in CROP.iterdir()}
    stems = [f.name.rsplit('.', 1)[0] for f in files()]
    out = np.full((len(stems), 4 * 384), np.nan, np.float32)
    missing = 0
    for j, m in enumerate(MAPS):
        for i in range(0, len(stems), 64):
            ims, idx = [], []
            for k in range(i, min(i + 64, len(stems))):
                p = on_disk.get(f'{stems[k]}_{m}.jpg'.lower())
                if p is None:
                    missing += 1
                    continue
                im = Image.open(p).convert('RGB')
                ims.append(ImageOps.mirror(im) if meta.eye.iat[k] == 'left' else im)
                idx.append(k)
            if not ims:
                continue
            with torch.no_grad():
                h = model(**proc(images=ims, return_tensors='pt').to(dev)).last_hidden_state[:, 1:]
            out[idx, j * 384:(j + 1) * 384] = h.mean(1).float().cpu().numpy()
        print(m, 'done', flush=True)
    np.save(DATA / 'emb_D.npy', out)
    print(out.shape, 'missing crops', missing, 'rows complete', int(np.isfinite(out).all(1).sum()))


if __name__ == '__main__':
    sys.exit(main())
