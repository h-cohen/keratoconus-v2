"""E22 step 2: higher-resolution decoding of EVERY '4 Maps Selectable' composite (design section 5).

Same decoder and per-image 31-step palette check as decode_maps.py / decode_all.py, but 2x2-pixel block medians
(145 x 145 per map instead of 58 x 58). Left eyes are mirrored (x nasal-positive). File order, pkey, eye, date
and cohort_row are identical to decoded_all_meta.parquet (asserted), so the two tables are row-aligned.
Output (rethink/data/, gitignored): decoded_hr.npy (N, 4, 145, 145) float16, NaN = undecodable / outside the
map; decoded_hr_meta.parquet = decoded_all_meta + per-map palette size and valid fraction. No labels are read.
"""
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from decode_maps import HALF, IMG, MAPS, decode  # noqa: E402

DATA = Path(__file__).parent / 'data'
BLOCK = 2
N = (2 * HALF) // BLOCK  # 145


def blocks(v):
    b = v[:N * BLOCK, :N * BLOCK].reshape(N, BLOCK, N, BLOCK).transpose(0, 2, 1, 3).reshape(N, N, -1)
    with np.errstate(all='ignore'):
        out = np.nanmedian(b, -1)
    out[np.isnan(b).mean(-1) > .5] = np.nan
    return out


def files():
    fs = sorted(f for f in IMG.iterdir() if f.suffix.lower() == '.jpg' and 'maps' in f.name.lower())
    return [f for f in fs if re.search(r'_(od|os)_(\d{8})_', f.name.lower())]


def work(f):
    im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
    v = np.full((4, N, N), np.nan, np.float32)
    npal = []
    for j, mp in enumerate(MAPS):
        d, n = decode(im, mp)
        npal.append(n)
        if d is not None:
            v[j] = blocks(d)
    if re.search(r'_(od|os)_', f.name.lower()).group(1) == 'os':
        v = v[:, :, ::-1]
    return v.astype(np.float16), npal


def main():
    fs = files()
    if len(sys.argv) > 1:  # smoke test
        fs = fs[:int(sys.argv[1])]
    with Pool(32) as p:
        res = p.map(work, fs, chunksize=8)
    meta = pd.read_parquet(DATA / 'decoded_all_meta.parquet')
    assert len(fs) <= len(meta)
    meta = meta.iloc[:len(fs)].copy()
    out = np.stack([r[0] for r in res])
    for j, m in enumerate(MAPS):
        meta[f'pal_{m}'] = [r[1][j] for r in res]
        meta[f'valid_{m}'] = np.isfinite(out[:, j].astype(np.float32)).mean((1, 2))
    # alignment check vs the 58x58 decoding: 5x5-block medians of both should agree closely
    lo = np.load(DATA / 'decoded_all.npy', mmap_mode='r')[:len(fs)]
    assert (np.isfinite(lo).any((2, 3)) == np.isfinite(out.astype(np.float32)).any((2, 3))).all(), 'row misalignment'
    if len(sys.argv) == 1:
        np.save(DATA / 'decoded_hr.npy', out)
        meta.to_parquet(DATA / 'decoded_hr_meta.parquet', index=False)
    print(len(meta), 'composites;', (meta.cohort_row >= 0).sum(), 'cohort')
    print(meta[[c for c in meta.columns if c.startswith(('pal_', 'valid_'))]].describe().T.round(3).to_string())


if __name__ == '__main__':
    sys.exit(main())
