"""E17 step 2: decode the four Pentacam maps of each composite into physical values (single visit).

Each composite carries its own colour bars (31 steps each, separated by black lines). For every image the
bars are sampled and every map pixel is assigned the value of the nearest palette colour; pixels far from
all palette colours (text, symbols, axis marks) become NaN.
  thickness  : 300 + 20k um       (k = 0..30, top to bottom)
  curvature  : sagittal front, D;  labelled steps 60, 50, 48, ... 17.5 with midpoints between them
  elevation  : front / back vs BFS, -75 + 5k um
Map boxes are those of preprocessing/3_improved_crop_script.py (290 px, centres below). Output, in
cohort order: rethink/data/decoded_maps.npy (N, 4, 58, 58) float32, 5x5-pixel block medians, and
rethink/data/decoded_meta.parquet (palette check, thinnest-band centroid in px for scale calibration).
No labels are read.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
IMG = Path('/home/hadar/Cloud/Work/projects/keratoconus-v2/data/image_scans')
MAPS = ['corneal_thickness', 'curvature_front', 'elevation_front', 'elevation_back']
CENTRE = {'corneal_thickness': (593, 243), 'curvature_front': (967, 243),
          'elevation_front': (593, 641), 'elevation_back': (967, 641)}
BAR = {'corneal_thickness': (385, 45, 400), 'curvature_front': (1180, 45, 400),
       'elevation_front': (385, 445, 800), 'elevation_back': (1180, 445, 800)}
_K = [60, 50, 48, 46, 44.5, 43.5, 42.5, 41.5, 40.5, 39.5, 38.5, 37.5, 32.5, 27.5, 22.5, 17.5]
VALUES = {
    'corneal_thickness': 300 + 20 * np.arange(31.),
    'curvature_front': np.array([v for a, b in zip(_K, _K[1:] + [None]) for v in ([a, (a + b) / 2] if b else [a])]),
    'elevation_front': -75 + 5 * np.arange(31.),
    'elevation_back': -75 + 5 * np.arange(31.),
}
HALF, BLOCK, MAXDIST = 145, 5, 45.0


def palette(im, x, y0, y1):
    """The 31 colours of a colour bar, top to bottom (black separators and grey background removed)."""
    col = im[y0:y1, x - 2:x + 3].mean(1)
    dark = col.sum(1) < 150
    segs, s = [], None
    for i, d in enumerate(list(dark) + [True]):
        if not d and s is None:
            s = i
        if d and s is not None:
            if i - s >= 4:
                c = np.median(col[s + 1:i - 1], 0)
                if not (np.ptp(c) < 12 and c.mean() > 200):  # grey background
                    segs.append(c)
            s = None
    return np.array(segs)


def decode(im, m):
    pal = palette(im, *BAR[m])
    if len(pal) != 31:
        return None, len(pal)
    cx, cy = CENTRE[m]
    box = im[cy - HALF:cy + HALF, cx - HALF:cx + HALF]
    d = ((box[:, :, None, :] - pal[None, None]) ** 2).sum(-1)
    k = d.argmin(-1)
    v = VALUES[m][k].astype(np.float32)
    v[np.sqrt(d.min(-1)) > MAXDIST] = np.nan
    yy, xx = np.mgrid[-HALF:HALF, -HALF:HALF]
    v[np.hypot(xx, yy) > HALF - 4] = np.nan
    return v, 31


def blocks(v):
    n = (2 * HALF) // BLOCK
    b = v[:n * BLOCK, :n * BLOCK].reshape(n, BLOCK, n, BLOCK).transpose(0, 2, 1, 3).reshape(n, n, -1)
    with np.errstate(all='ignore'):
        out = np.nanmedian(b, -1)
    out[np.isnan(b).mean(-1) > .5] = np.nan
    return out


def main():
    keys = pd.read_csv(DATA / 'private_keys.csv', dtype=str)
    on_disk = {f.name.lower(): f for f in IMG.glob('*.JPG')}
    n = (2 * HALF) // BLOCK
    out = np.full((len(keys), 4, n, n), np.nan, np.float32)
    meta = []
    for i, f in enumerate(keys.filename):
        im = np.asarray(Image.open(on_disk[f.lower()]).convert('RGB')).astype(np.float32)
        row = {}
        for j, m in enumerate(MAPS):
            v, npal = decode(im, m)
            row[f'pal_{m}'] = npal
            if v is None:
                continue
            out[i, j] = blocks(v)
            if m == 'corneal_thickness':  # centroid of the thinnest band, px from the map centre
                ys, xs = np.nonzero(v == np.nanmin(v))
                row.update(thin_px_x=xs.mean() - HALF, thin_px_y=ys.mean() - HALF, thin_band=float(np.nanmin(v)))
            if m == 'curvature_front':
                ys, xs = np.nonzero(v == np.nanmax(v))
                row.update(kmax_px_x=xs.mean() - HALF, kmax_px_y=ys.mean() - HALF, kmax_band=float(np.nanmax(v)))
        meta.append(row)
        if i % 100 == 0:
            print(i, flush=True)
    np.save(DATA / 'decoded_maps.npy', out)
    meta = pd.DataFrame(meta)
    meta.to_parquet(DATA / 'decoded_meta.parquet', index=False)
    print(meta.describe().T.round(2).to_string())


if __name__ == '__main__':
    sys.exit(main())
