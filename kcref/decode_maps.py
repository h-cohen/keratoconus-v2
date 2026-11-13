"""Map decoder (library for decode_all.py / decode_hr.py): the four Pentacam maps of a composite to physical values.

Each composite carries its own colour bars (31 steps each, separated by black lines). For every image the
bars are sampled and every map pixel is assigned the value of the nearest palette colour; pixels far from
all palette colours (text, symbols, axis marks) become NaN.
  thickness  : 300 + 20k um       (k = 0..30, top to bottom)
  curvature  : sagittal front, D;  labelled steps 60, 50, 48, ... 17.5 with midpoints between them
  elevation  : front / back vs BFS, -75 + 5k um
Map boxes are those of preprocessing/3_improved_crop_script.py (290 px, centres below). Map scale: 28.9 px per mm
(calibrated on the thickness map's ring spacing). No labels are read.
"""
import numpy as np

from common import SRC

IMG = SRC / 'image_scans'
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
