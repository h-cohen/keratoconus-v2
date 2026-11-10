"""E17 step 1: read the printed values of the Pentacam '4 Maps Selectable' composite (single visit).

The composite (data/image_scans/<filename>, 1200x910, one per cohort scan) prints, at fixed positions and
in one bitmap font, values the numeric export lacks: KMax location (x, y), cornea volume, KPD, anterior
chamber volume / angle / depth, pupil diameter, IOP correction. No OCR engine is installed, so glyphs are
segmented by column gaps and classified by nearest template. Templates are learned from fields whose
values ARE in the export (pachymetry at pupil / apex / thinnest point, thinnest x/y, KMax, pupil x/y),
on half of the images; accuracy is measured on the other half before the new fields are read.
Output: rethink/data/panel_ocr.parquet (cohort order); never reads labels.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
IMG = Path(os.environ.get('KC_SRC', '/home/hadar/Cloud/Work/projects/keratoconus-v2'), 'data/image_scans')
# name: (x0, x1, y_centre) of the value box
FIELDS = {
    'pachy_pupil': (136, 202, 609), 'pupil_x': (229, 264, 609), 'pupil_y': (283, 318, 609),
    'pachy_apex': (136, 202, 636),
    'pachy_thin': (136, 202, 663), 'thin_x': (229, 264, 663), 'thin_y': (283, 318, 663),
    'kmax': (136, 202, 690), 'kmax_x': (229, 264, 690), 'kmax_y': (283, 318, 690),
    'cornea_vol': (136, 202, 723), 'kpd': (267, 318, 723),
    'chamber_vol': (136, 202, 753), 'angle': (267, 318, 753),
    'acd': (136, 202, 783), 'pupil_dia': (267, 318, 783),
    'iop_corr': (136, 202, 813),
}
# known fields: (cohort/raw column, decimals, signed-with-plus)
KNOWN = {'pachy_pupil': ('Pachy Pupil:', 0, False), 'pachy_apex': ('Pachy Apex:', 0, False),
         'pachy_thin': ('Pachy Min:', 0, False), 'thin_x': ('Pachy Min Pos X:', 2, True),
         'thin_y': ('Pachy Min Pos Y:', 2, True), 'kmax': ('KMax Sagittal Front (D)', 1, False),
         'pupil_x': ('Pupil Pos X:', 2, True), 'pupil_y': ('Pupil Pos Y:', 2, True)}
W = 8  # glyph patch width


def glyphs(im, box):
    """Grey patches (11 x W) of the characters of the number in a value box (stops at the unit)."""
    x0, x1, yc = box
    g = im[yc - 6:yc + 5, x0 + 2:x1 - 1]
    cols = (g < 140).any(0)
    runs, s = [], None
    for i, c in enumerate(list(cols) + [False]):
        if c and s is None:
            s = i
        if not c and s is not None:
            runs.append((s, i))
            s = None
    out = []
    for k, (a, b) in enumerate(runs):
        if k and a - runs[k - 1][1] >= 4:  # a wide gap (digit gaps are <= 3 px) separates the number from its unit
            break
        p = np.full((11, W), 255.0)
        p[:, :min(b - a, W)] = g[:, a:a + min(b - a, W)]
        out.append(1 - p / 255)
    return out


def fmt(v, dec, signed):
    s = f'{v:.{dec}f}'
    return '+' + s if signed and v > 0 and s.strip('0.') else s


def read(templates, patches):
    keys = list(templates)
    T = np.stack([templates[k] for k in keys]).reshape(len(keys), -1)
    return ''.join(keys[int(np.argmin(((T - p.reshape(1, -1)) ** 2).sum(1)))] for p in patches)


def to_float(s):
    try:
        return float(s)
    except ValueError:
        return np.nan


def main():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    raw = pd.read_parquet(DATA / 'raw_extra.parquet')
    keys = pd.read_csv(DATA / 'private_keys.csv', dtype=str)
    known = pd.concat([coh, raw[['Pupil Pos X:', 'Pupil Pos Y:']]], axis=1)
    on_disk = {f.name.lower(): f for f in IMG.glob('*.JPG')}
    P = []
    for f in keys.filename:
        im = np.asarray(Image.open(on_disk[f.lower()]).convert('L')).astype(float)
        P.append({k: glyphs(im, b) for k, b in FIELDS.items()})

    half = np.random.default_rng(0).permutation(len(P)) < len(P) // 2
    lab = {}
    # x fields are excluded from template learning: the screen's x sign convention differs from the export's
    train_fields = {k: v for k, v in KNOWN.items() if not k.endswith('_x')}
    for i in np.flatnonzero(half):
        for k, (col, dec, sg) in train_fields.items():
            v = known[col].iloc[i]
            if pd.isna(v):
                continue
            s = fmt(v, dec, sg)
            if len(s) == len(P[i][k]):
                for ch, p in zip(s, P[i][k]):
                    lab.setdefault(ch, []).append(p)
    templates = {ch: np.mean(v, 0) for ch, v in lab.items()}
    print('templates:', {ch: len(v) for ch, v in sorted(lab.items())})

    for k, (col, dec, sg) in KNOWN.items():
        idx = [i for i in np.flatnonzero(~half) if pd.notna(known[col].iloc[i])]
        got = np.array([to_float(read(templates, P[i][k])) for i in idx])
        exp = known[col].values[idx]
        if k.endswith('_x'):
            eye = coh.eye.values[idx]
            same = np.isclose(got, exp, atol=.005)
            flip = np.isclose(got, -exp, atol=.005)
            print(f'held-out |x| match {k:12s} {np.mean(same | flip):.3f}  same sign: right {same[eye == "right"].mean():.2f}'
                  f' left {same[eye == "left"].mean():.2f}  (n={len(idx)})')
        else:
            print(f'held-out exact match {k:12s} {np.mean(np.isclose(got, exp, atol=.005)):.3f} (n={len(idx)})')

    out = pd.DataFrame({k: [to_float(read(templates, p[k])) for p in P] for k in FIELDS})
    out.to_parquet(DATA / 'panel_ocr.parquet', index=False)
    print(out.describe().T[['count', 'mean', 'min', 'max']].round(2).to_string())


if __name__ == '__main__':
    sys.exit(main())
