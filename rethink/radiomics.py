"""E25 block R: hand-crafted measurements of the decoded physical maps (decoded_hr.npy, left eyes mirrored so +x is
nasal for both eyes; +y superior). Label-free, per composite; written to rethink/data/emb_R.npy (2260 x n).
Geometry as zernike.py: 145 x 145 cells, 2 px per cell, 28.9 px per mm, map centre = corneal vertex.
Per map (thickness T, sagittal front curvature K, front / back elevation EF / EB), central 8 mm:
  ring x quadrant means: rings 0-1, 1-2, 2-3, 3-4 mm radius, quadrants superior / inferior / nasal / temporal (64);
  extremum (thinnest point for T, maximum for K / EF / EB): value, x, y, distance from centre (16);
  inferior - superior and nasal - temporal mean difference on the 2-4 mm annulus (8);
  map standard deviation within 3 mm (4).
Thickness progression around the thinnest point: mean thickness on 1, 2, 3 mm rings around it minus the thinnest
value, and the 3 mm value / 1 mm value ratio (4). Total 96. NaN where the map or > 30% of a region is missing.
Usage: python -I rethink/radiomics.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from harness import DATA  # noqa: E402
from zernike import CELL, PX_MM  # noqa: E402

NMAP = ['T', 'K', 'EF', 'EB']
RINGS = [(0, 1), (1, 2), (2, 3), (3, 4)]


def grid(n=145):
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    return (xx - c) * CELL / PX_MM, -(yy - c) * CELL / PX_MM


X, Y = grid()
R = np.hypot(X, Y)
TH = np.degrees(np.arctan2(Y, X))
QUAD = {'S': (TH > 45) & (TH <= 135), 'I': (TH < -45) & (TH >= -135), 'N': np.abs(TH) <= 45,
        'T': np.abs(TH) > 135}


def mean_ok(v, m):
    x = v[m]
    return np.nanmean(x) if np.isfinite(x).mean() >= 0.7 else np.nan


def names():
    out = [f'{mp}_r{a}{b}_{q}' for mp in NMAP for a, b in RINGS for q in QUAD]
    out += [f'{mp}_ext_{w}' for mp in NMAP for w in ('val', 'x', 'y', 'r')]
    out += [f'{mp}_{w}' for mp in NMAP for w in ('IminusS', 'NminusT')]
    out += [f'{mp}_sd3' for mp in NMAP]
    out += ['T_prog1', 'T_prog2', 'T_prog3', 'T_ratio31']
    return out


def features(v):
    f = []
    for j in range(4):
        m = v[j].astype(np.float64)
        for a, b in RINGS:
            ring = (R >= a) & (R < b)
            f += [mean_ok(m, ring & q) for q in QUAD.values()]
    for j in range(4):
        m = np.where(R <= 4, v[j].astype(np.float64), np.nan)
        if np.isfinite(m[R <= 4]).mean() < 0.7:
            f += [np.nan] * 4
            continue
        k = np.nanargmin(m) if j == 0 else np.nanargmax(m)
        f += [m.flat[k], X.flat[k], Y.flat[k], R.flat[k]]
    ann = (R >= 2) & (R < 4)
    for j in range(4):
        m = v[j].astype(np.float64)
        f += [mean_ok(m, ann & QUAD['I']) - mean_ok(m, ann & QUAD['S']),
              mean_ok(m, ann & QUAD['N']) - mean_ok(m, ann & QUAD['T'])]
    for j in range(4):
        m = v[j].astype(np.float64)[R <= 3]
        f.append(np.nanstd(m) if np.isfinite(m).mean() >= 0.7 else np.nan)
    t = np.where(R <= 4, v[0].astype(np.float64), np.nan)
    if np.isfinite(t[R <= 4]).mean() >= 0.7:
        k = np.nanargmin(t)
        d = np.hypot(X - X.flat[k], Y - Y.flat[k])
        rings = [mean_ok(v[0].astype(np.float64), (d >= r - 0.25) & (d < r + 0.25)) for r in (1, 2, 3)]
        f += [x - t.flat[k] for x in rings] + [(rings[2] - t.flat[k]) / (rings[0] - t.flat[k] + 1e-6)]
    else:
        f += [np.nan] * 4
    return f


def main():
    hr = np.load(DATA / 'decoded_hr.npy', mmap_mode='r')
    with np.errstate(all='ignore'):
        out = np.array([features(np.asarray(hr[i])) for i in range(len(hr))], np.float32)
    assert out.shape[1] == len(names())
    np.save(DATA / 'emb_R.npy', out)
    print(out.shape, 'NaN fraction', round(float(np.isnan(out).mean()), 3), 'rows all-NaN',
          int(np.isnan(out).all(1).sum()))


if __name__ == '__main__':
    sys.exit(main())
