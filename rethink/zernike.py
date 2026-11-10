"""E22 step 4: encoder Z - Zernike coefficients (analytic, no training, no labels; design section 5).

For every decoded composite (decoded_hr.npy, left eyes mirrored so x is nasal-positive) the front elevation, back
elevation and corneal thickness maps are fitted by least squares with OSA/ANSI Zernike polynomials up to radial
order 8 (45 terms) on the 6 mm zone centred on the map centre. Scale 28.9 px/mm (exp17_rawmaps calibration);
decoded_hr cells are 2 px, so the 3 mm radius is 43.35 cells. A fit needs >= 70% of the zone decoded, else NaN.
Output rethink/data/emb_Z.npy (N, 135) float32, row-aligned with decoded_hr_meta.parquet.
"""
import sys
from math import factorial
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / 'data'
ORDER, R_MM, PX_MM, CELL = 8, 3.0, 28.9, 2
MAPS = {'elevation_front': 2, 'elevation_back': 3, 'corneal_thickness': 0}


def osa():
    return [(n, m) for n in range(ORDER + 1) for m in range(-n, n + 1, 2)]


def zernike(n, m, rho, th):
    r = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        r += ((-1) ** k * factorial(n - k) / (factorial(k) * factorial((n + abs(m)) // 2 - k)
                                               * factorial((n - abs(m)) // 2 - k))) * rho ** (n - 2 * k)
    norm = np.sqrt(2 * (n + 1)) if m else np.sqrt(n + 1)
    return norm * r * (np.cos(m * th) if m >= 0 else np.sin(-m * th))


def basis(size):
    c = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size]
    x, y = (xx - c) * CELL / PX_MM, -(yy - c) * CELL / PX_MM  # mm, y up
    rho, th = np.hypot(x, y) / R_MM, np.arctan2(y, x)
    zone = rho <= 1
    B = np.stack([zernike(n, m, rho[zone], th[zone]) for n, m in osa()], 1)
    return zone, B


def main():
    hr = np.load(DATA / 'decoded_hr.npy', mmap_mode='r')
    zone, B = basis(hr.shape[-1])
    out = np.full((len(hr), len(MAPS) * B.shape[1]), np.nan, np.float32)
    for i in range(len(hr)):
        for j, ch in enumerate(MAPS.values()):
            v = hr[i, ch].astype(np.float64)[zone]
            ok = np.isfinite(v)
            if ok.mean() >= 0.7:
                out[i, j * B.shape[1]:(j + 1) * B.shape[1]] = np.linalg.lstsq(B[ok], v[ok], rcond=None)[0]
    np.save(DATA / 'emb_Z.npy', out)
    fit = np.isfinite(out).all(1)
    print(out.shape, 'complete fits', fit.sum(), 'of', len(out), '| zone cells', zone.sum())
    # sanity: residual RMS of the fit on a few maps
    for j, (m, ch) in enumerate(MAPS.items()):
        res = []
        for i in range(0, len(hr), 97):
            v = hr[i, ch].astype(np.float64)[zone]
            ok = np.isfinite(v)
            if ok.mean() >= 0.7:
                c = out[i, j * B.shape[1]:(j + 1) * B.shape[1]]
                res.append(np.sqrt(np.mean((B[ok] @ c - v[ok]) ** 2)))
        print(f'{m:20s} residual RMS median {np.median(res):.2f} (quantisation step 5 um / 20 um)')


if __name__ == '__main__':
    sys.exit(main())
