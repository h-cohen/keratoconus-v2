"""E22 step 3: decode the Pentacam report pages inside the '4 Maps Select.DCM' files (PRETRAINING ONLY, design C4).

Each DCM is an encapsulated 5-page PDF; every page holds one 1200 x 910 screenshot. All 1,590 DCM exams also
have a JPG composite (same file stem), so page 0 ('4 Maps Selectable') duplicates the composite and is skipped.
Page types and the maps decoded from them (same nearest-palette decoder as decode_maps.py):
  1 '4 Maps Refractive'   curvature (sagittal front, D), elevation front / back (BFS float 8 mm, -37.5 + 2.5k um),
                          thickness (um)                      scale 1.00 of the composite
  2 'Cataract Pre-OP'     curvature front (scale 0.865), thickness (scale 0.64); total corneal power not used
  3 'Holladay EKR'        EKR65 power on a RELATIVE 15-step scale -> cannot be decoded to physical units; its
                          palette check is reported and the page type is dropped
  4 'Belin/Ambrosio'      elevation front / back (BFS float 8 mm, -75 + 5k um, scale 0.66), thickness (0.71);
                          the 'enhanced' (exclusion-zone) and difference maps are not used
Geometry: centres and scales come from the printed axis labels (63 px per 4 mm on the composite, fixed across
files). Every map is resampled to the composite grid (145 x 145, 2-px blocks, left eyes mirrored), so channel j
means the same physical quantity at the same position as in decoded_hr.npy.
Palette check per map: exactly 31 steps AND every step within 30 (RGB L2) of the median palette of that page/map.
A page type is dropped if the palette check fails on > 5% of its files (pages 3 and 4: Holladay has a relative
15-step scale, Belin/Ambrosio elevation a continuous colour bar). Twin check per map: the same exam's JPG composite
shows the same quantity, so a decoded map is kept only if its central 4 mm median |difference| to the composite
is <= one palette step. Page 1 elevation uses the same colours for +-37.5 um (2.5 um steps) and +-75 um (5 um)
scales; both are tried and the one passing the twin check is used (no label is involved). Output (gitignored):
  rethink/data/decoded_dcm.npy (M, 4, 145, 145) float16 (NaN where absent / undecodable)
  rethink/data/decoded_dcm_meta.parquet: hr_row (row of the JPG composite of the same exam in decoded_hr.npy),
  pkey, eye, date, page. No labels, names or filenames are written.
"""
import io
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pymupdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from decode_hr import N, files  # noqa: E402
from decode_maps import IMG, MAXDIST, VALUES, palette  # noqa: E402

DATA = Path(__file__).parent / 'data'
REF = 63.0  # px per 4 mm on the composite
CH = {'thick': 0, 'curv': 1, 'ef': 2, 'eb': 3}
VAL = {'thick': VALUES['corneal_thickness'], 'curv': VALUES['curvature_front'], 'ef': VALUES['elevation_front'],
       'eb': VALUES['elevation_back'], 'ef25': -37.5 + 2.5 * np.arange(31.), 'eb25': -37.5 + 2.5 * np.arange(31.)}
# page -> map -> (centre x, centre y, px per 4 mm, bar (x, y0, y1), value table)
LAYOUT = {
    1: {'curv': (592, 275, 63.0, (385, 85, 400), 'curv'), 'ef': (967, 275, 63.0, (1180, 85, 400), 'ef25'),
        'thick': (592, 652, 63.0, (385, 462, 777), 'thick'), 'eb': (967, 652, 63.0, (1180, 462, 777), 'eb25')},
    2: {'curv': (247, 294, 54.5, (62, 113, 418), 'curv'), 'thick': (1042, 294, 40.3, (862, 113, 418), 'thick')},
    4: {'ef': (213, 204, 41.5, (50, 47, 733), 'ef'), 'eb': (489, 204, 41.5, (50, 47, 733), 'eb'),
        'thick': (1015, 200, 44.5, (1186, 45, 292), 'thick')},
}
HOLLADAY_BAR = (1170, 45, 705)


def pages(f):
    doc = pymupdf.open(stream=bytes(pydicom.dcmread(f).EncapsulatedDocument), filetype='pdf')
    out = []
    for p in doc:
        x = p.get_images()[0][0]
        out.append(np.asarray(Image.open(io.BytesIO(doc.extract_image(x)['image'])).convert('RGB')).astype(np.float32))
    return out


def resample(im, cx, cy, s, pal, vals):
    """Decode the map onto the composite grid: canonical offsets (2-px steps, +-145 px) scaled to page pixels."""
    off = (np.arange(N) * 2 + 1 - 145) * (s / REF)
    xs, ys = np.rint(cx + off).astype(int), np.rint(cy + off).astype(int)
    ok_x, ok_y = (xs >= 0) & (xs < im.shape[1]), (ys >= 0) & (ys < im.shape[0])
    sub = im[np.clip(ys, 0, im.shape[0] - 1)][:, np.clip(xs, 0, im.shape[1] - 1)]
    d = ((sub[:, :, None, :] - pal[None, None]) ** 2).sum(-1)
    v = vals[d.argmin(-1)].astype(np.float32)
    v[np.sqrt(d.min(-1)) > MAXDIST] = np.nan
    v[~ok_y] = np.nan
    v[:, ~ok_x] = np.nan
    yy, xx = np.mgrid[:N, :N]
    v[np.hypot(xx - (N - 1) / 2, yy - (N - 1) / 2) > (N - 1) / 2 - 2] = np.nan
    return v


def pal_only(f):
    try:
        pg = pages(f)
    except Exception:  # unreadable file
        return None
    pals = {(k, m): palette(pg[k], *lay[3]) for k, mm in LAYOUT.items() for m, lay in mm.items()}
    pals[(3, 'ekr')] = palette(pg[3], *HOLLADAY_BAR)
    return pals


_HR = None


def twin_ok(v, twin, step):
    """Twin check: central 4 mm (canonical radius 63 px = 31 grid cells), median |v - twin| <= one palette step."""
    yy, xx = np.mgrid[:N, :N]
    c = np.isfinite(v) & np.isfinite(twin) & (np.hypot(xx - (N - 1) / 2, yy - (N - 1) / 2) < 31)
    return c.sum() > 500 and np.median(np.abs(v[c] - twin[c])) <= step + 1e-6


def decode_file(args):
    """-> {page: (4, N, N) float16 in composite orientation} for maps passing palette AND twin checks; stats."""
    global _HR
    f, row, mirror, okmap = args
    if _HR is None:
        _HR = np.load(DATA / 'decoded_hr.npy', mmap_mode='r')
    twin = _HR[row].astype(np.float32)
    if mirror:  # back to screen orientation
        twin = twin[:, :, ::-1]
    pg = pages(f)
    out, stats = {}, {}
    for k, mm in LAYOUT.items():
        v = np.full((4, N, N), np.nan, np.float32)
        for m, (cx, cy, s, bar, vt) in mm.items():
            ok = False
            if okmap[(k, m)]:
                pal = palette(pg[k], *bar)
                # p1 elevation: the colour bar is the same for +-37.5 / 2.5 um and +-75 / 5 um scales; try both
                tables = [VAL[vt], 2 * VAL[vt]] if vt in ('ef25', 'eb25') else [VAL[vt]]
                for t in tables:
                    d = resample(pg[k], cx, cy, s, pal, t)
                    if twin_ok(d, twin[CH[m]], np.diff(np.sort(t)).min() if vt != 'curv' else 1.0):
                        v[CH[m]], ok = d, True
                        break
            stats[(k, m)] = ok
        if np.isfinite(v).any():
            out[k] = (v[:, :, ::-1] if mirror else v).astype(np.float16)
    return out, stats


def main():
    jpg = files()
    stem_row = {f.name.lower().rsplit('.', 1)[0]: i for i, f in enumerate(jpg)}
    dcm = sorted(f for f in IMG.iterdir() if f.name.lower().endswith('4 maps select.dcm'))
    if len(sys.argv) > 1:
        dcm = dcm[:int(sys.argv[1])]
    hr_row = [stem_row.get(f.name.lower().rsplit('.', 1)[0], -1) for f in dcm]
    meta_hr = pd.read_parquet(DATA / 'decoded_hr_meta.parquet')
    with Pool(24) as p:
        res = p.map(pal_only, dcm, chunksize=4)
    keys = [(k, m) for k, mm in LAYOUT.items() for m in mm] + [(3, 'ekr')]
    good = [r for r in res if r is not None]
    ref = {}
    for key in keys:
        ok = [r[key] for r in good if len(r[key]) == 31]
        ref[key] = np.median(np.stack(ok), 0) if ok else None
    passed, rows = {}, []
    for key in keys:
        flags = np.array([r is not None and len(r[key]) == 31 and ref[key] is not None
                          and np.sqrt(((r[key] - ref[key]) ** 2).sum(1)).max() < 30 for r in res])
        passed[key] = flags
        rows.append(dict(page=key[0], map=key[1], files=len(res),
                         n_steps_median=int(np.median([len(r[key]) for r in good])),
                         palette_fail_rate=round(1 - flags.mean(), 4)))
    rep = pd.DataFrame(rows)
    rep['page_type_dropped'] = rep.groupby('page').palette_fail_rate.transform('max') > 0.05
    keep_pages = set(rep[~rep.page_type_dropped].page)
    jobs = [(f, hr_row[i], meta_hr.eye.iat[hr_row[i]] == 'left',
             {(k, m): bool(passed[(k, m)][i]) and k in keep_pages for k, mm in LAYOUT.items() for m in mm})
            for i, f in enumerate(dcm) if res[i] is not None and hr_row[i] >= 0]
    with Pool(24) as p:
        dec = p.map(decode_file, jobs, chunksize=4)
    twin = {key: np.mean([d[1][key] for d in dec]) for key in keys if key[0] in keep_pages}
    rep['twin_pass_rate'] = [round(twin.get((r.page, r.map), np.nan), 4) for r in rep.itertuples()]
    print(rep.to_string(index=False))
    out, meta = [], []
    for (f, row, _, _), (pgs, _) in zip(jobs, dec):
        for k, v in pgs.items():
            out.append(v)
            mh = meta_hr.iloc[row]
            meta.append(dict(hr_row=row, pkey=mh.pkey, eye=mh.eye, date=mh.date, page=k))
    meta = pd.DataFrame(meta)
    print(len(dcm), 'DCM files;', sum(r is None for r in res), 'unreadable;', sum(h < 0 for h in hr_row),
          'without a JPG twin;', len(meta), 'decoded page samples:', meta.page.value_counts().to_dict())
    if len(sys.argv) == 1:
        np.save(DATA / 'decoded_dcm.npy', np.stack(out))
        meta.to_parquet(DATA / 'decoded_dcm_meta.parquet', index=False)
        rep.to_csv(Path(__file__).parent / 'results' / 'dcm_palette_check.csv', index=False)


if __name__ == '__main__':
    sys.exit(main())
