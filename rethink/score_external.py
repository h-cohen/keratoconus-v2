"""Score the frozen models (rethink/freeze_models.py) on an external validation file. Prints aggregates only.

Usage:  python -I rethink/score_external.py VALIDATION.csv|.parquet [--mapping cols.csv] [--images DIR]

The file needs one row per scan with:
  pid        patient identifier (any string; used for the fellow-eye join and the cluster bootstrap)
  eye        'right'/'left' (mirrors the training cohort; OD/OS are mapped)
  scan_date  visit date (used ONLY to find the fellow eye scanned at the same visit; never a feature)
  y          1 if the eye was later referred for / underwent CXL, else 0
  age        and the Pentacam numerics with the training names (rethink/numeric_features.txt).
             --mapping takes a 2-column CSV (external_name,cohort_name) to rename columns first.
  image_stem (only with --images) map images are DIR/<image_stem>_<map>.jpg for the four maps
             corneal_thickness, curvature_front, elevation_front, elevation_back, cropped by the same
             pipeline as data/cropped_output_v2. Needed for E10c and E13 (the final model).
Scans taken after the eye's CXL must be excluded beforehand, as in the training cohort.
Needs rethink/frozen/models.pkl and, with --images, the facebook/dinov2-small weights (HF cache).
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import freeze_models as fm  # noqa: E402
from compare import paired_delta  # noqa: E402
from harness import ROOT, cluster_boot_auc  # noqa: E402


def image_grid(df, img_dir):
    from extract_dino_grid import grid_features, load_encoder
    proc, model = load_encoder()
    return np.stack([grid_features([Path(img_dir) / f'{s}_{m}.jpg' for s in df.image_stem], proc, model)
                     .reshape(len(df), -1) for m in fm.MAPS], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--mapping')
    ap.add_argument('--images')
    a = ap.parse_args()
    df = pd.read_parquet(a.path) if a.path.endswith('.parquet') else pd.read_csv(a.path)
    if a.mapping:
        m = pd.read_csv(a.mapping, header=None, names=['ext', 'coh'])
        df = df.rename(columns=dict(zip(m.ext, m.coh)))
    with open(ROOT / 'frozen' / 'models.pkl', 'rb') as f:
        models = pickle.load(f)
    need = ['pid', 'eye', 'scan_date', 'y'] + models['numerics'] + (['image_stem'] if a.images else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f'missing columns: {missing}')
    df = df.reset_index(drop=True)
    df['pid'] = df.pid.astype(str)
    df['eye'] = df.eye.replace({'OD': 'right', 'OS': 'left'})
    df['scan_date'] = pd.to_datetime(df.scan_date)
    y, g = df.y.values.astype(int), df.pid.values

    p = {'age': models['age'].predict_proba(df[['age']].values)[:, 1],
         'E02': fm.predict_E02(models, df), 'E06b': fm.predict_E06b(models, df), 'E12b': fm.predict_E12b(models, df)}
    if a.images:
        G = image_grid(df, a.images)
        p['E10c'], p['E13'] = fm.predict_E10c(models, df, G), fm.predict_E13(models, df, G)
    has = fm.fellow_index(df) >= 0
    print(f'n={len(df)} scans, {df.pid.nunique()} patients, {y.sum()} positive; same-visit fellow for {has.sum()}')
    for k, v in p.items():
        lo, hi = cluster_boot_auc(y, v, g)
        print(f'{k:5s} AUC {roc_auc_score(y, v):.3f} [{lo:.3f}-{hi:.3f}]')
    for k in [k for k in p if k != 'age']:
        d, (lo, hi), pr = paired_delta(y, p[k], p['age'], g)
        print(f'{k} - age: dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P(d>0)={pr:.2f}')
    first = df.sort_values('scan_date').drop_duplicates(['pid', 'eye']).index
    if 0 < y[first].sum() < len(first):
        print('first visit per eye: ' + ', '.join(f'{k} {roc_auc_score(y[first], v[first]):.3f}' for k, v in p.items()))
    for nm, m in [('bilateral visits', has), ('single-eye visits', ~has)]:
        if 0 < y[m].sum() < m.sum():
            print(f'{nm} (n={m.sum()}, pos={y[m].sum()}): ' +
                  ', '.join(f'{k} {roc_auc_score(y[m], v[m]):.3f}' for k, v in p.items()))
    cal = p['E12b']
    print(f'E12b calibration: mean predicted {cal.mean():.3f} vs observed {y.mean():.3f}, '
          f'Brier {np.mean((cal - y) ** 2):.3f}')


if __name__ == '__main__':
    main()
