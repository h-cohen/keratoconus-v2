"""Score the frozen models (rethink/freeze_models.py) on an external validation file. Prints aggregates only.

Usage:  python -I rethink/score_external.py VALIDATION.csv|.parquet [--mapping cols.csv]

The file needs one row per scan with:
  pid        patient identifier (any string; used for the fellow-eye join and the cluster bootstrap)
  eye        'OD'/'OS' (or any two distinct values per patient)
  scan_date  visit date (used ONLY to find the fellow eye scanned at the same visit; never a feature)
  y          1 if the eye was later referred for / underwent CXL, else 0
  age        and the 55 Pentacam numerics with the same names as the training cohort
             (list: data/multimodal_dataset_features.txt). --mapping takes a 2-column CSV
             (external_name,cohort_name) to rename columns first.
Scans taken after the eye's CXL must be excluded beforehand, as in the training cohort.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from compare import paired_delta  # noqa: E402
from exp06_combo import fellow_X  # noqa: E402
from harness import ROOT, cluster_boot_auc  # noqa: E402


def predict(models, df):
    X = fellow_X(df)
    e02, e06 = models['E02'], models['E06b']
    return {
        'age': models['age'].predict_proba(df[['age']].values)[:, 1],
        'E02': e02['model'].predict_proba(X[e02['cols']].fillna(e02['median']).values)[:, 1],
        'E06b': e06['model'].predict_proba(X[e06['cols']].values)[:, 1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--mapping')
    a = ap.parse_args()
    df = pd.read_parquet(a.path) if a.path.endswith('.parquet') else pd.read_csv(a.path)
    if a.mapping:
        m = pd.read_csv(a.mapping, header=None, names=['ext', 'coh'])
        df = df.rename(columns=dict(zip(m.ext, m.coh)))
    with open(ROOT / 'frozen' / 'models.pkl', 'rb') as f:
        models = pickle.load(f)
    need = ['pid', 'eye', 'scan_date', 'y'] + models['numerics']
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f'missing columns: {missing}')
    df = df.reset_index(drop=True)
    df['pid'] = df.pid.astype(str)
    df['scan_date'] = pd.to_datetime(df.scan_date)
    y, g = df.y.values.astype(int), df.pid.values
    p = predict(models, df)
    has_fel = fellow_X(df).has_fellow.values
    print(f'n={len(df)} scans, {df.pid.nunique()} patients, {y.sum()} positive; '
          f'same-visit fellow for {int(has_fel.sum())}')
    for k, v in p.items():
        lo, hi = cluster_boot_auc(y, v, g)
        print(f'{k:5s} AUC {roc_auc_score(y, v):.3f} [{lo:.3f}-{hi:.3f}]')
    for k in ['E02', 'E06b']:
        d, (lo, hi), pr = paired_delta(y, p[k], p['age'], g)
        print(f'{k} - age: dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P(d>0)={pr:.2f}')
    first = df.sort_values('scan_date').drop_duplicates(['pid', 'eye']).index
    yf = y[first]
    if 0 < yf.sum() < len(yf):
        print('first visit per eye: ' + ', '.join(f'{k} {roc_auc_score(yf, v[first]):.3f}' for k, v in p.items()))


if __name__ == '__main__':
    main()
