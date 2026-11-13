"""E05: extra scan-level indices from the raw Pentacam export (prepare_raw.py), with age, L2 LR.

TKC is a categorical grade; clean grades are ordinal-encoded, free-text annotations become NaN
(median-imputed inside the fold). Axis angles enter as sin/cos (2θ). Nothing from KC diagnosis,
CXL or keratoplasty fields is used.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from exp01_tabular import _inner_cv  # noqa: E402
from harness import DATA, evaluate, load, ROOT  # noqa: E402

TKC = {'-': 0, 'KC 1': 1, 'KC 1-2': 1.5, 'KC 2': 2, 'KC 2-3': 2.5, 'KC 3': 3, 'KC 3-4': 3.5}


def raw_block(ex, rows):
    e = ex.iloc[rows]
    out = pd.DataFrame({'tkc': e['TKC:'].astype(str).map(TKC).values})
    for c in ['Pupil Pos X:', 'Pupil Pos Y:', 'RSagMin:', 'Pachy Min Pos X:', 'Pachy Min Pos Y:']:
        out[c] = pd.to_numeric(e[c], errors='coerce').values
    for c in ['Axis F (flat):', 'Axis B (flat):']:
        a = np.deg2rad(pd.to_numeric(e[c], errors='coerce').values)
        out[c + ' sin'] = np.sin(2 * a)
        out[c + ' cos'] = np.cos(2 * a)
    return out


def extra_lr():
    ex_all = pd.read_parquet(DATA / 'raw_extra.parquet')

    def fp(dtr, ctr, dte, cte, seed):
        Btr = raw_block(ex_all, dtr.crow.values)
        Bte = raw_block(ex_all, dte.crow.values)
        med = Btr.median()
        Xtr = np.hstack([Btr.fillna(med).values, dtr[['age']].values])
        Xte = np.hstack([Bte.fillna(med).values, dte[['age']].values])
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        C = _inner_cv(Xtr, y, g, mk, [dict(C=c) for c in [0.01, 0.03, 0.1, 0.3, 1]], seed)['C']
        return mk(C).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    lock = set(pd.read_csv(ROOT / 'lockbox_pids.csv').pid)
    keep = ~coh.pid.isin(lock).values
    df = coh[keep].reset_index(drop=True)
    cnn = np.load(DATA / 'cnn_frozen.npy')[keep]
    evaluate('E05_raw_extra_lr', extra_lr(), 'age + TKC grade + pupil/pachy-min position + RSagMin + axis sin/cos, L2 LR',
             df=df, cnn=cnn)
