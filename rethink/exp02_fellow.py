"""E02: fellow eye at the same visit. Single-visit compatible (PROTOCOL.md 'Allowed inputs').

For each scan: the other eye of the same patient scanned on the same date (if any). Only that
scan's numerics are used, never its label. Missing fellow scans are imputed with the training-fold
median and flagged with has_fellow, which describes the index visit, not the future.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from exp01_tabular import _inner_cv, engineer  # noqa: E402
from harness import evaluate, load  # noqa: E402


def fellow_table(df):
    """Engineered indices of the same-visit fellow eye, plus asymmetry, aligned to df rows."""
    own = engineer(df)
    other = df[['pid', 'scan_date', 'eye']].copy()
    fel = engineer(df).copy()
    fel.index = df.index
    fel = pd.concat([other, fel], axis=1)
    fel['key'] = fel.pid + '|' + fel.scan_date.astype(str)
    own_key = df.pid + '|' + df.scan_date.astype(str)
    # the fellow is the other eye of the same key; keep only keys with exactly two eyes
    cnt = fel.groupby('key').eye.transform('nunique')
    rows = []
    for i, (k, eye) in enumerate(zip(own_key, df.eye)):
        m = fel[(fel.key == k) & (fel.eye != eye)]
        if len(m) == 1 and cnt.iloc[m.index[0]] == 2 if len(m) else False:
            rows.append(m.index[0])
        else:
            rows.append(None)
    out = pd.DataFrame(index=df.index)
    out['has_fellow'] = [1.0 if r is not None else 0.0 for r in rows]
    fcols = [c for c in own.columns if c != 'age']
    for c in fcols:
        vals = [fel.at[r, c] if r is not None else np.nan for r in rows]
        out[f'fel_{c}'] = vals
        out[f'asym_{c}'] = own[c].values - np.array(vals, dtype=float)
    return out


def fellow_lr(use_asym=True, no_flag=False):
    def fp(dtr, ctr, dte, cte, seed):
        def build(d):
            e = engineer(d)
            f = fellow_table(d)
            X = e if no_flag else pd.concat([e, f[['has_fellow']]], axis=1)
            cols = [c for c in f.columns if c.startswith('fel_') or (use_asym and c.startswith('asym_'))]
            X = pd.concat([X, f[cols]], axis=1)
            return X
        Xtr, Xte = build(dtr), build(dte)
        med = Xtr.median()
        Xtr, Xte = Xtr.fillna(med).values, Xte.fillna(med).values
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        C = _inner_cv(Xtr, y, g, mk, [dict(C=c) for c in [0.01, 0.03, 0.1, 0.3, 1]], seed)['C']
        return mk(C).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


if __name__ == '__main__':
    df, _ = load('dev')
    f = fellow_table(df)
    print(f'scans with a same-visit fellow scan: {int(f.has_fellow.sum())} of {len(df)}')
    evaluate('E02_fellow_lr', fellow_lr(), 'age + engineered indices + same-visit fellow eye + asymmetry, L2 LR')
    evaluate('E02b_fellow_noflag', fellow_lr(no_flag=True),
             'as E02_fellow_lr but without the has_fellow indicator (ablation)')
