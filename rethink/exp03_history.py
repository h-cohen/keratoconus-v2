"""E03: visit + history. Separate model, reported apart from the single-visit claim (PROTOCOL.md).

Scope: scans with at least one earlier scan of the same eye (visit_idx >= 1). The reference is
age-only LR evaluated on the SAME scans, so the paired comparison is like-for-like.
History features use only earlier scans of the same eye (no labels, no later data), so they are
built once on the dev table. Ages/time are never used as calendar date.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from compare import paired_delta  # noqa: E402
from exp01_tabular import _inner_cv, engineer  # noqa: E402
from harness import evaluate, load, RES  # noqa: E402

KEY = ['KMax Sagittal Front (D)', 'Pachy Min:', 'Km F (D):', 'K2 F (D):', 'KI:', 'ISV:', 'IHA:',
       'IHD:', 'C.Vol D 3mm:', 'D0mm Pachy']


def history_table(df):
    """Current engineered indices + changes from the PREVIOUS scan of the same eye."""
    d = df.sort_values(['eid', 'scan_date']).copy()
    cur = engineer(d)
    prev = d.groupby('eid')[KEY].shift(1)
    dt = d.groupby('eid')['scan_date'].diff().dt.days / 365.25
    out = cur.copy()
    out['dt_years'] = dt
    for c in KEY:
        out[f'prev_{c}'] = prev[c]
        out[f'd_{c}'] = d[c] - prev[c]
        out[f'rate_{c}'] = (d[c] - prev[c]) / dt.replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)
    out['has_prior'] = dt.notna().astype(float)
    return out.loc[df.index]


def hist_lr():
    """Current + history indices, L2 LR, C by inner CV. Fitted on scans with a prior only."""
    def fp(dtr, ctr, dte, cte, seed):
        Xtr, Xte = history_table(dtr), history_table(dte)
        cols = [c for c in Xtr.columns if c != 'has_prior' and Xtr[c].notna().all()]
        Xtr, Xte = Xtr[cols], Xte[cols]
        med = Xtr.median()
        Xtr, Xte = Xtr.fillna(med).values, Xte.fillna(med).values
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        C = _inner_cv(Xtr, y, g, mk, [dict(C=c) for c in [0.01, 0.03, 0.1, 0.3, 1]], seed)['C']
        return mk(C).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


def hist_small_lr(cols=('age', 'rate_KMax Sagittal Front (D)', 'rate_Pachy Min:', 'dt_years')):
    """Few-feature variant: age + yearly change in KMax and Pachy Min, L2 LR (C fixed by inner CV)."""
    def fp(dtr, ctr, dte, cte, seed):
        Xtr = history_table(dtr)[list(cols)]
        Xte = history_table(dte)[list(cols)]
        med = Xtr.median()
        Xtr, Xte = Xtr.fillna(med).values, Xte.fillna(med).values
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        C = _inner_cv(Xtr, y, g, mk, [dict(C=c) for c in [0.01, 0.1, 1]], seed)['C']
        return mk(C).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


def age_lr():
    def fp(dtr, ctr, dte, cte, seed):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        return m.fit(dtr[['age']].values, dtr.y.values).predict_proba(dte[['age']].values)[:, 1]
    return fp


if __name__ == '__main__':
    full, cnn = load('dev')
    keep = (full.visit_idx >= 1).values
    sub, cs = full[keep].reset_index(drop=True), cnn[keep]
    print(f'history subset: {len(sub)} scans, {sub.eid.nunique()} eyes, {sub.pid.nunique()} patients, '
          f'{sub.y.sum()} positive scans')
    # the features need the previous scan, which may be a visit_idx==0 row: build on the full dev table
    hist_full = history_table(full)
    sub_hist = hist_full[keep].reset_index(drop=True)
    sub = sub.assign(**{c: sub_hist[c].values for c in sub_hist.columns if c not in sub.columns})

    s_ref, oof_ref = evaluate('E03_age_lr_on_history_subset', age_lr(), 'age only, same history subset',
                              df=sub, cnn=cs)
    s_m, oof_m = evaluate('E03_hist_lr', hist_lr(), 'current + prior-scan KC indices & rates, L2 LR',
                          df=sub, cnn=cs)
    s_s, oof_s = evaluate('E03_hist_small_lr', hist_small_lr(),
                          'age + yearly rate of KMax and Pachy Min (prior scan), L2 LR', df=sub, cnn=cs)
    y, g = sub.y.values, sub.pid.values
    for name, oof in [('E03_hist_lr', oof_m), ('E03_hist_small_lr', oof_s)]:
        m, (lo, hi), pgt = paired_delta(y, oof.mean(0), oof_ref.mean(0), g)
        print(f'{name} - age (history subset): dAUC {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(d>0)={pgt:.3f}')
