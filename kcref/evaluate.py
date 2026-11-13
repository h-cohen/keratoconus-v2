"""Evaluation: repeated patient-grouped cross-validation, AUC with patient-cluster bootstrap CIs, paired deltas.

Outer CV: 5-fold StratifiedGroupKFold by patient, repeated 5 times (seeds 2026-2030). fit_predict(dtr, dte, seed)
receives the training and test cohort rows and must fit everything (preprocessing, tuning) on dtr only.
Each run appends a row to results/experiments.csv and saves its out-of-fold scores to results/oof/<name>.npy.
"""
import json
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common import RES, ROOT

N_SPLITS, N_REPEATS, SEED, N_BOOT = 5, 5, 2026, 1000


def cluster_boot_auc(y, p, groups, seed=0, n=N_BOOT):
    """95% CI of the AUC, resampling patients."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    out = []
    for _ in range(n):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug))])
        if y[idx].min() != y[idx].max():
            out.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(out, [2.5, 97.5])


def paired_delta(y, p1, p2, groups, seed=0, n=N_BOOT):
    """AUC(p1) − AUC(p2) on the same scans, patient-cluster bootstrap: (mean, [2.5, 97.5] %, P(delta > 0))."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    d = []
    for _ in range(n):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug))])
        if y[idx].min() != y[idx].max():
            d.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    d = np.array(d)
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5]), float(np.mean(d > 0))


def cv_oof(df, fit_predict, n_repeats=N_REPEATS, seed=SEED):
    y, g = df.y.values, df.pid.values
    oof = np.zeros((n_repeats, len(df)))
    for r in range(n_repeats):
        for tr, te in StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed + r).split(df, y, g):
            oof[r, te] = fit_predict(df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True), seed + r)
    return oof


def summarise(df, oof):
    """Pooled AUC of the repeat-averaged scores (all scans and first visit, with CIs) and per-repeat AUCs."""
    y, g, fv = df.y.values, df.pid.values, df.first_visit.values
    p = oof.mean(0)
    rep = [roc_auc_score(y, o) for o in oof]
    lo, hi = cluster_boot_auc(y, p, g)
    flo, fhi = cluster_boot_auc(y[fv], p[fv], g[fv])
    return dict(auc=roc_auc_score(y, p), auc_lo=lo, auc_hi=hi, auc_rep_mean=float(np.mean(rep)),
                auc_rep_sd=float(np.std(rep)), auc_first=roc_auc_score(y[fv], p[fv]), auc_first_lo=flo,
                auc_first_hi=fhi, n=len(y), n_pos=int(y.sum()))


def evaluate(name, fit_predict, df, desc='', params=None):
    t = time.time()
    oof = cv_oof(df, fit_predict)
    s = summarise(df, oof)
    try:
        git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        git = 'na'
    s.update(name=name, desc=desc, params=json.dumps(params or {}), git=git, secs=round(time.time() - t, 1),
             when=time.strftime('%Y-%m-%d %H:%M'))
    (RES / 'oof').mkdir(parents=True, exist_ok=True)
    log = RES / 'experiments.csv'
    pd.DataFrame([s]).to_csv(log, mode='a', header=not log.exists(), index=False)
    np.save(RES / 'oof' / f'{name}.npy', oof)
    print(f"{name:24s} AUC {s['auc']:.3f} [{s['auc_lo']:.3f}-{s['auc_hi']:.3f}]  rep {s['auc_rep_mean']:.3f}"
          f"±{s['auc_rep_sd']:.3f}  first visit {s['auc_first']:.3f} [{s['auc_first_lo']:.3f}-"
          f"{s['auc_first_hi']:.3f}]  ({s['secs']}s)")
    return s, oof
