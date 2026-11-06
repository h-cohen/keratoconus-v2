"""Shared evaluation harness for the clean re-analysis. See PROTOCOL.md.

Every experiment goes through `evaluate()`, which
  * uses ONLY the development set (lockbox patients are removed here and nowhere else),
  * runs repeated patient-grouped stratified K-fold CV,
  * hands the model a fit/predict pair so ALL preprocessing (scaling, selection,
    imputation, early stopping, tuning) happens inside the training fold,
  * appends one row to rethink/results/experiments.csv (the full search log; nothing is deleted).
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
RES = ROOT / 'results'
N_SPLITS, N_REPEATS, SEED = 5, 5, 2026
N_BOOT = 1000


def load(split='dev'):
    """split: 'dev' (default), 'lockbox' or 'all'. Returns (df, cnn) row-aligned."""
    df = pd.read_parquet(DATA / 'cohort.parquet')
    cnn = np.load(DATA / 'cnn_frozen.npy')
    if split == 'all':
        return df.reset_index(drop=True), cnn
    lock = set(pd.read_csv(ROOT / 'lockbox_pids.csv').pid)
    m = df.pid.isin(lock).values
    m = m if split == 'lockbox' else ~m
    return df[m].reset_index(drop=True), cnn[m]


def numeric_features():
    df = pd.read_parquet(DATA / 'cohort.parquet')
    meta = {'pid', 'eid', 'eye', 'scan_date', 'visit_idx', 'first_visit', 'y', 'months_to_cxl', 'legacy_row'}
    return [c for c in df.columns if c not in meta]


def cluster_boot_auc(y, p, groups, seed=0, n=N_BOOT):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    out = []
    for _ in range(n):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug))])
        if y[idx].min() != y[idx].max():
            out.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(out, [2.5, 97.5])


def cv_oof(df, cnn, fit_predict, n_repeats=N_REPEATS, seed=SEED):
    """fit_predict(df_tr, cnn_tr, df_te, cnn_te, seed) -> probabilities for df_te."""
    y, g = df.y.values, df.pid.values
    oof = np.zeros((n_repeats, len(df)))
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(df, y, g):
            oof[r, te] = fit_predict(df.iloc[tr].reset_index(drop=True), cnn[tr],
                                     df.iloc[te].reset_index(drop=True), cnn[te], seed + r)
    return oof


def summarise(df, oof):
    y, g = df.y.values, df.pid.values
    rep = [roc_auc_score(y, o) for o in oof]
    p = oof.mean(0)
    fv = df.first_visit.values
    lo, hi = cluster_boot_auc(y, p, g)
    flo, fhi = cluster_boot_auc(y[fv], p[fv], g[fv])
    return dict(auc=roc_auc_score(y, p), auc_lo=lo, auc_hi=hi,
                auc_rep_mean=float(np.mean(rep)), auc_rep_sd=float(np.std(rep)),
                auc_first=roc_auc_score(y[fv], p[fv]), auc_first_lo=flo, auc_first_hi=fhi,
                n=len(y), n_pos=int(y.sum()))


def _git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return 'na'


def evaluate(name, fit_predict, desc='', params=None, df=None, cnn=None, save_oof=True):
    if df is None:
        df, cnn = load('dev')
    t = time.time()
    oof = cv_oof(df, cnn, fit_predict)
    s = summarise(df, oof)
    s.update(name=name, desc=desc, params=json.dumps(params or {}), git=_git_hash(),
             secs=round(time.time() - t, 1), when=time.strftime('%Y-%m-%d %H:%M'))
    RES.mkdir(exist_ok=True)
    log = RES / 'experiments.csv'
    pd.DataFrame([s]).to_csv(log, mode='a', header=not log.exists(), index=False)
    if save_oof:
        (RES / 'oof').mkdir(exist_ok=True)
        np.save(RES / 'oof' / f'{name}.npy', oof)
    print(f"{name:32s} AUC {s['auc']:.3f} [{s['auc_lo']:.3f}-{s['auc_hi']:.3f}]  "
          f"rep {s['auc_rep_mean']:.3f}±{s['auc_rep_sd']:.3f}  first-visit {s['auc_first']:.3f}  "
          f"({s['secs']}s)")
    return s, oof
