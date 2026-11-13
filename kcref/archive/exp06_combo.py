"""E06: pre-declared combinations of the best components, all fitted inside the training fold.

E06a: fellow-eye features (E02) + DINOv2 PCA blocks (E04) + age, L2 LR, C by inner CV.
E06b: fellow-eye features (E02) + age, LightGBM depth 2, age monotone (-), inner CV on trees.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from exp01_tabular import _inner_cv, engineer  # noqa: E402
from exp02_fellow import fellow_table  # noqa: E402
from exp04_dino import dino_block  # noqa: E402
from harness import DATA, ROOT, evaluate  # noqa: E402


def fellow_X(d):
    e, f = engineer(d), fellow_table(d)
    X = pd.concat([e, f[['has_fellow']]], axis=1)
    cols = [c for c in f.columns if c.startswith('fel_') or c.startswith('asym_')]
    return pd.concat([X, f[cols]], axis=1)


def combo_lr():
    def fp(dtr, ctr, dte, cte, seed):
        itr, ite = dtr.crow.values, dte.crow.values
        Ftr, Fte = fellow_X(dtr), fellow_X(dte)
        med = Ftr.median()
        Ftr, Fte = Ftr.fillna(med).values, Fte.fillna(med).values
        Btr, Bte = dino_block(itr, ite, 10, seed)
        Xtr = np.hstack([Ftr, Btr, dtr[['age']].values])
        Xte = np.hstack([Fte, Bte, dte[['age']].values])
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        C = _inner_cv(Xtr, y, g, mk, [dict(C=c) for c in [0.01, 0.03, 0.1, 0.3]], seed)['C']
        return mk(C).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


def combo_lgbm():
    def fp(dtr, ctr, dte, cte, seed):
        Xtr, Xte = fellow_X(dtr), fellow_X(dte)
        cols = list(Xtr.columns)
        mono = [-1 if c == 'age' else 0 for c in cols]
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda n_estimators, learning_rate, num_leaves: LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, max_depth=2,
            min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            monotone_constraints=mono, random_state=seed, verbose=-1, n_jobs=2,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
        grid = [dict(n_estimators=n, learning_rate=0.03, num_leaves=l) for n in [100, 300] for l in [2, 4]]
        hp = _inner_cv(Xtr.values, y, g, mk, grid, seed)
        return mk(**hp).fit(Xtr.values, y).predict_proba(Xte.values)[:, 1]
    return fp


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    lock = set(pd.read_csv(ROOT / 'lockbox_pids.csv').pid)
    keep = ~coh.pid.isin(lock).values
    df = coh[keep].reset_index(drop=True)
    cnn = np.load(DATA / 'cnn_frozen.npy')[keep]
    evaluate('E06a_fellow_dino_lr', combo_lr(), 'E02 fellow features + E04 DINOv2 PCA10 + age, L2 LR', df=df, cnn=cnn)
    evaluate('E06b_fellow_mono_lgbm', combo_lgbm(), 'E02 fellow features + age, monotone LightGBM', df=df, cnn=cnn)
