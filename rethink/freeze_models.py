"""Freeze the single-visit models on all 944 scans for external scoring (rethink/score_external.py).

Models (hyperparameters re-chosen by the same inner patient-grouped CV used in development):
  age     : age only, logistic regression (the reference to beat)
  E02     : age + engineered indices + same-visit fellow eye + asymmetry, L2 LR (best lockbox AUC)
  E06b    : the same fellow features, monotone LightGBM (age -), best dev AUC
Writes rethink/frozen/models.pkl (gitignored: trained on patient data).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from exp01_tabular import _inner_cv  # noqa: E402
from exp06_combo import fellow_X  # noqa: E402
from harness import DATA, ROOT, numeric_features  # noqa: E402

SEED = 2026
LR = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))  # noqa: E731


def fit_all(df):
    y, g = df.y.values, df.pid.values
    out = {'numerics': numeric_features(), 'n_train': len(df), 'seed': SEED}
    out['age'] = LogisticRegression(max_iter=5000).fit(df[['age']].values, y)

    X = fellow_X(df)
    med = X.median()
    Xf = X.fillna(med).values
    C = _inner_cv(Xf, y, g, LR, [dict(C=c) for c in [0.01, 0.03, 0.1, 0.3, 1]], SEED)['C']
    out['E02'] = dict(model=LR(C).fit(Xf, y), cols=list(X.columns), median=med, C=C)

    cols = list(X.columns)
    mono = [-1 if c == 'age' else 0 for c in cols]
    mk = lambda n_estimators, learning_rate, num_leaves: LGBMClassifier(  # noqa: E731
        n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, max_depth=2,
        min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        monotone_constraints=mono, random_state=SEED, verbose=-1, n_jobs=2,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
    grid = [dict(n_estimators=n, learning_rate=0.03, num_leaves=l) for n in [100, 300] for l in [2, 4]]
    hp = _inner_cv(X.values, y, g, mk, grid, SEED)
    out['E06b'] = dict(model=mk(**hp).fit(X.values, y), cols=cols, hp=hp)
    return out


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    models = fit_all(coh)
    d = ROOT / 'frozen'
    d.mkdir(exist_ok=True)
    with open(d / 'models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print(f"frozen on {len(coh)} scans: E02 C={models['E02']['C']}, E06b {models['E06b']['hp']}")
