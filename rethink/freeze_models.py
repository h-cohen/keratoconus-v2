"""Freeze the single-visit models on all 944 scans for external scoring (rethink/score_external.py).

Models (hyperparameters re-chosen by the same inner patient-grouped CV used in development):
  age     : age only, logistic regression (the reference to beat)
  E02     : age + engineered indices + same-visit fellow eye + asymmetry, L2 LR (best lockbox AUC)
  E06b    : the same fellow features, monotone LightGBM (age -), best dev AUC
  E12b    : E02 + E06b quantile average, Platt-calibrated on inner OOF
  E10c    : own spatial DINOv2 PCA + 55 numerics + fellow-eye image difference, L2 LR (needs map images)
  E13     : mean quantile of E12b and E10c -- the final model (all-data CV AUC 0.719); a ranking score,
            not a probability (use E12b for calibrated risk)
Writes rethink/frozen/models.pkl (gitignored: trained on patient data).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from exp01_tabular import _inner_cv  # noqa: E402
from exp02_fellow import fellow_lr  # noqa: E402
from exp06_combo import combo_lgbm, fellow_X  # noqa: E402
from exp10_spatial import MAPS, fellow_index  # noqa: E402
from exp12_ensemble import _rank  # noqa: E402
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

    # E12b: quantiles of E02 and E06b within their training predictions, averaged, Platt on inner OOF
    lr, gbm = fellow_lr(), combo_lgbm()

    def e12_score(dtr, dte):
        a = _rank(lr(dtr, None, dtr, None, SEED), lr(dtr, None, dte, None, SEED))
        b = _rank(gbm(dtr, None, dtr, None, SEED), gbm(dtr, None, dte, None, SEED))
        return (a + b) / 2
    oof = np.zeros(len(df))
    for a, b in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(df, y, g):
        oof[b] = e12_score(df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True))
    out['E12b'] = dict(cal=LogisticRegression(max_iter=5000).fit(oof[:, None], y),
                       ref_E02=predict_E02(out, df), ref_E06b=predict_E06b(out, df))

    # E10c: own spatial DINOv2 PCA(10)/map + 55 numerics + fellow PCA-score difference + has_fellow
    G = np.stack([np.load(DATA / f'dinogrid_{m}.npy').reshape(len(df), -1) for m in MAPS], 1)
    pcas = []
    for j in range(len(MAPS)):
        sc = StandardScaler().fit(G[:, j])
        pcas.append((sc, PCA(n_components=10, random_state=SEED).fit(sc.transform(G[:, j]))))
    out['E10c'] = dict(pcas=pcas)
    Xc = e10c_features(out, df, G)
    C = _inner_cv(Xc, y, g, LR, [dict(C=c) for c in [0.01, 0.03, 0.1, 0.3, 1]], SEED)['C']
    out['E10c'].update(model=LR(C).fit(Xc, y), C=C)

    # E13: mean of the quantiles of E12b and E10c within their training predictions (ranks only, no fit)
    out['E13'] = dict(ref_E12b=predict_E12b(out, df), ref_E10c=out['E10c']['model'].predict_proba(Xc)[:, 1])
    return out


def predict_E02(models, df):
    m = models['E02']
    return m['model'].predict_proba(fellow_X(df)[m['cols']].fillna(m['median']).values)[:, 1]


def predict_E06b(models, df):
    m = models['E06b']
    return m['model'].predict_proba(fellow_X(df)[m['cols']].values)[:, 1]


def predict_E12b(models, df):
    m = models['E12b']
    s = (_rank(m['ref_E02'], predict_E02(models, df)) + _rank(m['ref_E06b'], predict_E06b(models, df))) / 2
    return m['cal'].predict_proba(s[:, None])[:, 1]


def e10c_features(models, df, G):
    """G: (len(df), 4 maps, 16*384) DINOv2 grid features aligned with df rows."""
    S = np.hstack([p.transform(sc.transform(G[:, j])) for j, (sc, p) in enumerate(models['E10c']['pcas'])])
    fel = fellow_index(df.reset_index(drop=True))
    has = fel >= 0
    D = np.where(has[:, None], S - S[np.maximum(fel, 0)], 0.0)
    return np.hstack([S, df[models['numerics']].values, D, has[:, None].astype(float)])


def predict_E10c(models, df, G):
    return models['E10c']['model'].predict_proba(e10c_features(models, df, G))[:, 1]


def predict_E13(models, df, G):
    m = models['E13']
    return (_rank(m['ref_E12b'], predict_E12b(models, df)) + _rank(m['ref_E10c'], predict_E10c(models, df, G))) / 2


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    models = fit_all(coh)
    d = ROOT / 'frozen'
    d.mkdir(exist_ok=True)
    with open(d / 'models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print(f"frozen on {len(coh)} scans: E02 C={models['E02']['C']}, E06b {models['E06b']['hp']}, "
          f"E10c C={models['E10c']['C']}")
