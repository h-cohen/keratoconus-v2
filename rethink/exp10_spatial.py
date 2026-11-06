"""E10: spatial image features + raw numerics, joined several ways. All 944 scans, outer CV.

Image blocks (all fitted inside each training fold):
  S  spatial DINOv2: each map's 4x4-cell features (16 x 384) flattened, standardised, PCA(10) per map.
  C  colour grid: 4 maps x 16 cells x (cos hue, sin hue, sat, val) = 256 values, standardised, PCA(10).
Raw: the 55 Pentacam numerics including age.
Mixture difference (D): for scans with a fellow eye scanned at the same visit, the difference between
  this eye's and the fellow's S-block PCA scores, plus a has_fellow flag (0 difference when absent).

E10a_spatial_lr        : [S | numerics] -> L2 LR (C by inner CV)
E10b_spatial_color_lgbm: [S | C | numerics] -> LightGBM (small fixed trees)
E10c_spatial_diff_lr   : [S | numerics | D] -> L2 LR (C by inner CV)
E10d_stack             : late fusion. Image LR on [S | C] and numeric LightGBM, inner OOF, logistic combiner.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from harness import DATA, evaluate, numeric_features  # noqa: E402

MAPS = ['corneal_thickness', 'curvature_front', 'elevation_front', 'elevation_back']
NUM = numeric_features()
K = 10
LR = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))  # noqa: E731
GRID = np.concatenate([np.load(DATA / f'dinogrid_{m}.npy').reshape(-1, 16 * 384) for m in MAPS], 1)
COLOR = np.load(DATA / 'colorgrid.npy').reshape(-1, 4 * 16 * 4)


def fellow_index(df):
    """Row index of the other eye scanned on the same date for the same patient, or -1."""
    key = df.pid + '|' + df.scan_date.astype(str)
    idx = np.full(len(df), -1)
    for k, grp in df.groupby(key).groups.items():
        rows = list(grp)
        if len(rows) == 2 and df.eye.iloc[rows[0]] != df.eye.iloc[rows[1]]:
            idx[rows[0]], idx[rows[1]] = rows[1], rows[0]
    return idx


def pca_blocks(crow_fit, crow_apply, k=K, seed=0):
    """Per-map (S) and colour (C) PCA fitted on crow_fit rows, applied to crow_apply rows."""
    S_fit, S_app = [], []
    for j in range(len(MAPS)):
        X = GRID[:, j * 16 * 384:(j + 1) * 16 * 384]
        sc = StandardScaler().fit(X[crow_fit])
        p = PCA(n_components=k, random_state=seed).fit(sc.transform(X[crow_fit]))
        S_fit.append(p.transform(sc.transform(X[crow_fit])))
        S_app.append(p.transform(sc.transform(X[crow_apply])))
    sc = StandardScaler().fit(COLOR[crow_fit])
    p = PCA(n_components=k, random_state=seed).fit(sc.transform(COLOR[crow_fit]))
    C_fit, C_app = p.transform(sc.transform(COLOR[crow_fit])), p.transform(sc.transform(COLOR[crow_apply]))
    return np.hstack(S_fit), np.hstack(S_app), C_fit, C_app


def _features(d, crow_fit, seed, use):
    """Build the requested block matrix for the rows of `d`, with PCA fitted on `crow_fit`.
    Folds are patient-grouped, so a same-visit fellow is always inside `d`."""
    crow_apply = d.crow.values
    S_fit, S_app, C_fit, C_app = pca_blocks(crow_fit, crow_apply, seed=seed)
    parts = {'S': S_app, 'C': C_app, 'num': d[NUM].values}
    if 'D' in use:
        fel = fellow_index(d)
        has = (fel >= 0).astype(float)
        # fellow scores come from the same PCA, computed for the fellow rows (crow of the fellow)
        _, S_fel, _, _ = pca_blocks(crow_fit, np.where(fel >= 0, crow_apply[np.maximum(fel, 0)], 0), seed=seed)
        diff = np.where(has[:, None] > 0, S_app - S_fel, 0.0)
        parts['D'] = np.hstack([diff, has[:, None]])
    return np.hstack([parts[u] for u in use])


def _make(use, model, seed_name):
    def fp(dtr, ctr, dte, cte, seed):
        crow_tr = dtr.crow.values
        y, g = dtr.y.values, dtr.pid.values
        Xtr = _features(dtr, crow_tr, seed, use)
        Xte = _features(dte, crow_tr, seed, use)
        if model == 'lr':
            best, best_s = 0.1, -1
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            for C in [0.01, 0.03, 0.1, 0.3, 1]:
                s = []
                for a, b in inner.split(dtr, y, g):
                    da, db = dtr.iloc[a].reset_index(drop=True), dtr.iloc[b].reset_index(drop=True)
                    Xa = _features(da, crow_tr[a], seed, use)
                    Xb = _features(db, crow_tr[a], seed, use)
                    s.append(roc_auc_score(y[b], LR(C).fit(Xa, y[a]).predict_proba(Xb)[:, 1]))
                if np.mean(s) > best_s:
                    best, best_s = C, np.mean(s)
            return LR(best).fit(Xtr, y).predict_proba(Xte)[:, 1]
        m = LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=2,
                           min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                           random_state=seed, verbose=-1, n_jobs=4,
                           scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
        return m.fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


def stack():
    """Image LR on [S|C] and numeric LightGBM, combined by a logistic combiner on inner OOF logits."""
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def level1(dtr, dte, seed):
        Xi_tr = _features(dtr, dtr.crow.values, seed, ['S', 'C'])
        Xi_te = _features(dte, dtr.crow.values, seed, ['S', 'C'])
        pi = LR(0.1).fit(Xi_tr, dtr.y.values).predict_proba(Xi_te)[:, 1]
        m = LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=2, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=seed, verbose=-1,
                           n_jobs=4, scale_pos_weight=(dtr.y == 0).sum() / max((dtr.y == 1).sum(), 1))
        pn = m.fit(dtr[NUM].values, dtr.y.values).predict_proba(dte[NUM].values)[:, 1]
        return pi, pn

    def fp(dtr, ctr, dte, cte, seed):
        y, g = dtr.y.values, dtr.pid.values
        oof = np.zeros((len(dtr), 2))
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        for a, b in inner.split(dtr, y, g):
            da, db = dtr.iloc[a].reset_index(drop=True), dtr.iloc[b].reset_index(drop=True)
            oof[b, 0], oof[b, 1] = level1(da, db, seed)
        comb = LogisticRegression(C=1.0, max_iter=5000).fit(logit(oof), y)
        pi, pn = level1(dtr, dte, seed)
        return comb.predict_proba(logit(np.column_stack([pi, pn])))[:, 1]
    return fp


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    evaluate('E10a_spatial_lr', _make(['S', 'num'], 'lr', 'a'),
             'spatial DINOv2 (4x4 cells, PCA10/map) + 55 numerics, L2 LR', df=coh, cnn=cnn)
    evaluate('E10b_spatial_color_lgbm', _make(['S', 'C', 'num'], 'lgbm', 'b'),
             'spatial DINOv2 + colour grid + numerics, LightGBM', df=coh, cnn=cnn)
    evaluate('E10c_spatial_diff_lr', _make(['S', 'num', 'D'], 'lr', 'c'),
             'spatial DINOv2 + numerics + fellow-eye image difference, L2 LR', df=coh, cnn=cnn)
    evaluate('E10d_stack', stack(), 'late fusion: image LR on [spatial|colour] + numeric LGBM', df=coh, cnn=cnn)
