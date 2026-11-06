"""E04: frozen DINOv2-small features (4 maps x 768) -> within-fold standardisation + PCA -> L2 LR.

Age is always included. PCA size (k per map) and C are chosen by inner patient-grouped CV on the
training fold only. Features come from extract_dino.py, which reads no labels.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from harness import DATA, evaluate, load  # noqa: E402

MAPS = ['corneal_thickness', 'curvature_front', 'elevation_front', 'elevation_back']
ALL = np.concatenate([np.load(DATA / f'dino_{m}.npy') for m in MAPS], 1)  # rows = cohort.parquet


def dino_block(idx_tr, idx_te, k, seed):
    """Within-fold: per-map standardise + PCA(k) fitted on training rows only."""
    tr_parts, te_parts = [], []
    for j in range(len(MAPS)):
        X = ALL[:, j * 768:(j + 1) * 768]
        sc = StandardScaler().fit(X[idx_tr])
        p = PCA(n_components=k, random_state=seed).fit(sc.transform(X[idx_tr]))
        tr_parts.append(p.transform(sc.transform(X[idx_tr])))
        te_parts.append(p.transform(sc.transform(X[idx_te])))
    return np.hstack(tr_parts), np.hstack(te_parts)


def dino_lr(k=10):
    """Needs each scan's row in the feature matrices, carried as column `crow`."""
    def fp(dtr, ctr, dte, cte, seed):
        itr, ite = dtr.crow.values, dte.crow.values
        Btr, Bte = dino_block(itr, ite, k, seed)
        Xtr = np.hstack([Btr, dtr[['age']].values])
        Xte = np.hstack([Bte, dte[['age']].values])
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        best, best_s = 0.1, -1
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        for C in [0.01, 0.03, 0.1, 0.3, 1]:
            s = []
            for a, b in inner.split(Xtr, y, g):
                # PCA inside the inner fold too, so tuning never sees its validation rows
                Ba, Bb = dino_block(itr[a], itr[b], k, seed)
                m = mk(C).fit(np.hstack([Ba, dtr.age.values[a][:, None]]), y[a])
                s.append(roc_auc_score(y[b], m.predict_proba(np.hstack([Bb, dtr.age.values[b][:, None]]))[:, 1]))
            if np.mean(s) > best_s:
                best, best_s = C, np.mean(s)
        return mk(best).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


if __name__ == '__main__':
    import pandas as pd
    from harness import ROOT
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))  # row in the feature matrices
    lock = set(pd.read_csv(ROOT / 'lockbox_pids.csv').pid)
    df = coh[~coh.pid.isin(lock)].reset_index(drop=True)  # same rows, same order as load('dev')
    cnn = np.load(DATA / 'cnn_frozen.npy')[~coh.pid.isin(lock).values]
    for k in [10]:
        evaluate(f'E04_dino_pca{k}_lr', dino_lr(k), f'DINOv2-small, per-map PCA k={k} (4 maps) + age, L2 LR',
                 df=df, cnn=cnn)
