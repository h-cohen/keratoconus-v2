"""E09: IMAGE + RAW NUMERIC fusion, all 944 scans, outer CV as harness.evaluate.

Image block: frozen DINOv2-small, 4 maps, per-map standardise + PCA(k) fitted in-fold.
Raw block: the 55 Pentacam numerics including age (harness.numeric_features()).

E09a_concat_lr    : [image PCA | standardised numerics] -> L2 LR, C by inner CV (PCA refit per inner fold).
E09b_concat_lgbm  : [image PCA | numerics] -> LightGBM, fixed small trees (depth 2, 200 trees, lr 0.03).
E09c_stack        : late fusion. Level-1 image LR and numeric LR (C=0.1) give patient-grouped inner
                    out-of-fold scores; a logistic combiner on their logits is fitted on those OOF scores,
                    then applied to level-1 models refit on the whole outer training fold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from exp01_tabular import _inner_cv  # noqa: E402
from exp04_dino import dino_block  # noqa: E402
from harness import DATA, evaluate, numeric_features  # noqa: E402

NUM = numeric_features()
K = 5
LR = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))  # noqa: E731


def _split_block(dtr, itr, ite, dte, seed, k=K):
    """Image PCA block fitted on itr only, applied to ite."""
    return dino_block(itr, ite, k, seed)


def concat_lr():
    def fp(dtr, ctr, dte, cte, seed):
        itr, ite = dtr.crow.values, dte.crow.values
        y, g = dtr.y.values, dtr.pid.values

        def X(a_idx, b_idx, da, db):
            Bi, Bb = dino_block(a_idx, b_idx, K, seed)
            return np.hstack([Bi, da[NUM].values]), np.hstack([Bb, db[NUM].values])

        best, best_s = 0.1, -1
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        for C in [0.01, 0.03, 0.1, 0.3, 1]:
            s = []
            for a, b in inner.split(dtr, y, g):
                da, db = dtr.iloc[a].reset_index(drop=True), dtr.iloc[b].reset_index(drop=True)
                Xa, Xb = X(itr[a], itr[b], da, db)
                s.append(roc_auc(y[b], LR(C).fit(Xa, y[a]).predict_proba(Xb)[:, 1]))
            if np.mean(s) > best_s:
                best, best_s = C, np.mean(s)
        Xtr, Xte = X(itr, ite, dtr, dte)
        return LR(best).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


def roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def concat_lgbm():
    def fp(dtr, ctr, dte, cte, seed):
        itr, ite = dtr.crow.values, dte.crow.values
        Bi, Bb = dino_block(itr, ite, K, seed)
        Xtr = np.hstack([Bi, dtr[NUM].values])
        Xte = np.hstack([Bb, dte[NUM].values])
        y = dtr.y.values
        m = LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=2,
                           min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                           random_state=seed, verbose=-1, n_jobs=2,
                           scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
        return m.fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


def _level1(dtr, dte, seed):
    """Image-LR and numeric-LR fitted on dtr, applied to dte. Returns (img_prob, num_prob)."""
    Bi, Bb = dino_block(dtr.crow.values, dte.crow.values, K, seed)
    y = dtr.y.values
    pi = LR(0.1).fit(Bi, y).predict_proba(Bb)[:, 1]
    pn = LR(0.1).fit(dtr[NUM].values, y).predict_proba(dte[NUM].values)[:, 1]
    return pi, pn


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack():
    def fp(dtr, ctr, dte, cte, seed):
        y, g = dtr.y.values, dtr.pid.values
        oof = np.zeros((len(dtr), 2))
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        for a, b in inner.split(dtr, y, g):
            da, db = dtr.iloc[a].reset_index(drop=True), dtr.iloc[b].reset_index(drop=True)
            pi, pn = _level1(da, db, seed)
            oof[b, 0], oof[b, 1] = pi, pn
        comb = LogisticRegression(C=1.0, max_iter=5000).fit(_logit(oof), y)
        pi, pn = _level1(dtr, dte, seed)
        return comb.predict_proba(_logit(np.column_stack([pi, pn])))[:, 1]
    return fp


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    evaluate('E09a_concat_lr', concat_lr(), 'DINOv2 PCA5/map + 55 numerics, L2 LR (C inner CV), all 944',
             df=coh, cnn=cnn)
    evaluate('E09b_concat_lgbm', concat_lgbm(), 'DINOv2 PCA5/map + 55 numerics, small LightGBM, all 944',
             df=coh, cnn=cnn)
    evaluate('E09c_stack', stack(), 'late fusion: image LR + numeric LR -> logistic combiner, all 944',
             df=coh, cnn=cnn)
