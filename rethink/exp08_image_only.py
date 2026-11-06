"""E08: IMAGE-ONLY directions. No age, no numeric Pentacam features, no fellow-eye features.

Input = frozen DINOv2-small features of the four colour maps (extract_dino.py), per-map
standardise + PCA fitted in-fold, then L2 logistic regression. C and PCA size chosen by inner
patient-grouped CV on the training fold. Run on all 944 scans with outer CV (harness.evaluate).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from exp04_dino import dino_block  # noqa: E402
from harness import DATA, evaluate  # noqa: E402


def image_only_lr(k):
    def fp(dtr, ctr, dte, cte, seed):
        itr, ite = dtr.crow.values, dte.crow.values
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))

        def block(a, b):
            return dino_block(itr[a], itr[b], k, seed)

        best, best_s = 0.1, -1
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        for C in [0.01, 0.03, 0.1, 0.3, 1]:
            s = []
            for a, b in inner.split(dtr, y, g):
                Ba, Bb = dino_block(itr[a], itr[b], k, seed)
                m = mk(C).fit(Ba, y[a])
                s.append(roc_auc_score(y[b], m.predict_proba(Bb)[:, 1]))
            if np.mean(s) > best_s:
                best, best_s = C, np.mean(s)
        Btr, Bte = dino_block(itr, ite, k, seed)
        return mk(best).fit(Btr, y).predict_proba(Bte)[:, 1]
    return fp


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    for k in [5, 20]:
        evaluate(f'E08_dino_imageonly_pca{k}', image_only_lr(k),
                 f'DINOv2-small 4 maps, no age, PCA k={k} per map, L2 LR', df=coh, cnn=cnn)
