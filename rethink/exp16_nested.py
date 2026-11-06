"""E16: fully nested model selection on ALL 944 scans (no lockbox). Pre-declared candidate pools.

Outer: harness.cv_oof folds (5-fold StratifiedGroupKFold by patient x 5 repeats, seeds 2026-2030), identical
to every other all-data run, so paired comparisons with E07_age_all are on the same folds.
Inner: inside each outer training fold, 4-fold patient-grouped CV scores every candidate in the pool;
the best mean inner AUC is refit on the whole outer training fold and predicts the outer test fold.
Each candidate tunes its own hyper-parameters by a further inner CV inside its fit. Nothing is chosen
by looking at outer-fold results, which removes the post-hoc selection of E13.

Pool A, single eye (the submission's setting: the scanned eye's numerics + maps + age):
  age_lr, num_lr (55 numerics, L2 LR), num_lgbm, img_num_lr ([DINOv2 4x4 PCA10/map | numerics], LR),
  img_num_lgbm ([DINOv2 | colour grid | numerics], LightGBM)
Pool B, pool A plus the same-day fellow eye:
  E02 fellow LR, E06b fellow monotone LightGBM, E12b (E02+E06b quantile average, recalibrated),
  E10c_fast ([DINOv2 | numerics | fellow PCA-score difference | has_fellow], LR), E13 (mean of E12b and
  E10c_fast probabilities). Image PCA is fitted once per training set (unsupervised) and only C is
  tuned inside, unlike E10c, which refit the PCA in every inner fold.
Choices per outer fold -> rethink/results/nested16_choices.csv.
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
from compare import paired_delta  # noqa: E402
from exp01_tabular import _inner_cv  # noqa: E402
from exp02_fellow import fellow_lr  # noqa: E402
from exp06_combo import combo_lgbm  # noqa: E402
from exp10_spatial import MAPS, _color, _grid, fellow_index  # noqa: E402
from exp12_ensemble import avg_recal  # noqa: E402
from harness import DATA, RES, evaluate, numeric_features  # noqa: E402

NUM = numeric_features()
LR = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))  # noqa: E731
CGRID = [dict(C=c) for c in [0.01, 0.03, 0.1, 0.3, 1]]


def _lgbm(y, seed):
    return LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=2, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=seed, verbose=-1,
                          n_jobs=4, scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))


def _pca(Xfit, Xapp, seed, k=10):
    sc = StandardScaler().fit(Xfit)
    p = PCA(n_components=k, random_state=seed).fit(sc.transform(Xfit))
    return p.transform(sc.transform(Xfit)), p.transform(sc.transform(Xapp))


def img_blocks(dtr, dte, seed, use):
    """[S | C | num | D] for train and test; PCA fitted on the training rows only."""
    G, COL = _grid(), _color()
    rtr, rte = dtr.crow.values, dte.crow.values
    S = [_pca(G[rtr, j * 6144:(j + 1) * 6144], G[rte, j * 6144:(j + 1) * 6144], seed) for j in range(len(MAPS))]
    Str, Ste = np.hstack([s[0] for s in S]), np.hstack([s[1] for s in S])
    parts_tr, parts_te = [Str], [Ste]
    if 'C' in use:
        a, b = _pca(COL[rtr], COL[rte], seed)
        parts_tr.append(a), parts_te.append(b)
    parts_tr.append(dtr[NUM].values), parts_te.append(dte[NUM].values)
    if 'D' in use:
        for S_, d, out in [(Str, dtr, parts_tr), (Ste, dte, parts_te)]:
            fel = fellow_index(d)
            has = fel >= 0
            out += [np.where(has[:, None], S_ - S_[np.maximum(fel, 0)], 0.0), has[:, None].astype(float)]
    return np.hstack(parts_tr), np.hstack(parts_te)


def lr_on(Xtr, Xte, dtr, seed):
    y, g = dtr.y.values, dtr.pid.values
    C = _inner_cv(Xtr, y, g, LR, CGRID, seed)['C']
    return LR(C).fit(Xtr, y).predict_proba(Xte)[:, 1]


def age_lr(dtr, dte, seed):
    return LogisticRegression(max_iter=5000).fit(dtr[['age']].values, dtr.y.values).predict_proba(dte[['age']].values)[:, 1]


def num_lr(dtr, dte, seed):
    return lr_on(dtr[NUM].values, dte[NUM].values, dtr, seed)


def num_lgbm(dtr, dte, seed):
    return _lgbm(dtr.y.values, seed).fit(dtr[NUM].values, dtr.y.values).predict_proba(dte[NUM].values)[:, 1]


def img_num_lr(dtr, dte, seed):
    return lr_on(*img_blocks(dtr, dte, seed, ''), dtr, seed)


def img_num_lgbm(dtr, dte, seed):
    Xtr, Xte = img_blocks(dtr, dte, seed, 'C')
    return _lgbm(dtr.y.values, seed).fit(Xtr, dtr.y.values).predict_proba(Xte)[:, 1]


def e10c_fast(dtr, dte, seed):
    return lr_on(*img_blocks(dtr, dte, seed, 'D'), dtr, seed)


_E02, _E06b, _E12b = fellow_lr(), combo_lgbm(), avg_recal()
CANDS = {
    'age_lr': age_lr, 'num_lr': num_lr, 'num_lgbm': num_lgbm, 'img_num_lr': img_num_lr, 'img_num_lgbm': img_num_lgbm,
    'E02': lambda a, b, s: _E02(a, None, b, None, s), 'E06b': lambda a, b, s: _E06b(a, None, b, None, s),
    'E12b': lambda a, b, s: _E12b(a, None, b, None, s), 'E10c_fast': e10c_fast,
}
POOL_A = ['age_lr', 'num_lr', 'num_lgbm', 'img_num_lr', 'img_num_lgbm']
POOL_B = POOL_A + ['E02', 'E06b', 'E12b', 'E10c_fast', 'E13']
_CACHE, CHOICES = {}, []


def predict(name, dtr, dte, seed):
    key = (name, seed, hash(dtr.crow.values.tobytes()), hash(dte.crow.values.tobytes()))
    if key not in _CACHE:
        if name == 'E13':
            _CACHE[key] = (predict('E12b', dtr, dte, seed) + predict('E10c_fast', dtr, dte, seed)) / 2
        else:
            _CACHE[key] = CANDS[name](dtr, dte, seed)
    return _CACHE[key]


def select(pool, tag):
    def fp(dtr, ctr, dte, cte, seed):
        y, g = dtr.y.values, dtr.pid.values
        scores = {k: [] for k in pool}
        for a, b in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(dtr, y, g):
            da, db = dtr.iloc[a].reset_index(drop=True), dtr.iloc[b].reset_index(drop=True)
            for k in pool:
                scores[k].append(roc_auc_score(y[b], predict(k, da, db, seed)))
        means = {k: float(np.mean(v)) for k, v in scores.items()}
        best = max(means, key=means.get)
        CHOICES.append(dict(pool=tag, seed=seed, chosen=best, **{f'inner_{k}': round(v, 4) for k, v in means.items()}))
        print(tag, seed, best, round(means[best], 3), flush=True)
        return predict(best, dtr, dte, seed)
    return fp


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    _, oa = evaluate('E16a_nested_single_eye', select(POOL_A, 'A'),
                     'nested selection, single-eye pool (age, numerics LR/LGBM, images+numerics LR/LGBM), all 944',
                     df=coh, cnn=cnn)
    _, ob = evaluate('E16b_nested_with_fellow', select(POOL_B, 'B'),
                     'nested selection, pool A + same-day fellow models (E02, E06b, E12b, E10c_fast, E13), all 944',
                     df=coh, cnn=cnn)
    ch = pd.DataFrame(CHOICES)
    ch.to_csv(RES / 'nested16_choices.csv', index=False)
    print(ch.groupby('pool').chosen.value_counts().to_string())
    y, g = coh.y.values, coh.pid.values
    age = np.load(RES / 'oof' / 'E07_age_all.npy').mean(0)
    has = fellow_index(coh) >= 0
    for n, o in [('E16a', oa), ('E16b', ob)]:
        p = o.mean(0)
        d, (lo, hi), pr = paired_delta(y, p, age, g)
        print(f'{n} vs age: dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P={pr:.2f} | bilateral {roc_auc_score(y[has], p[has]):.3f} '
              f'(age {roc_auc_score(y[has], age[has]):.3f}) | single-eye visits {roc_auc_score(y[~has], p[~has]):.3f} '
              f'(age {roc_auc_score(y[~has], age[~has]):.3f})')
