"""E28 — learner / regularisation variants of the final non-stack model bi_R (numerics + fellow + age + R map
measurements), pre-declared in PROTOCOL.md 2026-11-12. Same input rows as E25_bi_R_lgbm (extra exams, first-visit
weight 2); every hyper-parameter chosen by patient-grouped inner CV inside the training fold; seeds fixed by harness.
Variants (KIND):
  xgb       XGBoost, grid depth {2,3} x trees {200,500}, eta 0.03, subsample 0.8, colsample 0.5
  lgbm_deep LightGBM, leaves {4,8,15} x trees {300,800}, eta 0.02, colsample 0.3, min_child 30, lambda 1
  lgbm_bag  the E25 LightGBM recipe, averaged over 5 seeds (tuned once)
  et        ExtraTrees 500, min_samples_leaf {5,20} x max_features {0.1, sqrt}
  lr_en     elastic-net logistic, C {0.01,0.03,0.1,0.3} x l1_ratio {0.2,0.5,0.8}
  svm       RBF SVM (Platt), C {0.1,0.3,1} x gamma {scale/4, scale}
  blend     fixed equal-weight mean of ranks of xgb, lgbm_bag, lr_en (no fitted weights)
Added 2026-11-13 (input-side variants, E25 LightGBM recipe unless stated):
  lgbm_nan   missing values left as NaN (LightGBM routes them natively) instead of median imputation
  lgbm_nofel R block without the fellow copy (own + own-fellow only)
  pca_lr     StandardScaler -> PCA(k) -> L2 logistic, k {20,40,80} x C {0.01,0.1,1}
Arms: noH (primary) and H (auxiliary). Run names E28_{noH,H}_{KIND}.
Usage: python -I rethink/exp28_tricks.py {noH|H} KIND [KIND ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
import exp20_bilateral as e20  # noqa: E402
import exp22_bilateral as e22  # noqa: E402
from exp01_tabular import _inner_cv  # noqa: E402
from harness import DATA, evaluate  # noqa: E402

W_FIRST = 2.0


def spw(y):
    return (y == 0).sum() / max((y == 1).sum(), 1)


def learner(kind, seed, y, cols):
    """(make(**hp) -> estimator, grid, sample-weight fit kwarg name or None)."""
    mono = [-1 if c == 'age' else 0 for c in cols]
    if kind == 'xgb':
        mk = lambda max_depth, n_estimators: XGBClassifier(  # noqa: E731
            max_depth=max_depth, n_estimators=n_estimators, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
            min_child_weight=5, reg_lambda=1.0, scale_pos_weight=spw(y), random_state=seed, n_jobs=4,
            tree_method='hist', verbosity=0)
        return mk, [dict(max_depth=d, n_estimators=n) for d in [2, 3] for n in [200, 500]], 'sample_weight'
    if kind == 'lgbm_deep':
        mk = lambda num_leaves, n_estimators: LGBMClassifier(  # noqa: E731
            num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=0.02, colsample_bytree=0.3,
            subsample=0.8, subsample_freq=1, min_child_samples=30, reg_lambda=1.0, monotone_constraints=mono,
            scale_pos_weight=spw(y), random_state=seed, n_jobs=4, verbose=-1)
        return mk, [dict(num_leaves=l, n_estimators=n) for l in [4, 8, 15] for n in [300, 800]], 'sample_weight'
    if kind == 'et':
        mk = lambda min_samples_leaf, max_features: ExtraTreesClassifier(  # noqa: E731
            n_estimators=500, min_samples_leaf=min_samples_leaf, max_features=max_features,
            class_weight='balanced', random_state=seed, n_jobs=4)
        return mk, [dict(min_samples_leaf=m, max_features=f) for m in [5, 20] for f in [0.1, 'sqrt']], 'sample_weight'
    if kind == 'lr_en':
        mk = lambda C, l1_ratio: make_pipeline(StandardScaler(), LogisticRegression(  # noqa: E731
            C=C, penalty='elasticnet', l1_ratio=l1_ratio, solver='saga', max_iter=5000, class_weight='balanced'))
        return mk, [dict(C=c, l1_ratio=r) for c in [0.01, 0.03, 0.1, 0.3] for r in [0.2, 0.5, 0.8]], \
            'logisticregression__sample_weight'
    if kind == 'svm':
        mk = lambda C, gm: make_pipeline(StandardScaler(), SVC(  # noqa: E731
            C=C, gamma=gm / len(cols), probability=True, class_weight='balanced', random_state=seed))
        return mk, [dict(C=c, gm=g) for c in [0.1, 0.3, 1] for g in [0.25, 1.0]], 'svc__sample_weight'
    if kind == 'pca_lr':
        from sklearn.decomposition import PCA
        mk = lambda k, C: make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed),  # noqa: E731
                                        LogisticRegression(C=C, max_iter=5000, class_weight='balanced'))
        return mk, [dict(k=k, C=c) for k in [20, 40, 80] for c in [0.01, 0.1, 1]], 'logisticregression__sample_weight'
    raise ValueError(kind)


def design(D, rows, use_h, seed, impute=True, fellow=True):
    encfit = e22.fit_enc_any(D, 'R', rows, seed)
    X = e22.assemble(D, rows, 'N', 'R', use_h, encfit)
    if not fellow:
        X = X.loc[:, ~X.columns.str.startswith('R_fel')]
    lo, hi = X.quantile(0.005), X.quantile(0.995)
    med = X.clip(lo, hi, axis=1).median()

    def prep(idx):
        Z = e22.assemble(D, idx, 'N', 'R', use_h, encfit)[X.columns].clip(lo, hi, axis=1)
        return (Z.fillna(med).fillna(0) if impute else Z).values
    return prep, list(X.columns)


def fit_predict(D, kind, use_h, dtr, dte, seed):
    rows = e22.train_rows(D, dtr, True)
    w = np.where(D['first'][rows], W_FIRST, 1.0)
    prep, cols = design(D, rows, use_h, seed, impute=kind != 'lgbm_nan', fellow=kind != 'lgbm_nofel')
    Xtr, Xte = prep(rows), prep(dte.crow.values)
    y, g = D['V'].y.values[rows], D['V'].pid.values[rows]
    if kind == 'blend':
        return np.mean([rankdata(fit_predict(D, k, use_h, dtr, dte, seed)) / len(dte)
                        for k in ('xgb', 'lgbm_bag', 'lr_en')], axis=0)
    if kind in ('lgbm_nan', 'lgbm_nofel'):
        mk = e20._lgbm(seed, y, cols)
        hp = _inner_cv(Xtr, y, g, mk, [dict(n_estimators=n, num_leaves=l) for n in [100, 300] for l in [2, 4]], seed)
        return mk(**hp).fit(Xtr, y, sample_weight=w).predict_proba(Xte)[:, 1]
    if kind == 'lgbm_bag':
        mk = e20._lgbm(seed, y, cols)
        hp = _inner_cv(Xtr, y, g, mk, [dict(n_estimators=n, num_leaves=l) for n in [100, 300] for l in [2, 4]], seed)
        return np.mean([e20._lgbm(seed + 97 * s, y, cols)(**hp).fit(Xtr, y, sample_weight=w).predict_proba(Xte)[:, 1]
                        for s in range(5)], axis=0)
    mk, grid, wkey = learner(kind, seed, y, cols)
    hp = _inner_cv(Xtr, y, g, mk, grid, seed)
    return mk(**hp).fit(Xtr, y, **{wkey: w}).predict_proba(Xte)[:, 1]


if __name__ == '__main__':
    arm, kinds = sys.argv[1], sys.argv[2:]
    assert arm in ('noH', 'H') and kinds
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    D = e22.build(coh)
    for k in kinds:
        evaluate(f'E28_{arm}_{k}', lambda dtr, ctr, dte, cte, seed, k=k: fit_predict(D, k, arm == 'H', dtr, dte, seed),
                 f'E28 learner variant {k} on bi_R (numerics + fellow + age + R), {arm}, extra exams, first-visit w 2',
                 params=dict(kind=k, arm=arm), df=coh, cnn=cnn)
