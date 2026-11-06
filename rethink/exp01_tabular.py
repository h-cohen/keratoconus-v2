"""E01: tabular done right. Nonlinear age, monotone trees, engineered KC indices.

Everything (scaling, C / depth tuning by inner patient-grouped CV) is fitted inside the training fold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from harness import evaluate, numeric_features  # noqa: E402

F = numeric_features()


def engineer(d):
    """KC indices from the Pentacam numerics. Pure functions of one scan."""
    o = pd.DataFrame(index=d.index)
    o['age'] = d.age
    o['kmax_minus_km'] = d['KMax Sagittal Front (D)'] - d['Km F (D):']
    o['pachy_min_over_apex'] = d['Pachy Min:'] / d['Pachy Apex:']
    o['pachy_min_over_pupil'] = d['Pachy Min:'] / d['Pachy Pupil:']
    o['pachy_min_per_age'] = d['Pachy Min:'] / d.age
    prog = [c for c in F if c.endswith('Prog') and c.startswith('D')]
    o['prog_mean'] = d[prog].mean(axis=1)
    o['prog_max'] = d[prog].max(axis=1)
    o['pachy_drop_4mm'] = d['D0mm Pachy'] - d['D4mm Pachy']
    o['ecc_front_back'] = d['Num. Ecc. F:'] - d['Num. Ecc. B:']
    o['asph_ratio'] = d['Asph. Q F:'] / d['Asph. Q B:'].replace(0, np.nan)
    o['elev_ratio'] = d['Rf F (mm):'] / d['Rf B (mm):']
    o['cvol_ratio'] = d['C.Vol D 3mm:'] / d['C.Vol D 10mm:']
    o = o.replace([np.inf, -np.inf], np.nan)
    return o.fillna(o.median())


def _inner_cv(X, y, g, make, grid, seed):
    """Pick the best grid point by mean inner-fold AUC (patient-grouped, inside the training fold)."""
    from sklearn.metrics import roc_auc_score
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    best, best_s = None, -1
    for hp in grid:
        s = []
        for a, b in inner.split(X, y, g):
            m = make(**hp).fit(X[a], y[a])
            s.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
        if np.mean(s) > best_s:
            best, best_s = hp, np.mean(s)
    return best


def spline_age_lr():
    """Age as a cubic spline (4 knots), LR. Tuned C by inner CV."""
    def fp(dtr, ctr, dte, cte, seed):
        mk = lambda C: make_pipeline(SplineTransformer(n_knots=4, degree=3), StandardScaler(),
                                     LogisticRegression(C=C, max_iter=5000))
        X, Xt = dtr[['age']].values, dte[['age']].values
        C = _inner_cv(X, dtr.y.values, dtr.pid.values, mk, [dict(C=c) for c in [0.01, 0.1, 1, 10]], seed)['C']
        return mk(C).fit(X, dtr.y.values).predict_proba(Xt)[:, 1]
    return fp


def eng_lr(cols=None):
    """Engineered indices + age, L2 LR, C tuned by inner CV."""
    def fp(dtr, ctr, dte, cte, seed):
        Xtr, Xte = engineer(dtr), engineer(dte)
        cols_ = list(Xtr.columns) if cols is None else cols
        Xtr, Xte = Xtr[cols_].values, Xte[cols_].values
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        C = _inner_cv(Xtr, y, g, mk, [dict(C=c) for c in [0.01, 0.03, 0.1, 0.3, 1]], seed)['C']
        return mk(C).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


def eng_mono_lgbm():
    """Engineered indices + age, LightGBM depth-2 with age monotone decreasing, tuned by inner CV."""
    def fp(dtr, ctr, dte, cte, seed):
        Xtr, Xte = engineer(dtr), engineer(dte)
        cols = list(Xtr.columns)
        mono = [-1 if c == 'age' else 0 for c in cols]
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda n_estimators, learning_rate, num_leaves: LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
            max_depth=2, min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, monotone_constraints=mono, random_state=seed, verbose=-1,
            n_jobs=2, scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
        grid = [dict(n_estimators=n, learning_rate=0.03, num_leaves=l)
                for n in [100, 300] for l in [2, 4]]
        hp = _inner_cv(Xtr.values, y, g, mk, grid, seed)
        return mk(**hp).fit(Xtr.values, y).predict_proba(Xte.values)[:, 1]
    return fp


RUNS = {
    'E01_age_spline_lr': (spline_age_lr, 'age as cubic spline, LR, C by inner CV'),
    'E01_eng_lr': (eng_lr, 'engineered KC indices + age, L2 LR, C by inner CV'),
    'E01_eng_mono_lgbm': (eng_mono_lgbm, 'engineered indices + age, LightGBM depth 2, age monotone(-), inner CV'),
}

if __name__ == '__main__':
    # usage: exp01_tabular.py [RUN ...]; each run is logged once, so pass only the ones not yet logged
    for name in (sys.argv[1:] or RUNS):
        make, desc = RUNS[name]
        evaluate(name, make(), desc)
