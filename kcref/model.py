"""The final model: an equal-weight rank blend of XGBoost, a 5-seed LightGBM bag and an elastic-net logistic
regression on N + R (+ H in the H arm), without the 11 derived indices (E31 adoption, FINAL_DROP). See
publication_v2/report.html for the full description.

Per training fold: training rows = fold cohort scans + extra exams (features.train_rows); first-visit scans
weight 2; R scaler, 0.5/99.5% clipping and median imputation fitted on the training rows; each member's
hyper-parameters chosen by 4-fold patient-grouped inner CV (mean AUC), then refitted with the sample weights.
Blend in CV: mean of the members' within-test-fold ranks. Frozen model: mean of the members' percentiles among
their scores on the 944 cohort scans (same ordering for a batch, defined for a single new scan).

Usage (repo root):
  python -I kcref/model.py cv {noH|H}     5 x 5 CV, logged as final_{arm}
  python -I kcref/model.py cv {noH|H} --blocks {L1|L2|L2b}   ladder rung on the final feature set, logged as
                                                             E31_ladder_{blocks}_{arm} (E29 rungs before E31)
  python -I kcref/model.py cv {noH|H} --drop {noeng|noagepachy}   derived-index ablation (E31), logged as E31_{drop}_{arm}
  python -I kcref/model.py cv {noH|H} --drop none   all columns (the E28 blend), logged as E28_{arm}_blend
  python -I kcref/model.py freeze         refit both arms on all scans -> kcref/frozen/final.pkl (patient data)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
import features as F  # noqa: E402
from common import DATA, ROOT  # noqa: E402
from evaluate import evaluate  # noqa: E402

W_FIRST = 2.0
SEED = 2026
MEMBERS = ('xgb', 'lgbm_bag', 'lr_en')
ENG = ('kmax_minus_km', 'pachy_min_over_apex', 'pachy_min_over_pupil', 'pachy_min_per_age', 'prog_mean', 'prog_max',
       'pachy_drop_4mm', 'ecc_front_back', 'asph_ratio', 'elev_ratio', 'cvol_ratio')  # features.engineer outputs
FINAL_DROP = 'noeng'  # E31 adoption (2026-11-16, pre-declared rule): the final model has no derived indices


def spw(y):
    return (y == 0).sum() / max((y == 1).sum(), 1)


def learner(kind, seed, y, cols):
    """(make(**hp) -> estimator, hyper-parameter grid, name of the sample-weight fit argument)."""
    if kind == 'xgb':
        mk = lambda max_depth, n_estimators: XGBClassifier(  # noqa: E731
            max_depth=max_depth, n_estimators=n_estimators, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
            min_child_weight=5, reg_lambda=1.0, scale_pos_weight=spw(y), random_state=seed, n_jobs=4,
            tree_method='hist', verbosity=0)
        return mk, [dict(max_depth=d, n_estimators=n) for d in [2, 3] for n in [200, 500]], 'sample_weight'
    if kind == 'lgbm':
        mono = [-1 if c == 'age' else 0 for c in cols]  # referral decreases with age
        mk = lambda n_estimators, num_leaves: LGBMClassifier(  # noqa: E731
            n_estimators=n_estimators, learning_rate=0.03, num_leaves=num_leaves, max_depth=2, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, monotone_constraints=mono, random_state=seed,
            verbose=-1, n_jobs=4, scale_pos_weight=spw(y))
        return mk, [dict(n_estimators=n, num_leaves=l) for n in [100, 300] for l in [2, 4]], 'sample_weight'
    if kind == 'lr_en':
        mk = lambda C, l1_ratio: make_pipeline(StandardScaler(), LogisticRegression(  # noqa: E731
            C=C, penalty='elasticnet', l1_ratio=l1_ratio, solver='saga', max_iter=5000, class_weight='balanced'))
        return mk, [dict(C=c, l1_ratio=r) for c in [0.01, 0.03, 0.1, 0.3] for r in [0.2, 0.5, 0.8]], \
            'logisticregression__sample_weight'
    raise ValueError(kind)


def inner_cv(X, y, g, make, grid, seed):
    """Grid point with the best mean AUC over a 4-fold patient-grouped split of the training rows."""
    best, best_s = None, -1
    for hp in grid:
        s = [roc_auc_score(y[b], make(**hp).fit(X[a], y[a]).predict_proba(X[b])[:, 1])
             for a, b in StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(X, y, g)]
        if np.mean(s) > best_s:
            best, best_s = hp, np.mean(s)
    return best


def select_columns(cols, blocks):
    """Columns of a feature-block ladder rung (E29); H columns are kept whenever present.
    L1 own eye (age + numerics + indices); L2 block N; L2b L1 + own-eye maps; final everything."""
    fellow = ('fel_', 'asym_', 'worse_')

    def keep(c):
        if blocks == 'final' or c in F.H_COLS:
            return True
        if blocks == 'L2':
            return not c.startswith('R_') and c not in ('has_fellow_image', 'own_image')
        own = not c.startswith(fellow + ('R_',)) and c not in ('has_fellow', 'has_fellow_image', 'own_image')
        if blocks == 'L1':
            return own
        if blocks == 'L2b':
            return own or c.startswith('R_own') or c == 'own_image'
        raise ValueError(blocks)
    return [c for c in cols if keep(c)]


def drop_columns(cols, drop):
    """E31 ablation: 'noeng' removes the 11 engineered indices (own, fel_, asym_, worse_ forms); 'noagepachy' removes
    thinnest pachymetry / age (own, fel_, asym_); None keeps all."""
    if drop is None:
        return list(cols)
    names = {'noeng': ENG, 'noagepachy': ('pachy_min_per_age',)}[drop]
    gone = {p + n for n in names for p in ('', 'fel_', 'asym_', 'worse_')}
    return [c for c in cols if c not in gone]


def matrix(fz, D, idx):
    X = F.assemble(D, idx, fz['use_h'], fz['encfit'])[fz['cols']]
    return X.clip(fz['lo'], fz['hi'], axis=1).fillna(fz['med']).fillna(0).values


def fit(D, rows, use_h, seed, blocks='final', drop=FINAL_DROP):
    """Fit preprocessing and the three members on visit rows `rows` (columns of ladder rung `blocks`)."""
    w = np.where(D['first'][rows], W_FIRST, 1.0)
    sc = F.fit_R(D, rows)
    X = F.assemble(D, rows, use_h, sc)
    X = X[drop_columns(select_columns(list(X.columns), blocks), drop)]
    lo, hi = X.quantile(0.005), X.quantile(0.995)
    fz = dict(use_h=use_h, encfit=sc, cols=list(X.columns), lo=lo, hi=hi, med=X.clip(lo, hi, axis=1).median(),
              models={})
    Xtr = matrix(fz, D, rows)
    y, g = D['V'].y.values[rows], D['V'].pid.values[rows]
    for k in MEMBERS:
        mk, grid, wkey = learner('lgbm' if k == 'lgbm_bag' else k, seed, y, fz['cols'])
        hp = inner_cv(Xtr, y, g, mk, grid, seed)
        seeds = [seed + 97 * s for s in range(5)] if k == 'lgbm_bag' else [seed]
        fz['models'][k] = [learner('lgbm' if k == 'lgbm_bag' else k, s, y, fz['cols'])[0](**hp)
                           .fit(Xtr, y, **{wkey: w}) for s in seeds]
    return fz


def member_scores(fz, X):
    return {k: np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0) for k, ms in fz['models'].items()}


def cv_fit_predict(D, use_h, blocks='final', drop=FINAL_DROP):
    """fit_predict for evaluate.cv_oof: blend score = mean of the members' within-test-fold ranks."""
    def fp(dtr, dte, seed):
        fz = fit(D, F.train_rows(D, dtr), use_h, seed, blocks, drop)
        s = member_scores(fz, matrix(fz, D, dte.crow.values))
        return np.mean([rankdata(s[k]) / len(dte) for k in MEMBERS], axis=0)
    return fp


def predict(fz, D, idx):
    """Frozen-model score in [0, 1] for visit rows idx (mean member percentile vs the cohort reference)."""
    s = member_scores(fz, matrix(fz, D, idx))
    return np.mean([np.searchsorted(fz['ref'][k], s[k], side='right') / len(fz['ref'][k]) for k in MEMBERS], axis=0)


def cohort():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    return coh


def main():
    coh = cohort()
    D = F.build(coh)
    if sys.argv[1] == 'cv':
        arm = sys.argv[2]
        blocks = sys.argv[sys.argv.index('--blocks') + 1] if '--blocks' in sys.argv else 'final'
        drop = sys.argv[sys.argv.index('--drop') + 1] if '--drop' in sys.argv else None
        h = ' + H' if arm == 'H' else ''
        if drop == 'none':
            evaluate(f'E28_{arm}_blend', cv_fit_predict(D, arm == 'H', drop=None), coh,
                     f'E28 blend on all N + R{h} columns (re-run 2026-11-16 after the working-copy loss)',
                     dict(arm=arm, drop='none'))
        elif drop is not None:
            evaluate(f'E31_{drop}_{arm}', cv_fit_predict(D, arm == 'H', drop=drop), coh,
                     f'E31 ablation {drop}: final blend on N + R{h} without the dropped derived indices',
                     dict(arm=arm, drop=drop))
        elif blocks == 'final':
            evaluate(f'final_{arm}', cv_fit_predict(D, arm == 'H'), coh,
                     f'final model (rank blend xgb + lgbm_bag + lr_en on N + R{h})', dict(arm=arm))
        else:
            evaluate(f'E31_ladder_{blocks}_{arm}', cv_fit_predict(D, arm == 'H', blocks), coh,
                     f'ladder rung {blocks}{h}: final blend on the rung columns, no derived indices',
                     dict(arm=arm, blocks=blocks, drop=FINAL_DROP))
    elif sys.argv[1] == 'freeze':
        out = {}
        for arm in ('noH', 'H'):
            fz = fit(D, F.train_rows(D, coh), arm == 'H', SEED)
            fz['name'] = f'final_{arm}'
            fz['ref'] = {k: np.sort(v) for k, v in member_scores(fz, matrix(fz, D, coh.crow.values)).items()}
            out[arm] = fz
            print(fz['name'], 'apparent (in-sample) AUC', round(roc_auc_score(coh.y, predict(fz, D, coh.crow.values)), 3))
        (ROOT / 'frozen').mkdir(exist_ok=True)
        with open(ROOT / 'frozen' / 'final.pkl', 'wb') as f:
            pickle.dump(out, f)


if __name__ == '__main__':
    sys.exit(main())
