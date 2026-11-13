"""Final model freeze (user decision 2026-11-13): E28 blend = equal-weight mean of the percentile ranks of xgb,
lgbm_bag (5 seeds) and lr_en on numerics + fellow + age + R map measurements, refitted on all 944 scans (+ extra exams,
first-visit weight 2, hyper-parameters by the same inner CV). Two arms: noH (primary) and H (auxiliary).
A single new scan cannot be ranked within a test fold, so each member's score is mapped to its percentile among the
944 cohort scans' scores of the frozen member (in-sample reference), then the three percentiles are averaged.
Writes rethink/frozen/final.pkl (gitignored: fitted on patient data). Usage: python -I rethink/freeze_final.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import exp20_bilateral as e20  # noqa: E402
import exp22_bilateral as e22  # noqa: E402
from exp01_tabular import _inner_cv  # noqa: E402
from exp28_tricks import W_FIRST, learner  # noqa: E402
from harness import DATA  # noqa: E402

SEED = 2026
FROZEN = Path(__file__).parent / 'frozen'
MEMBERS = ('xgb', 'lgbm_bag', 'lr_en')


def matrix(fz, D, idx):
    X = e22.assemble(D, idx, 'N', 'R', fz['use_h'], fz['encfit'])[fz['cols']]
    return X.clip(fz['lo'], fz['hi'], axis=1).fillna(fz['med']).fillna(0).values


def member_scores(fz, X):
    return {k: np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0) for k, ms in fz['models'].items()}


def predict(fz, D, idx):
    """Final blend score in [0, 1] for visit rows idx."""
    s = member_scores(fz, matrix(fz, D, idx))
    return np.mean([np.searchsorted(fz['ref'][k], s[k], side='right') / len(fz['ref'][k]) for k in MEMBERS], axis=0)


def freeze(D, coh, use_h):
    rows = e22.train_rows(D, coh, True)
    w = np.where(D['first'][rows], W_FIRST, 1.0)
    encfit = e22.fit_enc_any(D, 'R', rows, SEED)
    X = e22.assemble(D, rows, 'N', 'R', use_h, encfit)
    lo, hi = X.quantile(0.005), X.quantile(0.995)
    fz = dict(name=f'E28_{"H" if use_h else "noH"}_blend', use_h=use_h, encfit=encfit, cols=list(X.columns),
              lo=lo, hi=hi, med=X.clip(lo, hi, axis=1).median(), models={})
    Xtr = matrix(fz, D, rows)
    y, g = D['V'].y.values[rows], D['V'].pid.values[rows]
    mk = e20._lgbm(SEED, y, fz['cols'])
    hp = _inner_cv(Xtr, y, g, mk, [dict(n_estimators=n, num_leaves=l) for n in [100, 300] for l in [2, 4]], SEED)
    fz['models']['lgbm_bag'] = [e20._lgbm(SEED + 97 * s, y, fz['cols'])(**hp).fit(Xtr, y, sample_weight=w)
                                for s in range(5)]
    for k in ('xgb', 'lr_en'):
        mk, grid, wkey = learner(k, SEED, y, fz['cols'])
        fz['models'][k] = [mk(**_inner_cv(Xtr, y, g, mk, grid, SEED)).fit(Xtr, y, **{wkey: w})]
    fz['ref'] = {k: np.sort(v) for k, v in member_scores(fz, matrix(fz, D, coh.crow.values)).items()}
    return fz


def main():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    D = e22.build(coh)
    out = {'noH': freeze(D, coh, False), 'H': freeze(D, coh, True)}
    FROZEN.mkdir(exist_ok=True)
    with open(FROZEN / 'final.pkl', 'wb') as f:
        pickle.dump(out, f)
    from sklearn.metrics import roc_auc_score
    for a, fz in out.items():
        p = predict(fz, D, coh.crow.values)
        print(fz['name'], 'apparent AUC (in-sample, not a result)', round(roc_auc_score(coh.y, p), 3), flush=True)


if __name__ == '__main__':
    sys.exit(main())
