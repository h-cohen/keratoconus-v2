"""E23: first-visit addendum, pre-declared in PROTOCOL.md (2026-11-10, after E22 results were seen).

Pool per run = the 21 E22 candidates refitted with first-visit scans weighted 2 (prefix w_ / wnoH_) + ftL, the
nested fine-tuned encoder L (ft_L.py cache; same first-visit weighting). Instead of picking one candidate per outer
fold, a non-negative logistic stacker is fitted on the 4-fold inner OOF predictions (logits) of the whole pool
inside the outer training fold (first-visit weight 2, L2 1e-2 on the weights), then applied to the pool refitted
on the whole outer training fold. Outer folds, inner folds and seeds are those of E22 (harness; exp16 select).
Runs: E23n_H (with block H), E23n_noH (without). Stacker weights -> results/stack23_weights.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import exp16_nested as e16  # noqa: E402
import exp22_bilateral as e22  # noqa: E402
from ft_L import CACHE, key  # noqa: E402
from harness import DATA, RES, evaluate  # noqa: E402

W_FIRST, L2 = 2.0, 1e-2
WEIGHTS = []


def ftL(dtr, dte, seed):
    p = CACHE / f'{key(seed, dtr.crow.values, dte.crow.values)}.npy'
    if not p.exists():
        raise FileNotFoundError(f'ftL cache missing for split {p.name}; run ft_L.py first')
    return np.load(p).astype(float)


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def fit_stacker(Z, y, w):
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    k = Z.shape[1]

    def f(t):
        s = Z @ t[:k] + t[k]
        ll = np.logaddexp(0, s) - y * s
        return (w * ll).sum() / w.sum() + L2 * (t[:k] ** 2).sum()
    t0 = np.r_[np.full(k, 1.0 / k), 0.0]
    r = minimize(f, t0, method='L-BFGS-B', bounds=[(0, None)] * k + [(None, None)])
    return r.x


def stack(pool, tag):
    def fp(dtr, ctr, dte, cte, seed):
        y, g = dtr.y.values, dtr.pid.values
        P = np.zeros((len(dtr), len(pool)))
        for a, b in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(dtr, y, g):
            da, db = dtr.iloc[a].reset_index(drop=True), dtr.iloc[b].reset_index(drop=True)
            for j, k in enumerate(pool):
                P[b, j] = e16.predict(k, da, db, seed)
        Z = logit(P)
        mu, sd = Z.mean(0), Z.std(0) + 1e-9
        t = fit_stacker(Z, y, np.where(dtr.first_visit.values, W_FIRST, 1.0))
        WEIGHTS.append(dict(run=tag, seed=seed, **{k: round(v, 4) for k, v in zip(pool, t[:-1])}))
        Zt = (logit(np.column_stack([e16.predict(k, dtr, dte, seed) for k in pool])) - mu) / sd
        return 1 / (1 + np.exp(-(Zt @ t[:-1] + t[-1])))
    return fp


if __name__ == '__main__':
    stage = sys.argv[1]  # H | noH
    exp = sys.argv[2] if len(sys.argv) > 2 else 'E23'  # E24 = every candidate trained with the extra exams
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    D = e22.build(coh)
    use_h = stage == 'H'
    pre = {'E23': ('w_', 'wnoH_'), 'E24': ('a_', 'anoH_'), 'E25': ('a_', 'anoH_'), 'E27': ('a_', 'anoH_')}[exp][0 if use_h else 1]
    c = e22.candidates(D, prefix=pre, use_h=use_h, w_first=W_FIRST, aug_all=exp in ('E24', 'E25', 'E27'),
                         extra_enc={'E25': ('R', 'probe', 'multi'), 'E27': ('R', 'probe', 'multi', 'G0', 'G')}.get(exp, ()))
    c['ftL'] = ftL
    e16.CANDS.update(c)
    evaluate(f'{exp}n_{stage}', stack(list(c), stage),
             f'{exp} (pre-declared'+{'E23': '', 'E24': ', all candidates + extra exams', 'E25': ', E24 pool + image candidates R / probe / multi', 'E27': ', E25 pool + position-aware SSL encoders G0 / G'}[exp]+'): in-fold non-negative logistic stack of 21 first-visit-weighted E22 candidates '
             f'{"with" if use_h else "without"} block H + nested fine-tuned L', df=coh, cnn=cnn)
    pd.DataFrame(WEIGHTS).to_csv(RES / f'stack{exp[1:]}_{stage}_weights.csv', index=False)
