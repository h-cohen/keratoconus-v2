"""E11: visit + history from the FULL raw exam record (prepare_history.py). Reported separately from
the single-visit claim (PROTOCOL.md). All 944 cohort scans are eligible; outer CV as harness.evaluate.

Pre-declared before any E11 result was seen (2026-11-06):
  History subset H: cohort scans whose same eye has >= 1 raw exam on an earlier date, the earliest
    of them >= 91 days before the index scan, within a 3-year look-back. Any QC status.
  Features (Global Consensus progression criteria, no tuning on outcomes):
    age; current KMax and Pachy Min; Theil-Sen per-year slopes (prior exams + index scan) of
    KMax, K2 front, K2 back, Pachy Min; years since the most recent prior exam.
  Forbidden and NOT used: number of prior exams, total follow-up span, any date or later exam.
  E11a  L2 LR (C by inner CV)                  on H, vs age-only LR on H (paired bootstrap)
  E11b  LightGBM, monotone: age(-) KMax/K2 slopes(+) pachy slope(-), small fixed trees, on H
  E11c  E11a + the same slopes for the FELLOW eye (its prior exams before the index date), on H
History availability itself is never a feature: scans with history are positive 10.8% vs 24.8%
without, so it would act as a follow-up proxy.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import theilslopes
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from compare import paired_delta  # noqa: E402
from exp01_tabular import _inner_cv  # noqa: E402
from harness import DATA, RES, evaluate  # noqa: E402

SLOPE = {'KMax Sagittal Front (D)': 'kmax', 'K2 F (D):': 'k2f', 'K2 B (D):': 'k2b', 'Pachy Min:': 'pmin'}
LOOKBACK, MIN_GAP = pd.Timedelta(days=3 * 365), pd.Timedelta(days=91)


def history_features(coh, ex):
    """One row per cohort scan. Uses only exams dated strictly before the index scan."""
    grp = {k: g for k, g in ex.groupby(['pid', 'eye'])}
    other = {'right': 'left', 'left': 'right'}
    rows = []
    for _, r in coh.iterrows():
        rec = {}
        for side, eye in [('', r.eye), ('fel_', other[r.eye])]:
            g = grp.get((r.pid, eye))
            prev = g[(g.date < r.scan_date) & (g.date >= r.scan_date - LOOKBACK)] if g is not None else None
            ok = prev is not None and len(prev) > 0 and prev.date.min() <= r.scan_date - MIN_GAP
            if side == '':
                rec['in_H'] = ok
                rec['yrs_since_last'] = (r.scan_date - prev.date.max()).days / 365.25 if ok else np.nan
            for col, nm in SLOPE.items():
                rec[f'{side}slope_{nm}'] = np.nan
                if not ok:
                    continue
                t = (prev.date - r.scan_date).dt.days.values / 365.25
                v = prev[col].values.astype(float)
                if side == '':  # the index scan is the last point of its own eye's series
                    t, v = np.append(t, 0.0), np.append(v, float(r[col]))
                m = np.isfinite(v)
                if m.sum() >= 2 and np.ptp(t[m]) > 0:
                    rec[f'{side}slope_{nm}'] = theilslopes(v[m], t[m])[0]
        rows.append(rec)
    return pd.DataFrame(rows)


OWN = ['age', 'KMax Sagittal Front (D)', 'Pachy Min:', 'slope_kmax', 'slope_k2f', 'slope_k2b', 'slope_pmin',
       'yrs_since_last']
FEL = ['fel_slope_kmax', 'fel_slope_k2f', 'fel_slope_k2b', 'fel_slope_pmin']


def lr_on(cols):
    def fp(dtr, ctr, dte, cte, seed):
        med = dtr[cols].median()
        Xtr, Xte = dtr[cols].fillna(med).values, dte[cols].fillna(med).values
        y, g = dtr.y.values, dtr.pid.values
        mk = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))  # noqa: E731
        C = _inner_cv(Xtr, y, g, mk, [dict(C=c) for c in [0.01, 0.03, 0.1, 0.3, 1]], seed)['C']
        return mk(C).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


def mono_lgbm(cols):
    sign = {'age': -1, 'slope_kmax': 1, 'slope_k2f': 1, 'slope_k2b': 1, 'slope_pmin': -1}

    def fp(dtr, ctr, dte, cte, seed):
        y = dtr.y.values
        m = LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=4, max_depth=2, min_child_samples=15,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
                           n_jobs=4, monotone_constraints=[sign.get(c, 0) for c in cols],
                           scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
        return m.fit(dtr[cols].values, y).predict_proba(dte[cols].values)[:, 1]
    return fp


def age_lr():
    return lr_on(['age'])


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    ex = pd.read_parquet(DATA / 'raw_exams.parquet')
    h = history_features(coh, ex)
    d = pd.concat([coh.reset_index(drop=True), h], axis=1)
    H = d[d.in_H].reset_index(drop=True)
    cnn = np.zeros((len(H), 1))
    print(f'H: {len(H)} scans, {H.eid.nunique()} eyes, {H.pid.nunique()} patients, {int(H.y.sum())} positive scans; '
          f'fellow slopes available for {int(H.fel_slope_kmax.notna().sum())}')
    _, o_ref = evaluate('E11_age_lr_H', age_lr(), 'age only, on history subset H', df=H, cnn=cnn)
    _, o_a = evaluate('E11a_hist_lr', lr_on(OWN), 'age + current KMax/PachyMin + Theil-Sen slopes, L2 LR, on H',
                      df=H, cnn=cnn)
    _, o_b = evaluate('E11b_hist_mono_lgbm', mono_lgbm([c for c in OWN]), 'same features, monotone LightGBM, on H',
                      df=H, cnn=cnn)
    _, o_c = evaluate('E11c_hist_fellow_lr', lr_on(OWN + FEL), 'E11a + fellow-eye slopes, L2 LR, on H', df=H, cnn=cnn)
    y, g = H.y.values, H.pid.values
    for n, o in [('E11a_hist_lr', o_a), ('E11b_hist_mono_lgbm', o_b), ('E11c_hist_fellow_lr', o_c)]:
        m, (lo, hi), pg = paired_delta(y, o.mean(0), o_ref.mean(0), g)
        print(f'{n} - age(H): dAUC {m:+.3f} [{lo:+.3f}, {hi:+.3f}] P={pg:.3f}')
    H.to_parquet(DATA / 'history_subset.parquet', index=False)
