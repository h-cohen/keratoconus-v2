"""Reviewer analyses on the all-data OOF predictions of E07_E06b_all vs E07_age_all (no refitting).

  1. AUC within age strata (<25, 25-29, >=30) and by Pentacam TKC stage (no Belin ABCD in the export).
  2. Operating point: Youden threshold chosen on one half of the patients' OOF, applied to the other half
     (and swapped), so sens/spec/PPV/NPV are not tuned on the scans they are reported on.
  3. Calibration: intercept and slope of logit(OOF), Brier score.
  4. Decision curve: net benefit at thresholds 0.10-0.40 vs treat-all.
  5. Time to CXL: AUC of each positive time bin against all negatives (descriptive).
  6. Secondary endpoint, objective progression: KMax rise >= 1 D at a later exam 6-15 months after the
     index scan (raw exam record, before any CXL of that eye). Eyes without such a follow-up are excluded.
Writes rethink/results/analysis_e07.txt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from exp05_raw import TKC  # noqa: E402
from harness import DATA, ROOT, cluster_boot_auc  # noqa: E402

OOF = ROOT / 'results' / 'oof'
MODELS = {'E06b': 'E07_E06b_all', 'age': 'E07_age_all'}


def auc_ci(y, p, g):
    if y.sum() < 3 or y.sum() == len(y):
        return 'n/a'
    lo, hi = cluster_boot_auc(y, p, g)
    return f'{roc_auc_score(y, p):.3f} [{lo:.3f}-{hi:.3f}]'


def subgroup(lines, name, mask, y, g, P):
    m = np.asarray(mask)
    lines.append(f'  {name:14s} n={m.sum():4d} pos={int(y[m].sum()):3d}  ' +
                 '  '.join(f'{k} {auc_ci(y[m], p[m], g[m])}' for k, p in P.items()))


def operating_point(y, p, g, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    half = np.isin(g, rng.choice(ug, len(ug) // 2, replace=False))
    pred = np.zeros(len(y), bool)
    for fit, app in [(half, ~half), (~half, half)]:
        ts = np.unique(p[fit])
        j = [(p[fit] >= t)[y[fit] == 1].mean() - (p[fit] >= t)[y[fit] == 0].mean() for t in ts]
        pred[app] = p[app] >= ts[int(np.argmax(j))]
    tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
    fn, tn = (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    return dict(sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1))


def calibration(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    slope = LogisticRegression(C=1e6, max_iter=5000).fit(lp[:, None], y)
    # calibration-in-the-large: intercept with the logit as an offset, by Newton on one parameter
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp)))
        a += (y - q).sum() / max((q * (1 - q)).sum(), 1e-9)
    return dict(citl=a, slope=float(slope.coef_[0, 0]), brier=float(np.mean((p - y) ** 2)), mean_p=p.mean())


def net_benefit(y, p, t):
    n = len(y)
    tp, fp = ((p >= t) & (y == 1)).sum(), ((p >= t) & (y == 0)).sum()
    return tp / n - fp / n * t / (1 - t)


def progression_label(coh):
    """1/0 objective progression for scans with a qualifying follow-up exam, NaN otherwise."""
    ex = pd.read_parquet(DATA / 'raw_exams.parquet')
    ex = ex[ex['KMax Sagittal Front (D)'].notna()]
    eye_map = {'OD': 'right', 'OS': 'left', 'R': 'right', 'L': 'left'}
    ex_by = {k: v for k, v in ex.groupby(['pid', 'eye'])}
    lab = np.full(len(coh), np.nan)
    kmax = coh['KMax Sagittal Front (D)'].values
    for i, r in enumerate(coh.itertuples()):
        e = ex_by.get((r.pid, eye_map.get(r.eye, r.eye)))
        if e is None:
            continue
        t0 = r.scan_date
        stop = t0 + pd.DateOffset(months=int(r.months_to_cxl)) if pd.notna(r.months_to_cxl) else pd.Timestamp.max
        fu = e[(e.date >= t0 + pd.DateOffset(months=6)) & (e.date <= t0 + pd.DateOffset(months=15)) & (e.date < stop)]
        if len(fu):
            lab[i] = float((fu['KMax Sagittal Front (D)'] - kmax[i] >= 1.0).any())
    return lab


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    y, g = coh.y.values, coh.pid.values
    P = {k: np.load(OOF / f'{v}.npy').mean(0) for k, v in MODELS.items()}
    L = [f'OOF analyses, all {len(coh)} scans ({int(y.sum())} positive), models: {MODELS}']

    L.append('1a. age strata')
    for name, m in [('<25', coh.age < 25), ('25-29', (coh.age >= 25) & (coh.age < 30)), ('>=30', coh.age >= 30)]:
        subgroup(L, name, m, y, g, P)
    L.append('1b. TKC stage (Pentacam topographic KC grade)')
    tkc = pd.read_parquet(DATA / 'raw_extra.parquet')['TKC:'].astype(str).map(TKC).values
    for name, m in [('none/poss.', ~(tkc >= 1)), ('KC1-1.5', (tkc >= 1) & (tkc < 2)),
                    ('KC2-2.5', (tkc >= 2) & (tkc < 3)), ('KC3-4', tkc >= 3)]:
        subgroup(L, name, m, y, g, P)

    L.append('2. operating point (Youden threshold from the other half of patients)')
    for k, p in P.items():
        L.append(f'  {k:5s} ' + '  '.join(f'{a} {b:.2f}' for a, b in operating_point(y, p, g).items()))
    L.append('3. calibration')
    for k, p in P.items():
        L.append(f'  {k:5s} ' + '  '.join(f'{a} {b:.3f}' for a, b in calibration(y, p).items()))
    L.append('4. net benefit (treat-all in brackets)')
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        L.append(f'  t={t:.2f} ' + '  '.join(f'{k} {net_benefit(y, p, t):+.3f}' for k, p in P.items()) +
                 f'  [all {net_benefit(y, np.ones_like(y, float), t):+.3f}]')

    L.append('5. positives by months to CXL vs all negatives')
    neg = y == 0
    for name, lo, hi in [('0-6', 0, 6), ('6-12', 6, 12), ('12-24', 12, 24), ('>24', 24, 1e9)]:
        m = neg | ((y == 1) & (coh.months_to_cxl >= lo) & (coh.months_to_cxl < hi)).values
        subgroup(L, name, m, y, g, P)

    L.append('6. secondary endpoint: KMax rise >= 1 D at an exam 6-15 months later (before CXL)')
    prog = progression_label(coh)
    has = ~np.isnan(prog)
    yp = np.nan_to_num(prog).astype(int)
    L.append(f'  scans with qualifying follow-up: {has.sum()}, progressed: {int(yp[has].sum())}, '
             f'overlap with CXL label: {int((yp[has] & y[has]).sum())}')
    subgroup(L, 'progression', has, yp, g, P)
    text = '\n'.join(L)
    print(text)
    (ROOT / 'results' / 'analysis_e07.txt').write_text(text + '\n')
