"""E21: E20 with a LABEL-FREE fellow definition (pre-declared after the E20 leakage check).

E20 took recovered fellows from extra_exams.parquet, which keeps only pre-CXL exams. So a fellow eye already
cross-linked before the visit (273 of 316 'no fellow' scans) had no fellow, and has_fellow effectively meant
"the other eye is not yet treated" - known at the visit, but read from the clinical sheet, not the scan.
The image fellows of E02/E13 have the same property (post-CXL scans were excluded from the cohort).
Here the fellow is ANY same-day exam of the other eye in the raw export (raw_all_exams.parquet; no label,
CXL or keratoplasty field is read), preferring status OK, then fewest missing numerics. A treated fellow
contributes its own measured numerics. Training rows, features, candidates and folds are as in E20; the
label-dependent old fellow candidates (E02, E06b, E12b, E10c_fast, E13, b_E13) are not used.
Runs: E21_<candidate> for the eight E20 numeric candidates, E21n_nested = inner-CV selection over E16 pool A
(single eye) + the eight.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import exp16_nested as e16  # noqa: E402
import exp20_bilateral as e20  # noqa: E402
from compare import paired_delta  # noqa: E402
from harness import DATA, RES, evaluate  # noqa: E402

NUM = e20.NUM


def visits_clean(coh):
    """Cohort rows (index = crow), extra exams (training only), then raw exams that serve only as fellows."""
    v = e20.visits(coh).drop(columns=['fel', 'fel_src'])
    r = pd.read_parquet(DATA / 'raw_all_exams.parquet')
    r = r.sort_values(['status_ok', 'n_missing'], ascending=[False, True], kind='stable')
    r = r.drop_duplicates(['pid', 'eye', 'date'])
    r = r.assign(y=np.nan, src='fellow_only', cohort_patient=r.pid.isin(set(coh.pid)))[list(v.columns)]
    n = len(v)
    v = pd.concat([v, r], ignore_index=True)
    lut = {(p, e, d): n + i for i, (p, e, d) in enumerate(zip(r.pid, r.eye, r.date))}
    other = np.where(v.eye.values[:n] == 'right', 'left', 'right')
    fel = np.full(len(v), -1)
    fel[:n] = [lut.get((p, o, d), -1) for p, o, d in zip(v.pid.values[:n], other, v.date.values[:n])]
    v['fel'] = fel
    return v


NEW = {k: f for k, f in e20.NEW.items() if k != 'b_E13'}

if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    old_src = e20.visits(coh).fel_src.values[:len(coh)]  # E20 strata, for reporting only
    e20.V = visits_clean(coh)
    e20.FEAT = e20.features(e20.V)
    V = e20.V
    has = V.fel.values[:len(coh)] >= 0
    y, g = coh.y.values, coh.pid.values
    print(f'cohort scans with a label-free same-day fellow: {has.sum()} of {len(coh)}; flag alone AUC '
          f'{roc_auc_score(y, has):.3f}; augmentation rows with fellow: {int(((V.src == "extra") & (V.fel >= 0)).sum())}')
    e16.CANDS.update({f'c_{k}': f for k, f in NEW.items()})
    age = np.load(RES / 'oof' / 'E07_age_all.npy').mean(0)
    oofs = {}
    for k in NEW:
        _, oofs[f'E21_{k}'] = evaluate(f'E21_{k}', lambda dtr, ctr, dte, cte, seed, k=k: e16.predict(f'c_{k}', dtr, dte, seed),
                                       f'bilateral single visit, label-free fellows ({k})', df=coh, cnn=cnn)
    _, oofs['E21n_nested'] = evaluate('E21n_nested', e16.select(e16.POOL_A + [f'c_{k}' for k in NEW], 'A+clean'),
                                      'nested selection: E16 pool A + E21 label-free bilateral candidates, all 944',
                                      df=coh, cnn=cnn)
    ch = pd.DataFrame(e16.CHOICES)
    ch.to_csv(RES / 'nested21_choices.csv', index=False)
    print(ch.chosen.value_counts().to_string())
    print(f'{"run":28s} {"dAUC vs age [CI]":28s} ' + ' '.join(f'{s:>15s}' for s in ['image', 'raw', 'none']))
    for n, o in oofs.items():
        p = o.mean(0)
        d, (lo, hi), _ = paired_delta(y, p, age, g)
        strata = ' '.join(f'{roc_auc_score(y[old_src == s], p[old_src == s]):.3f} ({roc_auc_score(y[old_src == s], age[old_src == s]):.3f})'
                          for s in ['image', 'raw', 'none'])
        print(f'{n:28s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]   {strata}', flush=True)
