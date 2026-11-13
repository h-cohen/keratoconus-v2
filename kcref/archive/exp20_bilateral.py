"""E20: bilateral single-visit model, squeezed (pre-declared). All 944 scans, harness folds, paired vs age.

New information over E02/E06/E13:
  1. Recovered fellows. 584 cohort scans have no same-day fellow IMAGE, but for 252 of them the other eye's
     same-day exam is in the raw export (extra_exams.parquet: pre-CXL, complete numerics, any status).
     Same visit, numerics only, so still single-visit. Bilateral coverage 360 -> 612 of 944 scans.
     Fellow choice: the cohort scan if there is one, else a status-OK exam, else the first exam that day.
  2. Full feature set. E02 used 12 engineered indices per eye; 'full' uses age + all 54 numerics + the 12
     indices for the own eye, the fellow, own - fellow, and worse-eye indicators (own steeper KMax / K2 /
     ISV / KI / IHD / IVA / prog_max, own thinner Pachy Min).
  3. Bilateral training augmentation: extra exams (with their own same-day fellows) of patients outside
     the cohort and of training-fold cohort patients are added to the training rows only.
Fellow lookup never uses labels; training rows never include test-fold patients.
Candidates (LR: L2, C by patient-grouped inner CV; LGBM: depth 2, age monotone decreasing, 4-config inner CV):
  b_eng_lr, b_eng_lgbm, b_full_lr, b_full_lgbm      cohort training rows
  b_eng_aug_lr, b_eng_aug_lgbm, b_full_aug_lr, b_full_aug_lgbm   + augmentation rows
  b_E13 : (mean(b_eng_aug_lr, b_eng_aug_lgbm) + E10c_fast) / 2   (E13 recipe with recovered fellows)
Runs: E20_<candidate> for each of the nine, E20n_nested = inner-CV selection over E16 pool B + the nine.
Reported within strata: image fellow (360), recovered fellow (252), no fellow (332).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import exp16_nested as e16  # noqa: E402
from compare import paired_delta  # noqa: E402
from exp01_tabular import _inner_cv, engineer  # noqa: E402
from exp10_spatial import fellow_index  # noqa: E402
from harness import DATA, RES, evaluate  # noqa: E402

NUM = e16.NUM  # age + 54
WORSE = {'KMax Sagittal Front (D)': 1, 'K2 F (D):': 1, 'ISV:': 1, 'KI:': 1, 'IHD:': 1, 'IVA:': 1, 'prog_max': 1,
         'Pachy Min:': -1}


def visits(coh):
    """Cohort rows (index = crow) followed by extra exams; fellow row index (or -1) and fellow source."""
    ex = pd.read_parquet(DATA / 'extra_exams.parquet')
    c = coh[['pid', 'eye', 'y'] + NUM].copy()
    c['date'], c['status_ok'], c['src'] = pd.to_datetime(coh.scan_date), True, 'cohort'
    e = ex[['pid', 'eye', 'y', 'date', 'status_ok'] + NUM].copy()
    e['src'] = 'extra'
    v = pd.concat([c, e], ignore_index=True)
    v['cohort_patient'] = v.pid.isin(set(coh.pid))
    fi = fellow_index(coh)
    fel = np.full(len(v), -1)
    fel[:len(coh)] = fi
    rank = (v.src != 'cohort') * 2 + (~v.status_ok.astype(bool)) * 1  # prefer cohort, then status OK
    order = v.assign(rank=rank).sort_values('rank', kind='stable')
    first = order.groupby(['pid', 'eye', 'date']).head(1)
    lut = {(p, e_, d): i for i, p, e_, d in zip(first.index, first.pid, first.eye, first.date)}
    for i in np.flatnonzero(fel < 0):
        other = 'left' if v.eye.iat[i] == 'right' else 'right'
        fel[i] = lut.get((v.pid.iat[i], other, v.date.iat[i]), -1)
    v['fel'] = fel
    image = np.zeros(len(v), bool)
    image[:len(coh)] = fi >= 0
    v['fel_src'] = np.where(fel < 0, 'none', np.where(image, 'image', 'raw'))
    return v


def features(v):
    """Label-free feature matrices for every visit row: 'eng' (E02-style) and 'full'. NaN where undefined."""
    eng = engineer(v.assign(age=v.age)).replace([np.inf, -np.inf], np.nan)
    own_full = pd.concat([v[NUM], eng.drop(columns='age')], axis=1)
    has = (v.fel.values >= 0)
    fidx = np.maximum(v.fel.values, 0)
    out = {}
    for name, own in [('eng', eng), ('full', own_full)]:
        cols = [c for c in own.columns if c != 'age']
        fel = own[cols].iloc[fidx].reset_index(drop=True).where(np.repeat(has[:, None], len(cols), 1))
        fel.columns = ['fel_' + c for c in cols]
        asym = pd.DataFrame(own[cols].values - fel.values, columns=['asym_' + c for c in cols])
        parts = [own.reset_index(drop=True), fel, asym, pd.DataFrame({'has_fellow': has.astype(float)})]
        if name == 'full':
            parts.append(pd.DataFrame({f'worse_{c}': np.where(has, (s * (own[c].values - own[c].values[fidx]) > 0)
                                                               .astype(float), np.nan) for c, s in WORSE.items()}))
        out[name] = pd.concat(parts, axis=1)
    return out


def _lgbm(seed, y, cols):
    mono = [-1 if c == 'age' else 0 for c in cols]
    return lambda n_estimators, num_leaves: LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.03, num_leaves=num_leaves, max_depth=2, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, monotone_constraints=mono, random_state=seed,
        verbose=-1, n_jobs=4, scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))


def make(fs, aug, kind):
    def f(dtr, dte, seed):
        X = FEAT[fs]
        rows = list(dtr.crow.values)
        if aug:
            ok = (V.src == 'extra') & (~V.cohort_patient | V.pid.isin(set(dtr.pid)))
            rows += list(np.flatnonzero(ok.values))
        rows = np.array(rows)
        Xtr, Xte = X.iloc[rows], X.iloc[dte.crow.values]
        med = Xtr.median()
        Xtr, Xte = Xtr.fillna(med).fillna(0).values, Xte.fillna(med).fillna(0).values
        y, g = V.y.values[rows], V.pid.values[rows]
        if kind == 'lr':
            C = _inner_cv(Xtr, y, g, e16.LR, e16.CGRID, seed)['C']
            return e16.LR(C).fit(Xtr, y).predict_proba(Xte)[:, 1]
        mk = _lgbm(seed, y, list(X.columns))
        hp = _inner_cv(Xtr, y, g, mk, [dict(n_estimators=n, num_leaves=l) for n in [100, 300] for l in [2, 4]], seed)
        return mk(**hp).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return f


NEW = {f'b_{fs}{"_aug" if aug else ""}_{k}': make(fs, aug, k)
       for aug in [False, True] for fs in ['eng', 'full'] for k in ['lr', 'lgbm']}


def b_e13(dtr, dte, seed):
    p = lambda k: e16.predict(k, dtr, dte, seed)  # noqa: E731
    return ((p('b_eng_aug_lr') + p('b_eng_aug_lgbm')) / 2 + p('E10c_fast')) / 2


NEW['b_E13'] = b_e13

if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    V = visits(coh)
    FEAT = features(V)
    src = V.fel_src.values[:len(coh)]
    print('cohort fellow source:', pd.Series(src).value_counts().to_dict(),
          '| augmentation rows:', int((V.src == 'extra').sum()), 'with fellow:', int(((V.src == 'extra') & (V.fel >= 0)).sum()))
    e16.CANDS.update(NEW)
    y, g = coh.y.values, coh.pid.values
    age = np.load(RES / 'oof' / 'E07_age_all.npy').mean(0)
    oofs = {}
    for k in NEW:
        _, oofs[f'E20_{k}'] = evaluate(f'E20_{k}', lambda dtr, ctr, dte, cte, seed, k=k: e16.predict(k, dtr, dte, seed),
                                       f'bilateral single visit, recovered fellows ({k})', df=coh, cnn=cnn)
    _, oofs['E20n_nested'] = evaluate('E20n_nested', e16.select(e16.POOL_B + list(NEW), 'B+bilat'),
                                      'nested selection: E16 pool B + E20 bilateral candidates, all 944', df=coh, cnn=cnn)
    ch = pd.DataFrame(e16.CHOICES)
    ch.to_csv(RES / 'nested20_choices.csv', index=False)
    print(ch.chosen.value_counts().to_string())
    print(f'{"run":28s} {"dAUC vs age [CI]":28s} ' + ' '.join(f'{s:>15s}' for s in ['image', 'raw', 'none']))
    for n, o in oofs.items():
        p = o.mean(0)
        d, (lo, hi), _ = paired_delta(y, p, age, g)
        strata = ' '.join(f'{roc_auc_score(y[src == s], p[src == s]):.3f} ({roc_auc_score(y[src == s], age[src == s]):.3f})'
                          for s in ['image', 'raw', 'none'])
        print(f'{n:28s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]   {strata}', flush=True)
