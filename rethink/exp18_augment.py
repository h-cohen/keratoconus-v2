"""E18: single-eye, single-visit model trained on MORE labelled exams (pre-declared). All 944 scans.

Training augmentation from rethink/data/extra_exams.parquet (prepare_extra.py; 1,487 labelled pre-CXL
exams of CXL-clinic patients that are not cohort scans: 587 of 269 patients outside the cohort, 900 of
cohort patients). Inside every fit, the training rows are the cohort training rows plus
  - all extra exams of patients outside the cohort, and
  - extra exams of cohort patients who are in the training rows (never those of test-fold patients).
Evaluation is unchanged: cohort scans only, the harness folds (5x5, seeds 2026-2030), so paired
comparisons with E07_age_all are on the same folds. Features: age + 54 numerics (the extras have no images).
Candidates (LR: L2, C by patient-grouped inner CV on the augmented rows; LightGBM: E16's fixed small model):
  aug_ok_lr / aug_ok_lgbm   : extras with status OK only (558 exams; all cohort scans are status OK)
  aug_all_lr / aug_all_lgbm : extras of any status (1,487 exams)
  aug_age_lr                : age only, all extras (control: does more data only sharpen the age curve?)
Runs: E18a-d each numeric candidate alone, E18_age_aug the control, E18e nested inner-CV selection over
E16's pool A + the five candidates (single eye).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
import exp16_nested as e16  # noqa: E402
from compare import paired_delta  # noqa: E402
from harness import DATA, RES, evaluate  # noqa: E402

NUM = e16.NUM
EXTRA = pd.read_parquet(DATA / 'extra_exams.parquet')


def augment(dtr, status_ok_only):
    e = EXTRA[EXTRA.status_ok] if status_ok_only else EXTRA
    e = e[~e.cohort_patient | e.pid.isin(set(dtr.pid))]
    return pd.concat([dtr[['pid', 'y'] + NUM], e[['pid', 'y'] + NUM]], ignore_index=True)


def aug_lr(ok):
    def f(dtr, dte, seed):
        a = augment(dtr, ok)
        return e16.lr_on(a[NUM].values, dte[NUM].values, a, seed)
    return f


def aug_lgbm(ok):
    def f(dtr, dte, seed):
        a = augment(dtr, ok)
        return e16._lgbm(a.y.values, seed).fit(a[NUM].values, a.y.values).predict_proba(dte[NUM].values)[:, 1]
    return f


def aug_age_lr(dtr, dte, seed):
    a = augment(dtr, False)
    return LogisticRegression(max_iter=5000).fit(a[['age']].values, a.y.values).predict_proba(dte[['age']].values)[:, 1]


NEW = {'aug_ok_lr': aug_lr(True), 'aug_ok_lgbm': aug_lgbm(True), 'aug_all_lr': aug_lr(False),
       'aug_all_lgbm': aug_lgbm(False), 'aug_age_lr': aug_age_lr}

if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    e16.CANDS.update(NEW)
    oofs = {}
    runs = [('E18a_aug_ok_lr', 'aug_ok_lr'), ('E18b_aug_ok_lgbm', 'aug_ok_lgbm'), ('E18c_aug_all_lr', 'aug_all_lr'),
            ('E18d_aug_all_lgbm', 'aug_all_lgbm'), ('E18_age_aug', 'aug_age_lr')]
    for run, k in runs:
        _, oofs[run] = evaluate(run, lambda dtr, ctr, dte, cte, seed, k=k: e16.predict(k, dtr, dte, seed),
                                f'single eye, numerics, trained with extra labelled exams ({k})', df=coh, cnn=cnn)
        print(run, 'done', flush=True)
    _, oofs['E18e'] = evaluate('E18e_nested_single_eye_aug', e16.select(e16.POOL_A + list(NEW), 'A+aug'),
                               'nested selection: E16 pool A + augmented-training candidates, single eye, all 944',
                               df=coh, cnn=cnn)
    ch = pd.DataFrame(e16.CHOICES)
    ch.to_csv(RES / 'nested18_choices.csv', index=False)
    print(ch.chosen.value_counts().to_string())
    y, g = coh.y.values, coh.pid.values
    age = np.load(RES / 'oof' / 'E07_age_all.npy').mean(0)
    for n, o in oofs.items():
        d, (lo, hi), pr = paired_delta(y, o.mean(0), age, g)
        print(f'{n} vs age: dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P={pr:.2f}')
