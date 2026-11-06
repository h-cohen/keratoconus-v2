"""E07: nested CV on ALL 944 scans (dev + lockbox). The lockbox is no longer held out.

Outer: 5-fold StratifiedGroupKFold by patient, repeated 5x (seeds 2026-2030), as harness.evaluate.
Inner (inside each outer training fold): 4-fold patient-grouped CV scores each candidate; the best
mean inner AUC wins and is refit on the whole outer training fold. Each candidate's own
hyper-parameters are tuned by a further inner CV inside its fit.

Also run on the same outer folds: E06b alone and age-only LR, for paired comparisons.
Choices per outer fold are written to rethink/results/nested_choices.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from compare import paired_delta  # noqa: E402
from exp02_fellow import fellow_lr  # noqa: E402
from exp06_combo import combo_lgbm, combo_lr  # noqa: E402
from harness import DATA, RES, evaluate  # noqa: E402
from score_lockbox import age_lr  # noqa: E402

CANDS = {'age_lr': age_lr(), 'E02_fellow_lr': fellow_lr(), 'E06a_fellow_dino_lr': combo_lr(),
         'E06b_fellow_mono_lgbm': combo_lgbm()}
CHOICES = []


def select(cands):
    def fp(dtr, ctr, dte, cte, seed):
        y, g = dtr.y.values, dtr.pid.values
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        scores = {k: [] for k in cands}
        for a, b in inner.split(dtr, y, g):
            da, db = dtr.iloc[a].reset_index(drop=True), dtr.iloc[b].reset_index(drop=True)
            for k, f in cands.items():
                p = f(da, ctr[a], db, ctr[b], seed)
                scores[k].append(roc_auc_score(y[b], p))
        means = {k: float(np.mean(v)) for k, v in scores.items()}
        best = max(means, key=means.get)
        CHOICES.append(dict(seed=seed, chosen=best, **{f'inner_{k}': round(v, 4) for k, v in means.items()}))
        return cands[best](dtr, ctr, dte, cte, seed)
    return fp


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    evaluate('E07_age_all', age_lr(), 'age only, all 944 scans, outer CV', df=coh, cnn=cnn)
    evaluate('E07_E06b_all', combo_lgbm(), 'E06b alone, all 944 scans, outer CV', df=coh, cnn=cnn)
    s, oof = evaluate('E07_nested_select_all', select(CANDS),
                      'nested selection among age/E02/E06a/E06b, all 944 scans', df=coh, cnn=cnn)
    pd.DataFrame(CHOICES).to_csv(RES / 'nested_choices.csv', index=False)
    y, g = coh.y.values, coh.pid.values
    for ref in ['E07_age_all', 'E07_E06b_all']:
        r = np.load(RES / 'oof' / f'{ref}.npy').mean(0)
        m, (lo, hi), pgt = paired_delta(y, oof.mean(0), r, g)
        print(f'E07_nested_select_all - {ref}: dAUC {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(d>0)={pgt:.3f}')
    print(pd.Series([c['chosen'] for c in CHOICES]).value_counts().to_string())
