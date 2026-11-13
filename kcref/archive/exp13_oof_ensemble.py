"""E13: equal-weight rank average of E12b (fellow numerics: E02 LR + E06b LGBM) and E10c (spatial DINOv2 +
numerics + fellow-eye image difference). All 944 scans.

Both inputs come from harness.cv_oof on the same cohort with the same seeds, so the folds are identical
and per-repeat rank averaging of their OOF predictions is itself an out-of-fold ensemble (no weights
are fitted). Decided after seeing E12b and E10c, so it counts as a configuration and a post-hoc choice.
Also reports AUC within the has_fellow strata (the flag alone is a workflow confound, AUC 0.58).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from compare import paired_delta  # noqa: E402
from exp10_spatial import fellow_index  # noqa: E402
from harness import DATA, RES, _git_hash, summarise  # noqa: E402

NAME, PARTS = 'E13_E12b_E10c_rankavg', ['E12b_E02_E06b_avg', 'E10c_spatial_diff_lr']

if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    y, g = coh.y.values, coh.pid.values
    oofs = [np.load(RES / 'oof' / f'{n}.npy') for n in PARTS]
    ens = np.stack([np.mean([rankdata(o[r]) / len(y) for o in oofs], 0) for r in range(oofs[0].shape[0])])
    s = summarise(coh, ens)
    s.update(name=NAME, desc='equal-weight rank average of E12b and E10c OOF (same folds), all 944 scans',
             params='{"parts": "' + '+'.join(PARTS) + '"}', git=_git_hash(), secs=0,
             when=time.strftime('%Y-%m-%d %H:%M'))
    pd.DataFrame([s]).to_csv(RES / 'experiments.csv', mode='a', header=False, index=False)
    np.save(RES / 'oof' / f'{NAME}.npy', ens)
    print(f"{NAME} AUC {s['auc']:.3f} [{s['auc_lo']:.3f}-{s['auc_hi']:.3f}] rep {s['auc_rep_mean']:.3f} "
          f"first-visit {s['auc_first']:.3f}")

    p, age = ens.mean(0), np.load(RES / 'oof' / 'E07_age_all.npy').mean(0)
    d, (lo, hi), pr = paired_delta(y, p, age, g)
    print(f'vs age: dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P={pr:.2f}')
    has = fellow_index(coh) >= 0
    for nm, m in [('fellow', has), ('no fellow', ~has)]:
        d, (lo, hi), pr = paired_delta(y[m], p[m], age[m], g[m])
        print(f'{nm:9s} n={m.sum()} pos={y[m].sum()}  AUC {roc_auc_score(y[m], p[m]):.3f} vs age '
              f'{roc_auc_score(y[m], age[m]):.3f}  dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}]')
