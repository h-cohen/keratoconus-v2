"""E31 derived-index ablation: AUC of each variant and paired difference vs the final blend (E28), both arms.
Writes kcref/results/E31_compare.csv and prints it. Decision rule (PROTOCOL.md, E31): 'equivalent' if the first-visit
difference is >= -0.005 and its 95% CI includes 0. Run: python -I kcref/e31_compare.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, ROOT  # noqa: E402
from evaluate import cluster_boot_auc, paired_delta  # noqa: E402


def oof(name):
    p = np.load(ROOT / 'results' / 'oof' / f'{name}.npy')
    return p.mean(0) if p.ndim > 1 else p


def main():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    y, g, fv = coh.y.values, coh.pid.values, coh.first_visit.values
    rows = []
    for arm in ('noH', 'H'):
        ref = oof(f'E28_{arm}_blend')
        for drop in ('noeng', 'noagepachy'):
            p = oof(f'E31_{drop}_{arm}')
            r = {'arm': arm, 'variant': drop}
            for scope, m in (('first', fv), ('all', np.ones(len(y), bool))):
                lo, hi = cluster_boot_auc(y[m], p[m], g[m])
                d, (dlo, dhi), pp = paired_delta(y[m], p[m], ref[m], g[m])
                r.update({f'auc_{scope}': roc_auc_score(y[m], p[m]), f'lo_{scope}': lo, f'hi_{scope}': hi,
                          f'final_{scope}': roc_auc_score(y[m], ref[m]),
                          f'd_{scope}': roc_auc_score(y[m], p[m]) - roc_auc_score(y[m], ref[m]),
                          f'dlo_{scope}': dlo, f'dhi_{scope}': dhi, f'p_gt0_{scope}': pp})
            r['equivalent'] = bool(r['d_first'] >= -0.005 and r['dlo_first'] <= 0 <= r['dhi_first'])
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / 'results' / 'E31_compare.csv', index=False)
    print(t.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
