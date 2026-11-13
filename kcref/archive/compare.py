"""Paired cluster bootstrap of dAUC = AUC(model) - AUC(reference) on the dev set.

Usage: python -I rethink/compare.py MODEL [REFERENCE=E00_age_lr]
Reads the pooled OOF predictions saved by harness.evaluate(). Same patient clusters are resampled
for both models, so the CI reflects the paired difference, not two independent AUCs.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from harness import N_BOOT, RES, load  # noqa: E402


def paired_delta(y, p1, p2, groups, seed=0, n=N_BOOT):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    d = []
    for _ in range(n):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug))])
        if y[idx].min() != y[idx].max():
            d.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    d = np.array(d)
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5]), float(np.mean(d > 0))


if __name__ == '__main__':
    model = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else 'E00_age_lr'
    df, _ = load('dev')
    y, g = df.y.values, df.pid.values
    p1 = np.load(RES / 'oof' / f'{model}.npy').mean(0)
    p2 = np.load(RES / 'oof' / f'{ref}.npy').mean(0)
    m, (lo, hi), pgt = paired_delta(y, p1, p2, g)
    print(f'{model} - {ref}: dAUC {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(d>0)={pgt:.3f}')
