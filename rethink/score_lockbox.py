"""ONE-SHOT lockbox scoring for the three candidates declared in PROTOCOL.md (2026-11-06).

Fits each candidate on the full dev set (713 scans, seed 2026), predicts the lockbox (231 scans),
and reports AUC with a patient-cluster bootstrap CI and the paired ΔAUC against age-only LR fitted
the same way. Refuses to run twice: the marker file records that the lockbox has been spent.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from compare import paired_delta  # noqa: E402
from exp02_fellow import fellow_lr  # noqa: E402
from exp06_combo import combo_lgbm, combo_lr  # noqa: E402
from harness import DATA, RES, ROOT, cluster_boot_auc  # noqa: E402

SHA = 'f875b78075576864c9b8acf25cbdd87d2700647dbfbd93105af37043dca04956'
MARK = RES / 'lockbox_spent.flag'
SEED = 2026


def age_lr():
    def fp(dtr, ctr, dte, cte, seed):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        return m.fit(dtr[['age']].values, dtr.y.values).predict_proba(dte[['age']].values)[:, 1]
    return fp


def main():
    if MARK.exists():
        sys.exit('lockbox already scored; refusing (see ' + str(MARK) + ')')
    assert hashlib.sha256((ROOT / 'lockbox_pids.csv').read_bytes()).hexdigest() == SHA, 'lockbox changed'
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    lock = set(pd.read_csv(ROOT / 'lockbox_pids.csv').pid)
    is_lock = coh.pid.isin(lock).values
    dev, lb = coh[~is_lock].reset_index(drop=True), coh[is_lock].reset_index(drop=True)
    cnn_dev, cnn_lb = cnn[~is_lock], cnn[is_lock]
    print(f'dev {len(dev)} scans / {dev.pid.nunique()} patients; lockbox {len(lb)} scans / {lb.pid.nunique()} patients, '
          f'{int(lb.y.sum())} positive scans')
    MARK.write_text('spent\n')  # written before scoring: a crash still counts as spent

    cands = {'E06b_fellow_mono_lgbm': combo_lgbm(), 'E06a_fellow_dino_lr': combo_lr(),
             'E02_fellow_lr': fellow_lr()}
    ref = age_lr()(dev, cnn_dev, lb, cnn_lb, SEED)
    y, g = lb.y.values, lb.pid.values
    rows = []
    for name, fp in cands.items():
        p = fp(dev, cnn_dev, lb, cnn_lb, SEED)
        auc = roc_auc_score(y, p)
        lo, hi = cluster_boot_auc(y, p, g)
        d, (dlo, dhi), pgt = paired_delta(y, p, ref, g)
        rows.append(dict(name=name, auc=auc, auc_lo=lo, auc_hi=hi, dauc=d, dauc_lo=dlo, dauc_hi=dhi, p_gt0=pgt))
        print(f'{name:24s} lockbox AUC {auc:.3f} [{lo:.3f}-{hi:.3f}]  dAUC vs age {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]')
    ref_auc = roc_auc_score(y, ref)
    print(f'age-only (reference)     lockbox AUC {ref_auc:.3f}')
    pd.DataFrame(rows).to_csv(RES / 'lockbox_scores.csv', index=False)


if __name__ == '__main__':
    main()
