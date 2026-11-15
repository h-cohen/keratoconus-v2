"""Age-group and operating-characteristic helpers (E30). Synthetic data only.
Run: python -I publication_v2/test_age_ops.py"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import age_ops as AO  # noqa: E402


def test_age_group_cuts():
    a = np.array([15, 21.9, 22, 26.9, 27, 33.9, 34, 69])
    assert AO.age_group(a).tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


def test_wilson():
    lo, hi = AO.wilson(0, 10)
    assert lo == 0 and 0.25 < hi < 0.35
    lo, hi = AO.wilson(5, 10)
    assert lo < 0.5 < hi and abs((0.5 - lo) - (hi - 0.5)) < 1e-9


def test_within_group_auc_removes_group_signal():
    # score = group only: crude AUC high, within-group AUC exactly 0.5 (all ties)
    grp = np.repeat([1, 2], 100)
    y = np.r_[np.ones(60), np.zeros(40), np.ones(10), np.zeros(90)].astype(int)
    p = (grp == 1).astype(float)
    assert roc_auc_score(y, p) > 0.7
    assert AO.within_group_auc(y, p, grp) == 0.5


def test_within_group_auc_weights_pairs():
    # group 1: perfect (1 pos x 2 neg = 2 pairs); group 2: inverted (2 pos x 2 neg = 4 pairs) -> 2/6
    y = np.array([1, 0, 0, 1, 1, 0, 0])
    grp = np.array([1, 1, 1, 2, 2, 2, 2])
    p = np.array([0.9, 0.1, 0.2, 0.1, 0.2, 0.8, 0.9])
    assert abs(AO.within_group_auc(y, p, grp) - 2 / 6) < 1e-12
    # a one-class group contributes nothing
    assert abs(AO.within_group_auc(np.r_[y, 0, 0], np.r_[p, 5, 6], np.r_[grp, 3, 3]) - 2 / 6) < 1e-12


def test_threshold_at_sensitivity():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.75, 0.5, 0.3, 0.1, 0.05, 0.0])
    r = AO.operating_point(y, p, 0.75)
    assert r['threshold'] == 0.7 and r['sens'] == 0.75
    assert r['spec'] == 5 / 6 and r['flagged'] == 4 and r['ppv'] == 3 / 4 and r['npv'] == 5 / 6


if __name__ == '__main__':
    for t in [v for k, v in dict(globals()).items() if k.startswith('test_')]:
        t()
    print('ok')
