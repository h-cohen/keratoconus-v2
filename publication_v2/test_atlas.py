"""Atlas helpers (Appendix C) and plots.col_importance. Synthetic data only.
Run: python -I publication_v2/test_atlas.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import atlas as A  # noqa: E402
import plots as PL  # noqa: E402


def test_col_importance_uses_only_rows():
    # rows 0-1 selected: |SHAP| 1 and 3 -> mean 2; the unselected row (100) must not count
    sh = {'Zm': np.array([[1.0], [-3.0], [100.0]]), 'mask': np.ones((3, 1), bool)}
    got = PL.col_importance(sh, np.array([True, True, False]))
    assert got[0] == 2.0, got


def test_col_importance_skips_missing_map():
    sh = {'Zm': np.array([[2.0], [0.0]]), 'mask': np.array([[True], [False]])}
    assert PL.col_importance(sh, np.array([True, True]))[0] == 2.0


def test_crossing_interpolates_first_sign_change():
    tx = np.array([0.0, 1.0, 2.0, 3.0])
    ty = np.array([-1.0, -1.0, 1.0, -1.0])
    assert A.crossing(tx, ty) == 1.5
    assert np.isnan(A.crossing(tx, np.ones(4)))


def test_dirword():
    assert A.dirword(0.5) == 'higher → referral' and A.dirword(-0.5) == 'lower → referral'
    assert A.dirword(0.1) == 'non-monotone' and A.dirword(np.nan) == 'non-monotone'


if __name__ == '__main__':
    for t in [v for k, v in dict(globals()).items() if k.startswith('test_')]:
        t()
    print('ok')
