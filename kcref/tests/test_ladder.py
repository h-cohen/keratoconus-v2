"""Column subsets of the feature-block ladder (E29). Run: python -I kcref/tests/test_ladder.py"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
from model import select_columns  # noqa: E402
from tests_cols import columns  # noqa: E402


def test_counts():
    for use_h, extra in ((False, 0), (True, 5)):
        cols = columns(use_h)
        assert len(cols) == 495 + extra, len(cols)
        for blocks, n in (('L1', 66), ('L2', 205), ('L2b', 163), ('final', 495)):
            got = select_columns(cols, blocks)
            assert len(got) == n + extra, (blocks, use_h, len(got))
            it = iter(cols)
            assert all(c in it for c in got), 'order not preserved'


def test_l1_is_own_eye_only():
    got = select_columns(columns(True), 'L1')
    assert not any(c.startswith(('fel_', 'asym_', 'worse_', 'R_')) or c in ('has_fellow', 'own_image') for c in got)
    assert 'age' in got and 'fellow_cxl_before_visit' in got


if __name__ == '__main__':
    test_counts()
    test_l1_is_own_eye_only()
    print('ok')
