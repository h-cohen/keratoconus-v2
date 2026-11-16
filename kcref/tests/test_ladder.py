"""Column subsets of the feature-block ladder (E29). Run: python -I kcref/tests/test_ladder.py"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
from model import FINAL_DROP, drop_columns, select_columns  # noqa: E402
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


def test_e31_drops():
    for use_h in (False, True):
        cols = columns(use_h)
        eng = drop_columns(cols, 'noeng')
        assert len(cols) - len(eng) == 34, len(cols) - len(eng)
        assert not any('per_age' in c or 'ratio' in c or 'kmax_minus_km' in c or 'prog_max' in c for c in eng)
        assert 'Pachy Min:' in eng and 'fel_D4mm Prog' in eng and 'age' in eng
        ap = drop_columns(cols, 'noagepachy')
        assert sorted(set(cols) - set(ap)) == ['asym_pachy_min_per_age', 'fel_pachy_min_per_age', 'pachy_min_per_age']
        assert drop_columns(cols, None) == cols



def test_final_feature_set_is_noeng():
    # E31 adoption (2026-11-16): the final model drops the 34 derived-index columns by default
    assert FINAL_DROP == 'noeng'
    for use_h, extra in ((False, 0), (True, 5)):
        got = drop_columns(select_columns(columns(use_h), 'final'), FINAL_DROP)
        assert len(got) == 495 + extra - 34, len(got)
        assert 'pachy_min_per_age' not in got and 'Pachy Min:' in got


if __name__ == '__main__':
    test_counts()
    test_l1_is_own_eye_only()
    test_e31_drops()
    test_final_feature_set_is_noeng()
    print('ok')
