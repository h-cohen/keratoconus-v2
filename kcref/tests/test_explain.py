"""Exact member SHAP, blend SHAP and explanation metadata (E29). Synthetic data only.
Run: python -I kcref/tests/test_explain.py"""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
import model as M  # noqa: E402
import explain as E  # noqa: E402


def synthetic_fz(n=300, p=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (2.5 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n) > 0.8).astype(int)
    cols = [f'f{j}' for j in range(p)]
    hp = {'xgb': dict(max_depth=2, n_estimators=50), 'lgbm': dict(n_estimators=50, num_leaves=4),
          'lr_en': dict(C=0.3, l1_ratio=0.5)}
    models = {}
    for k in M.MEMBERS:
        kind = 'lgbm' if k == 'lgbm_bag' else k
        seeds = [7 + s for s in range(5)] if k == 'lgbm_bag' else [7]
        models[k] = [M.learner(kind, s, y, cols)[0](**hp[kind]).fit(X, y) for s in seeds]
    return dict(models=models, cols=cols), X


def test_additivity():
    fz, X = synthetic_fz()
    sh, mg, bias = E.member_shap(fz, X), E.member_margin(fz, X), E.member_bias(fz, X)
    for k in M.MEMBERS:
        assert sh[k].shape == X.shape
        assert np.allclose(sh[k].sum(1) + bias[k], mg[k], atol=1e-4), k


def test_blend_shape():
    fz, X = synthetic_fz()
    b = E.blend_shap(fz, X)
    assert b.shape == X.shape and np.isfinite(b).all()


def test_member_agreement():
    fz, X = synthetic_fz()
    sh = E.member_shap(fz, X)
    for k in M.MEMBERS:
        assert np.abs(sh[k]).mean(0).argmax() == 0, k


def columns(use_h=False):
    from tests_cols import columns as c
    return c(use_h)


def test_family_counts():
    from collections import Counter
    got = Counter(E.family(c) for c in columns())
    assert got == {'own_num': 65, 'age': 1, 'fel_num': 65, 'asym_num': 74, 'R_own': 96, 'R_fel': 96, 'R_asym': 96,
                   'flags': 2}, got
    assert Counter(E.family(c) for c in columns(True))['H'] == 5


def test_source_partition():
    srcs = {'own_num', 'own_maps', 'fellow_num', 'fellow_maps', 'age', 'flags', 'H'}
    assert all(E.source(c) in srcs for c in columns(True))
    assert E.source('asym_Pachy Min:') == 'fellow_num' and E.source('R_asym3') == 'fellow_maps'
    assert E.source('R_own3') == 'own_maps' and E.source('worse_ISV:') == 'fellow_num'


def test_r_name():
    assert E.r_name('R_own0') == 'T_r01_S'
    assert E.r_name('R_asym95') == 'T_ratio31'
    assert E.r_name('R_own48') == 'EB_r01_S'


def test_sector():
    assert E.sector('EB_r12_I') == ('EB', 1, 2, 'I')
    assert E.sector('T_ext_val') is None


def test_r_mask():
    cols = ['age', 'R_own0', 'R_fel0', 'R_asym0']
    flags = np.array([[1, 1], [0, 1], [1, 0]], float)  # columns: has_fellow_image, own_image
    m = E.r_mask(cols, flags)
    assert m.tolist() == [[True, True, True, True], [True, True, False, False], [True, False, True, False]]


def test_labels_unique():
    labs = [E.label(c) for c in columns(True)]
    assert all(labs) and len(set(labs)) == len(labs) == 500
    assert E.label('R_own48') == 'Back elevation (µm), 0–1 mm superior, own'
    assert E.label('asym_KMax Sagittal Front (D)') == 'KMax sagittal front (D), own − fellow'


def test_abcd_examples():
    assert E.abcd('Pachy Min:') == 'C' and E.abcd('fel_Pachy Min:') == 'C'
    assert E.abcd('R_own48') == 'B' and E.abcd('Rm B (mm):') == 'B'
    assert E.abcd('Km F (D):') == 'A' and E.abcd('R_asym16') == 'A'
    assert E.abcd('age') == 'other' and E.abcd('ISV:') == 'other'


def test_match_rules():
    import pandas as pd
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(dict(pid=rng.integers(0, 120, n), y=(rng.random(n) < 0.25).astype(int), km=rng.normal(46, 3, n),
                           age=rng.normal(25, 6, n), stage=rng.integers(1, 4, n)))
    pr = E.match_pairs(df)
    assert len(pr) > 10
    assert (df.y.values[pr.pos_idx] == 1).all() and (df.y.values[pr.neg_idx] == 0).all()
    assert (df.stage.values[pr.pos_idx] == df.stage.values[pr.neg_idx]).all()
    assert (df.pid.values[pr.pos_idx] != df.pid.values[pr.neg_idx]).all()
    assert pr.pos_idx.is_unique and pr.neg_idx.is_unique
    st = E.pair_stats(pr, s_final=df.y.values + 0.1 * rng.normal(size=n), s_L1=rng.normal(size=n))
    assert st['n_pairs'] == len(pr) and st['frac_final'] > st['frac_L1'] and 0 <= st['p_sign'] <= 1


if __name__ == '__main__':
    for t in [v for k, v in dict(globals()).items() if k.startswith('test_')]:
        t()
    print('ok')
