"""Explanations of the final blend (E29): exact out-of-fold SHAP and the feature metadata used by the figures.

SHAP per member, in that member's margin (log-odds):
  XGBoost, LightGBM: native exact TreeSHAP (pred_contribs / pred_contrib); the LightGBM bag is the mean over its seeds.
  Elastic-net LR: linear SHAP, coef x standardised value (reference = training mean, which the scaler maps to 0).
Blend attribution: each member's SHAP divided by the SD of its margin over the rows explained, averaged over members
with equal weights, the same equal weighting as the rank blend.

Usage (repo root): python -I kcref/explain.py shap {noH|H}   -> kcref/results/shap/{arm}_oof.npz (patient data)
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import DMatrix

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as EV  # noqa: E402
import features as F  # noqa: E402
import model as M  # noqa: E402
import radiomics  # noqa: E402
from common import RES  # noqa: E402


# ---------------------------------------------------------------- feature metadata
R_NAMES = radiomics.names()
MAPS = {'T': 'Thickness (µm)', 'K': 'Front sagittal curvature (D)', 'EF': 'Front elevation (µm)',
        'EB': 'Back elevation (µm)'}
QUAD = {'S': 'superior', 'I': 'inferior', 'N': 'nasal', 'T': 'temporal'}
SIDE = {'own': 'own', 'fel': 'fellow', 'asym': 'own − fellow'}
FLAGS = {'has_fellow_image': 'Fellow maps available', 'own_image': 'Own maps available'}
ENG = {'kmax_minus_km': 'KMax − Km front (D)', 'pachy_min_over_apex': 'Thinnest / apex pachymetry',
       'pachy_min_over_pupil': 'Thinnest / pupil pachymetry', 'pachy_min_per_age': 'Thinnest pachymetry / age',
       'prog_mean': 'Mean pachymetric progression', 'prog_max': 'Max pachymetric progression',
       'pachy_drop_4mm': 'Pachymetry drop centre to 4 mm (µm)', 'ecc_front_back': 'Eccentricity front − back',
       'asph_ratio': 'Asphericity Q front / back', 'elev_ratio': 'Rf front / Rf back',
       'cvol_ratio': 'Corneal volume 3 mm / 10 mm'}
H_LAB = {'fellow_cxl_before_visit': 'Fellow eye cross-linked before visit',
         'years_since_fellow_cxl': 'Years since fellow-eye CXL', 'fellow_kp_before_visit': 'Fellow keratoplasty before visit',
         'fellow_cxl_unknown': 'Fellow CXL date unknown', 'fellow_tx_missing': 'Not in treatment sheet'}
ABCD_A = {'Rf F (mm):', 'Rs F (mm):', 'Rh F (mm):', 'Rv F (mm):', 'K1 F (D):', 'K2 F (D):', 'Rm F (mm):', 'Km F (D):',
          'Astig F (D):', 'R Per F (mm)', 'R Min (mm)', 'KMax Sagittal Front (D)', 'RSagMin:', 'kmax_minus_km'}
ABCD_B = {'Rf B (mm):', 'Rs B (mm):', 'Rh B (mm):', 'Rv B (mm):', 'K1 B (D):', 'K2 B (D):', 'Rm B (mm):', 'Km B (mm):',
          'Astig B (D):', 'R Per B (mm):', 'R Min B (mm):'}


def _split(col):
    """(kind, side, base): kind in num / worse / R / H / flag / age / has_fellow."""
    if col.startswith('R_'):
        m = re.match(r'R_(own|fel|asym)(\d+)$', col)
        return 'R', m.group(1), R_NAMES[int(m.group(2))]
    if col in F.H_COLS:
        return 'H', '', col
    if col in FLAGS or col in ('age', 'has_fellow'):
        return {'age': 'age', 'has_fellow': 'has_fellow'}.get(col, 'flag'), '', col
    for pre, side in (('fel_', 'fel'), ('asym_', 'asym'), ('worse_', 'worse')):
        if col.startswith(pre):
            return ('worse' if side == 'worse' else 'num'), side, col[len(pre):]
    return 'num', 'own', col


def family(col):
    kind, side, _ = _split(col)
    if kind == 'R':
        return f'R_{side}'
    if kind in ('worse', 'has_fellow'):
        return 'asym_num'
    if kind == 'num':
        return f'{side}_num'
    return {'flag': 'flags'}.get(kind, kind)


def source(col):
    f = family(col)
    return {'own_num': 'own_num', 'R_own': 'own_maps', 'fel_num': 'fellow_num', 'asym_num': 'fellow_num',
            'R_fel': 'fellow_maps', 'R_asym': 'fellow_maps'}.get(f, f)


def r_name(col):
    return _split(col)[2]


def sector(rname):
    m = re.match(r'(T|K|EF|EB)_r(\d)(\d)_([SINT])$', rname)
    return (m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)) if m else None


def _num_label(b):
    if b in ENG:
        return ENG[b]
    if b == 'KMax Sagittal Front (D)':
        return 'KMax sagittal front (D)'
    m = re.match(r'D(\d+)mm (Pachy|Prog)$', b)
    if m:
        return f'{"Pachymetry" if m.group(2) == "Pachy" else "Pachymetric progression"} at {m.group(1)} mm'
    s = b.rstrip(':').strip().replace('C.Vol D', 'Corneal volume')
    s = re.sub(r' F( |$)', r' front\1', s)
    return re.sub(r' B( |$)', r' back\1', s)


def _r_label(rn):
    mp = rn.split('_')[0]
    sec = sector(rn)
    if sec:
        return f'{MAPS[sec[0]]}, {sec[1]}–{sec[2]} mm {QUAD[sec[3]]}'
    name, rest = MAPS[mp], rn[len(mp) + 1:]
    ext = 'thinnest point' if mp == 'T' else 'maximum'
    lab = {'ext_val': f'{ext} value', 'ext_x': f'{ext} x (mm, nasal +)', 'ext_y': f'{ext} y (mm, superior +)',
           'ext_r': f'{ext} distance from centre (mm)', 'IminusS': 'inferior − superior, 2–4 mm',
           'NminusT': 'nasal − temporal, 2–4 mm', 'sd3': 'SD within 3 mm', 'prog1': 'rise 1 mm around thinnest',
           'prog2': 'rise 2 mm around thinnest', 'prog3': 'rise 3 mm around thinnest',
           'ratio31': 'rise ratio 3 mm / 1 mm'}[rest]
    return f'{name}, {lab}'


def label(col):
    """Human-readable clinical label with units."""
    kind, side, b = _split(col)
    if kind == 'R':
        return f'{_r_label(b)}, {SIDE[side]}'
    if kind == 'H':
        return H_LAB[b]
    if kind == 'flag':
        return FLAGS[b]
    if kind == 'age':
        return 'Age (years)'
    if kind == 'has_fellow':
        return 'Fellow exam present'
    if kind == 'worse':
        return f'Worse than fellow: {_num_label(b)}'
    return _num_label(b) + ('' if side == 'own' else f', {SIDE[side]}')


def abcd(col):
    """Belin ABCD component: A anterior curvature, B posterior curvature, C thickness, else 'other'."""
    kind, _, b = _split(col)
    if kind == 'R':
        return {'T': 'C', 'K': 'A', 'EB': 'B'}.get(b.split('_')[0], 'other')
    if kind not in ('num', 'worse'):
        return 'other'
    if b in ABCD_A:
        return 'A'
    if b in ABCD_B:
        return 'B'
    if 'Pachy' in b or 'Prog' in b or b.startswith('pachy') or b.startswith('prog_'):
        return 'C'
    return 'other'


def r_mask(cols, flags):
    """(n, p) bool: True where the column's map exists for the row; flags = (n, 2) [has_fellow_image, own_image]."""
    fel, own = flags[:, 0] > 0, flags[:, 1] > 0
    out = np.ones((len(flags), len(cols)), bool)
    for j, c in enumerate(cols):
        f = family(c)
        out[:, j] = {'R_own': own, 'R_fel': fel, 'R_asym': own & fel}.get(f, out[:, j])
    return out


# ---------------------------------------------------------------- SHAP
def _parts(fz, X):
    """{member: (shap (n, p), bias (n,), margin (n,))}."""
    out = {}
    for k, ms in fz['models'].items():
        if k == 'xgb':
            m = ms[0]
            c = m.get_booster().predict(DMatrix(X), pred_contribs=True)
            out[k] = c[:, :-1], c[:, -1], m.predict(X, output_margin=True)
        elif k == 'lgbm_bag':
            c = np.mean([m.predict(X, pred_contrib=True) for m in ms], axis=0)
            out[k] = c[:, :-1], c[:, -1], np.mean([m.predict(X, raw_score=True) for m in ms], axis=0)
        elif k == 'lr_en':
            sc, lr = ms[0].steps[0][1], ms[0].steps[1][1]
            z = sc.transform(X)
            out[k] = z * lr.coef_[0], np.full(len(X), lr.intercept_[0]), ms[0].decision_function(X)
        else:
            raise ValueError(k)
    return out


def member_shap(fz, X):
    return {k: v[0] for k, v in _parts(fz, X).items()}


def member_bias(fz, X):
    return {k: v[1] for k, v in _parts(fz, X).items()}


def member_margin(fz, X):
    return {k: v[2] for k, v in _parts(fz, X).items()}


def blend_shap(fz, X):
    """(n, p) blend attribution: mean over members of SHAP / SD(member margin)."""
    p = _parts(fz, X)
    return np.mean([p[k][0] / p[k][2].std() for k in M.MEMBERS], axis=0)


def raw_values(D, idx, cols):
    """Un-standardised values of `cols` for visit rows idx (for colouring and stage summaries); NaN where missing."""
    E = D['R']

    def rmap(c):
        out = np.full((len(c), E.shape[1]), np.nan)
        out[c >= 0] = E[c[c >= 0]]
        return out
    o, f = rmap(D['own'][idx]), rmap(D['felc'][idx])
    R = pd.DataFrame(np.hstack([o, f, o - f]),
                     columns=[f'R_{w}{j}' for w in ('own', 'fel', 'asym') for j in range(E.shape[1])])
    tab = pd.concat([D['N'].iloc[idx].reset_index(drop=True), D['H'].iloc[idx].reset_index(drop=True),
                     D['flags'].iloc[idx].reset_index(drop=True), R], axis=1)
    return tab[cols].values.astype(np.float32)


def shap_pass(arm):
    """Re-run the 25 final-model fits of `arm`, storing out-of-fold blend SHAP; asserts the OOF equals E28."""
    coh = M.cohort()
    D = F.build(coh)
    use_h = arm == 'H'
    smoke = os.environ.get('KC_SHAP_SMOKE') == '1'
    ref = np.load(RES / 'oof' / f'E28_{arm}_blend.npy')
    st = dict(shap={}, fit_imp=[], member_imp=[], cols=None)

    def fp(dtr, dte, seed):
        fz = M.fit(D, F.train_rows(D, dtr), use_h, seed)
        rows = dte.crow.values
        X = M.matrix(fz, D, rows)
        s = M.member_scores(fz, X)
        parts = _parts(fz, X)
        sh = np.mean([parts[k][0] / parts[k][2].std() for k in M.MEMBERS], axis=0)
        first = D['first'][rows]
        st['shap'].setdefault(seed, np.full((len(coh), X.shape[1]), np.nan))[rows] = sh
        st['fit_imp'].append(np.abs(sh[first]).mean(0))
        st['member_imp'].append([np.abs(parts[k][0][first] / parts[k][2].std()).mean(0) for k in M.MEMBERS])
        st['cols'] = fz['cols']
        return np.mean([rankdata(s[k]) / len(dte) for k in M.MEMBERS], axis=0)

    if smoke:
        tr, te = next(StratifiedGroupKFold(EV.N_SPLITS, shuffle=True, random_state=EV.SEED)
                      .split(coh, coh.y.values, coh.pid.values))
        got = fp(coh.iloc[tr].reset_index(drop=True), coh.iloc[te].reset_index(drop=True), EV.SEED)
        d = float(np.abs(got - ref[0, te]).max())
        assert d < 1e-12, d
        print('smoke ok, max |diff|', d)
        return None
    oof = EV.cv_oof(coh, fp)
    d = float(np.abs(oof - ref).max())
    assert d < 1e-12, f'explanation pass drifted from E28_{arm}_blend: max |diff| {d}'
    out = RES / 'shap' / f'{arm}_oof.npz'
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, shap=np.nanmean([st['shap'][k] for k in sorted(st['shap'])], axis=0),
                        fit_imp=np.array(st['fit_imp']), member_imp=np.mean(st['member_imp'], axis=0),
                        cols=np.array(st['cols']), raw=raw_values(D, coh.crow.values, st['cols']), oof=oof.mean(0))
    print(f'saved results/shap/{arm}_oof.npz; oof identical (max |diff| {d})')
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'shap':
        shap_pass(sys.argv[2])
