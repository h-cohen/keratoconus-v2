"""E17: single-visit, single-eye raw data squeezed from the composites (pre-declared). All 944 scans.

New information per scan (no fellow eye, no history):
  P  printed panel values not in the export (ocr_panel.py, 100% exact on held-out known fields):
     KMax location (nasal / superior / decentration), thinnest-to-KMax distance, cornea volume, KPD,
     chamber volume, angle, depth, pupil diameter, IOP correction
  I  clinical indices from the physically decoded maps (decode_maps.py; 31-step palettes per image,
     scale 28.9 / 27.3 px/mm calibrated on the printed thinnest location, r = 0.99 / 0.95):
     back / front elevation at the thinnest point, max back / front elevation within the central 4 mm,
     I-S curvature (inferior - superior, ring 1-2.5 mm), inferior - superior thickness, cone area
     (K >= 48 D within r <= 3 mm), area of back elevation >= +20 um within r <= 3 mm
  R  ring/sector means of the decoded maps (rings 0-1-2-3-4 mm x 8 sectors x 4 maps = 128), left eyes
     mirrored to nasal-positive, PCA(10) per map fitted in the training fold
Candidates (L2 LR with C by inner CV, median imputation in-fold; LightGBM small fixed):
  raw_lr      : 55 numerics + P + I            raw_lgbm : same, LightGBM
  raw_ring_lr : 55 numerics + P + I + R-PCA
Runs:
  E17a_raw_lr, E17b_raw_lgbm, E17c_raw_ring_lr : each candidate alone (outer CV)
  E17d_nested_single_eye_raw : nested inner-CV selection over E16's pool A + the three candidates
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import exp16_nested as e16  # noqa: E402
from compare import paired_delta  # noqa: E402
from harness import DATA, RES, evaluate  # noqa: E402

SX, SY, BLOCK = 28.9, 27.3, 5


def grid_mm(n):
    """Cell centres in mm (x right on screen, y up) for the decoded 58x58 grid."""
    c = (np.arange(n) + .5) * BLOCK - 145
    return np.meshgrid(c / SX, -c / SY)


def panel_features(coh, o):
    nas = np.where(coh.eye == 'right', 1, -1)
    return pd.DataFrame({
        'kmax_nasal': o.kmax_x * nas, 'kmax_sup': o.kmax_y, 'kmax_decentr': np.hypot(o.kmax_x, o.kmax_y),
        'thin_to_kmax': np.hypot(o.kmax_x - o.thin_x, o.kmax_y - o.thin_y), 'cornea_vol': o.cornea_vol,
        'kpd': o.kpd, 'chamber_vol': o.chamber_vol, 'angle': o.angle, 'acd': o.acd,
        'pupil_dia': o.pupil_dia, 'iop_corr': o.iop_corr})


def map_indices(coh, o, V):
    n = V.shape[-1]
    X, Y = grid_mm(n)
    R, A = np.hypot(X, Y), np.degrees(np.arctan2(Y, X)) % 360
    rows = []
    for i in range(len(coh)):
        T, K, EF, EB = V[i]
        ti = np.unravel_index(np.argmin(np.hypot(X - o.thin_x.iloc[i], Y - o.thin_y.iloc[i])), X.shape)
        c4, c6 = R <= 2, R <= 3
        inf_ = (R >= 1) & (R <= 2.5) & (A > 240) & (A < 300)
        sup_ = (R >= 1) & (R <= 2.5) & (A > 60) & (A < 120)
        with np.errstate(all='ignore'):
            rows.append(dict(
                eb_thin=EB[ti], ef_thin=EF[ti], eb_max4=np.nanmax(EB[c4]), ef_max4=np.nanmax(EF[c4]),
                is_curv=np.nanmean(K[inf_]) - np.nanmean(K[sup_]), is_thick=np.nanmean(T[inf_]) - np.nanmean(T[sup_]),
                cone_area=np.nanmean(K[c6] >= 48), eb20_area=np.nanmean(EB[c6] >= 20)))
    return pd.DataFrame(rows)


def ring_sectors(coh, V):
    n = V.shape[-1]
    X, Y = grid_mm(n)
    left = (coh.eye == 'left').values
    out = np.full((len(coh), 4, 32), np.nan)
    for i in range(len(coh)):
        Xi = -X if left[i] else X  # nasal-positive
        R, A = np.hypot(Xi, Y), (np.degrees(np.arctan2(Y, Xi)) % 360) // 45
        for j in range(4):
            for r in range(4):
                for s in range(8):
                    m = (R >= r) & (R < r + 1) & (A == s)
                    with np.errstate(all='ignore'):
                        out[i, j, r * 8 + s] = np.nanmean(V[i, j][m])
    return out


def build(coh):
    o = pd.read_parquet(DATA / 'panel_ocr.parquet')
    V = np.load(DATA / 'decoded_maps.npy')
    tab = pd.concat([panel_features(coh, o), map_indices(coh, o, V)], axis=1)
    return tab, ring_sectors(coh, V)


def _X(dtr, dte, rings, seed):
    cols = e16.NUM + list(TAB.columns)
    Xtr, Xte = pd.concat([dtr[e16.NUM], TAB.iloc[dtr.crow.values].reset_index(drop=True)[TAB.columns]], axis=1), \
        pd.concat([dte[e16.NUM], TAB.iloc[dte.crow.values].reset_index(drop=True)[TAB.columns]], axis=1)
    med = Xtr.median()
    Xtr, Xte = Xtr[cols].fillna(med).values, Xte[cols].fillna(med).values
    if rings:
        for j in range(4):
            Rtr, Rte = RING[dtr.crow.values, j], RING[dte.crow.values, j]
            m = np.nanmedian(Rtr, 0)
            a, b = e16._pca(np.where(np.isnan(Rtr), m, Rtr), np.where(np.isnan(Rte), m, Rte), seed)
            Xtr, Xte = np.hstack([Xtr, a]), np.hstack([Xte, b])
    return Xtr, Xte


def raw_lr(dtr, dte, seed):
    return e16.lr_on(*_X(dtr, dte, False, seed), dtr, seed)


def raw_lgbm(dtr, dte, seed):
    Xtr, Xte = _X(dtr, dte, False, seed)
    return e16._lgbm(dtr.y.values, seed).fit(Xtr, dtr.y.values).predict_proba(Xte)[:, 1]


def raw_ring_lr(dtr, dte, seed):
    return e16.lr_on(*_X(dtr, dte, True, seed), dtr, seed)


NEW = {'raw_lr': raw_lr, 'raw_lgbm': raw_lgbm, 'raw_ring_lr': raw_ring_lr}

if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    TAB, RING = build(coh)
    TAB.to_parquet(DATA / 'rawmap_features.parquet', index=False)
    y = coh.y.values
    for c in TAB.columns:
        m = TAB[c].notna().values
        a = roc_auc_score(y[m], TAB[c].values[m])
        print(f'  univariate {c:14s} AUC {max(a, 1 - a):.3f} ({"+" if a >= .5 else "-"}) n={m.sum()}')

    e16.CANDS.update(NEW)
    oofs = {}
    for run, k in [('E17a_raw_lr', 'raw_lr'), ('E17b_raw_lgbm', 'raw_lgbm'), ('E17c_raw_ring_lr', 'raw_ring_lr')]:
        _, oofs[run] = evaluate(run, lambda dtr, ctr, dte, cte, seed, k=k: e16.predict(k, dtr, dte, seed),
                                f'single eye: 55 numerics + panel OCR + decoded-map indices{" + ring PCA" if "ring" in k else ""}',
                                df=coh, cnn=cnn)
    pool = e16.POOL_A + list(NEW)
    _, oofs['E17d'] = evaluate('E17d_nested_single_eye_raw', e16.select(pool, 'A+raw'),
                               'nested selection: E16 pool A + raw-map candidates, single eye, all 944', df=coh, cnn=cnn)
    ch = pd.DataFrame(e16.CHOICES)
    ch.to_csv(RES / 'nested17_choices.csv', index=False)
    print(ch.chosen.value_counts().to_string())
    g = coh.pid.values
    age = np.load(RES / 'oof' / 'E07_age_all.npy').mean(0)
    for n, o_ in oofs.items():
        d, (lo, hi), pr = paired_delta(y, o_.mean(0), age, g)
        print(f'{n} vs age: dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P={pr:.2f}')
