"""E14: bilateral-visit specialist (pre-declared). Scans with the other eye imaged the same day only (360 scans),
plus a routed whole-cohort model. Motivation: E13's gain lives entirely in this stratum (0.785 vs age 0.680).

Mirror alignment: Pentacam maps are shown in anatomical orientation, so left-eye maps are mirror images of
right-eye maps. Left-eye 4x4 cell grids (DINOv2 and colour) are flipped left-right into the right-eye
layout before any inter-eye difference is taken (deterministic, no fitting). E10c instead subtracted
unaligned PCA scores.

Feature blocks (unsupervised PCA fitted once per outer training fold; never sees the test fold):
  N  E02 numerics without has_fellow: own engineered indices, fellow indices, asymmetry, age
  S  own-eye spatial DINOv2, PCA(10) per map
  M  mirrored cell-wise difference own - fellow: DINOv2 PCA(10) per map, colour PCA(10),
     plus the L2 norm of the raw difference per map (asymmetry magnitude, 4 values)
Runs (all on the 360-scan bilateral subset, patient-grouped 5x5 CV, L2 LR with C by inner CV):
  E14_age_pair     age only (reference within the stratum)
  E14a_pair_num    N
  E14b_pair_mirror N + S + M
  E14c_pair_avg    mean of E14a and E14b probabilities (same folds, no fitted weights)
Routed whole cohort (reported, from OOF): bilateral scans -> E14c, single-eye scans -> E07_age_all.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from compare import paired_delta  # noqa: E402
from exp01_tabular import _inner_cv  # noqa: E402
from exp06_combo import fellow_X  # noqa: E402
from exp10_spatial import MAPS, fellow_index  # noqa: E402
from harness import DATA, RES, _git_hash, cluster_boot_auc, evaluate, summarise  # noqa: E402

K = 10
LR = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))  # noqa: E731
CGRID = [dict(C=c) for c in [0.003, 0.01, 0.03, 0.1, 0.3]]


def canonical(coh):
    """DINOv2 (N,4,4,4,384) and colour (N,4,4,4,4) grids with left eyes mirrored to the right-eye layout."""
    dino = np.stack([np.load(DATA / f'dinogrid_{m}.npy') for m in MAPS], 1).reshape(len(coh), 4, 4, 4, -1)
    col = np.load(DATA / 'colorgrid.npy').reshape(len(coh), 4, 4, 4, 4)
    left = (coh.eye == 'left').values
    dino[left], col[left] = dino[left][:, :, :, ::-1], col[left][:, :, :, ::-1]
    return dino.reshape(len(coh), 4, -1), col.reshape(len(coh), -1)


def _pca(Xfit, Xapp, seed):
    sc = StandardScaler().fit(Xfit)
    p = PCA(n_components=K, random_state=seed).fit(sc.transform(Xfit))
    return p.transform(sc.transform(Xfit)), p.transform(sc.transform(Xapp))


def blocks(dtr, dte, seed, use):
    """Feature matrices for train and test rows; image PCA fitted on the training rows only."""
    out_tr, out_te = [], []
    if 'N' in use:
        Ntr, Nte = fellow_X(dtr).drop(columns='has_fellow'), fellow_X(dte).drop(columns='has_fellow')
        med = Ntr.median()
        out_tr.append(Ntr.fillna(med).values), out_te.append(Nte.fillna(med).values)
    rtr, rte = dtr.crow.values, dte.crow.values
    ftr, fte = rtr[fellow_index(dtr)], rte[fellow_index(dte)]
    if 'S' in use:
        for j in range(len(MAPS)):
            a, b = _pca(DINO[rtr, j], DINO[rte, j], seed)
            out_tr.append(a), out_te.append(b)
    if 'M' in use:
        for j in range(len(MAPS)):
            Dtr, Dte = DINO[rtr, j] - DINO[ftr, j], DINO[rte, j] - DINO[fte, j]
            a, b = _pca(Dtr, Dte, seed)
            out_tr += [a, np.linalg.norm(Dtr, axis=1)[:, None]]
            out_te += [b, np.linalg.norm(Dte, axis=1)[:, None]]
        a, b = _pca(COL[rtr] - COL[ftr], COL[rte] - COL[fte], seed)
        out_tr.append(a), out_te.append(b)
    return np.hstack(out_tr), np.hstack(out_te)


def pair_lr(use):
    def fp(dtr, ctr, dte, cte, seed):
        Xtr, Xte = blocks(dtr, dte, seed, use)
        y, g = dtr.y.values, dtr.pid.values
        C = _inner_cv(Xtr, y, g, LR, CGRID, seed)['C']
        return LR(C).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


def age_lr():
    def fp(dtr, ctr, dte, cte, seed):
        return LogisticRegression(max_iter=5000).fit(dtr[['age']].values, dtr.y.values) \
            .predict_proba(dte[['age']].values)[:, 1]
    return fp


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    DINO, COL = canonical(coh)
    cnn = np.load(DATA / 'cnn_frozen.npy')
    has = fellow_index(coh) >= 0
    sub, csub = coh[has].reset_index(drop=True), cnn[has]
    print(f'bilateral subset: {len(sub)} scans, {sub.pid.nunique()} patients, {sub.y.sum()} positive')

    _, o_age = evaluate('E14_age_pair', age_lr(), 'age only, bilateral-visit subset', df=sub, cnn=csub)
    _, o_a = evaluate('E14a_pair_num', pair_lr('N'), 'E02 numerics w/o flag, bilateral subset, LR',
                      df=sub, cnn=csub)
    _, o_b = evaluate('E14b_pair_mirror', pair_lr('NSM'),
                      'numerics + own DINOv2 + mirrored inter-eye map differences, bilateral subset, LR',
                      df=sub, cnn=csub)
    o_c = (o_a + o_b) / 2
    s = summarise(sub, o_c)
    s.update(name='E14c_pair_avg', desc='mean of E14a and E14b probabilities (same folds), bilateral subset',
             params='{}', git=_git_hash(), secs=0, when=time.strftime('%Y-%m-%d %H:%M'))
    pd.DataFrame([s]).to_csv(RES / 'experiments.csv', mode='a', header=False, index=False)
    np.save(RES / 'oof' / 'E14c_pair_avg.npy', o_c)
    print(f"E14c_pair_avg AUC {s['auc']:.3f} [{s['auc_lo']:.3f}-{s['auc_hi']:.3f}] rep {s['auc_rep_mean']:.3f} "
          f"first-visit {s['auc_first']:.3f}")

    y, g, age = sub.y.values, sub.pid.values, o_age.mean(0)
    for n, o in [('E14a', o_a), ('E14b', o_b), ('E14c', o_c)]:
        d, (lo, hi), pr = paired_delta(y, o.mean(0), age, g)
        print(f'{n} - age (bilateral): dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P={pr:.2f}')
    d, (lo, hi), pr = paired_delta(y, o_b.mean(0), o_a.mean(0), g)
    print(f'E14b - E14a (value of maps): dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P={pr:.2f}')
    e13 = np.load(RES / 'oof' / 'E13_E12b_E10c_rankavg.npy').mean(0)[has]
    print(f'E13 on the same scans (different folds): {roc_auc_score(y, e13):.3f}')

    yall, gall = coh.y.values, coh.pid.values
    routed = np.load(RES / 'oof' / 'E07_age_all.npy').mean(0)
    routed[has] = o_c.mean(0)
    lo, hi = cluster_boot_auc(yall, routed, gall)
    d, (dlo, dhi), pr = paired_delta(yall, routed, np.load(RES / 'oof' / 'E07_age_all.npy').mean(0), gall)
    print(f'routed whole cohort: AUC {roc_auc_score(yall, routed):.3f} [{lo:.3f}-{hi:.3f}]  '
          f'vs age dAUC {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}] P={pr:.2f}')
