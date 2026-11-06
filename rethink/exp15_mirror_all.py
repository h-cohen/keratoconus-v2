"""E15: E10c redone with mirror-aligned inter-eye map differences, trained on all 944 scans (pre-declared).

E14 showed (a) mirror alignment helps the map difference a little (+0.015) and (b) a bilateral-only
specialist loses to models trained on all scans. E15 combines both: all-data training, mirrored diffs.

Features: N = E02 fellow numerics incl. has_fellow; S = own spatial DINOv2 PCA(10)/map;
M = mirrored cell-wise own - fellow difference (DINOv2 PCA(10)/map + per-map L2 norm, colour PCA(10)),
set to 0 for scans without a same-day fellow (the flag is in N). PCA fitted on the outer training fold
(unsupervised; difference PCA on training rows that have a fellow). L2 LR, C by inner CV.
  E15a_mirror_all  : N + S + M, LR
  E15b_E12b_E15a   : equal-weight rank average of E12b and E15a OOF (same folds)
  E15c_E12b_E10c_E15a: equal-weight rank average of E12b, E10c and E15a OOF (same folds)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import exp14_pair as e14  # noqa: E402
from compare import paired_delta  # noqa: E402
from exp01_tabular import _inner_cv  # noqa: E402
from exp06_combo import fellow_X  # noqa: E402
from exp10_spatial import MAPS, fellow_index  # noqa: E402
from harness import DATA, RES, _git_hash, evaluate, summarise  # noqa: E402


def _diff_pca(Dtr, Dte, htr, hte, seed):
    """PCA fitted on training rows with a fellow; rows without one get zeros."""
    sc = e14.StandardScaler().fit(Dtr[htr])
    p = e14.PCA(n_components=e14.K, random_state=seed).fit(sc.transform(Dtr[htr]))
    a, b = np.zeros((len(Dtr), e14.K)), np.zeros((len(Dte), e14.K))
    a[htr], b[hte] = p.transform(sc.transform(Dtr[htr])), p.transform(sc.transform(Dte[hte]))
    return a, b


def blocks(dtr, dte, seed):
    Ntr, Nte = fellow_X(dtr), fellow_X(dte)
    med = Ntr.median()
    out_tr, out_te = [Ntr.fillna(med).values], [Nte.fillna(med).values]
    rtr, rte = dtr.crow.values, dte.crow.values
    itr, ite = fellow_index(dtr), fellow_index(dte)
    htr, hte = itr >= 0, ite >= 0
    ftr, fte = rtr[np.where(htr, itr, 0)], rte[np.where(hte, ite, 0)]
    for j in range(len(MAPS)):
        a, b = e14._pca(DINO[rtr, j], DINO[rte, j], seed)
        out_tr.append(a), out_te.append(b)
        Dtr = np.where(htr[:, None], DINO[rtr, j] - DINO[ftr, j], 0.0)
        Dte = np.where(hte[:, None], DINO[rte, j] - DINO[fte, j], 0.0)
        a, b = _diff_pca(Dtr, Dte, htr, hte, seed)
        out_tr += [a, np.linalg.norm(Dtr, axis=1)[:, None]]
        out_te += [b, np.linalg.norm(Dte, axis=1)[:, None]]
    Ctr = np.where(htr[:, None], COL[rtr] - COL[ftr], 0.0)
    Cte = np.where(hte[:, None], COL[rte] - COL[fte], 0.0)
    a, b = _diff_pca(Ctr, Cte, htr, hte, seed)
    out_tr.append(a), out_te.append(b)
    return np.hstack(out_tr), np.hstack(out_te)


def mirror_lr():
    def fp(dtr, ctr, dte, cte, seed):
        Xtr, Xte = blocks(dtr, dte, seed)
        y, g = dtr.y.values, dtr.pid.values
        C = _inner_cv(Xtr, y, g, e14.LR, e14.CGRID, seed)['C']
        return e14.LR(C).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


def rank_avg(name, parts, coh):
    oofs = [np.load(RES / 'oof' / f'{n}.npy') for n in parts]
    ens = np.stack([np.mean([rankdata(o[r]) / len(coh) for o in oofs], 0) for r in range(oofs[0].shape[0])])
    s = summarise(coh, ens)
    s.update(name=name, desc='equal-weight rank average of ' + ' + '.join(parts) + ' OOF (same folds)',
             params='{}', git=_git_hash(), secs=0, when=time.strftime('%Y-%m-%d %H:%M'))
    pd.DataFrame([s]).to_csv(RES / 'experiments.csv', mode='a', header=False, index=False)
    np.save(RES / 'oof' / f'{name}.npy', ens)
    print(f"{name:24s} AUC {s['auc']:.3f} [{s['auc_lo']:.3f}-{s['auc_hi']:.3f}] rep {s['auc_rep_mean']:.3f} "
          f"first-visit {s['auc_first']:.3f}")
    return ens


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    DINO, COL = e14.canonical(coh)
    cnn = np.load(DATA / 'cnn_frozen.npy')
    evaluate('E15a_mirror_all', mirror_lr(),
             'E02 numerics + own DINOv2 + mirrored inter-eye map differences, all 944 scans, LR', df=coh, cnn=cnn)
    rank_avg('E15b_E12b_E15a', ['E12b_E02_E06b_avg', 'E15a_mirror_all'], coh)
    rank_avg('E15c_E12b_E10c_E15a', ['E12b_E02_E06b_avg', 'E10c_spatial_diff_lr', 'E15a_mirror_all'], coh)

    y, g = coh.y.values, coh.pid.values
    age = np.load(RES / 'oof' / 'E07_age_all.npy').mean(0)
    has = fellow_index(coh) >= 0
    for n in ['E15a_mirror_all', 'E15b_E12b_E15a', 'E15c_E12b_E10c_E15a', 'E13_E12b_E10c_rankavg']:
        p = np.load(RES / 'oof' / f'{n}.npy').mean(0)
        d, (lo, hi), pr = paired_delta(y, p, age, g)
        db, (blo, bhi), _ = paired_delta(y[has], p[has], age[has], g[has])
        print(f'{n:24s} vs age {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P={pr:.2f} | bilateral '
              f'{roc_auc_score(y[has], p[has]):.3f} ({db:+.3f} [{blo:+.3f}, {bhi:+.3f}]) | single-eye '
              f'{roc_auc_score(y[~has], p[~has]):.3f}')
