"""E22: bilateral single-visit programme (gpu_run/DESIGN.md section 6). All 944 scans, harness folds (5 x 5, seeds
2026-2030), every run logged by harness.evaluate.

Rows (E21 `visits_clean`): cohort scans (crow 0..943), extra labelled exams (training-only augmentation; a cohort
patient's extras only when that patient is in the training fold) and raw exams that serve only as fellows.
Blocks per row (label-free except H, which uses dated treatment strictly before the visit, U2/C1):
  N  E20 'full': age + 54 numerics + 12 indices, own / fellow / own - fellow, worse-eye flags, has_fellow
  H  fellow CXL before visit, years since, fellow keratoplasty before visit, undated-CXL flag, not-in-sheet flag
  P  panel OCR (11 features) own / fellow / difference      } fellow part only where the fellow has a same-day
  M  decoded-map indices (8) own / fellow / difference      } composite (has_fellow_image)
  I_e  encoder e in {Z, D, L, X, T}: StandardScaler + PCA(10) fitted on the training rows' own composites, applied
       to own and (mirrored) fellow composite embeddings; own, fellow, own - fellow, has_fellow_image
Missing values (no fellow image; augmentation rows have no image at all, flagged by own_image) are filled with
the training-row median. All features are clipped to the training rows' 0.5 / 99.5% quantiles (OCR outliers).
Learners (as E20): LR = StandardScaler + L2 LR, C by patient-grouped inner CV over {0.01..1}; LGBM = depth 2,
age monotone decreasing, 4-config inner CV (100/300 trees x 2/4 leaves).
Candidates (21): bn_{lr,lgbm}[_aug] (4), bnpm_{lr,lgbm} (2), bi_<e>_{lr,lgbm} (10), ens_<e> = mean(bn_aug_lgbm,
bi_<e>_lr) (5).
Runs (CLI stage):
  ref     E22_ref_single_num  nested {num_lr, num_lgbm} (E16 candidates, scanned eye's 55 numerics)
  cands   E22_<candidate> for each of the 21
  nested  E22n_nested         nested selection over the 21 (choices -> nested22_choices.csv)
  noH     E22n_noH            the same with block H removed from every candidate
  neg     E22_negctrl         nested over the 4 bn candidates with every row's whole fellow (numerics, H, images)
                              taken from a random other patient within +-2 years of age (C7)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import exp16_nested as e16  # noqa: E402
import exp20_bilateral as e20  # noqa: E402
import exp21_bilateral_clean as e21  # noqa: E402
from exp01_tabular import _inner_cv  # noqa: E402
from harness import DATA, RES, evaluate  # noqa: E402
from prepare_fellow_tx import H_COLS, tx_features  # noqa: E402

ENC = ['Z', 'D', 'L', 'X', 'T']
PM_COLS = None
_EMB = {}


def emb(e):
    if e not in _EMB:
        _EMB[e] = np.load(DATA / f'emb_{e}.npy').astype(np.float64)
    return _EMB[e]


def build(coh, neg_seed=None):
    """Feature tables for every visit row. neg_seed: negative control, each row's whole fellow swapped (C7)."""
    V = e21.visits_clean(coh)
    n_lab = int((V.src != 'fellow_only').sum())  # cohort + extra rows come first
    meta = pd.read_parquet(DATA / 'decoded_hr_meta.parquet')
    meta['d'] = pd.to_datetime(meta.date).dt.normalize()
    lut = {}
    for i, (p, e, d) in enumerate(zip(meta.pkey, meta.eye, meta.d)):
        lut.setdefault((p, e, d), i)
    d = pd.to_datetime(V.date).dt.normalize()
    other = np.where(V.eye.values == 'right', 'left', 'right')
    own = np.array([lut.get(k, -1) for k in zip(V.pid, V.eye, d)])
    cr = meta.cohort_row.values
    own[:len(coh)] = [int(np.flatnonzero(cr == i)[0]) for i in range(len(coh))]  # exact composite of each scan
    felc = np.array([lut.get(k, -1) for k in zip(V.pid, other, d)])
    H = tx_features(V.pid.values, V.eye.values, V.date.values)
    if neg_seed is not None:
        rng = np.random.default_rng(neg_seed)
        age, pid = V.age.values, V.pid.values
        donor = np.arange(len(V))
        for i in range(n_lab):
            cand = np.flatnonzero((np.abs(age[:n_lab] - age[i]) <= 2) & (pid[:n_lab] != pid[i]))
            if len(cand):
                donor[i] = rng.choice(cand)
        V = V.copy()
        V['fel'] = V.fel.values[donor]
        felc = felc[donor]
        H = H.iloc[donor].reset_index(drop=True)
    N = e20.features(V)['full']
    pa = pd.read_parquet(DATA / 'panel_all.parquet')
    has_fi = felc >= 0
    po = pa.reindex(np.where(own >= 0, own, -1)).reset_index(drop=True)
    pf = pa.reindex(np.where(has_fi, felc, -1)).reset_index(drop=True)
    PM = pd.concat([po.add_prefix('own_'), pf.add_prefix('fel_'), (po - pf).add_prefix('asym_')], axis=1)
    flags = pd.DataFrame({'has_fellow_image': has_fi.astype(float), 'own_image': (own >= 0).astype(float)})
    first = np.zeros(len(V), bool)
    first[:len(coh)] = coh.first_visit.values
    return dict(V=V, N=N, H=H, PM=PM, flags=flags, own=own, felc=felc, first=first)


def fit_enc(D, e, rows, seed, k=10):
    """StandardScaler + PCA(k) on the training rows' own composite embeddings (unsupervised, in fold)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    E = D.get('emb', {}).get(e) if 'emb' in D else emb(e)
    own = D['own']
    fit_rows = np.unique(own[rows][own[rows] >= 0])
    fit_rows = fit_rows[np.isfinite(E[fit_rows]).all(1)]
    sc = StandardScaler().fit(E[fit_rows])
    return sc, PCA(n_components=k, random_state=seed).fit(sc.transform(E[fit_rows]))


def enc_apply(D, e, idx, sc, pca):
    """own, (mirrored) fellow and own - fellow PCA scores; NaN where the composite is missing."""
    E = D.get('emb', {}).get(e) if 'emb' in D else emb(e)
    k = pca.n_components_

    def tr(c):
        out = np.full((len(c), k), np.nan)
        ok = (c >= 0)
        ok[ok] = np.isfinite(E[c[ok]]).all(1)
        out[ok] = pca.transform(sc.transform(E[c[ok]]))
        return out

    o, f = tr(D['own'][idx]), tr(D['felc'][idx])
    return pd.DataFrame(np.hstack([o, f, o - f]), columns=[f'{e}_{w}{j}' for w in ('own', 'fel', 'asym') for j in range(k)])


def assemble(D, idx, blocks, enc, use_h, encfit=None):
    """Design matrix (before clipping / imputation) for visit rows idx."""
    parts = [D['N'].iloc[idx].reset_index(drop=True)]
    if use_h:
        parts.append(D['H'].iloc[idx].reset_index(drop=True))
    if 'PM' in blocks:
        parts.append(D['PM'].iloc[idx].reset_index(drop=True))
    if 'PM' in blocks or enc:
        parts.append(D['flags'].iloc[idx].reset_index(drop=True))
    if enc:
        parts.append(enc_apply(D, enc, idx, *encfit))
    return pd.concat(parts, axis=1)


def train_rows(D, dtr, aug):
    V = D['V']
    rows = list(dtr.crow.values)
    if aug:
        ok = (V.src == 'extra') & (~V.cohort_patient | V.pid.isin(set(dtr.pid)))
        rows += list(np.flatnonzero(ok.values))
    return np.array(rows)


def fit_model(D, rows, blocks, kind, enc, use_h, seed, w=None):
    """Everything fitted on the training rows only; returns the frozen pipeline (used for CV and for freezing)."""
    V = D['V']
    encfit = fit_enc(D, enc, rows, seed) if enc else None
    Xtr = assemble(D, rows, blocks, enc, use_h, encfit)
    lo, hi = Xtr.quantile(0.005), Xtr.quantile(0.995)
    Xtr = Xtr.clip(lo, hi, axis=1)
    med = Xtr.median()
    cols = list(Xtr.columns)
    Xtr = Xtr.fillna(med).fillna(0).values
    y, g = V.y.values[rows], V.pid.values[rows]
    if kind == 'lr':
        hp = _inner_cv(Xtr, y, g, e16.LR, e16.CGRID, seed)
        m = e16.LR(**hp).fit(Xtr, y, **({} if w is None else {'logisticregression__sample_weight': w}))
    else:
        mk = e20._lgbm(seed, y, cols)
        hp = _inner_cv(Xtr, y, g, mk, [dict(n_estimators=n, num_leaves=l) for n in [100, 300] for l in [2, 4]], seed)
        m = mk(**hp).fit(Xtr, y, sample_weight=w)
    return dict(blocks=blocks, kind=kind, enc=enc, use_h=use_h, encfit=encfit, lo=lo, hi=hi, med=med, cols=cols,
                model=m, hp=hp)


def predict_frozen(fz, D, idx):
    X = assemble(D, idx, fz['blocks'], fz['enc'], fz['use_h'], fz['encfit'])[fz['cols']]
    X = X.clip(fz['lo'], fz['hi'], axis=1).fillna(fz['med']).fillna(0).values
    return fz['model'].predict_proba(X)[:, 1]


def make(D, blocks, aug, kind, enc=None, use_h=True, w_first=None):
    def f(dtr, dte, seed):
        rows = train_rows(D, dtr, aug)
        w = None if w_first is None else np.where(D['first'][rows], w_first, 1.0)
        fz = fit_model(D, rows, blocks, kind, enc, use_h, seed, w)
        return predict_frozen(fz, D, dte.crow.values)
    return f


def candidates(D, prefix='', use_h=True, w_first=None):
    c = {}
    for aug in [False, True]:
        for k in ['lr', 'lgbm']:
            c[f'{prefix}bn{"_aug" if aug else ""}_{k}'] = make(D, 'N', aug, k, use_h=use_h, w_first=w_first)
    for k in ['lr', 'lgbm']:
        c[f'{prefix}bnpm_{k}'] = make(D, 'PM', False, k, use_h=use_h, w_first=w_first)
    for e in ENC:
        for k in ['lr', 'lgbm']:
            c[f'{prefix}bi_{e}_{k}'] = make(D, 'I', False, k, enc=e, use_h=use_h, w_first=w_first)
    for e in ENC:
        c[f'{prefix}ens_{e}'] = (lambda e: lambda dtr, dte, seed: (e16.predict(f'{prefix}bn_aug_lgbm', dtr, dte, seed)
                                                                    + e16.predict(f'{prefix}bi_{e}_lr', dtr, dte, seed)) / 2)(e)
    return c


def run_fp(k):
    return lambda dtr, ctr, dte, cte, seed: e16.predict(k, dtr, dte, seed)


def smoke(coh, D):
    """One outer split, timing and feature counts only (no AUC printed, nothing logged)."""
    import time
    from sklearn.model_selection import StratifiedGroupKFold
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(coh, coh.y, coh.pid))
    dtr, dte = coh.iloc[tr].reset_index(drop=True), coh.iloc[te].reset_index(drop=True)
    for k in sys.argv[2:] or list(CANDS):
        t = time.time()
        p = e16.predict(k, dtr, dte, 2026)
        print(f'{k:20s} ok, {len(p)} predictions, finite {np.isfinite(p).all()}, {time.time() - t:.1f}s', flush=True)


if __name__ == '__main__':
    stage = sys.argv[1]
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    D = build(coh)
    CANDS = candidates(D)
    e16.CANDS.update(CANDS)
    has_fi = D['felc'][:len(coh)] >= 0
    print(f'blocks: N {D["N"].shape[1]}, H {D["H"].shape[1]}, PM {D["PM"].shape[1]}; cohort scans with a numeric '
          f'fellow {int((D["V"].fel.values[:len(coh)] >= 0).sum())}, with a fellow composite {int(has_fi.sum())}',
          flush=True)
    if stage == 'smoke':
        smoke(coh, D)
    elif stage == 'ref':
        evaluate('E22_ref_single_num', e16.select(['num_lr', 'num_lgbm'], 'ref'),
                 'primary comparator: nested selection over {num_lr, num_lgbm}, scanned eye 55 numerics, all 944',
                 df=coh, cnn=cnn)
        pd.DataFrame(e16.CHOICES).to_csv(RES / 'nested22ref_choices.csv', index=False)
    elif stage in ('cands', 'nested'):
        if stage == 'cands':
            for k in CANDS:
                evaluate(f'E22_{k}', run_fp(k), f'E22 candidate {k} (bilateral single visit), all 944', df=coh, cnn=cnn)
        evaluate('E22n_nested', e16.select(list(CANDS), 'E22n'),
                 'HEADLINE: nested inner-CV selection over the 21 E22 candidates, all 944', df=coh, cnn=cnn)
        pd.DataFrame(e16.CHOICES).to_csv(RES / 'nested22_choices.csv', index=False)
    elif stage == 'noH':
        c = candidates(D, prefix='noH_', use_h=False)
        e16.CANDS.update(c)
        evaluate('E22n_noH', e16.select(list(c), 'noH'),
                 'label-free sensitivity: E22n nested selection with block H removed from all 21 candidates',
                 df=coh, cnn=cnn)
        pd.DataFrame(e16.CHOICES).to_csv(RES / 'nested22noH_choices.csv', index=False)
    elif stage == 'neg':
        Dn = build(coh, neg_seed=2026)
        c = {k: f for k, f in candidates(Dn, prefix='neg_').items() if k.startswith('neg_bn_')}
        e16.CANDS.update(c)
        evaluate('E22_negctrl', e16.select(list(c), 'neg'),
                 'negative control: whole fellow (numerics, H) from a random other patient within +-2 y of age; '
                 'nested over the 4 bn candidates', df=coh, cnn=cnn)
        pd.DataFrame(e16.CHOICES).to_csv(RES / 'nested22neg_choices.csv', index=False)
