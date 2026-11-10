"""E22 analyses (design 6.4-6.5) on the logged OOF predictions. Writes rethink/results/analysis_e22.txt plus
analysis_e22_*.csv. No model is selected here; everything reads OOF files saved by harness.evaluate, except the
descriptive feature importance (section 8), which refits the numeric bilateral model on all 944 scans.

Thresholds / recalibration "in fold": for repeat r and outer fold k (the harness splits, regenerated with the
same seeds), the Youden threshold and the Platt map are fitted on the OOF predictions of the other four folds of
that repeat and applied to fold k; metrics are pooled over folds and averaged over the 5 repeats.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from compare import paired_delta  # noqa: E402
from harness import DATA, N_REPEATS, N_SPLITS, RES, SEED, cluster_boot_auc  # noqa: E402

OUT = []


def log(s=''):
    print(s, flush=True)
    OUT.append(str(s))


def oof(name):
    return np.load(RES / 'oof' / f'{name}.npy')


def auc_ci(y, p, g):
    if y.min() == y.max():
        return np.nan, np.nan, np.nan
    lo, hi = cluster_boot_auc(y, p, g)
    return roc_auc_score(y, p), lo, hi


def folds(coh):
    y, g = coh.y.values, coh.pid.values
    out = []
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + r)
        out.append([te for _, te in cv.split(coh, y, g)])
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def infold(coh, O, F):
    """Per repeat: Platt-calibrated OOF and Youden-threshold decisions, fitted on the other folds."""
    y = coh.y.values
    cal = np.zeros_like(O)
    dec = np.zeros_like(O, dtype=bool)
    for r in range(O.shape[0]):
        for te in F[r]:
            tr = np.setdiff1d(np.arange(len(y)), te)
            pl = LogisticRegression(C=1e6, max_iter=1000).fit(logit(O[r, tr])[:, None], y[tr])
            cal[r, te] = pl.predict_proba(logit(O[r, te])[:, None])[:, 1]
            ths = np.unique(O[r, tr])
            j = [(O[r, tr][y[tr] == 1] >= t).mean() + (O[r, tr][y[tr] == 0] < t).mean() for t in ths]
            dec[r, te] = O[r, te] >= ths[int(np.argmax(j))]
    return cal, dec


def op_metrics(y, dec):
    rows = []
    for d in dec:
        tp, fp = (d & (y == 1)).sum(), (d & (y == 0)).sum()
        fn, tn = (~d & (y == 1)).sum(), (~d & (y == 0)).sum()
        rows.append(dict(sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1)))
    return pd.DataFrame(rows).mean()


def calib(y, p):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
    return dict(slope=lr.coef_[0, 0], intercept=lr.intercept_[0], brier=np.mean((p - y) ** 2), mean_pred=p.mean())


def net_benefit(y, p, t):
    n = len(y)
    tp, fp = ((p >= t) & (y == 1)).sum(), ((p >= t) & (y == 0)).sum()
    return tp / n - fp / n * t / (1 - t)


def main():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    y, g = coh.y.values, coh.pid.values
    exp = pd.read_csv(RES / 'experiments.csv')
    names = [n for n in exp.name.unique() if n.startswith('E22_') and n not in ('E22_ref_single_num', 'E22_negctrl')]
    ref, age = oof('E22_ref_single_num').mean(0), oof('E07_age_all').mean(0)
    head = oof('E22n_nested')

    log('# E22 analyses (all 944 scans, pooled OOF over 5 repeats)\n')
    log('## 1. Headline and every candidate')
    rows = []
    for n in ['E22n_nested', 'E22n_noH', 'E22_negctrl', 'E22_ref_single_num', 'E07_age_all'] + names:
        if not (RES / 'oof' / f'{n}.npy').exists():
            continue
        p = oof(n).mean(0)
        a, lo, hi = auc_ci(y, p, g)
        fv = coh.first_visit.values
        r = dict(run=n, auc=a, lo=lo, hi=hi, first=roc_auc_score(y[fv], p[fv]))
        for tag, q in [('ref', ref), ('age', age)]:
            seeds = [0, 1, 2] if n == 'E22n_nested' else [0]
            ds = [paired_delta(y, p, q, g, seed=s) for s in seeds]
            r[f'd_{tag}'] = ds[0][0]
            r[f'd_{tag}_ci'] = ' / '.join(f'[{d[1][0]:+.3f}, {d[1][1]:+.3f}]' for d in ds)
            r[f'P_{tag}'] = ds[0][2]
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(RES / 'analysis_e22_candidates.csv', index=False)
    log(T.round(3).to_string(index=False))
    ch = pd.read_csv(RES / 'nested22_choices.csv')
    log('\nE22n choices (25 outer folds): ' + str(ch[ch.pool == 'E22n'].chosen.value_counts().to_dict()))
    for f, tag in [('nested22noH_choices.csv', 'noH'), ('nested22neg_choices.csv', 'neg'), ('nested22ref_choices.csv', 'ref')]:
        if (RES / f).exists():
            c = pd.read_csv(RES / f)
            log(f'{tag} choices: ' + str(c[c.pool == tag].chosen.value_counts().to_dict()))
    rep = [roc_auc_score(y, o) for o in head]
    log(f'E22n per-repeat AUC {np.mean(rep):.3f} +- {np.std(rep):.3f}')
    F = folds(coh)
    within = [roc_auc_score(y[te], head[r, te]) for r in range(N_REPEATS) for te in F[r]]
    within_ref = [roc_auc_score(y[te], oof('E22_ref_single_num')[r, te]) for r in range(N_REPEATS) for te in F[r]]
    log(f'E22n mean within-fold AUC {np.mean(within):.3f}; ref {np.mean(within_ref):.3f}; '
        f'folds where E22n > ref: {int(np.sum(np.array(within) > np.array(within_ref)))}/25')

    models = {'E22n': head.mean(0), 'ref_single_num': ref, 'age': age}
    log('\n## 2. Severity: KMax of the scanned eye (original bins + > 55 D)')
    km = coh['KMax Sagittal Front (D)'].values
    stage = pd.cut(km, [0, 46.5, 48, 53, 55, np.inf], labels=['S0 <=46.5', 'S1 46.5-48', 'S2 48-53', 'S3 53-55', 'S4 >55'],
                   include_lowest=True)
    rows = []
    for s in stage.categories:
        m = (stage == s)
        r = dict(stage=s, n=int(m.sum()), pos=int(y[m].sum()))
        for k, p in models.items():
            a, lo, hi = auc_ci(y[m], p[m], g[m])
            r[k] = f'{a:.3f} [{lo:.3f}-{hi:.3f}]'
        if 0 < y[m].sum() < m.sum():
            d = paired_delta(y[m], models['E22n'][m], ref[m], g[m])
            r['dAUC vs ref'] = f'{d[0]:+.3f} [{d[1][0]:+.3f}, {d[1][1]:+.3f}]'
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(RES / 'analysis_e22_severity.csv', index=False)
    log(S.to_string(index=False))
    # interaction: gain in early (KMax <= 48) vs later disease
    early = km <= 48
    rng = np.random.default_rng(0)
    ug = np.unique(g)
    ix = {q: np.flatnonzero(g == q) for q in ug}
    diffs = []
    for _ in range(1000):
        idx = np.concatenate([ix[q] for q in rng.choice(ug, len(ug))])
        e, l_ = idx[early[idx]], idx[~early[idx]]
        if min(y[e].sum(), y[l_].sum()) == 0 or min((1 - y[e]).sum(), (1 - y[l_]).sum()) == 0:
            continue
        de = roc_auc_score(y[e], models['E22n'][e]) - roc_auc_score(y[e], ref[e])
        dl = roc_auc_score(y[l_], models['E22n'][l_]) - roc_auc_score(y[l_], ref[l_])
        diffs.append(de - dl)
    log(f'stage x model interaction: (gain at KMax <= 48) - (gain at KMax > 48) = {np.mean(diffs):+.3f} '
        f'[{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}] (n early {early.sum()}, pos {y[early].sum()})')

    log('\n## 3. Fellow strata (mutually exclusive, in this order)')
    import exp22_bilateral as e22
    D = e22.build(coh)
    n = len(coh)
    treated = D['H'].fellow_cxl_before_visit.values[:n] == 1
    fimg = D['flags'].has_fellow_image.values[:n] == 1
    fnum = D['V'].fel.values[:n] >= 0
    st = np.where(treated, 'fellow treated before visit', np.where(fimg, 'fellow composite (image)',
                  np.where(fnum, 'fellow numerics only (raw)', 'no fellow')))
    rows = []
    for s in ['fellow composite (image)', 'fellow numerics only (raw)', 'fellow treated before visit', 'no fellow']:
        m = st == s
        r = dict(stratum=s, n=int(m.sum()), pos=int(y[m].sum()))
        for k, p in list(models.items()) + [('noH', oof('E22n_noH').mean(0))]:
            a, lo, hi = auc_ci(y[m], p[m], g[m])
            r[k] = f'{a:.3f} [{lo:.3f}-{hi:.3f}]' if np.isfinite(a) else 'n/a'
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(RES / 'analysis_e22_fellow_strata.csv', index=False)
    log(S.to_string(index=False))
    log(f'has_fellow_image alone: AUC {roc_auc_score(y, fimg):.3f}; fellow treated alone: {roc_auc_score(y, treated):.3f}')

    log('\n## 4. Label-free sensitivity and negative control')
    for nm in ['E22n_noH', 'E22_negctrl']:
        p = oof(nm).mean(0)
        a, lo, hi = auc_ci(y, p, g)
        d1, d2 = paired_delta(y, p, ref, g), paired_delta(y, head.mean(0), p, g)
        log(f'{nm}: AUC {a:.3f} [{lo:.3f}-{hi:.3f}] | vs ref {d1[0]:+.3f} [{d1[1][0]:+.3f}, {d1[1][1]:+.3f}] | '
            f'E22n minus it {d2[0]:+.3f} [{d2[1][0]:+.3f}, {d2[1][1]:+.3f}]')

    log('\n## 5. Age strata and first visit')
    ag = coh.age.values
    rows = []
    for lab, m in [('<25', ag < 25), ('25-29', (ag >= 25) & (ag < 30)), ('>=30', ag >= 30), ('first visit', coh.first_visit.values)]:
        r = dict(stratum=lab, n=int(m.sum()), pos=int(y[m].sum()))
        for k, p in models.items():
            a, lo, hi = auc_ci(y[m], p[m], g[m])
            r[k] = f'{a:.3f} [{lo:.3f}-{hi:.3f}]'
        d = paired_delta(y[m], models['E22n'][m], ref[m], g[m])
        r['dAUC vs ref'] = f'{d[0]:+.3f} [{d[1][0]:+.3f}, {d[1][1]:+.3f}]'
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(RES / 'analysis_e22_age_first.csv', index=False)
    log(S.to_string(index=False))

    log('\n## 6. Operating point (Youden, in fold), calibration (raw and Platt in fold), decision curve')
    rows, dca = [], []
    for k, nm in [('E22n', 'E22n_nested'), ('ref_single_num', 'E22_ref_single_num'), ('age', 'E07_age_all')]:
        O = oof(nm)
        cal, dec = infold(coh, O, F)
        r = dict(model=k, **op_metrics(y, dec).round(3).to_dict())
        c0, c1 = calib(y, O.mean(0)), calib(y, cal.mean(0))
        r.update({f'raw_{a}': round(b, 3) for a, b in c0.items()})
        r.update({f'platt_{a}': round(b, 3) for a, b in c1.items()})
        rows.append(r)
        for t in np.arange(0.05, 0.51, 0.05):
            dca.append(dict(model=k, threshold=round(t, 2), net_benefit=np.mean([net_benefit(y, c, t) for c in cal])))
    for t in np.arange(0.05, 0.51, 0.05):
        dca.append(dict(model='treat all', threshold=round(t, 2), net_benefit=net_benefit(y, np.ones_like(age), t)))
    S = pd.DataFrame(rows)
    S.to_csv(RES / 'analysis_e22_operating_calibration.csv', index=False)
    log(S.T.to_string())
    Dc = pd.DataFrame(dca).pivot(index='threshold', columns='model', values='net_benefit')
    Dc.to_csv(RES / 'analysis_e22_dca.csv')
    log('\nnet benefit (Platt-calibrated in fold; treat none = 0):')
    log(Dc.round(4).to_string())

    log('\n## 7. Leakage checks: each new block on its own (univariate, descriptive)')
    H = D['H'].iloc[:n]
    for c in H.columns:
        if H[c].std() > 0:
            log(f'H {c}: AUC {roc_auc_score(y, H[c].fillna(0)):.3f}')
    meta_rows = D['own'][:n]
    for e in e22.ENC:
        E = e22.emb(e)
        own = E[meta_rows]
        felc = D['felc'][:n]
        ok = np.isfinite(own).all(1)
        a_own = [abs(roc_auc_score(y[ok], own[ok, j]) - .5) for j in range(own.shape[1]) if own[ok, j].std() > 0]
        m = (felc >= 0)
        dif = own[m] - E[felc[m]]
        okd = np.isfinite(dif).all(1)
        a_dif = [abs(roc_auc_score(y[m][okd], dif[okd, j]) - .5) for j in range(dif.shape[1]) if dif[okd, j].std() > 0]
        log(f'I_{e}: max univariate |AUC-0.5| own {max(a_own):.3f} (median {np.median(a_own):.3f}), '
            f'own-fellow {max(a_dif):.3f} (median {np.median(a_dif):.3f}), dims {own.shape[1]}')

    log('\n## 8. Interpretation (descriptive; numeric bilateral LGBM bn_lgbm fitted on all 944)')
    from lightgbm import LGBMClassifier  # noqa: F401
    import exp20_bilateral as e20
    from exp01_tabular import _inner_cv
    X = pd.concat([D['N'].iloc[:n].reset_index(drop=True), D['H'].iloc[:n].reset_index(drop=True)], axis=1)
    lo, hi = X.quantile(0.005), X.quantile(0.995)
    X = X.clip(lo, hi, axis=1)
    X = X.fillna(X.median()).fillna(0)
    mk = e20._lgbm(2026, y, list(X.columns))
    hp = _inner_cv(X.values, y, g, mk, [dict(n_estimators=a, num_leaves=b) for a in [100, 300] for b in [2, 4]], 2026)
    m = mk(**hp).fit(X.values, y)
    shap = m.booster_.predict(X.values, pred_contrib=True)
    imp = pd.Series(np.abs(shap[:, :-1]).mean(0), index=X.columns).sort_values(ascending=False)
    fam = imp.groupby(lambda c: 'H (fellow treatment)' if c in e22.H_COLS else c.split('_')[0] if c.split('_')[0] in
                      ('fel', 'asym', 'worse') else 'age' if c == 'age' else 'own').sum().sort_values(ascending=False)
    log(f'hyper-parameters {hp}; mean |SHAP| by family: ' + ', '.join(f'{k} {v:.3f}' for k, v in fam.items()))
    log('top 15 features by mean |SHAP|:')
    log(imp.head(15).round(4).to_string())
    imp.to_csv(RES / 'analysis_e22_shap.csv')
    (RES / 'analysis_e22.txt').write_text('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    sys.exit(main())
