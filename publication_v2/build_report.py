"""Build publication_v2/report.html: the methods and results note for the final model.

Every number is recomputed from the logged artefacts (kcref/results/experiments.csv and kcref/results/oof/*.npy),
so re-running this script after new work updates the report. Aggregate statistics only: no names, ids, dates of
birth or original filenames. Writes report.html (self-contained, figures embedded), tables/*.csv, figures/*.png.
Usage: python -I publication_v2/build_report.py
"""
import base64
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent
sys.path[:0] = [str(ROOT / 'kcref'), str(OUT)]
import plots as PL  # noqa: E402
import diagrams as DG  # noqa: E402
from common import DATA, RES  # noqa: E402
from evaluate import cluster_boot_auc, paired_delta  # noqa: E402
from radiomics import names as r_names  # noqa: E402
from staging import stage_A, stage_B  # noqa: E402

ARMS = {'noH': 'noH — single visit (primary)', 'H': 'H — visit + fellow-eye treatment history (auxiliary)'}
FINAL = {a: f'E28_{a}_blend' for a in ARMS}
MEMBERS = {'xgb': 'XGBoost', 'lgbm_bag': 'LightGBM (5-seed bag)', 'lr_en': 'Elastic-net logistic'}
REF = {'noH': 'E25_noH_bi_R_lgbm', 'H': 'E25_bi_R_lgbm'}


def oof(name):
    p = np.load(RES / 'oof' / f'{name}.npy')
    return p.mean(0) if p.ndim > 1 else p


def ci(y, p, g):
    if len(np.unique(y)) < 2 or y.sum() < 3:
        return np.nan, np.nan, np.nan
    lo, hi = cluster_boot_auc(y, p, g)
    return roc_auc_score(y, p), lo, hi


def fmt(a, lo, hi):
    return '—' if np.isnan(a) else f'{a:.3f} [{lo:.3f}–{hi:.3f}]'


def age_only(coh):
    """Age-only logistic reference on the same 944 scans and the same 5 x 5 folds (seeds 2026-2030)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    y, g, x = coh.y.values, coh.pid.values, coh[['age']].values
    o = np.zeros((5, len(coh)))
    for r in range(5):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=2026 + r).split(x, y, g):
            o[r, te] = LogisticRegression().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return o.mean(0)


def img(path):
    """Embed a saved PNG (figures/) in the self-contained report."""
    return f'<img alt="{path.stem}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


PENDING = '<div class="box warn">Pending: {what}. Rebuild after the run finishes.</div>'


def e29_sections(coh, y, g, fv, P, pa):
    """HTML for the E29 sections (ladder, SHAP, cornea, stage, robustness, pairs); 'pending' until artefacts exist."""
    out = {}
    have = {f'{r}_{a}': (RES / 'oof' / f'E29_ladder_{r}_{a}.npy').exists() for r in ('L1', 'L2', 'L2b') for a in ARMS}
    if all(have.values()):
        lad, dl = [], []
        for a in ARMS:
            o = {r: oof(f'E29_ladder_{r}_{a}') for r in ('L1', 'L2b', 'L2')}
            o['final'] = P[a]
            for r, p in o.items():
                lad.append(dict(arm=a, rung=r, **dict(zip(('auc', 'lo', 'hi'), ci(y[fv], p[fv], g[fv])))))
            for s0, s1 in (('L1', 'L2'), ('L2', 'final'), ('L1', 'L2b'), ('L2b', 'final')):
                d = paired_delta(y[fv], o[s1][fv], o[s0][fv], g[fv])
                dl.append(dict(arm=a, step=f'{s0}→{s1}', d=d[0], lo=d[1][0], hi=d[1][1], p_gt0=d[2]))
        lad, dl = pd.DataFrame(lad), pd.DataFrame(dl)
        out['ladder'] = img(PL.fig_ladder(lad, dl, roc_auc_score(y[fv], pa[fv]))) + table(dl.round(3))
    else:
        out['ladder'] = PENDING.format(what='ladder runs E29_ladder_{L1,L2,L2b}_{noH,H}')
    if all((RES / 'shap' / f'{a}_oof.npz').exists() for a in ARMS):
        info = PL.cohort_info()
        shs = {a: PL.load_shap(a) for a in ARMS}
        out['shap'] = (img(PL.fig_sources(shs, info)) + img(PL.fig_beeswarm(shs['noH'], info, 'noH')) +
                       img(PL.fig_dependence(shs['noH'], info, 'noH')) +
                       '<details><summary>H arm: top features and dependence</summary>' +
                       img(PL.fig_beeswarm(shs['H'], info, 'H')) + img(PL.fig_dependence(shs['H'], info, 'H')) +
                       '</details>')
        tab = PL.r_table(shs['noH'], info)
        out['cornea'] = (img(PL.fig_cornea(shs['noH'], info, 'noH')) + img(PL.fig_cornea(shs['noH'], info, 'noH', True))
                         + img(PL.fig_mean_maps(info, tab)) +
                         '<details><summary>H arm: corneal importance</summary>' +
                         img(PL.fig_cornea(shs['H'], info, 'H')) + img(PL.fig_cornea(shs['H'], info, 'H', True)) +
                         '</details>')
        out['stage_shap'] = img(PL.fig_stage_shap(shs, info, 'A')) + img(PL.fig_stage_shap(shs, info, 'B'))
        ab = pd.concat([pd.read_csv(OUT / 'tables' / f'shap_top_{a}.csv').assign(arm=a) for a in ARMS])
        ab.to_csv(OUT / 'tables' / 'abcd.csv', index=False)
        out['robust'] = (img(PL.fig_robustness(shs['noH'], info, 'noH')) + img(PL.fig_robustness(shs['H'], info, 'H'))
                         + table(ab[['arm', 'rank', 'feature', 'abcd', 'source', 'mean_abs_shap']].round(4), 'small'))
        out['pairs'] = (table(PL.pair_table(shs['noH'], info)) if have['L1_noH'] else
                        PENDING.format(what='ladder rung L1 (needed for the pair comparison)'))
    else:
        for k in ('shap', 'cornea', 'stage_shap', 'robust', 'pairs'):
            out[k] = PENDING.format(what='OOF SHAP explanation pass (kcref/explain.py shap {noH,H})')
    return out


def table(df, cls=''):
    return df.to_html(index=False, escape=True, border=0, classes=cls, na_rep='—')


def main():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    y, g, fv = coh.y.values, coh.pid.values, coh.first_visit.values
    log = pd.read_csv(RES / 'experiments.csv').drop_duplicates('name', keep='last').set_index('name')
    (OUT / 'tables').mkdir(exist_ok=True)
    P = {a: oof(FINAL[a]) for a in ARMS}
    pa = age_only(coh)

    # ---- headline
    rows = []
    for a in ARMS:
        r = log.loc[FINAL[a]]
        rows.append({'arm': ARMS[a], 'model': FINAL[a], 'scans': int(r.n), 'positive scans': int(r.n_pos),
                     'AUC, all scans [95% CI]': fmt(r.auc, r.auc_lo, r.auc_hi),
                     'AUC, first visit [95% CI]': fmt(r.auc_first, r.auc_first_lo, r.auc_first_hi),
                     'per-repeat AUC (mean ± SD)': f'{r.auc_rep_mean:.3f} ± {r.auc_rep_sd:.3f}'})
    a0 = ci(y, pa, g)
    a1 = ci(y[fv], pa[fv], g[fv])
    rows.append({'arm': 'reference', 'model': 'age only (logistic, same folds)', 'scans': len(y),
                 'positive scans': int(y.sum()), 'AUC, all scans [95% CI]': fmt(*a0),
                 'AUC, first visit [95% CI]': fmt(*a1), 'per-repeat AUC (mean ± SD)': '—'})
    head = pd.DataFrame(rows)
    head.to_csv(OUT / 'tables' / 'headline.csv', index=False)

    # ---- paired deltas
    rows = []
    for a in ARMS:
        for lab, q in [('age only', pa), ('single-scan numerics reference', oof('E22_ref_single_num')),
                       ('LightGBM member alone', oof(REF[a]))]:
            d0 = paired_delta(y, P[a], q, g)
            d1 = paired_delta(y[fv], P[a][fv], q[fv], g[fv])
            rows.append({'arm': a, 'final blend minus': lab,
                         'ΔAUC all [95% CI]': f'{d0[0]:+.3f} [{d0[1][0]:+.3f}, {d0[1][1]:+.3f}]',
                         'ΔAUC first visit [95% CI]': f'{d1[0]:+.3f} [{d1[1][0]:+.3f}, {d1[1][1]:+.3f}]',
                         'P(Δ>0), first visit': f'{d1[2]:.2f}'})
    deltas = pd.DataFrame(rows)
    deltas.to_csv(OUT / 'tables' / 'paired_deltas.csv', index=False)

    # ---- members
    rows = []
    for a in ARMS:
        for k, lab in MEMBERS.items():
            r = log.loc[f'E28_{a}_{k}']
            rows.append({'arm': a, 'member': lab, 'AUC all': fmt(r.auc, r.auc_lo, r.auc_hi),
                         'AUC first visit': fmt(r.auc_first, r.auc_first_lo, r.auc_first_hi)})
    members = pd.DataFrame(rows)
    members.to_csv(OUT / 'tables' / 'members.csv', index=False)

    # ---- strata: stages A / B, age
    strata = {'A (Km F, D)': (stage_A(coh), {1: '< 48', 2: '48–53', 3: '53–55', 4: '≥ 55'}),
              'B (Rm B, mm)': (stage_B(coh), {1: '≥ 5.70', 2: '5.15–5.70', 3: '4.95–5.15', 4: '< 4.95'})}
    ag = np.select([coh.age < 25, coh.age < 30], [1, 2], 3)
    strata['age (years)'] = (ag, {1: '< 25', 2: '25–29', 3: '≥ 30'})
    rows = []
    for sname, (s, lab) in strata.items():
        for k, l in lab.items():
            for scope, m0 in [('all scans', np.ones(len(y), bool)), ('first visit', fv)]:
                m = (s == k) & m0
                row = {'stratification': sname, 'stratum': f'{k}: {l}', 'scans': scope, 'n': int(m.sum()),
                       'positive': int(y[m].sum())}
                for a in ARMS:
                    row[f'AUC {a} [95% CI]'] = fmt(*ci(y[m], P[a][m], g[m]))
                row['AUC age only'] = fmt(*ci(y[m], pa[m], g[m]))
                rows.append(row)
    st = pd.DataFrame(rows)
    st.to_csv(OUT / 'tables' / 'strata.csv', index=False)

    # ---- figures (E29 style)
    l1 = {a: oof(f'E29_ladder_L1_{a}') if (RES / 'oof' / f'E29_ladder_L1_{a}.npy').exists() else None for a in ARMS}
    curves = {'Final, noH': (P['noH'], PL.S.C['noH'], '-'), 'Final, H': (P['H'], PL.S.C['H'], '-')}
    if l1['noH'] is not None:
        curves['Single-eye numerics (L1)'] = (l1['noH'], PL.S.C['L1'], (0, (4, 2)))
    curves['Age only'] = (pa, PL.S.C['age'], (0, (1, 1.5)))
    roc_img = img(PL.fig_roc(y, g, fv, curves))
    stage_imgs = {}
    for k, s_ in (('A', stage_A(coh)), ('B', stage_B(coh))):
        sc = {'Age only': pa[fv], 'Final, noH': P['noH'][fv], 'Final, H': P['H'][fv]}
        if l1['noH'] is not None:
            sc['Single-eye numerics (L1)'] = l1['noH'][fv]
        df = PL.stage_auc(y[fv], sc, s_[fv], g[fv])
        if l1['noH'] is None:
            df = pd.concat([df, pd.DataFrame([dict(method='Single-eye numerics (L1)', stage=j, n=int((s_[fv] == j).sum()),
                                                   pos=int(y[fv][s_[fv] == j].sum()), auc=np.nan, lo=np.nan,
                                                   hi=np.nan, small=True) for j in (1, 2, 3, 4)])])
        df.to_csv(OUT / 'tables' / f'stage_auc_{k}.csv', index=False)
        stage_imgs[k] = img(PL.fig_stage_auc(df, k))
    strata_img = stage_imgs['A'] + stage_imgs['B']

    # ---- R feature dictionary
    desc = {'r': 'mean of the map on ring {a}–{b} mm, quadrant {q}', 'ext_val': 'extremum value',
            'ext_x': 'extremum x (mm, nasal +)', 'ext_y': 'extremum y (mm, superior +)',
            'ext_r': 'extremum distance from map centre (mm)', 'IminusS': 'inferior − superior mean, 2–4 mm annulus',
            'NminusT': 'nasal − temporal mean, 2–4 mm annulus', 'sd3': 'standard deviation within 3 mm'}
    mp = {'T': 'thickness (µm)', 'K': 'sagittal front curvature (D)', 'EF': 'front elevation vs BFS (µm)',
          'EB': 'back elevation vs BFS (µm)'}
    rows = []
    for n in r_names():
        m, rest = n.split('_', 1)
        if n.startswith('T_prog') or n == 'T_ratio31':
            d = ('ratio (3 mm − thinnest) / (1 mm − thinnest)' if n == 'T_ratio31' else
                 f'mean thickness on a {n[-1]} mm ring (±0.25 mm) around the thinnest point minus the thinnest value')
            rows.append({'feature': n, 'map': mp['T'], 'definition': d})
            continue
        if rest.startswith('r') and rest[1:3].isdigit():
            a, b, q = rest[1], rest[2], rest.split('_')[1]
            d = desc['r'].format(a=a, b=b, q={'S': 'superior', 'I': 'inferior', 'N': 'nasal', 'T': 'temporal'}[q])
        else:
            d = desc[rest]
            if rest.startswith('ext'):
                d = d.replace('extremum', 'thinnest point' if m == 'T' else 'maximum')
        rows.append({'feature': n, 'map': mp[m], 'definition': d})
    rdict = pd.DataFrame(rows)
    rdict.to_csv(OUT / 'tables' / 'r_features.csv', index=False)

    e29 = e29_sections(coh, y, g, fv, P, pa)
    n_first, pos_first = int(fv.sum()), int(y[fv].sum())
    body = TEMPLATE.format(
        date=dt.date.today().isoformat(), n=len(y), pos=int(y.sum()), n_pat=coh.pid.nunique(),
        n_eye=coh.eid.nunique(), n_first=n_first, pos_first=pos_first,
        age_mean=f'{coh.age.mean():.1f}', age_sd=f'{coh.age.std():.1f}', age_min=int(coh.age.min()),
        age_max=int(coh.age.max()),
        head=table(head, 'hl'), deltas=table(deltas), members=table(members), strata=table(st, 'small'),
        roc=roc_img, strata_img=strata_img, flow_img=img(DG.fig_dataflow()), filter_img=img(DG.fig_filtering()), rdict=table(rdict, 'small'), **e29,
        noH_first=f'{log.loc[FINAL["noH"]].auc_first:.3f}', H_first=f'{log.loc[FINAL["H"]].auc_first:.3f}')
    (OUT / 'report.html').write_text(body, encoding='utf-8')
    print('wrote', OUT / 'report.html')


TEMPLATE = open(Path(__file__).parent / 'report_template.html', encoding='utf-8').read() if (
    Path(__file__).parent / 'report_template.html').exists() else ''

if __name__ == '__main__':
    sys.exit(main())
