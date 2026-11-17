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
import age_ops as AO  # noqa: E402
import atlas as AT  # noqa: E402
import howto as HT  # noqa: E402
from common import DATA, RES  # noqa: E402
from evaluate import cluster_boot_auc, paired_delta  # noqa: E402
from radiomics import names as r_names  # noqa: E402
from staging import stage_A, stage_B  # noqa: E402

ARMS = {'noH': 'noH — single visit (primary)', 'H': 'H — visit + fellow-eye treatment history (auxiliary)'}
FINAL = {a: f'E31_noeng_{a}' for a in ARMS}  # E31 adoption: no derived indices
MEMBERS = {'xgb': 'XGBoost', 'lgbm_bag': 'LightGBM (5-seed bag)', 'lr_en': 'Elastic-net logistic'}


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


def img(path, howto=True):
    """Embed a saved PNG (figures/) in the self-contained report, followed by its 'how to read it' note
    (howto.py); already-built HTML passes through."""
    if isinstance(path, str) and path.lstrip().startswith('<'):
        return path
    path = Path(path)
    return (f'<img alt="{path.stem}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'
            + (HT.box(path.stem) if howto else ''))


PENDING = '<div class="box warn">Pending: {what}. Rebuild after the run finishes.</div>'


def e29_sections(coh, y, g, fv, P, pa):
    """HTML for the E29 sections (ladder, SHAP, cornea, stage, robustness, pairs); 'pending' until artefacts exist."""
    out = {}
    have = {f'{r}_{a}': (RES / 'oof' / f'E31_ladder_{r}_{a}.npy').exists() for r in ('L1', 'L2', 'L2b') for a in ARMS}
    if all(have.values()):
        lad, dl = [], []
        for a in ARMS:
            o = {r: oof(f'E31_ladder_{r}_{a}') for r in ('L1', 'L2b', 'L2')}
            o['final'] = P[a]
            for r, p in o.items():
                lad.append(dict(arm=a, rung=r, **dict(zip(('auc', 'lo', 'hi'), ci(y[fv], p[fv], g[fv])))))
            for s0, s1 in (('L1', 'L2'), ('L2', 'final'), ('L1', 'L2b'), ('L2b', 'final')):
                d = paired_delta(y[fv], o[s1][fv], o[s0][fv], g[fv])
                dl.append(dict(arm=a, step=f'{s0}→{s1}', d=d[0], lo=d[1][0], hi=d[1][1], p_gt0=d[2]))
        lad, dl = pd.DataFrame(lad), pd.DataFrame(dl)
        out['ladder'] = img(PL.fig_ladder(lad, dl, roc_auc_score(y[fv], pa[fv]))) + table(dl.round(3))
    else:
        out['ladder'] = PENDING.format(what='ladder runs E31_ladder_{L1,L2,L2b}_{noH,H}')
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
        out['pairs'] = (img(PL.fig_pairs(shs, info)) + table(PL.pair_table(shs, info)) if have['L1_noH'] and have['L1_H'] else
                        PENDING.format(what='ladder rung L1 (needed for the pair comparison)'))
    else:
        for k in ('shap', 'cornea', 'stage_shap', 'robust', 'pairs'):
            out[k] = PENDING.format(what='OOF SHAP explanation pass (kcref/explain.py shap {noH,H})')
    return out


def atlas_sections():
    """Appendix C: clinical reading atlas (aggregate figures from atlas.py); individual eyes stay local (cases.py)."""
    if not all((RES / 'shap' / f'{a}_oof.npz').exists() for a in ARMS):
        return {'atlas': PENDING.format(what='OOF SHAP explanation pass')}
    info = PL.cohort_info()
    shs = {a: PL.load_shap(a) for a in ARMS}
    figs = {a: dict(read=AT.fig_reading_map(shs[a], info, a), gal=AT.fig_r_gallery(shs[a], info, a),
                    **{by: AT.fig_atlas(shs[a], info, a, by) for by in ('label', 'stageA', 'stageB', 'age')})
            for a in ARMS}
    rm = pd.read_csv(PL.S.tab('reading_map_noH.csv'))
    rg = pd.read_csv(PL.S.tab('reading_guide_noH.csv'))
    top3 = '; '.join(f'{r.map.lower()} {r.location} (mostly {r.leading_side}, {r.direction})'
                     for r in rm.head(3).itertuples())
    by_map = rm.groupby('map').importance.sum().sort_values(ascending=False)
    tab_read = rm[['rank', 'map', 'location', 'leading_side', 'direction', 'rho_leading', 'share_own', 'share_fel',
                   'share_asym']].rename(columns={'share_own': 'share own', 'share_fel': 'share fellow',
                                                  'share_asym': 'share own − fellow', 'rho_leading': 'ρ'})
    tab_guide = rg[['rank', 'feature', 'side', 'direction', 'trend_zero_crossing', 'median_referred',
                    'median_not_referred', 'n_eyes']].rename(columns={
                        'trend_zero_crossing': 'SHAP trend crosses 0 at', 'median_referred': 'median, referred',
                        'median_not_referred': 'median, not referred'})
    det = lambda title, *ps: f'<details><summary>{title}</summary>' + ''.join(img(q) for q in ps) + '</details>'
    html = f"""<div class="box"><b>How to read this appendix.</b> The figures answer two clinical questions about the final
model: <i>where on the cornea</i> its decision is made, and <i>which direction</i> of each corneal measurement raises the
referral score. All figures pool first-visit eyes (no individual eye is shown). Importance = mean |SHAP| of the region,
summed over the own eye, the fellow eye and the own − fellow difference. Superior is up and nasal is right in every
dartboard (left eyes mirrored). <b>Short version (noH):</b> look first at {top3}. By map, the ten leading sectors are
{", ".join(f"{m.lower()} {100 * v / by_map.sum():.0f}%" for m, v in by_map.items())} of their combined importance.</div>
<h3>C.1 Where to look: the ten leading corneal sectors</h3>
{img(figs['noH']['read'])}{table(tab_read.round(2), 'small')}{det('H arm: leading sectors', figs['H']['read'])}
<h3>C.2 The twelve leading corneal measurements, one by one</h3>
<p>Each panel plots the measurement (x, in the map unit) against its SHAP value (y; above 0 raises the referral score),
for referred (navy) and not-referred (grey) first-visit eyes. The black line is the rolling median; the dashed line marks
where it crosses 0, i.e. the value at which this measurement turns from lowering to raising the score. These reading
points are descriptive: they are learned on this cohort, interact with the other features and are not validated
clinical thresholds.</p>
{img(figs['noH']['gal'])}{table(tab_guide.round(3), 'small')}{det('H arm: twelve leading measurements', figs['H']['gal'])}
<h3>C.3 Does the place to look change with the patient?</h3>
<p>Colour is each sector's share of its group's total sector importance, so groups whose SHAP magnitudes differ (advanced
stages carry much more) read on one scale; the group total is printed beside each row.</p>
<h4>By referral status</h4>{img(figs['noH']['label'])}
<h4>By stage A (Km F)</h4>{img(figs['noH']['stageA'])}
<h4>By stage B (Rm B)</h4>{img(figs['noH']['stageB'])}
<h4>By age group (§21)</h4>{img(figs['noH']['age'])}
{det('H arm: atlas by referral status, stage A, stage B and age', *(figs['H'][k] for k in ('label', 'stageA', 'stageB', 'age')))}
<h3>C.4 What referred corneas look like: all four maps</h3>
<p>Mean map of referred and not-referred first-visit eyes and their difference, with the top 3 own-eye sectors of each map
outlined.</p>
{img(PL.fig_mean_maps(info, PL.r_table(shs['noH'], info), maps=('T', 'K', 'EF', 'EB'), name='cornea_mean_maps_all'))}
<h3>C.5 Individual eyes</h3>
<p>A gallery of 12 individual eyes (correct referrals, correct non-referrals, false positives and false negatives) with
their maps, the sectors that moved their score and their per-family contributions is built by <code>cases.py</code> into
<code>publication_v2/local/report_cases.html</code>. It stays on the local machine (individual maps are not part of the
repository; user decision 2026-11-13).</p>"""
    return {'atlas': html}


def allscans_sections(coh, y, g, fv, P):
    """Appendix D: every explanation figure recomputed over all 944 cohort scans (all visits) instead of first-visit
    eyes, plus AUC by visit scope and a first-visit vs all-scans importance comparison. Same OOF SHAP file."""
    if not all((RES / 'shap' / f'{a}_oof.npz').exists() for a in ARMS):
        return {'allscans': PENDING.format(what='OOF SHAP explanation pass')}
    info = PL.cohort_info()
    info['age_group'] = AO.age_group(info['age'])
    shs = {a: PL.load_shap(a) for a in ARMS}
    ia = PL.all_scans(info)
    fu = ~fv
    rows = []
    for a in ARMS:
        for lab, m in (('first visit (one per eye)', fv), ('follow-up visits only', fu), ('all scans', np.ones(len(y), bool))):
            rows.append({'arm': a, 'scans': lab, 'n': int(m.sum()), 'positive': int(y[m].sum()),
                         'patients': len(set(g[m])), 'AUC [95% CI]': fmt(*ci(y[m], P[a][m], g[m]))})
    auc = pd.DataFrame(rows)
    auc.to_csv(OUT / 'tables' / 'auc_by_scope.csv', index=False)
    cmp_rows, rho = [], {}
    for a in ARMS:
        i1, i2 = PL.col_importance(shs[a], fv), PL.col_importance(shs[a], np.ones(len(y), bool))
        rho[a] = pd.Series(i1).corr(pd.Series(i2), method='spearman')
        r1, r2 = (-i1).argsort().argsort() + 1, (-i2).argsort().argsort() + 1
        for j in sorted(set(np.argsort(-i1)[:15]) | set(np.argsort(-i2)[:15]), key=lambda j: r2[j]):
            cmp_rows.append({'arm': a, 'feature': PL.E.label(shs[a]['cols'][j]), 'rank, all scans': int(r2[j]),
                             'rank, first visit': int(r1[j]), 'mean |SHAP|, all scans': i2[j],
                             'mean |SHAP|, first visit': i1[j]})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(OUT / 'tables' / 'shap_scope_compare.csv', index=False)
    det = lambda title, *ps: f'<details><summary>{title}</summary>' + ''.join(img(q) for q in ps) + '</details>'
    with PL.S.all_scans():
        src = img(PL.fig_sources(shs, ia))
        bee = {a: (PL.fig_beeswarm(shs[a], ia, a), PL.fig_dependence(shs[a], ia, a)) for a in ARMS}
        cor = {a: (PL.fig_cornea(shs[a], ia, a), PL.fig_cornea(shs[a], ia, a, True)) for a in ARMS}
        mm = PL.fig_mean_maps(ia, PL.r_table(shs['noH'], ia), maps=('T', 'K', 'EF', 'EB'), name='cornea_mean_maps_all')
        stg = [PL.fig_stage_shap(shs, ia, k) for k in ('A', 'B', 'age_group')]
        age = AO.fig_age_shap(shs, ia)
        atl = {a: dict(read=AT.fig_reading_map(shs[a], ia, a), gal=AT.fig_r_gallery(shs[a], ia, a),
                       **{by: AT.fig_atlas(shs[a], ia, a, by) for by in ('label', 'stageA', 'stageB', 'age')})
               for a in ARMS}
        rm = pd.read_csv(PL.S.tab('reading_map_noH.csv'))
    html = f"""<div class="box">The main text explains the model on <b>first-visit eyes</b> (one scan per eye, {int(fv.sum())} eyes), the
pre-declared primary population. This appendix recomputes <b>every explanation figure over all {len(y)} cohort scans</b>
(all visits; {int(fu.sum())} are follow-up scans of eyes already in the first-visit set). It uses the same out-of-fold SHAP
values (each scan explained by the fit that did not train on its patient); only the rows averaged change. Eyes with many
visits weigh more here, and follow-up scans of the same eye are correlated, so CIs on all scans are patient-cluster
bootstrap. Spearman ρ between the per-column importances of the two scopes: noH {rho['noH']:.3f}, H {rho['H']:.3f}.
Robustness across fits (§19) and matched pairs (§20) are defined on first-visit eyes only and are not repeated.</div>
<h3>D.1 AUC by visit scope</h3>{table(auc)}
<h3>D.2 Feature ranks: all scans vs first visit</h3>
<p>Union of the top 15 features of each scope, ordered by the all-scans rank.</p>{table(cmp.round(4), 'small')}
<h3>D.3 Information sources</h3>{img(src)}
<h3>D.4 Top features and dependence (noH)</h3>{img(bee['noH'][0])}{img(bee['noH'][1])}
{det('H arm: top features and dependence, all scans', *bee['H'])}
<h3>D.5 Where on the cornea</h3>{img(cor['noH'][0])}{img(cor['noH'][1])}{img(mm)}
{det('H arm: corneal importance, all scans', *cor['H'])}
<h3>D.6 Explanations by stage and age</h3>{''.join(img(q) for q in stg)}{img(age)}
<h3>D.7 Clinical reading atlas, all scans</h3>
<p>Short version (noH, all scans): look first at {'; '.join(f'{r.map.lower()} {r.location} (mostly {r.leading_side})' for r in rm.head(3).itertuples())}.</p>
{img(atl['noH']['read'])}{img(atl['noH']['gal'])}
{''.join(img(atl['noH'][k]) for k in ('label', 'stageA', 'stageB', 'age'))}
{det('H arm: reading atlas, all scans', *(atl['H'][k] for k in ('read', 'gal', 'label', 'stageA', 'stageB', 'age')))}"""
    return {'allscans': html}


def exec_summary(coh, y, g, fv, P, head, deltas, key_findings):
    """Executive summary (opens the report): the best figures and numbers of this build, compact. Every figure and
    table here is also in the full report; nothing is computed only for the summary except the short texts."""
    F = lambda n, h=True: img(OUT / 'figures' / f'{n}.png', h)
    T = lambda n: pd.read_csv(OUT / 'tables' / f'{n}.csv')
    fin = {a: roc_auc_score(y[fv], P[a][fv]) for a in ARMS}
    adj = T('age_adjusted').set_index(['arm', 'method'])
    lt = T('ladder_deltas').set_index(['arm', 'step'])
    ops = []
    for n, lab in (('operating_points', f'first visit ({int(fv.sum())} eyes)'), ('operating_points_allscans',
                                                                              f'all scans ({len(y)}, all visits)')):
        o = T(n)
        o['target sensitivity'] = o['target sensitivity'].map('{:.2f}'.format)
        o = o[o['target sensitivity'] == '0.90'].drop(columns='target sensitivity')
        ops.append(o.rename(columns=lambda c: 'flagged' if c.endswith(' flagged') else c).assign(scope=lab))
    ops = pd.concat(ops)[['scope', 'method', 'sensitivity', 'specificity [95% CI]', 'PPV', 'NPV', 'flagged']]
    scope = T('auc_by_scope')
    rm = T('reading_map_noH')
    d = lambda a, st: f"{lt.loc[(a, st)].d:+.3f} [{lt.loc[(a, st)].lo:+.3f}, {lt.loc[(a, st)].hi:+.3f}]"
    look = ''.join(f'<li><b>{r.map}</b>, {r.location} ({r.leading_side}): {r.direction}</li>' for r in rm.head(5).itertuples())
    hd = head.drop(columns=['scans', 'positive scans'])
    return f"""<div class="box"><b>In one paragraph.</b> From a single Pentacam visit (both eyes' numerics and four maps, and age) the
model predicts whether a keratoconic eye will be referred for corneal cross-linking. In {len(y)} scans of {coh.eid.nunique()} eyes
({coh.pid.nunique()} patients, {int(y.sum())} referred scans), cross-validated by patient, the final model reaches a first-visit
ROC-AUC of <b>{fin['noH']:.3f}</b> (single visit, primary) and <b>{fin['H']:.3f}</b> with the fellow eye's treatment history (auxiliary),
against {roc_auc_score(y[fv], age_only(coh)[fv]):.3f} for age alone. The gain comes from the <b>fellow eye</b>: adding it to a single-eye
model adds {d('noH', 'L1→L2')} AUC. Within an age group the final model still separates referred eyes (age-adjusted AUC
{adj.loc[('noH', 'Final')].auc_within:.3f}), where a single-eye model is near chance. SHAP explanations point to elevation maps (front
elevation nasal in the fellow eye, back elevation in the scanned eye) and to the fellow eye's vertical-asymmetry indices.</div>
{key_findings}
<h3>ES.1 Data: from exports to the analysis cohort</h3>
<p>Every scan passes the same inclusion filters; extra exams (no label at the visit) are used for training only, never scored.</p>
{F('diagram_filtering')}
<h3>ES.2 The model</h3>
<p>Both eyes of one visit → numerics (block N, 171 columns) + 96 map measurements per eye (block R, 288) [+ fellow treatment
history, H] → three learners (XGBoost, LightGBM bag, elastic-net LR) tuned inside each training fold → equal-weight rank blend.
Evaluation: 5-fold patient-grouped CV × 5 repeats, patient-cluster bootstrap CIs.</p>
{F('diagram_dataflow')}
<h3>ES.3 Final results</h3>
{table(hd, 'hl')}
{F('roc_final')}
<p><b>Paired comparisons</b> (same scans, patient bootstrap). The second row per arm is the simplification that removed 11 derived
indices for interpretability (pre-declared tolerance 0.010).</p>
{table(deltas)}
<p><b>By visit scope.</b> First visit is the primary population; follow-up visits are harder and fewer.</p>
{table(scope)}
<h3>ES.4 Where the gain comes from: the fellow eye</h3>
<p>Same model, nested column sets. Fellow eye (L1 → L2): {d('noH', 'L1→L2')} noH, {d('H', 'L1→L2')} H. Own maps instead
(L1 → L2b): {d('noH', 'L1→L2b')}. Maps on top of both eyes' numerics (L2 → final): {d('noH', 'L2→final')}.</p>
{F('ladder')}
<h3>ES.5 Performance by disease stage and beyond age</h3>
<div class="grid2">{F('stage_auc_A', False)}{F('stage_auc_B', False)}</div>{HT.box('stage_auc_A')}
{F('age_adjusted')}
<h3>ES.6 What the model uses (SHAP)</h3>
<p>Mean |SHAP| by information source, and the leading features with the direction of their effect (first visit; the same ranking
holds over all scans, Spearman ρ ≥ 0.99, Appendix D).</p>
{F('shap_sources')}
{F('shap_beeswarm_noH')}
<h3>ES.7 Where on the cornea to look</h3>
<p>The five leading corneal sectors for the referral decision (noH, first visit):</p><ol>{look}</ol>
{F('reading_map_noH')}
{F('cornea_mean_maps_all')}
{F('r_gallery_noH')}
<h3>ES.8 Operating point</h3>
<p>At sensitivity 0.90 (thresholds set on the same out-of-fold scores, so descriptive), for first-visit eyes (one scan
per eye, the primary population) and for every scan of the cohort (all visits; a threshold set over all scans):</p>
{table(ops)}
{F('operating')}
{F('operating_allscans', False)}
<div class="box warn"><b>Caveats.</b> One centre, retrospective; the label is the clinic's referral decision, not progression.
The model was selected and evaluated on cross-validation of the same cohort (no external or held-out set). SHAP describes what the
model uses, not causes. Full methods, every analysis for both arms and the development record follow below.</div>"""


def age_sections(coh, y, g, fv, P, pa):
    """E30 (PROTOCOL.md 2026-11-15): age groups, age-adjusted AUC, operating characteristics, key findings."""
    out = {}
    grp = AO.age_group(coh.age.values)
    info = PL.cohort_info()
    info['age_group'] = AO.age_group(info['age'])
    lad = {(r, a): oof(f'E31_ladder_{r}_{a}') for r in ('L1', 'L2b', 'L2') for a in ARMS}
    out['age_first'] = f'{roc_auc_score(y[fv], pa[fv]):.3f}'
    out['age_cohort'] = img(AO.fig_age_cohort(info)) + table(AO.age_cohort_table(info).round(3), 'small')
    sc = {'Age only': pa[fv], 'Single-eye numerics (L1)': lad[('L1', 'noH')][fv], 'Final, noH': P['noH'][fv],
          'Final, H': P['H'][fv]}
    df = PL.stage_auc(y[fv], sc, grp[fv], g[fv])
    df.to_csv(OUT / 'tables' / 'stage_auc_age_group.csv', index=False)
    fin = df[df.method.str.startswith('Final')].auc
    out['age_range'] = f'{fin.min():.2f}–{fin.max():.2f}'
    det = ''
    for a in ARMS:
        da = PL.stage_auc(y[fv], {'Age only': pa[fv], 'Single-eye numerics (L1)': lad[('L1', a)][fv],
                                  'Both eyes, numerics (L2)': lad[('L2', a)][fv], f'Final, {a}': P[a][fv]},
                          grp[fv], g[fv])
        da.to_csv(OUT / 'tables' / f'stage_auc_age_group_ladder_{a}.csv', index=False)
        det += img(PL.fig_stage_auc(da, 'age_group', f'stage_auc_age_group_ladder_{a}',
                                    f'{a} arm: ladder rungs by age group'))
    out['age_auc'] = (img(PL.fig_stage_auc(df, 'age_group')) + img(AO.fig_age_roc(y, g, fv, P, grp)) +
                      '<details><summary>Per arm: ladder rungs (age, L1, L2, final) by age group</summary>' + det +
                      '</details>')
    meth = {'Age only': None, 'Single-eye numerics (L1)': 'L1', 'L1 + own maps (L2b)': 'L2b',
            'Both eyes, numerics (L2)': 'L2', 'Final': 'final'}
    scores = {(a, m): (pa if r is None else P[a] if r == 'final' else lad[(r, a)]) for a in ARMS for m, r in meth.items()}
    adj = AO.age_adjusted_table(y, g, fv, grp, scores)
    t = adj.assign(**{'crude AUC [95% CI]': [fmt(r.auc, r.lo, r.hi) for r in adj.itertuples()],
                      'age-adjusted AUC [95% CI]': [fmt(r.auc_within, r.lo_within, r.hi_within) for r in adj.itertuples()]})
    out['age_adj'] = img(AO.fig_age_adjusted(adj)) + table(t[['arm', 'method', 'crude AUC [95% CI]',
                                                              'age-adjusted AUC [95% CI]']])
    shs = {a: PL.load_shap(a) for a in ARMS}
    out['age_shap'] = img(AO.fig_age_shap(shs, info)) + img(PL.fig_stage_shap(shs, info, 'age_group'))
    curves = {'Final, noH': (P['noH'], PL.S.C['noH'], '-'), 'Final, H': (P['H'], PL.S.C['H'], '-'),
              'Single-eye numerics (L1)': (lad[('L1', 'noH')], PL.S.C['L1'], (0, (4, 2))),
              'Age only': (pa, PL.S.C['age'], (0, (1, 1.5)))}
    ops = AO.operating_table(y, g, fv, {'Final, noH': P['noH'], 'Final, H': P['H'],
                                        'Single-eye numerics (L1), noH': lad[('L1', 'noH')], 'Age only': pa})
    fig = AO.fig_operating(y, g, fv, curves, P)
    pr = pd.read_csv(OUT / 'tables' / 'precision_recall.csv')
    pr = pd.DataFrame({'method': pr.method, 'average precision [95% CI]': [fmt(r.ap, r.lo, r.hi) for r in pr.itertuples()],
                       'prevalence': pr.prevalence.round(3)})
    allm = np.ones(len(y), bool)
    with PL.S.all_scans():
        ops_all = AO.operating_table(y, g, allm, {'Final, noH': P['noH'], 'Final, H': P['H'],
                                                  'Single-eye numerics (L1), noH': lad[('L1', 'noH')], 'Age only': pa})
        fig_all = AO.fig_operating(y, g, allm, curves, P)
    out['operating'] = (img(fig) + table(pr) + table(ops) +
                        f'<h4>All scans ({len(y)} scans, all visits)</h4><p>The same analysis over every cohort scan; the '
                        'threshold is set over all scans, and CIs are patient-cluster bootstrap.</p>' +
                        img(fig_all, False) + table(ops_all))
    # key findings (numbers from this build)
    lt = pd.read_csv(OUT / 'tables' / 'ladder_deltas.csv').set_index(['arm', 'step'])
    ps = pd.read_csv(OUT / 'tables' / 'pairs_stats.csv').set_index('arm')
    A = adj.set_index(['arm', 'method'])
    o90 = ops[(ops.method == 'Final, noH') & (ops['target sensitivity'] == '0.90')].iloc[0]
    o90l = ops[(ops.method == 'Single-eye numerics (L1), noH') & (ops['target sensitivity'] == '0.90')].iloc[0]
    d = lambda a, s: f"{lt.loc[(a, s)].d:+.3f} [{lt.loc[(a, s)].lo:+.3f}, {lt.loc[(a, s)].hi:+.3f}]"
    lab = 'Single-eye numerics (L1)'
    sv = pd.read_csv(OUT / 'tables' / 'shap_sources.csv')
    sv = sv[(sv.view == 'Information source') & (sv.arm == 'noH')].sort_values('mean_abs_shap', ascending=False)
    src = [s.lower().replace(',', '') for s in sv.group[:2]]
    ci_ = pd.read_csv(OUT / 'tables' / 'cornea_importance_noH.csv').sort_values('mean_abs_shap', ascending=False)
    reg = '; '.join(ci_.label.str.replace(' (µm)', '', regex=False)[:3])
    out['key_findings'] = f"""<div class="box"><b>Key findings</b> (first visit, noH primary; H in brackets)<ol>
<li><b>Discrimination.</b> First-visit AUC {roc_auc_score(y[fv], P['noH'][fv]):.3f} ({roc_auc_score(y[fv], P['H'][fv]):.3f}), against {out['age_first']} for age alone and {roc_auc_score(y[fv], lad[('L1', 'noH')][fv]):.3f} for the single-eye numerics of the previous paper (§13, §15).</li>
<li><b>The fellow eye carries the gain.</b> Adding the fellow eye to the single-eye model: ΔAUC {d('noH', 'L1→L2')} ({d('H', 'L1→L2')}). Adding the own eye's maps instead: {d('noH', 'L1→L2b')}. Maps on top of both eyes' numerics: {d('noH', 'L2→final')}, a small, borderline increment (§15).</li>
<li><b>More than age.</b> Among eyes of the same age group, the single-eye model is near chance (age-adjusted AUC {A.loc[('noH', lab)].auc_within:.3f}); the final model keeps {A.loc[('noH', 'Final')].auc_within:.3f} ({A.loc[('H', 'Final')].auc_within:.3f}) (§21).</li>
<li><b>Matched pairs.</b> In {int(ps.loc['noH', 'matched pairs'])} pairs matched on stage, Km F and age, the final model orders {ps.loc['noH', 'ordered correctly, final']} correctly against {ps.loc['noH', 'ordered correctly, single-eye L1']} for the single-eye model (sign test p = {ps.loc['noH', 'sign test p (two-sided)']}) (§20).</li>
<li><b>What it uses.</b> By information source, {src[0]} and {src[1]} carry the most SHAP. Among map measurements, elevation dominates: the top regions are {reg} (§16–§17).</li>
<li><b>Operating point.</b> At sensitivity 0.90 the final noH model has specificity {o90['specificity [95% CI]']} and flags {o90['eyes flagged']} eyes; the single-eye model flags {o90l['eyes flagged']} (§13.4).</li>
</ol></div>"""
    devrec = OUT / 'dev_record.html'
    out['devrec'] = devrec.read_text(encoding='utf-8') if devrec.exists() else PENDING.format(what='development record')
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
        for lab, q in [('age only', pa), ('E28 blend, all columns incl. derived indices', oof(f'E28_{a}_blend'))]:
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
    strata['age group (years)'] = (AO.age_group(coh.age.values), AO.AGE_LAB)
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
    l1 = {a: oof(f'E31_ladder_L1_{a}') if (RES / 'oof' / f'E31_ladder_L1_{a}.npy').exists() else None for a in ARMS}
    curves = {'Final, noH': (P['noH'], PL.S.C['noH'], '-'), 'Final, H': (P['H'], PL.S.C['H'], '-')}
    if l1['noH'] is not None:
        curves['Single-eye numerics (L1)'] = (l1['noH'], PL.S.C['L1'], (0, (4, 2)))
    curves['Age only'] = (pa, PL.S.C['age'], (0, (1, 1.5)))
    roc_img = img(PL.fig_roc(y, g, fv, curves))
    stage_imgs = {}
    ladder_oof = {(r, a): oof(f'E31_ladder_{r}_{a}') for r in ('L1', 'L2') for a in ARMS
                  if (RES / 'oof' / f'E31_ladder_{r}_{a}.npy').exists()}
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
        if len(ladder_oof) == 4:      # per arm: the ladder rungs per stage (age, L1, L2, final)
            for a in ARMS:
                sca = {'Age only': pa[fv], 'Single-eye numerics (L1)': ladder_oof[('L1', a)][fv],
                       'Both eyes, numerics (L2)': ladder_oof[('L2', a)][fv], f'Final, {a}': P[a][fv]}
                da = PL.stage_auc(y[fv], sca, s_[fv], g[fv])
                da.to_csv(OUT / 'tables' / f'stage_auc_{k}_ladder_{a}.csv', index=False)
                stage_imgs[f'{k}_{a}'] = img(PL.fig_stage_auc(da, k, f'stage_auc_{k}_ladder_{a}',
                                                              f'{a} arm: ladder rungs by stage {k}'))
    if len(ladder_oof) == 4:
        for a in ARMS:
            stage_imgs[f'roc_{a}'] = img(PL.fig_roc(y, g, fv, {
                f'Final, {a}': (P[a], PL.S.C[a], '-'), 'Both eyes, numerics (L2)': (ladder_oof[('L2', a)], PL.S.C['L2'], '-'),
                'Single-eye numerics (L1)': (ladder_oof[('L1', a)], PL.S.C['L1'], (0, (4, 2))),
                'Age only': (pa, PL.S.C['age'], (0, (1, 1.5)))}, f'roc_ladder_{a}', f'{a} arm: ladder rungs'))
    strata_img = stage_imgs['A'] + stage_imgs['B'] + ''.join(
        f'<details><summary>{a} arm: ladder rungs per stage and ROC</summary>' + stage_imgs.get(f'A_{a}', '') +
        stage_imgs.get(f'B_{a}', '') + stage_imgs.get(f'roc_{a}', '') + '</details>' for a in ARMS)

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
    e30 = age_sections(coh, y, g, fv, P, pa)
    e31a = atlas_sections()
    e31b = allscans_sections(coh, y, g, fv, P)
    n_first, pos_first = int(fv.sum()), int(y[fv].sum())
    exs = exec_summary(coh, y, g, fv, P, head, deltas, e30['key_findings'])
    body = TEMPLATE.format(exec=exs, 
        date=dt.date.today().isoformat(), n=len(y), pos=int(y.sum()), n_pat=coh.pid.nunique(),
        n_eye=coh.eid.nunique(), n_first=n_first, pos_first=pos_first,
        age_mean=f'{coh.age.mean():.1f}', age_sd=f'{coh.age.std():.1f}', age_min=int(coh.age.min()),
        age_max=int(coh.age.max()),
        head=table(head, 'hl'), deltas=table(deltas), members=table(members), strata=table(st, 'small'),
        roc=roc_img, strata_img=strata_img, flow_img=img(DG.fig_dataflow()), filter_img=img(DG.fig_filtering()), filter_cohort=table(DG.funnel_tables()['cohort']), filter_extra=table(DG.funnel_tables()['extra']), rdict=table(rdict, 'small'), **e29, **e30, **e31a, **e31b,
        noH_first=f'{log.loc[FINAL["noH"]].auc_first:.3f}', H_first=f'{log.loc[FINAL["H"]].auc_first:.3f}')
    (OUT / 'report.html').write_text(body, encoding='utf-8')
    print('wrote', OUT / 'report.html')


TEMPLATE = open(Path(__file__).parent / 'report_template.html', encoding='utf-8').read() if (
    Path(__file__).parent / 'report_template.html').exists() else ''

if __name__ == '__main__':
    sys.exit(main())
