"""Publication figures for the final model (E29). Every function draws from stored artefacts (OOF scores, SHAP files)
and writes the numbers behind the figure to tables/. Aggregates only; individual-eye figures live in cases.py."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Wedge
from sklearn.metrics import roc_auc_score, roc_curve

import style as S

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kcref'))
from evaluate import cluster_boot_auc  # noqa: E402
import radiomics  # noqa: E402

S.apply()
TAB = S.OUT / 'tables'
STAGE_LAB = {'A': {1: '< 48', 2: '48–53', 3: '53–55', 4: '≥ 55'},
             'B': {1: '≥ 5.70', 2: '5.15–5.70', 3: '4.95–5.15', 4: '< 4.95'}}
STAGE_TITLE = {'A': 'Stage A (Km F, D)', 'B': 'Stage B (Rm B, mm)'}


def auc_ci(y, p, g):
    """(AUC, lo, hi) with patient-cluster bootstrap; NaNs when a class is missing."""
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    lo, hi = cluster_boot_auc(y, p, g)
    return roc_auc_score(y, p), lo, hi


def stage_auc(y, scores, stage, groups, min_class=10):
    """AUC per stage and method; `small` = fewer than min_class positives or negatives (no CI reported)."""
    rows = []
    for name, p in scores.items():
        for k in (1, 2, 3, 4):
            m = stage == k
            pos, n = int(y[m].sum()), int(m.sum())
            small = pos < min_class or n - pos < min_class
            a = auc_ci(y[m], p[m], groups[m]) if n else (np.nan,) * 3
            rows.append(dict(method=name, stage=k, n=n, pos=pos, auc=a[0], lo=np.nan if small else a[1],
                             hi=np.nan if small else a[2], small=small))
    return pd.DataFrame(rows)


METHOD_COL = {'Age only': S.C['age'], 'Single-eye numerics (L1)': S.C['L1'], 'Final, noH': S.C['noH'],
              'Final, H': S.C['H']}


def framed(ax, grid='both'):
    """Full thin frame + light grid (the AUC figures' look)."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color('#4A4A4A')
    ax.grid(axis=grid, color=S.C['grid'], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction='in', top=False, right=False)


def shades(hex_, n=4):
    """n steps of one hue from light to the full colour (ordinal stages)."""
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(hex_))
    return [tuple(1 - (1 - c) * t) for t in np.linspace(0.38, 1.0, n)]


def fig_stage_auc(df, staging):
    """One panel per method; bars = first-visit AUC per stage, CI whiskers, AUC printed above; hatched = too few eyes
    for a CI (< 10 referred or < 10 not referred)."""
    methods = [m for m in METHOD_COL if m in set(df.method)] + [m for m in df.method.unique() if m not in METHOD_COL]
    fig, axes = plt.subplots(2, 2, figsize=(S.W2, 5.0), sharey=True)
    axes = axes.ravel()
    for ax, m, letter in zip(axes, methods, 'abcd'):
        S.panel_letter(ax, letter, x=-0.12, y=1.03)
        d = df[df.method == m].set_index('stage')
        base = METHOD_COL.get(m, S.C['mute'])
        cols = shades(base)
        framed(ax, 'y')
        ax.axhline(0.5, color='#9A9A9A', lw=0.8, ls=(0, (1, 2)), zorder=1)
        if d.auc.isna().all():
            ax.text(2.5, 0.65, 'pending', ha='center', color=S.C['mute'], fontsize=7)
        for k in (1, 2, 3, 4):
            r = d.loc[k]
            if np.isnan(r.auc):
                ax.text(k, 0.03, 'n/a', ha='center', va='bottom', fontsize=6, color=S.C['mute'])
                continue
            ax.bar(k, r.auc, width=0.62, color='white' if r.small else cols[k - 1], edgecolor='#333333',
                   hatch='////' if r.small else None, lw=0.7, zorder=2)
            if r.small:
                ax.bar(k, r.auc, width=0.62, color='none', edgecolor=cols[k - 1], hatch='////', lw=0, zorder=2)
            else:
                ax.errorbar(k, r.auc, yerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt='none', ecolor='#333333', lw=0.7,
                            capsize=2.2, zorder=3)
            top = r.auc if r.small else r.hi
            ax.text(k, top + 0.018, f'AUC = {r.auc:.2f}', ha='center', va='bottom', fontsize=6.3, fontweight='bold',
                    color=S.C['ink'])
        ax.set_xticks([1, 2, 3, 4], [f'{STAGE_LAB[staging][k]}\n(Stage {k})\n{int(d.loc[k].pos)}/{int(d.loc[k].n)}'
                                     for k in (1, 2, 3, 4)], fontsize=6.3)
        ax.set(ylim=(0, 1.1), xlim=(0.4, 4.6))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(m, loc='center', color=base, fontsize=8)
    for ax in axes[::2]:
        ax.set_ylabel('ROC-AUC, first visit (95% CI)')
    for ax in axes:
        ax.text(4.55, 0.515, 'Random', ha='right', va='bottom', fontsize=5.8, color='#8A8A8A')
    fig.text(0.5, -0.015, f'{STAGE_TITLE[staging]}  ·  under each bar: range, stage, referred / eyes  ·  '
             'hatched: < 10 referred or < 10 not referred, no CI', ha='center', fontsize=6.2, color=S.C['mute'])
    fig.tight_layout(w_pad=1.5, h_pad=1.8)
    return S.save(fig, f'stage_auc_{staging}')


FPR = np.linspace(0, 1, 201)


def roc_band(y, p, g, n=1000, seed=0):
    """Mean ROC (TPR on a fixed FPR grid) with a 95% patient-cluster bootstrap band."""
    rng = np.random.default_rng(seed)
    pids = np.unique(g)
    rows = {u: np.flatnonzero(g == u) for u in pids}
    out = []
    for _ in range(n):
        i = np.concatenate([rows[u] for u in rng.choice(pids, len(pids))])
        if len(np.unique(y[i])) < 2:
            continue
        f, t, _ = roc_curve(y[i], p[i])
        out.append(np.interp(FPR, f, t))
    out = np.array(out)
    f, t, _ = roc_curve(y, p)
    tp = np.interp(FPR, f, t)
    tp[0] = 0
    return tp, np.percentile(out, 2.5, 0), np.percentile(out, 97.5, 0)


def fig_roc(y, g, fv, curves):
    """ROC, all scans and first visit, with 95% patient-cluster bootstrap bands for the shaded curves.
    curves: {label: (score, colour, linestyle)}; solid curves get a band, dashed / dotted references do not."""
    fig, axes = plt.subplots(1, 2, figsize=(S.W2, 3.45))
    rows = []
    for ax, (title, m), letter in zip(axes, [('All scans (944)', np.ones(len(y), bool)),
                                             (f'First visit ({int(fv.sum())} eyes)', fv)], 'ab'):
        framed(ax)
        ax.plot([0, 1], [0, 1], color='#9A9A9A', lw=0.8, ls=(0, (1, 2)), zorder=1)
        for lab, (p, col, ls) in curves.items():
            a, lo, hi = auc_ci(y[m], p[m], g[m])
            if ls == '-':
                tp, blo, bhi = roc_band(y[m], p[m], g[m])
                ax.fill_between(FPR, blo, bhi, color=col, alpha=0.14, lw=0, zorder=2)
                ax.plot(FPR, tp, color=col, lw=1.9, zorder=4, solid_capstyle='round',
                        label=f'{lab}   AUC = {a:.3f} [{lo:.3f}–{hi:.3f}]')
            else:
                f, t, _ = roc_curve(y[m], p[m])
                ax.plot(f, t, color=col, ls=ls, lw=1.1, zorder=3, label=f'{lab}   AUC = {a:.3f} [{lo:.3f}–{hi:.3f}]')
            rows.append(dict(scope=title, method=lab, auc=a, lo=lo, hi=hi))
        ax.set(xlim=(0, 1), ylim=(0, 1.005), xlabel='False positive rate (1 − specificity)',
               ylabel='True positive rate (sensitivity)', aspect='equal')
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_title(title, loc='center')
        lg = ax.legend(loc='lower right', fontsize=5.7, handlelength=2.2, frameon=True, fancybox=False,
                       edgecolor='#CCCCCC', framealpha=0.95, borderpad=0.6)
        lg.get_frame().set_linewidth(0.5)
        S.panel_letter(ax, letter, x=-0.2, y=1.02)
    fig.text(0.5, -0.01, 'Shaded: 95% patient-cluster bootstrap band (1000 resamples); AUC with 95% CI.', ha='center',
             fontsize=6.2, color=S.C['mute'])
    fig.tight_layout(w_pad=2.5)
    pd.DataFrame(rows).to_csv(TAB / 'roc.csv', index=False)
    return S.save(fig, 'roc_final')


def legend_sources(ax, keys, **kw):
    ax.legend(handles=[Patch(color=S.SRC[k], label=S.SRC_LABEL[k]) for k in keys], **kw)


# ---------------------------------------------------------------- cornea geometry (radiomics.py conventions)
QUAD_DEG = {'N': (-45, 45), 'S': (45, 135), 'T': (135, 225), 'I': (225, 315)}  # +x nasal, +y superior
MAP_EXTENT = (radiomics.X.min(), radiomics.X.max(), radiomics.Y.min(), radiomics.Y.max())  # imshow, origin='upper'


def dartboard(ax, values, vmax, cmap=S.SEQ, vmin=0.0, edge='white'):
    """16 ring x quadrant sectors (rings 0-1 ... 3-4 mm) coloured by values[(r0, r1, quad)]; missing = light grey.
    Returns the patches (gid 'r{r0}{r1}_{q}')."""
    cm, nm = plt.get_cmap(cmap), Normalize(vmin, vmax)
    out = []
    for r0 in range(4):
        for q, (t0, t1) in QUAD_DEG.items():
            v = values.get((r0, r0 + 1, q))
            w = Wedge((0, 0), r0 + 1, t0, t1, width=1, facecolor='#EEEEEE' if v is None else cm(nm(v)),
                      edgecolor=edge, lw=0.8, gid=f'r{r0}{r0 + 1}_{q}')
            ax.add_patch(w)
            out.append(w)
    ax.set(xlim=(-4.4, 4.4), ylim=(-4.4, 4.4), aspect='equal')
    ax.axis('off')
    return out


def map_centroid_mm(a):
    """Value-weighted centroid (x, y) in mm of a 145 x 145 map, in radiomics.grid coordinates."""
    w = np.nan_to_num(np.asarray(a, float))
    return float((w * radiomics.X).sum() / w.sum()), float((w * radiomics.Y).sum() / w.sum())


# ---------------------------------------------------------------- SHAP summaries
import explain as E  # noqa: E402
from common import DATA, RES  # noqa: E402
from staging import stage_A, stage_B  # noqa: E402

FAMILY_LABEL = {'own_num': 'Own numerics', 'fel_num': 'Fellow numerics', 'asym_num': 'Own − fellow numerics',
                'R_own': 'Own maps (R)', 'R_fel': 'Fellow maps (R)', 'R_asym': 'Own − fellow maps (R)',
                'age': 'Age', 'flags': 'Map availability', 'H': 'Fellow treatment history'}
SRC_ORDER = ['own_num', 'own_maps', 'fellow_num', 'fellow_maps', 'age', 'flags', 'H']


def cohort_info():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    return dict(y=coh.y.values, first=coh.first_visit.values.astype(bool), pid=coh.pid.values,
                A=stage_A(coh), B=stage_B(coh), km=coh['Km F (D):'].values, age=coh.age.values)


def load_shap(arm):
    z = np.load(RES / 'shap' / f'{arm}_oof.npz', allow_pickle=False)
    return prepare({k: z[k] for k in z.files})


def prepare(sh):
    """Adds cols (list), mask (R availability) and the masked SHAP Zm (unavailable map attributions set to 0)."""
    sh = dict(sh)
    sh['cols'] = [str(c) for c in sh['cols']]
    j = [sh['cols'].index('has_fellow_image'), sh['cols'].index('own_image')]
    sh['mask'] = E.r_mask(sh['cols'], np.nan_to_num(sh['raw'][:, j]))
    sh['Zm'] = np.where(sh['mask'], sh['shap'], 0.0)
    return sh


def col_importance(sh, rows):
    """Mean |SHAP| per column over `rows`, counting only rows where the column's map exists."""
    m = sh['mask'] & rows[:, None]
    return np.abs(sh['Zm']).sum(0) / np.maximum(m.sum(0), 1)


def group_importance(sh, rows, key=E.source):
    """Mean |sum of SHAP within group| over `rows` with at least one available column of the group."""
    keys = np.array([key(c) for c in sh['cols']])
    out = {}
    for k in dict.fromkeys(keys):
        j = keys == k
        r = rows & sh['mask'][:, j].any(1)
        out[k] = float(np.abs(sh['Zm'][r][:, j].sum(1)).mean()) if r.any() else np.nan
    return out


def fig_sources(shs, info):
    """E2a: (a) information sources, (b) the 9 encoded families; mean |grouped SHAP|, first visit, noH vs H."""
    fv = info['first']
    fig, axes = plt.subplots(1, 2, figsize=(S.W2, 2.7), gridspec_kw=dict(width_ratios=[1, 1.1]))
    rows = []
    for ax, key, order, lab, letter, title in (
            (axes[0], E.source, SRC_ORDER, S.SRC_LABEL, 'a', 'Information source'),
            (axes[1], E.family, list(FAMILY_LABEL), FAMILY_LABEL, 'b', 'Encoded feature family')):
        imp = {a: group_importance(sh, fv, key) for a, sh in shs.items()}
        order = [k for k in order if any(k in v for v in imp.values())]
        order = sorted(order, key=lambda k: imp['noH'].get(k, imp['H'].get(k, 0)))
        yy = np.arange(len(order))
        for i, (a, off) in enumerate((('noH', 0.19), ('H', -0.19))):
            v = [imp[a].get(k, np.nan) for k in order]
            ax.barh(yy + off, v, height=0.36, color=S.C[a], label=f'{a}', zorder=2)
            for k, x in zip(order, v):
                rows.append(dict(view=title, group=lab[k], arm=a, mean_abs_shap=x))
        ax.set_yticks(yy, [lab[k] for k in order])
        framed(ax, 'x')
        ax.set_xlabel('Mean |SHAP| (member-standardised), first visit')
        ax.set_title(title)
        ax.tick_params(axis='y', length=0)
        S.panel_letter(ax, letter, x=-0.5 if letter == 'a' else -0.45)
    axes[0].legend(loc='lower right', handlelength=1, title='Arm', title_fontsize=6.5)
    fig.tight_layout(w_pad=3)
    pd.DataFrame(rows).to_csv(TAB / 'shap_sources.csv', index=False)
    return S.save(fig, 'shap_sources')


def _swarm_y(x, width=0.36, nb=40):
    """Vertical offsets that spread points by local density (simple beeswarm)."""
    if len(x) < 2:
        return np.zeros(len(x))
    b = np.digitize(x, np.linspace(x.min(), x.max(), nb))
    off = np.zeros(len(x))
    for k in np.unique(b):
        i = np.flatnonzero(b == k)
        n = len(i)
        off[i] = (np.arange(n) - (n - 1) / 2) / max(n, 1) * min(1, n / 12) * 2 * width
    return off


def fig_beeswarm(sh, info, arm, top=20):
    """E2b: top features, one dot per first-visit eye; colour = feature value percentile; chip = Belin ABCD."""
    fv = info['first']
    imp = col_importance(sh, fv)
    idx = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(S.W2 * 0.78, 4.6))
    cm = plt.get_cmap(S.DIV)
    rows = []
    for yi, j in enumerate(idx):
        r = fv & sh['mask'][:, j]
        x, v = sh['shap'][r, j], sh['raw'][r, j]
        ok = np.isfinite(v)
        pct = np.full(len(v), 0.5)
        pct[ok] = (pd.Series(v[ok]).rank(pct=True)).values
        o = np.argsort(np.abs(x))
        ax.scatter(x[o], yi + _swarm_y(x)[o], c=cm(pct[o]), s=3.2, lw=0, alpha=0.9, rasterized=True, zorder=2)
        rows.append(dict(rank=top - yi, feature=E.label(sh['cols'][j]), column=sh['cols'][j],
                         abcd=E.abcd(sh['cols'][j]), source=E.source(sh['cols'][j]), mean_abs_shap=imp[j]))
    ax.axvline(0, color=S.C['mute'], lw=0.6, zorder=1)
    ax.set_yticks(range(len(idx)), [E.label(sh['cols'][j]) for j in idx], fontsize=6.4)
    for yi, j in enumerate(idx):
        t = E.abcd(sh['cols'][j])
        ax.annotate(t if t != 'other' else '·', xy=(0, yi), xycoords=('axes fraction', 'data'), xytext=(-3, 0),
                    textcoords='offset points', ha='right', va='center', fontsize=5.5, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.25', fc=S.ABCD[t], ec='none'))
    ax.tick_params(axis='y', length=0, pad=16)
    ax.set_ylim(-0.7, len(idx) - 0.3)
    ax.set_xlabel('SHAP value (member-standardised)  ←  lower referral score   ·   higher referral score  →')
    ax.set_title(f'Top {top} features, {arm} arm, first visit ({int(fv.sum())} eyes)')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(['low', 'high'])
    cb.set_label('Feature value (percentile)', fontsize=6.5)
    cb.outline.set_visible(False)
    ax.legend(handles=[Patch(color=S.ABCD[k], label=lab) for k, lab in
                       (('A', 'A: anterior curvature'), ('B', 'B: posterior surface'), ('C', 'C: thickness'))],
              loc='lower right', fontsize=6, title='Belin ABCD', title_fontsize=6)
    fig.tight_layout()
    pd.DataFrame(rows[::-1]).to_csv(TAB / f'shap_top_{arm}.csv', index=False)
    return S.save(fig, f'shap_beeswarm_{arm}')


def _trend(x, y, k=41):
    o = np.argsort(x)
    return x[o], pd.Series(y[o]).rolling(k, center=True, min_periods=k // 3).median().values


def fig_dependence(sh, info, arm, top=4):
    """E2c: SHAP vs feature value for the top features, coloured by stage A, with a rolling-median trend."""
    fv = info['first']
    imp = col_importance(sh, fv)
    idx = np.argsort(imp)[::-1][:top]
    fig, axes = plt.subplots(1, top, figsize=(S.W2, 2.15))
    cm = plt.get_cmap(S.SEQ)
    for ax, j, letter in zip(axes, idx, 'abcdefgh'):
        r = fv & sh['mask'][:, j] & np.isfinite(sh['raw'][:, j])
        x, z, st = sh['raw'][r, j].astype(float), sh['shap'][r, j], info['A'][r]
        ax.axhline(0, color=S.C['chance'], lw=0.6)
        ax.scatter(x, z, c=cm((np.nan_to_num(st, nan=1) - 1) / 3), s=4, lw=0, alpha=0.8, rasterized=True)
        tx, ty = _trend(x, z)
        ax.plot(tx, ty, color=S.C['ink'], lw=1.2)
        ax.set_xlabel(E.label(sh['cols'][j]), fontsize=6.3, wrap=True)
        S.panel_letter(ax, letter, x=-0.25)
    axes[0].set_ylabel('SHAP value')
    fig.legend(handles=[Patch(color=cm(k / 3), label=f'A{k + 1}: {STAGE_LAB["A"][k + 1]}') for k in range(4)],
               loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=6.3, title='Stage A (Km F, D)',
               title_fontsize=6.3)
    fig.tight_layout(w_pad=1.2)
    return S.save(fig, f'shap_dependence_{arm}')


SIDES = (('own', 'Own eye'), ('fel', 'Fellow eye'), ('asym', 'Own − fellow'))
MAP_ORDER = (('T', 'Thickness'), ('K', 'Front curvature'), ('EF', 'Front elevation'), ('EB', 'Back elevation'))


def r_table(sh, info, signed=False):
    """Per R feature x side: mean |SHAP| (or Spearman rho of SHAP vs value) over first-visit eyes with the map."""
    fv = info['first']
    imp = col_importance(sh, fv)
    rows = []
    for j, c in enumerate(sh['cols']):
        if not c.startswith('R_'):
            continue
        side = E._split(c)[1]
        r = fv & sh['mask'][:, j] & np.isfinite(sh['raw'][:, j])
        rho = pd.Series(sh['raw'][r, j]).rank().corr(pd.Series(sh['shap'][r, j]).rank()) if r.sum() > 10 else np.nan
        rows.append(dict(col=c, side=side, rname=E.r_name(c), label=E.label(c), mean_abs_shap=imp[j], rho=rho,
                         n=int(r.sum())))
    return pd.DataFrame(rows)


def fig_cornea(sh, info, arm, signed=False):
    """E3: where on the cornea. Rows own / fellow / own − fellow; columns the four maps. Dartboard = 16 sectors
    (0-4 mm, S/N/I/T; +x nasal for both eyes); below, the non-sector measurements of each map as a heat strip."""
    tab = r_table(sh, info)
    val = 'rho' if signed else 'mean_abs_shap'
    vmax = np.nanmax(np.abs(tab[val])) if signed else np.nanmax(tab[val])
    vmin, cmap = (-vmax, S.DIV) if signed else (0.0, S.SEQ)
    fig = plt.figure(figsize=(S.W2, 6.6))
    gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 1.15], width_ratios=[0.32, 1, 1, 1, 1], hspace=0.12,
                          wspace=0.05)
    for i, (side, slab) in enumerate(SIDES):
        ax = fig.add_subplot(gs[i, 0])
        ax.axis('off')
        ax.text(0.95, 0.5, slab, ha='right', va='center', fontsize=8, fontweight='bold', rotation=90)
        for k, (mp, mlab) in enumerate(MAP_ORDER):
            ax = fig.add_subplot(gs[i, k + 1])
            d = tab[(tab.side == side) & tab.rname.str.match(rf'{mp}_r\d\d_')]
            vals = {E.sector(r)[1:]: v for r, v in zip(d.rname, d[val])}
            dartboard(ax, vals, vmax=vmax, vmin=vmin, cmap=cmap)
            for (r0, r1, q), v in vals.items():
                top3 = d[val].abs().nlargest(3).values if signed else d[val].nlargest(3).values
                if abs(v) >= top3.min() and r1 >= 1:
                    t0, t1 = QUAD_DEG[q]
                    ax.add_patch(Wedge((0, 0), r1, t0, t1, width=1, fc='none', ec=S.C['ink'], lw=1.3))
            if i == 0:
                ax.set_title(mlab, loc='center', fontsize=8)
            if i == 0 and k == 0:
                for s_, (x, y) in {'S': (0, 4.35), 'I': (0, -4.35), 'N': (4.35, 0), 'T': (-4.35, 0)}.items():
                    ax.text(x, y, s_, ha='center', va='center', fontsize=6, color=S.C['mute'])
    # non-sector measurements
    for k, (mp, mlab) in enumerate(MAP_ORDER):
        ax = fig.add_subplot(gs[3, k + 1])
        d = tab[tab.rname.str.startswith(mp + '_') & ~tab.rname.str.match(rf'{mp}_r\d\d_')]
        names = list(dict.fromkeys(d.rname))
        M_ = np.array([[d[(d.rname == n) & (d.side == s)][val].values[0] for s, _ in SIDES] for n in names])
        ax.imshow(M_, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_yticks(range(len(names)), [E._r_label(n).split(', ', 1)[1] for n in names], fontsize=5.2)
        ax.set_xticks(range(3), ['own', 'fel', 'o−f'], fontsize=5.5)
        ax.tick_params(length=0)
        ax.yaxis.tick_right() if k == 3 else None
        if k not in (0, 3):
            ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cax = fig.add_axes([0.93, 0.5, 0.012, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax)), cax=cax)
    cb.set_label('Spearman ρ (value, SHAP)' if signed else 'Mean |SHAP|', fontsize=6.3)
    cb.outline.set_visible(False)
    fig.text(0.5, 0.005, 'Sectors: rings 0–1, 1–2, 2–3, 3–4 mm × quadrants; +x nasal (left eyes mirrored), +y superior. '
             'Black outline: top 3 sectors of the panel. First-visit eyes with the map.',
             ha='center', fontsize=6, color=S.C['mute'])
    name = f'cornea_{"signed" if signed else "importance"}_{arm}'
    tab.to_csv(TAB / f'{name}.csv', index=False)
    return S.save(fig, name)


def mean_maps(info):
    """Mean decoded map (4 x 145 x 145) of referred and not-referred first-visit eyes; group means only."""
    hr = np.load(DATA / 'decoded_hr.npy', mmap_mode='r')
    meta = pd.read_parquet(DATA / 'decoded_hr_meta.parquet')
    row = {int(c): i for i, c in enumerate(meta.cohort_row.values) if c >= 0}
    out = {}
    for lab, yv in (('referred', 1), ('not referred', 0)):
        idx = [row[i] for i in np.flatnonzero(info['first'] & (info['y'] == yv)) if i in row]
        acc = np.zeros((4, 145, 145))
        cnt = np.zeros((4, 145, 145))
        for i in idx:
            a = np.asarray(hr[i], np.float64)
            ok = np.isfinite(a)
            acc[ok] += a[ok]
            cnt += ok
        out[lab] = (np.where(cnt > len(idx) * 0.5, acc / np.maximum(cnt, 1), np.nan), len(idx))
    return out


def fig_mean_maps(info, tab, maps=('T', 'EB')):
    """E3c: mean map of referred vs not-referred first-visit eyes and their difference, with the top-3 own-eye
    sectors of each map outlined (from the noH importance table)."""
    mm = mean_maps(info)
    from scipy.ndimage import gaussian_filter

    def smooth(a, sd=1.5):
        """NaN-aware Gaussian smoothing (sd in cells, 1 cell = 0.069 mm) to remove colour-decoding contour steps."""
        ok = np.isfinite(a)
        num, den = gaussian_filter(np.where(ok, a, 0), sd), gaussian_filter(ok.astype(float), sd)
        return np.where(ok, num / np.maximum(den, 1e-9), np.nan)
    fig, axes = plt.subplots(len(maps), 3, figsize=(S.W2 * 0.8, 2.45 * len(maps)))
    R = radiomics.R
    for i, mp in enumerate(maps):
        j = [m for m, _ in MAP_ORDER].index(mp)
        a, b = smooth(mm['referred'][0][j]), smooth(mm['not referred'][0][j])
        a, b = np.where(R <= 4, a, np.nan), np.where(R <= 4, b, np.nan)
        lo, hi = np.nanpercentile(np.r_[a.ravel(), b.ravel()], [2, 98])
        dl = np.nanpercentile(np.abs(a - b), 98)
        d = tab[(tab.side == 'own') & tab.rname.str.match(rf'{mp}_r\d\d_')].nlargest(3, 'mean_abs_shap')
        for k, (img_, cmap, vmin, vmax, title) in enumerate((
                (a, S.MAP_CMAP[mp], lo, hi, f'Referred (n = {mm["referred"][1]})'),
                (b, S.MAP_CMAP[mp], lo, hi, f'Not referred (n = {mm["not referred"][1]})'),
                (a - b, S.DIV, -dl, dl, 'Difference'))):
            ax = axes[i, k]
            im = ax.imshow(img_, cmap=cmap, vmin=vmin, vmax=vmax, extent=MAP_EXTENT, origin='upper')
            for rn in d.rname:
                _, r0, r1, q = E.sector(rn)
                t0, t1 = QUAD_DEG[q]
                ax.add_patch(Wedge((0, 0), r1, t0, t1, width=r1 - r0, fc='none', ec='black', lw=1.1))
            ax.set(xlim=(-4.3, 4.3), ylim=(-4.3, 4.3))
            ax.axis('off')
            if i == 0:
                ax.set_title(title, loc='center', fontsize=7.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=5.5)
            cb.outline.set_visible(False)
        axes[i, 0].text(-5.2, 0, dict(MAP_ORDER)[mp] + (' (µm)' if mp != 'K' else ' (D)'), rotation=90, ha='center',
                        va='center', fontsize=7.5, fontweight='bold')
    fig.text(0.5, 0.0, 'First-visit eyes, central 8 mm, +x nasal; group means, smoothed (Gaussian, 0.1 mm); outlined: top 3 own-eye sectors by mean |SHAP| (noH).',
             ha='center', fontsize=6, color=S.C['mute'])
    fig.tight_layout()
    return S.save(fig, 'cornea_mean_maps')


def fig_stage_shap(shs, info, staging):
    """E5: share of each information source in total |grouped SHAP| per stage (first visit), noH and H."""
    st = info[staging]
    fig, axes = plt.subplots(1, 2, figsize=(S.W2, 2.4), sharey=True)
    rows = []
    for ax, (a, sh), letter in zip(axes, shs.items(), 'ab'):
        srcs = [s for s in SRC_ORDER if any(E.source(c) == s for c in sh['cols'])]
        M_ = np.full((len(srcs), 4), np.nan)
        ns = []
        for k in (1, 2, 3, 4):
            r = info['first'] & (st == k)
            ns.append(int(r.sum()))
            if r.sum() >= 10:
                g = group_importance(sh, r)
                tot = sum(g[s] for s in srcs if np.isfinite(g.get(s, np.nan)))
                M_[:, k - 1] = [g.get(s, np.nan) / tot for s in srcs]
            for i, s in enumerate(srcs):
                rows.append(dict(arm=a, stage=k, n=ns[-1], source=S.SRC_LABEL[s], share=M_[i, k - 1]))
        ax.imshow(np.ma.masked_invalid(M_), cmap=S.SEQ, vmin=0, vmax=0.5, aspect='auto')
        for i in range(len(srcs)):
            for k in range(4):
                v = M_[i, k]
                ax.text(k, i, '—' if np.isnan(v) else f'{v:.0%}', ha='center', va='center', fontsize=6.3,
                        color=S.C['ink'] if (np.isnan(v) or v < 0.3) else 'white')
        ax.set_facecolor('#DDDDDD')
        ax.set_xticks(range(4), [f'{k}: {STAGE_LAB[staging][k]}\nn = {n}' for k, n in zip((1, 2, 3, 4), ns)],
                      fontsize=6.2)
        ax.set_yticks(range(len(srcs)), [S.SRC_LABEL[s] for s in srcs])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'{a} arm', color=S.C[a])
        S.panel_letter(ax, letter, x=-0.55 if letter == 'a' else -0.08)
    fig.text(0.5, -0.06, f'{STAGE_TITLE[staging]}; cell = share of total |grouped SHAP| (first visit); '
             '— = fewer than 10 eyes', ha='center', fontsize=6.3, color=S.C['mute'])
    fig.tight_layout(w_pad=1)
    pd.DataFrame(rows).to_csv(TAB / f'shap_by_stage_{staging}.csv', index=False)
    return S.save(fig, f'shap_by_stage_{staging}')


def fig_robustness(sh, info, arm, top=20):
    """E6: (a) rank of each top feature across the 25 CV fits (median, 5-95%); (b) source share per fit."""
    fi = sh['fit_imp']
    ranks = (-fi).argsort(1).argsort(1) + 1
    imp = col_importance(sh, info['first'])
    idx = np.argsort(imp)[::-1][:top][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(S.W2, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    rows = []
    for yi, j in enumerate(idx):
        r = ranks[:, j]
        q = np.percentile(r, [5, 50, 95])
        col = S.ABCD[E.abcd(sh['cols'][j])]
        ax.plot([q[0], q[2]], [yi, yi], color=col, lw=1.6, solid_capstyle='round')
        ax.plot(q[1], yi, 'o', ms=4, mfc='white', mec=col, mew=1.2)
        rows.append(dict(feature=E.label(sh['cols'][j]), abcd=E.abcd(sh['cols'][j]), median_rank=q[1], p5=q[0],
                         p95=q[2]))
    ax.set_yticks(range(len(idx)), [E.label(sh['cols'][j]) for j in idx], fontsize=6)
    ax.set_xscale('log')
    tk = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500) if t <= max(ranks[:, idx].max(), 2)]
    ax.set_xticks(tk, [str(t) for t in tk])
    ax.minorticks_off()
    ax.set_xlabel('Rank across 25 CV fits (median, 5–95%; 1 = most important)')
    framed(ax, 'x')
    ax.tick_params(axis='y', length=0)
    ax.invert_xaxis()
    S.panel_letter(ax, 'a', x=-0.95)
    ax = axes[1]
    keys = np.array([E.source(c) for c in sh['cols']])
    srcs = [s for s in SRC_ORDER if (keys == s).any()]
    share = np.array([[fi[f, keys == s].sum() for s in srcs] for f in range(len(fi))])
    share /= share.sum(1, keepdims=True)
    rng = np.random.default_rng(0)
    for i, s in enumerate(srcs):
        ax.scatter(share[:, i], i + rng.uniform(-0.18, 0.18, len(share)), s=7, color=S.SRC[s], lw=0, alpha=0.85)
        ax.plot([np.median(share[:, i])] * 2, [i - 0.3, i + 0.3], color=S.C['ink'], lw=1.2)
    ax.set_yticks(range(len(srcs)), [S.SRC_LABEL[s] for s in srcs])
    framed(ax, 'x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.set_xlabel('Share of total mean |SHAP| per fit (per-column sum)')
    ax.grid(axis='x', zorder=0)
    ax.tick_params(axis='y', length=0)
    S.panel_letter(ax, 'b', x=-0.6)
    fig.suptitle(f'{arm} arm', x=0.01, ha='left', fontsize=8.5, fontweight='bold', color=S.C[arm])
    fig.tight_layout(w_pad=2)
    pd.DataFrame(rows[::-1]).to_csv(TAB / f'robustness_{arm}.csv', index=False)
    return S.save(fig, f'robustness_{arm}')


# ---------------------------------------------------------------- S1 matched pairs (statistics only)
def pairs(sh, info):
    """Matched pairs on first-visit eyes with both maps (explain.match_pairs). Returns (pairs on the sub-frame,
    sub -> cohort row index, final noH OOF, L1 OOF)."""
    j = [sh['cols'].index('has_fellow_image'), sh['cols'].index('own_image')]
    both = (np.nan_to_num(sh['raw'][:, j]) > 0).all(1)
    sub = np.flatnonzero(info['first'] & both & np.isfinite(info['A']))
    df = pd.DataFrame(dict(pid=info['pid'][sub], y=info['y'][sub], km=info['km'][sub], age=info['age'][sub],
                           stage=info['A'][sub]))
    pr = E.match_pairs(df)
    l1 = np.load(RES / 'oof' / 'E29_ladder_L1_noH.npy').mean(0)
    return pr, sub, sh['oof'], l1


def pair_table(sh, info):
    pr, sub, s_fin, s_l1 = pairs(sh, info)
    st = E.pair_stats(pr.assign(pos_idx=sub[pr.pos_idx], neg_idx=sub[pr.neg_idx]), s_fin, s_l1)
    t = pd.DataFrame([{'matched pairs': st['n_pairs'], 'ordered correctly, single-eye L1': f'{st["frac_L1"]:.1%}',
                       'ordered correctly, final noH': f'{st["frac_final"]:.1%}',
                       'only final correct': st['final_only'], 'only L1 correct': st['L1_only'],
                       'sign test p (two-sided)': f'{st["p_sign"]:.2g}'}])
    t.to_csv(TAB / 'pairs_stats.csv', index=False)
    return t


# ---------------------------------------------------------------- E1 ladder
RUNG_LABEL = {'L1': 'Own eye numerics (L1)', 'L2b': 'L1 + own-eye maps (L2b)', 'L2': 'L1 + fellow eye (L2 = block N)',
              'final': 'L2 + all maps (final)'}


def fig_ladder(lad, dl, age_auc):
    """E1: first-visit AUC [95% CI] per rung, noH and H tracks; paired deltas annotated; age only as a reference line.
    lad: arm, rung, auc, lo, hi.  dl: arm, step ('L1→L2' ...), d, lo, hi."""
    rungs = [r for r in RUNG_LABEL if r in set(lad.rung)]
    fig, ax = plt.subplots(figsize=(S.W2, 2.9))
    framed(ax, 'x')
    ax.axvline(age_auc, color=S.C['age'], lw=0.9, ls=(0, (1, 2)))
    ax.text(age_auc, len(rungs) - 0.45, f' age only {age_auc:.3f}', color=S.C['age'], fontsize=6, va='bottom')
    for arm, off in (('noH', -0.13), ('H', 0.13)):
        d = lad[lad.arm == arm].set_index('rung')
        for yi, r in enumerate(rungs):
            if r not in d.index:
                continue
            q = d.loc[r]
            ax.plot([q.lo, q.hi], [yi + off] * 2, color=S.C[arm], lw=1.6, solid_capstyle='round', zorder=3)
            ax.plot(q.auc, yi + off, 'o', ms=5.5, mfc=S.C[arm], mec='white', mew=0.8, zorder=4,
                    label=arm if yi == 0 else None)
            ax.text(q.hi + 0.004, yi + off, f'{q.auc:.3f}', va='center', fontsize=6, color=S.C[arm])
    ax.set_yticks(range(len(rungs)), [RUNG_LABEL[r] for r in rungs])
    ax.invert_yaxis()
    ax.set_xlabel('First-visit AUC (95% patient-cluster bootstrap CI)')
    ax.legend(loc='lower right', title='Arm', title_fontsize=6.5)
    txt = '\n'.join(f'{a}  {s}: Δ {x.d:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}]'
                    for a in ('noH', 'H') for s, x in dl[dl.arm == a].set_index('step').iterrows())
    ax.text(1.02, 0.5, 'Paired Δ AUC, first visit\n' + txt, transform=ax.transAxes, fontsize=5.8, va='center',
            family='monospace', color=S.C['ink'])
    fig.tight_layout()
    lad.to_csv(TAB / 'ladder.csv', index=False)
    dl.to_csv(TAB / 'ladder_deltas.csv', index=False)
    return S.save(fig, 'ladder')
