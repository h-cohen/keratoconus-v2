"""Appendix C: clinical reading atlas. Where on the cornea the model looks, and which direction of each corneal
measurement raises the referral score. Aggregates over first-visit eyes only (no individual eye is drawn).

Location importance of a sector = mean |SHAP| of its R feature over first-visit eyes with the map, summed over the
three sides (own eye, fellow eye, own − fellow). Direction = Spearman rho between the feature value and its SHAP value.
The value where the rolling-median SHAP trend crosses 0 is reported as a descriptive reading point, not a validated
clinical threshold."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Wedge

import age_ops as AO
import plots as PL
import style as S

E = PL.E
TAB = PL.TAB
SIDE_LAB = {'own': 'own eye', 'fel': 'fellow eye', 'asym': 'own − fellow'}
MAPS = PL.MAP_ORDER
MIN_N = 10  # subgroups with fewer first-visit eyes are not drawn


def sector_importance(sh, rows):
    """{(map, r0, r1, q): mean |SHAP| summed over sides} and the per-column table for `rows`."""
    imp = PL.col_importance(sh, rows)
    out, recs = {}, []
    for j, c in enumerate(sh['cols']):
        if not c.startswith('R_'):
            continue
        _, side, rn = E._split(c)
        sec = E.sector(rn)
        recs.append(dict(col=c, side=side, rname=rn, sector=sec, imp=imp[j]))
        if sec:
            out[sec] = out.get(sec, 0.0) + imp[j]
    return out, pd.DataFrame(recs)


def direction(sh, rows, j):
    r = rows & sh['mask'][:, j] & np.isfinite(sh['raw'][:, j])
    if r.sum() <= 10:
        return np.nan
    return pd.Series(sh['raw'][r, j]).rank().corr(pd.Series(sh['shap'][r, j]).rank())


def dirword(rho):
    """Direction of a feature from the Spearman rho of value vs SHAP; |rho| < 0.3 = no single direction."""
    if not np.isfinite(rho) or abs(rho) < 0.3:
        return 'non-monotone'
    return 'higher → referral' if rho > 0 else 'lower → referral'


def outline(ax, sec, lw=1.3, color=None, ls='-'):
    _, r0, r1, q = sec
    t0, t1 = PL.QUAD_DEG[q]
    ax.add_patch(Wedge((0, 0), r1, t0, t1, width=r1 - r0, fc='none', ec=color or S.C['ink'], lw=lw, ls=ls, zorder=5))


def orient(ax, size=5.5):
    for s_, (x, y) in {'S': (0, 4.6), 'I': (0, -4.6), 'N': (4.6, 0), 'T': (-4.6, 0)}.items():
        ax.text(x, y, s_, ha='center', va='center', fontsize=size, color=S.C['mute'])


def colorbar(fig, rect, vmax, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=S.SEQ, norm=Normalize(0, vmax)), cax=cax, extend='max')
    cb.set_label(label, fontsize=6.3)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)


# ---------------------------------------------------------------- C1 where to look
def fig_reading_map(sh, info, arm, top=10):
    """One row of four maps; the `top` sectors (all maps together) numbered by rank, with a reading list."""
    fv = info['first']
    vals, tab = sector_importance(sh, fv)
    ranked = sorted(vals, key=vals.get, reverse=True)[:top]
    vmax = np.nanpercentile(list(vals.values()), 98)
    rows = []
    for k, sec in enumerate(ranked, 1):
        d = tab[tab.sector == sec].sort_values('imp', ascending=False)
        lead = d.iloc[0]
        rho = direction(sh, fv, sh['cols'].index(lead.col))
        share = {s: float(d[d.side == s].imp.sum() / d.imp.sum()) for s in SIDE_LAB}
        rows.append(dict(rank=k, map=dict(MAPS)[sec[0]], location=f'{sec[1]}–{sec[2]} mm {E.QUAD[sec[3]]}',
                         importance=vals[sec], leading_side=SIDE_LAB[lead.side], rho_leading=rho,
                         direction=dirword(rho),
                         **{f'share_{s}': share[s] for s in SIDE_LAB}, sec=sec))
    rt = pd.DataFrame(rows)
    fig = plt.figure(figsize=(S.W2, 3.55))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.78], hspace=0.08, wspace=0.06)
    for k, (mp, mlab) in enumerate(MAPS):
        ax = fig.add_subplot(gs[0, k])
        PL.dartboard(ax, {s[1:]: v for s, v in vals.items() if s[0] == mp}, vmax=vmax)
        ax.set_title(mlab, loc='center', fontsize=8)
        if k == 0:
            orient(ax)
        for r in rt[rt.sec.map(lambda s: s[0]) == mp].itertuples():
            _, r0, r1, q = r.sec
            outline(ax, r.sec)
            t = np.deg2rad(np.mean(PL.QUAD_DEG[q]))
            rr = (r0 + r1) / 2
            ax.text(rr * np.cos(t), rr * np.sin(t), str(r.rank), ha='center', va='center', fontsize=7,
                    fontweight='bold', color='white' if r.importance > 0.55 * vmax else S.C['ink'], zorder=6)
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    half = (len(rt) + 1) // 2
    for i, r in enumerate(rt.itertuples()):
        x, y = (0.0 if i < half else 0.5), 0.95 - (i % half) * 0.19
        ax.text(x, y, f'{r.rank:>2}.', fontsize=6.6, fontweight='bold', transform=ax.transAxes, va='top')
        ax.text(x + 0.025, y, f'{r.map}, {r.location}', fontsize=6.6, transform=ax.transAxes, va='top')
        ax.text(x + 0.025, y - 0.085, f'mostly {r.leading_side}; {r.direction} (ρ {r.rho_leading:+.2f})',
                fontsize=5.9, color=S.C['mute'], transform=ax.transAxes, va='top')
    colorbar(fig, [0.93, 0.5, 0.01, 0.33], vmax, 'Mean |SHAP|\n(own + fellow + own − fellow)')
    fig.text(0.5, -0.01, f'{arm} arm, {S.W["unit"]}. Numbers: the {top} most important sectors over all four maps '
             '(rings 0–1 … 3–4 mm × quadrants; +x nasal, left eyes mirrored). ρ: Spearman correlation of value '
             'and SHAP for the side that carries most of the sector.', ha='center', fontsize=5.8, color=S.C['mute'],
             wrap=True)
    fig.subplots_adjust(left=0.02, right=0.9, top=0.92, bottom=0.04)
    rt.drop(columns='sec').to_csv(S.tab(f'reading_map_{arm}.csv'), index=False)
    return S.save(fig, f'reading_map_{arm}')


# ---------------------------------------------------------------- C2 atlas by subgroup
def subgroups(info, by):
    fv = info['first']
    if by == 'label':
        return [('Referred', fv & (info['y'] == 1)), ('Not referred', fv & (info['y'] == 0))]
    if by == 'stageA':
        return [(f'Stage A{k}\nKm F {PL.STAGE_LAB["A"][k]} D', fv & (info['A'] == k)) for k in (1, 2, 3, 4)]
    if by == 'stageB':
        return [(f'Stage B{k}\nRm B {PL.STAGE_LAB["B"][k]} mm', fv & (info['B'] == k)) for k in (1, 2, 3, 4)]
    if by == 'age':
        g = AO.age_group(info['age'])
        return [(f'Age group {k}\n{AO.AGE_LAB[k]} years', fv & (g == k)) for k in (1, 2, 3, 4)]
    raise ValueError(by)


def fig_atlas(sh, info, arm, by):
    """Rows: subgroups; columns: the four maps. Colour = the sector's share (%) of the group's total sector importance,
    so groups with very different SHAP magnitudes (advanced stages carry far more) read on one scale. Outlined: top 3
    sectors of each panel; the group's total mean |SHAP| over all sectors is printed beside it."""
    grp = [(lab, r) for lab, r in subgroups(info, by) if r.sum() >= MIN_N]
    tot = {lab: sum(sector_importance(sh, r)[0].values()) for lab, r in grp}
    allv = {lab: {k: 100 * v / tot[lab] for k, v in sector_importance(sh, r)[0].items()} for lab, r in grp}
    vmax = np.nanpercentile([v for d in allv.values() for v in d.values()], 98)
    fig = plt.figure(figsize=(S.W2, 1.62 * len(grp) + 0.45))
    gs = fig.add_gridspec(len(grp), 5, width_ratios=[0.55, 1, 1, 1, 1], hspace=0.1, wspace=0.04)
    recs = []
    for i, (lab, r) in enumerate(grp):
        vals = allv[lab]
        ax = fig.add_subplot(gs[i, 0])
        ax.axis('off')
        ax.text(0.98, 0.6, lab, ha='right', va='center', fontsize=7, fontweight='bold', linespacing=1.3)
        ax.text(0.98, 0.3, f'{int(r.sum())} eyes, {int(info["y"][r].sum())} referred\ntotal |SHAP| {tot[lab]:.2f}',
                ha='right', va='center', fontsize=5.8, color=S.C['mute'], linespacing=1.3)
        for k, (mp, mlab) in enumerate(MAPS):
            ax = fig.add_subplot(gs[i, k + 1])
            v = {s[1:]: x for s, x in vals.items() if s[0] == mp}
            PL.dartboard(ax, v, vmax=vmax)
            for s in sorted(v, key=v.get, reverse=True)[:3]:
                outline(ax, (mp,) + s)
            if i == 0:
                ax.set_title(mlab, loc='center', fontsize=8)
            recs += [dict(group=lab, map=mp, r0=s[0], r1=s[1], quadrant=s[2], share_pct=x, group_total=tot[lab])
                     for s, x in v.items()]
    colorbar(fig, [0.92, 0.35, 0.01, 0.3], vmax, 'Share of the group\'s sector importance (%)')
    fig.text(0.5, 0.005, f'{arm} arm, {S.W["unit"]} with the map. Colour: sector share of the group total (64 '
             'sectors, own + fellow + own − fellow). Outlined: top 3 per panel. Superior up, nasal right (left eyes '
             f'mirrored). Groups with < {MIN_N} eyes not drawn.',
             ha='center', fontsize=5.8, color=S.C['mute'])
    fig.subplots_adjust(left=0.01, right=0.9, top=0.95, bottom=0.045)
    pd.DataFrame(recs).to_csv(S.tab(f'atlas_{by}_{arm}.csv'), index=False)
    return S.save(fig, f'atlas_{by}_{arm}')


# ---------------------------------------------------------------- C3 top corneal features, one panel each
def crossing(tx, ty):
    """First x where the rolling-median trend changes sign (linear interpolation); NaN if it never does."""
    ok = np.isfinite(ty)
    tx, ty = tx[ok], ty[ok]
    s = np.sign(ty)
    k = np.flatnonzero(s[:-1] * s[1:] < 0)
    if not len(k):
        return np.nan
    i = k[0]
    return float(tx[i] - ty[i] * (tx[i + 1] - tx[i]) / (ty[i + 1] - ty[i]))


def mini_board(ax, sec, side):
    """Inset locator: the sector in navy on a grey dartboard."""
    PL.dartboard(ax, {s: 0.0 for s in [(r, r + 1, q) for r in range(4) for q in 'SINT']}, vmax=1,
                 cmap=plt.matplotlib.colors.ListedColormap(['#E9ECF3']), edge='white')
    if sec:
        _, r0, r1, q = sec
        t0, t1 = PL.QUAD_DEG[q]
        ax.add_patch(Wedge((0, 0), r1, t0, t1, width=r1 - r0, fc=S.C['noH'], ec='white', lw=0.5))
    else:  # summary measure of the whole map (position, extreme value, asymmetry)
        ax.add_patch(Wedge((0, 0), 4, 0, 360, width=4, fc='none', ec=S.C['noH'], lw=1.2))
        ax.text(0, 0, 'whole\nmap', ha='center', va='center', fontsize=4.6, color=S.C['noH'])
    ax.text(0, -5.6, {'own': 'own', 'fel': 'fellow', 'asym': 'own−fel'}[side], ha='center', va='top', fontsize=5,
            color=S.C['mute'])


def fig_r_gallery(sh, info, arm, top=12):
    """The `top` corneal (R) features by mean |SHAP|: SHAP vs value (referred navy, not referred grey), rolling-median
    trend, the value where the trend crosses 0, and a locator dartboard."""
    fv = info['first']
    imp = PL.col_importance(sh, fv)
    rj = [j for j in np.argsort(imp)[::-1] if sh['cols'][j].startswith('R_')][:top]
    nc = 4
    nr = int(np.ceil(len(rj) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(S.W2, 2.05 * nr))
    recs = []
    for n, (ax, j) in enumerate(zip(axes.flat, rj)):
        c = sh['cols'][j]
        _, side, rn = E._split(c)
        r = fv & sh['mask'][:, j] & np.isfinite(sh['raw'][:, j])
        x, z, y = sh['raw'][r, j].astype(float), sh['shap'][r, j], info['y'][r]
        lo, hi = np.percentile(x, [1, 99])
        ax.axhline(0, color=S.C['chance'], lw=0.6)
        ax.scatter(x[y == 0], z[y == 0], s=3.5, lw=0, color=AO.NOREF_COL, alpha=0.8, rasterized=True)
        ax.scatter(x[y == 1], z[y == 1], s=4.5, lw=0, color=S.C['noH'], alpha=0.85, rasterized=True)
        tx, ty = PL._trend(x, z)
        ax.plot(tx, ty, color=S.C['ink'], lw=1.1)
        x0 = crossing(tx, ty)
        if np.isfinite(x0) and lo <= x0 <= hi:
            ax.axvline(x0, color=S.C['H'], lw=0.9, ls=(0, (3, 2)))
            ax.text(x0, 1.0, f' {x0:.3g}', color=S.C['H'], fontsize=5.6, va='top', ha='left',
                    transform=ax.get_xaxis_transform())
        ax.set_xlim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
        ax.set_title(f'{n + 1}. ' + textwrap.fill(E._r_label(rn), 30), loc='left', fontsize=6.2, pad=3)
        ax.set_xlabel(SIDE_LAB[side], fontsize=5.8, labelpad=1)
        ax.tick_params(labelsize=5.5)
        if n % nc == 0:
            ax.set_ylabel('SHAP value', fontsize=6.3)
        ins = ax.inset_axes([0.78, 0.68, 0.22, 0.3] if np.corrcoef(x, z)[0, 1] < 0 else [0.0, 0.68, 0.22, 0.3])
        mini_board(ins, E.sector(rn), side)
        rho = direction(sh, fv, j)
        recs.append(dict(rank=n + 1, feature=E._r_label(rn), side=SIDE_LAB[side], mean_abs_shap=imp[j], rho=rho,
                         direction=dirword(rho), trend_zero_crossing=x0,
                         median_referred=np.median(x[y == 1]), median_not_referred=np.median(x[y == 0]),
                         n_eyes=int(r.sum())))
    for ax in list(axes.flat)[len(rj):]:
        ax.axis('off')
    fig.legend(handles=[Patch(color=S.C['noH'], label='Referred'), Patch(color=AO.NOREF_COL, label='Not referred'),
                        plt.Line2D([], [], color=S.C['ink'], lw=1.1, label='Rolling median'),
                        plt.Line2D([], [], color=S.C['H'], lw=0.9, ls=(0, (3, 2)), label='Trend crosses 0')],
               loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=6.3)
    fig.text(0.5, -0.005, f'{arm} arm, {S.W["unit"]} with the map; x axis in the map unit (µm or D; positions in '
             'mm). SHAP > 0 raises the referral score. Locator: the sector in navy, for the side named under the axis.',
             ha='center', fontsize=5.8, color=S.C['mute'])
    fig.tight_layout(rect=(0, 0.01, 1, 0.97), h_pad=1.0, w_pad=0.8)
    pd.DataFrame(recs).to_csv(S.tab(f'reading_guide_{arm}.csv'), index=False)
    return S.save(fig, f'r_gallery_{arm}')
