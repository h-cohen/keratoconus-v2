"""E30 (PROTOCOL.md 2026-11-15): age groups and operating characteristics of the final blends, both arms.
Descriptive only: no model is refitted. Aggregates only; every figure writes its numbers to tables/."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import plots as PL
import style as S

AGE_CUTS = (22, 27, 34)                       # first-visit age quartiles, rounded (pre-declared)
AGE_LAB = {1: '< 22', 2: '22–26', 3: '27–33', 4: '≥ 34'}
PL.STAGE_LAB['age_group'] = AGE_LAB
PL.STAGE_TITLE['age_group'] = 'Age group (years)'
REF_COL, NOREF_COL = '#34404D', '#BFC5CE'     # referred / not referred (data, not arms)
SLATE = '#4F6272'
TAB = PL.TAB


def age_group(age):
    """1..4 for < 22, 22-26, 27-33, >= 34 years."""
    a = np.asarray(age, float)
    return np.select([a < AGE_CUTS[0], a < AGE_CUTS[1], a < AGE_CUTS[2]], [1, 2, 3], 4)


def wilson(k, n, z=1.96):
    """Wilson 95% interval of a proportion k / n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def within_group_auc(y, p, grp):
    """AUC over referred / not-referred pairs from the same group only (each pair weighted equally)."""
    num = den = 0.0
    for k in np.unique(grp):
        m = grp == k
        npos = int(y[m].sum())
        nneg = int(m.sum()) - npos
        if npos and nneg:
            w = npos * nneg
            num += roc_auc_score(y[m], p[m]) * w
            den += w
    return num / den if den else np.nan


def _boot(stat, g, n=1000, seed=0):
    """Patient-cluster bootstrap of stat(idx) -> float; returns the 2.5 / 97.5 percentiles."""
    rng = np.random.default_rng(seed)
    pids = np.unique(g)
    rows = {u: np.flatnonzero(g == u) for u in pids}
    out = []
    for _ in range(n):
        v = stat(np.concatenate([rows[u] for u in rng.choice(pids, len(pids))]))
        if np.isfinite(v):
            out.append(v)
    return tuple(np.percentile(out, [2.5, 97.5]))


def within_group_ci(y, p, grp, g):
    return (within_group_auc(y, p, grp),) + _boot(lambda i: within_group_auc(y[i], p[i], grp[i]), g)


def operating_point(y, p, target):
    """Highest threshold t with sensitivity(p >= t) >= target; sens, spec, PPV, NPV and eyes flagged at t."""
    pos = np.sort(p[y == 1])[::-1]
    t = pos[int(np.ceil(target * len(pos) - 1e-9)) - 1]
    f = p >= t
    tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
    fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
    return dict(threshold=t, sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / max(tp + fp, 1),
                npv=tn / max(tn + fn, 1), flagged=tp + fp)


# ------------------------------------------------------------------------------------------------ tables
def age_cohort_table(info):
    fv, y, grp, st = info['first'], info['y'], info['age_group'], info['A']
    rows = []
    for k, lab in AGE_LAB.items():
        m = fv & (grp == k)
        n, r = int(m.sum()), int(y[m].sum())
        lo, hi = wilson(r, n)
        row = dict(group=f'{k}: {lab}', eyes=n, referred=r, rate=r / n, lo=lo, hi=hi,
                   age_median=float(np.median(info['age'][m])), KmF_median=float(np.nanmedian(info['km'][m])))
        for s in (1, 2, 3, 4):
            ms = m & (st == s)
            row[f'A{s} referred/eyes'] = f'{int(y[ms].sum())}/{int(ms.sum())}'
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(S.tab('age_cohort.csv'), index=False)
    return df


def age_adjusted_table(y, g, fv, grp, scores):
    """Crude and within-age-group AUC (first visit) per method; scores: {(arm, method): oof}."""
    rows = []
    for (arm, meth), p in scores.items():
        a = PL.auc_ci(y[fv], p[fv], g[fv])
        w = within_group_ci(y[fv], p[fv], grp[fv], g[fv])
        rows.append(dict(arm=arm, method=meth, auc=a[0], lo=a[1], hi=a[2], auc_within=w[0], lo_within=w[1],
                         hi_within=w[2]))
    df = pd.DataFrame(rows)
    df.to_csv(S.tab('age_adjusted.csv'), index=False)
    return df


def operating_table(y, g, fv, scores, targets=(0.80, 0.90, 0.95)):
    rows = []
    yf, gf = y[fv], g[fv]
    for lab, p in scores.items():
        pf = p[fv]
        for t in targets:
            r = operating_point(yf, pf, t)
            th = r['threshold']
            lo, hi = _boot(lambda i: ((pf[i] < th) & (yf[i] == 0)).sum() / max((yf[i] == 0).sum(), 1), gf)
            rows.append({'method': lab, 'target sensitivity': f'{t:.2f}', 'sensitivity': round(r['sens'], 3),
                         'specificity [95% CI]': f'{r["spec"]:.3f} [{lo:.3f}–{hi:.3f}]', 'PPV': round(r['ppv'], 3),
                         'NPV': round(r['npv'], 3), f'{S.W["n"]} flagged': f'{r["flagged"]}/{int(fv.sum())}'})
    df = pd.DataFrame(rows)
    df.to_csv(S.tab('operating_points.csv'), index=False)
    return df


# ------------------------------------------------------------------------------------------------ figures
def _cut_lines(ax, top_labels=True):
    for c in AGE_CUTS:
        ax.axvline(c, color='#8A8A8A', lw=0.7, ls=(0, (3, 2)), zorder=1)
    if top_labels:
        edges = (14,) + AGE_CUTS + (70,)
        for k in range(4):
            ax.text((edges[k] + edges[k + 1]) / 2 if k < 3 else 40, 1.0, f'{k + 1}', transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=6.3, color=S.C['mute'], fontweight='bold')


def fig_age_cohort(info):
    """a: first-visit age histogram by label; b: referral rate per age group (Wilson CI); c: age group x stage A."""
    tab = age_cohort_table(info)
    fv, y, age = info['first'], info['y'], info['age']
    fig = plt.figure(figsize=(S.W2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.95, 1.15], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    bins = np.arange(14, 72, 2)
    ax.hist([age[fv & (y == 1)], age[fv & (y == 0)]], bins=bins, stacked=True, color=[REF_COL, NOREF_COL],
            edgecolor='white', lw=0.4, label=['Referred', 'Not referred'], zorder=2)
    _cut_lines(ax)
    ax.set(xlabel='Age at first visit (years)', ylabel='Eyes', xlim=(14, 70))
    ax.legend(loc='upper right', fontsize=6.3, handlelength=1.2)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    S.panel_letter(ax, 'a', x=-0.2)

    ax = fig.add_subplot(gs[1])
    cols = PL.shades(SLATE)
    for k, r in enumerate(tab.itertuples(), 1):
        ax.bar(k, r.rate, width=0.64, color=cols[k - 1], edgecolor='#333333', lw=0.6, zorder=2)
        ax.errorbar(k, r.rate, yerr=[[r.rate - r.lo], [r.hi - r.rate]], fmt='none', ecolor='#333333', lw=0.7,
                    capsize=2.2, zorder=3)
        ax.text(k, r.hi + 0.02, f'{r.rate:.0%}', ha='center', va='bottom', fontsize=6.3, fontweight='bold')
    ax.set_xticks(range(1, 5), [f'{AGE_LAB[k]}\n{r.referred}/{r.eyes}' for k, r in zip(AGE_LAB, tab.itertuples())],
                  fontsize=6.1)
    ax.set(ylim=(0, 0.62), ylabel='Referred (first visit)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    S.panel_letter(ax, 'b', x=-0.32)

    ax = fig.add_subplot(gs[2])
    M = np.full((4, 4), np.nan)
    txt = [['' for _ in range(4)] for _ in range(4)]
    for k in range(4):
        for s in range(4):
            m = fv & (info['age_group'] == k + 1) & (info['A'] == s + 1)
            if m.sum():
                M[k, s] = y[m].mean()
                txt[k][s] = f'{y[m].mean():.0%}\n{int(y[m].sum())}/{int(m.sum())}'
    im = ax.imshow(np.ma.masked_invalid(M), cmap=S.SEQ, vmin=0, vmax=1.0, aspect='auto')
    for k in range(4):
        for s in range(4):
            ax.text(s, k, txt[k][s] or '—', ha='center', va='center', fontsize=5.6,
                    color='white' if np.nan_to_num(M[k, s]) > 0.55 else S.C['ink'])
    ax.set_xticks(range(4), [f'A{s}\n{PL.STAGE_LAB["A"][s]}' for s in range(1, 5)], fontsize=6)
    ax.set_yticks(range(4), [AGE_LAB[k] for k in range(1, 5)], fontsize=6.2)
    ax.set_xlabel('Stage A (Km F, D)')
    ax.set_ylabel('Age group (years)')
    ax.set_facecolor('#EEEEEE')
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label('Referred', fontsize=6.2)
    cb.ax.tick_params(labelsize=5.8)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    cb.outline.set_linewidth(0.4)
    S.panel_letter(ax, 'c', x=-0.33)
    fig.text(0.5, -0.1, 'First-visit eyes (one per eye). Dashed lines and numbers: age groups 1–4 (first-visit age '
             'quartiles, pre-declared). Whiskers: Wilson 95% CI. Cells: referral rate, referred / eyes.',
             ha='center', fontsize=6.1, color=S.C['mute'])
    return S.save(fig, 'age_cohort')


def fig_age_roc(y, g, fv, P, grp):
    """First-visit ROC of the final model per age group, one panel per arm."""
    fig, axes = plt.subplots(1, 2, figsize=(S.W2, 3.3))
    rows = []
    for ax, (a, p), letter in zip(axes, P.items(), 'ab'):
        PL.framed(ax)
        ax.plot([0, 1], [0, 1], color='#9A9A9A', lw=0.8, ls=(0, (1, 2)), zorder=1)
        cols = PL.shades(S.C[a], 5)[1:]
        for k, lab in AGE_LAB.items():
            m = fv & (grp == k)
            yy, pp = y[m], p[m]
            auc, lo, hi = PL.auc_ci(yy, pp, g[m])
            small = yy.sum() < 10 or (1 - yy).sum() < 10
            f, t, _ = roc_curve(yy, pp)
            ci = '' if small else f' [{lo:.2f}–{hi:.2f}]'
            ax.plot(f, t, color=cols[k - 1], lw=1.6, ls=(0, (3, 1.5)) if small else '-', zorder=3,
                    label=f'{lab} y ({int(yy.sum())}/{int(m.sum())})  AUC {auc:.2f}{ci}')
            rows.append(dict(arm=a, group=lab, n=int(m.sum()), pos=int(yy.sum()), auc=auc,
                             lo=np.nan if small else lo, hi=np.nan if small else hi))
        f, t, _ = roc_curve(y[fv], p[fv])
        ax.plot(f, t, color=S.C['ink'], lw=0.9, ls=(0, (1, 1.2)), zorder=4,
                label=f'All ages  AUC {roc_auc_score(y[fv], p[fv]):.2f}')
        ax.set(xlim=(0, 1), ylim=(0, 1.005), aspect='equal', xlabel='False positive rate (1 − specificity)',
               ylabel='True positive rate (sensitivity)')
        ax.set_title(f'Final model, {a} arm', color=S.C[a], loc='center')
        lg = ax.legend(loc='lower right', fontsize=5.6, frameon=True, fancybox=False, edgecolor='#CCCCCC',
                       framealpha=0.95, title='Age group (referred / eyes)', title_fontsize=5.8)
        lg.get_frame().set_linewidth(0.5)
        S.panel_letter(ax, letter, x=-0.2, y=1.02)
    fig.text(0.5, -0.01, 'First visit. Dashed: < 10 referred or < 10 not referred, no CI. CI: 95% patient-cluster '
             'bootstrap.', ha='center', fontsize=6.2, color=S.C['mute'])
    fig.tight_layout(w_pad=2.5)
    pd.DataFrame(rows).to_csv(S.tab('roc_age.csv'), index=False)
    return S.save(fig, 'roc_age')


def fig_age_adjusted(df):
    """Crude vs within-age-group AUC per method, one panel per arm."""
    meths = list(dict.fromkeys(df.method))
    fig, axes = plt.subplots(1, 2, figsize=(S.W2, 2.35), sharey=True)
    for ax, a, letter in zip(axes, ('noH', 'H'), 'ab'):
        d = df[df.arm == a].set_index('method')
        PL.framed(ax, 'x')
        ax.axvline(0.5, color='#9A9A9A', lw=0.8, ls=(0, (1, 2)))
        for i, m in enumerate(meths):
            r = d.loc[m]
            col = PL.METHOD_COL.get(m if m != 'Final' else f'Final, {a}', S.C['L2b'])
            ax.errorbar(r.auc, i - 0.14, xerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt='o', mfc='white', mec=col,
                        ecolor=col, ms=4.5, lw=0.9, capsize=1.8)
            ax.errorbar(r.auc_within, i + 0.14, xerr=[[r.auc_within - r.lo_within], [r.hi_within - r.auc_within]],
                        fmt='o', color=col, ms=4.5, lw=0.9, capsize=1.8)
            ax.text(1.13, i, f'{r.auc:.3f} → {r.auc_within:.3f}', ha='right', va='center', fontsize=6,
                    color=S.C['ink'])
        ax.set_yticks(range(len(meths)), meths)
        ax.set(xlim=(0.35, 1.15), xlabel='ROC-AUC, first visit (95% CI)')
        ax.set_xticks(np.arange(0.4, 1.01, 0.1))
        ax.axvline(1.0, color='#CCCCCC', lw=0.6)
        ax.invert_yaxis()
        ax.set_title(f'{a} arm', color=S.C[a], loc='center')
        S.panel_letter(ax, letter, x=-0.5 if letter == 'a' else -0.06, y=1.02)
    h = [Line2D([], [], marker='o', ls='', mfc='white', mec=S.C['ink'], label='Crude AUC'),
         Line2D([], [], marker='o', ls='', color=S.C['ink'], label='Age-adjusted (pairs within the same age group)')]
    fig.legend(handles=h, loc='upper center', ncol=2, bbox_to_anchor=(0.55, 1.08), fontsize=6.4)
    fig.tight_layout(w_pad=1.2)
    return S.save(fig, 'age_adjusted')


def fig_age_shap(shs, info):
    """SHAP of the age column against age (first visit), per arm, coloured by label, with a rolling median."""
    fig, axes = plt.subplots(1, 2, figsize=(S.W2, 2.35), sharey=True)
    fv, y = info['first'], info['y']
    rows = []
    for ax, (a, sh), letter in zip(axes, shs.items(), 'ab'):
        j = sh['cols'].index('age')
        x, z = sh['raw'][fv, j].astype(float), sh['shap'][fv, j]
        ax.axhline(0, color=S.C['chance'], lw=0.6)
        _cut_lines(ax)
        for lab, col, sel in (('Not referred', NOREF_COL, y[fv] == 0), ('Referred', REF_COL, y[fv] == 1)):
            ax.scatter(x[sel], z[sel], s=5, color=col, lw=0, alpha=0.85, label=lab, rasterized=True, zorder=2)
        tx, ty = PL._trend(x, z)
        ax.plot(tx, ty, color=S.C[a], lw=1.6, zorder=3, label='Rolling median')
        ax.set(xlabel=f'Age at visit ({S.W["scope"]}, years)', xlim=(14, 70))
        ax.set_title(f'{a} arm', color=S.C[a], loc='center', pad=14)
        S.panel_letter(ax, letter, x=-0.16, y=1.06)
        for k in AGE_LAB:
            m = info['age_group'][fv] == k
            rows.append(dict(arm=a, group=AGE_LAB[k], mean_shap_age=float(z[m].mean()),
                             mean_abs_shap_age=float(np.abs(z[m]).mean())))
    axes[0].set_ylabel('SHAP of age (member-standardised)')
    axes[1].legend(loc='upper right', fontsize=6.2, markerscale=1.8)
    fig.text(0.5, -0.04, f'{S.W["scope"].capitalize()}. Above 0: age raises the referral score.', ha='center', fontsize=6.1,
             color=S.C['mute'])
    fig.tight_layout(w_pad=1.2)
    pd.DataFrame(rows).to_csv(S.tab('shap_age.csv'), index=False)
    return S.save(fig, 'shap_age')


def fig_operating(y, g, fv, curves, thr_scores):
    """a: precision-recall (first visit); b, c: score distributions by label per arm with the 0.90-sensitivity
    threshold. curves: {label: (score, colour, linestyle)}; thr_scores: {arm: score}."""
    fig = plt.figure(figsize=(S.W2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    PL.framed(ax)
    yf = y[fv]
    prev = yf.mean()
    ax.axhline(prev, color='#9A9A9A', lw=0.8, ls=(0, (1, 2)))
    ax.text(0.98, prev - 0.025, f'prevalence {prev:.2f}', fontsize=5.8, color=S.C['mute'], ha='right', va='top')
    rows = []
    for lab, (p, col, ls) in curves.items():
        pr, rc, _ = precision_recall_curve(yf, p[fv])
        ap = average_precision_score(yf, p[fv])
        lo, hi = _boot(lambda i: average_precision_score(yf[i], p[fv][i]) if yf[i].min() < yf[i].max() else np.nan,
                       g[fv])
        ax.plot(rc, pr, color=col, ls=ls, lw=1.7 if ls == '-' else 1.1, label=f'{lab}  AP {ap:.2f}')
        rows.append(dict(method=lab, ap=ap, lo=lo, hi=hi, prevalence=prev))
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Recall (sensitivity)', ylabel='Precision (PPV)')
    ax.legend(loc='upper right', fontsize=5.6, frameon=True, fancybox=False, edgecolor='#CCCCCC', framealpha=0.95)
    ax.set_title(f"Precision–recall, {S.W['scope']}", loc='center', fontsize=7.5)
    S.panel_letter(ax, 'a', x=-0.22, y=1.02)
    pd.DataFrame(rows).to_csv(S.tab('precision_recall.csv'), index=False)
    for (a, p), letter, k in zip(thr_scores.items(), 'bc', (1, 2)):
        ax = fig.add_subplot(gs[k])
        pf = p[fv]
        bins = np.linspace(pf.min(), pf.max(), 26)
        ax.hist(pf[yf == 0], bins=bins, histtype='stepfilled', color=NOREF_COL, alpha=0.75, density=True, zorder=2)
        ax.hist(pf[yf == 0], bins=bins, histtype='step', color='#7D8794', lw=1.0, density=True, label='Not referred',
                zorder=4)
        ax.hist(pf[yf == 1], bins=bins, histtype='stepfilled', color=REF_COL, alpha=0.22, density=True, zorder=3)
        ax.hist(pf[yf == 1], bins=bins, histtype='step', color=REF_COL, lw=1.0, density=True, label='Referred',
                zorder=4)
        t = operating_point(yf, pf, 0.90)['threshold']
        ax.axvline(t, color=S.C[a], lw=1.2, ls=(0, (4, 2)), zorder=5)
        ax.text(t, 1.0, ' sensitivity 0.90', transform=ax.get_xaxis_transform(), fontsize=5.8, color=S.C[a],
                va='bottom', ha='center')
        ax.set(xlabel=f'Final score, {a} arm (rank)', ylabel='Density' if k == 1 else '')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        if k == 2:
            ax.legend(loc='upper left', fontsize=5.8, handlelength=1.2)
        S.panel_letter(ax, letter, x=-0.08, y=1.02)
    fig.text(0.5, -0.06, f"{S.W['scope'].capitalize()} ({int(fv.sum())} {S.W['n']}). Blend scores are within-fold ranks (not probabilities); the threshold is set "
             'on the same out-of-fold scores (descriptive).', ha='center', fontsize=6.1, color=S.C['mute'])
    return S.save(fig, 'operating')
