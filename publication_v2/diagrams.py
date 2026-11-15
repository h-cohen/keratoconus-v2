"""Method diagrams for the paper (E29): the data flow with tensor sizes, and the cohort / training-row filtering funnels.

Funnel counts come from the data-preparation logs: kcref/data/cohort_report.txt (prepare_data.py) and the printout of
prepare_extra.py, plus the raw counts of data/multimodal_dataset.csv. They are constants here because recomputing them
needs the identifiable source exports; test_figures.test_funnel_matches_cohort checks the final row against the cohort.
"""
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
import style as S  # noqa: E402

S.apply()

# (label, scans, eyes, patients, positive scans, reason for the cut that leads to this row)
COHORT_FUNNEL = [
    ('Image export, all scans', 1218, 677, 481, 283, None),
    ('All four maps present', 1017, 574, 413, 232, 'one or more of the four maps missing in the composite'),
    ('Complete numerics', 1017, 574, 413, 232, 'missing Pentacam numeric (none)'),
    ('Km F ≤ 55 D', 997, 560, 401, 220, 'inclusion range of the original study (advanced cones)'),
    ('Study cohort', 944, 523, 385, 167, 'scan on/after the eye’s CXL date (already treated)'),
]
FIRST_VISIT = (523, 125)
EXTRA_FUNNEL = [
    ('Raw Pentacam export', 11760, None, None, None, None),
    ('Patients of the CXL clinic', 4683, None, None, None,
     'patient not followed in the CXL clinic: label not comparable'),
    ('Complete numerics, Km F ≤ 55 D', 4167, None, None, None, 'same rules as the cohort'),
    ('Before CXL / keratoplasty', 2431, None, None, None, 'exam on/after CXL or keratoplasty, or CXL undated'),
    ('Extra training exams', 1487, None, 503, 299, 'exam is one of the 944 cohort scans'),
]

INK, MUTE, LINE = S.C['ink'], S.C['mute'], '#B8B8B8'
FILL = {'in': '#EEF3F9', 'block': '#FFFFFF', 'model': '#F6EFE4', 'out': '#E7F0E9', 'h': '#F1EBF6'}


def box(ax, x, y, w, h, title, body='', kind='block', edge=None, tsize=7, bsize=5.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012', fc=FILL[kind],
                                ec=edge or LINE, lw=0.9 if edge else 0.7))
    ax.text(x + w / 2, y + h - 0.012, title, ha='center', va='top', fontsize=tsize, fontweight='bold', color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.012 - tsize / 330, body, ha='center', va='top', fontsize=bsize, color=INK,
                linespacing=1.35)


def arrow(ax, p, q, color=MUTE, ls='-', rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, lw=0.8, color=color, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=1, shrinkB=1))


def elbow(ax, pts, color=MUTE, ls='-'):
    """Orthogonal connector through `pts`, arrow head on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=0.8, ls=ls, solid_capstyle='butt')
    arrow(ax, pts[-2], pts[-1], color=color, ls=ls)


def fig_dataflow():
    """One visit -> encoding -> blocks (with sizes) -> design matrix (in-fold preprocessing) -> 3 members -> blend."""
    fig = plt.figure(figsize=(S.W2, 4.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    cA, cD, cB, cX, cM, cS = (0.01, 0.15), (0.19, 0.14), (0.36, 0.165), (0.555, 0.125), (0.71, 0.135), (0.875, 0.115)
    for (x, w), t in ((cA, 'Inputs (one visit)'), (cD, 'Map encoding'), (cB, 'Feature blocks'),
                      (cX, 'Design matrix'), (cM, 'Ensemble members'), (cS, 'Output')):
        ax.text(x + w / 2, 0.955, t, ha='center', fontsize=7, fontweight='bold', color=MUTE)
        ax.plot([x, x + w], [0.935, 0.935], color=LINE, lw=0.6)
    # inputs
    box(ax, cA[0], 0.68, cA[1], 0.22, 'Scanned eye',
        'age, 54 Pentacam numerics\n4 maps (T, K, EF, EB)\n4 × 145 × 145\n0.069 mm per cell', 'in', bsize=5.6)
    box(ax, cA[0], 0.42, cA[1], 0.22, 'Fellow eye',
        'same-day exam\n54 numerics\n4 maps, 4 × 145 × 145\nchosen label-free', 'in', bsize=5.6)
    box(ax, cA[0], 0.22, cA[1], 0.13, 'Treatment sheet', 'fellow-eye CXL /\nkeratoplasty dates\nbefore the visit', 'h',
        edge=S.SRC['H'], bsize=5.6)
    # encoding
    box(ax, cD[0], 0.42, cD[1], 0.24, 'Decode → R',
        'colour → µm or D\nleft eyes mirrored\n(+x nasal)\n96 values per eye:\n64 ring × quadrant\n16 extrema, 8 I−S / N−T\n4 SD, 4 thickness rise',
        'block', bsize=5.4)
    # blocks
    box(ax, cB[0], 0.72, cB[1], 0.18, 'N  ·  205',
        'own: age + 54 + 11 indices (66)\nfellow 65  ·  own − fellow 65\n8 worse-eye flags, has_fellow', 'block',
        edge=S.SRC['own_num'], bsize=5.4)
    box(ax, cB[0], 0.46, cB[1], 0.18, 'R  ·  288', 'own 96  ·  fellow 96\nown − fellow 96', 'block',
        edge=S.SRC['own_maps'], bsize=5.6)
    box(ax, cB[0], 0.355, cB[1], 0.075, 'Flags  ·  2', 'own map, fellow map present', 'block', tsize=6.6, bsize=5.4)
    box(ax, cB[0], 0.22, cB[1], 0.11, 'H  ·  5   (H arm only)', 'fellow CXL before visit, years since,\nkeratoplasty, unknown, missing',
        'h', edge=S.SRC['H'], tsize=6.6, bsize=5.4)
    # design matrix with in-fold preprocessing
    box(ax, cX[0], 0.22, cX[1], 0.68, 'X', 'n × 495   noH\nn × 500   H\n\nn = training rows\nof the fold', 'block',
        edge=INK, tsize=8)
    ax.plot([cX[0] + 0.012, cX[0] + cX[1] - 0.012], [0.60, 0.60], color=LINE, lw=0.6)
    ax.text(cX[0] + cX[1] / 2, 0.58, 'In-fold preprocessing', ha='center', va='top', fontsize=6.4, fontweight='bold')
    ax.text(cX[0] + cX[1] / 2, 0.545, 'clip 0.5 / 99.5 %\nmedian impute\nstandardise R\n\nfitted on the\ntraining rows only',
            ha='center', va='top', fontsize=5.6, linespacing=1.4)
    # members and output
    ys = (0.73, 0.505, 0.28)
    for y, (t, b) in zip(ys, (('XGBoost', 'depth 2–3\n200–500 trees'), ('LightGBM × 5 seeds', 'depth 2, 100–300 trees\nage monotone'),
                              ('Elastic-net LR', 'C 0.01–0.3\nL1 ratio 0.2–0.8'))):
        box(ax, cM[0], y, cM[1], 0.17, t, b, 'model', tsize=6.6, bsize=5.6)
        arrow(ax, (cX[0] + cX[1], y + 0.085), (cM[0], y + 0.085))
        arrow(ax, (cM[0] + cM[1], y + 0.085), (cS[0], y + 0.085))
    box(ax, cS[0], 0.22, cS[1], 0.68, 'Referral\nscore', '\n\neach member →\npercentile rank\n\nmean of the\nthree ranks\n\nscore in [0, 1]',
        'out', edge=S.C['noH'], bsize=5.6)
    # connectors
    xa, xd, xb = cA[0] + cA[1], cD[0], cB[0]
    arrow(ax, (xa, 0.84), (xb, 0.84))                                    # own numerics -> N
    elbow(ax, [(xa, 0.60), (0.172, 0.60), (0.172, 0.76), (xb, 0.76)])     # fellow numerics -> N
    elbow(ax, [(xa, 0.71), (cD[0] + cD[1] / 2, 0.71), (cD[0] + cD[1] / 2, 0.66)])   # own maps -> decode
    arrow(ax, (xa, 0.50), (xd, 0.50))                                    # fellow maps -> decode
    arrow(ax, (cD[0] + cD[1], 0.55), (xb, 0.55))                         # decode -> R
    elbow(ax, [(cD[0] + cD[1] / 2, 0.42), (cD[0] + cD[1] / 2, 0.39), (xb, 0.39)])   # availability -> flags
    arrow(ax, (xa, 0.275), (xb, 0.275), color=S.SRC['H'], ls=(0, (3, 2)))           # sheet -> H
    for y, c, ls in ((0.81, MUTE, '-'), (0.55, MUTE, '-'), (0.39, MUTE, '-'), (0.275, S.SRC['H'], (0, (3, 2)))):
        arrow(ax, (cB[0] + cB[1], y), (cX[0], y), color=c, ls=ls)
    ax.text(0.99, 0.185, 'dashed: H arm only', ha='right', fontsize=5.4, color=S.SRC['H'], style='italic')
    # training panel
    ax.add_patch(FancyBboxPatch((0.01, 0.015), 0.98, 0.15, boxstyle='round,pad=0.004,rounding_size=0.01',
                                fc='#FAFAF8', ec=LINE, lw=0.6))
    ax.text(0.025, 0.15, 'Training and evaluation', fontsize=6.8, fontweight='bold', va='top')
    ax.text(0.025, 0.118,
            'Training rows per outer fold: the fold’s cohort scans + 1,487 extra labelled exams (no maps: R imputed, '
            'own map flag 0); first-visit scans weight 2.\n'
            'Hyper-parameters: 4-fold patient-grouped inner CV per member.   Outer: 5-fold StratifiedGroupKFold by patient '
            '× 5 repeats (944 scans, 385 patients).\n'
            'Blend in CV: mean of the members’ within-fold ranks.   Frozen model: each member’s percentile among its '
            'scores on the 944 cohort scans, then the mean.\n'
            'Explanations: exact TreeSHAP (XGBoost, LightGBM) and linear SHAP (LR) per member, divided by the SD of the '
            'member margin, then averaged.',
            fontsize=5.6, va='top', linespacing=1.5, color=INK)
    return S.save(fig, 'diagram_dataflow')


def _funnel(ax, rows, title, unit_note):
    ax.set(xlim=(0, 1), ylim=(-0.45, len(rows) - 0.4))
    ax.axis('off')
    ax.invert_yaxis()
    ax.text(0, -0.35, title, fontsize=8, fontweight='bold', va='bottom')
    top = rows[0][1]
    for i, (lab, n, eyes, pts, pos, why) in enumerate(rows):
        w = max(0.07, 0.42 * (n / top) ** 0.5)
        last = i == len(rows) - 1
        ax.add_patch(FancyBboxPatch((0.25 - w / 2, i - 0.32), w, 0.64, boxstyle='round,pad=0,rounding_size=0.02',
                                    fc=S.C['noH'] if last else '#CBD3EC', ec='none'))
        narrow = w < 0.13
        ax.text(0.25 + w / 2 + 0.012 if narrow else 0.25, i, f'{n:,}', ha='left' if narrow else 'center',
                va='center', fontsize=7, fontweight='bold', color='white' if last and not narrow else INK)
        det = '  ·  '.join(x for x in (f'{eyes:,} eyes' if eyes else '', f'{pts:,} patients' if pts else '',
                                         f'{pos:,} CXL-labelled' if pos is not None else '') if x)
        ax.text(0.53, i - 0.08, lab, fontsize=6.8, fontweight='bold', va='bottom')
        ax.text(0.53, i + 0.02, det, fontsize=5.8, va='top', color=MUTE)
        if why is not None:
            lost = rows[i - 1][1] - n
            ax.annotate('', xy=(0.25, i - 0.33), xytext=(0.25, i - 0.67),
                        arrowprops=dict(arrowstyle='-|>', color=MUTE, lw=0.7, mutation_scale=6))
            ax.text(0.53, i - 0.5, textwrap.fill(f'− {lost:,}  {why}', 58), fontsize=5.6, va='center', linespacing=1.2,
                    color=S.C['H'] if lost else MUTE, style='italic')
    ax.text(0, len(rows) - 0.35, unit_note, fontsize=5.6, color=MUTE, va='top')


def fig_filtering():
    """Cohort (evaluated scans) and extra training exams: rows lost at each rule and why."""
    fig, axes = plt.subplots(1, 2, figsize=(S.W2, 3.5))
    _funnel(axes[0], COHORT_FUNNEL, 'a   Study cohort (evaluated)',
            f'Unit: scans. First visit: {FIRST_VISIT[0]} eyes, {FIRST_VISIT[1]} referred. Label: eye ever cross-linked.')
    _funnel(axes[1], EXTRA_FUNNEL, 'b   Extra training exams (training folds only)',
            'Unit: exams. Used only when the patient is outside the test fold; never scored.')
    fig.tight_layout(w_pad=1.5)
    return S.save(fig, 'diagram_filtering')


def funnel_tables():
    """The two funnels as tables (step, counts, removed, reason); same constants as the figure."""
    import pandas as pd
    out = {}
    for key, rows in (('cohort', COHORT_FUNNEL), ('extra', EXTRA_FUNNEL)):
        t = []
        for i, (lab, n, eyes, pts, pos, why) in enumerate(rows):
            t.append({'step': lab, 'scans' if key == 'cohort' else 'exams': n, 'eyes': eyes, 'patients': pts,
                      'CXL-labelled': pos, 'removed': None if i == 0 else rows[i - 1][1] - n,
                      'reason for removal': why})
        d = pd.DataFrame(t)
        for c in d.columns[1:6]:
            d[c] = d[c].astype('Int64')
        out[key] = d
        out[key].to_csv(S.OUT / 'tables' / f'filtering_{key}.csv', index=False)
    return out


if __name__ == '__main__':
    print(fig_dataflow(), fig_filtering())
