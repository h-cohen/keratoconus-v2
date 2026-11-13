"""Method diagrams for the paper (E29): the data flow with tensor sizes, and the cohort / training-row filtering funnels.

Funnel counts come from the data-preparation logs: kcref/data/cohort_report.txt (prepare_data.py) and the printout of
prepare_extra.py, plus the raw counts of data/multimodal_dataset.csv. They are constants here because recomputing them
needs the identifiable source exports; test_figures.test_funnel_matches_cohort checks the final row against the cohort.
"""
import sys
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
     'patients outside the clinic sheet were never followed for CXL: label not comparable'),
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


def fig_dataflow():
    """One visit -> blocks (with sizes) -> design matrix -> in-fold preprocessing -> three members -> rank blend."""
    fig = plt.figure(figsize=(S.W2, 4.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    # column 1: inputs of one visit
    ax.text(0.10, 0.975, 'One visit', ha='center', fontsize=7.5, fontweight='bold', color=MUTE)
    box(ax, 0.01, 0.74, 0.18, 0.19, 'Scanned eye', '54 Pentacam numerics\n+ age\n4 maps, 4 × 145 × 145\n(T, K, EF, EB; 0.069 mm/cell)',
        'in')
    box(ax, 0.01, 0.50, 0.18, 0.19, 'Fellow eye (same day)', '54 numerics\n4 maps, 4 × 145 × 145\n(any same-day exam,\nchosen label-free)', 'in')
    box(ax, 0.01, 0.30, 0.18, 0.15, 'Clinic treatment sheet', 'fellow-eye CXL / keratoplasty\ndates before the visit\n(H arm only)', 'h',
        edge=S.SRC['H'])
    # column 2: map path
    box(ax, 0.235, 0.43, 0.17, 0.25, 'Map decoding → R',
        'colour → µm / D per cell\nleft eyes mirrored (+x nasal)\n96 measurements per map set:\n64 ring × quadrant means\n16 extremum value / position\n8 I−S, N−T asymmetries\n4 SD, 4 thickness rise', 'block')
    # column 3: blocks
    ax.text(0.545, 0.975, 'Feature blocks (columns)', ha='center', fontsize=7.5, fontweight='bold', color=MUTE)
    box(ax, 0.445, 0.74, 0.20, 0.19, 'N  ·  205',
        'own: age + 54 + 11 indices (66)\nfellow: 65  ·  own − fellow: 65\n8 worse-eye flags + has_fellow', 'block',
        edge=S.SRC['own_num'])
    box(ax, 0.445, 0.50, 0.20, 0.19, 'R  ·  288',
        'own 96  ·  fellow 96\nown − fellow 96\nstandardised in fold', 'block', edge=S.SRC['own_maps'])
    box(ax, 0.445, 0.385, 0.20, 0.085, 'Flags  ·  2', 'own map, fellow map present', 'block', tsize=6.6, bsize=5.6)
    box(ax, 0.445, 0.27, 0.20, 0.09, 'H  ·  5  (H arm)', 'fellow CXL before visit, years since,\nkeratoplasty, unknown, missing', 'h',
        edge=S.SRC['H'], tsize=6.6, bsize=5.6)
    # column 4: matrix + preprocessing
    box(ax, 0.69, 0.66, 0.13, 0.27, 'Design matrix X',
        'n × 495  (noH)\nn × 500  (H)\n\nn = training rows\nof the fold', 'block', edge=INK)
    box(ax, 0.69, 0.38, 0.13, 0.24, 'In-fold\npreprocessing',
        '\n\nclip 0.5 / 99.5 %\nmedian impute\nfitted on the\ntraining rows only', 'block')
    # column 5: members + blend
    ax.text(0.91, 0.975, 'Rank blend', ha='center', fontsize=7.5, fontweight='bold', color=MUTE)
    box(ax, 0.845, 0.80, 0.135, 0.13, 'XGBoost', 'depth 2–3, 200–500 trees', 'model', tsize=6.8, bsize=5.6)
    box(ax, 0.845, 0.645, 0.135, 0.13, 'LightGBM × 5 seeds', 'depth 2, 100–300 trees\nage monotone', 'model',
        tsize=6.8, bsize=5.6)
    box(ax, 0.845, 0.49, 0.135, 0.13, 'Elastic-net LR', 'C 0.01–0.3, L1 0.2–0.8\nstandardised', 'model', tsize=6.8,
        bsize=5.6)
    box(ax, 0.845, 0.30, 0.135, 0.15, 'Referral score',
        'mean of the three\npercentile ranks\n(0 to 1)', 'out', edge=S.C['noH'])
    # arrows
    arrow(ax, (0.19, 0.86), (0.445, 0.86))                       # own numerics -> N
    arrow(ax, (0.19, 0.66), (0.445, 0.79), rad=-0.1)             # fellow numerics -> N
    arrow(ax, (0.19, 0.77), (0.30, 0.68), rad=0.15)              # own maps -> decoding
    arrow(ax, (0.19, 0.55), (0.235, 0.58))                       # fellow maps -> decoding
    arrow(ax, (0.405, 0.595), (0.445, 0.595))                    # decoding -> R
    arrow(ax, (0.19, 0.375), (0.445, 0.315), color=S.SRC['H'], ls=(0, (3, 2)))
    for y0 in (0.835, 0.595, 0.43, 0.315):
        arrow(ax, (0.645, y0), (0.69, 0.80 if y0 > 0.7 else 0.70))
    arrow(ax, (0.755, 0.66), (0.755, 0.62))
    for y1 in (0.865, 0.71, 0.555):
        arrow(ax, (0.82, 0.50), (0.845, y1))
    for y in (0.865, 0.71, 0.555):                               # members -> bracket -> score
        ax.plot([0.98, 0.988], [y, y], color=MUTE, lw=0.8)
    ax.plot([0.988, 0.988], [0.865, 0.375], color=MUTE, lw=0.8)
    arrow(ax, (0.988, 0.375), (0.98, 0.375))
    # training panel
    ax.add_patch(FancyBboxPatch((0.01, 0.025), 0.98, 0.205, boxstyle='round,pad=0.004,rounding_size=0.01',
                                fc='#FAFAF8', ec=LINE, lw=0.6))
    ax.text(0.025, 0.205, 'Training and evaluation', fontsize=7, fontweight='bold', va='top')
    ax.text(0.025, 0.165,
            'Training rows per outer fold: the fold’s cohort scans + 1,487 extra labelled exams (no maps; R imputed, '
            'own_image = 0); first-visit scans weight 2.\n'
            'Hyper-parameters: 4-fold patient-grouped inner CV per member.   Outer: 5-fold StratifiedGroupKFold by patient '
            '× 5 repeats (944 scans, 385 patients).\n'
            'Blend in CV: mean of the members’ within-fold ranks.   Frozen model: each member’s percentile among its '
            'scores on the 944 cohort scans, then the mean.\n'
            'Explanations: exact TreeSHAP (XGBoost, LightGBM) and linear SHAP (LR) per member, divided by the SD of the '
            'member margin, then averaged.',
            fontsize=5.9, va='top', linespacing=1.55, color=INK)
    return S.save(fig, 'diagram_dataflow')


def _funnel(ax, rows, title, unit_note):
    ax.set(xlim=(0, 1), ylim=(-0.2, len(rows) - 0.4))
    ax.axis('off')
    ax.invert_yaxis()
    ax.text(0, -0.35, title, fontsize=8, fontweight='bold', va='bottom')
    top = rows[0][1]
    for i, (lab, n, eyes, pts, pos, why) in enumerate(rows):
        w = 0.5 * n / top
        last = i == len(rows) - 1
        ax.add_patch(FancyBboxPatch((0.25 - w / 2, i - 0.32), w, 0.64, boxstyle='round,pad=0,rounding_size=0.02',
                                    fc=S.C['noH'] if last else '#C9D7EA', ec='none'))
        narrow = w < 0.1
        ax.text(0.25 - w / 2 - 0.015 if narrow else 0.25, i, f'{n:,}', ha='right' if narrow else 'center',
                va='center', fontsize=7, fontweight='bold', color='white' if last and not narrow else INK)
        det = '  ·  '.join(x for x in (f'{eyes:,} eyes' if eyes else '', f'{pts:,} patients' if pts else '',
                                         f'{pos:,} CXL-labelled' if pos is not None else '') if x)
        ax.text(0.53, i - 0.08, lab, fontsize=6.8, fontweight='bold', va='bottom')
        ax.text(0.53, i + 0.02, det, fontsize=5.8, va='top', color=MUTE)
        if why is not None:
            lost = rows[i - 1][1] - n
            ax.annotate('', xy=(0.25, i - 0.33), xytext=(0.25, i - 0.67),
                        arrowprops=dict(arrowstyle='-|>', color=MUTE, lw=0.7, mutation_scale=6))
            ax.text(0.53, i - 0.5, f'− {lost:,}  {why}' if lost else f'− 0  {why}', fontsize=5.6, va='center',
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
