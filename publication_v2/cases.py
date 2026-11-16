"""S1 matched-pair cases (E29): individual eyes' decoded maps. Output goes ONLY to publication_v2/local/ (gitignored;
user decision 2026-11-13: individual maps stay out of the repository until submission).

Pairs: first-visit eyes with both maps; each referred eye paired with the nearest not-referred eye on z-scored
(Km F, age), same stage A, different patient (explain.match_pairs, pre-declared in PROTOCOL.md E29). Pairs are ranked by
(final noH score gap) - (single-eye L1 score gap); the top 3 form the main case figure, the next 5 the supplement.
Labels are 'Pair k a/b' with Km F and age rounded; no ids, dates or filenames.
Usage: python -I publication_v2/cases.py
"""
import base64
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plots as PL  # noqa: E402
import style as S  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kcref'))
import explain as E  # noqa: E402
import features as F  # noqa: E402
import model as M  # noqa: E402

LOCAL = S.OUT / 'local'
SHOW = (('own', 0, 'Thickness'), ('own', 3, 'Back elev.'), ('fel', 0, 'Fellow thickness'), ('fel', 3, 'Fellow back elev.'))
MAP_KEY = {0: 'T', 3: 'EB'}
RANGE = {0: (420, 680), 3: (-60, 60)}


def ranked_pairs():
    """Matched pairs (with scores) ranked by final-minus-L1 gap; the noH and H SHAP dicts and the cohort info."""
    info = PL.cohort_info()
    shs = {a: PL.load_shap(a) for a in ('noH', 'H')}
    pr, sub, s_fin, s_l1 = PL.pairs(shs['noH'], info, 'noH')
    p, q = sub[pr.pos_idx.values], sub[pr.neg_idx.values]
    pr = pr.assign(gain=(s_fin[p] - s_fin[q]) - (s_l1[p] - s_l1[q]))
    return pr.sort_values('gain', ascending=False).reset_index(drop=True), sub, shs, info, s_l1


def draw(pairs, sub, shs, info, s_l1, name):
    """Matched pairs: rows 'Pair k a' (referred) and 'Pair k b' (not referred)."""
    rows = []
    for p, pair in pairs.iterrows():
        for e, (key, lab) in enumerate(((pair.pos_idx, 'referred'), (pair.neg_idx, 'not referred'))):
            rows.append((sub[int(key)], f'Pair {p + 1}{"ab"[e]}', lab, e == 0))
    return draw_rows(rows, shs, info, s_l1, name)


def gallery_rows(shs, info):
    """12 first-visit eyes with both maps, by the final noH OOF score: the 4 highest-scored referred (true positives),
    the 4 lowest-scored not referred (true negatives), the 2 highest-scored not referred (false-positive type) and the 2
    lowest-scored referred (false-negative type)."""
    D = F.build(M.cohort())
    sh = shs['noH']
    n = len(info['y'])  # cohort rows come first in the visit table
    ok = info['first'] & (D['own'][:n] >= 0) & (D['felc'][:n] >= 0)
    s = sh['oof']
    pos, neg = np.flatnonzero(ok & (info['y'] == 1)), np.flatnonzero(ok & (info['y'] == 0))
    pos, neg = pos[np.argsort(-s[pos])], neg[np.argsort(-s[neg])]
    pick = [(i, 'correct referral', True) for i in pos[:4]] + [(i, 'correct non-referral', False) for i in neg[-4:][::-1]]
    pick += [(i, 'not referred, high score', False) for i in neg[:2]] + [(i, 'referred, low score', True) for i in pos[-2:]]
    return [(i, f'Eye {k + 1}', lab, red) for k, (i, lab, red) in enumerate(pick)]


def draw_rows(rows, shs, info, s_l1, name):
    """One row per eye: (cohort row, title, sub-label, referred?)."""
    hr = np.load(PL.DATA / 'decoded_hr.npy', mmap_mode='r')
    D = F.build(M.cohort())
    sh = shs['noH']
    fam = np.array([E.family(c) for c in sh['cols']])
    fams = list(PL.FAMILY_LABEL)
    n = len(rows)
    lim = 1.05 * max(abs(sh['shap'][i][fam == f].sum()) for i, *_ in rows for f in fams)
    fig = plt.figure(figsize=(S.W2, 1.025 * n))
    gs = fig.add_gridspec(n, 6, width_ratios=[0.95, 1, 1, 1, 1, 1.6], hspace=0.45, wspace=0.08)
    for r, (i, title, lab, referred) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0])
        ax.axis('off')
        ax.text(0, 0.62, title, fontsize=7.5, fontweight='bold', transform=ax.transAxes)
        ax.text(0, 0.42, lab, fontsize=6.5, color=S.C['H'] if referred else S.C['mute'], transform=ax.transAxes)
        ax.text(0, 0.18, f'Km F {info["km"][i]:.0f} D · age {info["age"][i]:.0f}', fontsize=5.8,
                transform=ax.transAxes)
        ax.text(0, -0.02, f'score L1 {s_l1[i]:.2f} → final {sh["oof"][i]:.2f}', fontsize=5.8,
                transform=ax.transAxes)
        for k, (side, j, mtitle) in enumerate(SHOW):
            ax = fig.add_subplot(gs[r, k + 1])
            c = (D['own'] if side == 'own' else D['felc'])[i]
            ax.axis('off')
            if c < 0:
                continue
            img = np.where(PL.radiomics.R <= 4, np.asarray(hr[c, j], float), np.nan)
            ax.imshow(img, cmap=S.MAP_CMAP[MAP_KEY[j]], vmin=RANGE[j][0], vmax=RANGE[j][1], extent=PL.MAP_EXTENT,
                      origin='upper')
            cols = [q for q, cc in enumerate(sh['cols']) if cc.startswith(f'R_{side}') and
                    E.sector(E.r_name(cc)) and E.sector(E.r_name(cc))[0] == MAP_KEY[j]]
            for q in sorted(cols, key=lambda q: -abs(sh['shap'][i, q]))[:3]:
                _, r0, r1, quad = E.sector(E.r_name(sh['cols'][q]))
                t0, t1 = PL.QUAD_DEG[quad]
                ax.add_patch(Wedge((0, 0), r1, t0, t1, width=r1 - r0, fc='none', lw=1.0,
                                   ec='black' if sh['shap'][i, q] > 0 else 'white'))
            ax.set(xlim=(-4.2, 4.2), ylim=(-4.2, 4.2))
            if r == 0:
                ax.set_title(mtitle, loc='center', fontsize=6.8)
        ax = fig.add_subplot(gs[r, 5])
        v = [sh['shap'][i, fam == f].sum() for f in fams]
        ax.barh(range(len(fams)), v, color=[S.C['H'] if x > 0 else S.C['noH'] for x in v], height=0.7)
        ax.axvline(0, color=S.C['mute'], lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.grid(axis='x')
        ax.set_yticks(range(len(fams)), [PL.FAMILY_LABEL[f] for f in fams] if r == 0 else [],
                      fontsize=5.6)
        ax.yaxis.tick_right()
        ax.tick_params(labelsize=5.2, length=1.5)
        ax.invert_yaxis()
        if r == 0:
            ax.set_title('Relative contribution\n(member-standardised SHAP)', loc='center', fontsize=6.3)
    fig.text(0.5, 0.0, 'Maps from decoded composites (left eyes mirrored, +x nasal); outlined: 3 sectors with the '
             'largest |SHAP| for that eye (black raises, white lowers the score). Thickness 420–680 µm, '
             'back elevation ±60 µm.', ha='center', fontsize=6, color=S.C['mute'])
    return S.save(fig, name, local=True)


def main():
    pr, sub, shs, info, s_l1 = ranked_pairs()
    paths = [draw(pr.iloc[:3], sub, shs, info, s_l1, 'cases_main'),
             draw(pr.iloc[3:8].reset_index(drop=True), sub, shs, info, s_l1, 'cases_supplement'),
             draw_rows(gallery_rows(shs, info), shs, info, s_l1, 'cases_gallery')]
    html = ''.join(f'<h2>{p.stem}</h2><img style="max-width:100%" src="data:image/png;base64,'
                   f'{base64.b64encode(p.read_bytes()).decode()}">' for p in paths)
    (LOCAL / 'report_cases.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Matched-pair cases (local only)</title>'
        '<body style="font-family:sans-serif;max-width:1100px;margin:auto">'
        '<p><b>Local only, not for the repository.</b> Individual eyes; see report section S1 for the statistics.</p>'
        + html, encoding='utf-8')
    print('wrote', *paths, LOCAL / 'report_cases.html')


if __name__ == '__main__':
    sys.exit(main())
