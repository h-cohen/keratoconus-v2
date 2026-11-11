"""E24 freeze for external validation: the E24n stacks refitted on all 944 scans (user decision 2026-11-11:
keep the highest first-visit AUC model, E24n_H). Writes rethink/frozen/bilateral.pkl (the E22 freeze is kept as
bilateral_e22.pkl) and rethink/frozen/ftL.pt (gitignored: fitted on patient data).

Exactly one outer fold of exp23_firstvisit.py run on all 944 scans (seed 2026):
  1. 4-fold patient-grouped inner CV: OOF predictions of every pool member (21 E24 candidates + ftL);
  2. non-negative logistic stacker on their standardised logits (first-visit weight 2, L2 1e-2);
  3. every member with a non-zero weight refitted on all 944 scans (+ extra exams, first-visit weight 2), ftL
     fine-tuned on all 944 scans (ft_L.py, same recipe).
  headline    E24n_H stack   (needs fellow numerics, --tx, composites; crops if a DINOv2 member is used)
  fallback_a  E24n_noH stack (no --tx)
  fallback_b, ref, age: unchanged from the E22 freeze.
Usage: python -I rethink/freeze_stack.py
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import exp16_nested as e16  # noqa: E402
import exp22_bilateral as e22  # noqa: E402
import exp23_firstvisit as e23  # noqa: E402
import ft_L  # noqa: E402
from freeze_bilateral import FROZEN, SEED, encoders_used, spec, uses_pm  # noqa: E402
from harness import DATA, RES  # noqa: E402


def run_ftl(coh, jobs):
    todo = [(s, a, b) for s, a, b in jobs if not (ft_L.CACHE / f'{ft_L.key(s, a, b)}.npy').exists()]
    gpus = ft_L.GPUS[:max(1, len(todo))]
    if todo:
        from multiprocessing import get_context
        with get_context('spawn').Pool(len(gpus)) as p:
            p.map(ft_L.worker, [(g, todo[i::len(gpus)]) for i, g in enumerate(gpus)])


def freeze_member(D, coh, name, pre):
    if name == 'ftL':
        return dict(name='ftL', ftl=True)
    s = spec(name[len(pre):] if name.startswith(pre) else name)
    if s[0] == 'ens':
        return dict(name=name, ens=[freeze_member(D, coh, f'{pre}bn_aug_lgbm', pre),
                                    freeze_member(D, coh, f'{pre}bi_{s[1]}_lr', pre)])
    blocks, aug, kind, enc, _ = s
    use_h = pre == 'a_'
    rows = e22.train_rows(D, coh, True)  # E24: every candidate trains with the extra exams
    w = np.where(D['first'][rows], e23.W_FIRST, 1.0)
    fz = e22.fit_model(D, rows, blocks, kind, enc, use_h, SEED, w)
    fz['name'] = name
    return fz


def freeze_stack(D, coh, pre):
    use_h = pre == 'a_'
    c = e22.candidates(D, prefix=pre, use_h=use_h, w_first=e23.W_FIRST, aug_all=True)
    c['ftL'] = e23.ftL
    e16.CANDS.update(c)
    pool = list(c)
    y, g = coh.y.values, coh.pid.values
    P = np.zeros((len(coh), len(pool)))
    for a, b in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(coh, y, g):
        da, db = coh.iloc[a].reset_index(drop=True), coh.iloc[b].reset_index(drop=True)
        for j, k in enumerate(pool):
            P[b, j] = e16.predict(k, da, db, SEED)
    Z = e23.logit(P)
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    t = e23.fit_stacker(Z, y, np.where(coh.first_visit.values, e23.W_FIRST, 1.0))
    keep = [j for j in range(len(pool)) if t[j] > 0]
    members = [freeze_member(D, coh, pool[j], pre) for j in keep]
    print(pre, 'stack members:', {pool[j]: round(t[j], 3) for j in keep}, flush=True)
    return dict(name=f'E24n_{"H" if use_h else "noH"}', stack=members, mu=mu[keep], sd=sd[keep], t=t[keep],
                b=float(t[-1]), use_h=use_h)


def main():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    D = e22.build(coh)
    assert 'age' in D['V'].columns
    y, g = coh.y.values, coh.pid.values
    jobs = [(SEED, coh.crow.values[a], coh.crow.values[b])
            for a, b in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(coh, y, g)]
    jobs.append((SEED, coh.crow.values, np.array([], dtype=coh.crow.dtype)))
    run_ftl(coh, jobs)
    final = ft_L.CACHE / f'{ft_L.key(*jobs[-1])}.pt'
    FROZEN.mkdir(exist_ok=True)
    old = FROZEN / 'bilateral.pkl'
    if not (FROZEN / 'bilateral_e22.pkl').exists():
        shutil.copy(old, FROZEN / 'bilateral_e22.pkl')
    with open(FROZEN / 'bilateral_e22.pkl', 'rb') as f:
        out = pickle.load(f)
    out['headline'] = freeze_stack(D, coh, 'a_')
    out['fallback_a'] = freeze_stack(D, coh, 'anoH_')
    shutil.copy(final, FROZEN / 'ftL.pt')
    used = set().union(*[encoders_used_s(out[k]) for k in ('headline', 'fallback_a', 'fallback_b')])
    for e in used - {'Z', 'D'} - set(out['encoders']):
        ck = torch.load(RES / 'ssl' / e / 'ckpt.pt', map_location='cpu', weights_only=False)
        enc = {k[4:]: v for k, v in ck['net'].items() if k.startswith('enc.')}
        torch.save(dict(enc=enc, cfg=ck['cfg'], epoch=ck['epoch']), FROZEN / f'enc_{e}.pt')
    if 'ocr_templates' not in out and any(uses_pm_s(out[k]) for k in ('headline', 'fallback_a', 'fallback_b')):
        import panel_all
        out['ocr_templates'] = panel_all.templates()
    out['encoders'] = sorted(used)
    with open(FROZEN / 'bilateral.pkl', 'wb') as f:
        pickle.dump(out, f)
    rows = [dict(slot=s, model=out[s]['name'], member=m['name'], weight=round(float(w), 4))
            for s in ('headline', 'fallback_a') for m, w in zip(out[s]['stack'], out[s]['t'])]
    pd.DataFrame(rows).to_csv(RES / 'freeze24_choices.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), '| encoders needed:', sorted(used))


def encoders_used_s(fz):
    if 'stack' in fz:
        return set().union(*[encoders_used_s(m) for m in fz['stack']])
    if fz.get('ftl'):
        return set()
    return encoders_used(fz)


def uses_pm_s(fz):
    if 'stack' in fz:
        return any(uses_pm_s(m) for m in fz['stack'])
    return False if fz.get('ftl') else uses_pm(fz)


if __name__ == '__main__':
    sys.exit(main())
