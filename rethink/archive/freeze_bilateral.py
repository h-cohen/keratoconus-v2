"""E22 freeze for external validation (design 6.6). Writes rethink/frozen/bilateral.pkl and the encoder weights
rethink/frozen/enc_<e>.pt (gitignored: fitted on patient data).

Selection is run ONCE on all 944 scans, exactly as one outer fold of E22n: 4-fold patient-grouped inner CV
(seed 2026) scores each candidate of a pool; the best is refitted on all 944 scans (+ augmentation rows if the
candidate uses them) with exp22_bilateral.fit_model (same code path as in CV).
  headline     pool = the 21 E22 candidates                       (needs whatever blocks the winner uses)
  fallback_a   pool = the 21 candidates without block H (E22n_noH)  (no fellow treatment dates)
  fallback_b   pool = noH_bn_{lr,lgbm}[_aug] (bilateral numerics N only, no H, no images)
  ref          E22_ref_single_num: {num_lr, num_lgbm} on the scanned eye's 55 numerics (primary comparator)
  age          age-only LR (sub-analysis comparator)
Also stored: the numeric column list, OCR glyph templates (block P), NORM constants; the external scorer
(score_external.py) rebuilds every block from raw inputs and never needs the cohort.
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import exp16_nested as e16  # noqa: E402
import exp22_bilateral as e22  # noqa: E402
from harness import DATA, RES, ROOT  # noqa: E402

SEED = 2026
FROZEN = ROOT / 'frozen'


def spec(name):
    """Candidate name -> (blocks, aug, kind, enc, use_h) or ('ens', enc, use_h)."""
    use_h = not name.startswith('noH_')
    n = name.replace('noH_', '')
    if n.startswith('ens_'):
        return ('ens', n.split('_')[1], use_h)
    parts = n.split('_')
    kind, aug = parts[-1], 'aug' in parts
    if parts[0] == 'bn':
        return ('N', aug, kind, None, use_h)
    if parts[0] == 'bnpm':
        return ('PM', aug, kind, None, use_h)
    return ('I', aug, kind, parts[1], use_h)


def select_all(coh, pool):
    y, g = coh.y.values, coh.pid.values
    scores = {k: [] for k in pool}
    for a, b in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(coh, y, g):
        da, db = coh.iloc[a].reset_index(drop=True), coh.iloc[b].reset_index(drop=True)
        for k in pool:
            scores[k].append(roc_auc_score(y[b], e16.predict(k, da, db, SEED)))
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    return max(means, key=means.get), means


def freeze(D, coh, name):
    s = spec(name)
    pre = 'noH_' if name.startswith('noH_') else ''
    if s[0] == 'ens':
        return dict(name=name, ens=[freeze(D, coh, f'{pre}bn_aug_lgbm'), freeze(D, coh, f'{pre}bi_{s[1]}_lr')])
    blocks, aug, kind, enc, use_h = s
    rows = e22.train_rows(D, coh, aug)
    fz = e22.fit_model(D, rows, blocks, kind, enc, use_h, SEED)
    fz['name'] = name
    return fz


def encoders_used(fz):
    if 'ens' in fz:
        return set().union(*[encoders_used(f) for f in fz['ens']])
    return {fz['enc']} if fz['enc'] else set()


def main():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    D = e22.build(coh)
    full = e22.candidates(D)
    noh = e22.candidates(D, prefix='noH_', use_h=False)
    e16.CANDS.update(full), e16.CANDS.update(noh)
    out = {'numerics': e16.NUM, 'seed': SEED, 'n_train': len(coh), 'norm': e22_norm()}
    log = []
    for slot, pool in [('headline', list(full)), ('fallback_a', list(noh)),
                       ('fallback_b', [k for k in noh if k.startswith('noH_bn_')])]:
        best, means = select_all(coh, pool)
        log.append(dict(slot=slot, chosen=best, inner_auc=round(means[best], 4)))
        print(slot, best, round(means[best], 3), flush=True)
        out[slot] = freeze(D, coh, best)
    best, means = select_all(coh, ['num_lr', 'num_lgbm'])
    log.append(dict(slot='ref', chosen=best, inner_auc=round(means[best], 4)))
    out['ref'] = ref_fit(coh, best)
    out['age'] = LogisticRegression(max_iter=5000).fit(coh[['age']].values, coh.y.values)
    used = set().union(*[encoders_used(out[k]) for k in ('headline', 'fallback_a', 'fallback_b')])
    FROZEN.mkdir(exist_ok=True)
    for e in used - {'Z', 'D'}:
        ck = torch.load(RES / 'ssl' / e / 'ckpt.pt', map_location='cpu', weights_only=False)
        enc = {k[4:]: v for k, v in ck['net'].items() if k.startswith('enc.')}
        torch.save(dict(enc=enc, cfg=ck['cfg'], epoch=ck['epoch']), FROZEN / f'enc_{e}.pt')
    if any(uses_pm(out[k]) for k in ('headline', 'fallback_a', 'fallback_b')):
        import panel_all
        out['ocr_templates'] = panel_all.templates()
    out['encoders'] = sorted(used)
    with open(FROZEN / 'bilateral.pkl', 'wb') as f:
        pickle.dump(out, f)
    pd.DataFrame(log).to_csv(RES / 'freeze22_choices.csv', index=False)
    print(pd.DataFrame(log).to_string(index=False), '| encoders needed:', sorted(used))


def e22_norm():
    from pretrain_jepa import NORM
    return NORM


def uses_pm(fz):
    if 'ens' in fz:
        return any(uses_pm(f) for f in fz['ens'])
    return fz['blocks'] == 'PM'


def ref_fit(coh, name):
    """The single-eye numeric comparator refitted on all 944 scans (same code as exp16_nested)."""
    from exp01_tabular import _inner_cv
    X, y, g = coh[e16.NUM].values, coh.y.values, coh.pid.values
    if name == 'num_lr':
        hp = _inner_cv(X, y, g, e16.LR, e16.CGRID, SEED)
        return dict(name=name, model=e16.LR(**hp).fit(X, y))
    return dict(name=name, model=e16._lgbm(y, SEED).fit(X, y))


if __name__ == '__main__':
    sys.exit(main())
