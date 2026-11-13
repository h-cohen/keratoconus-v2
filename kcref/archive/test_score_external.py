"""Test score_external.py the way the user will run it: from a copy of the repo WITHOUT data/ and rethink/data/.

1. Builds an external-format input from a random subset of cohort patients (pseudonymous pid only): scored rows
   (cohort scans), fellow-only rows (same-day raw exams of the other eye, y empty), a per-eye treatment table,
   and copies of their composites under neutral names (c00000.jpg ...). Written to OUTDIR (scratch), never to git.
2. Copies only the code, numeric_features.txt and frozen/ into OUTDIR/repo and runs the scorer there, three times:
   all inputs; without --tx; without --tx and composites.
3. Prints the internal apparent AUCs of the same frozen models on the same scored rows (exp22 path), which the
   scorer's output must reproduce (aggregates only on both sides).
Usage: python -I rethink/test_score_external.py OUTDIR [N_PATIENTS]
"""
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import exp22_bilateral as e22  # noqa: E402
from decode_hr import files  # noqa: E402
from harness import DATA, ROOT  # noqa: E402

REPO = ROOT.parent


def main():
    out = Path(sys.argv[1]).resolve()
    n_pat = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    out.mkdir(parents=True, exist_ok=True)
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    pids = np.random.default_rng(7).choice(coh.pid.unique(), n_pat, replace=False)
    D = e22.build(coh)
    D['hr'] = np.load(DATA / 'decoded_hr.npy')  # for the ftL stack member
    V = D['V']
    num = e22.e16.NUM
    sc = coh[coh.pid.isin(pids)]
    meta = pd.read_parquet(DATA / 'decoded_hr_meta.parquet')
    fs = files()
    comp = (out / 'composites')
    comp.mkdir(exist_ok=True)
    names = {}
    crops = out / 'crops'
    crops.mkdir(exist_ok=True)
    from dino_all import CROP, MAPS
    on_disk = {p.name.lower(): p for p in CROP.iterdir()}

    def cname(r):
        if r < 0:
            return ''
        if r not in names:
            names[r] = f'c{len(names):05d}.jpg'
            shutil.copy(fs[r], comp / names[r])
            stem = fs[r].name.rsplit('.', 1)[0]
            for m in MAPS:  # cropped maps for encoder D, neutral names
                src = on_disk.get(f'{stem}_{m}.jpg'.lower())
                if src is not None:
                    shutil.copy(src, crops / f'{names[r][:-4]}_{m}.jpg')
        return names[r]

    rows = []
    for crow in sc.crow.values:
        rows.append(dict(pid=V.pid.iat[crow], eye=V.eye.iat[crow], scan_date=V.date.iat[crow], y=V.y.iat[crow],
                         composite=cname(D['own'][crow]), image_stem=names[D['own'][crow]][:-4] if D['own'][crow] >= 0 else '', **{c: V[c].iat[crow] for c in num}))
    # fellow-only rows: the fellow chosen in training, unless it is itself a scored cohort scan
    fel_rows = set(int(f) for f in V.fel.values[sc.crow.values] if f >= 0) - set(sc.crow.values)
    for r in sorted(fel_rows):
        fc = D['felc'][[c for c in sc.crow.values if V.fel.iat[c] == r][0]]
        rows.append(dict(pid=V.pid.iat[r], eye=V.eye.iat[r], scan_date=V.date.iat[r], y=np.nan,
                         composite=cname(fc), image_stem=names[fc][:-4] if fc >= 0 else '', **{c: V[c].iat[r] for c in num}))
    ext = pd.DataFrame(rows)
    ext.to_csv(out / 'external.csv', index=False)
    tx = pd.read_parquet(DATA / 'eye_tx.parquet')
    tx[tx.pid.isin(pids)].to_csv(out / 'tx.csv', index=False)
    print(f'external test set: {len(sc)} scored scans, {len(fel_rows)} fellow-only rows, {len(names)} composites')

    # internal reference: same frozen models, internal feature path, same scored rows
    with open(ROOT / 'frozen' / 'bilateral.pkl', 'rb') as f:
        model = pickle.load(f)
    idx, y = sc.crow.values, sc.y.values
    from score_external import predict
    for slot in ('headline', 'fallback_a', 'fallback_b'):
        print(f'internal {slot:10s} apparent AUC {roc_auc_score(y, predict(model[slot], D, idx)):.3f}')
    print(f'internal ref        apparent AUC {roc_auc_score(y, model["ref"]["model"].predict_proba(sc[num].values)[:, 1]):.3f}')

    repo = out / 'repo'
    if repo.exists():
        shutil.rmtree(repo)
    (repo / 'rethink').mkdir(parents=True)
    for p in (REPO / 'rethink').glob('*.py'):
        shutil.copy(p, repo / 'rethink' / p.name)
    shutil.copy(REPO / 'rethink' / 'numeric_features.txt', repo / 'rethink')
    (repo / 'rethink' / 'frozen').mkdir()
    for p in (REPO / 'rethink' / 'frozen').iterdir():
        if p.name != 'models.pkl':  # legacy models are tested separately (freeze_models.py)
            shutil.copy(p, repo / 'rethink' / 'frozen' / p.name)
    assert not (repo / 'data').exists() and not (repo / 'rethink' / 'data').exists()
    py = sys.executable
    for label, extra in [('all inputs', ['--tx', str(out / 'tx.csv'), '--composites', str(comp), '--images', str(crops)]),
                         ('no --tx', ['--composites', str(comp), '--images', str(crops)]), ('numerics only', [])]:
        print(f'\n=== scorer from the data-free copy: {label} ===', flush=True)
        r = subprocess.run([py, '-I', 'rethink/score_external.py', str(out / 'external.csv')] + extra, cwd=repo,
                           capture_output=True, text=True)
        print(r.stdout[-4000:], r.stderr[-2000:] if r.returncode else '', flush=True)


if __name__ == '__main__':
    sys.exit(main())
