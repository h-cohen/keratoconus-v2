"""Step 5: decode every '4 Maps Selectable' composite in data/image_scans (cohort or not) to physical maps, 58 x 58.

Same decoder as decode_maps.py (per-image 31-step palettes). Left eyes are mirrored so x is nasal-positive.
Output (kcref/data/, gitignored): decoded_all.npy (N, 4, 58, 58) float32 and decoded_all_meta.parquet with
pkey (patient key: cohort pid when the name matches a cohort patient, else a salted hash of the name),
eye, date, cohort_row (row in cohort.parquet, -1 if not a cohort scan). No labels are read.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from decode_maps import IMG, MAPS, blocks, decode  # noqa: E402
from prepare_data import pid_of  # noqa: E402

DATA = Path(__file__).parent / 'data'


def name_key(stem):
    return '_'.join(stem.lower().split('_')[:2])


def main():
    keys = pd.read_csv(DATA / 'private_keys.csv', dtype=str)
    stems = keys.filename.str.lower().str.rsplit('.', n=1).str[0]
    cohort_row = {s: i for i, s in enumerate(stems)}
    name_pid = dict(zip(stems.map(name_key), keys.pid))
    files = sorted(f for f in IMG.iterdir() if f.suffix.lower() == '.jpg' and 'maps' in f.name.lower())
    out, meta = [], []
    for i, f in enumerate(files):
        s = f.name.lower().rsplit('.', 1)[0]
        m = re.search(r'_(od|os)_(\d{8})_', s)
        if not m:
            continue
        im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
        v = np.full((4, 58, 58), np.nan, np.float32)
        for j, mp in enumerate(MAPS):
            d, _ = decode(im, mp)
            if d is not None:
                v[j] = blocks(d)
        if m.group(1) == 'os':
            v = v[:, :, ::-1]
        out.append(v)
        nk = name_key(s)
        meta.append(dict(pkey=name_pid.get(nk, pid_of('name:' + nk)), eye={'od': 'right', 'os': 'left'}[m.group(1)],
                         date=pd.to_datetime(m.group(2), format='%d%m%Y'), cohort_row=cohort_row.get(s, -1),
                         ok=bool(np.isfinite(v).any(axis=(1, 2)).all())))
        if i % 200 == 0:
            print(i, flush=True)
    np.save(DATA / 'decoded_all.npy', np.stack(out))
    meta = pd.DataFrame(meta)
    meta.to_parquet(DATA / 'decoded_all_meta.parquet', index=False)
    print(len(meta), 'composites;', (meta.cohort_row >= 0).sum(), 'cohort;', meta.pkey.nunique(), 'patients;',
          meta.ok.mean().round(3), 'with all four maps decoded')


if __name__ == '__main__':
    sys.exit(main())
