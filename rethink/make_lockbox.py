"""Set aside a patient-level lockbox ONCE. Run only at protocol freeze; never re-drawn.

25% of patients, stratified by whether the patient has any CXL-labelled eye.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / 'data'
out = DATA / 'lockbox_pids.csv'
assert not out.exists(), 'lockbox already drawn - do not redraw'
df = pd.read_parquet(DATA / 'cohort.parquet')
pat = df.groupby('pid').y.max()
rng = np.random.default_rng(20261106)
lock = []
for lab in (0, 1):
    p = pat[pat == lab].index.values
    lock += list(rng.choice(p, int(round(0.25 * len(p))), replace=False))
pd.DataFrame({'pid': sorted(lock)}).to_csv(out, index=False)
d = df[df.pid.isin(lock)]
print(f'lockbox: {len(lock)} patients, {d.eid.nunique()} eyes, {len(d)} scans, {d.y.sum()} positive scans, '
      f'{d.groupby("eid").y.first().sum()} positive eyes')
