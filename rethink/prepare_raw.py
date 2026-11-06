"""Align the raw Pentacam export (Raw_Combined_IdsMesh.xlsx) to the frozen cohort, row by row.

Match key: patient id (idp), eye and exam time (HHMMSS). The exam date in this export has day/month
swaps (CLAUDE.md), so the filename date is NOT used as a key. Output: rethink/data/raw_extra.parquet,
one row per cohort scan, in cohort.parquet order. Columns are only the scan's own indices; the
label-like columns of the export (KC diagnosis, CXL R/L, KERATOPLASTY, NOTES) are never copied.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
SRC = Path('/home/hadar/Cloud/Work/projects/keratoconus-v2/data/221204_v2/Raw_Combined_IdsMesh.xlsx')
# scan-level indices not already in the 55 numerics; no names, dates, labels or notes
EXTRA = ['TKC:', 'D0mm Prog', 'Ecc. Zone:', 'Pupil Pos X:', 'Pupil Pos Y:', 'Axis F (flat):',
         'Axis B (flat):', 'RSagMin:', 'Pachy Min Pos X:', 'Pachy Min Pos Y:']


def main():
    raw = pd.read_excel(SRC)
    raw['eye'] = raw['Exam Eye:'].str.strip().str.lower()
    raw['time'] = raw['Exam Time:'].astype(str).str.replace(':', '').str[:6]
    raw['idn'] = raw['idp'].astype(str).str.zfill(8)
    k = pd.read_csv(DATA / 'private_keys.csv', dtype=str)
    m = re.compile(r'_(od|os)_(\d{8})_(\d{6})_')
    parts = [m.search(f).groups() for f in k.filename]
    k['eye'] = [{'od': 'right', 'os': 'left'}[p[0]] for p in parts]
    k['time'] = [p[2] for p in parts]
    k['idn'] = k.id_norm.str.zfill(8)
    cols = ['idn', 'eye', 'time', 'Status'] + EXTRA
    keyed = raw[cols].drop_duplicates(['idn', 'eye', 'time'])
    out = k[['idn', 'eye', 'time']].merge(keyed, on=['idn', 'eye', 'time'], how='left')
    assert len(out) == len(k) and out.Status.notna().all(), 'raw match not 1:1'
    out.drop(columns=['idn', 'eye', 'time', 'Status']).to_parquet(DATA / 'raw_extra.parquet', index=False)
    print('matched', len(out), 'scans; status counts:', out.assign(s=keyed.set_index(['idn', 'eye', 'time']).loc[
        list(zip(k.idn, k.eye, k.time)), 'Status'].values).s.value_counts().to_dict())


if __name__ == '__main__':
    sys.exit(main())
