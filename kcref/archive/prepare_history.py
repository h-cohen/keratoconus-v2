"""Pseudonymised long-format table of ALL raw Pentacam exams, for 'visit + history' models.

Source: Raw_Combined_IdsMesh.xlsx (11,760 exams). It stacks two exports:
  rows <  10125 (old export): text dates are US month-first ('09/13/2017'); values Excel turned into
                              dates were read day-first, so their day and month are swapped.
  rows >= 10125 (new export): dates stored correctly.
The parsed date must equal the filename date for every cohort scan, or the script stops.

Output rethink/data/raw_exams.parquet: pid (same salted hash as cohort.parquet), eye, date, time,
status, and the scan's numeric indices. Names, birth dates, ids, notes, KC diagnosis, CXL and
keratoplasty fields are dropped.
"""
import os
import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data import pid_of  # noqa: E402

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
SRC = Path(os.environ.get('KC_SRC', Path(__file__).resolve().parents[1]), 'data/221204_v2/Raw_Combined_IdsMesh.xlsx')
NEW_EXPORT_FROM = 10125
KEEP = ['Km F (D):', 'K1 F (D):', 'K2 F (D):', 'Astig F (D):', 'K1 B (D):', 'K2 B (D):', 'Pachy Apex:',
        'Pachy Min:', 'ISV:', 'IVA:', 'KI:', 'CKI:', 'IHA:', 'IHD:', 'RSagMin:', 'KMax Sagittal Front (D)',
        'Asph. Q F:', 'Asph. Q B:', 'Num. Ecc. F:', 'C.Vol D 3mm:']


def parse_date(v, row):
    if isinstance(v, (dt.datetime, pd.Timestamp)):
        if row >= NEW_EXPORT_FROM:
            return pd.Timestamp(v.year, v.month, v.day)
        try:
            return pd.Timestamp(v.year, v.day, v.month)
        except ValueError:
            return pd.NaT
    if isinstance(v, str) and re.fullmatch(r'\d\d/\d\d/\d{4}', v.strip()):
        return pd.to_datetime(v.strip(), format='%m/%d/%Y')
    return pd.NaT


def main():
    raw = pd.read_excel(SRC)
    raw['date'] = [parse_date(v, i) for i, v in enumerate(raw['Exam Date:'])]
    raw['eye'] = raw['Exam Eye:'].astype(str).str.strip().str.lower()
    raw['time'] = raw['Exam Time:'].astype(str).str.replace(':', '').str[:6]
    raw['idn'] = raw['idp'].astype(str).str.zfill(8)
    raw = raw[raw.date.notna() & raw.eye.isin(['right', 'left'])].copy()

    # validation against the cohort's filename dates
    k = pd.read_csv(DATA / 'private_keys.csv', dtype=str)
    parts = [re.search(r'_(od|os)_(\d{8})_(\d{6})_', f).groups() for f in k.filename]
    k['eye'] = [{'od': 'right', 'os': 'left'}[a] for a, _, _ in parts]
    k['fdate'] = [pd.to_datetime(d, format='%d%m%Y') for _, d, _ in parts]
    k['time'] = [t for _, _, t in parts]
    k['idn'] = k.id_norm.str.zfill(8)
    j = k.merge(raw.drop_duplicates(['idn', 'eye', 'time']), on=['idn', 'eye', 'time'], how='left')
    agree = int((j.date == j.fdate).sum())
    assert agree == len(k), f'date parse agrees for only {agree}/{len(k)} cohort scans'

    out = raw[['idn', 'eye', 'date', 'time', 'Status'] + KEEP].drop_duplicates(['idn', 'eye', 'date', 'time'])
    for c in KEEP:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out['status_ok'] = out.Status.astype(str).str.strip().eq('OK')
    out['pid'] = out.idn.map(pid_of)
    out = out.drop(columns=['idn', 'Status']).sort_values(['pid', 'eye', 'date', 'time']).reset_index(drop=True)
    out.to_parquet(DATA / 'raw_exams.parquet', index=False)
    print(f'dates agree for {agree}/{len(k)} cohort scans; wrote {len(out)} exams, '
          f'{out.pid.nunique()} patients, {int(out.status_ok.sum())} with status OK')


if __name__ == '__main__':
    main()
