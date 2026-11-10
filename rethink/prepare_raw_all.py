"""E21 step 1: label-free table of EVERY raw Pentacam exam, for same-day fellow lookup.

Source: Raw_Combined_IdsMesh.xlsx (all 11,760 exams). No CXL, keratoplasty, diagnosis or label field is
read, so whether a scan gets a fellow cannot depend on any label or treatment date.
Output rethink/data/raw_all_exams.parquet: pid (cohort hash), eye, date, status_ok, age + 54 numerics
(cohort sign convention; NaN where the export has none).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data import pid_of  # noqa: E402
from prepare_history import parse_date  # noqa: E402

ROOT = Path(__file__).parent
SRC = Path('/home/hadar/Cloud/Work/projects/keratoconus-v2/data/221204_v2/Raw_Combined_IdsMesh.xlsx')


def main():
    num = [l.strip() for l in open(ROOT / 'numeric_features.txt') if l.strip() and l.strip() != 'age']
    raw = pd.read_excel(SRC, usecols=lambda c: c in set(num) | {'idp', 'D.o.Birth:', 'Exam Date:', 'Exam Eye:', 'Status'})
    raw['date'] = [parse_date(v, i) for i, v in enumerate(raw['Exam Date:'])]
    raw['eye'] = raw['Exam Eye:'].astype(str).str.strip().str.lower()
    raw = raw[raw.date.notna() & raw.eye.isin(['right', 'left'])].copy()
    for c in num:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw.loc[raw.eye == 'right', 'Pachy Min Pos X:'] *= -1
    raw['age'] = raw.date.dt.year - pd.to_datetime(raw['D.o.Birth:'], errors='coerce').dt.year
    raw['pid'] = raw.idp.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8).map(pid_of)
    raw['status_ok'] = raw.Status.astype(str).str.strip().eq('OK')
    raw['n_missing'] = raw[num].isna().sum(1)
    out = raw[['pid', 'eye', 'date', 'status_ok', 'n_missing', 'age'] + num].reset_index(drop=True)
    out.to_parquet(ROOT / 'data' / 'raw_all_exams.parquet', index=False)
    print(len(out), 'exams', out.pid.nunique(), 'patients; complete numerics', int((out.n_missing == 0).sum()))


if __name__ == '__main__':
    sys.exit(main())
