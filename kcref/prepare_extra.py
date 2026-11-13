"""Step 3: extra labelled single-visit exams from the raw Pentacam export, for training only.

The raw export (Raw_Combined_IdsMesh.xlsx, 11,760 exams) carries the 55 numerics and the CXL fields for
every exam. Its CXL field agrees with the cohort label for 944/944 cohort scans, its numerics match the
cohort exactly (except the sign of 'Pachy Min Pos X:' for right eyes, harmonised here) and cohort age is
exactly scan year - birth year. Exams that are not cohort scans (no images, or not in the image export)
are kept when the patient is in the CXL clinic sheet (Combined_excel_KC.xlsx; patients outside it all
carry label 0 because they were never followed in the clinic, so their label does not mean the same
thing) and the cohort rules hold: complete numerics, Km F <= 55, CXL eyes only before their CXL date and
with a usable date, no exam on/after keratoplasty.
Output kcref/data/extra_exams.parquet: pid (cohort hash), eye, date, y, status_ok, cohort_patient,
age + 54 numerics. No names, ids or dates of birth. Extras of cohort patients may only be used when the
patient is in the training fold (features.train_rows enforces this).
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data import parse_date, pid_of, to_date  # noqa: E402

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
SRC = Path(os.environ.get('KC_SRC', Path(__file__).resolve().parents[1]), 'data/221204_v2')


def main():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    keys = pd.read_csv(DATA / 'private_keys.csv', dtype=str)
    num = [l.strip() for l in open(ROOT / 'numeric_features.txt') if l.strip() and l.strip() != 'age']
    raw = pd.read_excel(SRC / 'Raw_Combined_IdsMesh.xlsx')
    raw['date'] = [parse_date(v, i) for i, v in enumerate(raw['Exam Date:'])]
    raw['eye'] = raw['Exam Eye:'].astype(str).str.strip().str.lower()
    raw['time'] = raw['Exam Time:'].astype(str).str.replace(':', '').str[:6]
    raw['idn'] = raw['idp'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8)
    raw = raw[raw.date.notna() & raw.eye.isin(['right', 'left'])].copy()
    for c in num:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw.loc[raw.eye == 'right', 'Pachy Min Pos X:'] *= -1  # cohort convention
    raw['age'] = raw.date.dt.year - pd.to_datetime(raw['D.o.Birth:'], errors='coerce').dt.year

    # checks against the cohort
    keys['eye'], keys['idn'] = coh.eye.values, keys.id_norm.str.zfill(8)
    keys['time'] = [re.search(r'_(\d{6})_4', f).group(1) for f in keys.filename]
    j = keys.merge(raw.drop_duplicates(['idn', 'eye', 'time']), on=['idn', 'eye', 'time'], how='left')
    cxl = np.where(j.eye == 'right', j['CXL R'].notna(), j['CXL L'].notna()).astype(int)
    assert (cxl == coh.y.values).all(), 'raw CXL field disagrees with the cohort label'
    bad = ~np.isclose(j[num + ['age']].values, coh[num + ['age']].values, atol=1e-6)
    print('cells differing from the cohort:', {c: int(b) for c, b in zip(num + ['age'], bad.sum(0)) if b})
    assert bad.mean() < 1e-3, 'raw numerics differ'

    sheet = pd.read_excel(SRC / 'Combined_excel_KC.xlsx')
    sheet_ids = set(sheet['ID'].astype(str).str.replace('-', '').str.replace(r'\.0$', '', regex=True).str.zfill(8))
    r = raw[raw.idn.isin(sheet_ids)].copy()
    r['cxl_date'] = np.where(r.eye == 'right', r['CXL R'], r['CXL L'])
    r['y'] = pd.notna(r.cxl_date).astype(int)
    r['cxl_date'] = r.cxl_date.map(to_date)
    r['kp_date'] = np.where(r.eye == 'right', r['KERATOPLASTY R'], r['KERATOPLASTY L'])
    r['kp_date'] = r.kp_date.map(to_date)
    n = [('KC clinic patients, all exams', len(r))]
    r = r.dropna(subset=num + ['age'])
    r = r[r['Km F (D):'] <= 55]
    n.append(('complete numerics, Km F <= 55', len(r)))
    r = r[~((r.y == 1) & (r.cxl_date.isna() | ((r.cxl_date - r.date).dt.days <= 0)))]
    r = r[~(r.kp_date.notna() & (r.date >= r.kp_date))]
    n.append(('pre-CXL with dated CXL, pre-keratoplasty', len(r)))
    cohort_keys = set(zip(keys.idn, keys.eye, keys.time))
    r = r[[k not in cohort_keys for k in zip(r.idn, r.eye, r.time)]]
    n.append(('excluding the 944 cohort scans', len(r)))

    r['pid'] = r.idn.map(pid_of)
    r['cohort_patient'] = r.pid.isin(set(coh.pid))
    r['status_ok'] = r.Status.astype(str).str.strip().eq('OK')
    out = r[['pid', 'eye', 'date', 'y', 'status_ok', 'cohort_patient', 'age'] + num].reset_index(drop=True)
    out.to_parquet(DATA / 'extra_exams.parquet', index=False)
    for a, b in n:
        print(f'{a:45s} {b}')
    print(out.groupby(['cohort_patient', 'status_ok']).agg(scans=('y', 'size'), patients=('pid', 'nunique'),
                                                           positive=('y', 'sum'), rate=('y', 'mean')).round(3))


if __name__ == '__main__':
    sys.exit(main())
