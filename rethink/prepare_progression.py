"""E19 step 1: self-supervised progression pairs from the full raw Pentacam record (no referral labels).

Source: Raw_Combined_IdsMesh.xlsx, all 11,760 exams of all patients. For each eye, every status-OK exam
with complete numerics is paired with the next status-OK exam of the same eye 91 days to 3 years later.
Targets (per year): change in KMax, thinnest pachymetry and K2 front. Exams on/after a CXL or keratoplasty
of that eye are dropped (treatment changes the cornea), as are eyes with an undated CXL. The CXL fields
are used only for that filter; no label is written.
Output rethink/data/progression_pairs.parquet: pid (cohort hash), eye, gap_years, the index exam's age +
54 numerics (cohort sign convention), d_kmax, d_pachy, d_k2 (per year).
Which pairs may train the pretext model is decided per fit in exp19_progression.py (never test-fold patients).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data import pid_of, to_date  # noqa: E402
from prepare_history import parse_date  # noqa: E402

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
SRC = Path('/home/hadar/Cloud/Work/projects/keratoconus-v2/data/221204_v2/Raw_Combined_IdsMesh.xlsx')
TARGETS = {'d_kmax': 'KMax Sagittal Front (D)', 'd_pachy': 'Pachy Min:', 'd_k2': 'K2 F (D):'}


def main():
    num = [l.strip() for l in open(ROOT / 'numeric_features.txt') if l.strip() and l.strip() != 'age']
    raw = pd.read_excel(SRC)
    raw['date'] = [parse_date(v, i) for i, v in enumerate(raw['Exam Date:'])]
    raw['eye'] = raw['Exam Eye:'].astype(str).str.strip().str.lower()
    raw['idn'] = raw['idp'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8)
    raw = raw[raw.date.notna() & raw.eye.isin(['right', 'left'])].copy()
    if '--any-status' not in sys.argv:
        raw = raw[raw.Status.astype(str).str.strip().eq('OK')]
    for c in num:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw.loc[raw.eye == 'right', 'Pachy Min Pos X:'] *= -1
    raw['age'] = raw.date.dt.year - pd.to_datetime(raw['D.o.Birth:'], errors='coerce').dt.year
    raw = raw.dropna(subset=num + ['age'])

    cxl_raw = np.where(raw.eye == 'right', raw['CXL R'], raw['CXL L'])
    cxl = pd.Series(cxl_raw, index=raw.index).map(to_date)
    kp = pd.Series(np.where(raw.eye == 'right', raw['KERATOPLASTY R'], raw['KERATOPLASTY L']), index=raw.index).map(to_date)
    treated = (pd.notna(cxl_raw) & (cxl.isna() | (raw.date >= cxl))) | (kp.notna() & (raw.date >= kp))
    raw = raw[~treated.values]
    raw = raw.sort_values(['idn', 'eye', 'date']).drop_duplicates(['idn', 'eye', 'date'])

    rows = []
    for (idn, eye), g in raw.groupby(['idn', 'eye']):
        d = g.date.values
        for i in range(len(g)):
            gap = (d - d[i]).astype('timedelta64[D]').astype(int)
            nxt = np.flatnonzero((gap >= 91) & (gap <= 3 * 365))
            if not len(nxt):
                continue
            j, yrs = nxt[0], gap[nxt[0]] / 365.25
            r = {'pid': pid_of(idn), 'eye': eye, 'gap_years': yrs, 'age': g.age.iloc[i]}
            r.update({c: g[c].iloc[i] for c in num})
            r.update({t: (g[c].iloc[j] - g[c].iloc[i]) / yrs for t, c in TARGETS.items()})
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_parquet(DATA / 'progression_pairs.parquet', index=False)
    print(f'{len(out)} pairs, {out.pid.nunique()} patients, {out.groupby(["pid", "eye"]).ngroups} eyes')
    print(out[['gap_years'] + list(TARGETS)].describe().round(2).to_string())


if __name__ == '__main__':
    sys.exit(main())
