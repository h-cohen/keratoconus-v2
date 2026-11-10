"""E22 step 1: per-eye treatment dates for the fellow-eye treatment block H (design U2, C1).

Source: the clinic sheet data/221204_v2/Combined_excel_KC.xlsx (the same sheet that defines the label in
prepare_data.py). For every patient-eye it gives the CXL date and the keratoplasty date. A non-empty CXL /
keratoplasty cell that cannot be parsed as a date is recorded as 'undated' (C1: never treated as past).
Output rethink/data/eye_tx.parquet (gitignored): pid (cohort hash), eye, cxl_date, cxl_undated, kp_date,
kp_undated. No names, ids or dates of birth.

Block H is built at run time by `tx_features(pid, eye_of_fellow, visit_date)` with STRICT `<`: a fellow CXL
on the visit day counts as not yet treated. Patients absent from the sheet get NaN (unknown) everywhere.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data import pid_of, to_date  # noqa: E402

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
SRC = Path(os.environ.get('KC_SRC', Path(__file__).resolve().parents[1]), 'data/221204_v2')
H_COLS = ['fellow_cxl_before_visit', 'years_since_fellow_cxl', 'fellow_kp_before_visit', 'fellow_cxl_unknown',
          'fellow_tx_missing']


def _filled(v):
    return pd.notna(v) and str(v).strip() not in ('', 'nan', 'NaT')


def main():
    c = pd.read_excel(SRC / 'Combined_excel_KC.xlsx')
    c['id'] = c['ID'].astype(str).str.replace('-', '').str.replace(r'\.0$', '', regex=True).str.zfill(8)
    rows = []
    for side, s in [('right', 'R'), ('left', 'L')]:
        for _, r in c[['id', f'CXL {s}', f'KERATOPLASTY {s}']].iterrows():
            cx, kp = r[f'CXL {s}'], r[f'KERATOPLASTY {s}']
            cd, kd = to_date(cx), to_date(kp)
            rows.append(dict(pid=pid_of(r['id']), eye=side, cxl_date=cd, cxl_undated=_filled(cx) and pd.isna(cd),
                             kp_date=kd, kp_undated=_filled(kp) and pd.isna(kd)))
    out = pd.DataFrame(rows).drop_duplicates(['pid', 'eye']).reset_index(drop=True)
    out.to_parquet(DATA / 'eye_tx.parquet', index=False)
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    print(len(out), 'patient-eyes;', out.cxl_date.notna().sum(), 'dated CXL;', out.cxl_undated.sum(), 'undated CXL;',
          out.kp_date.notna().sum(), 'dated KP;', out.kp_undated.sum(), 'undated KP;',
          'cohort patients in sheet:', coh.pid.isin(set(out.pid)).mean().round(3))


_TX = None


def tx_features(pid, eye, date, table=None):
    """Block H for rows (pid, own eye, visit date) -> DataFrame of H_COLS. Uses the OTHER eye's dates, `<` only.
    table: optional DataFrame (pid, eye, cxl_date, cxl_undated, kp_date, kp_undated); default eye_tx.parquet."""
    global _TX
    if table is not None:
        tx = table.set_index(['pid', 'eye'])
    else:
        if _TX is None:
            _TX = pd.read_parquet(DATA / 'eye_tx.parquet').set_index(['pid', 'eye'])
        tx = _TX
    other = np.where(np.asarray(eye) == 'right', 'left', 'right')
    t = tx.reindex(pd.MultiIndex.from_arrays([np.asarray(pid), other]))
    date = pd.to_datetime(pd.Series(np.asarray(date)))
    cxl, kp = pd.to_datetime(t.cxl_date.values), pd.to_datetime(t.kp_date.values)
    missing = t.cxl_undated.isna().values  # patient-eye not in the sheet
    before = (cxl < date.values)
    yrs = np.where(before, (date.values - cxl).days / 365.25, 0.0)
    unk = t.cxl_undated.fillna(False).astype(bool).values
    h = pd.DataFrame({'fellow_cxl_before_visit': before.astype(float), 'years_since_fellow_cxl': yrs,
                      'fellow_kp_before_visit': (kp < date.values).astype(float),
                      'fellow_cxl_unknown': unk.astype(float), 'fellow_tx_missing': missing.astype(float)})
    h.loc[missing, H_COLS[:4]] = np.nan
    return h


if __name__ == '__main__':
    sys.exit(main())
