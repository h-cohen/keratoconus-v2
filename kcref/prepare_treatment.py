"""Step 4: per-eye treatment dates for block H (fellow-eye treatment before the visit; H arm only).

Source: the clinic sheet data/221204_v2/Combined_excel_KC.xlsx, the same sheet that defines the label. For every
patient-eye it gives the CXL date and the keratoplasty date. A non-empty cell that cannot be parsed as a date is
recorded as undated and is never treated as past. features.block_H uses only dates strictly before the visit.
Output kcref/data/eye_tx.parquet (gitignored): pid, eye, cxl_date, cxl_undated, kp_date, kp_undated.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, SRC  # noqa: E402
from prepare_data import pid_of, to_date  # noqa: E402


def _filled(v):
    return pd.notna(v) and str(v).strip() not in ('', 'nan', 'NaT')


def main():
    c = pd.read_excel(SRC / '221204_v2' / 'Combined_excel_KC.xlsx')
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


if __name__ == '__main__':
    sys.exit(main())
