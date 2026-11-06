"""Build the frozen, pseudonymised analysis cohort for the clean re-analysis.

Inputs (never committed):
  multimodal_dataset.csv, multimodal_dataset_features.txt   (scan-level table)
  Combined_excel_KC.xlsx                                     (per-patient CXL / keratoplasty dates)
  cnn_features.npy                                           (frozen ResNet-18 features, row-aligned
                                                              with the legacy 997-scan cohort)
Outputs (rethink/data/, gitignored):
  cohort.parquet     one row per eligible scan; no names / national IDs
  cnn_frozen.npy     CNN features re-aligned to cohort.parquet
  private_keys.csv   pid <-> original id / image filename (stays local)
  cohort_report.txt  exclusion flow

Cohort rules (frozen 2026-11-06, see PROTOCOL.md):
  1. legacy inclusion: all four maps present, no missing numeric feature, 'Km F (D):' <= 55
     (kept identical to the submitted pipeline so results are comparable; note that the
     filter is on mean front K, not KMax)
  2. drop scans of CXL-labelled eyes taken on/after the eye's CXL date (already treated)
  3. drop CXL-labelled eyes with no usable CXL date (timing cannot be verified)
  4. drop scans taken on/after a keratoplasty date in that eye
"""
import argparse
import os
import datetime
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SALT = 'kc-v2-rethink'


def to_date(v):
    if isinstance(v, (datetime.datetime, pd.Timestamp)):
        return pd.Timestamp(v)
    if isinstance(v, str):
        t = pd.to_datetime(v, errors='coerce', dayfirst=True)
        return t if pd.notna(t) and t.year > 1990 else pd.NaT
    return pd.NaT


def pid_of(x):
    return 'P' + hashlib.sha256((SALT + str(x)).encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.environ.get('KC_SRC', str(Path(__file__).resolve().parents[1])),
                    help='repo checkout that holds data/ and results/age_subanalysis/ (env KC_SRC)')
    ap.add_argument('--out', default=str(Path(__file__).parent / 'data'))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    feats = [l.strip() for l in open(src / 'data/multimodal_dataset_features.txt') if l.strip()]
    df = pd.read_csv(src / 'data/multimodal_dataset.csv', dtype={'id': str})
    log.append(f'raw rows: {len(df)}')
    df = df[df['has_all_images']].dropna(subset=feats).copy()
    df = df[df['Km F (D):'] <= 55].reset_index(drop=True)
    log.append(f'legacy cohort (4 maps, complete numerics, Km F<=55): {len(df)} scans, '
               f'{df.ideye.nunique()} eyes, {df.id.nunique()} patients, {df.y.sum()} CXL-labelled')
    cnn = np.load(src / 'results/age_subanalysis/cnn_features.npy')
    assert len(cnn) == len(df), 'CNN cache misaligned with legacy cohort'
    df['legacy_row'] = np.arange(len(df))

    c = pd.read_excel(src / 'data/221204_v2/Combined_excel_KC.xlsx')
    c['id'] = c['ID'].astype(str).str.replace('-', '').str.replace(r'\.0$', '', regex=True).str.zfill(8)
    rows = []
    for side, s in [('right', 'R'), ('left', 'L')]:
        t = c[['id', f'CXL {s}', f'KERATOPLASTY {s}']].copy()
        t.columns = ['id', 'cxl_raw', 'kp_raw']
        t['eye'] = side
        rows.append(t)
    clin = pd.concat(rows).drop_duplicates(['id', 'eye'])
    clin['cxl_date'] = clin['cxl_raw'].map(to_date)
    clin['kp_date'] = clin['kp_raw'].map(to_date)

    df['eye'] = df['ideye'].str.split('-').str[-1]
    df['id_norm'] = df['id'].astype(str).str.zfill(8)
    df = df.merge(clin[['id', 'eye', 'cxl_date', 'kp_date']], left_on=['id_norm', 'eye'],
                  right_on=['id', 'eye'], how='left', suffixes=('', '_c'))
    df['scan_date'] = pd.to_datetime(df['scan_date'])  # filename date; 'Exam Date:' has d/m swaps
    df['months_to_cxl'] = (df['cxl_date'] - df['scan_date']).dt.days / 30.44

    n0 = len(df)
    post = (df.y == 1) & (df.months_to_cxl <= 0)
    df = df[~post]
    log.append(f'excluded {post.sum()} CXL-labelled scans taken on/after CXL date')
    nodate = (df.y == 1) & df.cxl_date.isna()
    df = df[~nodate]
    log.append(f'excluded {nodate.sum()} CXL-labelled scans with no usable CXL date')
    kp = df.kp_date.notna() & (df.scan_date >= df.kp_date)
    df = df[~kp]
    log.append(f'excluded {kp.sum()} scans on/after keratoplasty')
    neg_with_cxl = (df.y == 0) & df.cxl_date.notna()
    log.append(f'WARNING: {neg_with_cxl.sum()} label-0 scans have a CXL date in the clinical sheet '
               '(kept; label/sheet disagreement to review)')
    df = df.reset_index(drop=True)

    df['pid'] = df['id_norm'].map(pid_of)
    df['eid'] = df['pid'] + '-' + df['eye']
    df = df.sort_values(['eid', 'scan_date', 'scan_time']).reset_index(drop=True)
    df['visit_idx'] = df.groupby('eid').cumcount()
    df['first_visit'] = df['visit_idx'] == 0

    keep = ['pid', 'eid', 'eye', 'scan_date', 'visit_idx', 'first_visit', 'y', 'months_to_cxl',
            'legacy_row'] + feats
    df[['pid', 'id_norm', 'filename']].to_csv(out / 'private_keys.csv', index=False)
    df[keep].to_parquet(out / 'cohort.parquet', index=False)
    np.save(out / 'cnn_frozen.npy', cnn[df['legacy_row'].values])

    e = df.groupby('eid').y.first()
    log.append(f'FINAL: {len(df)} scans, {df.eid.nunique()} eyes, {df.pid.nunique()} patients; '
               f'{df.y.sum()} positive scans ({df.y.mean():.1%}); positive eyes {e.sum()}/{len(e)}')
    log.append(f'first-visit scans: {df.first_visit.sum()} ({df[df.first_visit].y.sum()} positive)')
    (out / 'cohort_report.txt').write_text('\n'.join(log) + '\n')
    print('\n'.join(log))


if __name__ == '__main__':
    main()
