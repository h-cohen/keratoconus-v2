"""Feature construction for one visit. Every row of the design matrix is a scan; everything except block H is
label-free.

Rows (`visits`): the 944 cohort scans (row = cohort row), then extra labelled exams (training only), then raw
exams that serve only as same-day fellows.
Blocks:
  N      age + 54 numerics + 11 engineered indices for the scanned eye; the same 65 non-age values for the fellow;
         own − fellow; 8 'worse eye' indicators; has_fellow                                   (205 columns)
  R      96 map measurements (radiomics.py) of the own and the fellow map, standardised in fold;
         own, fellow, own − fellow                                                            (288 columns)
  flags  has_fellow_image, own_image                                                          (2 columns)
  H      fellow-eye treatment before the visit (H arm only)                                   (5 columns)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import DATA, numeric_features

NUM = numeric_features()  # age + 54
WORSE = {'KMax Sagittal Front (D)': 1, 'K2 F (D):': 1, 'ISV:': 1, 'KI:': 1, 'IHD:': 1, 'IVA:': 1, 'prog_max': 1,
         'Pachy Min:': -1}
H_COLS = ['fellow_cxl_before_visit', 'years_since_fellow_cxl', 'fellow_kp_before_visit', 'fellow_cxl_unknown',
          'fellow_tx_missing']


def engineer(d):
    """11 keratoconus indices (+ age) from one scan's numerics."""
    o = pd.DataFrame(index=d.index)
    o['age'] = d.age
    o['kmax_minus_km'] = d['KMax Sagittal Front (D)'] - d['Km F (D):']
    o['pachy_min_over_apex'] = d['Pachy Min:'] / d['Pachy Apex:']
    o['pachy_min_over_pupil'] = d['Pachy Min:'] / d['Pachy Pupil:']
    o['pachy_min_per_age'] = d['Pachy Min:'] / d.age
    prog = [c for c in NUM if c.endswith('Prog') and c.startswith('D')]
    o['prog_mean'] = d[prog].mean(axis=1)
    o['prog_max'] = d[prog].max(axis=1)
    o['pachy_drop_4mm'] = d['D0mm Pachy'] - d['D4mm Pachy']
    o['ecc_front_back'] = d['Num. Ecc. F:'] - d['Num. Ecc. B:']
    o['asph_ratio'] = d['Asph. Q F:'] / d['Asph. Q B:'].replace(0, np.nan)
    o['elev_ratio'] = d['Rf F (mm):'] / d['Rf B (mm):']
    o['cvol_ratio'] = d['C.Vol D 3mm:'] / d['C.Vol D 10mm:']
    o = o.replace([np.inf, -np.inf], np.nan)
    return o.fillna(o.median())


def visits(coh):
    """Cohort rows, extra exams and fellow-only raw exams; `fel` = row of the same-day fellow exam or -1.
    The fellow is any same-day exam of the other eye in the raw export (status OK first, then fewest missing
    numerics); no label or treatment field is read."""
    c = coh[['pid', 'eye', 'y'] + NUM].copy()
    c['date'], c['status_ok'], c['src'] = pd.to_datetime(coh.scan_date), True, 'cohort'
    ex = pd.read_parquet(DATA / 'extra_exams.parquet')
    e = ex[['pid', 'eye', 'y', 'date', 'status_ok'] + NUM].copy()
    e['src'] = 'extra'
    v = pd.concat([c, e], ignore_index=True)
    v['cohort_patient'] = v.pid.isin(set(coh.pid))
    r = pd.read_parquet(DATA / 'raw_all_exams.parquet')
    r = r.sort_values(['status_ok', 'n_missing'], ascending=[False, True], kind='stable')
    r = r.drop_duplicates(['pid', 'eye', 'date'])
    r = r.assign(y=np.nan, src='fellow_only', cohort_patient=r.pid.isin(set(coh.pid)))[list(v.columns)]
    n = len(v)
    v = pd.concat([v, r], ignore_index=True)
    lut = {(p, e_, d): n + i for i, (p, e_, d) in enumerate(zip(r.pid, r.eye, r.date))}
    other = np.where(v.eye.values[:n] == 'right', 'left', 'right')
    fel = np.full(len(v), -1)
    fel[:n] = [lut.get((p, o, d), -1) for p, o, d in zip(v.pid.values[:n], other, v.date.values[:n])]
    v['fel'] = fel
    return v


def block_N(v):
    eng = engineer(v.assign(age=v.age)).replace([np.inf, -np.inf], np.nan)
    own = pd.concat([v[NUM], eng.drop(columns='age')], axis=1)
    has = v.fel.values >= 0
    fidx = np.maximum(v.fel.values, 0)
    cols = [c for c in own.columns if c != 'age']
    fel = own[cols].iloc[fidx].reset_index(drop=True).where(np.repeat(has[:, None], len(cols), 1))
    fel.columns = ['fel_' + c for c in cols]
    asym = pd.DataFrame(own[cols].values - fel.values, columns=['asym_' + c for c in cols])
    worse = pd.DataFrame({f'worse_{c}': np.where(has, (s * (own[c].values - own[c].values[fidx]) > 0).astype(float),
                                                 np.nan) for c, s in WORSE.items()})
    return pd.concat([own.reset_index(drop=True), fel, asym, pd.DataFrame({'has_fellow': has.astype(float)}), worse],
                     axis=1)


def block_H(pid, eye, date):
    """Fellow-eye CXL / keratoplasty strictly before the visit date (clinic treatment sheet, eye_tx.parquet).
    Patients absent from the sheet: NaN except fellow_tx_missing = 1."""
    tx = pd.read_parquet(DATA / 'eye_tx.parquet').set_index(['pid', 'eye'])
    other = np.where(np.asarray(eye) == 'right', 'left', 'right')
    t = tx.reindex(pd.MultiIndex.from_arrays([np.asarray(pid), other]))
    date = pd.to_datetime(pd.Series(np.asarray(date)))
    cxl, kp = pd.to_datetime(t.cxl_date.values), pd.to_datetime(t.kp_date.values)
    missing = t.cxl_undated.isna().values
    before = cxl < date.values
    h = pd.DataFrame({'fellow_cxl_before_visit': before.astype(float),
                      'years_since_fellow_cxl': np.where(before, (date.values - cxl).days / 365.25, 0.0),
                      'fellow_kp_before_visit': (kp < date.values).astype(float),
                      'fellow_cxl_unknown': t.cxl_undated.fillna(False).astype(bool).values.astype(float),
                      'fellow_tx_missing': missing.astype(float)})
    h.loc[missing, H_COLS[:4]] = np.nan
    return h


def build(coh):
    """All label-free tables for every visit row (+ H). D['own'] / D['felc']: row in decoded_hr / emb_R or -1."""
    V = visits(coh)
    meta = pd.read_parquet(DATA / 'decoded_hr_meta.parquet')
    meta['d'] = pd.to_datetime(meta.date).dt.normalize()
    lut = {}
    for i, k in enumerate(zip(meta.pkey, meta.eye, meta.d)):
        lut.setdefault(k, i)
    d = pd.to_datetime(V.date).dt.normalize()
    other = np.where(V.eye.values == 'right', 'left', 'right')
    own = np.array([lut.get(k, -1) for k in zip(V.pid, V.eye, d)])
    cr = meta.cohort_row.values
    own[:len(coh)] = [int(np.flatnonzero(cr == i)[0]) for i in range(len(coh))]  # exact composite of each scan
    felc = np.array([lut.get(k, -1) for k in zip(V.pid, other, d)])
    flags = pd.DataFrame({'has_fellow_image': (felc >= 0).astype(float), 'own_image': (own >= 0).astype(float)})
    first = np.zeros(len(V), bool)
    first[:len(coh)] = coh.first_visit.values
    return dict(V=V, N=block_N(V), H=block_H(V.pid.values, V.eye.values, V.date.values), flags=flags, own=own,
                felc=felc, first=first, R=np.load(DATA / 'emb_R.npy').astype(np.float64))


def train_rows(D, dtr):
    """Training rows of a fold: its cohort scans + extra exams of patients outside the cohort or in the fold."""
    V = D['V']
    ok = (V.src == 'extra') & (~V.cohort_patient | V.pid.isin(set(dtr.pid)))
    return np.array(list(dtr.crow.values) + list(np.flatnonzero(ok.values)))


def fit_R(D, rows):
    """StandardScaler on the training rows' own map measurements (complete rows only)."""
    E, own = D['R'], D['own']
    fit = np.unique(own[rows][own[rows] >= 0])
    return StandardScaler().fit(E[fit[np.isfinite(E[fit]).all(1)]])


def block_R(D, idx, sc):
    E = D['R']

    def tr(c):
        out = np.full((len(c), E.shape[1]), np.nan)
        ok = c >= 0
        ok[ok] = np.isfinite(E[c[ok]]).all(1)
        out[ok] = sc.transform(E[c[ok]])
        return out

    o, f = tr(D['own'][idx]), tr(D['felc'][idx])
    return pd.DataFrame(np.hstack([o, f, o - f]),
                        columns=[f'R_{w}{j}' for w in ('own', 'fel', 'asym') for j in range(E.shape[1])])


def assemble(D, idx, use_h, sc):
    """Design matrix (before clipping / imputation) for visit rows idx: N [+ H] + flags + R."""
    parts = [D['N'].iloc[idx].reset_index(drop=True)]
    if use_h:
        parts.append(D['H'].iloc[idx].reset_index(drop=True))
    parts += [D['flags'].iloc[idx].reset_index(drop=True), block_R(D, idx, sc)]
    return pd.concat(parts, axis=1)
