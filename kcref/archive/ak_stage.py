"""Amsler-Krumeich stage from the Pentacam numerics (user-supplied table, 2026-11-11).
Mean K = 'Km F (D):' (front mean K): I < 48.0, II 48.0-53.0, III 53.0-55.0, IV > 55.0 D.
Minimal corneal thickness = 'Pachy Min:' (thinnest point; the table says 'minimal apical'): III 300-400 um,
IV < 300 um; > 400 um does not raise the stage. Stage = the highest stage met by either criterion.
Refraction (myopia / astigmatism) and scarring are not in the data and are not used.
Boundaries: K 48.0 -> II, 53.0 -> III, 55.0 -> IV; thickness < 400 -> III, < 300 -> IV (400 itself does not raise).
Note: the cohort filter Km F <= 55 D means K alone almost never gives stage IV here."""
import numpy as np
import pandas as pd


def ak_stage(df):
    k = df['Km F (D):'].values
    t = df['Pachy Min:'].values
    sk = np.select([k < 48, k < 53, k < 55], [1, 2, 3], 4).astype(float)
    sk[np.isnan(k)] = np.nan
    st = np.select([t < 300, t < 400], [4, 3], 1).astype(float)
    st[np.isnan(t)] = np.nan
    return pd.Series(np.fmax(sk, st), index=df.index, name='ak_stage')
