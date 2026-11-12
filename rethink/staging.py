"""Final staging for the stage analyses (user, 2026-11-12), two approaches reported side by side.
A  'KMax Sagittal Front (D)' (user 2026-11-12: KMax sagittal, not Km F): 1 < 48; 2 48-53; 3 53-55; 4 >= 55 D.
B  'Rm B (mm):' (back mean radius): 1 >= 5.70; 2 5.15-5.70; 3 4.95-5.15; 4 < 4.95 mm.
Boundary values go to the milder stage for B (5.15 -> 2, 4.95 -> 3) and to the higher stage for A (48.0 -> 2),
i.e. each stage is a half-open interval [lower, upper) in the parameter's own direction."""
import numpy as np
import pandas as pd


def stage_A(df):
    k = df['KMax Sagittal Front (D)'].values
    s = np.select([k < 48, k < 53, k < 55], [1, 2, 3], 4).astype(float)
    s[np.isnan(k)] = np.nan
    return s


def stage_B(df):
    r = df['Rm B (mm):'].values
    s = np.select([r >= 5.70, r >= 5.15, r >= 4.95], [1, 2, 3], 4).astype(float)
    s[np.isnan(r)] = np.nan
    return s
