"""E26: images alone. Features = only the image block of the scanned eye and its same-day fellow (own, fellow,
own - fellow) + has_fellow_image; no numerics, no age, no block H. LR with inner patient-grouped CV over C (same
code path as exp22 fit_model). Rows: cohort scans only (no augmentation: extra exams have no image).
Usage: python -I rethink/exp26_imgonly.py ENC [ENC ...]   (Z D L X T R probe multi)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import exp16_nested as e16  # noqa: E402
import exp22_bilateral as e22  # noqa: E402
from exp01_tabular import _inner_cv  # noqa: E402
from harness import DATA, evaluate  # noqa: E402


def make(D, enc):
    def fp(dtr, ctr, dte, cte, seed):
        rows, idx = dtr.crow.values, dte.crow.values
        ef = e22.fit_enc_any(D, enc, rows, seed)

        def X(i):
            return pd.concat([e22.enc_apply_any(D, enc, i, ef),
                              D['flags'][['has_fellow_image']].iloc[i].reset_index(drop=True)], axis=1)
        Xtr, Xte = X(rows), X(idx)
        med = Xtr.median()
        Xtr, Xte = Xtr.fillna(med).fillna(0).values, Xte.fillna(med).fillna(0).values
        y, g = dtr.y.values, dtr.pid.values
        hp = _inner_cv(Xtr, y, g, e16.LR, e16.CGRID, seed)
        return e16.LR(**hp).fit(Xtr, y).predict_proba(Xte)[:, 1]
    return fp


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    D = e22.build(coh)
    for enc in sys.argv[1:]:
        evaluate(f'E26_img_{enc}', make(D, enc), f'E26 images only ({enc}): own/fellow/asym, no numerics/age/H, LR',
                 df=coh, cnn=cnn)
