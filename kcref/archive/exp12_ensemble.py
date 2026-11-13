"""E12: fellow-eye model ensemble on all 944 scans, outer patient-grouped CV (pre-declared, 2 runs).

E12a_E02_all      : E02 fellow LR alone on all scans (its all-data reference; lockbox-best model).
E12b_E02_E06b_avg : mean of the rank-normalised E02 LR and E06b monotone LightGBM predictions,
                    then Platt-recalibrated on inner patient-grouped OOF of the training fold
                    (recalibration fixes the scale_pos_weight bias for calibration/DCA; AUC unaffected
                    by the monotone map, the averaging is the only thing that can change AUC).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from exp02_fellow import fellow_lr  # noqa: E402
from exp06_combo import combo_lgbm  # noqa: E402
from harness import DATA, evaluate  # noqa: E402


def _rank(p_ref, p):
    """Map p to its empirical quantile within the reference (training) predictions."""
    return np.searchsorted(np.sort(p_ref), p, side='right') / len(p_ref)


def avg_recal():
    lr, gbm = fellow_lr(), combo_lgbm()

    def score(dtr, dte, seed):
        a_tr, b_tr = lr(dtr, None, dtr, None, seed), gbm(dtr, None, dtr, None, seed)
        a_te, b_te = lr(dtr, None, dte, None, seed), gbm(dtr, None, dte, None, seed)
        return (_rank(a_tr, a_te) + _rank(b_tr, b_te)) / 2

    def fp(dtr, ctr, dte, cte, seed):
        y, g = dtr.y.values, dtr.pid.values
        oof = np.zeros(len(dtr))
        for a, b in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(dtr, y, g):
            oof[b] = score(dtr.iloc[a].reset_index(drop=True), dtr.iloc[b].reset_index(drop=True), seed)
        cal = LogisticRegression(max_iter=5000).fit(oof[:, None], y)
        return cal.predict_proba(score(dtr, dte, seed)[:, None])[:, 1]
    return fp


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    cnn = np.load(DATA / 'cnn_frozen.npy')
    evaluate('E12a_E02_all', fellow_lr(), 'E02 fellow LR, all 944 scans, outer CV', df=coh, cnn=cnn)
    evaluate('E12b_E02_E06b_avg', avg_recal(),
             'rank-average of E02 LR + E06b LGBM, Platt-recalibrated on inner OOF, all 944 scans',
             df=coh, cnn=cnn)
