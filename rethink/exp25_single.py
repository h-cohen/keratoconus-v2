"""E25 standalone runs (for the record): each new image candidate alone, with block H, extra exams and first-visit
weight 2, through harness.evaluate. Usage: python -I rethink/exp25_single.py NAME [NAME ...] (R_lr, probe_lgbm, ...)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import exp22_bilateral as e22  # noqa: E402
from harness import DATA, evaluate  # noqa: E402

if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    D = e22.build(coh)
    c = e22.candidates(D, prefix='', use_h=True, w_first=2.0, aug_all=True, extra_enc=('R', 'probe', 'multi'))
    for n in sys.argv[1:]:
        f = c[f'bi_{n}']
        evaluate(f'E25_bi_{n}', lambda dtr, ctr, dte, cte, seed, f=f: f(dtr, dte, seed),
                 f'E25 standalone: bi_{n}, block H, extra exams, first-visit weight 2', df=coh, cnn=cnn)
