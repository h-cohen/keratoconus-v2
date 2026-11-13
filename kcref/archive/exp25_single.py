"""E25 standalone runs (for the record): each new image candidate alone, with block H, extra exams and first-visit
weight 2, through harness.evaluate. Usage: python -I rethink/exp25_single.py [--noH] NAME [NAME ...] (R_lr, probe_lgbm, ...).
--noH: no block H (primary arm, user decision 2026-11-12); run names get a noH_ infix."""
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
    noh = '--noH' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--noH']
    c = e22.candidates(D, prefix='', use_h=not noh, w_first=2.0, aug_all=True, extra_enc=('R', 'probe', 'multi', 'G0', 'G'))
    for n in args:
        f = c[f'bi_{n}']
        evaluate(f'{"E27" if n.startswith("G") else "E25"}_{"noH_" if noh else ""}bi_{n}', lambda dtr, ctr, dte, cte, seed, f=f: f(dtr, dte, seed),
                 f'E25 standalone: bi_{n}, {"no block H" if noh else "block H"}, extra exams, first-visit weight 2', df=coh, cnn=cnn)
