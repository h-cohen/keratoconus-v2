"""Design-matrix column names in model order, built from the feature definitions (no patient data read)."""
import numpy as np
import pandas as pd

import features as F


def columns(use_h):
    eng = F.engineer(pd.DataFrame(np.ones((1, len(F.NUM))), columns=F.NUM))
    own = list(F.NUM) + [c for c in eng.columns if c != 'age']
    rest = [c for c in own if c != 'age']
    n = own + ['fel_' + c for c in rest] + ['asym_' + c for c in rest] + ['has_fellow'] + \
        ['worse_' + c for c in F.WORSE]
    r = [f'R_{w}{j}' for w in ('own', 'fel', 'asym') for j in range(96)]
    return n + (F.H_COLS if use_h else []) + ['has_fellow_image', 'own_image'] + r
