"""Figure helpers (E29). Synthetic data only. Run: python -I publication_v2/test_figures.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plots as FG  # noqa: E402


def test_stage_auc_small():
    rng = np.random.default_rng(0)
    n = 400
    stage = np.r_[np.repeat([1, 2, 3], 132), np.full(4, 4)].astype(float)
    y = (rng.random(n) < 0.3).astype(int)
    y[stage == 4] = 0
    g = np.arange(n)
    s = y + rng.normal(size=n)
    df = FG.stage_auc(y, {'m': s}, stage, g)
    r4 = df[(df.method == 'm') & (df.stage == 4)].iloc[0]
    assert r4.small and np.isnan(r4.auc) and r4.n == 4
    for k in (1, 2, 3):
        r = df[(df.method == 'm') & (df.stage == k)].iloc[0]
        assert not r.small and r.lo <= r.auc <= r.hi, r


def test_sector_orientation():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    patches = FG.dartboard(ax, {(2, 3, 'N'): 1.0, (0, 1, 'S'): 0.5}, vmax=1.0)
    def centre(gid):
        v = patches[[p.get_gid() for p in patches].index(gid)]
        xy = v.get_patch_transform().transform(v.get_path().vertices)  # data coordinates (mm)
        return xy[:, 0].mean(), xy[:, 1].mean()
    x, y = centre('r23_N')
    assert x > 0 and abs(y) < x, (x, y)
    x, y = centre('r01_S')
    assert y > 0 and abs(x) < y, (x, y)
    assert len(patches) == 16
    plt.close(fig)


def test_map_orientation():
    import radiomics
    a = np.zeros((145, 145))
    a[30:40, 100:110] = 1  # rows above the centre, columns right of it
    x, y = FG.map_centroid_mm(a)
    assert x > 0 and y > 0  # nasal (+x) and superior (+y), as in radiomics.grid
    assert FG.MAP_EXTENT == (radiomics.X.min(), radiomics.X.max(), radiomics.Y.min(), radiomics.Y.max())


def test_no_identifiers():
    """Committed outputs carry no identifier columns or cohort pids; the local folder is gitignored."""
    import subprocess
    import pandas as pd
    root = Path(__file__).parent
    bad = {'pid', 'pkey', 'date', 'scan_date', 'name', 'file', 'eid'}
    for t in (root / 'tables').glob('*.csv'):
        assert not bad & {c.lower() for c in pd.read_csv(t, nrows=1).columns}, t
    assert subprocess.run(['git', 'check-ignore', '-q', str(root / 'local' / 'x.png')]).returncode == 0
    html = (root / 'report.html').read_text(encoding='utf-8')
    pids = set(pd.read_parquet(FG.DATA / 'cohort.parquet').pid.astype(str))
    assert not any(p in html for p in pids if len(p) >= 6)


if __name__ == '__main__':
    for t in [v for k, v in dict(globals()).items() if k.startswith('test_')]:
        t()
    print('ok')
