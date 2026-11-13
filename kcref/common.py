"""Paths and the numeric feature list shared by every step.

KC_SRC (optional): folder holding the source `data/` (exports, images); default is the repository root.
Built tables go to kcref/data/ and logs to kcref/results/ (both gitignored: patient data).
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
RES = ROOT / 'results'
SRC = Path(os.environ.get('KC_SRC', ROOT.parent)) / 'data'
META = {'pid', 'eid', 'eye', 'scan_date', 'visit_idx', 'first_visit', 'y', 'months_to_cxl', 'legacy_row'}


def numeric_features():
    """Age + the 54 Pentacam numerics, in cohort column order (numeric_features.txt when no cohort is built)."""
    if not (DATA / 'cohort.parquet').exists():
        return (ROOT / 'numeric_features.txt').read_text().split('\n')[:-1]
    return [c for c in pd.read_parquet(DATA / 'cohort.parquet').columns if c not in META]
