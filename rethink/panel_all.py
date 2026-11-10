"""E22 step 6: blocks P (printed panel values) and M (decoded-map indices) for EVERY composite, so that fellow eyes
whose composite is not a cohort scan get them too (design 6.1). No labels are read.

P: the template OCR of ocr_panel.py (templates learned exactly as there, from the same half of the cohort images
   and the same known fields), applied to all composites; exp17_rawmaps.panel_features turns the fields into
   features (KMax location nasal / superior / decentration, thinnest-to-KMax distance, cornea and chamber volume,
   KPD, angle, ACD, pupil diameter, IOP correction).
M: exp17_rawmaps.map_indices on decoded_all.npy (58 x 58; left eyes un-mirrored back to screen orientation, as
   map_indices expects), using the OCR'd thinnest location.
Check: for the 944 cohort composites both blocks must equal rawmap_features.parquet (E17).
Output rethink/data/panel_all.parquet (2,260 rows, aligned with decoded_hr_meta.parquet).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import ocr_panel as op  # noqa: E402
from decode_hr import files  # noqa: E402
from exp17_rawmaps import map_indices, panel_features  # noqa: E402

DATA = Path(__file__).parent / 'data'


def templates():
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    raw = pd.read_parquet(DATA / 'raw_extra.parquet')
    keys = pd.read_csv(DATA / 'private_keys.csv', dtype=str)
    known = pd.concat([coh, raw[['Pupil Pos X:', 'Pupil Pos Y:']]], axis=1)
    on_disk = {f.name.lower(): f for f in op.IMG.glob('*.JPG')}
    half = np.random.default_rng(0).permutation(len(keys)) < len(keys) // 2
    lab = {}
    train_fields = {k: v for k, v in op.KNOWN.items() if not k.endswith('_x')}
    for i in np.flatnonzero(half):
        im = np.asarray(Image.open(on_disk[keys.filename[i].lower()]).convert('L')).astype(float)
        for k, (col, dec, sg) in train_fields.items():
            v = known[col].iloc[i]
            if pd.isna(v):
                continue
            s = op.fmt(v, dec, sg)
            p = op.glyphs(im, op.FIELDS[k])
            if len(s) == len(p):
                for ch, q in zip(s, p):
                    lab.setdefault(ch, []).append(q)
    return {ch: np.mean(v, 0) for ch, v in lab.items()}


def main():
    T = templates()
    fs = files()
    meta = pd.read_parquet(DATA / 'decoded_hr_meta.parquet')
    rows = []
    for f in fs:
        im = np.asarray(Image.open(f).convert('L')).astype(float)
        rows.append({k: op.to_float(op.read(T, op.glyphs(im, b))) for k, b in op.FIELDS.items()})
    o = pd.DataFrame(rows)
    P = panel_features(meta[['eye']], o)
    V = np.load(DATA / 'decoded_all.npy').copy()
    left = meta.eye.values == 'left'
    V[left] = V[left][:, :, :, ::-1]
    M = map_indices(meta, o, V)
    out = pd.concat([P, M], axis=1)
    out.to_parquet(DATA / 'panel_all.parquet', index=False)
    ref = pd.read_parquet(DATA / 'rawmap_features.parquet')
    c = meta.cohort_row.values
    mine = out.iloc[np.flatnonzero(c >= 0)].reset_index(drop=True)
    ref = ref.iloc[c[c >= 0]].reset_index(drop=True)
    eq = {k: float(np.mean(np.isclose(mine[k], ref[k], equal_nan=True))) for k in ref.columns}
    print('agreement with rawmap_features on the 944 cohort composites:', eq)
    print(out.describe().T[['count', 'mean', 'min', 'max']].round(2).to_string())


if __name__ == '__main__':
    sys.exit(main())
