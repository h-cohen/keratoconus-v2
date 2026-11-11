"""Score the frozen models on an external validation set. Prints AGGREGATES ONLY (no per-patient output).

Usage:
  python -I rethink/score_external.py VALIDATION.csv|.parquet [--mapping cols.csv]
         [--tx TREATMENT.csv] [--composites DIR] [--images DIR]

Models scored (whichever the inputs support):
  Bilateral single visit (rethink/freeze_stack.py -> frozen/bilateral.pkl; E22 freeze kept as bilateral_e22.pkl):
    headline    E24n_H stack refitted on all 944 scans         needs: fellow numerics, --tx, composites [+ crops]
    fallback_a  E24n_noH stack (no block H)                    needs: fellow numerics, composites [+ crops]
    fallback_b  bilateral numerics only (block N)              needs: fellow numerics
    ref         single-eye numeric model (primary comparator)  needs: the scanned eye's numerics
    age         age only (sub-analysis)
  Legacy (rethink/freeze_models.py -> frozen/models.pkl, if present): age, E02, E06b, E12b [, E10c, E13 with --images].

Input file: one row per Pentacam exam.
  pid        patient identifier (any string; fellow join and cluster bootstrap)
  eye        'right'/'left' (OD/OS are mapped)
  scan_date  visit date (used ONLY to find the other eye's exam of the same visit; never a feature)
  y          1 if the eye was later referred for / underwent CXL, 0 if not; EMPTY for exams that only serve as
             the fellow eye of a scored exam (e.g. an already treated fellow eye, or one outside the inclusion rules)
  age + the Pentacam numerics with the training names (rethink/numeric_features.txt); --mapping renames columns
             (2-column CSV external_name,cohort_name).
  composite  (optional) file name, inside --composites DIR, of the exam's full 'OCULUS PENTACAM 4 Maps Selectable'
             screenshot (1200 x 910 JPG, same layout as the training composites). Needed for image blocks.
  image_stem (optional, legacy models and encoder D only) crops DIR/<image_stem>_<map>.jpg as data/cropped_output_v2.
Scored rows: y in {0, 1}. Scans taken on/after the eye's own CXL must be excluded beforehand, as in training.
--tx: per-eye treatment table: pid, eye, cxl_date, kp_date (empty if none), optional cxl_undated / kp_undated
      (1 if the eye was treated but the date is unknown). Only dates strictly BEFORE the scored visit are used.
Needs rethink/frozen/ (bilateral.pkl, enc_*.pt) and rethink/numeric_features.txt; never the training cohort.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from compare import paired_delta  # noqa: E402
from harness import ROOT, cluster_boot_auc  # noqa: E402


# --------------------------------------------------------------------------------------------- E22 bilateral
def needs(fz):
    """(needs H, needs composites, needs crops) of a frozen candidate."""
    if fz.get('ftl'):
        return False, True, False
    if 'stack' in fz or 'ens' in fz:
        r = [needs(f) for f in fz.get('stack', fz.get('ens'))]
        return tuple(any(x[i] for x in r) for i in range(3))
    if 'ens' in fz:
        r = [needs(f) for f in fz['ens']]
        return tuple(any(x[i] for x in r) for i in range(3))
    return fz['use_h'], fz['blocks'] == 'PM' or fz['enc'] in ('Z', 'L', 'X', 'T'), fz['enc'] == 'D'


def decode_composite(path, eye):
    """(4, 145, 145) physical maps (decode_hr) and (4, 58, 58) (decode_maps) of one composite; left eyes mirrored
    in the first, screen orientation in the second (as in training)."""
    from PIL import Image
    import decode_hr
    import decode_maps
    im = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    hr = np.full((4, 145, 145), np.nan, np.float32)
    lo = np.full((4, 58, 58), np.nan, np.float32)
    for j, m in enumerate(decode_maps.MAPS):
        d, _ = decode_maps.decode(im, m)
        if d is not None:
            hr[j], lo[j] = decode_hr.blocks(d), decode_maps.blocks(d)
    return (hr[:, :, ::-1] if eye == 'left' else hr), lo


def embed(model, hrs, crops, eyes):
    """Embeddings per composite for every encoder the frozen models use (same code as training)."""
    import torch
    import torch.nn.functional as F
    out = {}
    for e in model['encoders']:
        if e == 'Z':
            import zernike
            zone, B = zernike.basis(145)
            Z = np.full((len(hrs), 3 * B.shape[1]), np.nan, np.float32)
            for i, v in enumerate(hrs):
                for j, ch in enumerate(zernike.MAPS.values()):
                    x = v[ch].astype(np.float64)[zone]
                    ok = np.isfinite(x)
                    if ok.mean() >= 0.7:
                        Z[i, j * B.shape[1]:(j + 1) * B.shape[1]] = np.linalg.lstsq(B[ok], x[ok], rcond=None)[0]
            out[e] = Z.astype(np.float64)
        elif e == 'D':
            out[e] = dino(crops, eyes)
        else:
            from pretrain_jepa import Encoder
            ck = torch.load(ROOT / 'frozen' / f'enc_{e}.pt', map_location='cpu', weights_only=False)
            net = Encoder()
            net.load_state_dict(ck['enc'])
            net.eval()
            x = np.stack(hrs).astype(np.float32)
            for j, (m, s) in enumerate(model['norm']):
                x[:, j] = (x[:, j] - m) / s
            valid = np.isfinite(x)
            x = torch.from_numpy(np.concatenate([np.nan_to_num(x), valid], 1).astype(np.float16)).float()
            embs = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    z = F.interpolate(x[i:i + 64], size=ck['cfg']['global_size'], mode='bilinear', align_corners=False)
                    z = torch.cat([z[:, :4] * (z[:, 4:] > 0.5), (z[:, 4:] > 0.5).float()], 1)
                    embs.append(net(z).numpy())
            out[e] = np.concatenate(embs).astype(np.float64)
    return out


def dino(stems, eyes):
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained('facebook/dinov2-small', do_center_crop=False,
                                              size={'height': 224, 'width': 224})
    model = AutoModel.from_pretrained('facebook/dinov2-small').eval()
    maps = ['corneal_thickness', 'curvature_front', 'elevation_front', 'elevation_back']
    out = np.full((len(stems), 4 * 384), np.nan)
    for i, (s, e) in enumerate(zip(stems, eyes)):
        if s is None:
            continue
        ims = [Image.open(f'{s}_{m}.jpg').convert('RGB') for m in maps]
        ims = [ImageOps.mirror(im) if e == 'left' else im for im in ims]
        with torch.no_grad():
            h = model(**proc(images=ims, return_tensors='pt')).last_hidden_state[:, 1:].mean(1)
        out[i] = h.numpy().reshape(-1)
    return out


def panel(model, comp_paths, lo_maps, eyes):
    """Block P + M per composite (ocr_panel templates frozen with the model; exp17 features)."""
    from PIL import Image
    import ocr_panel as op
    from exp17_rawmaps import map_indices, panel_features
    T = model['ocr_templates']
    rows = []
    for p in comp_paths:
        im = np.asarray(Image.open(p).convert('L')).astype(float)
        rows.append({k: op.to_float(op.read(T, op.glyphs(im, b))) for k, b in op.FIELDS.items()})
    o = pd.DataFrame(rows)
    meta = pd.DataFrame({'eye': eyes})
    return pd.concat([panel_features(meta, o), map_indices(meta, o, np.stack(lo_maps))], axis=1)


def build_external(df, model, tx, comp_dir, crop_dir):
    """Every E22 block for the external rows, from raw inputs only (mirrors exp22_bilateral.build)."""
    import exp20_bilateral as e20
    from prepare_fellow_tx import tx_features
    num = model['numerics']
    V = df[['pid', 'eye'] + num].copy()
    V['y'] = df.y.values
    V['date'] = pd.to_datetime(df.scan_date).dt.normalize()
    V['scored'] = df.y.notna().values
    V['n_missing'] = V[num].isna().sum(1)
    # fellow = the other eye's exam on the same date; prefer a scored exam, then the fewest missing numerics
    order = V.assign(r=np.arange(len(V))).sort_values(['scored', 'n_missing'], ascending=[False, True], kind='stable')
    first = order.drop_duplicates(['pid', 'eye', 'date'])
    lut = {(p, e, d): r for p, e, d, r in zip(first.pid, first.eye, first.date, first.r)}
    other = np.where(V.eye.values == 'right', 'left', 'right')
    V['fel'] = [lut.get((p, o, d), -1) for p, o, d in zip(V.pid, other, V.date)]
    D = {'V': V, 'N': e20.features(V)['full']}
    D['H'] = (tx_features(V.pid.values, V.eye.values, V.date.values, table=tx) if tx is not None
              else pd.DataFrame(np.nan, index=range(len(V)), columns=__import__('prepare_fellow_tx').H_COLS))
    own = np.full(len(V), -1)
    hrs, los, paths, stems, eyes = [], [], [], [], []
    if comp_dir is not None and 'composite' in df.columns:
        for i, c in enumerate(df.composite.values):
            if isinstance(c, str) and c and (Path(comp_dir) / c).exists():
                hr, lo = decode_composite(Path(comp_dir) / c, V.eye.iat[i])
                own[i] = len(hrs)
                hrs.append(hr), los.append(lo), paths.append(Path(comp_dir) / c), eyes.append(V.eye.iat[i])
                s = df.image_stem.iat[i] if 'image_stem' in df.columns else None
                stems.append(str(Path(crop_dir) / s) if (crop_dir and isinstance(s, str)) else None)
    clut = {}
    for i in np.flatnonzero(own >= 0):
        clut.setdefault((V.pid.iat[i], V.eye.iat[i], V.date.iat[i]), own[i])
    felc = np.array([clut.get((p, o, d), -1) for p, o, d in zip(V.pid, other, V.date)])
    D.update(own=own, felc=felc, hr=np.stack(hrs) if hrs else np.zeros((0, 4, 145, 145)), emb=embed(model, hrs, stems, eyes) if hrs else {})
    D['flags'] = pd.DataFrame({'has_fellow_image': (felc >= 0).astype(float), 'own_image': (own >= 0).astype(float)})
    if hrs and 'ocr_templates' in model:
        pa = panel(model, paths, [lo for lo in los], eyes)
        po = pa.reindex(np.where(own >= 0, own, -1)).reset_index(drop=True)
        pf = pa.reindex(np.where(felc >= 0, felc, -1)).reset_index(drop=True)
        D['PM'] = pd.concat([po.add_prefix('own_'), pf.add_prefix('fel_'), (po - pf).add_prefix('asym_')], axis=1)
    return D


def ftl_predict(D, idx):
    """E24 member ftL: encoder L fine-tuned on all 944 scans (frozen/ftL.pt), same inputs and eval transform as
    ft_L.py: own and same-day fellow composite (zeros + flag when none), age standardised with the training stats."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pretrain_jepa import Encoder
    ck = torch.load(ROOT / 'frozen' / 'ftL.pt', map_location='cpu', weights_only=False)
    enc = Encoder()
    enc.load_state_dict(ck['enc'])
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(384 * 3 + 2, 1))
    head.load_state_dict(ck['head'])
    enc.eval(), head.eval()
    hr = D['hr'].astype(np.float32)
    for j, (m, s) in enumerate(ck['norm']):
        hr[:, j] = (hr[:, j] - m) / s
    X = torch.from_numpy(np.concatenate([np.nan_to_num(hr), np.isfinite(hr)], 1).astype(np.float16)).float()

    def ev(rows):
        x = torch.stack([X[int(r)] if r >= 0 else torch.zeros_like(X[0]) for r in rows])
        x = F.interpolate(x, size=ck['cfg']['global_size'], mode='bilinear', align_corners=False)
        return torch.cat([x[:, :4] * (x[:, 4:] > 0.5), (x[:, 4:] > 0.5).float()], 1)
    age = torch.tensor((D['V'].age.values[idx] - ck['age_mu']) / ck['age_sd'], dtype=torch.float32)
    hf = torch.tensor(D['felc'][idx] >= 0, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            sl = slice(i, i + 64)
            zo, zf = enc(ev(D['own'][idx][sl])), enc(ev(D['felc'][idx][sl])) * hf[sl, None]
            out.append(torch.sigmoid(head(torch.cat([zo, zf, zo - zf, hf[sl, None], age[sl, None]], 1))).squeeze(1))
    return torch.cat(out).numpy().astype(float)


def predict(fz, D, idx):
    import exp22_bilateral as e22
    if fz.get('ftl'):
        return ftl_predict(D, idx)
    if 'stack' in fz:
        P = np.clip(np.column_stack([predict(m, D, idx) for m in fz['stack']]), 1e-5, 1 - 1e-5)
        Z = (np.log(P / (1 - P)) - fz['mu']) / fz['sd']
        return 1 / (1 + np.exp(-(Z @ fz['t'] + fz['b'])))
    if 'ens' in fz:
        return np.mean([predict(f, D, idx) for f in fz['ens']], 0)
    return e22.predict_frozen(fz, D, idx)


def report(name, p, y, g, refs, first):
    lo, hi = cluster_boot_auc(y, p, g)
    s = f'{name:12s} AUC {roc_auc_score(y, p):.3f} [{lo:.3f}-{hi:.3f}]'
    if 0 < y[first].sum() < first.sum():
        s += f'  first visit {roc_auc_score(y[first], p[first]):.3f}'
    for rn, rp in refs.items():
        if rn != name:
            d, (dl, dh), pr = paired_delta(y, p, rp, g)
            s += f' | vs {rn} {d:+.3f} [{dl:+.3f}, {dh:+.3f}]'
    print(s)


def score_bilateral(df, a):
    with open(ROOT / 'frozen' / 'bilateral.pkl', 'rb') as f:
        model = pickle.load(f)
    tx = None
    if a.tx:
        tx = pd.read_csv(a.tx) if a.tx.endswith('.csv') else pd.read_parquet(a.tx)
        tx['pid'] = tx.pid.astype(str)
        tx['eye'] = tx.eye.replace({'OD': 'right', 'OS': 'left'})
        for c in ('cxl', 'kp'):
            tx[f'{c}_date'] = pd.to_datetime(tx[f'{c}_date'], errors='coerce')
            tx[f'{c}_undated'] = tx[f'{c}_undated'].fillna(0).astype(bool) if f'{c}_undated' in tx else False
    D = build_external(df, model, tx, a.composites, a.images)
    idx = np.flatnonzero(df.y.notna().values)
    y, g = df.y.values[idx].astype(int), df.pid.values[idx]
    sc = df.iloc[idx]
    first = (sc.assign(_d=pd.to_datetime(sc.scan_date)).sort_values('_d').groupby(['pid', 'eye']).cumcount() == 0
             ).reindex(sc.index).values
    have_comp, have_crop = (D['own'][idx] >= 0).all(), D['own'][idx].min() >= 0 and 'image_stem' in df.columns
    p = {'age': model['age'].predict_proba(sc[['age']].values)[:, 1],
         'ref': model['ref']['model'].predict_proba(sc[model['numerics']].values)[:, 1]}
    for slot in ('headline', 'fallback_a', 'fallback_b'):
        nh, nc, ncr = needs(model[slot])
        miss = [w for w, n, h in [('--tx', nh, tx is not None), ('composites', nc, have_comp), ('crops', ncr, have_crop)]
                if n and not h]
        if miss:
            print(f'{slot}: not scored (input lacks {", ".join(miss)})')
            continue
        p[slot] = predict(model[slot], D, idx)
    has_f = D['V'].fel.values[idx] >= 0
    print(f'E22 bilateral: n={len(idx)} scans, {len(set(g))} patients, {y.sum()} positive; same-visit fellow exam for '
          f'{has_f.sum()}, fellow composite for {(D["felc"][idx] >= 0).sum()}')
    for slot in ('headline', 'fallback_a', 'fallback_b', 'ref'):
        if slot in model and isinstance(model[slot], dict):
            print(f'  {slot}: {model[slot].get("name")}')
    refs = {k: p[k] for k in ('ref', 'age')}
    for k, v in p.items():
        report(k, v, y, g, refs, first)
    for k, v in p.items():
        print(f'{k:12s} calibration-in-the-large: mean predicted {v.mean():.3f} vs observed {y.mean():.3f}, '
              f'Brier {np.mean((v - y) ** 2):.3f}')
    for nm, m in [('with same-visit fellow', has_f), ('without', ~has_f)]:
        if 0 < y[m].sum() < m.sum():
            print(f'{nm} (n={m.sum()}, pos={y[m].sum()}): ' +
                  ', '.join(f'{k} {roc_auc_score(y[m], v[m]):.3f}' for k, v in p.items()))


# ------------------------------------------------------------------------------------------------ legacy models
def image_grid(df, img_dir):
    import freeze_models as fm
    from extract_dino_grid import grid_features, load_encoder
    proc, model = load_encoder()
    return np.stack([grid_features([Path(img_dir) / f'{s}_{m}.jpg' for s in df.image_stem], proc, model)
                     .reshape(len(df), -1) for m in fm.MAPS], 1)


def score_legacy(df, a):
    import freeze_models as fm
    with open(ROOT / 'frozen' / 'models.pkl', 'rb') as f:
        models = pickle.load(f)
    df = df[df.y.notna()].reset_index(drop=True)
    y, g = df.y.values.astype(int), df.pid.values
    p = {'age': models['age'].predict_proba(df[['age']].values)[:, 1],
         'E02': fm.predict_E02(models, df), 'E06b': fm.predict_E06b(models, df), 'E12b': fm.predict_E12b(models, df)}
    if a.images and 'image_stem' in df.columns:
        G = image_grid(df, a.images)
        p['E10c'], p['E13'] = fm.predict_E10c(models, df, G), fm.predict_E13(models, df, G)
    print(f'legacy models: n={len(df)} scans, {df.pid.nunique()} patients, {y.sum()} positive')
    for k, v in p.items():
        lo, hi = cluster_boot_auc(y, v, g)
        print(f'{k:5s} AUC {roc_auc_score(y, v):.3f} [{lo:.3f}-{hi:.3f}]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--mapping')
    ap.add_argument('--tx')
    ap.add_argument('--composites')
    ap.add_argument('--images')
    a = ap.parse_args()
    df = pd.read_parquet(a.path) if a.path.endswith('.parquet') else pd.read_csv(a.path)
    if a.mapping:
        m = pd.read_csv(a.mapping, header=None, names=['ext', 'coh'])
        df = df.rename(columns=dict(zip(m.ext, m.coh)))
    from harness import numeric_features
    need = ['pid', 'eye', 'scan_date', 'y'] + numeric_features()
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f'missing columns: {missing}')
    df = df.reset_index(drop=True)
    df['pid'] = df.pid.astype(str)
    df['eye'] = df.eye.replace({'OD': 'right', 'OS': 'left'})
    df['scan_date'] = pd.to_datetime(df.scan_date)
    if (ROOT / 'frozen' / 'bilateral.pkl').exists():
        score_bilateral(df, a)
    if (ROOT / 'frozen' / 'models.pkl').exists():
        score_legacy(df, a)


if __name__ == '__main__':
    main()
