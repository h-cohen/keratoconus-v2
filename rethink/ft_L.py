"""E23 step 1: nested supervised fine-tuning of encoder L (PROTOCOL 2026-11-10 E23 item 3).

For every training/test split that E23 will need - the 25 harness outer folds (5-fold StratifiedGroupKFold by
patient, seeds 2026-2030) and, inside each, the 4 inner folds of exp16_nested.select (same seeds) - encoder L
(last pretraining epoch) + a linear head is trained on the TRAINING cohort scans only and predicts the test scans.
Fixed recipe (declared before any run): 20 epochs, AdamW lr 1e-4, wd 0.05, batch 32, pretraining global-view
augmentation, first-visit scans weight 2, last epoch kept, no early stopping, nothing chosen by AUC.
Inputs per scan: own composite and same-day fellow composite (both mirrored nasal-positive; zeros + flag when the
fellow has none), plus age. Head input = [z_own, z_fel, z_own - z_fel, has_fellow_image, age_std], dropout 0.3.
Predictions are cached in rethink/results/ftL/<key>.npy, key = seed + hash of train / test crows (as e16 cache).
Usage: python -I rethink/ft_L.py [--smoke]   (one worker process per GPU in GPUS)
"""
import hashlib
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from harness import DATA, N_REPEATS, N_SPLITS, RES, SEED  # noqa: E402

GPUS = [0, 1, 2, 3, 4, 5, 6]  # torch indices; the faulty card is not enumerated
CACHE = RES / 'ftL'
EPOCHS, LR, WD, BATCH, W_FIRST = 20, 1e-4, 0.05, 32, 2.0


def key(seed, tr, te):
    h = hashlib.sha1(np.sort(tr).tobytes() + b'|' + np.sort(te).tobytes()).hexdigest()[:16]
    return f'{seed}_{h}'


def splits(coh):
    from sklearn.model_selection import StratifiedGroupKFold
    y, g = coh.y.values, coh.pid.values
    out = []
    for r in range(N_REPEATS):
        seed = SEED + r
        for tr, te in StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(coh, y, g):
            out.append((seed, coh.crow.values[tr], coh.crow.values[te]))
            dtr = coh.iloc[tr].reset_index(drop=True)
            for a, b in StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(dtr, dtr.y.values, dtr.pid.values):
                out.append((seed, dtr.crow.values[a], dtr.crow.values[b]))
    return out


def worker(args):
    gpu, jobs = args
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import exp22_bilateral as e22
    from pretrain_jepa import CFG, Augment, Encoder, NORM
    dev = torch.device(f'cuda:{gpu}')
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    D = e22.build(coh)
    n = len(coh)
    own, felc = D['own'][:n], D['felc'][:n]
    hr = np.load(DATA / 'decoded_hr.npy').astype(np.float32)
    for j, (m, s) in enumerate(NORM):
        hr[:, j] = (hr[:, j] - m) / s
    valid = np.isfinite(hr)
    X = torch.from_numpy(np.concatenate([np.nan_to_num(hr), valid], 1).astype(np.float16)).to(dev)
    zero = torch.zeros_like(X[0])
    y_all = torch.tensor(coh.y.values, dtype=torch.float32, device=dev)
    w_all = torch.tensor(np.where(coh.first_visit.values, W_FIRST, 1.0), dtype=torch.float32, device=dev)
    age = coh.age.values.astype(np.float32)
    has_f = torch.tensor(felc >= 0, dtype=torch.float32, device=dev)
    ck = torch.load(RES / 'ssl' / 'L' / 'ckpt.pt', map_location='cpu', weights_only=False)
    enc_state = {k[4:]: v for k, v in ck['net'].items() if k.startswith('enc.')}
    c = {**CFG['common'], **CFG['L']}
    aug = Augment(c, dev)

    def batch_x(rows, train):
        o = X[torch.as_tensor(own[rows], device=dev)]
        f = torch.stack([X[int(i)] if i >= 0 else zero for i in felc[rows]])
        if train:
            return aug.view(o, c['global_size'], c['global_scale']), aug.view(f, c['global_size'], c['global_scale'])
        def ev(x):
            x = F.interpolate(x.float(), size=c['global_size'], mode='bilinear', align_corners=False)
            return torch.cat([x[:, :4] * (x[:, 4:] > 0.5), (x[:, 4:] > 0.5).float()], 1)
        return ev(o), ev(f)

    for seed, tr, te in jobs:
        out = CACHE / f'{key(seed, tr, te)}.npy'
        if out.exists():
            continue
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        enc = Encoder().to(dev)
        enc.load_state_dict(enc_state)
        head = nn.Sequential(nn.Dropout(0.3), nn.Linear(384 * 3 + 2, 1)).to(dev)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=LR, weight_decay=WD)
        mu, sd = age[tr].mean(), age[tr].std()
        ages = torch.tensor((age - mu) / sd, device=dev)

        def logits(rows, train):
            o, f = batch_x(rows, train)
            zo, zf = enc(o), enc(f) * has_f[rows][:, None]
            return head(torch.cat([zo, zf, zo - zf, has_f[rows][:, None], ages[rows][:, None]], 1)).squeeze(1)

        for ep in range(EPOCHS):
            enc.train(), head.train()
            perm = rng.permutation(tr)
            for b in range(0, len(perm) - 1, BATCH):
                rows = perm[b:b + BATCH]
                if len(rows) < 2:
                    continue
                with torch.autocast('cuda', dtype=torch.float16):
                    lg = logits(rows, True).float()
                loss = (F.binary_cross_entropy_with_logits(lg, y_all[rows], reduction='none') * w_all[rows]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        enc.eval(), head.eval()
        if len(te) == 0:  # final fit on all scans (freeze_stack.py): keep the weights, nothing to predict
            torch.save(dict(enc=enc.state_dict(), head=head.state_dict(), age_mu=float(mu), age_sd=float(sd),
                            cfg=c, norm=NORM), out.with_suffix('.pt'))
            np.save(out, np.zeros(0))
            print(gpu, 'final fit saved', flush=True)
            continue
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            p = torch.cat([torch.sigmoid(logits(te[i:i + 128], False).float()) for i in range(0, len(te), 128)])
        np.save(out, p.cpu().numpy())
        print(gpu, out.name, flush=True)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    jobs = splits(coh)
    if '--smoke' in sys.argv:
        global EPOCHS
        jobs = jobs[:1]
        worker((GPUS[0], [(s, a, b) for s, a, b in jobs]))
        return
    chunks = [(g, jobs[i::len(GPUS)]) for i, g in enumerate(GPUS)]
    with get_context('spawn').Pool(len(GPUS)) as p:
        p.map(worker, chunks)
    print('done', len(jobs), 'splits', flush=True)


if __name__ == '__main__':
    sys.exit(main())
