"""E19: self-supervised image representation (masked map autoencoder), single eye, single visit. Pre-declared.

Pretraining corpus: every decoded '4 Maps Selectable' composite in data/image_scans (decode_all.py; 2,259
exams, physical thickness / curvature / front and back elevation, left eyes mirrored). No labels.
Per outer fold the encoder is pretrained only on images of (a) patients outside the cohort and (b) cohort
patients in the outer training rows; test-fold patients' images are never seen. Inner-CV calls of the
nested run reuse their outer fold's encoder (its pretraining set excludes the outer test fold).
Pretext: inputs are the 4 normalised maps + 4 visibility channels; 50% of 8x8 patches are hidden
independently per map, so the net must predict one map from the others and from the neighbourhood; loss =
MSE on hidden valid pixels + 0.2 x MSE on visible valid pixels. Augmentation: shifts of up to 3 px.
Encoder: 4 stride-2 conv blocks -> global average -> 64-d embedding; decoder mirrors it. 40 epochs, AdamW.
Candidates (LR, L2, C by patient-grouped inner CV):
  ssl_lr      : [age | 64-d SSL embedding]
  ssl_num_lr  : [SSL embedding | 55 numerics]
Runs: E19a_ssl_lr, E19b_ssl_num_lr, E19c_nested_single_eye_ssl (E16 pool A + both candidates).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import exp16_nested as e16  # noqa: E402
from compare import paired_delta  # noqa: E402
from harness import DATA, RES, evaluate  # noqa: E402

torch.set_num_threads(4)
NORM = [(550., 60.), (45., 4.), (0., 15.), (0., 15.)]  # thickness um, curvature D, elevations um
EPOCHS, BATCH, EMB = 40, 64, 64
_V = None
_META = None
_ENC = {}
CURRENT = {'pids': None}


def _data():
    global _V, _META
    if _V is None:
        v = np.load(DATA / 'decoded_all.npy')
        for j, (m, s) in enumerate(NORM):
            v[:, j] = (v[:, j] - m) / s
        valid = np.isfinite(v)
        _V = np.concatenate([np.nan_to_num(v), valid], 1).astype(np.float32)  # (N, 8, 58, 58)
        _META = pd.read_parquet(DATA / 'decoded_all_meta.parquet')
    return _V, _META


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.GELU(), nn.Conv2d(o, o, 3, 1, 1),
                         nn.BatchNorm2d(o), nn.GELU())


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(block(8, 32), block(32, 64), block(64, 128), block(128, 128))  # 64 -> 4
        self.proj = nn.Linear(128, EMB)
        self.up = nn.Linear(EMB, 128 * 16)
        self.dec = nn.Sequential(*[nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GELU())
                                   for i, o in [(256, 128), (128, 64), (64, 32)]], nn.ConvTranspose2d(32, 4, 4, 2, 1))

    def embed(self, x):
        h = self.enc(nn.functional.pad(x, (3, 3, 3, 3)))
        return self.proj(h.mean((2, 3))), h

    def forward(self, x):
        z, h = self.embed(x)
        u = self.up(z).view(-1, 128, 4, 4)
        return self.dec(torch.cat([h, u], 1))[:, :, 3:61, 3:61]


def _mask(n, rng):
    m = rng.random((n, 4, 8, 8)) < .5
    m = np.kron(m, np.ones((1, 1, 8, 8), bool))[:, :, :58, :58]
    return torch.from_numpy(m)


def pretrain(rows, seed):
    V, _ = _data()
    X = torch.from_numpy(V[rows])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = MAE()
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * (len(rows) // BATCH + 1))
    net.train()
    for _ in range(EPOCHS):
        for b in torch.randperm(len(rows)).split(BATCH):
            x = X[b]
            dx, dy = rng.integers(-3, 4, 2)
            x = torch.roll(x, (int(dx), int(dy)), (2, 3))
            val, tgt = x[:, 4:] > .5, x[:, :4]
            hid = _mask(len(b), rng) & val
            vis = val & ~hid
            inp = torch.cat([tgt * vis, vis.float()], 1)
            err = (net(inp) - tgt) ** 2
            loss = err[hid].mean() + .2 * err[vis].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return net, float(loss.detach())


def encoder(seed):
    """Encoder for the current outer training set: pretrained on non-cohort + outer-training patients."""
    pids = CURRENT['pids']
    key = (seed, hash(tuple(sorted(pids))))
    if key not in _ENC:
        _, meta = _data()
        cohort_pids = set(pd.read_parquet(DATA / 'cohort.parquet', columns=['pid']).pid)
        allowed = ~meta.pkey.isin(cohort_pids) | meta.pkey.isin(pids)
        rows = np.flatnonzero(allowed.values & meta.ok.values)
        net, loss = pretrain(rows, seed)
        V, _ = _data()
        with torch.no_grad():
            Z = torch.cat([net.embed(torch.from_numpy(V[i:i + 256]))[0] for i in range(0, len(V), 256)]).numpy()
        cmap = dict(zip(meta.cohort_row[meta.cohort_row >= 0], np.flatnonzero(meta.cohort_row >= 0)))
        _ENC[key] = Z[[cmap[i] for i in range(len(cmap))]]  # cohort order
        print(f'  pretrained on {len(rows)} images (seed {seed}), final loss {loss:.3f}', flush=True)
    return _ENC[key]


def ssl_lr(dtr, dte, seed):
    Z = encoder(seed)
    return e16.lr_on(np.hstack([dtr[['age']].values, Z[dtr.crow.values]]),
                     np.hstack([dte[['age']].values, Z[dte.crow.values]]), dtr, seed)


def ssl_num_lr(dtr, dte, seed):
    Z = encoder(seed)
    return e16.lr_on(np.hstack([dtr[e16.NUM].values, Z[dtr.crow.values]]),
                     np.hstack([dte[e16.NUM].values, Z[dte.crow.values]]), dtr, seed)


NEW = {'ssl_lr': ssl_lr, 'ssl_num_lr': ssl_num_lr}


def outer(fp):
    def f(dtr, ctr, dte, cte, seed):
        CURRENT['pids'] = set(dtr.pid)
        return fp(dtr, ctr, dte, cte, seed)
    return f


if __name__ == '__main__':
    coh = pd.read_parquet(DATA / 'cohort.parquet')
    coh['crow'] = np.arange(len(coh))
    cnn = np.load(DATA / 'cnn_frozen.npy')
    e16.CANDS.update(NEW)
    oofs = {}
    for run, k in [('E19a_ssl_lr', 'ssl_lr'), ('E19b_ssl_num_lr', 'ssl_num_lr')]:
        _, oofs[run] = evaluate(run, outer(lambda dtr, ctr, dte, cte, seed, k=k: e16.predict(k, dtr, dte, seed)),
                                f'single eye: self-supervised masked-map embedding ({k})', df=coh, cnn=cnn)
    _, oofs['E19c'] = evaluate('E19c_nested_single_eye_ssl', outer(e16.select(e16.POOL_A + list(NEW), 'A+ssl')),
                               'nested selection: E16 pool A + SSL candidates, single eye, all 944', df=coh, cnn=cnn)
    ch = pd.DataFrame(e16.CHOICES)
    ch.to_csv(RES / 'nested19_choices.csv', index=False)
    print(ch.chosen.value_counts().to_string())
    y, g = coh.y.values, coh.pid.values
    age = np.load(RES / 'oof' / 'E07_age_all.npy').mean(0)
    for n, o in oofs.items():
        d, (lo, hi), pr = paired_delta(y, o.mean(0), age, g)
        print(f'{n} vs age: dAUC {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P={pr:.2f}')
