"""E22 step 5: self-supervised encoders L, X, T on ALL decoded maps (design section 5, U5). Labels are never read.

Corpus: decoded_hr.npy (2,260 composites, 4 maps, 145 x 145, left eyes mirrored) + decoded_dcm.npy (3,180 DCM page
samples of the same exams, partial channels). Input = 4 maps normalised with exp19_ssl.NORM + 4 validity masks.

Encoders (one fixed configuration each, CFG below, written before any training; last-epoch checkpoint is used):
  L  LeJEPA: 2 global + 6 local views of one sample; every view's projection predicts the mean projection of the
     global views; loss = (1 - lam) * prediction + lam * SIGReg (Epps-Pulley, 256 random slices, 17 knots).
  X  cross-eye JEPA: context = one eye's maps at a visit, target = the same-day fellow eye (both already mirrored to
     nasal-positive); predictor MLP maps the context projection to the fellow's; symmetric; + lam * SIGReg.
     Pairs: the 786 same-day bilateral composite visits.
  T  temporal JEPA: context = an eye at visit t, target = the same eye at a later visit; the predictor also gets
     the gap (years); + lam * SIGReg. Pairs: all ordered visit pairs of all eyes with a strictly later date.
  For X and T each exam is shown as its composite (p = 0.5) or one of its DCM page renderings.
Backbone: ResNet-18 layout (2-2-2-2 basic blocks, widths 48/96/192/384, 3x3 stride-2 stem, no max-pool), global
average pool -> 384-d embedding (the design's cap). Projector 384-1024-1024-128 (BN, ReLU).
Augmentations (physically meaningful only): random resized crop, rotation +-10 deg, Gaussian noise on decoded
values, channel dropout, patch dropout. No flips, no colour jitter.
Output: rethink/results/ssl/<enc>/ckpt.pt (resumable, every 10 epochs), curve.csv (label-free diagnostics), and
rethink/data/emb_<enc>.npy (2,260 x 384, row-aligned with decoded_hr_meta) from the last epoch.
Usage: python -I rethink/pretrain_jepa.py L|X|T [--smoke] [--gpu N]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
NORM = [(550., 60.), (45., 4.), (0., 15.), (0., 15.)]  # exp19_ssl.NORM: thickness um, curvature D, elevations um

CFG = {  # FIXED a priori (LeJEPA paper defaults where they exist). Never tuned on referral.
    'common': dict(epochs=300, warmup=10, wd=0.05, lam=0.05, slices=256, global_size=128, local_size=64,
                   global_scale=(0.4, 1.0), local_scale=(0.1, 0.35), rot_deg=10, noise=0.05, ch_drop=0.15,
                   patch_drop=0.5, emb=384, proj=128, seed=2026),
    'L': dict(batch=128, lr=1e-3, n_global=2, n_local=6),
    'X': dict(batch=64, lr=5e-4, n_global=2, n_local=0),
    'T': dict(batch=64, lr=5e-4, n_global=2, n_local=0),
}


# ----------------------------------------------------------------------------------------------------------- data
def load_corpus():
    hr = np.load(DATA / 'decoded_hr.npy')
    dc = np.load(DATA / 'decoded_dcm.npy')
    meta = pd.read_parquet(DATA / 'decoded_hr_meta.parquet')
    dmeta = pd.read_parquet(DATA / 'decoded_dcm_meta.parquet')
    x = np.concatenate([hr, dc]).astype(np.float32)
    for j, (m, s) in enumerate(NORM):
        x[:, j] = (x[:, j] - m) / s
    valid = np.isfinite(x)
    x = np.concatenate([np.nan_to_num(x), valid], 1).astype(np.float16)  # (N, 8, 145, 145)
    owner = np.concatenate([np.arange(len(hr)), dmeta.hr_row.values])  # composite row of the exam
    return torch.from_numpy(x), owner, meta


def renderings(owner, n_hr):
    """For each composite row: indices of its DCM page renderings in the corpus."""
    r = {i: [] for i in range(n_hr)}
    for k in range(n_hr, len(owner)):
        r[int(owner[k])].append(k)
    return r


def pairs_X(meta):
    m = meta.reset_index().rename(columns={'index': 'row'})
    rr = m[m.eye == 'right'].drop_duplicates(['pkey', 'date'])
    ll = m[m.eye == 'left'].drop_duplicates(['pkey', 'date'])
    j = rr.merge(ll, on=['pkey', 'date'], suffixes=('_r', '_l'))
    return np.stack([j.row_r.values, j.row_l.values], 1)


def pairs_T(meta):
    m = meta.reset_index().rename(columns={'index': 'row'})
    out = []
    for _, g in m.groupby(['pkey', 'eye']):
        g = g.sort_values('date')
        for a in range(len(g)):
            for b in range(len(g)):
                gap = (g.date.iat[b] - g.date.iat[a]).days / 365.25
                if gap > 0:
                    out.append((g.row.iat[a], g.row.iat[b], gap))
    return np.array(out)


class Augment:
    def __init__(self, c, dev):
        self.c, self.dev = c, dev

    def view(self, x, size, scale):
        """Random resized crop + rotation in one affine warp, then noise / channel / patch dropout. x: (B, 8, H, W)."""
        c, B = self.c, x.shape[0]
        s = torch.empty(B, device=self.dev).uniform_(*scale)
        logr = torch.empty(B, device=self.dev).uniform_(math.log(3 / 4), math.log(4 / 3))
        sx, sy = torch.sqrt(s * torch.exp(logr)).clamp(max=1), torch.sqrt(s / torch.exp(logr)).clamp(max=1)
        tx, ty = (1 - sx) * torch.empty(B, device=self.dev).uniform_(-1, 1), (1 - sy) * torch.empty(B, device=self.dev).uniform_(-1, 1)
        a = torch.empty(B, device=self.dev).uniform_(-c['rot_deg'], c['rot_deg']) * math.pi / 180
        th = torch.zeros(B, 2, 3, device=self.dev)
        th[:, 0, 0], th[:, 0, 1], th[:, 0, 2] = sx * torch.cos(a), -sy * torch.sin(a), tx
        th[:, 1, 0], th[:, 1, 1], th[:, 1, 2] = sx * torch.sin(a), sy * torch.cos(a), ty
        grid = F.affine_grid(th, (B, 8, size, size), align_corners=False)
        v = F.grid_sample(x.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        val, msk = v[:, :4], (v[:, 4:] > 0.5).float()
        val = (val + c['noise'] * torch.randn_like(val)) * msk
        keep = (torch.rand(B, 4, device=self.dev) > c['ch_drop']).float()
        keep[keep.sum(1) == 0, torch.randint(0, 4, (1,)).item()] = 1  # keep at least one map
        keep = keep * (msk.flatten(2).sum(-1) > 0).float() + (keep.sum(1, keepdim=True) == 0).float()
        val, msk = val * keep[:, :, None, None], msk * keep[:, :, None, None]
        if c['patch_drop'] > 0:
            for _ in range(2):
                ps = int(size * 0.2)
                cy, cx = torch.randint(0, size - ps, (B,), device=self.dev), torch.randint(0, size - ps, (B,), device=self.dev)
                yy = torch.arange(size, device=self.dev)[None, :, None]
                xx = torch.arange(size, device=self.dev)[None, None, :]
                box = ((yy >= cy[:, None, None]) & (yy < cy[:, None, None] + ps) & (xx >= cx[:, None, None])
                       & (xx < cx[:, None, None] + ps) & (torch.rand(B, 1, 1, device=self.dev) < c['patch_drop']))
                val, msk = val * (~box[:, None]), msk * (~box[:, None])
        return torch.cat([val, msk], 1)


# ---------------------------------------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o)
        self.c2, self.b2 = nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o)
        self.sc = nn.Sequential() if s == 1 and i == o else nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))


class Encoder(nn.Module):
    def __init__(self, w=(48, 96, 192, 384)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(8, w[0], 3, 2, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        layers, i = [], w[0]
        for k, o in enumerate(w):
            layers += [Block(i, o, 1 if k == 0 else 2), Block(o, o, 1)]
            i = o
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(self.stem(x)).mean((2, 3))


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
                         nn.Linear(h, o))


class Net(nn.Module):
    def __init__(self, c, enc):
        super().__init__()
        self.enc, self.proj = Encoder(), mlp(c['emb'], 1024, c['proj'])
        self.pred = None if enc == 'L' else mlp(c['proj'] + (1 if enc == 'T' else 0), 512, c['proj'])

    def forward(self, x):
        return self.proj(self.enc(x))


def sigreg(z, step, slices):
    """Epps-Pulley test of z (N, K) projections against N(0, 1), as in LeJEPA (minimal reference code)."""
    z = z.float()
    g = torch.Generator(device=z.device)
    g.manual_seed(int(step))
    A = torch.randn(z.shape[1], slices, generator=g, device=z.device)
    A = A / A.norm(dim=0)
    t = torch.linspace(-5, 5, 17, device=z.device)
    w = torch.exp(-0.5 * t ** 2)
    xt = (z @ A).unsqueeze(2) * t
    err = (torch.cos(xt).mean(0) - w) ** 2 + torch.sin(xt).mean(0) ** 2
    return (torch.trapezoid(err * w, t, dim=1) * z.shape[0]).mean()


# ---------------------------------------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('enc', choices=['L', 'X', 'T'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    c = {**CFG['common'], **CFG[a.enc]}
    if a.smoke:
        c.update(epochs=2, warmup=1)
    dev = torch.device(f'cuda:{a.gpu}')
    torch.manual_seed(c['seed'])
    np.random.seed(c['seed'])
    rng = np.random.default_rng(c['seed'])
    out = ROOT / 'results' / 'ssl' / (a.enc + ('_smoke' if a.smoke else ''))
    out.mkdir(parents=True, exist_ok=True)
    X, owner, meta = load_corpus()
    n_hr = len(meta)
    Xg = X.to(dev)
    rend = renderings(owner, n_hr)
    if a.enc == 'L':
        units = np.arange(len(X))
    elif a.enc == 'X':
        units = pairs_X(meta)
    else:
        units = pairs_T(meta)
    if a.smoke:
        units = units[:256]
    print(a.enc, 'training units', len(units), 'corpus', tuple(X.shape), flush=True)
    net = Net(c, a.enc).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=c['wd'])
    scaler = torch.amp.GradScaler()
    steps_ep = max(1, len(units) // c['batch'])
    total = c['epochs'] * steps_ep
    sched = lambda s: min(1.0, (s + 1) / (c['warmup'] * steps_ep)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))  # noqa: E731
    aug = Augment(c, dev)
    ep0, curve = 0, []
    ck = out / 'ckpt.pt'
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(s['net']), opt.load_state_dict(s['opt']), scaler.load_state_dict(s['scaler'])
        ep0, curve, rng = s['epoch'], s['curve'], s['rng']
        torch.set_rng_state(s['torch_rng'].cpu()), torch.cuda.set_rng_state(s['cuda_rng'].cpu(), dev)
        print('resumed at epoch', ep0, flush=True)

    def pick(rows):  # one rendering per exam: composite (p = 0.5) or one of its DCM pages
        out_ = []
        for r in rows:
            r = int(r)
            alt = rend[r]
            out_.append(r if (not alt or rng.random() < 0.5) else alt[rng.integers(len(alt))])
        return torch.as_tensor(out_, device=dev)

    for ep in range(ep0, c['epochs']):
        net.train()
        t0, perm = time.time(), rng.permutation(len(units))
        agg = np.zeros(3)
        for b in range(steps_ep):
            step = ep * steps_ep + b
            for gr in opt.param_groups:
                gr['lr'] = c['lr'] * sched(step)
            u = units[perm[b * c['batch']:(b + 1) * c['batch']]]
            with torch.autocast('cuda', dtype=torch.float16):
                if a.enc == 'L':
                    x = Xg[torch.as_tensor(u, device=dev)]
                    views = [aug.view(x, c['global_size'], c['global_scale']) for _ in range(c['n_global'])]
                    views += [aug.view(x, c['local_size'], c['local_scale']) for _ in range(c['n_local'])]
                    zg = net(torch.cat(views[:c['n_global']])).float().view(c['n_global'], len(u), -1)
                    zl = net(torch.cat(views[c['n_global']:])).float().view(c['n_local'], len(u), -1)
                    z = torch.cat([zg, zl])
                    pred = (zg.mean(0) - z).square().mean()
                    reg = torch.stack([sigreg(zv, step, c['slices']) for zv in z]).mean()
                else:
                    ia, ib = pick(u[:, 0]), pick(u[:, 1])
                    za = torch.stack([net(aug.view(Xg[ia], c['global_size'], c['global_scale'])).float()
                                      for _ in range(c['n_global'])])
                    zb = torch.stack([net(aug.view(Xg[ib], c['global_size'], c['global_scale'])).float()
                                      for _ in range(c['n_global'])])
                    if a.enc == 'X':  # symmetric cross-eye prediction
                        pred = 0.5 * ((net.pred(za.mean(0)).float() - zb.mean(0)).square().mean()
                                      + (net.pred(zb.mean(0)).float() - za.mean(0)).square().mean())
                    else:  # forward in time, gap as predictor input
                        gap = torch.as_tensor(u[:, 2], device=dev, dtype=torch.float32)[:, None] / 3.0
                        pred = (net.pred(torch.cat([za.mean(0), gap], 1)).float() - zb.mean(0)).square().mean()
                    reg = torch.stack([sigreg(zv, step, c['slices']) for zv in torch.cat([za, zb])]).mean()
                loss = (1 - c['lam']) * pred + c['lam'] * reg
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            agg += [loss.item(), pred.item(), reg.item()]
        agg /= steps_ep
        curve.append(dict(epoch=ep + 1, loss=agg[0], pred=agg[1], sigreg=agg[2], lr=c['lr'] * sched(step),
                          secs=round(time.time() - t0, 1)))
        print(curve[-1], flush=True)
        if (ep + 1) % 10 == 0 or ep + 1 == c['epochs']:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=ep + 1,
                            curve=curve, rng=rng, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(dev),
                            cfg=c), ck)
            pd.DataFrame(curve).to_csv(out / 'curve.csv', index=False)
    # embed every composite with the LAST-epoch encoder: full map resized to the global view size, no augmentation
    net.eval()
    embs = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, n_hr, 256):
            x = F.interpolate(Xg[i:min(i + 256, n_hr)].float(), size=c['global_size'], mode='bilinear', align_corners=False)
            x = torch.cat([x[:, :4] * (x[:, 4:] > 0.5), (x[:, 4:] > 0.5).float()], 1)
            embs.append(net.enc(x).float().cpu().numpy())
    E = np.concatenate(embs)
    sv = np.linalg.svd(E - E.mean(0), compute_uv=False)
    p = sv ** 2 / (sv ** 2).sum()
    erank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    print(a.enc, 'embedding', E.shape, 'effective rank', round(erank, 1), 'mean feature std', float(E.std(0).mean()), flush=True)
    pd.DataFrame([dict(enc=a.enc, effective_rank=erank, feat_std=float(E.std(0).mean()))]).to_csv(out / 'embed_stats.csv', index=False)
    np.save((out / 'emb.npy') if a.smoke else (DATA / f'emb_{a.enc}.npy'), E)


if __name__ == '__main__':
    sys.exit(main())
