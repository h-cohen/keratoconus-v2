"""E27 step 1: position-aware SSL encoders G0 and G (pre-declared in PROTOCOL.md 2026-11-11). Labels are never read.

Motivation: L/X/T were trained with random resized crops (40-100 % of the map, anywhere), which makes the embedding
invariant to WHERE a feature sits; cone location and inferior-superior asymmetry are diagnostic, so position must
be kept. Both encoders = LeJEPA as L (2 global + 6 local views, SIGReg, lam 0.05, same backbone widths, 300 epochs,
last epoch kept), with three fixed changes:
  1. three coordinate channels (x nasal+, y superior+, radius; in units of the 5 mm map half-width) are warped
     together with the maps, so every pixel - also in a local crop - carries its absolute corneal position;
  2. global views: scale 0.85-1.0, aspect 1, rotation +-5 deg, shift +-0.3 mm (no flips); local views as L;
  3. input 11 channels (4 maps, 4 masks, 3 coordinates).
  G additionally has an auxiliary label-free head: linear 384 -> 96 predicting the standardised block R map
  measurements (radiomics.py) of the exam from each global view; loss += MU * masked MSE (MU = 0.1 fixed).
Corpus as L (composites + DCM page renderings; a DCM rendering's R target is that of its exam's composite).
Output: rethink/results/ssl/<enc>/ckpt.pt, curve.csv; rethink/data/emb_<enc>.npy (2260 x 384, last epoch).
Usage: python -I rethink/pretrain_geo.py G0|G [--smoke] [--gpu N]
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

sys.path.insert(0, str(Path(__file__).parent))
from pretrain_jepa import CFG as CFG_L, DATA, ROOT, Block, Augment, load_corpus, mlp, sigreg  # noqa: E402

CFG = {**CFG_L['common'], **CFG_L['L'], 'global_scale': (0.85, 1.0), 'global_rot': 5.0, 'global_shift_mm': 0.3,
       'half_mm': 5.0, 'mu': 0.1}
IN_CH = 11


class GeoEncoder(nn.Module):
    def __init__(self, w=(48, 96, 192, 384)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(IN_CH, w[0], 3, 2, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        layers, i = [], w[0]
        for k, o in enumerate(w):
            layers += [Block(i, o, 1 if k == 0 else 2), Block(o, o, 1)]
            i = o
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(self.stem(x)).mean((2, 3))


def coords(n, dev):
    t = torch.linspace(-1, 1, n, device=dev)
    yy, xx = torch.meshgrid(-t, t, indexing='ij')  # y up
    return torch.stack([xx, yy, torch.sqrt(xx ** 2 + yy ** 2)])  # (3, n, n)


def with_coords(x):
    """(B, 8, H, W) maps + masks -> (B, 11, H, W) with coordinate channels."""
    return torch.cat([x.float(), coords(x.shape[-1], x.device).expand(x.shape[0], -1, -1, -1)], 1)


class GeoAugment(Augment):
    def view(self, x, size, scale, glob):
        """Affine warp of maps + masks + coordinates; then noise / channel / patch dropout on the maps only."""
        c, B, dev = self.c, x.shape[0], self.dev
        x = with_coords(x)
        if glob:
            s = torch.empty(B, device=dev).uniform_(*scale).sqrt()
            sx = sy = s
            sh = c['global_shift_mm'] / c['half_mm']
            tx, ty = (torch.empty(B, device=dev).uniform_(-sh, sh) for _ in range(2))
            rot = c['global_rot']
        else:
            s = torch.empty(B, device=dev).uniform_(*scale)
            logr = torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3))
            sx, sy = torch.sqrt(s * torch.exp(logr)).clamp(max=1), torch.sqrt(s / torch.exp(logr)).clamp(max=1)
            tx = (1 - sx) * torch.empty(B, device=dev).uniform_(-1, 1)
            ty = (1 - sy) * torch.empty(B, device=dev).uniform_(-1, 1)
            rot = c['rot_deg']
        a = torch.empty(B, device=dev).uniform_(-rot, rot) * math.pi / 180
        th = torch.zeros(B, 2, 3, device=dev)
        th[:, 0, 0], th[:, 0, 1], th[:, 0, 2] = sx * torch.cos(a), -sy * torch.sin(a), tx
        th[:, 1, 0], th[:, 1, 1], th[:, 1, 2] = sx * torch.sin(a), sy * torch.cos(a), ty
        grid = F.affine_grid(th, (B, IN_CH, size, size), align_corners=False)
        v = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        mapsmask = super().view(v[:, :8], size, (1.0, 1.0), identity=True)
        return torch.cat([mapsmask, v[:, 8:]], 1)


def _identity_view(self, x, size, scale, identity=False):
    """Augment.view with the geometric warp disabled (identity=True): noise / channel / patch dropout only."""
    if not identity:
        return _orig_view(self, x, size, scale)
    c, B = self.c, x.shape[0]
    val, msk = x[:, :4], (x[:, 4:] > 0.5).float()
    val = (val + c['noise'] * torch.randn_like(val)) * msk
    keep = (torch.rand(B, 4, device=self.dev) > c['ch_drop']).float()
    keep[keep.sum(1) == 0, torch.randint(0, 4, (1,)).item()] = 1
    keep = keep * (msk.flatten(2).sum(-1) > 0).float() + (keep.sum(1, keepdim=True) == 0).float()
    val, msk = val * keep[:, :, None, None], msk * keep[:, :, None, None]
    if c['patch_drop'] > 0:
        for _ in range(2):
            ps = int(size * 0.2)
            cy = torch.randint(0, size - ps, (B,), device=self.dev)
            cx = torch.randint(0, size - ps, (B,), device=self.dev)
            yy = torch.arange(size, device=self.dev)[None, :, None]
            xx = torch.arange(size, device=self.dev)[None, None, :]
            box = ((yy >= cy[:, None, None]) & (yy < cy[:, None, None] + ps) & (xx >= cx[:, None, None])
                   & (xx < cx[:, None, None] + ps) & (torch.rand(B, 1, 1, device=self.dev) < c['patch_drop']))
            val, msk = val * (~box[:, None]), msk * (~box[:, None])
    return torch.cat([val, msk], 1)


_orig_view = Augment.view
Augment.view = _identity_view


def eval_input(x, size):
    """Unaugmented full map at the global view size, with coordinates (as used for the embeddings)."""
    x = F.interpolate(x.float(), size=size, mode='bilinear', align_corners=False)
    x = torch.cat([x[:, :4] * (x[:, 4:] > 0.5), (x[:, 4:] > 0.5).float()], 1)
    return with_coords(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('enc', choices=['G0', 'G'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    c = dict(CFG)
    if a.smoke:
        c.update(epochs=2, warmup=1)
    dev = torch.device(f'cuda:{a.gpu}')
    torch.manual_seed(c['seed'])
    rng = np.random.default_rng(c['seed'])
    out = ROOT / 'results' / 'ssl' / (a.enc + ('_smoke' if a.smoke else ''))
    out.mkdir(parents=True, exist_ok=True)
    X, owner, meta = load_corpus()
    n_hr = len(meta)
    Xg = X.to(dev)
    R = np.load(DATA / 'emb_R.npy').astype(np.float32)
    R = (R - np.nanmean(R, 0)) / (np.nanstd(R, 0) + 1e-6)
    Rm = torch.tensor(np.isfinite(R[owner]), device=dev, dtype=torch.float32)
    Rt = torch.tensor(np.nan_to_num(R[owner]), device=dev)
    units = np.arange(len(X))[:256] if a.smoke else np.arange(len(X))
    print(a.enc, 'training units', len(units), flush=True)
    enc = GeoEncoder().to(dev)
    proj = mlp(c['emb'], 1024, c['proj']).to(dev)
    aux = nn.Linear(c['emb'], R.shape[1]).to(dev)
    params = list(enc.parameters()) + list(proj.parameters()) + (list(aux.parameters()) if a.enc == 'G' else [])
    opt = torch.optim.AdamW(params, lr=c['lr'], weight_decay=c['wd'])
    scaler = torch.amp.GradScaler()
    steps_ep = max(1, len(units) // c['batch'])
    total = c['epochs'] * steps_ep
    sched = lambda s: min(1.0, (s + 1) / (c['warmup'] * steps_ep)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))  # noqa: E731
    aug = GeoAugment(c, dev)
    ep0, curve = 0, []
    ck = out / 'ckpt.pt'
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        enc.load_state_dict(s['enc']), proj.load_state_dict(s['proj']), aux.load_state_dict(s['aux'])
        opt.load_state_dict(s['opt']), scaler.load_state_dict(s['scaler'])
        ep0, curve, rng = s['epoch'], s['curve'], s['rng']
        torch.set_rng_state(s['torch_rng'].cpu()), torch.cuda.set_rng_state(s['cuda_rng'].cpu(), dev)
        print('resumed at epoch', ep0, flush=True)
    for ep in range(ep0, c['epochs']):
        enc.train(), proj.train(), aux.train()
        t0, perm = time.time(), rng.permutation(len(units))
        agg = np.zeros(4)
        for b in range(steps_ep):
            step = ep * steps_ep + b
            for gr in opt.param_groups:
                gr['lr'] = c['lr'] * sched(step)
            u = torch.as_tensor(units[perm[b * c['batch']:(b + 1) * c['batch']]], device=dev)
            x = Xg[u]
            with torch.autocast('cuda', dtype=torch.float16):
                gv = [aug.view(x, c['global_size'], c['global_scale'], True) for _ in range(c['n_global'])]
                lv = [aug.view(x, c['local_size'], c['local_scale'], False) for _ in range(c['n_local'])]
                hg = enc(torch.cat(gv))
                zg = proj(hg).float().view(c['n_global'], len(u), -1)
                zl = proj(enc(torch.cat(lv))).float().view(c['n_local'], len(u), -1)
                z = torch.cat([zg, zl])
                pred = (zg.mean(0) - z).square().mean()
                reg = torch.stack([sigreg(zv, step, c['slices']) for zv in z]).mean()
                loss = (1 - c['lam']) * pred + c['lam'] * reg
                if a.enc == 'G':
                    r = aux(hg).float().view(c['n_global'], len(u), -1)
                    m = Rm[u][None]
                    ax = ((r - Rt[u][None]) ** 2 * m).sum() / (m.sum() * c['n_global'] + 1e-6) * c['n_global']
                    loss = loss + c['mu'] * ax
                else:
                    ax = torch.zeros((), device=dev)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            agg += [loss.item(), pred.item(), reg.item(), ax.item()]
        agg /= steps_ep
        curve.append(dict(epoch=ep + 1, loss=agg[0], pred=agg[1], sigreg=agg[2], aux=agg[3], lr=c['lr'] * sched(step),
                          secs=round(time.time() - t0, 1)))
        print(curve[-1], flush=True)
        if (ep + 1) % 10 == 0 or ep + 1 == c['epochs']:
            torch.save(dict(enc=enc.state_dict(), proj=proj.state_dict(), aux=aux.state_dict(), opt=opt.state_dict(),
                            scaler=scaler.state_dict(), epoch=ep + 1, curve=curve, rng=rng,
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(dev), cfg=c), ck)
            pd.DataFrame(curve).to_csv(out / 'curve.csv', index=False)
    enc.eval()
    embs = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, n_hr, 256):
            embs.append(enc(eval_input(Xg[i:min(i + 256, n_hr)], c['global_size'])).float().cpu().numpy())
    E = np.concatenate(embs)
    sv = np.linalg.svd(E - E.mean(0), compute_uv=False)
    p = sv ** 2 / (sv ** 2).sum()
    erank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    print(a.enc, 'embedding', E.shape, 'effective rank', round(erank, 1), flush=True)
    pd.DataFrame([dict(enc=a.enc, effective_rank=erank)]).to_csv(out / 'embed_stats.csv', index=False)
    np.save((out / 'emb.npy') if a.smoke else (DATA / f'emb_{a.enc}.npy'), E)


if __name__ == '__main__':
    sys.exit(main())
