"""Is the model using what it is shown?  python -m wm.diag --ckpt RUN/model.pt --data VAL_DIR

Prints the loss by position in the window (position 0 has no context; later ones should be much easier), at a few
noise levels, and the same with every episode's context swapped for another episode's (if that costs nothing, the
context is being ignored)."""
import argparse

import numpy as np
import torch

from wm import data as D
from wm import roll

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=16)
    a = p.parse_args()
    dev = "cuda"
    m, mean, std = roll.load(a.ckpt, dev)
    F = m.cfg.frames
    lat, act, lvl = D.Chunks(a.data, F).fixed(a.n, 1)[0]
    lat = (torch.from_numpy(lat).to(dev).float() - mean[:, None]) / std[:, None]
    act, lvl = torch.from_numpy(act).to(dev), torch.from_numpy(lvl.astype(np.int64)).to(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    eps = torch.randn(lat.shape, device=dev, generator=g)
    eps_c = torch.randn(lat.shape, device=dev, generator=g)
    tau = torch.full(lat.shape[:2], 0.05, device=dev)
    print("data: variance of a latent around its episode's mean frame", round(float((lat - lat.mean(1, keepdim=True)).var()), 3),
          " mean squared change between consecutive frames", round(float(((lat[:, 1:] - lat[:, :-1]) ** 2).mean()), 3))
    for tv in (0.2, 0.5, 0.8, 0.98):
        t = torch.full(lat.shape[:2], tv, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            l = m(lat, act, lvl, t, tau, eps, eps_c).float().mean(0)
            # other episodes' frames as context: the context copies come from a rolled batch, targets stay
            B = lat.shape[0]
            x = lat.roll(1, 0)
            tt, tc = t[:, :, None, None, None], tau[:, :, None, None, None]
            xin = torch.cat(((1 - tc) * x + tc * eps_c, (1 - tt) * lat + tt * eps), 1).to(m.pos.dtype)
            cond = torch.cat((m._cond(tau, act, lvl, 0), m._cond(t, act, lvl, 1)), 1)
            h = m.inp(m._tok(xin)) + m.pos
            idx = torch.arange(F, device=dev)
            low, strict, eye = idx[:, None] >= idx[None], idx[:, None] > idx[None], idx[:, None] == idx[None]
            mask = torch.cat((torch.cat((low, torch.zeros_like(low)), 1), torch.cat((strict, eye), 1)), 0)
            for blk in m.blocks:
                h = blk(h, cond, torch.cat((idx, idx)), mask)
            v = m._head(h[:, F:], cond[:, F:])
            l2 = ((v.float() - (eps - lat)) ** 2).mean((2, 3, 4)).mean(0)
        f = lambda z: " ".join(f"{float(z[i]):.3f}" for i in (0, 1, 2, 4, 8, 16, 31) if i < F)
        print(f"t={tv}: loss at positions 0 1 2 4 8 16 31: {f(l)}   | with a wrong context: {f(l2)}")
