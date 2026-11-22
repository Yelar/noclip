"""Does the model use its memory of places?   python tools/mem_gain.py CKPT VAL_DIR [WINDOWS]

The loss on validation windows (recorded frames as context, the same noise each time) three ways: with the window's
memory frames, with none, and with another window's memory frames (the wrong place). Only windows that have memory
frames count. A model that uses the memory has a lower loss with it than without, most of all on the window's first
frames (which have little context of their own) and on the noisiest frames to draw; a wrong memory should not help.

Then the same for the moment the memory is for: windows whose last frame has turned to a view that the window's
other frames do not show in full but an earlier frame of the episode does (the player looks back at something).
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm import data as D  # noqa: E402
from wm import memory as MEM  # noqa: E402
from wm import roll  # noqa: E402

ckpt, val = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 192
dev, B = "cuda", 4
m, mean, std = roll.load(ckpt, dev, dtype=torch.float32)
c = m.cfg
if not c.mem:
    raise SystemExit("this model has no memory")
ch = D.Chunks(val, c.frames, c.mem, c.A, 0.0)
g = torch.Generator(device=dev).manual_seed(7)
f = lambda x: (torch.from_numpy(x).to(dev).float() - mean[:, None]) / std[:, None]
ints = lambda x: torch.from_numpy(x.astype(np.int64)).to(dev)
rnd = lambda *sh: torch.randn(sh, device=dev, generator=g)
with_, without, wrong, ts, slots = [], [], [], [], []
with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
    for b in ch.fixed(B, n // B, seed=99):
        lat, act, lvl = f(b[0]), torch.from_numpy(b[1]).to(dev), ints(b[2])
        mlat, mlvl, mok = f(b[3]), ints(b[4]), torch.from_numpy(b[5]).to(dev)
        Fr = lat.shape[1]
        t = torch.rand(B, Fr, device=dev, generator=g).clamp_min(0.02)
        tau, mtau = torch.full((B, Fr), 0.05, device=dev), torch.full(mok.shape, 0.05, device=dev)
        eps, eps_c = rnd(*lat.shape), rnd(*lat.shape)
        mlat = 0.95 * mlat + 0.05 * rnd(*mlat.shape)
        run = lambda ml, mv, ok: m(lat, act, lvl, t, tau, eps, eps_c, mem=(ml, mv, ok, mtau)).float().cpu()
        la, lb = run(mlat, mlvl, mok), run(mlat, mlvl, torch.zeros_like(mok))
        lc = run(mlat.roll(1, 0), mlvl.roll(1, 0), mok.roll(1, 0))
        has = (mok.any(1) & mok.roll(1, 0).any(1)).cpu()
        with_.append(la[has]); without.append(lb[has]); wrong.append(lc[has]); ts.append(t.cpu()[has])
        slots.append(mok.sum(1).cpu())
with_, without, wrong, ts, slots = (torch.cat(x) for x in (with_, without, wrong, ts, slots))
print(f"{ckpt}: {len(slots)} windows, {float((slots > 0).float().mean()):.0%} with memory frames "
      f"({float(slots[slots > 0].float().mean()):.1f} of {c.mem} slots on average), {len(with_)} compared")
Fr = with_.shape[1]
idx = torch.arange(Fr)[None].expand_as(ts)
for name, sel in (("all frames", idx >= 0), ("the window's first 4 frames", idx < 4), ("the other frames", idx >= 4),
                  ("first 4 frames, drawn from much noise (t > 0.6)", (idx < 4) & (ts > 0.6)),
                  ("other frames, drawn from much noise (t > 0.6)", (idx >= 4) & (ts > 0.6))):
    a, b_, w = float(with_[sel].mean()), float(without[sel].mean()), float(wrong[sel].mean())
    print(f"  {name:48s} loss with memory {a:.4f}, without {b_:.4f} ({100 * (a / b_ - 1):+.1f}%), with a wrong memory {w:.4f} ({100 * (w / b_ - 1):+.1f}%)")


# ---- the moment of looking back: the window's last frame has turned at least TURN radians beyond every other frame of
# the window (so a strip of it is new to the window), and an earlier frame of the episode was taken from nearly the
# same view (score below NEAR, wm/memory.py). This is the case the memory is for.
TURN, NEAR = 0.15, 1.2


def returns(n, rng):
    out, tries, Fr = [], 0, c.frames
    while len(out) < n and tries < 60 * n:
        tries += 1
        lat, act, lvl, aux = ch.files[int(rng.integers(len(ch.files)))]
        if aux is None or lat.shape[1] < Fr + 2 * MEM.GAP:
            continue
        b = int(rng.integers(lat.shape[0]))
        a = np.asarray(aux[b], np.float32)
        for i in rng.permutation(np.arange(Fr + MEM.GAP, a.shape[0]))[:16]:
            s = int(i) - Fr + 1
            if a[s:i + 1, D.AUX_TRANS].any() or (a[s:i + 1, D.AUX_EP] != a[i, D.AUX_EP]).any():
                continue
            sc = MEM.score(a[:i, D.AUX_X], a[:i, D.AUX_Z], a[:i, D.AUX_YAW], a[:i, D.AUX_PITCH],
                           a[i, D.AUX_X], a[i, D.AUX_Z], a[i, D.AUX_YAW], a[i, D.AUX_PITCH])
            pre = np.where((a[:s, D.AUX_EP] == a[i, D.AUX_EP]) & (a[:s, D.AUX_TRANS] == 0), sc[:s], 1e9)
            if np.abs(MEM.wrap(a[s:i, D.AUX_YAW] - a[i, D.AUX_YAW], 2 * math.pi)).min() < TURN or pre.min() > NEAR:
                continue
            idx = sorted(MEM.choose(pre, np.arange(s), c.mem))
            mlat, mlvl = np.zeros((c.mem,) + lat.shape[2:], lat.dtype), np.zeros(c.mem, lvl.dtype)
            mok = np.zeros(c.mem, bool)
            for k, j in enumerate(idx):
                q = c.mem - len(idx) + k
                mlat[q], mlvl[q], mok[q] = lat[b, j], lvl[b, j], True
            out.append((np.asarray(lat[b, s:i + 1]), ch._acts(act[b, s:i + 1], aux[b, s:i + 1]), np.asarray(lvl[b, s:i + 1]), mlat, mlvl, mok))
            break
    return out, tries


wins, tries = returns(n, np.random.default_rng(5))
res = []
with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
    for k in range(0, len(wins) - B + 1, B):
        b = tuple(np.stack([w[i] for w in wins[k:k + B]]) for i in range(6))
        lat, act, lvl = f(b[0]), torch.from_numpy(b[1]).to(dev), ints(b[2])
        mlat, mlvl, mok = f(b[3]), ints(b[4]), torch.from_numpy(b[5]).to(dev)
        Fr = lat.shape[1]
        t = torch.rand(B, Fr, device=dev, generator=g).clamp_min(0.02)
        tau, mtau = torch.full((B, Fr), 0.05, device=dev), torch.full(mok.shape, 0.05, device=dev)
        eps, eps_c = rnd(*lat.shape), rnd(*lat.shape)
        mlat = 0.95 * mlat + 0.05 * rnd(*mlat.shape)
        run = lambda ml, mv, ok: m(lat, act, lvl, t, tau, eps, eps_c, mem=(ml, mv, ok, mtau)).float().cpu()[:, -1]
        res.append(torch.stack((run(mlat, mlvl, mok), run(mlat, mlvl, torch.zeros_like(mok)),
                                run(mlat.roll(1, 0), mlvl.roll(1, 0), mok.roll(1, 0)), t.cpu()[:, -1])))
if res:
    la, lb, lc, tl = torch.cat(res, 1)
    print(f"looking back ({la.numel()} windows, found in {tries} episodes): the frame that turns to a view seen before, but not in the last {(c.frames - 1) / 10:.1f} s")
    for name, sel in (("that frame", tl >= 0), ("that frame, drawn from much noise (t > 0.6)", tl > 0.6)):
        a, b_, w = float(la[sel].mean()), float(lb[sel].mean()), float(lc[sel].mean())
        print(f"  {name:48s} loss with memory {a:.4f}, without {b_:.4f} ({100 * (a / b_ - 1):+.1f}%), with a wrong memory {w:.4f} ({100 * (w / b_ - 1):+.1f}%)")
else:
    print(f"looking back: no such windows found in {tries} episodes")
