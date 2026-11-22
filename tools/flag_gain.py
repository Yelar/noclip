"""Does the model use what it is told about the lamps?   python tools/flag_gain.py CKPT VAL_DIR [WINDOWS]

The `blackout` action column (sim/gen.py) says that the lamps are out. This measures, on recorded frames (recorded
context, the same noise each time, no memory frames), the loss three ways: with the actions as recorded, with the
column set to 0, and with the column inverted.

 - The frames right after a blackout ends are where it matters most: the lamps come back from one frame to the
   next, and nothing in the dark frames before says when. A model that uses the column draws the lit room there;
   one that does not keeps it dark, and its loss is the same whatever the column says.
 - Frames deep inside a blackout, and frames of windows without any: inverting the column should hurt a model that
   believes it.

Prints the mean loss per case and the change against the recorded actions.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm import data as D  # noqa: E402
from wm import roll  # noqa: E402

BLACK, AFTER = 9, 3      # the column; frames after the end that count as "the lamps come back"
ckpt, val = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 96
dev, B = "cuda", 4
m, mean, std = roll.load(ckpt, dev, dtype=torch.float32)
c = m.cfg
if c.A <= BLACK:
    raise SystemExit(f"this model takes {c.A} action columns: it is not told about blackouts")
Fr, R = c.frames, getattr(c, "mem", 0)
ch = D.Chunks(val, Fr, 0, c.A, 0.0)
rng = np.random.default_rng(3)
ends, plain = [], []
for lat, act, lvl, aux in ch.files:
    if act.shape[2] <= BLACK:
        raise SystemExit("this dataset has no blackout column")
    f = np.asarray(act[:, :, BLACK]) > 0.5                                   # [E, T]
    for b, i in zip(*np.nonzero(f[:, :-1] & ~f[:, 1:])):                     # the last dark frame is i, lit from i + 1
        s = int(i) + 1 + AFTER - Fr
        if s >= 0 and int(i) + 1 + AFTER <= f.shape[1] and f[b, s:int(i) + 1].sum() >= Fr - AFTER - 14:
            ends.append((lat, act, lvl, int(b), s))
    for _ in range(6):
        b, s = int(rng.integers(f.shape[0])), int(rng.integers(f.shape[1] - Fr))
        if not f[b, s:s + Fr].any():
            plain.append((lat, act, lvl, int(b), s))
ends = [ends[i] for i in rng.permutation(len(ends))[:n]]
plain = [plain[i] for i in rng.permutation(len(plain))[:n]]
g = torch.Generator(device=dev).manual_seed(7)


def losses(wins):
    """-> per-frame loss [n, Fr] with the actions as recorded, with the column 0, with it inverted; the column [n, Fr]"""
    out = [[], [], [], []]
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        for k in range(0, len(wins) - B + 1, B):
            w = wins[k:k + B]
            lat = torch.from_numpy(np.stack([np.asarray(x[0][x[3], x[4]:x[4] + Fr], np.float32) for x in w])).to(dev)
            lat = (lat - mean[:, None]) / std[:, None]
            act = torch.from_numpy(np.stack([ch._acts(x[1][x[3], x[4]:x[4] + Fr], None) for x in w])).to(dev)
            lvl = torch.from_numpy(np.stack([np.asarray(x[2][x[3], x[4]:x[4] + Fr]).astype(np.int64) for x in w])).to(dev)
            t = torch.rand(B, Fr, device=dev, generator=g).clamp_min(0.02)
            tau = torch.full((B, Fr), 0.05, device=dev)
            eps = torch.randn(lat.shape, device=dev, generator=g)
            eps_c = torch.randn(lat.shape, device=dev, generator=g)
            mem = None
            if R:
                mem = (torch.zeros((B, R) + lat.shape[2:], device=dev), torch.zeros(B, R, dtype=torch.long, device=dev),
                       torch.zeros(B, R, dtype=torch.bool, device=dev), torch.full((B, R), 0.05, device=dev))
            zero, inv = act.clone(), act.clone()
            zero[..., BLACK] = 0
            inv[..., BLACK] = 1 - inv[..., BLACK]
            for j, a in enumerate((act, zero, inv)):
                out[j].append(m(lat, a, lvl, t, tau, eps, eps_c, mem=mem).float().cpu())
            out[3].append(act[..., BLACK].float().cpu())
    return [torch.cat(x) for x in out]


def show(name, rec, other, sel, what):
    a, b = float(rec[sel].mean()), float(other[sel].mean())
    print(f"  {name:54s} loss as recorded {a:.4f}, {what} {b:.4f} ({100 * (b / a - 1):+.1f}%)  [{int(sel.sum())} frames]")


print(f"{ckpt}: {len(ends)} windows that end {AFTER} frames after a blackout does, {len(plain)} windows without one")
if len(ends) >= B:
    rec, zero, inv, flag = losses(ends)
    pos = torch.arange(Fr)[None].expand_as(rec)
    show("the first frame after the lamps come back", rec, inv, pos == Fr - AFTER, "told the blackout goes on")
    show(f"the {AFTER - 1} frames after that one", rec, inv, pos > Fr - AFTER, "told the blackout goes on")
    show("the dark frames before, told as recorded or not", rec, zero, (pos < Fr - AFTER) & (flag > 0), "told there is no blackout")
if len(plain) >= B:
    rec, zero, inv, flag = losses(plain)
    show("frames of windows without a blackout", rec, inv, torch.ones_like(rec, dtype=torch.bool), "told there is one")
