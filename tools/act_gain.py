"""How much does the model respond to each of its inputs?   python tools/act_gain.py CKPT VAL_DIR [WINDOWS]

For every action column: the frame the model infers for each moment of recorded windows (recorded context, targets
at a high noise level, the same noise each time) is computed twice, once with the actions as recorded and once with
that column changed in every frame (a key or a state column flipped between 0 and 1; forward, strafe, turn and
pitch negated). Printed: how far the two inferred frames are apart on average (in latent units, 1 = the latents'
spread), and the norm of the column's weights in the first layer of the action network. A column the model does
not respond to (a difference near 0) is one it has not learned: pressing that key, or the director setting that
state, changes nothing in the picture.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm import data as D  # noqa: E402
from wm import roll  # noqa: E402

NAMES = ["forward", "right", "turn", "pitch", "sprint", "light", "noclip", "use", "presence", "blackout", "red"]
SIGNED = (0, 1, 2, 3)
ckpt, val = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
dev, B = "cuda", 4
m, mean, std = roll.load(ckpt, dev, dtype=torch.float32)
c = m.cfg
Fr, R = c.frames, getattr(c, "mem", 0)
ch = D.Chunks(val, Fr, 0, c.A, 0.0)
g = torch.Generator(device=dev).manual_seed(7)
diff, floor = torch.zeros(c.A), 0.0
with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
    k = 0
    for b in ch.fixed(B, n // B, seed=21):
        lat = (torch.from_numpy(b[0]).to(dev).float() - mean[:, None]) / std[:, None]
        act, lvl = torch.from_numpy(b[1]).to(dev), torch.from_numpy(b[2].astype(np.int64)).to(dev)
        t = torch.full((B, Fr), 0.8, device=dev)
        tau = torch.full((B, Fr), 0.05, device=dev)
        eps = torch.randn(lat.shape, device=dev, generator=g)
        eps_c = torch.randn(lat.shape, device=dev, generator=g)
        mem = None
        if R:
            mem = (torch.zeros((B, R) + lat.shape[2:], device=dev), torch.zeros(B, R, dtype=torch.long, device=dev),
                   torch.zeros(B, R, dtype=torch.bool, device=dev), torch.full((B, R), 0.05, device=dev))
        base = m(lat, act, lvl, t, tau, eps, eps_c, want="x0", mem=mem).float()
        floor += float((m(lat, act.clone(), lvl, t, tau, eps, eps_c, want="x0", mem=mem).float() - base).abs().mean())
        for j in range(c.A):
            a = act.clone()
            a[..., j] = -a[..., j] if j in SIGNED else 1 - a[..., j]
            if j in SIGNED:                      # a column that is 0 in a frame would stay 0: give those frames a 1
                a[..., j] = torch.where(act[..., j] == 0, torch.ones_like(a[..., j]), a[..., j])
            diff[j] += float((m(lat, a, lvl, t, tau, eps, eps_c, want="x0", mem=mem).float() - base).abs().mean())
        k += 1
w = m.a_mlp[0].weight.float().norm(dim=0).cpu()
print(f"{ckpt}: {k * B} windows of {Fr} frames; the inferred frame's mean change when a column is changed, and the column's weight norm")
print(f"  (the same input twice: change {floor / k:.4f}, which is what no response looks like)")
for j in range(c.A):
    name = NAMES[j] if j < len(NAMES) else str(j)
    print(f"  {j:2d} {name:9s} change {diff[j] / k:.4f}   weight norm {float(w[j]):7.3f}")
