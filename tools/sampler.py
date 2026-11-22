"""How few passes can draw a frame as well as the play server's 8 Euler steps?

  python tools/sampler.py CKPT VAL_DIR [AE] [N]

For N recorded moments (a full window of recorded context each) the next frame is drawn from the same noise by
several samplers. The yardstick is the frame the model's own flow leads to from that noise (Euler with 64 steps):
a sampler that lands as close to it as 8 Euler steps do draws the same pictures, with fewer passes if it needs
fewer. Per sampler: passes per frame, distance to that frame in latent units (1 = the latents' spread) and, with an
autoencoder given, as PSNR of the decoded pictures.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm import data as D  # noqa: E402
from wm import fast, roll  # noqa: E402

ckpt, val = sys.argv[1], sys.argv[2]
ae_spec = sys.argv[3] if len(sys.argv) > 3 else ""
n = int(sys.argv[4]) if len(sys.argv) > 4 else 48
dev = "cuda"
m, mean, std = roll.load(ckpt, dev)
c = m.cfg
keep = c.frames - 1
ch = D.Chunks(val, keep + 1, A=c.A)
lat, act, lvl = ch.fixed(n, 1, seed=11)[0][:3]
lat = (torch.from_numpy(lat).to(dev).float() - mean[:, None]) / std[:, None]
act, lvl = torch.from_numpy(act).to(dev), torch.from_numpy(lvl.astype(np.int64)).to(dev)
fd = fast.FastDream(m, n, 0.05, 8, 1.0, seed=0)
fd.seed(lat[:, :keep], act[:, :keep], lvl[:, :keep])
a1, l1, pos = act[:, keep].to(fd.dt), lvl[:, keep], fd._pos()
g = torch.Generator(device=dev).manual_seed(5)
noise = torch.randn((n, c.C, c.h, c.w), device=dev, generator=g)


def times(steps, shift=1.0):
    s = torch.linspace(1, 0, steps + 1, device=dev)
    return shift * s / (1 + (shift - 1) * s)


@torch.no_grad()
def draw(method, steps, shift=1.0):
    """-> the frame, and how many passes it took"""
    ts, x, prev, passes = times(steps, shift), noise.clone(), None, 0
    vel = lambda x_, t_: fast.frame_pass(m, x_.to(fd.dt), t_.expand(n), a1, l1, 1, pos, fd.K, fd.V, fd.bias).float()
    for i in range(steps):
        h = ts[i + 1] - ts[i]
        v = vel(x, ts[i]); passes += 1
        if method == "euler" or (method == "ab2" and prev is None):
            x = x + h * v
        elif method == "ab2":           # the last step's velocity is used once more (no extra pass)
            r = h / (ts[i] - ts[i - 1])
            x = x + h * ((1 + r / 2) * v - (r / 2) * prev)
        elif method == "heun":          # a second pass at the step's end, except on the last step
            if i == steps - 1:
                x = x + h * v
            else:
                v2 = vel(x + h * v, ts[i + 1]); passes += 1
                x = x + h * (v + v2) / 2
        prev = v
    return x, passes


ref, _ = draw("euler", 64)
ae = None
if ae_spec:
    from wm import ae as AE
    ae = AE.load(ae_spec, dev, batch=16)
    dec = lambda z: ae.decode(z * std + mean).float()
    ref_img, true_img = dec(ref), dec(lat[:, keep])
psnr = lambda a_, b_: float((10 * torch.log10(255 ** 2 / ((a_ - b_) ** 2).mean((1, 2, 3)).clamp_min(1e-3))).mean())
print(f"{ckpt}: {n} moments, window {keep} frames; distance to the 64-step frame")
rows = [("euler", s, sh) for s in (8, 6, 5, 4, 3) for sh in (1.0,)] + [("euler", s, sh) for s in (6, 5, 4) for sh in (1.5, 2.0, 3.0)] \
    + [("ab2", s, sh) for s in (6, 5, 4, 3) for sh in (1.0, 1.5, 2.0, 3.0)] + [("heun", s, sh) for s in (4, 3, 2) for sh in (1.0, 2.0)]
for method, steps, shift in rows:
    x, passes = draw(method, steps, shift)
    line = f"  {method:5s} {steps} steps, shift {shift}: {passes} passes, latent distance {float(((x - ref) ** 2).mean()):.5f}"
    if ae is not None:
        img = dec(x)
        line += f", picture {psnr(img, ref_img):5.2f} dB to the 64-step frame, {psnr(img, true_img):5.2f} dB to the recorded one"
    print(line, flush=True)
if ae is not None:
    print(f"  the 64-step frame itself: {psnr(ref_img, true_img):5.2f} dB to the recorded one")
