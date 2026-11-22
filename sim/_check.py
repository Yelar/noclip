"""Checks: the same seed gives the same data; what the explorer does; how often things happen.
python -m sim._check [steps]"""
import sys
import time

import torch

from . import ACTIONS, AUX, LEVELS, Explorer, Sim

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
with torch.no_grad():
    runs = []
    for rep in range(2):
        sim, ex = Sim(8, "cuda", 3), Explorer(8, "cuda", 3)
        f, l, a = sim.reset()
        fs, au = [f.clone()], [a.clone()]
        for t in range(150):
            f, l, a = sim.step(ex.act(sim))
            fs.append(f.clone()); au.append(a.clone())
        runs.append((torch.stack(fs), torch.stack(au)))
    d = (runs[0][0].int() - runs[1][0].int()).abs()
    print(f"same seed twice, 8 episodes x 151 frames: {int((d > 0).sum())} of {d.numel()} values differ (max {int(d.max())}); "
          f"aux max difference {float((runs[0][1] - runs[1][1]).abs().max()):.2e}")
    del runs
    B = 64
    sim, ex = Sim(B, "cuda", 11), Explorer(B, "cuda", 11)
    sim.reset()
    A, X, L = [], [], []
    t0 = time.time()
    for t in range(steps):
        act = ex.act(sim)
        f, l, a = sim.step(act)
        A.append(act); X.append(a); L.append(l)
    print(f"{B * steps / (time.time() - t0):.0f} frames/s")
    A, X, L = torch.stack(A, 1), torch.stack(X, 1), torch.stack(L, 1)       # [B, T, .]
    ix = {n: i for i, n in enumerate(AUX)}
    alive = X[..., ix["trans"]] == 0
    n = alive.float().sum()
    fr = lambda m: f"{100 * float((m & alive).float().sum() / n):.1f}%"
    print("keys: forward", fr(A[..., 0] > 0), "back", fr(A[..., 0] < 0), "none", fr(A[..., 0] == 0), "| strafe left", fr(A[..., 1] < 0),
          "right", fr(A[..., 1] > 0), "| sprint", fr(A[..., 4] > 0), "| light on", fr(A[..., 5] > 0))
    tu = A[..., 2][alive].abs()
    pi = A[..., 3][alive].abs()
    q = lambda v: [round(float(torch.quantile(v[:200000], p)), 3) for p in (0.5, 0.9, 0.99)]
    print("abs turn quantiles 50/90/99%:", q(tu), " abs pitch:", q(pi), " pitch angle 5/50/95%:",
          [round(float(torch.quantile(X[..., ix["pitch"]].flatten()[:200000], p)), 2) for p in (0.05, 0.5, 0.95)])
    sp = X[..., ix["speed"]]
    print("speed: still", fr(sp < 0.1), "walking", fr((sp >= 0.1) & (sp < 2)), "running", fr(sp >= 2),
          "| pushing into a wall", fr((A[..., 0] > 0) & (sp < 0.3)))
    mins = B * steps / 600
    print(f"per minute of play: noclips {float(A[..., 6].sum()) / mins:.2f}, caught {float(((X[..., ix['dying']][:, 1:] > 0) & (X[..., ix['dying']][:, :-1] == 0)).sum()) / mins:.2f}")
    print("frames: something in view", fr(X[..., ix["ent_visible"]] > 0), "| danger > 0", fr(X[..., ix["danger"]] > 0), "| in transition",
          f"{100 * float((~alive).float().mean()):.1f}%")
    print("share of frames per level:", {LEVELS[i]: round(float((L == i).float().mean()), 3) for i in range(len(LEVELS))})
    print("brightness at the player, mean per level:", {LEVELS[i]: round(float(X[..., ix["brightness"]][L == i].mean()), 3) for i in range(len(LEVELS)) if bool((L == i).any())})
