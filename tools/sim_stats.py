"""What the scripted player meets in each level:  python tools/sim_stats.py [STEPS] [ENVS] [LEVELS]
Per level: share of frames with an entity in view / near, share looking far up or down, share in a blackout / under red lamps, share with a wall right in
front of the face (nearer than WALL), share with a near wall across the middle of the view (nearer than FLAT over
45 degrees: a picture with little in it), share of 2.4 s windows that show nothing else, door uses and captures
per player-minute, mean speed."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import LEVELS, Explorer, Sim  # noqa: E402

WALL, FLAT, WINDOW = 1.2, 2.5, 24      # metres; the model's window of frames

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
B = int(sys.argv[2]) if len(sys.argv) > 2 else 48
levels = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else range(len(LEVELS))
with torch.no_grad():
    for L in levels:
        sim, ex = Sim(B, "cuda", 100 + L, levels=[L]), Explorer(B, "cuda", L)
        sim.reset()
        seen = near = use = shut = caught = speed = nc = hits = tilt = wall = flat = walled = black = red = 0.0
        facing = torch.zeros(B, device="cuda")
        t0 = time.time()
        for t in range(steps):
            act = ex.act(sim)
            use += float(((act[:, 7] > 0.5) & (sim.door_now == 1)).sum())
            shut += float(((act[:, 7] > 0.5) & (sim.door_now == 3)).sum())
            nc += float((act[:, 6] > 0.5).sum())
            _, _, aux = sim.step(act)
            seen += float((aux[:, 8] > 0).float().mean())
            near += float((aux[:, 7] < 20).float().mean())
            caught += float((aux[:, 11] > 0).float().sum()) / 20
            hits += float((aux[:, 17] == 6).float().sum())
            speed += float(aux[:, 4].mean())
            tilt += float((aux[:, 3].abs() > 0.35).float().mean())
            black += float((aux[:, 18] > 0).float().mean()); red += float((aux[:, 19] > 0).float().mean())
            free = sim.probe(ex.rel[7:10])
            blank = (free.amax(1) < FLAT) & (sim.trans == 0)
            facing = torch.where(blank, facing + 1, torch.zeros_like(facing))
            wall += float(((free[:, 1] < WALL) & (sim.trans == 0)).float().mean())
            flat += float(blank.float().mean())
            walled += float((facing >= WINDOW).float().mean())
        mins = B * steps / 10 / 60
        print(f"{L:2d} {LEVELS[L]:11s} in view {100 * seen / steps:4.1f}%  near {100 * near / steps:4.1f}%  looking up/down {100 * tilt / steps:4.1f}%  blackout {100 * black / steps:4.1f}%  red {100 * red / steps:4.1f}%  wall in the face {100 * wall / steps:4.1f}%, across the view {100 * flat / steps:4.1f}%, whole windows of that {100 * walled / steps:4.1f}%  per minute: doors opened "
              f"{use / mins:4.2f}, shut {shut / mins:4.2f}, struck {hits / mins:4.2f}, died {caught / mins:4.2f}, noclips {nc / mins:4.2f}  speed {speed / steps:4.2f} m/s  "
              f"doors per map {int((sim.d_cell >= 0).sum()) // B:3d}  {B * steps / (time.time() - t0):5.0f} frames/s", flush=True)
