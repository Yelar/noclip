"""Training speed of the world model on random data:  python -m wm.bench SIZE MB [FRAMES] [COMPILE]"""
import sys
import time

import torch

from wm import model as M

size, mb = sys.argv[1], int(sys.argv[2])
frames = int(sys.argv[3]) if len(sys.argv) > 3 else 32
comp = int(sys.argv[4]) if len(sys.argv) > 4 else 1
d, depth, heads = M.SIZES[size]
c = M.Cfg(d=d, depth=depth, heads=heads, frames=frames)
m = M.build(c).cuda()
f = torch.compile(m) if comp else m
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
lat = torch.randn(mb, frames, c.C, c.h, c.w, device="cuda")
act = torch.randn(mb, frames, c.A, device="cuda")
lvl = torch.randint(0, 10, (mb, frames), device="cuda")
for i in range(26):
    if i == 6:
        torch.cuda.synchronize(); t0 = time.time()
    t = torch.rand(mb, frames, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = f(lat, act, lvl, t, t * 0.2).mean()
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
dt = (time.time() - t0) / 20
print(f"{size} mb {mb} frames {frames} compile {comp}: {dt * 1000:.0f} ms/step, {mb * frames / dt:.0f} frames/s, "
      f"{2 * mb * frames * c.N / dt / 1000:.0f}k tokens/s, {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
