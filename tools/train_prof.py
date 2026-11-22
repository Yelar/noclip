"""Where a training step's time goes, on real data and with the main run's settings:

  python tools/train_prof.py DATA_DIR [INIT.pt] [key=value ...]     e.g. hat_p=0 mem=0 compile=0 mb=3 accum=3 ops=1

Prints frames/s of whole optimizer steps (wm/task.py's own `step`), then the same step taken apart: waiting for
data, moving it to the GPU, the extra pass that makes the model's own context (hat), forward, backward, optimizer.
ops=1 adds the GPU operations that take the most time, measured without compilation (where they have names);
ops=2 the GPU kernels of the compiled model (matrix products against everything else).
"""
import os
import sys
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm import task as T  # noqa: E402

data = sys.argv[1]
init = sys.argv[2] if len(sys.argv) > 2 and "=" not in sys.argv[2] else ""
over = dict(kv.split("=", 1) for kv in sys.argv[2:] if "=" in kv)
ops, steps = int(over.pop("ops", 0)), int(over.pop("steps", 16))
a = dict(T.ARGS, data=data, val="", size="b", frames=24, mem=8, actions=9, mb=3, accum=3, lr=8e-5, hat_p=0.5, init=init)
for k, v in over.items():
    a[k] = type(T.ARGS[k])(v)
a = types.SimpleNamespace(**a)
task = T.build(a, "cuda", "prof")
sync = torch.cuda.synchronize


def whole(n):
    sync(); t0 = time.time(); fr = 0
    for _ in range(n):
        fr += task.step(1.0)[1]
    sync()
    return fr / (time.time() - t0), (time.time() - t0) / n


for _ in range(6):      # compilation, both kinds of micro-batch
    task.step(1.0)
fps, dt = whole(steps)
print(f"{' '.join(f'{k}={v}' for k, v in over.items()) or 'main run settings'}: {fps:.1f} frames/s, {dt * 1e3:.0f} ms per optimizer step "
      f"({a.accum} micro-batches of {a.mb} windows of {a.frames} frames + {a.mem} memory frames), {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)

# the same step, taken apart (one synchronisation per stage: slightly slower in total than the step above)
tm = dict(data=0.0, to_gpu=0.0, hat=0.0, forward=0.0, backward=0.0, optimizer=0.0)


def lap(name, t0):
    sync()
    tm[name] += time.time() - t0
    return time.time()


n_mb = 0
for _ in range(steps):
    for _ in range(a.accum):
        t0 = time.time()
        b = task.feed.get(); t0 = lap("data", t0)
        lat, act, lvl, mem = task._gpu(b)
        t, tau = task._levels(lat.shape[:2], task.gen)
        drop = torch.rand(lat.shape[0], 1, device="cuda", generator=task.gen) < a.lvl_drop
        lvl = torch.where(drop, torch.full_like(lvl, task.cfg.levels), lvl)
        mem = task._mem(mem, drop, task.gen); t0 = lap("to_gpu", t0)
        ctx = None
        if a.hat_p and torch.rand((), device="cuda", generator=task.gen) < a.hat_p:
            t0h = a.hat_lo + (1 - a.hat_lo) * torch.rand(lat.shape[:2], device="cuda", generator=task.gen)
            with torch.no_grad(), task.amp:
                hat = task.fwd(lat, act, lvl, t0h, tau, ctx=None, want="x0", mem=mem)
            pick = torch.rand(lat.shape[:2], device="cuda", generator=task.gen) < a.hat_frames
            ctx = torch.where(pick[:, :, None, None, None], hat.to(lat.dtype), lat); t0 = lap("hat", t0)
        with task.amp:
            loss = task.fwd(lat, act, lvl, t, tau, ctx=ctx, mem=mem).mean() / a.accum
        t0 = lap("forward", t0)
        loss.backward(); t0 = lap("backward", t0)
        n_mb += 1
    t0 = time.time()
    torch.nn.utils.clip_grad_norm_(task.params, a.clip)
    task.opt.step(); task.opt.zero_grad(set_to_none=True); lap("optimizer", t0)
tot = sum(tm.values())
print("  per optimizer step: " + ", ".join(f"{k} {1e3 * v / steps:.0f} ms ({100 * v / tot:.0f}%)" for k, v in tm.items()), flush=True)

if ops:
    from torch.profiler import ProfilerActivity, profile
    lat, act, lvl, mem = task._gpu(task.feed.get())
    t, tau = task._levels(lat.shape[:2], task.gen)
    mem = task._mem(mem, None, task.gen)
    net = task.fwd if ops == 2 else task.model
    for _ in range(2):
        with task.amp:
            net(lat, act, lvl, t, tau, mem=mem).mean().backward()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            with task.amp:
                net(lat, act, lvl, t, tau, mem=mem).mean().backward()
        sync()
    task.opt.zero_grad(set_to_none=True)
    ev = sorted(prof.key_averages(), key=lambda e: -getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)))
    tot = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ev)
    dt = lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
    kern = [e for e in ev if not e.key.startswith(("aten::", "Optimizer", "Torch", "autograd", "torch", "Command", "CompiledFunction", "## "))]
    gemm = sum(dt(e) for e in kern if any(w in e.key.lower() for w in ("gemm", "cutlass", "sgemm", "matmul")))
    kt = sum(dt(e) for e in kern)
    print(f"  GPU kernels of one micro-batch, forward + backward ({'compiled' if ops == 2 else 'not compiled'}): {kt / 3e3:.0f} ms, "
          f"matrix products {100 * gemm / max(1, kt):.0f}% of it")
    for e in sorted(kern, key=lambda e: -dt(e))[:16]:
        print(f"    {e.key[:70]:70s} {100 * dt(e) / max(1, kt):5.1f}%  {dt(e) / 3e3:7.1f} ms  x{e.count // 3}")
