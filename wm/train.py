"""Train the world model on one GPU (the same task object that elastic.py trains across many).

  python -m wm.train --data DIR --val DIR --out RUN --size s --steps 20000

Writes RUN/log.jsonl (loss, frames/s, validation loss) and RUN/model.pt (wm/sample.py and wm/play.py read it).
Started again with the same --out it continues.
"""
import argparse
import json
import math
import os
import time
import types

import torch

from wm import task as T

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--floor", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_s", type=float, default=600)
    for k, v in T.ARGS.items():
        p.add_argument("--" + k, type=type(v), default=v)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    task = T.build(types.SimpleNamespace(**{k: getattr(a, k) for k in T.ARGS}), "cuda", "solo")
    ck, step = os.path.join(a.out, "model.pt"), 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cuda")
        task.model.load_state_dict(st["model"]); task.opt.load_state_dict(st["opt"]); step = st["step"]
        task.n = step
    n = sum(q.numel() for q in task.params)
    print(f"{n / 1e6:.1f}M parameters, {len(task.chunks.files)} chunks, window {a.frames} frames x {task.cfg.N} tokens, "
          f"from step {step}", flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = t_save = time.time()
    acc, frames = 0.0, 0
    while step < a.steps:
        m = min(1.0, (step + 1) / a.warmup) * (a.floor + (1 - a.floor) * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        loss, f = task.step(m)
        step += 1; acc += loss; frames += f
        rec = None
        if step % a.log_every == 0:
            dt = time.time() - t0
            rec = {"step": step, "loss": round(acc / a.log_every, 5), "frames_s": round(frames / dt, 1), "lr_mult": round(m, 4)}
            acc, frames, t0 = 0.0, 0, time.time()
        if step % a.eval_every == 0 or step == a.steps:
            rec = {**(rec or {"step": step}), "val": round(task.evaluate(), 5)}
            t0 = time.time()
        if rec:
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
        if time.time() - t_save > a.save_s or step == a.steps:
            task.save(ck, opt=task.opt.state_dict(), step=step)
            t_save = time.time()
