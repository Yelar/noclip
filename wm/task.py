"""Training the world model: one object that owns the model, the data and the optimizer on one GPU.

It is a task for `elastic.py` (GPUs of several accounts that join and leave one run):

  python elastic.py coord --task PATH/wm/task.py --data DIR --val DIR --size b ...

and `wm/train.py` runs the same object alone on one GPU.
"""
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm import data as D  # noqa: E402
from wm import model as M  # noqa: E402

ARGS = dict(
    data="", val="",                 # dataset folders (comma-separated); val: a folder of other seeds
    size="b", d=0, depth=0, heads=0, patch=1, levels=32, frames=32,
    mb=8, accum=1,                   # windows per micro-batch, micro-batches per step
    lr=2e-4, wd=0.0, beta2=0.99, clip=1.0,
    tau_max=0.3, tau_p=0.8,          # noise put on the context copies: with probability tau_p, a level up to tau_max
    lvl_drop=0.1,                    # how often the level is not told (for guidance, and for inventing a level)
    hat_p=0.0,                       # how often a micro-batch's context is the model's own estimate of the frames
    hat_lo=0.3,                      # ... made from frames noised to a level between hat_lo and 1 (higher: blurrier)
    hat_frames=0.75,                 # ... for this share of its frames (the others stay the recorded ones)
    hat_deep=0.0,                    # ... and how often the estimate is itself made from estimates (errors on errors)
    hat_start=0, hat_ramp=1,         # hat_p applies from this optimizer step on, rising linearly over hat_ramp steps
    init="",                         # a checkpoint (Task.save) whose weights the model starts from
    act_boost=0.0,                   # ... with the action columns it does not respond to set to this share of the
                                     # usual column's size (wm/model.py boost_actions); 0: left as they are
    mem=0, mem_drop=0.15,            # memory frames per window (wm/memory.py), and how often a window gets none
    mem_look=0.25,                   # how often a window is looked for in which a frame looks back (wm/data.py)
    actions=0,                       # action columns (0: as the first chunk has them); older chunks are filled up
    skip_state="",                   # action columns that must be 0 in every frame of a training window, e.g. "9,10":
                                     # no window with a blackout or red light in it (they are states that outlast the
                                     # model's window; trained on them, its dreams go dark and stay dark)
    aux=1,                           # in an elastic run: 1: the optimizer's state is averaged over the workers every
                                     # round (twice the model's size on the wire, each way); 0: every worker keeps its
                                     # own, as in DiLoCo, and a worker that starts with none ...
    cold_warm=100,                   # ... raises its learning rate over this many of its own steps
    compile=1, eval_batches=6, threads=4, seed=0,
)


class Task:
    def __init__(self, a, dev, wid):
        self.a, self.dev = a, dev
        torch.manual_seed(a.seed)
        d, depth, heads = M.SIZES[a.size]
        self.chunks = D.Chunks(a.data, a.frames, a.mem, a.actions, a.mem_drop, a.mem_look)
        if not self.chunks.files:
            raise SystemExit(f"no chunks with at least {a.frames} frames under {a.data!r}")
        lat = self.chunks.files[0][0]
        self.chunks.A = A = a.actions or self.chunks.files[0][1].shape[2]
        self.chunks.skip = tuple(int(x) for x in str(a.skip_state).split(",") if x.strip())
        self.cfg = M.Cfg(C=lat.shape[2], h=lat.shape[3], w=lat.shape[4], patch=a.patch, d=a.d or d, depth=a.depth or depth,
                         heads=a.heads or heads, A=A, levels=a.levels, frames=a.frames, mem=a.mem)
        mean, std = D.load_stats(a.data)
        self.mean = torch.tensor(mean, device=dev)[None, None, :, None, None]
        self.std = torch.tensor(std, device=dev)[None, None, :, None, None]
        self.model = M.build(self.cfg).to(dev)
        if a.init:  # in an elastic run the first worker's weights become the run's, and every worker builds the same
            self.model.load_state_dict(M.upgrade(torch.load(a.init, map_location=dev)["model"], self.model, a.act_boost))
        self.params = list(self.model.parameters())
        self.fwd = torch.compile(self.model) if (a.compile and dev != "cpu") else self.model
        decay = [p for p in self.params if p.ndim >= 2]
        rest = [p for p in self.params if p.ndim < 2]
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": rest, "weight_decay": 0.0}],
                                     lr=a.lr, betas=(0.9, a.beta2), fused=dev != "cpu")
        seed = [a.seed, zlib.crc32(wid.encode()), int.from_bytes(os.urandom(4), "little")]
        self.feed = D.Feeder(self.chunks, a.mb, seed, a.threads)
        self.gen = torch.Generator(device=dev).manual_seed(int(np.random.default_rng(seed).integers(2 ** 62)))
        self.amp = torch.autocast(dev, dtype=torch.bfloat16) if dev != "cpu" else torch.autocast("cpu", enabled=False)
        self._val = None
        self.n = 0  # optimizer steps this run is old (elastic.py tells a joiner through set_steps)
        self.own = 0  # optimizer steps this process has taken (its optimizer state is that old, with aux=0)

    def _gpu(self, b):
        """-> lat, act, lvl, and the memory frames (latents, levels, which slots are filled) or None"""
        f = lambda x: (torch.from_numpy(x).to(self.dev).float() - self.mean) / self.std
        g = lambda x: torch.from_numpy(x.astype(np.int64)).to(self.dev)
        mem = (f(b[3]), g(b[4]), torch.from_numpy(b[5]).to(self.dev)) if len(b) > 3 else None
        return f(b[0]), torch.from_numpy(b[1]).to(self.dev), g(b[2]), mem

    def _mem(self, mem, drop, g):
        """The memory frames as the model takes them: noised like context, their level hidden where the window's is."""
        if mem is None:
            return None
        mlat, mlvl, mok = mem
        mtau = self._levels(mlat.shape[:2], g)[1]
        mt = mtau[:, :, None, None, None]
        mlat = (1 - mt) * mlat + mt * torch.randn(mlat.shape, device=self.dev, generator=g)
        if drop is not None:
            mlvl = torch.where(drop, torch.full_like(mlvl, self.cfg.levels), mlvl)
        return mlat, mlvl, mok, mtau

    def _levels(self, shape, g):
        """Noise levels: targets t (mostly logit-normal, some uniform), context copies tau (small, sometimes none)."""
        u = torch.rand(shape, device=self.dev, generator=g)
        t = torch.where(torch.rand(shape, device=self.dev, generator=g) < 0.9,
                        torch.sigmoid(torch.randn(shape, device=self.dev, generator=g)), u)
        tau = self.a.tau_max * torch.rand(shape, device=self.dev, generator=g) ** 1.5
        tau = tau * (torch.rand(shape, device=self.dev, generator=g) < self.a.tau_p)
        return t.clamp(1e-4, 1.0), tau

    def step(self, lr_mult, k=None):
        cold = 1.0 if self.a.aux else min(1.0, (self.own + 1) / max(1, self.a.cold_warm))
        for g in self.opt.param_groups:
            g["lr"] = self.a.lr * lr_mult * cold
        k, total = k or self.a.accum, 0.0
        for _ in range(k):
            lat, act, lvl, mem = self._gpu(self.feed.get())
            t, tau = self._levels(lat.shape[:2], self.gen)
            drop = torch.rand(lat.shape[0], 1, device=self.dev, generator=self.gen) < self.a.lvl_drop
            lvl = torch.where(drop, torch.full_like(lvl, self.cfg.levels), lvl)
            mem = self._mem(mem, drop, self.gen)
            ctx = None
            hat_p = self.a.hat_p * min(1.0, max(0.0, (self.n - self.a.hat_start + 1) / max(1, self.a.hat_ramp)))
            if hat_p and torch.rand((), device=self.dev, generator=self.gen) < hat_p:
                # the model's one-pass estimate of every frame from the recorded context: what it would hand itself.
                # Sometimes a second round: estimates made from those estimates, as errors build on errors in play.
                rnd = lambda *sh: torch.rand(sh, device=self.dev, generator=self.gen)
                deep = bool(rnd() < self.a.hat_deep)
                for _ in range(2 if deep else 1):
                    t0 = self.a.hat_lo + (1 - self.a.hat_lo) * rnd(*lat.shape[:2])
                    with torch.no_grad(), self.amp:
                        hat = self.fwd(lat, act, lvl, t0, tau, ctx=ctx, want="x0", mem=mem)
                    pick = rnd(*lat.shape[:2]) < self.a.hat_frames
                    ctx = torch.where(pick[:, :, None, None, None], hat.to(lat.dtype), lat)
            with self.amp:
                loss = self.fwd(lat, act, lvl, t, tau, ctx=ctx, mem=mem).mean() / k
            loss.backward()
            total += loss.item()
        if self.a.clip:
            torch.nn.utils.clip_grad_norm_(self.params, self.a.clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.n += 1
        self.own += 1
        return total, k * self.a.mb * self.a.frames

    @torch.no_grad()
    def evaluate(self):
        """Loss on fixed windows with fixed noise: the same number on every worker."""
        if self._val is None:
            ch = D.Chunks(self.a.val, self.a.frames, self.a.mem, self.cfg.A, 0.0) if self.a.val else self.chunks
            # two windows at a time: validation runs next to the training step's memory on the same GPU
            self._val = ch.fixed(2, self.a.eval_batches * max(1, self.a.mb // 2))
        g = torch.Generator(device=self.dev).manual_seed(1234)
        self.model.eval()
        tot = 0.0
        for b in self._val:
            lat, act, lvl, mem = self._gpu(b)
            t, tau = self._levels(lat.shape[:2], g)
            eps = torch.randn(lat.shape, device=self.dev, generator=g)
            eps_c = torch.randn(lat.shape, device=self.dev, generator=g)
            mem = self._mem(mem, None, g)
            with self.amp:
                tot += self.model(lat, act, lvl, t, tau, eps, eps_c, mem=mem).mean().item()
        self.model.train()
        return tot / len(self._val)

    # ---- what elastic.py needs besides step and evaluate
    def warmup(self):
        import elastic_tasks as et
        et._dry_step(self, [self.opt])

    def aux(self):
        if not self.a.aux:
            return []
        import elastic_tasks as et
        ts, self.n_log = et._state_tensors([self.opt])
        return ts

    def aux_betas(self):
        import elastic_tasks as et
        return et._state_betas([self.opt])

    def set_steps(self, n):
        import elastic_tasks as et
        et._set_steps([self.opt], n)
        self.n = max(self.n, int(n))

    # ---- a checkpoint that wm/sample.py and wm/play.py read
    def save(self, path, **extra):
        from dataclasses import asdict
        tmp = path + ".tmp"
        torch.save({"cfg": asdict(self.cfg), "model": self.model.state_dict(), "mean": self.mean.flatten().tolist(),
                    "std": self.std.flatten().tolist(), "args": dict(vars(self.a)), **extra}, tmp)
        os.replace(tmp, path)


def build(a, dev, wid):
    return Task(a, dev, wid)
