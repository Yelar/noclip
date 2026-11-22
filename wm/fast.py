"""A faster frame loop for playing: `FastDream` draws the same frames as `roll.Dream` with less work per frame.

`roll.Dream` redoes the pass over all context frames for every new frame. Here the context's temporal keys and
values live in a ring of `keep` slots per block, and a finished frame costs one more single-frame pass (as context)
that writes its keys and values into the oldest slot. Attention over the ring needs no copying: scores against the
ring and against the frame itself are computed separately and normalised together. Guidance runs both variants
(level told / not told) as one batch against the same ring.

What differs from `roll.Dream` in this mode (`exact=False`):
  - A context frame's keys and values are computed once, when the frame is finished, from the frames that were in
    the window then. `roll.Dream` recomputes them every frame from the frames in the window now, so there an old
    frame "forgets" what was before the window; here it does not. Until the window is full (the first `keep`
    frames of a dream) both give the same numbers.
  - Frame positions keep counting up instead of restarting at 0 in each window. Positions enter only as rotations
    of queries and keys, so only their differences matter; the ring is rotated back every few thousand frames.
With `exact=True` the ring is refilled by `model.context` on the current window before every frame: the same
computation as `roll.Dream` (and as training), at the cost of that pass.

Memory (models with cfg.mem slots): `set_memory` puts frames from longer ago behind the ring, in slots of their own,
before a frame is drawn. A memory frame's keys and values come from a pass over that frame alone, done once and
kept; only their rotation to the current frame positions is redone every frame.

  python -m wm.fast CKPT [--ae dcae] [--steps 8]     self-test against the model's own functions, then timings
"""
import time

import torch

from wm.model import MEM_STRIDE, _norm, _rot


def frame_pass(m, z, t, act, lvl, stream, pos, K, V, bias, want_kv=False):
    """One pass over one frame per batch row. z [Bt, C, h, w], t [Bt], act [Bt, A], lvl [Bt]; stream 0 (context) or
    1 (target); pos: the frame's position (a tensor, so that a compiled pass serves every frame). K, V [L, M0, H, keep, dh]: the ring of context keys and values, shared
    by the G = Bt * N / M0 groups of the batch; bias [keep]: 0 for filled slots, -inf for empty ones.
    Returns the head's output [Bt, C, h, w] and, with want_kv, this frame's own keys and values [L, Bt*N, H, dh]."""
    cond = m._cond(t[:, None], act[:, None], lvl[:, None], stream)
    x = m.inp(m._tok(z[:, None])) + m.pos
    Bt, _, N, d = x.shape
    H = m.cfg.heads
    dh = d // H
    G = Bt * N // K.shape[1]
    scale = dh ** -0.5
    M0 = K.shape[1]
    p = pos.view(1)
    ks, vs = [], []
    for i, blk in enumerate(m.blocks):
        s1, b1, s2, b2, s3, b3 = blk.ada(cond)[:, :, None].chunk(6, -1)
        x = x + blk.spatial((_norm(x) * (1 + s1) + b1).view(Bt, N, d)).view(Bt, 1, N, d)
        at = blk.temporal
        q, k, v = at.qkv((_norm(x) * (1 + s2) + b2).view(Bt * N, d)).view(Bt * N, 3, H, 1, dh).unbind(1)
        q, k, v = _rot(q, p)[:, :, 0], _rot(k, p)[:, :, 0], v[:, :, 0]               # [Bt*N, H, dh]
        # the G groups of the batch (level told / not told) become G query rows against the one ring: no copies
        qg = q.view(G, M0, H, dh).permute(1, 2, 0, 3)                                # [M0, H, G, dh]
        sc = torch.matmul(qg, K[i].transpose(-1, -2)).float() * scale + bias         # [M0, H, G, keep]
        ss = (q * k).sum(-1).view(G, M0, H).permute(1, 2, 0)[..., None].float() * scale   # the frame itself
        w = torch.softmax(torch.cat((sc, ss), -1), -1).to(q.dtype)
        y = torch.matmul(w[..., :-1], V[i]) + w[..., -1:] * v.view(G, M0, H, dh).permute(1, 2, 0, 3)
        y = y.permute(2, 0, 1, 3)                                                    # [G, M0, H, dh]
        x = x + at.out(y.reshape(Bt * N, d)).view(Bt, 1, N, d)
        x = x + blk.mlp(_norm(x) * (1 + s3) + b3)
        if want_kv:
            ks.append(k); vs.append(v)
    out = m._head(x, cond)[:, 0]
    return (out, torch.stack(ks), torch.stack(vs)) if want_kv else out


class FastDream:
    """B parallel dreams with `roll.Dream`'s interface (`seed`, `step`) and settings that can change between frames
    (`tau`, `steps`, `guide`, `shift`, `exact`). `ms` holds the last frame's stage times when `timing` is on."""

    REBASE = 4096

    def __init__(self, model, B=1, tau=0.05, steps=8, guide=1.0, shift=1.0, keep=None, seed=0, exact=False,
                 compile=False, timing=False, solver="euler"):
        self.m, self.B, self.tau, self.steps, self.guide, self.shift, self.exact = model, B, tau, steps, guide, shift, exact
        # "euler", or "ab2": every step after the first also uses the step before's velocity (second order, no more
        # passes), which lands as close to the flow's own frame in fewer steps (tools/sampler.py measures it)
        self.solver = solver
        c = model.cfg
        self.keep = keep or c.frames - 1
        self.R = getattr(c, "mem", 0)               # memory slots, behind the ring's
        self.dev, self.dt = model.pos.device, model.pos.dtype
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        shape = (c.depth, B * c.N, c.heads, self.keep + self.R, c.d // c.heads)
        self.K, self.V = torch.zeros(shape, device=self.dev, dtype=self.dt), torch.zeros(shape, device=self.dev, dtype=self.dt)
        self.timing, self.ms = timing, {}
        # compile: False, True, or a mode of torch.compile ("reduce-overhead": the pass replayed as one CUDA graph)
        self._pass = torch.compile(frame_pass, mode=compile if isinstance(compile, str) else None) if compile else frame_pass
        self.reset()

    def reset(self):
        self.lat, self.act, self.lvl = [], [], []   # the window: noised context copies, their actions and levels
        self.n, self.p, self.w = 0, 0, 0            # filled slots, position of the next frame, slot written next
        self.bias = torch.full((self.keep + self.R,), float("-inf"), device=self.dev)
        self.mem_kv, self.mem_n = {}, 0             # memory frames' keys and values by frame id; slots in use

    def _pos(self):
        return torch.tensor(self.p, device=self.dev)

    def _clock(self):
        if self.timing and self.dev.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def _push(self, z, act, lvl):
        """A finished frame becomes context: noise it once, and (fast mode) run it through as a context frame."""
        eps = torch.randn(z.shape, device=self.dev, generator=self.g).to(self.dt)
        zc = (1 - self.tau) * z.to(self.dt) + self.tau * eps
        # copies laid out afresh: a slice of a recorded episode ([:, i]) has other strides than the tensors of a played
        # frame, the compiled pass is compiled again for them, and a dream's first frame waits for that (a plain copy
        # keeps the slice's strides)
        act = act.to(self.dt).clone(memory_format=torch.contiguous_format)
        lvl = lvl.clone(memory_format=torch.contiguous_format)
        self.lat.append(zc); self.act.append(act); self.lvl.append(lvl)
        for q in (self.lat, self.act, self.lvl):
            del q[:-self.keep]
        if not self.exact:
            slot = self.w
            bias = self.bias.clone()
            bias[slot] = float("-inf")  # the slot about to be replaced holds the frame that leaves the window
            tau = torch.full((self.B,), self.tau, device=self.dev)
            _, k, v = self._pass(self.m, zc, tau, self.act[-1], lvl, 0, self._pos(), self.K, self.V, bias, True)
            self.K[:, :, :, slot], self.V[:, :, :, slot] = k, v
            self.bias[slot] = 0.0
            self.w = (self.w + 1) % self.keep
        self.n, self.p = min(self.n + 1, self.keep), self.p + 1
        if self.p >= self.REBASE and not self.exact:  # rotate the ring's keys back so that positions stay small
            back = self.p - self.keep
            self.K = _rot(self.K, torch.full((self.keep + self.R,), -back, device=self.dev))
            self.p -= back

    def _refill(self):
        """Exact mode: the ring is the context pass over the current window, positions 0 .. K-1."""
        Kn = len(self.lat)
        self.bias[:self.keep] = float("-inf")
        if Kn:
            tau = torch.full((self.B, Kn), self.tau, device=self.dev)
            cache = self.m.context(torch.stack(self.lat, 1), torch.stack(self.act, 1), torch.stack(self.lvl, 1), tau)
            for i, (k, v) in enumerate(cache):
                self.K[i, :, :, :Kn], self.V[i, :, :, :Kn] = k, v
            self.bias[:Kn] = 0.0
        self.n = self.p = Kn
        self.w = Kn % self.keep

    @torch.no_grad()
    def set_memory(self, frames):
        """The memory frames for the next frame: [(id, z [B, C, h, w] normalised latent, lvl [B])], oldest first, at
        most cfg.mem of them (call it before every `step`: the slots' positions move with the window)."""
        R, keep = self.R, self.keep
        if not R:
            return
        frames = frames[-R:]
        for fid, z, lvl in frames:
            if fid not in self.mem_kv:      # a pass over the frame alone: as a memory frame it attends to nothing else
                eps = torch.randn(z.shape, device=self.dev, generator=self.g).to(self.dt)
                zc = (1 - self.tau) * z.to(self.dt) + self.tau * eps
                tau = torch.full((self.B,), self.tau, device=self.dev)
                act = torch.zeros(self.B, self.m.cfg.A, device=self.dev, dtype=self.dt)
                _, k, v = self._pass(self.m, zc, tau, act, lvl, 2, torch.tensor(0, device=self.dev), self.K, self.V,
                                     torch.full_like(self.bias, float("-inf")), True)
                self.mem_kv[fid] = (k.clone(), v.clone())   # kept across passes: a CUDA graph reuses its outputs' memory
        ids = [f[0] for f in frames]
        if len(self.mem_kv) > 64:
            for fid in [i for i in self.mem_kv if i not in ids][: len(self.mem_kv) - 64]:
                del self.mem_kv[fid]
        n = self.mem_n = len(ids)
        self.bias[keep:] = float("-inf")
        if n:
            first = self.p - self.n         # the position of the oldest frame in the window, where training's window starts
            pos = first - (R - torch.arange(R - n, R, device=self.dev)) * MEM_STRIDE
            self.K[:, :, :, keep + R - n:] = _rot(torch.stack([self.mem_kv[i][0] for i in ids], 3), pos)
            self.V[:, :, :, keep + R - n:] = torch.stack([self.mem_kv[i][1] for i in ids], 3)
            self.bias[keep + R - n:] = 0.0

    @torch.no_grad()
    def set_exact(self, exact):
        """Switch modes in the middle of a dream; the window of frames is what both modes share."""
        if exact != self.exact:
            self.exact = exact
            self._refill()  # leaving exact mode: the ring continues from the exact pass's keys and values

    @torch.no_grad()    # as `step` is: with gradients on, the compiled pass is compiled a second time for it, and the
    def seed(self, z, act, lvl):    # first player to connect waited for that (2 s, 14 s on a fresh installation)
        for i in range(z.shape[1]):
            self._push(z[:, i], act[:, i], lvl[:, i])

    @torch.no_grad()
    def step(self, act, lvl):
        t0 = self._clock()
        if self.exact:
            self._refill()
        t1 = self._clock()
        c, B = self.m.cfg, self.B
        act = act.to(self.dt)
        x = torch.randn((B, c.C, c.h, c.w), device=self.dev, generator=self.g)     # the sum stays in float32
        s = torch.linspace(1, 0, self.steps + 1, device=self.dev)
        ts = self.shift * s / (1 + (self.shift - 1) * s)
        cfg = self.guide != 1.0
        if cfg:
            act2, lvl2 = torch.cat((act, act)), torch.cat((lvl, torch.full_like(lvl, c.levels)))
        pos, prev = self._pos(), None
        for i in range(self.steps):
            xi = x.to(self.dt)
            if cfg:
                v, v0 = self._pass(self.m, torch.cat((xi, xi)), ts[i].expand(2 * B), act2, lvl2, 1, pos, self.K, self.V,
                                   self.bias).float().chunk(2)
                v = v0 + self.guide * (v - v0)
            else:   # a copy: the next pass may write over what this one returned
                v = self._pass(self.m, xi, ts[i].expand(B), act, lvl, 1, pos, self.K, self.V, self.bias).to(torch.float32, copy=True)
            h = ts[i + 1] - ts[i]
            if self.solver == "ab2" and prev is not None:
                r = h / (ts[i] - ts[i - 1])
                x = x + h * ((1 + r / 2) * v - (r / 2) * prev)
            else:
                x = x + h * v
            prev = v
        t2 = self._clock()
        self._push(x, act, lvl)
        t3 = self._clock()
        if self.timing:
            self.ms = {"ctx": (t1 - t0 + t3 - t2) * 1e3, "den": (t2 - t1) * 1e3}
        return x


if __name__ == "__main__":
    import argparse

    from wm import roll
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--ae", default="")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--frames", type=int, default=60)
    a = p.parse_args()
    dev = "cuda"
    m, mean, std = roll.load(a.ckpt, dev)
    c = m.cfg
    B, keep = 2, c.frames - 1
    g = torch.Generator(device=dev).manual_seed(0)
    n = keep + 6
    act = torch.randn(B, n, c.A, device=dev, generator=g)
    lvl = torch.randint(0, 5, (B, n), device=dev, generator=g)

    # 1. Self-test in float32: the same seeds give the same frames as roll.Dream, always in exact mode, and in ring
    #    mode until the window is full (afterwards the two differ by design, see the docstring).
    m32 = roll.load(a.ckpt, dev, dtype=torch.float32)[0]
    for exact, guide in ((True, 1.0), (False, 1.0), (False, 1.7), (True, 1.7)):
        ref = roll.Dream(m32, B, 0.05, 4, guide, seed=3)
        fd = FastDream(m32, B, 0.05, 4, guide, seed=3, exact=exact)
        err = []
        for i in range(n):
            za, zb = ref.step(act[:, i], lvl[:, i]), fd.step(act[:, i], lvl[:, i])
            err.append(float((za - zb).abs().max() / za.abs().max().clamp_min(1e-6)))
        early, late = max(err[:keep + 1]), max(err[keep + 1:])
        print(f"self-test exact={int(exact)} guide={guide}: largest relative difference to roll.Dream {early:.2e} while the "
              f"window fills, {late:.2e} after", flush=True)
        assert early < 2e-3, err
        assert (not exact) or late < 2e-3, err
    del m32

    # 2. Timings, single player.
    def run(name, d, frames=a.frames, dec=None):
        act1, lvl1 = act[:1, 0], lvl[:1, 0]
        ctx, den, de, tot = [], [], [], []
        for i in range(frames):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            z = d.step(act1, lvl1)
            if dec is not None:
                torch.cuda.synchronize(); t1 = time.perf_counter()
                dec(z)
                torch.cuda.synchronize(); de.append((time.perf_counter() - t1) * 1e3)
            torch.cuda.synchronize(); tot.append((time.perf_counter() - t0) * 1e3)
            ctx.append(d.ms.get("ctx", 0)); den.append(d.ms.get("den", 0))
        k = frames // 2  # the window is full and compilation is over in the second half
        f = lambda v: sum(v[k:]) / max(1, len(v[k:]))
        print(f"{name:34s} context {f(ctx):6.1f} ms  denoise {f(den):6.1f} ms  decode {f(de):6.1f} ms  total {f(tot):6.1f} ms "
              f"= {1000 / f(tot):5.1f} frames/s", flush=True)

    nparam = sum(q.numel() for q in m.parameters()) / 1e6
    print(f"model {nparam:.0f}M parameters, window {keep} frames, {a.steps} steps", flush=True)
    for guide in (1.0, 1.5):
        run(f"exact, guide {guide}", FastDream(m, 1, 0.05, a.steps, guide, exact=True, timing=True))
        run(f"ring, guide {guide}", FastDream(m, 1, 0.05, a.steps, guide, timing=True))
        run(f"ring, compiled, guide {guide}", FastDream(m, 1, 0.05, a.steps, guide, timing=True, compile=True), frames=120)
    if a.ae:
        from wm import ae as AE
        for name in a.ae.split(","):
            ae = AE.load(name, dev)
            mean4, std4 = mean, std
            run(f"ring, guide 1.5, decoder {name}", FastDream(m, 1, 0.05, a.steps, 1.5, timing=True),
                dec=lambda z: ae.decode(z * std4 + mean4))
