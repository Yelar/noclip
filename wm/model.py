"""The world model: a transformer that draws the next frame (in the autoencoder's latent space) from the frames so
far, the player's actions and the level.

How it is trained is how it is played. A training window of F frames enters twice: as *context* (the frames as they
are, lightly noised so that the model tolerates its own imperfect frames later) and as *targets* (each frame mixed
with noise at a random level t). Target i sees context 0..i-1 and itself, and predicts the velocity that turns
noise into frame i (flow matching). So every frame of the window is a prediction from clean history, in one pass,
and frame 0 is a prediction from nothing: the model can also invent a level's first frame.

Attention is factorised: within a frame (spatial), and across frames at the same grid position (temporal, causal,
relative positions by rotation). Playing therefore costs one frame's tokens per denoising step: the context's
temporal keys and values are computed once per frame and reused.

Per frame the model is told: the noise level, which stream it is, the action that led to the frame, and the level.

Memory (cfg.mem > 0): besides its window the model can be shown a few frames from longer ago that were seen from
about the same place (wm/memory.py picks them). They are a third stream, context only: each is taken as it is
(it attends to no other frame), and every frame of the window may attend to all of them. Their frame positions lie
before the window's, cfg.mem slots MEM_STRIDE apart.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    C: int = 32          # latent channels
    h: int = 9           # latent grid
    w: int = 16
    patch: int = 1
    d: int = 768
    depth: int = 16
    heads: int = 12
    mlp: int = 4
    A: int = 7           # action columns
    levels: int = 32     # level ids 0..levels-1; id `levels` is "not told"
    frames: int = 32     # frames per training window
    mem: int = 0         # memory frames shown besides the window (0: the model has no memory stream)

    @property
    def N(self):
        return (self.h // self.patch) * (self.w // self.patch)


MEM_STRIDE = 4           # frame positions between two memory slots; the window starts at cfg.mem * MEM_STRIDE


def _rot(x, pos):
    """Rotate pairs of channels of x [M, heads, S, dh] by angles proportional to pos [S] (relative positions)."""
    dh = x.shape[-1]
    freq = 1.0 / (200.0 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    ang = pos.to(torch.float32)[:, None] * freq[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)


def _norm(x):
    """Scale each token to unit root-mean-square, in the computation's own precision (LayerNorm would switch to
    float32 under autocast: more memory and time for nothing here)."""
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, pos=None, mask=None, cache=None, store=False):
        """x [M, S, d]. pos: frame index per slot (temporal attention). cache: (k, v) of earlier frames to attend
        to besides x itself. store: also return this call's (k, v), with the cache's in front."""
        M, S, d = x.shape
        q, k, v = self.qkv(x).view(M, S, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if pos is not None:
            q, k = _rot(q, pos), _rot(k, pos)
        if cache is not None:
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = self.out(y.transpose(1, 2).reshape(M, S, d))
        return (y, (k, v)) if store else y


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ada = nn.Linear(c.d, 6 * c.d)
        self.spatial, self.temporal = Attn(c.d, c.heads), Attn(c.d, c.heads)
        self.mlp = nn.Sequential(nn.Linear(c.d, c.mlp * c.d), nn.GELU(approximate="tanh"), nn.Linear(c.mlp * c.d, c.d))
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond, pos, mask=None, cache=None, store=False):
        """x [B, S, N, d], cond [B, S, d]. Returns x (and the temporal keys and values when store)."""
        B, S, N, d = x.shape
        s1, b1, s2, b2, s3, b3 = self.ada(cond)[:, :, None].chunk(6, -1)
        x = x + self.spatial((_norm(x) * (1 + s1) + b1).view(B * S, N, d)).view(B, S, N, d)
        hx = (_norm(x) * (1 + s2) + b2).transpose(1, 2).reshape(B * N, S, d)
        y = self.temporal(hx, pos, mask, cache, store)
        y, kv = y if store else (y, None)
        x = x + y.view(B, N, S, d).transpose(1, 2)
        x = x + self.mlp(_norm(x) * (1 + s3) + b3)
        return (x, kv) if store else x


class WM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = c = cfg
        self.inp = nn.Linear(c.C * c.patch ** 2, c.d)
        self.pos = nn.Parameter(torch.randn(c.N, c.d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, c.d), nn.SiLU(), nn.Linear(c.d, c.d))
        self.a_mlp = nn.Sequential(nn.Linear(c.A, c.d), nn.SiLU(), nn.Linear(c.d, c.d))
        self.lvl = nn.Embedding(c.levels + 1, c.d)
        self.stream = nn.Embedding(3 if c.mem else 2, c.d)  # 0 context, 1 target, 2 memory
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.depth))
        self.ada = nn.Linear(c.d, 2 * c.d)
        self.out = nn.Linear(c.d, c.C * c.patch ** 2)
        for m in (self.ada, self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    # ---- pieces
    def _tok(self, z):
        """[B, S, C, h, w] -> [B, S, N, C*p*p]"""
        c, p = self.cfg, self.cfg.patch
        B, S = z.shape[:2]
        z = z.view(B, S, c.C, c.h // p, p, c.w // p, p).permute(0, 1, 3, 5, 2, 4, 6)
        return z.reshape(B, S, c.N, c.C * p * p)

    def _untok(self, x):
        c, p = self.cfg, self.cfg.patch
        B, S = x.shape[:2]
        x = x.view(B, S, c.h // p, c.w // p, c.C, p, p).permute(0, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, S, c.C, c.h, c.w)

    def _cond(self, t, act, lvl, stream):
        """noise level [B, S], actions [B, S, A], level ids [B, S], stream id (int) -> [B, S, d]"""
        half = 128
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        ang = t.float()[..., None] * 1000.0 * f
        te = torch.cat((ang.cos(), ang.sin()), -1).to(self.inp.weight.dtype)
        return self.t_mlp(te) + self.a_mlp(act.to(te.dtype)) + self.lvl(lvl) + self.stream.weight[stream]

    def _head(self, x, cond):
        s, b = self.ada(cond)[:, :, None].chunk(2, -1)
        return self._untok(self.out(_norm(x) * (1 + s) + b))

    # ---- training
    def forward(self, lat, act, lvl, t, tau, eps=None, eps_c=None, ctx=None, want="loss", mem=None):
        """lat [B, F, C, h, w] clean latents, act [B, F, A], lvl [B, F], t and tau [B, F] in [0, 1]: noise levels of the
        targets and of the context copies. Returns the loss per frame [B, F] (mean squared error of the velocity).

        mem: (latents [B, R, C, h, w] already noised to their levels [B, R], levels [B, R], which slots are filled
        [B, R] bool, those noise levels): the memory frames.

        ctx: frames to show as context instead of lat (the targets stay lat): the model's own estimates of the
        frames, so that it learns to continue from what it draws itself, not only from perfect frames.
        want="x0": return those estimates instead of the loss, the clean frames the model infers from the noisy ones."""
        B, Fr = lat.shape[:2]
        eps = torch.randn_like(lat) if eps is None else eps
        eps_c = torch.randn_like(lat) if eps_c is None else eps_c
        tt, tc = t[:, :, None, None, None].to(lat.dtype), tau[:, :, None, None, None].to(lat.dtype)
        noisy = (1 - tt) * lat + tt * eps
        x = torch.cat(((1 - tc) * (lat if ctx is None else ctx) + tc * eps_c, noisy), 1)
        cond = torch.cat((self._cond(tau, act, lvl, 0), self._cond(t, act, lvl, 1)), 1)
        x = self.inp(self._tok(x)) + self.pos
        idx = torch.arange(Fr, device=lat.device)
        pos = torch.cat((idx, idx))
        low = idx[:, None] >= idx[None]                       # context i sees context j <= i
        strict = idx[:, None] > idx[None]                     # target i sees context j < i
        eye = idx[:, None] == idx[None]                       # ... and itself
        mask = torch.cat((torch.cat((low, torch.zeros_like(low)), 1), torch.cat((strict, eye), 1)), 0)
        R = 0
        if mem is not None:
            mlat, mlvl, mok, mtau = mem
            R = mlat.shape[1]
            x = torch.cat((self.inp(self._tok(mlat.to(x.dtype))) + self.pos, x), 1)
            cond = torch.cat((self._cond(mtau, act.new_zeros(B, R, act.shape[2]), mlvl, 2), cond), 1)
            pos = torch.cat((torch.arange(R, device=lat.device) * MEM_STRIDE, pos + R * MEM_STRIDE))
            S = R + 2 * Fr
            full = torch.zeros(S, S, dtype=torch.bool, device=lat.device)
            full[R:, R:] = mask
            full = full[None].repeat(B, 1, 1)
            full[:, R:, :R] = mok[:, None, :]                 # the window's frames see the filled memory slots
            full[:, :R, :R] = torch.eye(R, dtype=torch.bool, device=lat.device)   # a memory frame sees only itself
            N = self.cfg.N
            mask = full[:, None, None].expand(B, N, 1, S, S).reshape(B * N, 1, S, S)
        for blk in self.blocks:
            x = blk(x, cond, pos, mask)
        v = self._head(x[:, R + Fr:], cond[:, R + Fr:])
        if want == "x0":
            return noisy.float() - tt.float() * v.float()
        return ((v.float() - (eps - lat).float()) ** 2).mean((2, 3, 4))

    # ---- playing
    @torch.no_grad()
    def context(self, lat, act, lvl, tau, first=0):
        """The keys and values that later frames attend to, for context frames lat [B, K, C, h, w] (already noised
        to level tau [B, K]) at frame positions first .. first+K-1. One entry per block."""
        B, K = lat.shape[:2]
        if K == 0:
            return None
        cond = self._cond(tau, act, lvl, 0)
        x = self.inp(self._tok(lat)) + self.pos
        idx = torch.arange(K, device=lat.device)
        mask, cache = idx[:, None] >= idx[None], []
        for blk in self.blocks:
            x, kv = blk(x, cond, idx + first, mask, store=True)
            cache.append(kv)
        return cache

    @torch.no_grad()
    def velocity(self, x, t, act, lvl, cache, pos):
        """One denoising evaluation for the frame at position pos: x [B, C, h, w] at noise level t [B]."""
        cond = self._cond(t[:, None], act[:, None], lvl[:, None], 1)
        h = self.inp(self._tok(x[:, None])) + self.pos
        p = torch.full((1,), pos, device=x.device)
        for i, blk in enumerate(self.blocks):
            h = blk(h, cond, p, cache=None if cache is None else cache[i])
        return self._head(h, cond)[:, 0]

    @torch.no_grad()
    def next_frame(self, cache, act, lvl, pos, steps=8, guide=1.0, shift=1.0, noise=None):
        """Draw the frame at position pos given the context's cache: Euler steps from noise (t=1) to the frame (t=0).
        guide > 1 pushes towards the named level (classifier-free guidance against "not told")."""
        c = self.cfg
        B = act.shape[0]
        dt = self.inp.weight.dtype
        x = (torch.randn(B, c.C, c.h, c.w, device=act.device) if noise is None else noise).float()  # the sum stays in float32
        s = torch.linspace(1, 0, steps + 1, device=act.device)
        ts = shift * s / (1 + (shift - 1) * s)
        null = torch.full_like(lvl, c.levels)
        for i in range(steps):
            t = ts[i].expand(B)
            v = self.velocity(x.to(dt), t, act, lvl, cache, pos).float()
            if guide != 1.0:
                v0 = self.velocity(x.to(dt), t, act, null, cache, pos).float()
                v = v0 + guide * (v - v0)
            x = x + (ts[i + 1] - ts[i]) * v
        return x


def build(cfg):
    return WM(cfg if isinstance(cfg, Cfg) else Cfg(**cfg))


def boost_actions(w, share, seed=0):
    """First-layer weights of the action network [d, A] in which every column the model cannot be responding to
    (under a tenth of the usual column's norm) is given `share` of the usual norm: its own direction if it has one,
    a random one if it is new. Why: a column that starts at zero stays negligible. `use` and `presence` were added
    at zero and after 40M frames had a norm of 0.11 against the keys' 6 to 10; changing them changed the inferred
    frame by 0.007, which is what running the model twice on the same input gives (tools/act_gain.py). At 8e-5 a
    weight moves by at most that much per step, and such a column's gradient is small and noisy."""
    n = w.float().norm(dim=0)
    usual = n[n > 0.1 * n.max()].median()
    g = torch.Generator().manual_seed(seed)
    out = w.float().clone()
    for j in (n < 0.1 * usual).nonzero()[:, 0].tolist():
        v = out[:, j] if n[j] > 1e-3 * usual else torch.randn(w.shape[0], generator=g).to(out.device)
        out[:, j] = v / v.norm() * share * usual
    return out.to(w.dtype)


def upgrade(sd, model, act_boost=0.0):
    """The weights of an older model made to fit `model`: new action columns start at zero (nothing changes for the
    model), or, with act_boost, at that share of the usual column's size (boost_actions: at zero they are not
    learned); the memory stream starts as a copy of the context stream."""
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    w, A = sd["a_mlp.0.weight"], model.cfg.A
    if w.shape[1] < A:
        sd["a_mlp.0.weight"] = torch.cat((w, w.new_zeros(w.shape[0], A - w.shape[1])), 1)
    if act_boost > 0:
        sd["a_mlp.0.weight"] = boost_actions(sd["a_mlp.0.weight"], act_boost)
    e, n = sd["stream.weight"], model.stream.weight.shape[0]
    if e.shape[0] < n:
        sd["stream.weight"] = torch.cat((e, e[:1].expand(n - e.shape[0], -1)), 0)
    return sd


SIZES = {  # name: (d, depth, heads)
    "xs": (384, 8, 6), "s": (512, 12, 8), "b": (768, 16, 12), "l": (1024, 20, 16),
}

if __name__ == "__main__":  # shapes, causality, and that playing with a cache equals the training pass
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = Cfg(d=128, depth=3, heads=4, frames=6)
    m = build(c).to(dev)
    for p in m.parameters():  # the zero-initialised parts would hide everything
        if p.abs().sum() == 0:
            nn.init.normal_(p, std=0.05)
    B, Fr = 2, c.frames
    lat = torch.randn(B, Fr, c.C, c.h, c.w, device=dev)
    act = torch.randn(B, Fr, c.A, device=dev)
    lvl = torch.randint(0, 5, (B, Fr), device=dev)
    t, tau = torch.rand(B, Fr, device=dev), torch.rand(B, Fr, device=dev) * 0.2
    eps, eps_c = torch.randn_like(lat), torch.randn_like(lat)
    loss = m(lat, act, lvl, t, tau, eps, eps_c)
    # causality: changing frame 4 must not change the loss of frames 0..3
    lat2 = lat.clone(); lat2[:, 4] += 1.0
    loss2 = m(lat2, act, lvl, t, tau, eps, eps_c)
    assert torch.allclose(loss[:, :4], loss2[:, :4], atol=1e-5) and not torch.allclose(loss[:, 4:], loss2[:, 4:], atol=1e-5)
    # the cached path gives the training pass's velocity for target 5
    tc = tau[:, :, None, None, None]
    ctx = ((1 - tc) * lat + tc * eps_c)[:, :5]
    cache = m.context(ctx, act[:, :5], lvl[:, :5], tau[:, :5])
    tt = t[:, 5, None, None, None]
    v = m.velocity((1 - tt) * lat[:, 5] + tt * eps[:, 5], t[:, 5], act[:, 5], lvl[:, 5], cache, 5)
    ref = ((v - (eps - lat)[:, 5]) ** 2).mean((1, 2, 3))
    assert torch.allclose(ref, loss[:, 5], atol=1e-4), (ref, loss[:, 5])
    x = m.next_frame(cache, act[:, 5], lvl[:, 5], 5, steps=4, guide=1.5)
    print("ok", tuple(x.shape), "params", sum(p.numel() for p in m.parameters()))
    for name, (d, depth, heads) in SIZES.items():
        n = sum(p.numel() for p in build(Cfg(d=d, depth=depth, heads=heads)).parameters())
        print(f"{name}: {n / 1e6:.0f}M parameters")
