"""Running the world model frame after frame: what sampling, evaluation and the play server share."""
import json
import os

import torch

from wm import model as M


def load(path, dev="cuda", cfg=None, stats=None, dtype=None):
    """A checkpoint of wm/task.py (`Task.save`), a released model (model.safetensors with its config.json beside it,
    wm/bundle.py), or a bare state dict (elastic.py eval --save) with cfg and stats given. dtype: bfloat16 on a
    CUDA device unless given, float32 elsewhere."""
    dtype = dtype or (torch.bfloat16 if str(dev).startswith("cuda") else torch.float32)
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        with open(os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")) as f:
            c = json.load(f)
        cfg, sd, stats = M.Cfg(**c["model"]), load_file(path), (c["mean"], c["std"])
    else:
        ck = torch.load(path, map_location="cpu")
        if "model" in ck and "cfg" in ck:
            cfg, sd, stats = M.Cfg(**ck["cfg"]), ck["model"], (ck["mean"], ck["std"])
        else:
            sd = ck
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m = M.build(cfg).to(dev, dtype).eval().requires_grad_(False)
    m.load_state_dict(sd)
    mean = torch.tensor(stats[0], device=dev, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(stats[1], device=dev, dtype=torch.float32)[None, :, None, None]
    return m, mean, std


class Dream:
    """B parallel dreams. Frames live here as normalised latents; `seed` adds known frames, `step` draws the next.

    The context is the last `frames - 1` frames, each noised once to level `tau` (the model was trained on lightly
    noised context; a little of it at play time keeps a long dream from drifting). The context's pass is redone
    for every frame, which is exactly the training computation (one pass over K frames, then `steps` passes over one).
    """

    def __init__(self, model, B, tau=0.05, steps=8, guide=1.0, shift=1.0, keep=None, seed=0):
        self.m, self.B, self.tau, self.steps, self.guide, self.shift = model, B, tau, steps, guide, shift
        self.keep = keep or model.cfg.frames - 1
        self.dev, self.dt = model.pos.device, model.pos.dtype
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        self.lat, self.act, self.lvl = [], [], []  # per frame: [B, C, h, w] noised context copy, [B, A], [B]

    def _push(self, z, act, lvl):
        eps = torch.randn(z.shape, device=self.dev, generator=self.g).to(self.dt)
        self.lat.append(((1 - self.tau) * z.to(self.dt) + self.tau * eps))
        self.act.append(act.to(self.dt)); self.lvl.append(lvl)
        for q in (self.lat, self.act, self.lvl):
            del q[:-self.keep]

    def seed(self, z, act, lvl):
        """z [B, K, C, h, w] normalised latents of known frames with their actions [B, K, A] and levels [B, K]."""
        for i in range(z.shape[1]):
            self._push(z[:, i], act[:, i], lvl[:, i])

    @torch.no_grad()
    def step(self, act, lvl):
        """act [B, A], lvl [B] -> the next frame's normalised latent [B, C, h, w] (float32)."""
        K = len(self.lat)
        cache = None
        if K:
            tau = torch.full((self.B, K), self.tau, device=self.dev)
            cache = self.m.context(torch.stack(self.lat, 1), torch.stack(self.act, 1), torch.stack(self.lvl, 1), tau)
        noise = torch.randn((self.B, self.m.cfg.C, self.m.cfg.h, self.m.cfg.w), device=self.dev, generator=self.g)
        z = self.m.next_frame(cache, act.to(self.dt), lvl, K, self.steps, self.guide, self.shift, noise)
        self._push(z, act, lvl)
        return z.float()
