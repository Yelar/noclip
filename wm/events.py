"""Reading game events off the frames the model draws: was the player just struck, has the player died?

The model draws the blow (a face over the picture) and the death ("YOU DIED") as it saw the simulator do; nothing
else tells the play server that they happened. A small network reads them off a frame's latent, so that the server
can play the sounds and forget the place the player died in. It is trained on a dataset's latents and the
simulator's record of them (aux: `scream` counts down during a blow's frames, `dying` is set during a death's).
"Dead" means the frames that show the words: a death's first frames show the same face as any blow, and its last
ones the place the player wakes up in, so those count neither for nor against it.

  python -m wm.events DATA_DIR [OUT.pt] [CHUNKS]     trains it (a minute on a GPU) -> DATA_DIR/events.pt
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

EVENTS = ("hit", "dead")
AUX_DYING, AUX_SCREAM = 11, 17      # columns of aux (sim/core.py)


class Reader(nn.Module):
    def __init__(self, C=32, d=96):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(C, d, 3, padding=1), nn.SiLU(), nn.Conv2d(d, d, 3, padding=1), nn.SiLU())
        self.out = nn.Linear(2 * d, len(EVENTS))

    def forward(self, z):
        """z [B, C, h, w] normalised latents -> logits [B, 2] (struck in this frame, dying in this frame)"""
        x = self.net(z)
        return self.out(torch.cat((x.mean((2, 3)), x.amax((2, 3))), 1))


def load(path, dev="cuda"):
    """The reader from a file this module wrote, or from a released events.safetensors (wm/bundle.py)."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path, device=str(dev))
        C = sd["net.0.weight"].shape[1]
    else:
        ck = torch.load(path, map_location=dev)
        sd, C = ck["model"], ck["C"]
    m = Reader(C).to(dev).eval().requires_grad_(False)
    m.load_state_dict(sd)
    return m


WORDS = (8, 16)      # the frames of a death (counted from 1) that surely show "YOU DIED" (sim/core.py: _overlay)


def labels(aux):
    """aux [B, T, .] -> labels [B*T, 2] (a blow's frame, a frame with the words) and which of them count [B*T, 2]"""
    hit, dying = np.asarray(aux[..., AUX_SCREAM]) > 0, np.asarray(aux[..., AUX_DYING]) > 0
    c = np.cumsum(dying, 1)
    k = c - np.maximum.accumulate(np.where(dying, 0, c), 1)         # 1, 2, ... along each death's frames
    cut = dying & (np.minimum.accumulate(dying, 1))                 # a death that began before the chunk did
    dead = dying & (k >= WORDS[0]) & (k <= WORDS[1]) & ~cut
    y = np.stack((hit, dead), -1).reshape(-1, 2)
    w = np.stack((np.ones_like(hit), dead | ~dying), -1).reshape(-1, 2)
    return y, w


def gather(files, rng, per=400):
    """From each chunk: the frames of blows and deaths, and as many others.
    -> latents [n, C, h, w], labels [n, 2], which labels count [n, 2]"""
    xs, ys, ws = [], [], []
    for f in files:
        y, w = labels(np.load(f[:-8] + ".aux.npy", mmap_mode="r"))
        pos = np.flatnonzero(y.any(1))
        pos = rng.permutation(pos)[:per]
        idx = np.unique(np.concatenate((pos, rng.integers(0, len(y), max(per // 2, len(pos))))))
        lat = np.load(f, mmap_mode="r")
        lat = lat.reshape(-1, *lat.shape[2:])
        xs.append(np.asarray(lat[idx], np.float32)); ys.append(y[idx]); ws.append(w[idx])
    return np.concatenate(xs), np.concatenate(ys).astype(np.float32), np.concatenate(ws).astype(np.float32)


if __name__ == "__main__":
    import glob

    from wm import data as D
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else os.path.join(src, "events.pt")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(src, "*.lat.npy")))
    files = [files[i] for i in rng.permutation(len(files))[:n]]
    hold = max(1, len(files) // 6)
    mean, std = (torch.tensor(v, device=dev)[None, :, None, None] for v in D.load_stats(src))
    prep = lambda a: (torch.from_numpy(a).to(dev) - mean) / std
    (xt, yt, wt), (xv, yv, wv) = gather(files[hold:], rng), gather(files[:hold], rng)
    xt, yt, wt, xv, yv, wv = prep(xt), *(torch.from_numpy(v).to(dev) for v in (yt, wt)), prep(xv), *(torch.from_numpy(v).to(dev) for v in (yv, wv))
    print(f"{len(files)} chunks: {len(xt)} frames to learn from ({int(yt[:, 0].sum())} of a blow, {int(yt[:, 1].sum())} with the words), "
          f"{len(xv)} held out", flush=True)
    torch.manual_seed(0)
    m = Reader(xt.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = 3000
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = 2e-3 * 0.5 * (1 + np.cos(np.pi * i / steps))
        j = torch.randint(0, len(xt), (256,), device=dev)
        x = xt[j]
        # the model's own frames are not as clean as the simulator's: noise of several strengths, as in its context
        x = x + torch.rand(256, 1, 1, 1, device=dev) * 0.4 * torch.randn_like(x)
        loss = nn.functional.binary_cross_entropy_with_logits(m(x), yt[j], weight=wt[j])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        res = {}
        for noise in (0.0, 0.3):
            p = torch.sigmoid(m(xv + noise * torch.randn_like(xv))) > 0.5
            for k, name in enumerate(EVENTS):
                on, off = (yv[:, k] > 0) & (wv[:, k] > 0), (yv[:, k] == 0) & (wv[:, k] > 0)
                tp, fp, fn = (p[:, k] & on).sum().item(), (p[:, k] & off).sum().item(), (~p[:, k] & on).sum().item()
                res[f"{name}, noise {noise}"] = {"found": round(tp / max(1, tp + fn), 3), "false alarms per 1000 other frames": round(1000 * fp / max(1, int(off.sum())), 2)}
    print(json.dumps(res, indent=1))
    torch.save({"model": m.state_dict(), "C": xt.shape[1], "events": EVENTS, "data": src, "held_out": res}, out + ".tmp")
    os.replace(out + ".tmp", out)
    print("->", out)
