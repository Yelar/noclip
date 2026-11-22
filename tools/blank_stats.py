"""How much is there to see in a dataset's frames, level by level?

  python tools/blank_stats.py DATA_DIR [CHUNKS] [SHEET.jpg AE]

A frame's content is measured on its latent: the spread of the (normalised) latent over the picture, averaged over
the channels. A picture of one flat thing (a wall in the face, darkness, fog) has little; a view down a corridor has
much. BLANK is set where the emptiest tenth of all frames begins. Per level: the median content, the share of blank
frames, and the share of the model's windows (WINDOW frames) in which every frame is blank: a window that tells the
model nothing about the place. No GPU needed. With SHEET.jpg and an autoencoder (wm/ae.py; a GPU), eight of the
blank frames of each of the six emptiest levels are decoded into one picture, a row per level in the order printed.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm import data as D  # noqa: E402

WINDOW, EVERY = 24, 1
src = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
mean, std = D.load_stats(src)
files = sorted(glob.glob(os.path.join(src, "*.lat.npy")))
rng = np.random.default_rng(0)
files = [files[i] for i in rng.permutation(len(files))[:n]]
names = json.load(open(os.path.join(src, "meta.json")))["levels"] if os.path.exists(os.path.join(src, "meta.json")) else None
cont, lvls, where = [], [], []
for f in files:
    lat = np.load(f, mmap_mode="r")                      # [B, T, C, h, w]
    lvl = np.load(f[:-8] + ".lvl.npy", mmap_mode="r")
    for b in rng.permutation(lat.shape[0])[:12]:         # whole episodes, so that windows can be counted
        x = (np.asarray(lat[b], np.float32) - mean[None, :, None, None]) / std[None, :, None, None]
        cont.append(x.reshape(x.shape[0], x.shape[1], -1).std(2).mean(1))
        lvls.append(np.asarray(lvl[b]).astype(np.int64))
        where.append((lat, int(b)))
allc = np.concatenate(cont)
blank = float(np.quantile(allc, 0.10))
print(f"{src}: {len(files)} chunks, {len(cont)} episodes, {len(allc)} frames; content median {np.median(allc):.3f}, "
      f"blank below {blank:.3f}")
rows = {}
for c, l in zip(cont, lvls):
    b = c < blank
    run = np.zeros(len(b), np.int64)                     # blank frames in a row up to and including this one
    k = 0
    for i, v in enumerate(b):
        k = k + 1 if v else 0
        run[i] = k
    for L in np.unique(l):
        m = l == L
        r = rows.setdefault(int(L), [[], 0, 0, 0])
        r[0].append(c[m]); r[1] += int(b[m].sum()); r[2] += int((run[m] >= WINDOW).sum()); r[3] += int(m.sum())
for L in sorted(rows, key=lambda L: -rows[L][1] / rows[L][3]):
    c, nb, nw, tot = rows[L]
    c = np.concatenate(c)
    name = names[L] if names and L < len(names) else str(L)
    print(f"  {L:2d} {name:11s} {tot:7d} frames  content median {np.median(c):.3f}  blank {100 * nb / tot:5.1f}%  "
          f"whole windows blank {100 * nw / tot:5.1f}%")

if len(sys.argv) > 4:
    import torch
    from PIL import Image

    from wm import ae as AE
    ae = AE.load(sys.argv[4], "cuda", batch=16)
    rows_ = []
    for L in sorted(rows, key=lambda L: -rows[L][1] / rows[L][3])[:6]:
        pick = [(e, i) for e, (c, l) in enumerate(zip(cont, lvls)) for i in np.flatnonzero((c < blank) & (l == L))]
        pick = [pick[i] for i in rng.permutation(len(pick))[:8]]
        z = np.stack([np.asarray(where[e][0][where[e][1], i], np.float32) for e, i in pick])
        with torch.no_grad():
            img = ae.decode(torch.from_numpy(z)).cpu().numpy()[:, ::2, ::2]
        img = np.concatenate(list(img) + [np.zeros_like(img[0])] * (8 - len(img)), 1)
        rows_.append(img)
    Image.fromarray(np.concatenate(rows_, 0)).save(sys.argv[3], quality=88)
    print("->", sys.argv[3])
