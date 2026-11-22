"""See what the world model has learned.

  python -m wm.sample --ckpt RUN/model.pt --data VAL_DIR --out DIR [--n 8 --ctx 8 --len 96 --steps 8 --ae dcae]

For n recorded episodes it shows the model the first `ctx` frames and then only the recorded actions, and lets it
dream `len` frames. Written: for each episode a video with the simulator's frames on top and the dream below, a
contact sheet, and in report.json how far the dream is from the simulator's frames at each step ahead (the dream
is not expected to stay equal for long: unseen places are invented; the first second says whether actions work).
With --free it also dreams each level from nothing while walking a fixed route.

A model with memory (wm/memory.py) dreams one episode at a time with the play engine, recalling frames it drew
earlier in the dream by the recorded poses; and once more without recalling, so that the report says what the
memory is worth on the frames that look at a place seen before (`revisit`).

`blank_last30` says whether dreams fall apart into pictures with nothing in them (a light on a bare wall, darkness):
the share of blank frames in the dreams' last 3 s against the share in the recorded frames of the same moments. A
frame is blank if its latent varies less over the picture than nine tenths of these episodes' recorded frames do
(tools/blank_stats.py has the same measure for a dataset). `by_level` has it per level (the one an episode's dream
starts in), with the PSNR and the contrast of the dreamed latents relative to the recorded ones. PSNR 12 s ahead
moves by more than 1 dB between settings on 8 episodes: for comparisons use --n 32 --video 0 --mem_cmp 0.
"""
import argparse
import json
import os

import numpy as np
import torch

from wm import ae as AE
from wm import data as D
from wm import roll


def video(path, frames, fps=10):
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)


def dream_mem(m, lat, act, lvl, aux, a, use):
    """-> the dreamed latents [n, T, C, h, w], and per frame how many earlier frames could be recalled for it."""
    from wm import fast
    from wm import memory as MEM
    outs, hits = [], []
    for b in range(lat.shape[0]):
        fd = fast.FastDream(m, 1, a.tau, a.steps, a.guide, a.shift, seed=a.seed, solver=a.solver)
        bank = MEM.Bank(m.cfg.mem, period=MEM.G)
        out, hit = [], []
        for i in range(a.ctx + a.len):
            pose = tuple(float(v) for v in aux[b, i, :4])
            if i and aux[b, i, 14] != aux[b, i - 1, 14]:        # the recorded player fell into another map
                bank.clear()
            got = bank.recall(pose, i, fd.keep + 1)
            hit.append(len(got))
            fd.set_memory([(fid, z_, l_) for fid, (z_, l_) in got] if use else [])
            if i < a.ctx:
                z = lat[b:b + 1, i]
                fd._push(z, act[b:b + 1, i], lvl[b:b + 1, i])
            else:
                z = fd.step(act[b:b + 1, i], lvl[b:b + 1, i])
            bank.add(i, pose, (z.clone(), lvl[b:b + 1, i]))
            out.append(z[0])
        outs.append(torch.stack(out)); hits.append(hit)
    return torch.stack(outs), np.array(hits)


def route(n, A, dev):
    """A walk that uses every control: forward, turns, a strafe, a look up and down, a stop."""
    act = torch.zeros(n, A, device=dev)
    for i in range(n):
        ph = (i // 12) % 8
        act[i, 0] = 1.0 if ph in (0, 1, 3, 5, 6) else (-1.0 if ph == 7 else 0.0)
        act[i, 2] = {1: 0.35, 2: -0.5, 5: -0.3}.get(ph, 0.0)
        act[i, 1] = 1.0 if ph == 4 else 0.0
        act[i, 3] = {2: 0.3, 3: -0.3}.get(ph, 0.0)
    return act


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ae", default="dcae")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--ctx", type=int, default=8)
    p.add_argument("--len", type=int, default=96)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--solver", default="euler", help="euler or ab2 (wm/fast.py): with --ring 1 or a model with memory")
    p.add_argument("--guide", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--shift", type=float, default=1.0)
    p.add_argument("--free", type=int, default=0, help="also dream this many frames of every level from nothing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", type=int, default=1, help="0: no videos, only the sheet and the report")
    p.add_argument("--mem_cmp", type=int, default=1, help="0: do not dream a second time without the memory")
    p.add_argument("--ring", type=int, default=0, help="1: dream with the play server's cached context (wm/fast.py) instead of "
                   "the exact computation, to see what that shortcut costs")
    a = p.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    m, mean, std = roll.load(a.ckpt, dev)
    ae = AE.load(a.ae, dev, batch=8)
    ch = D.Chunks(a.data, a.ctx + a.len, A=m.cfg.A)
    ch.want_aux = True
    lat, act, lvl, aux = ch.fixed(a.n, 1, seed=a.seed + 7)[0]
    lat = (torch.from_numpy(lat).to(dev).float() - mean[:, None]) / std[:, None]
    act, lvl = torch.from_numpy(act).to(dev), torch.from_numpy(lvl.astype(np.int64)).to(dev)
    mem = getattr(m.cfg, "mem", 0)
    out0 = hits = None
    if mem:
        out, hits = dream_mem(m, lat, act, lvl, aux, a, True)
        out0 = dream_mem(m, lat, act, lvl, aux, a, False)[0] if a.mem_cmp else None
    elif a.ring:
        from wm import fast
        dr = fast.FastDream(m, a.n, a.tau, a.steps, a.guide, a.shift, exact=False, compile=False, solver=a.solver)
    else:
        dr = roll.Dream(m, a.n, a.tau, a.steps, a.guide, a.shift, seed=a.seed)
    if not mem:
        dr.seed(lat[:, :a.ctx], act[:, :a.ctx], lvl[:, :a.ctx])
        out = [lat[:, i] for i in range(a.ctx)]
        for i in range(a.ctx, a.ctx + a.len):
            out.append(dr.step(act[:, i], lvl[:, i]))
        out = torch.stack(out, 1)
    err = ((out - lat) ** 2).mean((2, 3, 4)).mean(0)[a.ctx:]  # per step ahead, in normalised latent units (1 = noise)
    # is the dream still a picture? contrast of the dreamed latents relative to the recorded ones (1 = as lively;
    # towards 0 = faded to a flat frame), and how far their overall level has wandered from the recorded frames'
    live_ep = (out.std((2, 3, 4)) / lat.std((2, 3, 4)).clamp_min(1e-3))[:, a.ctx:]       # [n, len]
    live = live_ep.mean(0)
    cont = lambda z: z.flatten(3).std(3).mean(2)                                         # [n, T]: how much a picture holds
    c_rec, c_out = cont(lat), cont(out)
    blank = torch.quantile(c_rec.flatten(), 0.10)
    b_rec, b_out = (c_rec < blank)[:, -30:].float(), (c_out < blank)[:, -30:].float()    # [n, 30]
    wander = (out.mean((3, 4)) - lat.mean((3, 4))).abs().mean((0, 2))[a.ctx:]
    report = {"ckpt": a.ckpt, "steps": a.steps, "guide": a.guide, "tau": a.tau,
              "contrast_first10": round(float(live[:10].mean()), 3), "contrast_last10": round(float(live[-10:].mean()), 3),
              "level_shift_first10": round(float(wander[:10].mean()), 3), "level_shift_last10": round(float(wander[-10:].mean()), 3),
              "latent_mse_by_step": [round(float(e), 4) for e in err],
              "latent_mse_first10": round(float(err[:10].mean()), 4), "latent_mse_last10": round(float(err[-10:].mean()), 4),
              "blank_last30": {"recorded": round(float(b_rec.mean()), 3), "dreamed": round(float(b_out.mean()), 3)}}
    psnr, psnr0, sheet = [], [], []
    for b in range(a.n):
        real = ae.decode(lat[b] * std + mean)
        fake = ae.decode(out[b] * std + mean)
        mse = ((real.float() - fake.float()) ** 2).mean((1, 2, 3))[a.ctx:]
        psnr.append((10 * torch.log10(255 ** 2 / mse.clamp_min(1e-3))).cpu())
        if out0 is not None:
            mse0 = ((real.float() - ae.decode(out0[b] * std + mean).float()) ** 2).mean((1, 2, 3))[a.ctx:]
            psnr0.append((10 * torch.log10(255 ** 2 / mse0.clamp_min(1e-3))).cpu())
        both = torch.cat((real, fake), 1).cpu().numpy()
        if a.video:
            video(os.path.join(a.out, f"ep{b}-level{int(lvl[b, a.ctx])}.mp4"), both)
        sheet.append(np.concatenate([both[i][::2, ::2] for i in range(a.ctx - 1, a.ctx + a.len, max(1, a.len // 6))][:7], 1))
    ps = torch.stack(psnr).mean(0)
    try:
        with open(os.path.join(a.data.split(",")[0], "meta.json")) as f:
            lnames = json.load(f).get("levels", [])
    except OSError:
        lnames = []
    by = {}
    for b in range(a.n):
        L = int(lvl[b, a.ctx])
        by.setdefault(lnames[L] if L < len(lnames) else str(L), []).append(
            (float(psnr[b][-30:].mean()), float(live_ep[b, -30:].mean()), float(b_rec[b].mean()), float(b_out[b].mean())))
    col = lambda v, i: round(float(np.mean([x[i] for x in v])), 3)
    report["by_level"] = {k: {"episodes": len(v), "psnr_last30": round(col(v, 0), 2), "contrast_last30": col(v, 1),
                              "blank_recorded": col(v, 2), "blank_dreamed": col(v, 3)}
                          for k, v in sorted(by.items(), key=lambda kv: -(col(kv[1], 3) - col(kv[1], 2)))}
    report["psnr_by_step"] = [round(float(x), 2) for x in ps]
    report["psnr_first10"], report["psnr_last10"] = round(float(ps[:10].mean()), 2), round(float(ps[-10:].mean()), 2)
    if out0 is not None:        # what recalling is worth: all frames, and those that look at a place seen before
        pa, pb, rv = torch.stack(psnr), torch.stack(psnr0), torch.from_numpy(hits[:, a.ctx:] > 0)
        report["memory"] = {"psnr": round(float(pa.mean()), 2), "psnr_without": round(float(pb.mean()), 2),
                            "revisit_share": round(float(rv.float().mean()), 3),
                            "psnr_revisit": round(float(pa[rv].mean()), 2) if rv.any() else None,
                            "psnr_revisit_without": round(float(pb[rv].mean()), 2) if rv.any() else None,
                            "psnr_last10_without": round(float(pb.mean(0)[-10:].mean()), 2)}
    from PIL import Image
    Image.fromarray(np.concatenate(sheet, 0)).save(os.path.join(a.out, "sheet.jpg"), quality=85)
    if a.free:
        with open(os.path.join(a.data.split(",")[0], "meta.json")) as f:
            names = json.load(f).get("levels", [])
        ids = sorted(set(int(x) for x in np.unique(np.concatenate([np.unique(c[2]) for c in ch.files[:16]]))))
        r = route(a.free, m.cfg.A, dev)
        dr = roll.Dream(m, len(ids), a.tau, a.steps, max(a.guide, 1.5), a.shift, seed=a.seed + 1)
        lv = torch.tensor(ids, device=dev)
        zs = torch.stack([dr.step(r[i][None].expand(len(ids), -1), lv) for i in range(a.free)], 1)
        rows = []
        for j, i in enumerate(ids):
            fr = ae.decode(zs[j] * std + mean).cpu().numpy()
            name = (names[i] if i < len(names) else str(i)).replace(" ", "_").replace("/", "-")
            video(os.path.join(a.out, f"free-{i:02d}-{name}.mp4"), fr)
            rows.append(np.concatenate([fr[k][::2, ::2] for k in range(0, a.free, max(1, a.free // 7))][:7], 1))
        Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(a.out, "free.jpg"), quality=85)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("by_step")}))
