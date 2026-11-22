"""Write datasets with the simulator, or pictures to judge it by.

  python -m sim.gen --out DIR --name NAME --seed S [--envs 64] [--chunk 256] [--chunks N] [--rgb] [--ae dcae]
  python -m sim.gen --preview DIR [--levels 0,5] [--steps 240]      contact sheets and a video per level
  python -m sim.gen --bench [--envs 64]                             frames per second

A chunk is B parallel episodes x T frames (see the README for the files). The episodes go on from chunk to chunk.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import ACTIONS, AUX, LEVELS, PITCH_MAX, TITLES, TURN_MAX, Explorer, Sim
from .levels import CALM
from .core import BACK, SPRINT, STRAFE, WALK

VERSION = 5            # 5: blackout and red-light columns; the scripted player turns away from walls (4: no health bar)
PRESENCE_M = 20.0     # the `presence` column: 1 while something is within this many metres of the player
# Columns after the keys: what the game decides, not the player. The play server's director sets them (wm/play.py).
STATE = ["presence", "blackout", "red"]


def _save(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _meta(a, ae, lat_shape):
    p = os.path.join(a.out, "meta.json")
    if os.path.exists(p):
        return
    from . import levels as LV
    maps = [LV.make(i, 0) for i in range(len(LEVELS))]      # which levels have blackouts and red light at all
    m = dict(version=VERSION, blackouts=[i for i, mp in enumerate(maps) if mp.wave_rate > 0],
             reds=[i for i, mp in enumerate(maps) if mp.red_rate > 0],      # (the play server's director asks)
             fps=a.fps, res=[a.height, a.width], actions=ACTIONS + STATE, levels=LEVELS, presence_m=PRESENCE_M,
             move=[WALK, SPRINT, BACK, STRAFE], calm=[LEVELS.index(n) for n in CALM],
             level_titles=[TITLES[l] for l in LEVELS], aux=AUX, turn_max=TURN_MAX, pitch_max=PITCH_MAX,
             ae=ae, latent=lat_shape, envs=a.envs, chunk=a.chunk,
             note="act[t] turns frame t-1 into frame t; lvl[t] is the level the player is in or falling into; the last "
                  "three action columns are not keys: an entity is near, a blackout is on (from the first lamp going "
                  "out until they come back), the lamps are red (the game's director sets them when playing)")
    with open(p + f".{os.getpid()}.tmp", "w") as f:
        json.dump(m, f, indent=1)
    os.replace(p + f".{os.getpid()}.tmp", p)


def _levels(a):
    return None if not a.levels else [int(x) for x in a.levels.split(",")]


def generate(a):
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    B, T, H, W = a.envs, a.chunk, a.height, a.width
    sim = Sim(B, dev, a.seed, res=(H, W), fps=a.fps, levels=_levels(a))
    ex = Explorer(B, dev, a.seed)
    enc = None
    if a.ae:
        from wm import ae as wmae
        enc = wmae.load(a.ae, dev)
    frame, lvl, aux = sim.reset()
    act = torch.zeros(B, len(ACTIONS), device=dev)
    t0 = time.time()
    for k in range(a.chunks):
        stem = os.path.join(a.out, f"{a.name}-{k:05d}")
        acts = torch.zeros(B, T, len(ACTIONS) + len(STATE))
        lvls = torch.zeros(B, T, dtype=torch.int16)
        auxs = torch.zeros(B, T, len(AUX))
        rgb = np.lib.format.open_memmap(stem + ".rgb.npy.tmp", mode="w+", dtype=np.uint8, shape=(B, T, H, W, 3)) if a.rgb else None
        lats = []
        for t in range(T):
            if k or t:
                act = ex.act(sim)
                frame, lvl, aux = sim.step(act)
            acts[:, t, :len(ACTIONS)], lvls[:, t], auxs[:, t] = act.cpu(), lvl.cpu().to(torch.int16), aux.cpu()
            acts[:, t, len(ACTIONS)] = (aux[:, AUX.index("ent_dist")] < PRESENCE_M).float().cpu()
            acts[:, t, len(ACTIONS) + 1] = aux[:, AUX.index("blackout")].cpu()
            acts[:, t, len(ACTIONS) + 2] = aux[:, AUX.index("red")].cpu()
            if rgb is not None:
                rgb[:, t] = frame.cpu().numpy()
            if enc is not None:
                lats.append(enc.encode(frame).cpu())
        _save(stem + ".act.npy", acts.numpy())
        _save(stem + ".lvl.npy", lvls.numpy())
        _save(stem + ".aux.npy", auxs.numpy())
        if rgb is not None:
            rgb.flush()
            del rgb
            os.replace(stem + ".rgb.npy.tmp", stem + ".rgb.npy")
        lat_shape = None
        if enc is not None:
            lat = torch.stack(lats, 1).numpy().astype(np.float16)
            lat_shape = list(lat.shape[2:])
            _save(stem + ".lat.npy", lat)
        _meta(a, a.ae or None, lat_shape)
        n = (k + 1) * B * T
        print(f"chunk {k} done, {n} frames, {n / (time.time() - t0):.0f} frames/s", flush=True)


def preview(a):
    import imageio
    from PIL import Image
    dev = "cuda"
    os.makedirs(a.preview, exist_ok=True)
    lv = _levels(a) or list(range(len(LEVELS)))
    B, N = 12, a.steps
    over = []
    for L in lv:
        sim = Sim(B, dev, a.seed + 17 * L, res=(a.height, a.width), fps=a.fps, levels=[L])
        ex = Explorer(B, dev, a.seed + L)
        frame, _, _ = sim.reset()
        vid, shots = [frame[0].cpu().numpy()], {}
        ev = []
        st = dict(seen=0.0, near=0.0, use=0.0, caught=0.0)
        for t in range(1, N):
            act = ex.act(sim)
            st["use"] += float(((act[:, 7] > 0.5) & (sim.door_now > 0)).sum())
            frame, lvl, aux = sim.step(act)
            st["seen"] += float((aux[:, 8] > 0).float().mean()); st["near"] += float((aux[:, 7] < PRESENCE_M).float().mean())
            st["caught"] += float(((aux[:, 11] > 0) & (aux[:, 12] > 0)).float().mean()) / 12
            vid.append(frame[0].cpu().numpy())
            for b in range(B):
                if t == 12 + (b * (N - 20)) // B:
                    shots[b] = frame[b]
            hot = (aux[:, 8] > 0) & (aux[:, 7] < 12)        # an entity in view and close
            if bool(hot.any()) and len(ev) < 8 and t % 3 == 0:
                ev.append(frame[int(hot.float().argmax())].clone())
        tiles = [shots[b] for b in range(B)]
        tiles[-len(ev[:4]):] = ev[:4] if ev else tiles[-0:]
        g = torch.stack(tiles).float().permute(0, 3, 1, 2)
        g = torch.nn.functional.interpolate(g, size=(a.height * 3 // 4, a.width * 3 // 4), mode="area")
        hh, ww = g.shape[2:]
        sheet = g.view(3, 4, 3, hh, ww).permute(2, 0, 3, 1, 4).reshape(3, 3 * hh, 4 * ww)
        Image.fromarray(sheet.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()).save(
            os.path.join(a.preview, f"{L:02d}_{LEVELS[L]}.jpg"), quality=88)
        imageio.mimwrite(os.path.join(a.preview, f"{L:02d}_{LEVELS[L]}.mp4"), vid, fps=a.fps, quality=7, macro_block_size=16)
        over.append(shots[B // 2])
        mins = B * N / a.fps / 60
        print(f"level {L} {LEVELS[L]}: sheet and video written; something in view {100 * st['seen'] / N:.0f}% of the time, near "
              f"{100 * st['near'] / N:.0f}%; per player-minute {st['use'] / mins:.1f} doors used, {st['caught'] * B / mins:.2f} times caught; "
              f"{int((sim.d_cell >= 0).sum()) // B} doors per map", flush=True)
        del sim
    # doors: every third frame from the moment E is pressed at one
    dl = [l for l in (5, 15, 13, 4) if l in lv] or lv[:1]
    sim = Sim(16, dev, a.seed + 9, res=(a.height, a.width), fps=a.fps, levels=dl)
    ex = Explorer(16, dev, a.seed + 9)
    ex.curious[:] = 1.0
    sim.reset()
    strip, cur = [], None
    for t in range(1500):
        act = ex.act(sim)
        hit = (act[:, 7] > 0.5) & (sim.door_now == 1)
        if cur is None and bool(hit.any()) and len(strip) < 28:
            cur = (int(hit.float().argmax()), t)
        frame, _, _ = sim.step(act)
        if cur is not None and (t - cur[1]) % 3 == 0:
            strip.append(frame[cur[0]].clone())
            if t - cur[1] >= 18:
                cur = None
        if len(strip) >= 28 and cur is None:
            break
    if strip:
        strip += [torch.zeros_like(strip[0])] * (28 - len(strip))
        g = torch.nn.functional.interpolate(torch.stack(strip[:28]).float().permute(0, 3, 1, 2), size=(a.height // 2, a.width // 2), mode="area")
        hh, ww = g.shape[2:]
        sheet = g.view(4, 7, 3, hh, ww).permute(2, 0, 3, 1, 4).reshape(3, 4 * hh, 7 * ww)
        Image.fromarray(sheet.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()).save(os.path.join(a.preview, "doors.jpg"), quality=88)
    print(f"doors: {len([f for f in strip if f.any()])} frames from door openings", flush=True)
    # a noclip and a capture, frame by frame
    sim = Sim(4, dev, a.seed + 5, res=(a.height, a.width), fps=a.fps, levels=lv)
    ex = Explorer(4, dev, a.seed + 5)
    sim.reset()
    strip = []
    for t in range(40):
        act = ex.act(sim)
        act[:, 6] = float(t == 22)
        if t >= 22:
            act[:, :2] = 0
        frame, _, _ = sim.step(act, dest=torch.full((4,), lv[len(lv) // 2], device=dev))
        if 20 <= t < 34:
            strip.append(frame[0])
    sim.e_type[:, 0], sim.e_spr[:, 0], sim.e_alpha[:, 0] = 4, 4, 1.0     # a hound right behind, and little health left
    sim.hp[:] = 0.3
    sim.e_pos[:, 0] = sim.pos + 2.0
    for t in range(18):
        frame, _, _ = sim.step(torch.zeros(4, 7, device=dev))
        if t >= 4:
            strip.append(frame[0])
    g = torch.nn.functional.interpolate(torch.stack(strip).float().permute(0, 3, 1, 2), size=(a.height // 2, a.width // 2), mode="area")
    hh, ww = g.shape[2:]
    sheet = g.view(4, 7, 3, hh, ww).permute(2, 0, 3, 1, 4).reshape(3, 4 * hh, 7 * ww)
    Image.fromarray(sheet.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()).save(os.path.join(a.preview, "transitions.jpg"), quality=88)
    # lamps going out towards the player (every third frame)
    sim = Sim(4, dev, a.seed + 6, res=(a.height, a.width), fps=a.fps, levels=[lv[0]])
    sim.reset()
    for _ in range(3):
        sim.step(torch.zeros(4, 7, device=dev))
    best = sim.probe(ex.rel).argmax(1)
    sim.yaw = sim.yaw + ex.rel[best]
    sim.wave_rate[:] = 1e6
    strip = []
    for t in range(42):
        frame, _, _ = sim.step(torch.zeros(4, 7, device=dev))
        sim.wave_rate[:] = 0
        if t % 3 == 0:
            strip.append(frame[0])
    g = torch.nn.functional.interpolate(torch.stack(strip).float().permute(0, 3, 1, 2), size=(a.height // 2, a.width // 2), mode="area")
    sim.w_act[:] = False
    sim.red_rate[:] = 1e6
    for t in range(14):
        frame, _, _ = sim.step(torch.zeros(4, 7, device=dev))
        sim.red_rate[:] = 0
        if t % 2 == 0:
            strip.append(frame[0])
    g = torch.nn.functional.interpolate(torch.stack(strip).float().permute(0, 3, 1, 2), size=(a.height // 2, a.width // 2), mode="area")
    sheet = g.view(3, 7, 3, hh, ww).permute(2, 0, 3, 1, 4).reshape(3, 3 * hh, 7 * ww)
    Image.fromarray(sheet.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()).save(os.path.join(a.preview, "lamp_wave.jpg"), quality=88)
    # the entities, each 4 m in front of the player
    from .levels import SPR_ID
    kinds = [(1, "figure"), (1, "tall"), (1, "faceless"), (2, "scarecrow"), (4, "hound"), (6, "party"), (5, "horde"), (3, "smiler")]
    sim = Sim(len(kinds), dev, a.seed + 7, res=(a.height, a.width), fps=a.fps, levels=[lv[0]])
    sim.reset()
    sim.rates[:] = 0
    free = sim.probe(ex.rel)
    best = free.argmax(1)
    sim.yaw = sim.yaw + ex.rel[best]
    d = (free.gather(1, best[:, None])[:, 0] - 0.6).clamp(1.5, 4.0)
    for b, (typ, spr) in enumerate(kinds):
        sim.e_type[b, 0], sim.e_spr[b, 0], sim.e_alpha[b, 0] = typ, SPR_ID[spr], 1.0
        sim.e_pos[b, 0] = sim.pos[b] + d[b] * torch.stack([torch.cos(sim.yaw[b]), torch.sin(sim.yaw[b])])
    sim.light[:] = False
    frame, _, _ = sim.step(torch.zeros(len(kinds), 7, device=dev))
    g = torch.nn.functional.interpolate(frame.float().permute(0, 3, 1, 2), size=(a.height * 3 // 4, a.width * 3 // 4), mode="area")
    h3, w3 = g.shape[2:]
    sheet = g.view(2, 4, 3, h3, w3).permute(2, 0, 3, 1, 4).reshape(3, 2 * h3, 4 * w3)
    Image.fromarray(sheet.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()).save(os.path.join(a.preview, "entities.jpg"), quality=88)
    g = torch.stack(over).float().permute(0, 3, 1, 2)
    g = torch.nn.functional.interpolate(g, size=(a.height // 2, a.width // 2), mode="area")
    n = len(over)
    cols = 4
    rows = (n + cols - 1) // cols
    sheet = torch.zeros(3, rows * g.shape[2], cols * g.shape[3], device=g.device)
    for i in range(n):
        sheet[:, (i // cols) * g.shape[2]:(i // cols + 1) * g.shape[2], (i % cols) * g.shape[3]:(i % cols + 1) * g.shape[3]] = g[i]
    Image.fromarray(sheet.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()).save(os.path.join(a.preview, "all_levels.jpg"), quality=88)


def bench(a):
    dev = "cuda"
    sim = Sim(a.envs, dev, a.seed, res=(a.height, a.width), fps=a.fps, levels=_levels(a))
    ex = Explorer(a.envs, dev, a.seed)
    sim.reset()
    for _ in range(10):
        sim.step(ex.act(sim))
    torch.cuda.synchronize()
    sim.prof = {}
    te = 0.0
    t0 = time.time()
    n = a.steps
    for _ in range(n):
        t1 = time.time()
        act = ex.act(sim)
        torch.cuda.synchronize()
        te += time.time() - t1
        sim.step(act)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"{a.envs} episodes at {a.width}x{a.height}: {n * a.envs / dt:.0f} frames/s  ({dt / n * 1e3:.1f} ms per step)")
    print(f"  explorer {te / n * 1e3:.1f} ms; " + "; ".join(f"{k} {v / n * 1e3:.1f} ms" for k, v in sim.prof.items()))
    print(f"  GPU memory peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out"); p.add_argument("--name", default="sim"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", type=int, default=64); p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--chunks", type=int, default=1); p.add_argument("--rgb", action="store_true")
    p.add_argument("--ae", default=""); p.add_argument("--levels", default="")
    p.add_argument("--height", type=int, default=288); p.add_argument("--width", type=int, default=512)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--preview", default=""); p.add_argument("--steps", type=int, default=240)
    p.add_argument("--bench", action="store_true")
    a = p.parse_args()
    with torch.no_grad():
        if a.bench:
            bench(a)
        elif a.preview:
            preview(a)
        else:
            assert a.out, "--out DIR"
            generate(a)


if __name__ == "__main__":
    main()
