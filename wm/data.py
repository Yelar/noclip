"""Reading datasets of chunks (see README: NAME-00000.{lat,act,lvl,aux}.npy, each [B, T, ...]).

Chunks are memory-mapped; a window is F consecutive frames of one episode inside one chunk. A folder that is still
being written can be trained on: `rescan()` picks up chunks that have appeared since. Folders are given separated
by commas; `DIR@2` makes that folder's chunks twice as likely to be drawn as their number says.

With mem=R a window comes with up to R *memory frames* (wm/memory.py): frames from before the window, in the same
episode, that were seen from nearly the same place in nearly the same direction as one frame of the window. That
frame is by preference one that *looks back*: it has turned to a view that the window's earlier frames do not show
in full and an older frame does. Only there does the memory tell the model something its last seconds do not, and
such moments are rare, so windows that have one are also drawn more often than chance would (mem_look).

Datasets of different ages can be mixed: A action columns are made of what a chunk has (older chunks have no
column for E, which nobody pressed in them, and none for "something is near", which is taken from their aux).

  python -m wm.data stats DIR [DIR ...]   per-channel mean and std of the latents -> DIR/stats.json (of the first DIR)
  python -m wm.data look DIR [FRAMES]     how often a window's memory is for a frame that looks back, for some mem_look
"""
import glob
import json
import math
import os
import queue
import sys
import threading
import time

import numpy as np

from wm import memory as MEM

AUX_X, AUX_Z, AUX_YAW, AUX_PITCH, AUX_ENT_DIST, AUX_TRANS, AUX_EP = 0, 1, 2, 3, 7, 12, 14     # columns of aux (sim/core.py)
PRESENCE_M = 20.0           # sim/gen.py: the last action column is 1 while something is within this distance
MEM_EARLY = 40              # with memory, most windows start at least this many frames into their chunk
LOOK_TURN = 0.15            # a frame looks back if it has turned this far (radians) beyond all the window's earlier frames
LOOK_NEAR = 1.2             # ... to a view that a frame from before the window shows (wm/memory.py's score below this)
SKIP_TRIES = 8
LOOK_TRIES = 4              # window starts tried when a window with such a frame is wanted


class Chunks:
    def __init__(self, dirs, frames, mem=0, A=0, mem_drop=0.15, mem_look=0.0):
        """A: action columns to return (0: as stored). mem: memory frames per window; mem_drop: how often a window
        gets none although there would be some (the model must also do without); mem_look: how often a window is
        looked for (among LOOK_TRIES starts) in which a frame looks back."""
        self.dirs, self.wts = [], []
        for d in (dirs.split(",") if isinstance(dirs, str) else dirs):
            if d:
                d, _, w = d.partition("@")
                self.dirs.append(d); self.wts.append(float(w or 1))
        self.frames, self.mem, self.A, self.mem_drop, self.mem_look = frames, mem, A, mem_drop, mem_look
        self.files, self.seen, self.fw, self.cum = [], set(), [], None
        self.want_aux = False       # windows also carry the simulator's record of the frames (pose, entities, ...)
        self.looks = 0              # windows so far whose memory was recalled for a frame that looks back
        self.skip = ()              # action columns that must be 0 in every frame of a window (wm/task.py skip_state)
        self.rescan()

    def rescan(self):
        for d, w in zip(self.dirs, self.wts):
            for f in sorted(glob.glob(os.path.join(d, "*.lat.npy"))):
                if f in self.seen:
                    continue
                try:
                    lat = np.load(f, mmap_mode="r")
                    act = np.load(f[:-8] + ".act.npy", mmap_mode="r")
                    lvl = np.load(f[:-8] + ".lvl.npy", mmap_mode="r")
                    aux = np.load(f[:-8] + ".aux.npy", mmap_mode="r") if os.path.exists(f[:-8] + ".aux.npy") else None
                except (OSError, ValueError):  # not there yet, or still being copied
                    continue
                if lat.shape[1] >= self.frames:
                    self.files.append((lat, act, lvl, aux))
                    self.fw.append(w)
                    self.seen.add(f)
        self.cum = np.cumsum(self.fw) / max(1e-9, float(np.sum(self.fw))) if self.fw else None
        return len(self.files)

    def _acts(self, act, aux):
        """The window's actions with self.A columns."""
        act = np.asarray(act, np.float32)
        if not self.A or act.shape[-1] == self.A:
            return act
        out = np.zeros(act.shape[:-1] + (self.A,), np.float32)
        k = min(act.shape[-1], self.A)
        out[..., :k] = act[..., :k]
        if act.shape[-1] == 7 and self.A >= 9 and aux is not None:
            out[..., 8] = np.asarray(aux)[..., AUX_ENT_DIST] < PRESENCE_M
        # blackout and red (columns 9 and 10) cannot be had from older chunks: their record does not say. Left at 0
        # there, which is wrong during those chunks' blackouts: do not mix such chunks into a model that has them.
        return out

    def _look(self, a, s, rng):
        """A frame of the window starting at s that looks back (see the top of this file), or None. a: the episode
        row's aux up to the window's end."""
        if s < MEM.GAP:
            return None
        w = a[s:s + self.frames]
        gap = np.abs(MEM.wrap(w[:, None, AUX_YAW] - w[None, :, AUX_YAW], 2 * math.pi))
        gap[np.triu_indices(len(w))] = 9.0                          # only the frames before count
        cand = np.flatnonzero((gap.min(1) >= LOOK_TURN) & (gap.min(1) < 9.0) & (w[:, AUX_TRANS] == 0))
        for i in rng.permutation(cand)[:4]:
            sc = MEM.score(a[:s, AUX_X], a[:s, AUX_Z], a[:s, AUX_YAW], a[:s, AUX_PITCH], w[i, AUX_X], w[i, AUX_Z], w[i, AUX_YAW], w[i, AUX_PITCH])
            if sc[(a[:s, AUX_EP] == w[i, AUX_EP]) & (a[:s, AUX_TRANS] == 0)].min(initial=9.0) < LOOK_NEAR:
                return s + int(i)
        return None

    def _memory(self, lat, lvl, aux, b, s, rng, ref=None):
        """Memory frames for the window that starts at frame s of episode row b: latents [R, ...], levels [R] and
        which of the R slots are filled [R] (filled from the back: the last slot is the nearest in the model's time).
        ref: the frame they are recalled for, if the caller has chosen it."""
        R = self.mem
        mlat = np.zeros((R,) + lat.shape[2:], lat.dtype)
        mlvl, mok = np.zeros(R, lvl.dtype), np.zeros(R, bool)
        if aux is None or s < MEM.GAP or rng.random() < self.mem_drop:
            return mlat, mlvl, mok
        a = np.asarray(aux[b, :s + self.frames], np.float32)
        if ref is None and self.mem_look and rng.random() < 0.75:   # a frame that looks back, if the window has one
            ref = self._look(a, s, rng)
        self.looks += ref is not None
        if ref is None:
            ref = s + int(rng.integers(self.frames))                # the frame the memory is recalled for
        ok = (a[:s, AUX_EP] == a[ref, AUX_EP]) & (a[:s, AUX_TRANS] == 0)
        if not ok.any():
            return mlat, mlvl, mok
        age = (ref - np.arange(s)).astype(np.float32)
        drift = 0.03 * np.sqrt(age)                                 # at play the poses are reckoned, and drift
        sc = MEM.score(a[:s, AUX_X] + rng.normal(0, 1, s) * drift, a[:s, AUX_Z] + rng.normal(0, 1, s) * drift, a[:s, AUX_YAW],
                       a[:s, AUX_PITCH], a[ref, AUX_X], a[ref, AUX_Z], a[ref, AUX_YAW], a[ref, AUX_PITCH])
        sc[~ok] = 1e9
        idx = sorted(MEM.choose(sc, np.arange(s), R))
        for k, j in enumerate(idx):
            slot = R - len(idx) + k
            mlat[slot], mlvl[slot], mok[slot] = lat[b, j], lvl[b, j], True
        return mlat, mlvl, mok

    def window(self, rng):
        lat, act, lvl, aux = self.files[min(int(np.searchsorted(self.cum, rng.random())), len(self.cum) - 1)]
        b, last = rng.integers(lat.shape[0]), lat.shape[1] - self.frames
        s = rng.integers(min(MEM_EARLY, last), last + 1) if (self.mem and rng.random() < 0.85) else rng.integers(last + 1)
        ref = None
        if self.mem and aux is not None and rng.random() < self.mem_look:
            a = np.asarray(aux[b], np.float32)
            for _ in range(LOOK_TRIES):
                ref = self._look(a[:s + self.frames], s, rng)
                if ref is not None:
                    break
                s = rng.integers(min(MEM_EARLY, last), last + 1)
        if self.skip and act.shape[2] > max(self.skip):     # e.g. no window with a blackout in it: drawn again
            for _ in range(SKIP_TRIES):
                if not (np.asarray(act[b, s:s + self.frames])[:, list(self.skip)] > 0.5).any():
                    break
                b, s, ref = rng.integers(lat.shape[0]), rng.integers(last + 1), None
        w = (lat[b, s:s + self.frames], self._acts(act[b, s:s + self.frames], None if aux is None else aux[b, s:s + self.frames]),
             lvl[b, s:s + self.frames])
        if self.mem:
            w = w + self._memory(lat, lvl, aux, b, s, rng, ref)
        return w + (np.asarray(aux[b, s:s + self.frames], np.float32),) if self.want_aux else w

    def batch(self, n, rng):
        ws = [self.window(rng) for _ in range(n)]
        return tuple(np.stack([w[i] for w in ws]) for i in range(len(ws[0])))

    def fixed(self, n, batches, seed=1234):
        """The same batches for everyone who has the same folder: validation."""
        rng = np.random.default_rng(seed)
        return [self.batch(n, rng) for _ in range(batches)]


class Feeder:
    """Batches read ahead by a few threads (the files are on a network filesystem)."""

    def __init__(self, chunks, n, seed, threads=4, depth=8, rescan_s=300):
        self.chunks, self.n, self.q, self.rescan_s = chunks, n, queue.Queue(depth), rescan_s
        self.last = time.time()
        for i in range(threads):
            threading.Thread(target=self._run, args=(np.random.default_rng([*np.atleast_1d(seed).tolist(), i]),), daemon=True).start()

    def _run(self, rng):
        while True:
            self.q.put(self.chunks.batch(self.n, rng))

    def get(self):
        if time.time() - self.last > self.rescan_s:
            self.last = time.time()
            self.chunks.rescan()
        return self.q.get()


def load_stats(dirs):
    d = (dirs.split(",") if isinstance(dirs, str) else dirs)[0]
    with open(os.path.join(d, "stats.json")) as f:
        s = json.load(f)
    return np.asarray(s["mean"], np.float32), np.asarray(s["std"], np.float32)


if __name__ == "__main__":
    if sys.argv[1] == "stats":
        ch = Chunks(sys.argv[2:], 1)
        rng = np.random.default_rng(0)
        x = np.concatenate([np.asarray(f[0][:, :: max(1, f[0].shape[1] // 16)], np.float32).reshape(-1, *f[0].shape[2:])
                            for f in [ch.files[i] for i in rng.permutation(len(ch.files))[:64]]])
        mean, std = x.mean((0, 2, 3)), x.std((0, 2, 3))
        out = os.path.join(ch.dirs[0], "stats.json")
        with open(out + ".tmp", "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist(), "frames": int(len(x)), "chunks": len(ch.files)}, f)
        os.replace(out + ".tmp", out)
        print(f"{len(ch.files)} chunks, {len(x)} frames sampled: mean {mean.mean():.3f} (per channel {mean.min():.2f}..{mean.max():.2f}), "
              f"std {std.mean():.3f} ({std.min():.2f}..{std.max():.2f}) -> {out}")
    elif sys.argv[1] == "look":
        fr = int(sys.argv[3]) if len(sys.argv) > 3 else 24
        for look in (0.0, 0.25, 0.5, 1.0):
            ch, rng, n = Chunks(sys.argv[2], fr, 8, 0, 0.0, look), np.random.default_rng(0), 600
            t0 = time.time()
            ws = [ch.window(rng) for _ in range(n)]
            dt = time.time() - t0
            if look == 0.0:     # by chance: windows that have such a frame at all
                aux_of = lambda f, b: np.asarray(f[3][b], np.float32)
                has = 0
                for _ in range(n):
                    f = ch.files[int(rng.integers(len(ch.files)))]
                    b, s = int(rng.integers(f[0].shape[0])), int(rng.integers(MEM_EARLY, f[0].shape[1] - fr + 1))
                    has += ch._look(aux_of(f, b)[:s + fr], s, rng) is not None
                print(f"windows of {fr} frames that have a frame looking back: {has / n:.1%}")
                ch.looks = 0
                ch.mem_look = 1e-9      # only the choice of the frame inside the window drawn
                ws = [ch.window(rng) for _ in range(n)]
                print(f"mem_look 0 but the frame chosen inside the window: memory for such a frame in {ch.looks / n:.1%} of the windows")
                continue
            slots = np.mean([w[5].sum() for w in ws])
            turn = np.mean([np.abs(w[1][:, 2]).mean() for w in ws])
            print(f"mem_look {look}: memory for such a frame in {ch.looks / n:.1%} of the windows, {slots:.1f} slots filled, "
                  f"mean |turn| {turn:.3f}, {n / dt:.0f} windows/s")
