"""Play the world model in a browser.

  python -m wm.play                                  the released model (wm/bundle.py; downloaded once)
  python -m wm.play --ckpt RUN/model.pt [--data DIR] [--ae dcae] [--port 8765] [--host 127.0.0.1] [--token T]

serves a page (wm/web/) and a websocket. For one player it runs a loop at the dataset's frame rate: take the
latest keys and the mouse movement since the last frame, let the model draw the next frame, decode it, send it as
a JPEG. `--data` (a dataset folder) gives the level names and lets a dream start from a few recorded frames instead
of from nothing. From a laptop, reach it through an ssh tunnel (the commands are printed at start; README).

What travels (all JSON text except frames):
  client -> server   {"t":"in","f":-1|0|1,"r":-1|0|1,"sp":0|1,"li":0|1,"dx":rad,"dy":rad,"ct":ms}   about 30 a second;
                       dx > 0 turns right, dy > 0 looks up, both summed since the last message; ct: the client's clock
                     {"t":"noclip","level":id or -1 for a random other level}
                     {"t":"use"}             E: open or close the door ahead
                     {"t":"reset","mode":"nothing"|"data","level":id or -1}
                     {"t":"set","steps":..,"guide":..,"tau":..,"shift":..,"exact":0|1,"dread":0..1}
                     {"t":"rec","on":0|1}    {"t":"pause","on":0|1}
  server -> client   {"t":"hello",...} once, {"t":"rec",...}, {"t":"note","msg":...}
                     a frame: uint32 n (little endian), n bytes of JSON {"i","lvl","ct","ms":{...},"n","rec",...}, a JPEG
                       (ct: the ct of the newest input the frame was drawn from; ms: time per stage on the server;
                        "ev": "hit" or "dead" when the frame shows a blow or the death, see wm/events.py)
"""
import argparse
import asyncio
import io
import json
import os
import random
import secrets
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from aiohttp import WSMsgType, web

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
# a model takes the first cfg.A; the last three are not keys but the director's (direct, lights)
ACTIONS = ["forward", "right", "turn", "pitch", "sprint", "light", "noclip", "use", "presence", "blackout", "red"]


def find_meta(a, ck_args):
    """meta.json of the dataset: --meta, --data, or the folder the checkpoint was trained on."""
    for d in (a.meta, a.data, (ck_args or {}).get("data", "")):
        for one in str(d or "").split(","):
            f = one if one.endswith(".json") else os.path.join(one, "meta.json")
            if one and os.path.isfile(f):
                with open(f) as fh:
                    return json.load(fh)
    return {}


class Engine:
    """Everything that touches the GPU; all its methods run on one worker thread."""

    def __init__(self, a):
        from wm import ae as AE
        from wm import roll
        from wm import memory as MEM
        self.a, self.dev = a, a.device
        self.m, self.mean, self.std = roll.load(a.ckpt, self.dev)
        ck = {} if a.ckpt.endswith(".safetensors") else torch.load(a.ckpt, map_location="cpu")
        self.meta = find_meta(a, ck.get("args") if isinstance(ck, dict) else None)
        del ck
        c = self.m.cfg
        self.fps = a.fps or float(self.meta.get("fps", 10))
        self.turn_max = float(self.meta.get("turn_max", 0.35))
        self.pitch_max = float(self.meta.get("pitch_max", 0.2))
        self.levels = list(self.meta.get("levels") or [f"Level {i}" for i in range(c.levels)])[:c.levels]
        self.ae = AE.load(a.ae or "dcae", self.dev, batch=4)  # it may share its GPU with a training worker: small decodes only
        self.chunks = None
        if a.data:
            from wm import data as D
            self.chunks = D.Chunks(a.data, a.seed_frames, A=c.A)
            if not self.chunks.files:
                self.chunks = None
        self.rng = np.random.default_rng()
        self.A, R = c.A, getattr(c, "mem", 0)
        # memory of places: the pose is reckoned from the keys, every third frame is kept with it (wm/memory.py)
        self.pose = MEM.Pose(self.fps, self.turn_max, self.pitch_max, tuple(self.meta.get("move") or (1.4, 3.0, 1.1, 1.2)))
        self.bank, self.quiet = (MEM.Bank(R) if R else None), 0
        # the director: stretches in which something is near alternate with quiet ones; dread (0..1) shortens the quiet
        self.dread, self.near, self.near_t = float(a.dread), False, 10 * self.fps
        self.black_t = self.red_t = 0                       # frames a blackout / the red light still lasts
        self.blackouts, self.reds = set(self.meta.get("blackouts") or []), set(self.meta.get("reds") or [])   # levels that have them
        self.calm = set(self.meta.get("calm") or [])        # levels in which nothing hunts: the director leaves them alone
        # the model draws the blow and the death itself; a small network reads them off its frames (wm/events.py)
        self.reader, self.event, self.hit_n, self.dead_n, self.hit_at = None, None, 0, 0, -10 ** 9
        ev = a.events or next((os.path.join(d, "events.pt") for d in a.data.split(",") if os.path.exists(os.path.join(d, "events.pt"))), "")
        if ev and self.A > 7:
            from wm import events as EV
            self.reader = EV.load(ev, self.dev)
        self.writer, self.rec_path, self.i = None, "", 0
        self.guard, self.ref, self.off, self.fall, self.fell_at = bool(a.guard), {}, 0, None, -10 ** 9
        if self.guard and self.chunks is not None:  # (decoding is slow without a CUDA device: fewer frames there)
            self.ref = self._reference(32 if self.dev.startswith("cuda") else 8)
        # compile 1: the frame pass and the decoder each run as one CUDA graph (no per-operation launch cost: 35 ms a
        # frame against 43 ms merely compiled, tools/play_prof.py); 2: compiled without graphs; 0: as written;
        # -1: 1 where that is known to work (a CUDA device under Linux), else 0
        comp = a.compile if a.compile >= 0 else int(self.dev.startswith("cuda") and sys.platform.startswith("linux"))
        t0, plain = time.time(), self.ae.dnet.decoder
        try:
            self._build(comp)
        except Exception as e:  # no compiler on this machine, or one this model's code does not get through
            if not comp:
                raise
            print(f"compiling did not work ({type(e).__name__}: {str(e)[:300]}); playing without it, frames take longer "
                  f"(--compile 0 skips the attempt)", flush=True)
            self.ae.dnet.decoder = plain
            self._build(0)
        print(f"ready after {time.time() - t0:.0f} s of warm-up", flush=True)

    def _build(self, comp):
        """The dream and the decoder, compiled as asked, and a few frames drawn so that playing starts at full speed."""
        from wm import fast
        a, c = self.a, self.m.cfg
        graph = "reduce-overhead" if comp == 1 else None
        if comp:  # the decoder is a plain convolutional network
            self.ae.dnet.decoder = torch.compile(self.ae.dnet.decoder.to(memory_format=torch.channels_last), mode=graph)
        self.dream = fast.FastDream(self.m, 1, a.tau, a.steps, a.guide, a.shift, exact=bool(a.exact),
                                    compile=graph or bool(comp), timing=True, solver=a.solver)
        for guide in (1.0, 1.5):  # both batch shapes of the compiled pass, and the decoder
            self.dream.guide = guide
            for _ in range(3):
                self.frame([0.0] * c.A, 0)
        if self.dream.R:  # the pass that takes a frame into memory
            self.dream.set_memory([(-1, torch.zeros(1, c.C, c.h, c.w, device=self.dev), torch.zeros(1, device=self.dev, dtype=torch.long))])
            self.frame([0.0] * c.A, 0)
        self.dream.guide = a.guide
        self.reset("nothing", 0)

    def _reference(self, per_level=32, tries=600):
        """What each level's frames normally look like (brightness and contrast of the decoded picture), from
        recorded frames: the range a dream has to stay in, or it is taken to have fallen apart."""
        got = {}
        for _ in range(tries):
            lat, _, lvl = self.chunks.batch(1, self.rng)
            L = int(lvl[0, -1])
            if len(got.setdefault(L, [])) < per_level and int(lvl[0, 0]) == L:
                got[L].append(lat[0, -1])
            if len(got) >= len(self.levels) and all(len(v) >= per_level for v in got.values()):
                break
        ref = {}
        for L, zs in got.items():
            if len(zs) < 8:
                continue
            img = self.ae.decode(torch.from_numpy(np.stack(zs)).to(self.dev).float()).float()
            mu, sd = img.mean((1, 2, 3)).cpu().numpy(), img.std((1, 2, 3)).cpu().numpy()
            ref[L] = (float(np.percentile(mu, 3)), float(np.percentile(mu, 97)), float(np.percentile(sd, 3)))
        return ref

    def _watch(self, img, level):
        """Count the frames in a row that no longer look like the level; after 2 s of them, ask for a fall into
        another level (the model was taught that transition), and if that does not help either, for a new dream."""
        if not self.guard or level not in self.ref:
            return
        lo, hi, sd_lo = self.ref[level]
        mu, sd = float(img.float().mean()), float(img.float().std())
        bad = sd < 0.5 * sd_lo or mu > 1.3 * hi + 6 or mu < 0.5 * lo - 3
        self.off = self.off + 1 if bad else 0
        if self.off >= int(2 * self.fps) and self.i - self.fell_at > int(3 * self.fps):
            self.fall = "reset" if self.i - self.fell_at < int(8 * self.fps) else "noclip"
            self.fell_at, self.off = self.i, 0

    def opts(self):
        d = self.dream
        return {"steps": d.steps, "guide": d.guide, "tau": d.tau, "shift": d.shift, "exact": int(d.exact), "guard": int(self.guard),
                "dread": self.dread, "mem": d.R}

    def set(self, o):
        d = self.dream
        d.steps = int(min(32, max(1, o.get("steps", d.steps))))
        d.guide = float(min(6.0, max(1.0, o.get("guide", d.guide))))
        d.tau = float(min(0.5, max(0.0, o.get("tau", d.tau))))
        d.shift = float(min(8.0, max(0.25, o.get("shift", d.shift))))
        if "exact" in o:
            d.set_exact(bool(o["exact"]))
        if "guard" in o:
            self.guard = bool(o["guard"])
        self.dread = float(min(1.0, max(0.0, o.get("dread", self.dread))))
        return self.opts()

    def _events(self, z):
        """A blow or the death in the frame just drawn, read off its latent: a blow counts once (its face stays for
        several frames); the death after three frames of it, and then the place is gone, and so is its memory."""
        hit, dead = (self.reader(z.float()) > 0).flatten().tolist()
        self.hit_n = self.hit_n + 1 if hit else 0
        if self.hit_n == 2 and self.i - self.hit_at > 2 * self.fps:
            self.event, self.hit_at = "hit", self.i
        self.dead_n = self.dead_n + 1 if dead else 0
        if self.dead_n == 3:
            self.event = "dead"
            self._forget()
            self.quiet = int(1.5 * self.fps)

    def direct(self, level=-1):
        """Is something near in the next frame? (The model's last action column; only models that have it use it.)"""
        if self.dread <= 0 or level in self.calm:
            return 0.0
        self.near_t -= 1
        if self.near_t <= 0:
            self.near = not self.near
            self.near_t = self.fps * (self.rng.uniform(6, 22) if self.near else 3 + self.rng.exponential(40.0 / (0.25 + 2.0 * self.dread)))
        return 1.0 if self.near else 0.0

    def lights(self, level=-1):
        """Is a blackout on, are the lamps red, in the next frame? (Action columns 9 and 10, for models that have
        them: without being told, a model cannot know how long a darkness lasts, and its dreams stay dark.) As in
        the data: a blackout lasts 11 to 21 s and brings something near, red light lasts 8 to 25 s; with dread 0.5
        a blackout comes about every three minutes, in the levels whose data has them (meta.json)."""
        if self.A <= 9 or self.dread <= 0 or level in self.calm:
            self.black_t = self.red_t = 0
            return 0.0, 0.0
        per_frame = self.dread / (60.0 * self.fps)
        if self.black_t > 0:
            self.black_t -= 1
        elif level in self.blackouts and self.rng.random() < 0.6 * per_frame:
            self.black_t = int(self.fps * self.rng.uniform(11, 21))
            if not self.near:
                self.near, self.near_t = True, int(self.black_t * self.rng.uniform(0.5, 1.0))
        if self.red_t > 0:
            self.red_t -= 1
        elif level in self.reds and self.black_t <= 0 and self.rng.random() < 0.2 * per_frame:
            self.red_t = int(self.fps * self.rng.uniform(8, 25))
        return float(self.black_t > 0), float(self.red_t > 0)

    def _forget(self):
        """A new place: nothing that was seen before is there to come back to."""
        self.black_t = self.red_t = 0
        self.pose.reset()
        if self.bank is not None:
            self.bank.clear()
        self.near, self.near_t = False, self.fps * self.rng.uniform(6, 20)

    def reset(self, mode, level):
        """A new dream: from nothing in `level`, or from recorded frames of a random episode. Returns the level."""
        self.dream.reset()
        self.off, self.fall = 0, None
        self._forget()
        self.event, self.hit_n, self.dead_n, self.hit_at = None, 0, 0, -10 ** 9
        if mode == "data" and self.chunks is not None:
            lat, act, lvl = self.chunks.batch(1, self.rng)
            z = (torch.from_numpy(lat).to(self.dev).float() - self.mean[:, None]) / self.std[:, None]
            lvl = torch.from_numpy(lvl.astype(np.int64)).to(self.dev)
            self.dream.seed(z, torch.from_numpy(act).to(self.dev), lvl)
            level = int(lvl[0, -1])
        return level

    def frame(self, act, level):
        """One frame: -> (JPEG bytes, milliseconds per stage)."""
        t0 = time.perf_counter()
        lv = torch.tensor([level], device=self.dev, dtype=torch.long)
        if len(act) > 6 and act[6] > 0.5:  # noclip: another map; what is drawn while falling is not worth keeping
            self._forget()
            self.quiet = 12
        pose = self.pose.step(act)
        if self.bank is not None:
            self.dream.set_memory([(fid, z_, l_) for fid, (z_, l_) in self.bank.recall(pose, self.i, self.dream.keep + 1)])
        z = self.dream.step(torch.tensor([act], device=self.dev, dtype=torch.float32), lv)
        if self.bank is not None:
            if self.quiet:
                self.quiet -= 1
            else:
                self.bank.add(self.i, pose, (z.clone(), lv))
        t1 = time.perf_counter()
        img = self.ae.decode(z * self.std + self.mean)[0]
        self._watch(img, level)
        if self.reader is not None:
            self._events(z)
        if self.dev.startswith("cuda"):
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        arr = img.cpu().numpy()
        buf = io.BytesIO()
        from PIL import Image
        Image.fromarray(arr).save(buf, format="JPEG", quality=self.a.jpeg)
        t3 = time.perf_counter()
        if self.writer is not None:
            self.writer.append_data(arr)
        self.i += 1
        ms = {"ctx": round(self.dream.ms.get("ctx", 0.0), 1), "den": round(self.dream.ms.get("den", 0.0), 1),
              "dec": round((t2 - t1) * 1e3, 1), "jpg": round((t3 - t2) * 1e3, 1), "tot": round((time.perf_counter() - t0) * 1e3, 1)}
        return buf.getvalue(), ms

    def record(self, on):
        """Start or stop writing the frames to an mp4. Returns the file's path."""
        if on and self.writer is None:
            import imageio.v2 as imageio
            d = self.a.rec or os.path.join(os.path.dirname(os.path.abspath(self.a.ckpt)), "rec")   # main(): "rec" for a released model
            try:
                os.makedirs(d, exist_ok=True)
                if not os.access(d, os.W_OK):
                    raise OSError
            except OSError:  # the checkpoint's folder belongs to another account
                d = os.path.expanduser("~/wm-rec")
                os.makedirs(d, exist_ok=True)
            self.rec_path = os.path.join(d, time.strftime("play-%Y%m%d-%H%M%S.mp4"))
            self.writer = imageio.get_writer(self.rec_path, fps=self.fps, codec="libx264", quality=8, macro_block_size=1)
        elif not on and self.writer is not None:
            self.writer.close()
            self.writer = None
        return self.rec_path


class Player:
    def __init__(self, ws, level):
        self.ws, self.level = ws, level
        self.f = self.r = self.sp = self.li = 0
        self.dx = self.dy = 0.0   # radians of mouse movement not yet turned into frames
        self.ct = 0.0
        self.noclip = None        # level to fall into at the next frame
        self.use = False          # E was pressed since the last frame
        self.reset = None         # (mode, level) to start a new dream at the next frame
        self.paused = False
        self.last_in = 0.0        # the dream starts with the first input and waits when inputs stop


def take_action(p, eng):
    """The action vector of the next frame from the player's state; uses up the mouse movement."""
    def axis(v, vmax):
        a = max(-1.0, min(1.0, v / vmax))
        rest = v - a * vmax
        return a, max(-vmax, min(vmax, rest))  # a flick faster than the model can turn carries over one frame at most
    turn, p.dx = axis(p.dx, eng.turn_max)
    pitch, p.dy = axis(p.dy, eng.pitch_max)
    noclip = 0.0
    if p.noclip is not None:
        p.level, p.noclip, noclip = p.noclip, None, 1.0
    use, p.use = float(p.use), False
    return [float(p.f), float(p.r), turn * eng.turn_sign, pitch * eng.pitch_sign, float(p.sp), float(p.li), noclip, use,
            eng.direct(p.level), *eng.lights(p.level)][:eng.A]


async def tick_loop(app, p):
    eng, pool, loop = app["state"]["engine"], app["pool"], asyncio.get_running_loop()
    period = 1.0 / eng.fps
    nxt = loop.time()
    while not p.ws.closed:
        if p.paused or time.time() - p.last_in > 5:  # nobody at the controls: the dream waits
            await asyncio.sleep(0.05)
            nxt = loop.time()
            continue
        if p.reset is not None:
            (mode, level), p.reset = p.reset, None
            p.level = await loop.run_in_executor(pool, eng.reset, mode, level)
            p.noclip = None
        ct = p.ct
        act = take_action(p, eng)
        jpeg, ms = await loop.run_in_executor(pool, eng.frame, act, p.level)
        if eng.fall is not None:  # the dream fell apart: the player falls out of it
            how, eng.fall = eng.fall, None
            others = [i for i in range(len(eng.levels)) if i != p.level]
            if how == "noclip" and p.noclip is None and others:
                p.noclip = int(eng.rng.choice(others))
            elif how == "reset":
                p.reset = ("data" if eng.chunks is not None else "nothing", int(eng.rng.choice(others)) if others else p.level)
        ev, eng.event = eng.event, None
        head = json.dumps({"i": eng.i, "lvl": p.level, "ct": ct, "ms": ms, "n": eng.dream.n, "mem": eng.dream.mem_n,
                           "near": int(eng.near and eng.dread > 0 and p.level not in eng.calm),
                           "dark": int(eng.black_t > 0), "red": int(eng.red_t > 0),
                           **({"ev": ev} if ev else {}), "rec": int(eng.writer is not None)}).encode()
        try:
            await p.ws.send_bytes(struct.pack("<I", len(head)) + head + jpeg)
        except (ConnectionError, RuntimeError):
            break
        nxt += period
        delay = nxt - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            nxt = loop.time()  # drawing takes longer than a frame's time: go as fast as it does


def allowed(request):
    tok = request.app["args"].token
    return not tok or secrets.compare_digest(request.query.get("token", ""), tok)


async def index(request):
    if not allowed(request):
        return web.Response(status=403, text="add ?token=... to the address")
    return web.FileResponse(os.path.join(WEB, "index.html"), headers={"Cache-Control": "no-store"})


async def ws_handler(request):
    if not allowed(request):
        return web.Response(status=403, text="token")
    app = request.app
    eng, pool, loop = app["state"]["engine"], app["pool"], asyncio.get_running_loop()
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 20)
    await ws.prepare(request)
    st = app["state"]
    old = st.get("player")
    if old is not None and not old.ws.closed:  # one player at a time: the newest connection plays
        await old.ws.close(code=4000, message=b"another player connected")
    p = st["player"] = Player(ws, 0)
    await ws.send_json({"t": "hello", "levels": eng.levels, "fps": eng.fps, "has_data": eng.chunks is not None,
                        "opts": eng.opts(), "turn_max": eng.turn_max, "pitch_max": eng.pitch_max, "level": p.level,
                        "ckpt": os.path.basename(os.path.dirname(os.path.abspath(eng.a.ckpt))), "ae": eng.a.ae,
                        "params_m": round(sum(q.numel() for q in eng.m.parameters()) / 1e6)})
    p.reset = ("data" if eng.chunks is not None else "nothing", 0)
    task = asyncio.create_task(tick_loop(app, p))
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                m = json.loads(msg.data)
                t = m.get("t")
                if t == "in":
                    p.f, p.r = int(max(-1, min(1, m.get("f", 0)))), int(max(-1, min(1, m.get("r", 0))))
                    p.sp, p.li = int(bool(m.get("sp"))), int(bool(m.get("li")))
                    p.dx += float(m.get("dx", 0.0)); p.dy += float(m.get("dy", 0.0))
                    p.ct, p.last_in = float(m.get("ct", 0.0)), time.time()
                elif t == "noclip":
                    lv = int(m.get("level", -1))
                    if not 0 <= lv < len(eng.levels):
                        lv = random.choice([i for i in range(len(eng.levels)) if i != p.level] or [0])
                    p.noclip = lv
                elif t == "use":
                    p.use = True
                elif t == "reset":
                    lv = int(m.get("level", -1))
                    p.reset = (m.get("mode", "nothing"), lv if 0 <= lv < len(eng.levels) else random.randrange(len(eng.levels)))
                elif t == "set":
                    await ws.send_json({"t": "opts", "opts": await loop.run_in_executor(pool, eng.set, m)})
                elif t == "rec":
                    path = await loop.run_in_executor(pool, eng.record, bool(m.get("on")))
                    await ws.send_json({"t": "rec", "on": int(bool(m.get("on"))), "path": path})
                elif t == "pause":
                    p.paused = bool(m.get("on"))
            except (ValueError, TypeError) as e:
                await ws.send_json({"t": "note", "msg": f"bad message: {e}"})
    finally:
        task.cancel()
        if st.get("player") is p:
            st["player"] = None
            await loop.run_in_executor(pool, eng.record, False)
    return ws


async def on_startup(app):
    a = app["args"]
    app["state"]["engine"] = eng = await asyncio.get_running_loop().run_in_executor(app["pool"], Engine, a)
    eng.turn_sign = -1.0 if a.flip_turn else float(eng.meta.get("turn_sign", 1.0))
    eng.pitch_sign = -1.0 if a.flip_pitch else float(eng.meta.get("pitch_sign", 1.0))
    q = f"?token={a.token}" if a.token else ""
    node, user = socket.gethostname(), os.environ.get("USER", "USER")
    print(f"playing {a.ckpt} at http://{a.host}:{a.port}/{q}\n"
          f"from a laptop, through an ssh alias that lands on this node:   ssh -N -L {a.port}:127.0.0.1:{a.port} ALIAS\n"
          f"or through the login node (needs --host 0.0.0.0 and --token): ssh -N -L {a.port}:{node}:{a.port} LOGIN\n"
          f"then open http://localhost:{a.port}/{q}", flush=True)


def main():
    from wm import bundle as BU
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BU.REPO, help="a released model: its folder, or its Hugging Face repository "
                   "(downloaded once). It brings its decoder, level names and starting frames; used when there is no --ckpt")
    p.add_argument("--ckpt", default="", help="a checkpoint of a training run instead (then say --data and --ae)")
    p.add_argument("--data", default="", help="dataset folder: level names, and recorded frames to start dreams from")
    p.add_argument("--meta", default="", help="a meta.json when there is no --data")
    p.add_argument("--ae", default="", help="the autoencoder, wm/ae.py (default: the released model's, or dcae)")
    p.add_argument("--device", default="auto", help="auto: cuda if there is one, else mps (Apple), else cpu; "
                   "only cuda is fast enough to play at the game's speed")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token", default="", help="required in the address as ?token=...; made up if --host is not local")
    p.add_argument("--fps", type=float, default=0, help="default: the dataset's")
    p.add_argument("--steps", type=int, default=12, help="denoising steps per frame: 12 keeps dreams from falling apart "
                   "far more often than 6 does (README, Playing); the page can change it")
    p.add_argument("--guide", type=float, default=1.0, help="guidance on the level; above 1 every step costs two passes "
                   "(1: none, which is what the dreams were measured with)")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--shift", type=float, default=1.0)
    p.add_argument("--exact", type=int, default=0, help="1: redo the context pass every frame (wm/roll.py's computation)")
    p.add_argument("--compile", type=int, default=-1, help="1: as CUDA graphs, 2: compiled only, 0: not at all; "
                   "-1: 1 on a CUDA device under Linux, else 0")
    p.add_argument("--solver", default="ab2", help="ab2 (second order: 6 steps land closer to the flow's own frame than 8 of\n"
                   "euler do, tools/sampler.py) or euler")
    p.add_argument("--seed_frames", type=int, default=8)
    p.add_argument("--jpeg", type=int, default=85)
    p.add_argument("--rec", default="", help="folder for recordings (default: rec/ next to the checkpoint)")
    p.add_argument("--guard", type=int, default=1, help="1: when the picture has stopped looking like the level for 2 s, "
                   "the player falls into another level (needs --data); 0: let a dream fall apart")
    p.add_argument("--events", default="", help="the reader of blows and deaths (wm/events.py); default: events.pt in --data")
    p.add_argument("--dread", type=float, default=0.5, help="0..1: how often something is near (0: never); the page can change it")
    p.add_argument("--flip_turn", type=int, default=0)
    p.add_argument("--flip_pitch", type=int, default=0)
    p.add_argument("--pid", default="", help="write the process id here")
    a = p.parse_args()
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if not a.ckpt:  # a released model: what the command line leaves open comes from its folder
        d = BU.fetch(a.model)
        for k, v in BU.play_args(d).items():
            if not getattr(a, k):
                setattr(a, k, v)
        a.rec = a.rec or "rec"
        print(f"model: {d}", flush=True)
    if a.device != "cuda":
        print(f"no CUDA device: drawing on {a.device}, in float32 and uncompiled. A frame will take longer than the tenth "
              f"of a second the game gives it, so the game runs slow; fewer steps (the page's settings) help", flush=True)
        import shutil
        if a.device == "cpu" and shutil.which("nvidia-smi"):  # the usual reason on a machine that has a GPU
            print(f"This machine has an NVIDIA driver, but the installed PyTorch ({torch.__version__}, built for CUDA "
                  f"{torch.version.cuda}) cannot use it: it is probably built for a newer driver. Install the PyTorch build "
                  f"for your driver (pytorch.org, Get Started) and start again", flush=True)
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.token:
        a.token = secrets.token_urlsafe(9)
    if a.pid:
        with open(a.pid, "w") as f:
            f.write(str(os.getpid()))
    app = web.Application()
    app["args"], app["pool"], app["state"] = a, ThreadPoolExecutor(max_workers=1), {"player": None, "engine": None}
    app.on_startup.append(on_startup)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/s", WEB)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
