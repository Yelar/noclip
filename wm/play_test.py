"""A player without a browser, for measuring wm/play.py:

  python -m wm.play_test --url http://127.0.0.1:8765/ [--seconds 20] [--out DIR] [--set steps=8,guide=1.5] [--reset data]

It connects, sends a scripted sequence of inputs 30 times a second (walk, turn, strafe, look up and down, stand,
flashlight, run, noclip, walk backwards, a fast turn) and reports the frame rate it received, the time from an
input to the frame drawn from it, and the server's time per stage. With --out it saves the frames as play_test.mp4
and the numbers as play_test.json.
"""
import argparse
import asyncio
import io
import json
import os
import struct
import time
from urllib.parse import urlsplit

import aiohttp


def script(t):
    """Input at time t (seconds): forward, right, sprint, light, turn rate and pitch rate in rad/s, noclip."""
    t = t % 20.0
    f = r = sp = 0
    turn = pitch = 0.0
    if t < 3: f = 1
    elif t < 5: f, turn = 1, 1.5
    elif t < 6.5: r = 1
    elif t < 7.25: pitch = 0.8
    elif t < 8: pitch = -0.8
    elif t < 9: pass
    elif t < 12: f, sp, turn = 1, 1, -1.0
    elif t < 16: f = 1
    elif t < 18: f = -1
    else: turn = 5.0
    return f, r, sp, int(9 <= t < 16), turn, pitch


async def main(a):
    u = urlsplit(a.url)
    ws_url = f"{'wss' if u.scheme == 'https' else 'ws'}://{u.netloc}/ws{'?' + u.query if u.query else ''}"
    frames, lat, stages, jpegs = [], [], [], []
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(ws_url, max_msg_size=0) as ws:
            hello = json.loads((await ws.receive()).data)
            if a.set:
                await ws.send_json({"t": "set", **{k: float(v) for k, v in (kv.split("=") for kv in a.set.split(","))}})
            if a.reset:
                await ws.send_json({"t": "reset", "mode": a.reset, "level": a.level})
            t0 = time.time()

            async def sender():
                last, clipped = time.time(), -1
                while True:
                    now = time.time()
                    t, dt = now - t0, now - last
                    last = now
                    f, r, sp, li, turn, pitch = script(t)
                    await ws.send_json({"t": "in", "f": f, "r": r, "sp": sp, "li": li, "dx": turn * dt, "dy": pitch * dt, "ct": now * 1000})
                    if int(t // 20) != clipped and t % 20 >= 12:  # one noclip per round of the script
                        clipped = int(t // 20)
                        await ws.send_json({"t": "noclip", "level": -1})
                    await asyncio.sleep(1 / 30)

            task = asyncio.create_task(sender())
            try:
                while time.time() - t0 < a.seconds:
                    msg = await ws.receive(timeout=30)
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        now = time.time()
                        n = struct.unpack_from("<I", msg.data)[0]
                        hdr = json.loads(msg.data[4:4 + n])
                        frames.append(now); stages.append(hdr["ms"])
                        if hdr["ct"]:
                            lat.append(now * 1000 - hdr["ct"])
                        if a.out:
                            jpegs.append(msg.data[4 + n:])
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                        break
            finally:
                task.cancel()
    k = [i for i, t in enumerate(frames) if t - frames[0] >= a.skip] or [0]  # leave the start out of the averages
    k0 = k[0]
    n = len(frames) - k0
    srt = sorted(lat[k0:]) or [0.0]
    rep = {"model": f"{hello.get('ckpt')} ({hello.get('params_m')}M)", "decoder": hello.get("ae"), "target_fps": hello.get("fps"),
           "first_frame_after_s": round(frames[0] - t0, 2) if frames else None,
           "seconds": round(frames[-1] - frames[0], 1) if frames else 0, "frames": len(frames),
           "fps": round((n - 1) / (frames[-1] - frames[k0]), 2) if n > 1 else 0.0,
           "input_to_frame_ms": {"mean": round(sum(srt) / len(srt), 1), "p50": round(srt[len(srt) // 2], 1), "p95": round(srt[int(len(srt) * 0.95)], 1)},
           "server_ms": {key: round(sum(m[key] for m in stages[k0:]) / max(1, n), 1) for key in (stages[0] if stages else {})},
           "opts": a.set or "server defaults"}
    if a.out and jpegs:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
        os.makedirs(a.out, exist_ok=True)
        with imageio.get_writer(os.path.join(a.out, "play_test.mp4"), fps=hello.get("fps", 10), codec="libx264", quality=8, macro_block_size=1) as w:
            for j in jpegs:
                w.append_data(np.asarray(Image.open(io.BytesIO(j)).convert("RGB")))
        with open(os.path.join(a.out, "play_test.json"), "w") as f:
            json.dump(rep, f, indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8765/")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--skip", type=float, default=3, help="seconds at the start left out of the averages")
    p.add_argument("--out", default="")
    p.add_argument("--set", default="", help="settings to send first, e.g. steps=8,guide=1.5,exact=0")
    p.add_argument("--reset", default="", help="start with a new dream: nothing or data")
    p.add_argument("--level", type=int, default=0)
    asyncio.run(main(p.parse_args()))
