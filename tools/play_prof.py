"""Where a played frame's time goes, and what each way of running the pieces costs:

  python tools/play_prof.py CKPT [AE] [STEPS] [ONLY]

ONLY: a word that a variant's name must hold for it to be run (e.g. "ab2").

One player, the window full, 8 memory frames in place, no guidance (as the play server runs). Per variant the mean
milliseconds per frame of: the passes that draw the frame, the pass that takes it into the context, decoding,
JPEG. Variants: the frame pass run as it is / compiled / compiled into one CUDA graph; the decoder likewise.
"""
import io
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wm import ae as AE  # noqa: E402
from wm import fast, roll  # noqa: E402

ckpt = sys.argv[1]
ae_spec = sys.argv[2] if len(sys.argv) > 2 else "dcae"
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
dev = "cuda"
m, mean, std = roll.load(ckpt, dev)
c = m.cfg
R = getattr(c, "mem", 0)
sync = torch.cuda.synchronize
print(f"{ckpt}: {sum(q.numel() for q in m.parameters()) / 1e6:.0f}M parameters, window {c.frames - 1} frames + {R} memory, {steps} steps", flush=True)


def jpeg_pil(img):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img.cpu().numpy()).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def jpeg_gpu(img):
    from torchvision.io import encode_jpeg
    return bytes(encode_jpeg(img.permute(2, 0, 1).contiguous(), quality=85).cpu().numpy())


def run(name, comp, dec_mode, jpeg=jpeg_pil, frames=140, solver="euler", n_steps=steps):
    torch.manual_seed(0)
    ae = AE.load(ae_spec, dev, batch=4)
    if "cl" in dec_mode:
        ae.dnet.decoder = ae.dnet.decoder.to(memory_format=torch.channels_last)
    if "compile" in dec_mode:
        ae.dnet.decoder = torch.compile(ae.dnet.decoder, mode="reduce-overhead" if "graph" in dec_mode else None)
    d = fast.FastDream(m, 1, 0.05, n_steps, 1.0, timing=True, compile=comp, solver=solver)
    act, lvl = torch.zeros(1, c.A, device=dev), torch.zeros(1, device=dev, dtype=torch.long)
    act[0, 0] = 1.0
    memf = [(i, torch.randn(1, c.C, c.h, c.w, device=dev), lvl) for i in range(R)]
    acc = dict(mem=0.0, den=0.0, ctx=0.0, dec=0.0, jpg=0.0, tot=0.0)
    k = frames // 2
    for i in range(frames):
        sync(); t0 = time.perf_counter()
        if R:
            d.set_memory(memf)
        sync(); t1 = time.perf_counter()
        z = d.step(act, lvl)
        sync(); t2 = time.perf_counter()
        img = ae.decode(z * std + mean)[0]
        sync(); t3 = time.perf_counter()
        jpeg(img)
        t4 = time.perf_counter()
        if i >= k:
            for key, v in (("mem", t1 - t0), ("den", d.ms["den"] / 1e3), ("ctx", d.ms["ctx"] / 1e3), ("dec", t3 - t2), ("jpg", t4 - t3), ("tot", t4 - t0)):
                acc[key] += v * 1e3 / (frames - k)
    print(f"  {name:44s} memory {acc['mem']:5.1f}  draw {acc['den']:5.1f}  context {acc['ctx']:5.1f}  decode {acc['dec']:5.1f}  jpeg {acc['jpg']:4.1f}  "
          f"total {acc['tot']:6.1f} ms = {1000 / acc['tot']:5.1f} frames/s", flush=True)
    del ae, d
    torch.cuda.empty_cache()


for name, comp, dec_mode, jp, kw in (
        ("as it is (nothing compiled)", False, "", jpeg_pil, {}),
        ("both compiled (the play server until Nov 19)", True, "compile", jpeg_pil, {}),
        ("pass as one CUDA graph", "reduce-overhead", "compile", jpeg_pil, {}),
        ("pass and decoder as CUDA graphs", "reduce-overhead", "compile graph", jpeg_pil, {}),
        ("... decoder channels-last", "reduce-overhead", "compile graph cl", jpeg_pil, {}),
        ("... 6 steps of ab2", "reduce-overhead", "compile graph cl", jpeg_pil, dict(solver="ab2", n_steps=6)),
        ("... JPEG on the GPU, 6 steps of ab2", "reduce-overhead", "compile graph cl", jpeg_gpu, dict(solver="ab2", n_steps=6)),
        ("... 12 steps of ab2", "reduce-overhead", "compile graph cl", jpeg_pil, dict(solver="ab2", n_steps=12)),
        ("... 16 steps of ab2", "reduce-overhead", "compile graph cl", jpeg_pil, dict(solver="ab2", n_steps=16))):
    if len(sys.argv) > 4 and sys.argv[4] not in name:
        continue
    try:
        run(name, comp, dec_mode, jp, **kw)
    except Exception as e:  # a variant this installation cannot run should not hide the others
        print(f"  {name}: failed, {type(e).__name__}: {str(e)[:200]}", flush=True)
