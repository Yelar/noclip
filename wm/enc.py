"""A fast encoder for the simulator's frames: a ConvNeXt trained to give the same latents as the pretrained
autoencoder's encoder, which costs about a trillion operations per frame and limits dataset rendering to about 50
frames/s per GPU.

  python -m wm.enc --data DIR --replay b:2000:64,c:3000:64 --val val:9000:32 --out OUT [--steps 20000]

Training needs (frame, latent) pairs. A dataset holds the latents but not the frames, so the frames are rendered
again: the simulator is deterministic, and running the generator's seed again reproduces its episodes (checked
against the stored ground truth of every frame; an episode that differs is left out). Writes OUT/enc.pt, used as
`--ae dcae-lite,enc=OUT/enc.pt`.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Student(nn.Module):
    def __init__(self, C=32, pretrained=False):
        super().__init__()
        import torchvision
        self.f = torchvision.models.convnext_small(weights="IMAGENET1K_V1" if pretrained else None).features
        self.mix = nn.Sequential(nn.Conv2d(768 + 4 * 384, 1024, 1), nn.GELU(), nn.Conv2d(1024, 1024, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(1024, C, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """x [N, 3, H, W] in [-1, 1] -> latents [N, C, H/32, W/32], in the scale datasets store."""
        x = ((x + 1) * 0.5 - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        for i, m in enumerate(self.f):
            x = m(x)
            if i == 5:
                mid = x  # 1/16 resolution: finer detail for the head
        return self.mix(torch.cat((x, F.pixel_unshuffle(mid, 2)), 1))


class Replay:
    """Render a generator's episodes again and pair each frame with the latent the dataset stored for it."""

    def __init__(self, data, spec, dev):
        from sim import Explorer, Sim
        name, seed, B = spec.split(":")
        self.data, self.name, self.B, self.dev = data, name, int(B), dev
        self.make = lambda: (Sim(self.B, dev, int(seed)), Explorer(self.B, dev, int(seed)))
        self.restart()
        self.bad = self.n = 0

    def restart(self):
        self.sim, self.ex = self.make()
        self.k, self.t, self.lat = 0, 0, None

    def ready(self):
        return self.lat is not None or os.path.exists(os.path.join(self.data, f"{self.name}-{self.k:05d}.lat.npy"))

    def step(self):
        """(frames uint8 [n, H, W, 3], latents float16 [n, C, h, w]) of the episodes that replayed exactly."""
        if self.lat is None:
            stem = os.path.join(self.data, f"{self.name}-{self.k:05d}")
            self.lat = torch.from_numpy(np.load(stem + ".lat.npy")).to(self.dev)
            self.aux = torch.from_numpy(np.load(stem + ".aux.npy")).to(self.dev)
        if self.k == 0 and self.t == 0:
            frame, _, aux = self.sim.reset()
        else:
            frame, _, aux = self.sim.step(self.ex.act(self.sim))
        ok = ((aux - self.aux[:, self.t]).abs() < 1e-3).all(-1)
        z = self.lat[:, self.t]
        self.bad += int((~ok).sum()); self.n += self.B
        self.t += 1
        if self.t == self.lat.shape[1]:
            self.k, self.t, self.lat = self.k + 1, 0, None
        return frame[ok], z[ok]


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean((1, 2, 3))
    return (10 * torch.log10(255 ** 2 / mse.clamp_min(1e-6))).mean().item()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--replay", required=True, help="NAME:SEED:ENVS,... of generators whose chunks are in --data")
    p.add_argument("--val", default="", help="NAME:SEED:ENVS in --val_data (default --data + '-val') for held-out frames")
    p.add_argument("--val_data", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ratio", type=float, default=1.5, help="training steps per rendered step")
    p.add_argument("--buffer", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--decoder", default="dcae-lite")
    p.add_argument("--init", default="", help="continue from an enc.pt (for a short finishing run at a low --lr)")
    a = p.parse_args()
    from wm import ae as AE
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    net = Student(pretrained=True).to(dev).to(memory_format=torch.channels_last)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev))
    ema = copy.deepcopy(net).requires_grad_(False).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.03, final_div_factor=50)
    ae = AE.load(a.decoder, dev)
    # held-out frames with the stored latents
    v = Replay(a.val_data or a.data.rstrip("/") + "-val", a.val, dev)
    hf, hz = [], []
    for i in range(96):
        f, z = v.step()
        if i % 6 == 5:
            hf.append(f[:16]); hz.append(z[:16])
    hf, hz = torch.cat(hf), torch.cat(hz)
    base = psnr(ae.decode(hz), hf)
    print(json.dumps({"held": len(hf), "replay_mismatch": v.bad / v.n, "psnr_teacher": round(base, 2),
                      "latent_var": round(hz.float().var().item(), 4)}), flush=True)
    del v
    reps = [Replay(a.data, s, dev) for s in a.replay.split(",")]
    H, W = hf.shape[1:3]
    buf_f = torch.zeros(a.buffer, H, W, 3, dtype=torch.uint8, device=dev)
    buf_z = torch.zeros(a.buffer, *hz.shape[1:], dtype=torch.float16, device=dev)
    fill = ptr = 0
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    step, credit, acc, t0, r, epochs = 0, 0.0, 0.0, time.time(), 0, 0
    while step < a.steps:
        live = [x for x in reps if x.ready()]
        if not live:
            if all(x.k == 0 and x.t == 0 for x in reps):  # the generators have not written a chunk yet
                time.sleep(20)
                continue
            for x in reps:  # every stored chunk has been replayed: start over
                x.restart()
            epochs += 1
            continue
        rep = live[r % len(live)]; r += 1
        f, z = rep.step()
        n = len(f)
        idx = (ptr + torch.arange(n, device=dev)) % a.buffer
        buf_f[idx], buf_z[idx] = f, z
        ptr, fill = (ptr + n) % a.buffer, min(a.buffer, fill + n)
        if fill < min(a.buffer, 4096):
            continue
        credit += a.ratio
        while credit >= 1 and step < a.steps:
            credit -= 1
            i = torch.randint(0, fill, (a.batch,), device=dev)
            x = (buf_f[i].permute(0, 3, 1, 2).float() / 127.5 - 1.0).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = net(x)
            loss = F.mse_loss(y.float(), buf_z[i].float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            with torch.no_grad():
                for e, q in zip(ema.parameters(), net.parameters()):
                    e.lerp_(q, 0.002)
            step += 1; acc += loss.item()
            if step % 500 == 0 or step == a.steps:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ys = torch.cat([ema((hf[j:j + 64].permute(0, 3, 1, 2).float() / 127.5 - 1.0).contiguous(
                        memory_format=torch.channels_last)) for j in range(0, len(hf), 64)]).float()
                rec = {"step": step, "loss": round(acc / 500, 5), "held_mse": round(F.mse_loss(ys, hz.float()).item(), 5),
                       "held_rel": round((F.mse_loss(ys, hz.float()) / hz.float().var()).item(), 4),
                       "psnr_student": round(psnr(ae.decode(ys), hf), 2), "psnr_teacher": round(base, 2),
                       "it_s": round(500 / (time.time() - t0), 2), "epochs": epochs,
                       "mismatch": round(sum(x.bad for x in reps) / max(1, sum(x.n for x in reps)), 4)}
                acc, t0 = 0.0, time.time()
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                torch.save(ema.state_dict(), os.path.join(a.out, "enc.tmp"))
                os.replace(os.path.join(a.out, "enc.tmp"), os.path.join(a.out, "enc.pt"))
