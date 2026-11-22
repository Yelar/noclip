"""Fine-tune the autoencoder's DECODER on the simulator's frames (the encoder stays as it is, so datasets of latents
remain valid). The pretrained decoder was made for photographs; on these frames it smears wallpaper and tiles.

  python -m wm.ae_tune --ae dcae --out DIR [--steps 20000 --batch 16 --envs 64]

Frames come straight from the simulator running on the same GPU, so there is no dataset and no repetition. The
latents get a little noise before decoding, so that the decoder also copes with the world model's imperfect ones.
Loss: L1 + LPIPS. Writes DIR/decoder.pt (use as `--ae dcae,dec=DIR/decoder.pt`) and DIR/log.jsonl with the PSNR on
held-out frames, before and during.
"""
import argparse
import copy
import json
import os
import time

import torch

from wm import ae as AE


def decode(net, z, scale):
    """Frames from latents with a float32 decoder (the one being trained, or its running average)."""
    out = []
    for i in range(0, len(z), 8):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = net(z[i:i + 8].float() / scale)
        out.append(((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1))
    return torch.cat(out)


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean((1, 2, 3))
    return (10 * torch.log10(255 ** 2 / mse.clamp_min(1e-6))).mean().item()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ae", default="dcae")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--skip", type=int, default=2, help="simulator steps between training steps")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lpips", type=float, default=0.5)
    p.add_argument("--znoise", type=float, default=0.05, help="std of the noise added to latents, in units of their std")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    import lpips
    from sim import Explorer, Sim
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    ae = AE.load(a.ae, dev)
    dec = ae.dnet.decoder.float().requires_grad_(True).train()
    ema = copy.deepcopy(dec).requires_grad_(False).eval()
    perc = lpips.LPIPS(net="vgg").to(dev).eval().requires_grad_(False)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.03, final_div_factor=20)

    def world(seed, B):
        s, e = Sim(B, dev, seed), Explorer(B, dev, seed)
        s.reset()
        return s, e

    def frames(s, e, n):
        for _ in range(n):
            f = s.step(e.act(s))[0]
        return f

    vs, ve = world(990001, 32)
    held = torch.cat([frames(vs, ve, 25) for _ in range(4)])  # 128 held-out frames from other maps
    del vs, ve
    zh = ae.encode(held)
    base = psnr(decode(dec, zh, ae.dscale), held)
    sim, exp = world(a.seed, a.envs)
    frames(sim, exp, 20)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    print(json.dumps({"step": 0, "psnr_held": round(base, 2)}), flush=True)
    log.write(json.dumps({"step": 0, "psnr_held": round(base, 2)}) + "\n"); log.flush()
    zstd = zh.float().std().item()
    g = torch.Generator(device=dev).manual_seed(a.seed)
    t0, acc = time.time(), [0.0, 0.0]
    for step in range(1, a.steps + 1):
        f = frames(sim, exp, a.skip)
        f = f[torch.randperm(a.envs, device=dev, generator=g)[:a.batch]]
        x = f.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        with torch.no_grad():
            z = ae.encode(f).float()
            z = (z + a.znoise * zstd * torch.randn(z.shape, device=dev, generator=g)) / ae.dscale
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = dec(z)
        y = y.float()
        l1 = (y - x).abs().mean()
        lp = perc(y, x).mean()
        (l1 + a.lpips * lp).backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        with torch.no_grad():
            for e, q in zip(ema.parameters(), dec.parameters()):
                e.lerp_(q, 0.002)
        acc[0] += l1.item(); acc[1] += lp.item()
        if step % 500 == 0 or step == a.steps:
            rec = {"step": step, "l1": round(acc[0] / 500, 4), "lpips": round(acc[1] / 500, 4),
                   "psnr_held": round(psnr(decode(ema, zh, ae.dscale), held), 2), "it_s": round(500 / (time.time() - t0), 2)}
            acc, t0 = [0.0, 0.0], time.time()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save(ema.state_dict(), os.path.join(a.out, "decoder.tmp"))
            os.replace(os.path.join(a.out, "decoder.tmp"), os.path.join(a.out, "decoder.pt"))
