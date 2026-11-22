"""Frame autoencoders: a frame (uint8, H x W x 3) <-> a small grid of latent vectors the world model works in.

Pretrained ones, frozen for encoding, so a dataset of latents stays valid; the decoder may be replaced by one
fine-tuned on the simulator's frames, and the encoder by a fast one trained to give the same latents
(`name,enc=PATH,dec=PATH`); `name,decnet=OTHER` decodes with another autoencoder of the same latent space.

  ae = load("dcae", "cuda");  z = ae.encode(frames_uint8_NHWC);  frames = ae.decode(z)
"""
import os

import torch


def weights(path, device="cpu"):
    """A state dict from a .safetensors file (the released ones) or a file written by torch.save."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=str(device))
    return torch.load(path, map_location=device)


# name: (kind, Hugging Face repo, downsampling factor, latent channels)
KNOWN = {
    "dcae": ("dc", "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", 32, 32),
    "dcae-in": ("dc", "mit-han-lab/dc-ae-f32c32-in-1.0-diffusers", 32, 32),
    "dcae-lite": ("dc", "mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers", 32, 32),
    "dcae-f64": ("dc", "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers", 64, 128),
    "sdvae": ("kl", "stabilityai/sd-vae-ft-mse", 8, 4),
    "taesd": ("tiny", "madebyollin/taesd", 8, 4),
}


class AE:
    def __init__(self, name, device="cuda", dtype=None, batch=64):
        name, *opts = name.split(",")
        opts = dict(o.split("=", 1) for o in opts)  # enc=PATH: the fast encoder of wm/enc.py; dec=PATH: a tuned decoder
        kind, repo, self.down, self.C = KNOWN[name]
        self.name, self.kind, self.device, self.batch = name, kind, device, batch
        self.dtype = dtype or (torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
        import diffusers
        cls = {"dc": diffusers.AutoencoderDC, "kl": diffusers.AutoencoderKL, "tiny": diffusers.AutoencoderTiny}[kind]
        self.net = cls.from_pretrained(repo, torch_dtype=torch.float32).to(device, self.dtype).eval().requires_grad_(False)
        self.scale = float(getattr(self.net.config, "scaling_factor", 1.0) or 1.0)
        self.student = None
        self.dnet, self.dscale = self.net, self.scale
        if "decnet" in opts:  # decode with another autoencoder of the same latent space (taesd encodes into sdvae's)
            k2, repo2, _, _ = KNOWN[opts["decnet"]]
            cls2 = {"dc": diffusers.AutoencoderDC, "kl": diffusers.AutoencoderKL, "tiny": diffusers.AutoencoderTiny}[k2]
            self.dnet = cls2.from_pretrained(repo2, torch_dtype=torch.float32).to(device, self.dtype).eval().requires_grad_(False)
            self.dscale = float(getattr(self.dnet.config, "scaling_factor", 1.0) or 1.0)
            self.dnet.encoder = None
        if "dec" in opts:  # a decoder fine-tuned on the simulator's frames (wm/ae_tune.py)
            self.dnet.decoder.load_state_dict(weights(opts["dec"], device))
        if "enc" in opts:
            from wm.enc import Student
            self.student = Student(self.C).to(device)
            self.student.load_state_dict(weights(opts["enc"], device))
            self.student = self.student.to(self.dtype).to(memory_format=torch.channels_last).eval().requires_grad_(False)
            self.net.encoder = None  # not used any more: frees its memory
        self._encode = self._enc
        if os.environ.get("WM_AE_COMPILE", "0") == "1" and device != "cpu":  # worth it for long encoding jobs
            torch.backends.cudnn.benchmark = True
            if self.student is not None:
                self.student = torch.compile(self.student)
            else:
                self.net.encoder = torch.compile(self.net.encoder)

    def _enc(self, x):
        if self.kind == "dc":
            return self.net.encode(x).latent
        if self.kind == "kl":
            return self.net.encode(x).latent_dist.mean
        return self.net.encode(x).latents

    @torch.no_grad()
    def encode(self, frames):
        """uint8 [N, H, W, 3] (any device) -> float16 [N, C, H/down, W/down] on the autoencoder's device."""
        out = []
        for i in range(0, len(frames), self.batch):
            x = frames[i:i + self.batch].to(self.device).permute(0, 3, 1, 2).to(self.dtype) / 127.5 - 1.0
            if self.student is not None:
                out.append(self.student(x.contiguous(memory_format=torch.channels_last)).to(torch.float16))
            else:
                out.append((self._enc(x) * self.scale).to(torch.float16))
        return torch.cat(out)

    @torch.no_grad()
    def decode(self, z):
        """latents [N, C, h, w] -> uint8 [N, H, W, 3] on the autoencoder's device."""
        out = []
        for i in range(0, len(z), self.batch):
            x = self.dnet.decode(z[i:i + self.batch].to(self.device, self.dtype) / self.dscale).sample
            out.append(((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1))
        return torch.cat(out)


def load(name, device="cuda", **kw):
    return AE(name, device, **kw)


if __name__ == "__main__":  # python -m wm.ae FRAMES.npy [names...]: reconstruction quality and speed on real frames
    import sys
    import time

    import numpy as np
    src = np.load(sys.argv[1], mmap_mode="r")
    fr = torch.from_numpy(np.ascontiguousarray(src.reshape(-1, *src.shape[-3:])[:: max(1, src[..., 0, 0, 0].size // 256)][:256]))
    for name in sys.argv[2:] or ["dcae"]:
        for dt in (torch.bfloat16, torch.float32):
            ae = load(name, "cuda", dtype=dt, batch=int(os.environ.get("WM_AE_BATCH", "64")))
            ae.encode(fr[:8]); torch.cuda.synchronize(); t0 = time.time()
            z = ae.encode(fr); torch.cuda.synchronize(); t1 = time.time()
            rec = ae.decode(z); torch.cuda.synchronize(); t2 = time.time()
            mse = ((rec.float().cpu() - fr.float()) ** 2).mean((1, 2, 3))
            psnr = (10 * torch.log10(255 ** 2 / mse.clamp_min(1e-6))).mean().item()
            print(f"{name:10s} {str(dt)[6:]:9s} latent {tuple(z.shape[1:])} std {z.float().std():.3f}  psnr {psnr:.2f} dB  "
                  f"enc {len(fr) / (t1 - t0):.0f} f/s  dec {len(fr) / (t2 - t1):.0f} f/s", flush=True)
            if dt == torch.float32:
                out = sys.argv[1].replace(".npy", f".{name.split(',')[0]}.rec.npy")
                np.save(out, rec[:32].cpu().numpy())
            del ae
