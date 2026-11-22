"""A released model: one folder, or a Hugging Face model repository, with everything that playing needs.

  config.json            the model's shape, the latents' mean and std, the autoencoder's name, how it was trained
  model.safetensors      the world model
  decoder.safetensors    the frame decoder: the autoencoder's own, fine-tuned on the simulator's frames (wm/ae_tune.py)
  encoder.safetensors    the fast frame encoder (wm/enc.py); only making datasets needs it
  events.safetensors     the reader of blows and deaths (wm/events.py)
  meta.json              level names, frame rate, how far a frame can turn
  start/                 recorded frames that dreams start from: a small dataset (README, Dataset format)
(and what its page shows: README.md, a picture, a video; playing fetches no video)

  python -m wm.play                          plays the released model (downloaded once into the Hugging Face cache)
  python -m wm.play --model DIR_OR_REPO      ... another one
  python -m wm.bundle OUT --ckpt RUN/final.pt --data DATASET --dec DECODER.pt [--enc ENCODER.pt] [--events EVENTS.pt]
                                             makes the folder OUT from a training run's files
"""
import argparse
import json
import os

import numpy as np

REPO = "yelarys/noclip"         # the released model's Hugging Face repository
SEG, PER_LEVEL = 24, 12         # start/: frames per recorded stretch, stretches per level
# columns of the dataset's files (sim/core.py, sim/gen.py)
ACT_NOCLIP, ACT_STATE = 6, (9, 10)
AUX_DYING, AUX_TRANS, AUX_SCREAM = 11, 12, 17


def fetch(model=REPO):
    """The folder of a released model: `model` itself if it is a folder, else the Hugging Face repository of that name."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model, ignore_patterns=["*.mp4"])      # (a repository may carry a video of the game)


def play_args(d):
    """What wm/play.py takes from a released model's folder, by the name of its command-line option."""
    def at(name):
        return os.path.join(d, name)
    with open(at("config.json")) as f:
        ae = json.load(f).get("ae", "dcae")
    out = {"ckpt": at("model.safetensors"), "meta": at("meta.json"),
           "ae": ae + (",dec=" + at("decoder.safetensors") if os.path.exists(at("decoder.safetensors")) else "")}
    if os.path.isdir(at("start")):
        out["data"] = at("start")
    if os.path.exists(at("events.safetensors")):
        out["events"] = at("events.safetensors")
    return out


def starts(data, mean, std, seed=0):
    """Stretches of SEG recorded frames to start dreams from, PER_LEVEL per level: lat, act, lvl, each [n, SEG, ...].
    A stretch stays in one level, has no blackout or red light, no fall, no blow and no death in it, and none of its
    frames is blank (the emptiest tenth of the frames looked at: a wall in the face, darkness)."""
    from wm import data as D
    rng = np.random.default_rng(seed)
    ch = D.Chunks(data, SEG)
    found, content = {}, []
    for lat, act, lvl, aux in ch.files:
        for b in rng.permutation(lat.shape[0]):
            a, l = np.asarray(act[b], np.float32), np.asarray(lvl[b]).astype(np.int64)
            x = (np.asarray(lat[b], np.float32) - mean[None, :, None, None]) / std[None, :, None, None]
            c = x.reshape(x.shape[0], x.shape[1], -1).std(2).mean(1)        # how much there is to see in each frame
            content.append(c)
            bad = a[:, ACT_NOCLIP] > 0.5
            for j in ACT_STATE:
                if a.shape[1] > j:
                    bad |= a[:, j] > 0.5
            if aux is not None:
                au = np.asarray(aux[b], np.float32)
                bad |= (au[:, AUX_DYING] > 0) | (au[:, AUX_TRANS] > 0)
                if au.shape[1] > AUX_SCREAM:
                    bad |= au[:, AUX_SCREAM] > 0
            for s in range(0, len(l) - SEG + 1, SEG):
                if not bad[s:s + SEG].any() and (l[s:s + SEG] == l[s]).all():
                    found.setdefault(int(l[s]), []).append((lat, act, lvl, int(b), s, float(c[s:s + SEG].min())))
    blank = float(np.quantile(np.concatenate(content), 0.10))
    lats, acts, lvls = [], [], []
    for L in sorted(found):
        ok = [f for f in found[L] if f[5] > blank]
        for i in rng.permutation(len(ok))[:PER_LEVEL]:
            lat, act, lvl, b, s, _ = ok[i]
            lats.append(np.asarray(lat[b, s:s + SEG]))
            acts.append(np.asarray(act[b, s:s + SEG], np.float32))
            lvls.append(np.asarray(lvl[b, s:s + SEG]))
        print(f"  level {L:2d}: {min(len(ok), PER_LEVEL)} of {len(ok)} stretches that qualify", flush=True)
    return np.stack(lats), np.stack(acts), np.stack(lvls)


def make(a):
    import torch
    from safetensors.torch import save_file

    from wm import data as D

    def plain(sd):
        return {k.removeprefix("_orig_mod."): v.detach().cpu().contiguous() for k, v in sd.items()}

    os.makedirs(os.path.join(a.out, "start"), exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    save_file(plain(ck["model"]), os.path.join(a.out, "model.safetensors"))
    with open(os.path.join(a.data.split(",")[0], "meta.json")) as f:
        meta = json.load(f)
    ae = str(meta.get("ae", "dcae")).split(",")[0]
    # The task's options as trained, without the places its files were in
    task = {k: (os.path.basename(str(v).rstrip("/")) if k in ("data", "val", "init") else v) for k, v in (ck.get("args") or {}).items()}
    cfg = {"model": ck["cfg"], "mean": [float(x) for x in ck["mean"]], "std": [float(x) for x in ck["std"]], "ae": ae,
           "parameters": int(sum(v.numel() for v in ck["model"].values())),
           "trained": {"frames": ck.get("tokens"), "rounds": ck.get("round"), "averaged_weights": ck.get("averaged"), "task": task}}
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    if a.dec:
        save_file(plain(torch.load(a.dec, map_location="cpu")), os.path.join(a.out, "decoder.safetensors"))
    if a.enc:
        save_file(plain(torch.load(a.enc, map_location="cpu")), os.path.join(a.out, "encoder.safetensors"))
    if a.events:
        save_file(plain(torch.load(a.events, map_location="cpu")["model"]), os.path.join(a.out, "events.safetensors"))
    # The lamps' states (action columns 9, 10) are left to 0 when playing: the lists that would let the play server's
    # director set them are not passed on (a model must have been trained to obey them, wm/play.py lights())
    keep = {k: v for k, v in meta.items() if k not in ("blackouts", "reds", "envs", "chunk")}
    keep["ae"] = ae
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(keep, f, indent=1)
    print("recorded stretches to start from:", flush=True)
    lat, act, lvl = starts(a.data, np.asarray(ck["mean"], np.float32), np.asarray(ck["std"], np.float32))
    for name, x in (("lat", lat), ("act", act), ("lvl", lvl)):
        np.save(os.path.join(a.out, "start", f"start-00000.{name}.npy"), x)
    with open(os.path.join(a.out, "start", "stats.json"), "w") as f:
        json.dump({"mean": cfg["mean"], "std": cfg["std"]}, f)
    for root, _, files in os.walk(a.out):
        for n in sorted(files):
            p = os.path.join(root, n)
            if n.startswith(".tmp"):        # left by a writer that was cut off: it must not travel with the release
                os.remove(p)
                continue
            print(f"  {os.path.getsize(p) / 1e6:9.1f} MB  {os.path.relpath(p, a.out)}")
    D.Chunks(os.path.join(a.out, "start"), 8, A=ck["cfg"]["A"]).batch(1, np.random.default_rng(0))    # it reads back


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--ckpt", required=True, help="a checkpoint of wm/task.py (wm/train.py's model.pt, wm/export.py's)")
    p.add_argument("--data", required=True, help="a dataset of the model's kind: its meta.json, and the frames to start from")
    p.add_argument("--dec", default="", help="a decoder tuned by wm/ae_tune.py")
    p.add_argument("--enc", default="", help="a fast encoder trained by wm/enc.py")
    p.add_argument("--events", default="", help="the reader trained by wm/events.py")
    make(p.parse_args())
