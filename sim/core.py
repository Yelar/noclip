"""The simulator: B Backrooms episodes at once on one GPU, rendered by ray casting.

How a frame is drawn. Walls stand on a grid, so which wall a ray hits depends only on its compass direction: a fan
of NA horizontal rays per episode is walked through the grid (DDA) and records the first full wall and up to KEV
blocks of partial height on the way (low walls, lintels). Every pixel then looks up its fan ray and decides, from its
own slope, whether it hits one of those, the floor or the ceiling; so the per-pixel work has no loop. Sprites are
upright billboards resolved the same way (two layers per fan ray). Light comes from a baked light map per episode
(with occlusion), a few time-varying channels for lamps that flicker, pulse or spark, and the flashlight.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import tex
from .levels import (E_HORDE, E_HOUND, E_NONE, E_PARTY, E_SMILER, E_STALKER, E_WATCHER, G, LEVELS, MAT_ID, MATS, NCH,
                     NK, SPR_ID, SPRITES, make)

ACTIONS = ["forward", "right", "turn", "pitch", "sprint", "light", "noclip", "use"]
AUX = ["x", "z", "yaw", "pitch", "speed", "eye", "light", "ent_dist", "ent_visible", "ent_type", "danger", "dying",
       "trans", "brightness", "episode", "door", "hp", "scream", "blackout", "red"]
# blackout: 1 from the moment a wave of lamps going out starts towards the player until the lamps come back (11 to
# 21 s); red: 1 while the lamps are red. Neither can be told from a frame for as long as it lasts (darkness looks the
# same in its first second and its last), so the generator hands both to the model as conditions (sim/gen.py).
FACELESS = ("office", "hotel", "mall", "city", "apartments")      # levels whose figures are mostly the faceless kind

TURN_MAX = 0.6          # radians per frame at turn = 1
PITCH_MAX = 0.35
PITCH_LIM = 0.8
WALK, SPRINT, BACK, STRAFE = 1.4, 3.0, 1.1, 1.2
EYE = 1.6
RAD = 0.22
FOV = math.radians(80)
NA = 2048               # fan rays
FAN_HALF = 1.27         # the fan spans +- this many radians around the heading
FAR = 48.0
NSTEP = 66
KEV = 2                 # partial-height blocks remembered per fan ray
NP, NPV, NE = 256, 16, 6
ND = 64                 # doors per map that the explorer and the rooms' dwellers know of
REACH = 1.9             # how far E reaches
NFAN = 9                # fan channels of the full wall, before those of the partial blocks (NEVT each)
NEVT = 10
NTER = 56               # steps of the march that finds ground that is not flat (0.25 m at first, growing, to about 60 m)
SKY_BG = 0.30           # the backdrop on the horizon reaches this many radians up
LG = 2 * G              # light map: 0.5 m texels
LMC = 9                 # 0-2 steady rgb, 3-6 channels 1..4, 7 occlusion, 8 stains
TR_NOCLIP, TR_DEATH = 8, 20
SCR, INV = 6, 20        # frames the attacker's face is on the screen; frames nothing can strike again after a blow
REGEN = 0.004           # health that comes back per second
DMG = [0.0, 0.0, 0.6, 0.6, 0.45, 1.0, 0.6]     # health a blow takes, by entity type (the horde ends it at once)
NTR = 64                # trail of past positions, one every 0.5 s
FL_P = 2.4
EXPO = 1.15
M = G - 1
TS, NMIP = tex.TS, tex.NMIP

# transition tables: per frame of the transition (eye height as a fraction: >=0 of EYE in the old map, the same
# measured down from the ceiling in the new one when negative), brightness, glitch strength
_NC_EYE = [0.85, 0.5, 0.12, 0.05, 0.05, -0.85, -0.35, 1.0]
_NC_FADE = [1.0, 0.8, 0.35, 0.0, 0.0, 0.45, 0.85, 1.0]
_NC_GL = [0.5, 1.0, 1.0, 0.6, 0.3, 0.5, 0.2, 0.0]
_NC_LOAD = 4
_DE_EYE = [1.0, 0.95, 0.8, 0.5] + [0.3] * 13 + [0.45, 0.75, 1.0]
_DE_FADE = [0.9, 0.6, 0.3, 0.1] + [0.0] * 12 + [0.25, 0.5, 0.8, 1.0]
_DE_GL = [0.3, 0.6, 0.9, 1.0, 0.5] + [0.0] * 10 + [0.2, 0.3, 0.1, 0.0, 0.0]
_DE_LOAD = 10
_SCR_COVER = [0.85, 1.0, 1.0, 1.0, 0.9, 0.5]            # how much of the picture the face takes, per frame of it
_SCR_FLASH = [0.6, 0.15, 0.0, 0.0, 0.0, 0.0]


def _wrap(d):
    return (d + G / 2) % G - G / 2


def _blur(x, sigma):
    """Gaussian blur of [N, C, H, W] that wraps around."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=torch.float32) ** 2 / (2 * sigma * sigma))
    k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="circular"), k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="circular"), k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


class Sim:
    def __init__(self, B, device="cuda", seed=0, res=(288, 512), fps=10, levels=None):
        self.B, self.dev = B, torch.device(device)
        self.H, self.W = res
        self.fps, self.dt = fps, 1.0 / fps
        self.seed = int(seed)
        self.levels = list(range(len(LEVELS))) if levels is None else [int(l) for l in levels]
        dev = self.dev
        self.g = torch.Generator(device=dev)
        self.g.manual_seed(self.seed * 7919 + 13)
        at = tex.build(dev)
        self.atlas, self.base, names = at["atlas"], at["base"], at["names"]
        self.mipsize = torch.tensor([TS >> l for l in range(NMIP)], device=dev)
        # materials: one row per property so that a single index gives all of them
        NM = len(MATS)
        self.NM = NM
        mt = torch.zeros(13, NM)
        for i, m in enumerate(MATS.values()):
            mt[:, i] = torch.tensor([names[m["tex"]], 1.0 / m["size"], (1.0 / m["v"]) if m["v"] > 0 else 0.0,
                                     float(m["v"] <= 0), m["em"], m["ch"], m["gloss"], m["water"], float(m["puddle"]),
                                     float(m["env"]), *m["emcol"]])
        self.mtab = mt.to(dev)
        self.m_emcol = self.mtab[10:13].t().contiguous()
        # sprite types (row 0 = none)
        st = torch.zeros(len(SPRITES) + 1, 5)
        for n, (t, w, h, base, emis, _blk) in SPRITES.items():
            st[SPR_ID[n]] = torch.tensor([names[t], w, h, base, emis])
        self.stab = st.to(dev)
        # camera
        th = math.tan(FOV / 2)
        self.px = ((torch.arange(self.W, device=dev) + 0.5) / self.W * 2 - 1).view(1, 1, -1) * th
        self.py = -((torch.arange(self.H, device=dev) + 0.5) / self.H * 2 - 1).view(1, -1, 1) * th * self.H / self.W
        self.pixang = 2 * th / self.W
        self.fan_az = torch.linspace(-FAN_HALF, FAN_HALF, NA, device=dev)
        r = torch.sqrt(self.px ** 2 + (self.py + 0.03) ** 2)[0]
        self.spot = (tex.sstep(0.46, 0.20, r) + 0.22 * tex.sstep(1.0, 0.3, r) + 0.02)
        rr = torch.sqrt((self.px / th) ** 2 + (self.py / th) ** 2)[0]
        self.vig = 1 - 0.28 * tex.sstep(0.55, 1.25, rr)
        self.dvig = tex.sstep(0.25, 1.1, rr)
        self.fl_col = torch.tensor([1.0, 0.96, 0.86], device=dev).view(1, 3, 1, 1)
        ar = torch.arange(B, device=dev)
        self.ar = ar
        self.boff = (ar * G * G).view(B, 1, 1)
        self.boff2 = (ar * 2 * G * G).view(B, 1, 1)
        self.koff = (ar * NK).view(B, 1)
        self.moff = (ar * NM).view(B, 1, 1)
        self.doff = (ar * NCH).view(B, 1, 1)
        f, z = torch.zeros, torch.long
        self.wall = f(B, G, G, dtype=z, device=dev)
        self.fc = f(B, 2, G, G, dtype=z, device=dev)
        self.emit = f(B, G, G, device=dev)
        self.echan = f(B, G, G, dtype=z, device=dev)
        self.block = f(B, G, G, dtype=torch.bool, device=dev)
        self.k_side, self.k_top, self.k_bot = f(B, NK, device=dev), f(B, NK, device=dev), f(B, NK, device=dev)
        self.k_y0, self.k_y1 = f(B, NK, device=dev), f(B, NK, device=dev)
        self.k_full = f(B, NK, dtype=torch.bool, device=dev)
        self.k_block = f(B, NK, dtype=torch.bool, device=dev)
        self.k_split, self.k_up = torch.full((B, NK), 1e9, device=dev), f(B, NK, device=dev)
        self.k_door, self.k_nxt, self.k_prv = (f(B, NK, dtype=z, device=dev) for _ in range(3))
        self.pblock = f(B, G, G, dtype=torch.bool, device=dev)
        self.d_cell = torch.full((B, ND), -1, dtype=z, device=dev)
        self.d_front, self.d_nrm, self.d_room = (f(B, ND, 2, device=dev) for _ in range(3))
        self.d_haunt = f(B, ND, dtype=z, device=dev)
        self.mv_cell, self.mv_dir = torch.full((B,), -1, dtype=z, device=dev), f(B, dtype=z, device=dev)
        self.door_now = f(B, dtype=z, device=dev)
        self.outdoor = f(B, dtype=torch.bool, device=dev)
        self.sky_tex = f(B, dtype=z, device=dev)
        self.sky_h, self.sky_z = f(B, 3, device=dev), f(B, 3, device=dev)
        self.sky_c, self.sky_s, self.sunk = f(B, device=dev), f(B, device=dev), f(B, device=dev)
        self.sun = f(B, 2, device=dev)
        self.sky_w, self.sky_bg = f(B, device=dev), torch.full((B,), -1, dtype=z, device=dev)
        # ground that is not flat: height and slope per cell (padded, for sampling that wraps), height per cell, under the player
        self.hilly = f(B, dtype=torch.bool, device=dev)
        self.hmap, self.hcell, self.gh = f(B, 3, G + 2, G + 2, device=dev), f(B, G * G, device=dev), f(B, device=dev)
        self.tex_names = names
        # health, and what is drawn over the finished picture
        self.hp = torch.ones(B, device=dev)
        self.scr, self.inv, self.scr_tex = (f(B, dtype=z, device=dev) for _ in range(3))
        self.dmg = torch.tensor(DMG, device=dev)
        hs = self.W / 512
        g_ = torch.tensor([[int(v) for v in row] for row in tex._FONT["E"].split()], device=dev)
        gs = max(1, round(2 * hs))
        gl = torch.ones(7 * gs + 6, 5 * gs + 6, dtype=z, device=dev)                    # 1: the box, 2: the letter
        gl[3:-3, 3:-3] = 1 + g_.repeat_interleave(gs, 0).repeat_interleave(gs, 1)
        self.glyph, self.glyph_at = gl, (int(self.H * 0.62), self.W // 2 - gl.shape[1] // 2)
        self.glyph_col = torch.tensor([[0, 0, 0], [16, 16, 16], [235, 235, 235]], dtype=torch.uint8, device=dev)
        self.ov_x = (((torch.arange(self.W, device=dev) + 0.5) / self.W - 0.5) * (self.W / self.H)).view(1, -1).expand(self.H, self.W)
        self.ov_y = ((torch.arange(self.H, device=dev) + 0.5) / self.H - 0.5).view(-1, 1).expand(self.H, self.W)
        self.lm = f(B, LMC, LG + 2, LG + 2, device=dev)
        self.Hc, self.fog, self.grime = f(B, device=dev), f(B, device=dev), f(B, device=dev)
        self.fogcol, self.amb, self.lcol, self.staincol = (f(B, 3, device=dev) for _ in range(4))
        self.dyncol = f(B, NCH, 3, device=dev)
        self.tint = f(B, NM, 3, device=dev)
        self.pulse = f(B, 4, device=dev)
        self.spark, self.wave_rate, self.flashp = f(B, device=dev), f(B, device=dev), f(B, device=dev)
        self.rates = f(B, 7, device=dev)
        self.axis = f(B, device=dev)
        self.pp, self.pt = f(B, NP, 2, device=dev), f(B, NP, dtype=z, device=dev)
        self.e_type, self.e_spr = f(B, NE, dtype=z, device=dev), f(B, NE, dtype=z, device=dev)
        self.e_pos = f(B, NE, 2, device=dev)
        self.e_vel = f(B, NE, 2, device=dev)
        self.e_seen, self.e_unseen, self.e_age, self.e_lit, self.e_tr = (f(B, NE, dtype=z, device=dev) for _ in range(5))
        self.e_alpha = f(B, NE, device=dev)
        self.e_near = torch.full((B, NE), 4.5, device=dev)      # a watcher fades when the player comes this close
        self.e_fade = f(B, NE, dtype=torch.bool, device=dev)
        self.pos, self.vel = f(B, 2, device=dev), f(B, 2, device=dev)
        self.yaw, self.pitch, self.bob, self.roll = (f(B, device=dev) for _ in range(4))
        self.eye_now = torch.full((B,), EYE, device=dev)
        self.light = f(B, dtype=torch.bool, device=dev)
        self.lvl, self.ep = f(B, dtype=z, device=dev), f(B, dtype=z, device=dev)
        self.trans, self.trans_kind = f(B, dtype=z, device=dev), f(B, dtype=z, device=dev)
        self.trail = f(B, NTR, 2, device=dev)
        self.dynm = torch.ones(B, NCH, device=dev)
        self.on1 = torch.ones(B, dtype=torch.bool, device=dev)
        self.on2 = f(B, dtype=torch.bool, device=dev)
        self.w_act = f(B, dtype=torch.bool, device=dev)
        self.w_org, self.w_dir = f(B, 2, device=dev), f(B, 2, device=dev)
        self.w_front, self.w_hold = f(B, device=dev), f(B, device=dev)
        self.red_t, self.red_ph, self.red_rate = f(B, device=dev), f(B, device=dev), f(B, device=dev)
        self.ltint = torch.ones(B, 3, device=dev)
        self.danger, self.bright = f(B, device=dev), f(B, device=dev)
        self.fade = torch.ones(B, device=dev)
        self.glitch = f(B, device=dev)
        self.red = f(B, device=dev)
        self.ent_dist, self.ent_vis, self.ent_typ = f(B, device=dev), f(B, device=dev), f(B, device=dev)
        self.t = 0
        self.tr_count = 1
        self._tab = {k: torch.tensor(v, device=dev) for k, v in dict(
            nc_eye=_NC_EYE, nc_fade=_NC_FADE, nc_gl=_NC_GL, de_eye=_DE_EYE, de_fade=_DE_FADE, de_gl=_DE_GL,
            scr_cover=_SCR_COVER, scr_flash=_SCR_FLASH).items()}
        self._started = False
        self.prof = None
        self._speed = torch.zeros(B, device=dev)

    # ------------------------------------------------------------------ random numbers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.dev)

    def _tick(self, name):
        if self.prof is not None:
            import time
            torch.cuda.synchronize()
            now = time.time()
            self.prof[name] = self.prof.get(name, 0.0) + now - getattr(self, "_t_last", now)
            self._t_last = now

    # ------------------------------------------------------------------ maps
    def reset(self):
        """Start every episode in a fresh map of a random level. Returns the first frame, lvl and aux."""
        lv = torch.tensor(self.levels, device=self.dev)[torch.randint(len(self.levels), (self.B,), generator=self.g, device=self.dev)]
        self.t, self.tr_count = 0, 1
        self.trans.zero_(); self.fade.fill_(1); self.glitch.zero_(); self.red.zero_(); self.danger.zero_()
        self.ent_dist.fill_(99.0)
        self.hp.fill_(1.0); self.scr.zero_(); self.inv.zero_()
        for b in range(self.B):
            self.lvl[b] = lv[b]
            self._load(b, int(lv[b]))
        self._started = True
        self._lights()
        self.gh = self._h_points(self.pos[:, None])[:, 0]
        self.eye_now = torch.full((self.B,), EYE, device=self.dev) + self.gh
        frame = self._render()
        return frame, self.lvl.clone(), self._aux()

    def _load(self, b, level):
        dev = self.dev
        mp = make(level, self.seed * 1000003 + b * 7919 + int(self.ep[b]))
        self.ep[b] += 1
        H = float(mp.H)
        tn = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        self.wall[b] = tn(mp.w, torch.long)
        self.fc[b, 0] = tn(mp.floor, torch.long)
        self.fc[b, 1] = tn(mp.ceil, torch.long)
        self.emit[b] = tn(mp.emit)
        self.echan[b] = tn(mp.echan, torch.long)
        ks = mp.kinds + [mp.kinds[0]] * (NK - len(mp.kinds))
        self.k_side[b] = tn([k["side"] for k in ks]); self.k_top[b] = tn([k["top"] for k in ks])
        self.k_bot[b] = tn([k["bot"] for k in ks])
        self.k_y0[b] = tn([k["y0"] for k in ks]); self.k_y1[b] = tn([(k["y1"] if k["y1"] > 0 else H) for k in ks])
        self.k_y0[b, 0], self.k_y1[b, 0] = 1e9, -1e9
        for i in range(len(mp.kinds), NK):
            self.k_y0[b, i], self.k_y1[b, i] = 1e9, -1e9
        self.k_full[b] = tn([k["full"] for k in ks], torch.bool)
        self.k_block[b] = tn([k["block"] for k in ks], torch.bool)
        self.k_split[b] = tn([k["split"] for k in ks]); self.k_up[b] = tn([k["up"] for k in ks])
        self.k_door[b] = tn([k["door"] for k in ks], torch.long)
        self.k_nxt[b] = tn([k["nxt"] for k in ks], torch.long); self.k_prv[b] = tn([k["prv"] for k in ks], torch.long)
        self.pblock[b] = tn(mp.pblock, torch.bool)
        self.block[b] = self.k_block[b][self.wall[b]] | self.pblock[b]
        self.d_cell[b], self.d_haunt[b] = -1, 0
        if mp.doors:
            a = tn(mp.doors[:ND])
            n_ = a.shape[0]
            self.d_cell[b, :n_] = torch.floor(a[:, 1]).long() * G + torch.floor(a[:, 0]).long()
            self.d_front[b, :n_], self.d_nrm[b, :n_], self.d_room[b, :n_] = (a[:, 0:2] + a[:, 2:4]) % G, a[:, 2:4], a[:, 4:6]
            self.d_haunt[b, :n_] = a[:, 6].long()
        self.mv_cell[b], self.mv_dir[b], self.door_now[b] = -1, 0, 0
        self.outdoor[b] = bool(mp.outdoor)
        st_, sh_, sz_, sc_, ss_ = mp.sky
        self.sky_w[b], self.sky_bg[b] = mp.clouds, (self.tex_names[mp.backdrop] if mp.backdrop else -1)
        self.hilly[b] = mp.hmap is not None
        if mp.hmap is None:
            self.hmap[b].zero_(); self.hcell[b].zero_()
        else:
            hm = tn(mp.hmap)
            gx_, gz_ = (torch.roll(hm, -1, 1) - torch.roll(hm, 1, 1)) * 0.5, (torch.roll(hm, -1, 0) - torch.roll(hm, 1, 0)) * 0.5
            self.hmap[b] = F.pad(torch.stack([hm, gx_, gz_])[None], (1, 1, 1, 1), mode="circular")[0]
            self.hcell[b] = hm.reshape(-1)
        self.sky_tex[b], self.sky_c[b], self.sky_s[b] = self.tex_names[st_], sc_, ss_
        self.sky_h[b], self.sky_z[b] = tn(sh_), tn(sz_)
        self.sun[b, 0], self.sun[b, 1], self.sunk[b] = mp.sun
        self.Hc[b], self.fog[b], self.grime[b] = H, mp.fog, mp.grime
        self.fogcol[b], self.amb[b], self.lcol[b], self.staincol[b] = tn(mp.fogcol), tn(mp.amb), tn(mp.lcol), tn(mp.staincol)
        self.tint[b] = tn(mp.tint)
        self.pulse[b] = tn(mp.pulse)
        self.spark[b], self.wave_rate[b], self.flashp[b] = mp.spark, mp.wave_rate, mp.flash
        rt = torch.zeros(7, device=dev)
        for k, v in mp.rates.items():
            rt[k] = v
        self.rates[b] = rt
        self.axis[b] = 0.0 if mp.axis is None else float(mp.axis[1])
        self.red_rate[b], self.red_t[b], self.red_ph[b] = mp.red_rate, (1e9 if mp.red_perm else 0.0), 10.0
        # light channels take the colour of their lamps
        dc = torch.zeros(NCH, 3, device=dev)
        cnt = torch.zeros(NCH, device=dev)
        for (_x, _z, r_, g_, b_, p_, _rad, ch) in mp.lights:
            dc[int(ch)] += torch.tensor([r_, g_, b_], device=dev)
            cnt[int(ch)] += 1
        self.dyncol[b] = dc / cnt.clamp(min=1)[:, None]
        self._bake(b, mp)
        # props
        self.pt[b].zero_()
        if mp.props:
            pr = mp.props
            if len(pr) > NP:
                pr = [pr[i] for i in mp.rng.choice(len(pr), NP, replace=False)]
            a = tn(pr)
            self.pp[b, : len(pr)] = a[:, :2] % G
            self.pt[b, : len(pr)] = a[:, 2].long()
        # player
        x, z, yaw = mp.spawn
        self.pos[b, 0], self.pos[b, 1], self.yaw[b] = float(x), float(z), float(yaw)
        self.pitch[b], self.bob[b], self.roll[b] = 0.0, 0.0, 0.0
        self.vel[b].zero_()
        self.trail[b] = self.pos[b]
        self.on1[b], self.on2[b] = True, False
        self.w_act[b] = False
        self.light[b] = bool(mp.rng.random() < mp.flash)
        # entities
        self.e_type[b].zero_(); self.e_alpha[b].zero_(); self.e_fade[b] = False
        if mp.name == "run":
            self._spawn_at(b, 0, E_HORDE, x - 9.0 * math.cos(yaw), z - 9.0 * math.sin(yaw), "horde")
        if mp.n_party:
            rz, rx = np.nonzero(mp.reach)
            d = np.hypot(_wrap(rx + 0.5 - x), _wrap(rz + 0.5 - z))
            ok = np.nonzero((d > 5) & (d < 20))[0]
            for i, j in enumerate(mp.rng.permutation(ok)[: min(mp.n_party, NE - 1)]):
                self._spawn_at(b, i, E_PARTY, rx[j] + 0.5, rz[j] + 0.5, "party")

    def _spawn_at(self, b, slot, typ, x, z, spr):
        self.e_type[b, slot] = typ
        self.e_spr[b, slot] = SPR_ID[spr]
        self.e_pos[b, slot, 0], self.e_pos[b, slot, 1] = float(x) % G, float(z) % G
        self.e_seen[b, slot] = 0; self.e_unseen[b, slot] = 0; self.e_age[b, slot] = 0; self.e_lit[b, slot] = 0
        self.e_tr[b, slot] = self.tr_count - 1
        self.e_alpha[b, slot] = 1.0
        self.e_fade[b, slot] = False
        self.e_vel[b, slot] = 0.0
        self.e_near[b, slot] = 4.5

    def _bake(self, b, mp):
        dev = self.dev
        opq = torch.from_numpy(np.array([k["full"] for k in mp.kinds])[mp.w]).to(dev)
        lm = torch.zeros(7, LG * LG)
        if mp.lights:
            L = torch.tensor(mp.lights, dtype=torch.float32, device=dev)
            P = int(math.ceil(float(L[:, 6].max()) / 0.5))
            o = torch.arange(-P, P + 1, device=dev)
            oz, ox = torch.meshgrid(o, o, indexing="ij")
            lt = torch.floor(L[:, :2] / 0.5).long()
            tx = lt[:, 0:1] + ox.reshape(1, -1)
            tz = lt[:, 1:2] + oz.reshape(1, -1)
            dx = (tx.float() + 0.5) * 0.5 - L[:, 0:1]
            dz = (tz.float() + 0.5) * 0.5 - L[:, 1:2]
            d2 = dx * dx + dz * dz
            fall = L[:, 5:6] / (1 + d2 / 1.6) * (1 - d2 / L[:, 6:7] ** 2).clamp(min=0) ** 2
            vis = torch.ones_like(fall, dtype=torch.bool)
            for j in range(16):
                fr = (j + 0.5) / 16
                cx = torch.floor(L[:, 0:1] + dx * fr).long() % G
                cz = torch.floor(L[:, 1:2] + dz * fr).long() % G
                vis &= ~opq[cz, cx]
            val = fall * vis
            ch = L[:, 7].long()
            vals = torch.zeros(7, *val.shape, device=dev)
            st = (ch == 0).float()[:, None]
            for c in range(3):
                vals[c] = val * st * L[:, 2 + c: 3 + c]
            for c in range(1, NCH):
                vals[2 + c] = val * (ch == c).float()[:, None]
            idx = ((tz % LG) * LG + (tx % LG)).reshape(-1)
            # summed on the CPU: adding into repeated indices on the GPU is not reproducible
            lm.index_add_(1, idx.cpu(), vals.reshape(7, -1).cpu())
        lmg = lm.to(dev).view(1, 7, LG, LG)
        lmg = lmg + 0.45 * _blur(lmg, 3.0)                                  # light that has bounced once
        opq2 = opq.repeat_interleave(2, 0).repeat_interleave(2, 1)
        of = (~opq2).float().view(1, 1, LG, LG)
        k3 = lambda x: F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="circular"), 3, 1)
        fill = k3(lmg * of) / k3(of).clamp(min=1e-3)
        lmg = torch.where(opq2.view(1, 1, LG, LG), fill, lmg)               # texels inside walls take their neighbours' light
        occ = _blur(1 - of, 1.1)
        n = mp.rng.standard_normal((LG, LG)).astype(np.float32)
        fy, fx = np.meshgrid(np.fft.fftfreq(LG, 1 / LG), np.fft.rfftfreq(LG, 1 / LG), indexing="ij")
        ff = np.sqrt(fy * fy + fx * fx)
        sp = np.fft.rfft2(n) * np.where((ff >= 1) & (ff <= 24), np.maximum(ff, 1) ** -1.3, 0)
        st_ = np.fft.irfft2(sp, s=(LG, LG))
        st_ = torch.from_numpy((st_ / (st_.std() + 1e-8)).astype(np.float32)).to(dev)
        stain = 0.5 * (1 + torch.erf(st_ / 1.4142)).view(1, 1, LG, LG)
        self.lm[b] = F.pad(torch.cat([lmg, occ, stain], 1), (1, 1, 1, 1), mode="circular")[0]

    # ------------------------------------------------------------------ rays through the grid
    def _cast(self, o, ang, steps, table, events=False):
        """Walk rays from o [B, 2] at absolute angles ang [B, N] through the grid until a cell whose kind is set in
        `table` [B, NK]. Returns distance (FAR if none), whether an x-line was crossed last, the kind hit, and, with
        events, the partial-height runs passed on the way."""
        B, N = ang.shape
        dev = self.dev
        dx, dz = torch.cos(ang), torch.sin(ang)
        dx = torch.where(dx.abs() < 1e-6, torch.full_like(dx, 1e-6), dx)
        dz = torch.where(dz.abs() < 1e-6, torch.full_like(dz, 1e-6), dz)
        ox, oz = o[:, 0:1], o[:, 1:2]
        mx = torch.floor(ox).long().expand(B, N).clone()
        mz = torch.floor(oz).long().expand(B, N).clone()
        sx, sz = torch.sign(dx).long(), torch.sign(dz).long()
        tdx, tdz = 1.0 / dx.abs(), 1.0 / dz.abs()
        fx, fz = ox - torch.floor(ox), oz - torch.floor(oz)
        tmx = torch.where(dx > 0, 1 - fx, fx) * tdx
        tmz = torch.where(dz > 0, 1 - fz, fz) * tdz
        wflat, tflat = self.wall.view(-1), table.reshape(-1)
        boff, koff = self.boff.view(B, 1), self.koff
        tw = torch.full((B, N), FAR, device=dev)
        sw = torch.zeros(B, N, dtype=torch.bool, device=dev)
        kw = torch.zeros(B, N, dtype=torch.long, device=dev)
        done = torch.zeros(B, N, dtype=torch.bool, device=dev)
        if events:
            fullf = self.k_full.view(-1)
            e_in = [torch.full((B, N), FAR, device=dev) for _ in range(KEV)]
            e_out = [torch.full((B, N), FAR, device=dev) for _ in range(KEV)]
            e_side = [torch.zeros(B, N, dtype=torch.bool, device=dev) for _ in range(KEV)]
            e_kind = [torch.zeros(B, N, dtype=torch.long, device=dev) for _ in range(KEV)]
            k0 = wflat[boff + (mz & M) * G + (mx & M)]                    # the cell the camera stands in (a lintel?)
            part0 = (k0 > 0) & ~fullf[koff + k0]
            e_in[0] = torch.where(part0, torch.zeros_like(tw), e_in[0])
            e_out[0] = torch.where(part0, torch.minimum(tmx, tmz), e_out[0])
            e_kind[0] = torch.where(part0, k0, e_kind[0])
            cnt = part0.long()
            prev = torch.where(part0, k0, torch.zeros_like(k0))
        for i in range(steps):
            side = tmx < tmz
            t_ent = torch.where(side, tmx, tmz)
            nside = ~side
            mx = mx + sx * side
            mz = mz + sz * nside
            tmx = tmx + tdx * side
            tmz = tmz + tdz * nside
            kind = wflat[boff + (mz & M) * G + (mx & M)]
            kind = torch.where(done, torch.zeros_like(kind), kind)
            hit = tflat[koff + kind]
            tw = torch.where(hit, t_ent, tw)
            sw = torch.where(hit, side, sw)
            kw = torch.where(hit, kind, kw)
            done = done | hit
            if events:
                part = (kind > 0) & ~hit
                ext = part & (kind == prev)
                new = part & ~ext
                t_out = torch.minimum(tmx, tmz)
                for k in range(KEV):
                    mk = new & (cnt == k)
                    e_in[k] = torch.where(mk, t_ent, e_in[k])
                    e_side[k] = torch.where(mk, side, e_side[k])
                    e_kind[k] = torch.where(mk, kind, e_kind[k])
                    e_out[k] = torch.where(mk | (ext & (cnt == k + 1)), t_out, e_out[k])
                cnt = cnt + new
                prev = kind
            if i % 6 == 5 and bool(done.all()):
                break
        if events:
            return tw, sw, kw, dx, dz, e_in, e_out, e_side, e_kind
        return tw, sw, kw

    def probe(self, rel):
        """Free distance from every player along headings yaw + rel [N] (things that stop a walker count)."""
        ang = self.yaw[:, None] + rel[None]
        return self._cast(self.pos, ang, 24, self.k_block)[0]

    def _light_at(self, p):
        """Light map sample [B, N, 9] at world points p [B, N, 2]."""
        g = ((p * (1.0 / G)) % 1.0) * (2.0 * LG / (LG + 2)) + (2.0 / (LG + 2) - 1.0)
        return F.grid_sample(self.lm, g[:, :, None, :], mode="bilinear", padding_mode="border", align_corners=False)[..., 0].transpose(1, 2)

    def _h_points(self, p):
        """Ground height [B, N] at world points p [B, N, 2] (0 where the ground is flat)."""
        g = ((p * (1.0 / G)) % 1.0) * (2.0 * G / (G + 2)) + (2.0 / (G + 2) - 1.0)
        return F.grid_sample(self.hmap[:, :1], g[:, :, None, :], mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]

    def _terrain(self, hi, dirx, dirz, s):
        """Where the rays of episodes hi [n] meet ground that is not flat: a march outward in growing steps, the crossing
        found by interpolation. dirx, dirz, s [n, H, W]. -> distance, height there, whether the ground is met at all,
        and how light the slope there is (it is lit from the sun's side)."""
        n = hi.numel()
        v = lambda a: a.view(n, 1, 1)
        ox, oz, ey = v(self.pos[hi, 0]), v(self.pos[hi, 1]), v(self.eye_now[hi])
        hm = self.hmap[hi]
        k1, k0 = 2.0 * G / (G + 2), 2.0 / (G + 2) - 1.0

        def at(t, ch):
            grid = torch.stack([(((ox + dirx * t) * (1.0 / G)) % 1.0) * k1 + k0, (((oz + dirz * t) * (1.0 / G)) % 1.0) * k1 + k0], -1)
            return F.grid_sample(hm[:, ch], grid, mode="bilinear", padding_mode="border", align_corners=False)

        t_hit = torch.full_like(s, 1e9)
        found = torch.zeros_like(s, dtype=torch.bool)
        tp, dp = 0.0, (ey - v(self.gh[hi])).expand_as(s)
        t, st = 0.0, 0.25
        for _ in range(NTER):
            t += st
            st *= 1.045
            d = ey + s * t - at(t, slice(0, 1))[:, 0]
            cross = ~found & (d <= 0)
            t_hit = torch.where(cross, tp + dp / (dp - d).clamp(min=1e-6) * (t - tp), t_hit)
            found = found | cross
            tp, dp = t, d
        gr = at(t_hit.clamp(max=2 * FAR), slice(1, 3))
        ndl = (0.7 - 0.7 * (gr[:, 0] * v(self.sun[hi, 0]) + gr[:, 1] * v(self.sun[hi, 1]))) / torch.sqrt(1 + gr[:, 0] ** 2 + gr[:, 1] ** 2)
        return t_hit, ey + s * t_hit, found, 0.50 + 0.72 * ndl.clamp(min=0)

    def _rgb_light(self, Ls):
        """amb + steady + channels, from light-map samples [..., 9] -> [..., 3]."""
        return self.amb[:, None] + (Ls[..., 0:3] + torch.einsum("bnc,bck->bnk", Ls[..., 3:7], self.dyncol_now)) * self.ltint[:, None]

    # ------------------------------------------------------------------ one step
    @torch.no_grad()
    def step(self, act, dest=None):
        """act [B, 8] (see ACTIONS; 7 columns: nobody presses E). dest [B]: the level to fall into for episodes in
        which noclip is pressed in this step; -1 or None picks one at random. Whoever is caught wakes up somewhere
        else in the same level. Returns frame uint8 [B, H, W, 3], lvl [B] and aux [B, len(AUX)]."""
        if not self._started:
            self.reset()
        B, dev, dt = self.B, self.dev, self.dt
        if self.prof is not None:
            import time
            torch.cuda.synchronize()
            self._t_last = time.time()
        act = act.to(dev).float()
        if act.shape[1] < len(ACTIONS):
            act = F.pad(act, (0, len(ACTIONS) - act.shape[1]))
        alive = self.trans == 0
        af = alive.float()
        fwd, right = act[:, 0].clamp(-1, 1), act[:, 1].clamp(-1, 1)
        turn, pit = act[:, 2].clamp(-1, 1), act[:, 3].clamp(-1, 1)
        sprint, lighton, nc = act[:, 4] > 0.5, act[:, 5] > 0.5, act[:, 6] > 0.5
        rnd = torch.tensor(self.levels, device=dev)[torch.randint(len(self.levels), (B,), generator=self.g, device=dev)]
        if len(self.levels) > 1:        # a random destination differs from where the player is
            again = torch.tensor(self.levels, device=dev)[torch.randint(len(self.levels), (B,), generator=self.g, device=dev)]
            rnd = torch.where(rnd == self.lvl, again, rnd)
        dest = rnd if dest is None else torch.where(dest.to(dev) < 0, rnd, dest.to(dev).long())
        # the player
        self.yaw = self.yaw + turn * TURN_MAX * af
        self.pitch = (self.pitch + pit * PITCH_MAX * af).clamp(-PITCH_LIM, PITCH_LIM)
        self.light = torch.where(alive, lighton, self.light)
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        sp = torch.where(sprint & (fwd > 0), torch.full_like(fwd, SPRINT), torch.full_like(fwd, WALK))
        vf = fwd * torch.where(fwd > 0, sp, torch.full_like(fwd, BACK))
        vr = right * STRAFE
        nrm = torch.sqrt(vf * vf + vr * vr)
        sc = torch.where(nrm > sp, sp / nrm.clamp(min=1e-6), torch.ones_like(nrm)) * af
        tv = torch.stack([(cy * vf - sy * vr) * sc, (sy * vf + cy * vr) * sc], 1)
        self.vel = self.vel + (tv - self.vel) * 0.7
        p0 = self.pos.clone()
        for _ in range(3):
            h = dt / 3
            nx = self.pos[:, 0] + self.vel[:, 0] * h
            bx = self._blocked(nx, self.pos[:, 1])
            nx = torch.where(bx, self.pos[:, 0], nx)
            nz = self.pos[:, 1] + self.vel[:, 1] * h
            bz = self._blocked(nx, nz)
            nz = torch.where(bz, self.pos[:, 1], nz)
            self.pos = torch.stack([nx, nz], 1)
        moved = (self.pos - p0).norm(dim=1)
        self.vel = (self.pos - p0) / dt * 0.5 + self.vel * 0.5       # walls take the speed out
        self.pos = self.pos % G
        speed = moved / dt
        self.bob = self.bob + moved * 0.75
        amp = (speed / WALK).clamp(max=1.6)
        eye = EYE + 0.035 * torch.sin(2 * math.pi * self.bob) * amp
        self.roll = 0.008 * torch.sin(math.pi * self.bob) * amp
        self._speed = speed
        if self.t % 5 == 0:
            self.trail[:, self.tr_count % NTR] = self.pos
            self.tr_count += 1
        self._tick("player")
        self.scr, self.inv = (self.scr - 1).clamp(min=0), (self.inv - 1).clamp(min=0)
        self._doors((act[:, 7] > 0.5) & alive)
        # transitions that start now
        start = alive & nc
        self._begin(start, dest, 0)
        caught = self._entities(alive & ~start)
        self._begin(caught, self.lvl.clone(), 1)
        self._tick("entities")
        # transitions in progress
        self.fade.fill_(1.0); self.glitch.zero_(); self.red.zero_()
        tr = self.trans > 0
        if bool(tr.any()):
            for b in tr.nonzero().flatten().tolist():
                kind = int(self.trans_kind[b])
                L = TR_DEATH if kind else TR_NOCLIP
                fidx = L - int(self.trans[b])
                if fidx == (_DE_LOAD if kind else _NC_LOAD):
                    self._load(b, int(self.lvl[b]))
                    if kind:
                        self.hp[b] = 1.0
                pre = "de" if kind else "nc"
                e = float(self._tab[pre + "_eye"][fidx])
                Hb = min(float(self.Hc[b]), 5.0)
                eye[b] = EYE * e if e >= 0 else EYE - e * (Hb - 0.3 - EYE)
                self.fade[b] = self._tab[pre + "_fade"][fidx]
                self.glitch[b] = self._tab[pre + "_gl"][fidx]
                if kind and fidx < 5:
                    self.red[b] = 1.0
                    self.roll[b] = 0.12 * math.sin(fidx * 2.1)
            self.trans = (self.trans - 1).clamp(min=0)
        self.gh = self._h_points(self.pos[:, None])[:, 0] if bool(self.hilly.any()) else torch.zeros_like(eye)
        self.eye_now = eye.clamp(min=0.08) + self.gh
        self._tick("transitions")
        self._lights()
        self.t += 1
        self._tick("lamps")
        frame = self._render()
        self._tick("post")
        return frame, self.lvl.clone(), self._aux()

    def _begin(self, mask, dest, kind):
        if not bool(mask.any()):
            return
        self.trans = torch.where(mask, torch.full_like(self.trans, TR_DEATH if kind else TR_NOCLIP), self.trans)
        self.trans_kind = torch.where(mask, torch.full_like(self.trans_kind, kind), self.trans_kind)
        self.lvl = torch.where(mask, dest, self.lvl)
        self.vel = self.vel * (~mask).float()[:, None]

    def _doors(self, use):
        """E. The door the player faces, within reach, starts to open or to close: one door per episode moves at a
        time, one kind of its chain per frame. Opening one can wake what waits in the room behind it."""
        B = self.B
        wf = self.wall.view(B, -1)
        mv = self.mv_cell >= 0
        if bool(mv.any()):
            idx = self.mv_cell.clamp(min=0)[:, None]
            cur = wf.gather(1, idx)
            nxt = torch.where(self.mv_dir[:, None] > 0, self.k_nxt.gather(1, cur), self.k_prv.gather(1, cur))
            new = torch.where(mv[:, None] & (nxt > 0), nxt, cur)
            wf.scatter_(1, idx, new)
            self.block.view(B, -1).scatter_(1, idx, self.k_block.gather(1, new) | self.pblock.view(B, -1).gather(1, idx))
            done = mv & ((self.k_door.gather(1, new)[:, 0] != 2) | (nxt[:, 0] == 0))
            self.mv_cell = torch.where(done, torch.full_like(self.mv_cell, -1), self.mv_cell)
        tw, _, kw = self._cast(self.pos, self.yaw[:, None], 4, self.k_full | (self.k_door > 0))
        st, tw = self.k_door.gather(1, kw)[:, 0], tw[:, 0]
        reach = (tw < REACH) & ((st == 1) | (st == 3))
        self.door_now = torch.where(reach, st, torch.zeros_like(st))
        go = use & reach & (self.mv_cell < 0)
        if bool(go.any()):
            cxl = torch.floor(self.pos[:, 0] + torch.cos(self.yaw) * (tw + 0.02))
            czl = torch.floor(self.pos[:, 1] + torch.sin(self.yaw) * (tw + 0.02))
            d = torch.maximum(_wrap(cxl + 0.5 - self.pos[:, 0]).abs(), _wrap(czl + 0.5 - self.pos[:, 1]).abs())
            go = go & ((st == 1) | (d > 0.55 + RAD))                # a doorway does not close on the player
            cell = (czl.long() & M) * G + (cxl.long() & M)
            self.mv_cell = torch.where(go, cell, self.mv_cell)
            self.mv_dir = torch.where(go, torch.where(st == 1, torch.ones_like(st), -torch.ones_like(st)), self.mv_dir)
            opening = go & (st == 1)
            if bool(opening.any()):
                self._dwellers(opening, cell)

    def _dwellers(self, opening, cell):
        """What stood behind a door that is being opened: a figure, something that runs out, eyes in the dark."""
        for b in opening.nonzero().flatten().tolist():
            i = (self.d_cell[b] == cell[b]).nonzero().flatten()
            if not i.numel():
                continue
            i = int(i[0])
            h = int(self.d_haunt[b, i])
            self.d_haunt[b, i] = 0
            slot = int((self.e_type[b] == 0).float().argmax())
            if h == 0 or int(self.e_type[b, slot]) != 0:
                continue
            x, z = float(self.d_room[b, i, 0]), float(self.d_room[b, i, 1])
            if h == 1:
                u = float(self._rand(1))
                lvl = LEVELS[int(self.lvl[b])]
                self._spawn_at(b, slot, E_WATCHER, x, z, "faceless" if lvl in FACELESS and u < 0.6 else ("figure" if u < 0.8 else "figure2"))
                self.e_near[b, slot] = 2.0
            elif h == 2:
                self._spawn_at(b, slot, E_HOUND, x, z, "hound")
            elif h == 3:
                self._spawn_at(b, slot, E_SMILER, x, z, "smiler")
            else:
                self._spawn_at(b, slot, E_PARTY, x, z, "party")

    def _blocked(self, x, z):
        b = torch.zeros_like(x, dtype=torch.bool)
        bf = self.block.view(-1)
        off = self.boff.view(-1)
        for ax in (-RAD, RAD):
            for az in (-RAD, RAD):
                b = b | bf[off + (torch.floor(z + az).long() & M) * G + (torch.floor(x + ax).long() & M)]
        return b

    def _aux(self):
        z = lambda v: v.float() if torch.is_tensor(v) else torch.full((self.B,), float(v), device=self.dev)
        sp = getattr(self, "_speed", torch.zeros(self.B, device=self.dev))
        cols = [self.pos[:, 0], self.pos[:, 1], self.yaw, self.pitch, sp, self.eye_now, self.light, self.ent_dist,
                self.ent_vis, self.ent_typ, self.danger, (self.trans > 0) & (self.trans_kind == 1), self.trans > 0,
                self.bright, self.ep, self.door_now, self.hp, self.scr, self.w_act, self.red_t > 0]
        return torch.stack([z(c) for c in cols], 1)

    # ------------------------------------------------------------------ lamps that change
    def _lights(self):
        B, dev = self.B, self.dev
        u = self._rand(B, 6)
        self.on1 = torch.where(self.on1, u[:, 0] > 0.07, u[:, 0] < 0.45)
        self.on2 = torch.where(self.on2, u[:, 1] > 0.45, u[:, 1] < 0.05)
        ph = (self.t * self.dt / self.pulse[:, 0]) % 1.0
        lo, hi = self.pulse[:, 1], self.pulse[:, 2]
        wav = torch.where(self.pulse[:, 3] > 0.5, (ph < 0.5).float(), 0.5 + 0.5 * torch.sin(2 * math.pi * ph))
        d = self.dynm
        d[:, 0] = 1.0
        d[:, 1] = torch.where(self.on1, 0.9 + 0.2 * u[:, 2], 0.10 + 0.1 * u[:, 2])
        d[:, 2] = torch.where(self.on2, 0.8 + 0.2 * u[:, 3], torch.full_like(ph, 0.02))
        d[:, 3] = lo + (hi - lo) * wav
        d[:, 4] = torch.where(u[:, 4] < self.spark, 1.2 + 2.0 * u[:, 5], d[:, 4] * 0.25)
        # a wave of lamps going out towards the player, then darkness for a while
        alive = self.trans == 0
        start = alive & ~self.w_act & (self._rand(B) < self.wave_rate / 60.0 * self.dt)
        if bool(start.any()):
            fwd = torch.stack([torch.cos(self.yaw), torch.sin(self.yaw)], 1)
            sgn = torch.where(self._rand(B) < 0.75, 1.0, -1.0)[:, None]
            self.w_org = torch.where(start[:, None], self.pos, self.w_org)
            self.w_dir = torch.where(start[:, None], fwd * sgn, self.w_dir)
            self.w_front = torch.where(start, torch.full_like(self.w_front, 27.0), self.w_front)
            self.w_hold = torch.where(start, 4.0 + 10.0 * self._rand(B), self.w_hold)
            self.w_act = self.w_act | start
        if bool(self.w_act.any()):
            a = self.w_act.float()
            self.w_front = self.w_front - 8.0 * self.dt * a
            self.w_hold = self.w_hold - self.dt * a * (self.w_front < -30).float()
            self.w_act = self.w_act & (self.w_hold > 0)
        self.emit_now = self.emit * d.gather(1, self.echan.view(B, -1)).view(B, G, G)
        if bool(self.w_act.any()):
            c = torch.arange(G, device=dev) + 0.5
            qx = _wrap(c.view(1, 1, G) - self.w_org[:, 0].view(B, 1, 1)) * self.w_dir[:, 0].view(B, 1, 1)
            qz = _wrap(c.view(1, G, 1) - self.w_org[:, 1].view(B, 1, 1)) * self.w_dir[:, 1].view(B, 1, 1)
            fr = torch.floor(self.w_front / 3.5) * 3.5
            dark = ((qx + qz) > fr.view(B, 1, 1)) & self.w_act.view(B, 1, 1)
            self.emit_now = self.emit_now * (~dark).float()
        # now and then every lamp goes out and comes back red, for a while
        rs = alive & (self.red_t <= 0) & ~self.w_act & (self._rand(B) < self.red_rate / 60.0 * self.dt)
        self.red_t = torch.where(rs, 8.0 + 17.0 * self._rand(B), self.red_t)
        self.red_ph = torch.where(rs, torch.zeros_like(self.red_ph), self.red_ph + 1)
        ra = self.red_t > 0
        self.red_t = (self.red_t - self.dt * ra.float()).clamp(min=0)
        off = ra & ((self.red_ph <= 2) | (self.red_t < 0.35))
        red = torch.tensor([0.95, 0.09, 0.06], device=dev)
        self.ltint = torch.where(off[:, None], torch.full((1, 3), 0.03, device=dev), torch.where(ra[:, None], red[None], torch.ones(1, 3, device=dev)))
        self.lcol_now = self.lcol * self.ltint
        self.dyncol_now = d[:, 1:, None] * self.dyncol[:, 1:]
        Ls = self._light_at(self.pos[:, None])
        rgb = self._rgb_light(Ls)[:, 0]
        if bool(self.w_act.any()):
            rgb = torch.where((self.w_act & (self.w_front < 0))[:, None], self.amb, rgb)
        self.bright = rgb.mean(1)

    # ------------------------------------------------------------------ entities
    def _entities(self, alive):
        """Move, spawn and remove the things that are not the player. Returns who was caught in this step."""
        B, dev, dt = self.B, self.dev, self.dt
        act = self.e_type > 0
        rel = _wrap(self.e_pos - self.pos[:, None])
        dist = rel.norm(dim=2)
        abs_ang = torch.atan2(rel[..., 1], rel[..., 0])
        ang = (abs_ang - self.yaw[:, None] + math.pi) % (2 * math.pi) - math.pi
        free = self._cast(self.pos, abs_ang, 28, self.k_full)[0]
        los = free > dist - 0.3
        vd = (3.2 / self.fog).clamp(max=40.0)[:, None]
        vis = act & (ang.abs() < 0.78) & los & (dist < vd)
        self.e_age = self.e_age + act.long()
        self.e_seen = self.e_seen + vis.long()
        self.e_unseen = torch.where(vis, torch.zeros_like(self.e_unseen), self.e_unseen + (self.e_seen > 0).long())
        ty = self.e_type
        speed = torch.zeros(B, NE, device=dev)
        toward = torch.ones(B, NE, device=dev)
        # watcher: stands there, and is gone when you look again or come close
        w = ty == E_WATCHER
        self.e_fade |= w & ((dist < self.e_near) | (self.e_age > 450) | ((self.e_vel.abs().sum(2) > 0) & (self.e_age > 45)))
        gone = w & (self.e_seen >= 4) & (self.e_unseen >= 3)
        # stalker and partygoer: only move while not looked at
        s = ty == E_STALKER
        speed = torch.where(s & ~vis, torch.full_like(speed, 2.8), speed)
        self.e_fade |= s & (self.e_age > 700)
        p = ty == E_PARTY
        speed = torch.where(p & ~vis & (dist < 14), torch.full_like(speed, 1.3), speed)
        # smiler: comes closer in the dark, backs off and dissolves in the flashlight beam
        m = ty == E_SMILER
        lit = m & vis & self.light[:, None] & (ang.abs() < 0.3) & (dist < 14)
        self.e_lit = self.e_lit + lit.long()
        speed = torch.where(m, torch.where(lit, torch.full_like(speed, 1.6), torch.full_like(speed, 1.1)), speed)
        toward = torch.where(lit, -toward, toward)
        self.e_fade |= m & ((self.e_lit > 10) | (self.e_age > 500) | (self.bright[:, None] > 0.35))
        # hound and horde: run at you
        hd = ty == E_HOUND
        speed = torch.where(hd, torch.full_like(speed, 3.2), speed)       # faster than the player can run: hide, or shut a door
        self.e_fade |= hd & (self.e_age > 350)
        ho = ty == E_HORDE
        speed = torch.where(ho, torch.full_like(speed, 2.75), speed)
        # where to: the player when in sight, else along the player's trail
        latest = self.tr_count - 1
        self.e_tr = torch.where(los, torch.full_like(self.e_tr, latest), self.e_tr).clamp(max(latest - NTR + 2, 0), latest)
        tgt_tr = self.trail.gather(1, (self.e_tr % NTR)[..., None].expand(B, NE, 2))
        dtr = _wrap(tgt_tr - self.e_pos)
        near = (dtr.norm(dim=2) < 0.6) & (self.e_tr < latest)
        self.e_tr = self.e_tr + near.long()
        dv = torch.where(los[..., None], -rel, dtr)
        dn = dv.norm(dim=2, keepdim=True).clamp(min=1e-6)
        stepm = torch.minimum(speed * dt, torch.where(toward > 0, dn[..., 0], torch.full_like(speed, 9.0)))
        mv = dv / dn * (stepm * toward * act.float() * alive[:, None].float())[..., None]
        newp = (self.e_pos + mv + self.e_vel * (dt * act.float() * alive[:, None].float())[..., None]) % G
        ci = (torch.floor(newp[..., 1]).long() & M) * G + (torch.floor(newp[..., 0]).long() & M)
        kd = self.k_door.gather(1, self.wall.view(B, -1).gather(1, ci))
        self.e_pos = torch.where(((kd == 1) | (kd == 2))[..., None], self.e_pos, newp)      # a shut door stops them
        # the horde never falls far behind
        far = ho & (dist > 20) & alive[:, None]
        if bool(far.any()):
            k = max(latest - 8, 0)
            back = self.trail[:, k % NTR]
            self.e_pos = torch.where(far[..., None], back[:, None].expand(B, NE, 2), self.e_pos)
            self.e_tr = torch.where(far, torch.full_like(self.e_tr, min(k + 1, latest)), self.e_tr)
        # fading out and leaving
        self.e_alpha = (self.e_alpha - 0.25 * self.e_fade.float()).clamp(0, 1)
        gone = gone | (self.e_fade & (self.e_alpha <= 0))
        self.e_type = torch.where(gone, torch.zeros_like(ty), ty)
        act = self.e_type > 0
        threat = act & (ty != E_WATCHER) & (self.e_alpha > 0.5)
        dthr = torch.where(threat, dist, torch.full_like(dist, 99.0))
        dmin, imin = dthr.min(dim=1)
        self.danger = ((7.0 - dmin) / 5.0).clamp(0, 1) * alive.float()
        dall = torch.where(act, dist, torch.full_like(dist, 99.0))
        da, ia = dall.min(dim=1)
        self.ent_dist = da
        self.ent_vis = vis.gather(1, ia[:, None])[:, 0].float() * (da < 99).float()
        self.ent_typ = self.e_type.gather(1, ia[:, None])[:, 0].float()
        # a blow: health is lost, the attacker's face fills the picture and it is gone; the last blow is death
        hit = alive & (dmin < 0.9) & (self.inv == 0)
        typ = self.e_type.gather(1, imin[:, None])[:, 0]
        self.hp = torch.where(hit, self.hp - self.dmg[typ], self.hp + REGEN * dt * alive.float()).clamp(max=1.0)
        self.scr = torch.where(hit, torch.full_like(self.scr, SCR), self.scr)
        self.scr_tex = torch.where(hit, torch.where(typ == E_SMILER, self.tex_names["s_smiler"], self.tex_names["s_scream"]), self.scr_tex)
        self.inv = torch.where(hit, torch.full_like(self.inv, INV), self.inv)
        struck = hit[:, None] & (torch.arange(NE, device=dev)[None] == imin[:, None]) & (self.e_type != E_HORDE)
        self.e_type = torch.where(struck, torch.zeros_like(self.e_type), self.e_type)
        act = self.e_type > 0
        caught = hit & (self.hp <= 0)
        # new arrivals
        nact = act.sum(1)
        boost = torch.where(self.w_act & (self.w_front < -5), 8.0, 1.0) * torch.where((self.red_t > 0) & (self.red_t < 1e8), 4.0, 1.0)
        pr = self.rates * (boost / 60.0 * dt)[:, None]
        u = self._rand(B, 7)
        want = (u < pr) & alive[:, None] & (nact < 3)[:, None]
        if bool(want.any()):
            self._spawn(want)
        return caught

    def _spawn(self, want):
        B, dev = self.B, self.dev
        rel = torch.linspace(-0.5, 0.5, 9, device=dev)
        free = self._cast(self.pos, self.yaw[:, None] + rel[None], 30, self.k_block)[0]
        fbest, ibest = free.max(dim=1)
        u = self._rand(B, 4)
        abest = self.yaw + rel[ibest]
        for b in want.any(1).nonzero().flatten().tolist():
            typ = int(want[b].float().argmax())
            slot = int((self.e_type[b] == 0).float().argmax())
            if int(self.e_type[b, slot]) != 0:
                continue
            lvl = LEVELS[int(self.lvl[b])]
            if typ in (E_WATCHER, E_SMILER):
                lo, hi = (9.0, 24.0) if typ == E_WATCHER else (5.0, 12.0)
                fb = float(fbest[b])
                if fb < 5.5:                # in narrow places they stand closer
                    continue
                d = min(fb - 1.0, lo + (hi - lo) * float(u[b, 0]))
                x = float(self.pos[b, 0]) + d * math.cos(float(abest[b]))
                z = float(self.pos[b, 1]) + d * math.sin(float(abest[b]))
                if typ == E_SMILER:
                    Ls = self._light_at(torch.tensor([[[x % G, z % G]]], device=dev).expand(B, 1, 2))
                    if float(self._rgb_light(Ls)[b, 0].mean()) > 0.12:
                        continue
                    spr = "smiler"
                else:
                    spr = "faceless" if lvl in FACELESS and u[b, 1] < 0.6 else ("figure" if u[b, 1] < 0.5 else "figure2")
                    if float(u[b, 3]) > 0.85 and float(self.Hc[b]) >= 3.3:
                        spr = "tall"
                self._spawn_at(b, slot, typ, x, z, spr)
                if typ == E_WATCHER and float(u[b, 2]) < 0.35:        # this one walks across and out of sight
                    sgn = 1.0 if float(u[b, 3]) < 0.5 else -1.0
                    ab = float(abest[b]) + sgn * math.pi / 2
                    self.e_vel[b, slot, 0], self.e_vel[b, slot, 1] = 1.3 * math.cos(ab), 1.3 * math.sin(ab)
                    self.e_spr[b, slot] = SPR_ID["figure2"]
            elif typ in (E_STALKER, E_HOUND):
                k = self.tr_count - 1 - int(12 + 28 * float(u[b, 0]))
                if k < max(self.tr_count - NTR + 1, 0):
                    continue
                p = self.trail[b, k % NTR]
                r = _wrap(p - self.pos[b])
                if float(r.norm()) < 7.0:
                    continue
                a = (math.atan2(float(r[1]), float(r[0])) - float(self.yaw[b]) + math.pi) % (2 * math.pi) - math.pi
                if abs(a) < 0.9:
                    continue
                spr = "hound" if typ == E_HOUND else ("scarecrow" if lvl == "fields" else ("faceless" if lvl in ("office", "mall", "city") else
                                                                                         ("figure2" if u[b, 1] < 0.6 else "figure")))
                self._spawn_at(b, slot, typ, float(p[0]), float(p[1]), spr)
                self.e_tr[b, slot] = k + 1

    # ------------------------------------------------------------------ drawing
    def _fan(self):
        B, dev = self.B, self.dev
        ang = self.yaw[:, None] + self.fan_az[None]
        tw, sw, kw, dx, dz, e_in, e_out, e_side, e_kind = self._cast(self.pos, ang, NSTEP, self.k_full, events=True)
        ox, oz = self.pos[:, 0:1], self.pos[:, 1:2]
        ks, kt, kb = self.k_side.view(-1), self.k_top.view(-1), self.k_bot.view(-1)
        y0, y1 = self.k_y0.view(-1), self.k_y1.view(-1)
        kiw = self.koff + kw
        sun = torch.where(sw, -torch.sign(dx) * self.sun[:, 0:1], -torch.sign(dz) * self.sun[:, 1:2])
        if bool(self.hilly.any()):       # what stands on ground that is not flat stands on the height of its cell
            cellh = lambda t: self.hcell.gather(1, (torch.floor(oz + (t + 0.01) * dz).long() & M) * G + (torch.floor(ox + (t + 0.01) * dx).long() & M))
            hw, hb = cellh(tw), [cellh(e_in[k]) for k in range(KEV)]
        else:
            hw = torch.zeros_like(tw)
            hb = [hw] * KEV
        ch = [tw, torch.where(sw, oz + tw * dz, ox + tw * dx), ks[kiw], torch.where(sw, dx.abs(), dz.abs()),
              y1[kiw], self.k_split.view(-1)[kiw], self.k_up.view(-1)[kiw], sun, hw]
        for k in range(KEV):
            ki = self.koff + e_kind[k]
            ch += [e_in[k], e_out[k], torch.where(e_side[k], oz + e_in[k] * dz, ox + e_in[k] * dx),
                   torch.where(e_side[k], dx.abs(), dz.abs()), y0[ki], y1[ki], ks[ki], kt[ki], kb[ki], hb[k]]
        fan = torch.stack(ch, 1)
        # sprites: the nearest props in view and the entities
        rel = _wrap(self.pp - self.pos[:, None])
        d = rel.norm(dim=2)
        a = (torch.atan2(rel[..., 1], rel[..., 0]) - self.yaw[:, None] + math.pi) % (2 * math.pi) - math.pi
        ok = (self.pt > 0) & (a.abs() < 1.35) & (d < FAR) & (d > 0.3)
        dk, ik = torch.topk(torch.where(ok, d, torch.full_like(d, 1e9)), NPV, dim=1, largest=False)
        pa, ptp = a.gather(1, ik), self.pt.gather(1, ik)
        ppos = self.pp.gather(1, ik[..., None].expand(B, NPV, 2))
        erel = _wrap(self.e_pos - self.pos[:, None])
        ed = erel.norm(dim=2)
        ea = (torch.atan2(erel[..., 1], erel[..., 0]) - self.yaw[:, None] + math.pi) % (2 * math.pi) - math.pi
        sd = torch.cat([dk, torch.where(self.e_type > 0, ed, torch.full_like(ed, 1e9))], 1)          # [B, S]
        sa = torch.cat([pa, ea], 1)
        stp = torch.cat([ptp, self.e_spr], 1)
        spos = torch.cat([ppos, self.e_pos], 1)
        salpha = torch.cat([torch.ones_like(dk), self.e_alpha], 1)
        tab = self.stab[stp]                                                                       # [B, S, 5]
        sw_, sh, sbase = tab[..., 1], tab[..., 2], tab[..., 3]
        Hc = self.Hc[:, None]
        sh = torch.where(sh <= 0, Hc.expand_as(sh), sh)
        sbase = torch.where(sbase < 0, Hc - sh, sbase) + self._h_points(spos)
        delta = self.fan_az.view(1, 1, NA) - sa[..., None]
        lat = sd[..., None] * torch.tan(delta.clamp(-1.4, 1.4))
        th = sd[..., None] / torch.cos(delta.clamp(-1.4, 1.4))
        cov = (sd[..., None] < 1e8) & (delta.abs() < 1.3) & (lat.abs() < 0.5 * sw_[..., None]) & (th < tw[:, None] + 0.05)
        key = torch.where(cov, th, torch.full_like(th, 1e9))
        st2, sid = torch.topk(key, 2, dim=1, largest=False)                                        # [B, 2, NA]
        su = (0.5 + lat / sw_[..., None].clamp(min=1e-3)).gather(1, sid)
        Ls = self._light_at(spos)
        slight = self._rgb_light(Ls)
        self._spr = dict(t=st2, u=su, id=sid, base=sbase, h=sh, tex=tab[..., 0].long(), emis=tab[..., 4],
                         alpha=salpha, light=slight)
        return fan

    @torch.no_grad()
    def _render(self):
        B, H, W, dev = self.B, self.H, self.W, self.dev
        v3 = lambda a: a.view(B, 1, 1)
        eye, Hc = v3(self.eye_now), v3(self.Hc)
        cr, sr = v3(torch.cos(self.roll)), v3(torch.sin(self.roll))
        x = self.px * cr - self.py * sr
        y = self.px * sr + self.py * cr
        cp, sp = v3(torch.cos(self.pitch)), v3(torch.sin(self.pitch))
        fh = cp - y * sp
        dy = sp + y * cp
        hl = torch.sqrt(x * x + fh * fh)
        s = dy / hl
        az = torch.atan2(x, fh)
        ai = ((az + FAN_HALF) * ((NA - 1) / (2 * FAN_HALF)) + 0.5).long().clamp_(0, NA - 1)
        cy, sy = v3(torch.cos(self.yaw)), v3(torch.sin(self.yaw))
        dirx = (cy * fh - sy * x) / hl
        dirz = (sy * fh + cy * x) / hl
        fan = self._fan()
        self._tick("fan")
        g = torch.gather(fan, 2, ai.view(B, 1, H * W).expand(B, fan.shape[1], H * W)).view(B, -1, H, W)
        s_ = torch.where(s.abs() < 1e-4, torch.full_like(s, 1e-4), s)
        up = s_ > 0
        ysurf = torch.where(up, Hc.expand_as(s), torch.zeros_like(s))
        t_pl = (ysurf - eye) / s_
        isceil, tshade = up, None                                   # where no wall stops a ray: the ceiling (sky), else the ground
        if bool(self.hilly.any()):
            hi = self.hilly.nonzero().flatten()
            t_h, y_h, met, sh_h = self._terrain(hi, dirx[hi], dirz[hi], s[hi])
            t_pl[hi] = torch.where(met, t_h, torch.where(up[hi], t_pl[hi], torch.full_like(t_h, 2 * FAR)))
            ysurf[hi] = torch.where(met, y_h, ysurf[hi])
            isceil = up.clone()
            isceil[hi] = up[hi] & ~met
            tshade = torch.ones_like(s)
            tshade[hi] = sh_h
        tw = g[:, 0]
        wtop, ybase = g[:, 4], g[:, 8]
        ywall = eye + s * tw - ybase                                # height above the wall's foot
        side = (tw < t_pl) & (ywall <= wtop + 1e-3)                 # above a wall's top the ray goes on, to the sky
        face = torch.zeros_like(side)                               # the top or the underside of a partial block is hit
        t = torch.where(side, tw, t_pl)
        u = g[:, 1]
        upper = ywall > g[:, 5]
        mat = torch.where(side, torch.where(upper, g[:, 6], g[:, 2]), torch.full_like(t, -1.0))
        cosi = g[:, 3]
        fith = torch.where(upper, wtop, torch.minimum(wtop, g[:, 5]))   # the height a stretched texture spans
        sunf = g[:, 7] * side.float()
        for k in reversed(range(KEV)):
            c0 = NFAN + NEVT * k
            ein, eout, eu, ecos, y0, y1, ms, mt_, mb, eb = (g[:, c0 + i] for i in range(NEVT))
            yin = eye + s * ein - eb
            yout = eye + s * eout - eb
            hs = (yin >= y0) & (yin <= y1) & (ein < t)
            ht = (yin > y1) & (yout < y1)
            hb = (yin < y0) & (yout > y0)
            yh = torch.where(ht, y1, y0) + eb
            th = (yh - eye) / s_
            hh = (ht | hb) & (th < t) & (th > 0)
            t = torch.where(hs, ein, torch.where(hh, th, t))
            side = (side & ~hh) | hs
            face = (face & ~hs) | hh
            ybase = torch.where(hs, eb, ybase)
            u = torch.where(hs, eu, u)
            cosi = torch.where(hs, ecos, cosi)
            mat = torch.where(hs, ms, torch.where(hh, torch.where(ht, mt_, mb), mat))
            ysurf = torch.where(hh, yh, ysurf)
            fith = torch.where(hs, y1, fith)
            sunf = torch.where(hs | hh, torch.zeros_like(sunf), sunf)
        del g, fan
        self._tick("hits")
        yhit = torch.where(side, eye + s * t, ysurf)
        yrel = yhit - ybase                                         # for walls: the height above their foot
        ceilf = torch.where(face, up, isceil)                       # the surface hit faces down
        wx = v3(self.pos[:, 0]) + dirx * t
        wz = v3(self.pos[:, 1]) + dirz * t
        ix = torch.floor(wx).long() & M
        iz = torch.floor(wz).long() & M
        cell = iz * G + ix
        plane = mat < 0
        matp = self.fc.view(-1)[self.boff2 + isceil.long() * (G * G) + cell]
        matl = torch.where(plane, matp, mat.long())
        cem = self.emit_now.view(-1)[self.boff + cell]
        mp_ = self.mtab[:, matl]                                    # [13, B, H, W]
        texi, us, vs, fit, em, chn, gloss, water, puddle = (mp_[i] for i in range(9))
        vsc = torch.where(fit > 0, 1.0 / fith.clamp(min=0.5), vs)
        tu = torch.where(side, u, wx) * us
        tv = torch.where(side, 1.0 - yrel * vsc, wz * us)
        s2 = torch.sqrt(1 + s * s)
        t3 = t * s2
        cosn = torch.where(side, cosi, s.abs() / s2).clamp_(min=0.3)
        tpp = t3 * (self.pixang * TS) * us / cosn
        lod = torch.log2(tpp.clamp_(min=1.0)).add_(0.3).long().clamp_(max=NMIP - 1)
        size = self.mipsize[lod]
        iu = torch.minimum(((tu - torch.floor(tu)) * size).long(), size - 1)
        iv = torch.minimum(((tv - torch.floor(tv)) * size).long(), size - 1)
        texel = self.atlas[self.base[texi.long() * NMIP + lod] + iv * size + iu]           # [B, H, W, 4] uint8
        alb = texel[..., :3].float() * (1.0 / 255)
        alb = alb * alb
        ema = texel[..., 3].float() * (1.0 / 255)
        del texel, tu, tv, iu, iv, size, tpp
        self._tick("texture")
        # light
        back = side.float() * 0.3
        lx, lz = wx - dirx * back, wz - dirz * back
        k1, k0 = 2.0 * LG / (LG + 2), 2.0 / (LG + 2) - 1.0
        grid = torch.stack([((lx * (1.0 / G)) % 1.0) * k1 + k0, ((lz * (1.0 / G)) % 1.0) * k1 + k0], -1)
        L = F.grid_sample(self.lm, grid, mode="bilinear", padding_mode="border", align_corners=False)
        light = (L[:, 0:3] + torch.einsum("bchw,bck->bkhw", L[:, 3:7], self.dyncol_now)) * self.ltint[:, :, None, None]
        if bool(self.w_act.any()):
            q = _wrap(lx - v3(self.w_org[:, 0])) * v3(self.w_dir[:, 0]) + _wrap(lz - v3(self.w_org[:, 1])) * v3(self.w_dir[:, 1])
            fr = v3(torch.floor(self.w_front / 3.5) * 3.5)
            wm = 1 - ((q - fr + 1.5) / 3.0).clamp(0, 1) * v3(self.w_act.float())
            light = light * wm[:, None]
        yn = yrel / Hc
        shade = torch.where(side,
                            (0.72 + 0.22 * yn) * (1 - 0.30 * (1 - yrel / 0.4).clamp_(0, 1) ** 2 - 0.30 * (1 - (Hc - yrel) / 0.4).clamp_(0, 1) ** 2),
                            torch.where(ceilf, 0.72, 1.0) * (1 - 0.85 * L[:, 7].clamp(max=0.55)))
        if tshade is not None:
            shade = torch.where(plane & ~isceil, shade * tshade, shade)
        shade = shade * (1 + v3(self.sunk) * sunf.clamp(min=0))
        fl = self.spot[None] * v3(self.light.float()) * (FL_P / (1.0 + t3 * t3 * 0.11))
        lit = (self.amb[:, :, None, None] + light) * shade[:, None] + fl[:, None] * self.fl_col
        stm = ((L[:, 8] - 0.5) * 2.9).clamp_(0, 1)
        tint = self.tint.view(-1, 3)[self.moff + matl]
        alb = alb * tint * (1 - (stm * v3(self.grime))[..., None] * (1 - self.staincol[:, None, None, :]))
        rgb = alb * lit.permute(0, 2, 3, 1)
        del alb, tint, light, grid
        self._tick("light")
        # things that glow
        E = torch.where(plane & isceil, cem, self.dynm.view(-1)[self.doff + chn.long()])
        emm = ema * em * E
        ei = (emm > 0.004).nonzero(as_tuple=True)
        if ei[0].numel():
            e_ = emm[ei]
            ec = self.m_emcol[matl[ei]] * torch.where(mp_[9][ei][:, None] > 0, self.lcol_now[ei[0]], torch.ones(1, 3, device=dev))
            rgb[ei] = rgb[ei] * (1 - e_.clamp(max=1.0))[:, None] + e_[:, None] * ec
        # wet floors and water
        gl = gloss * (~side & ~ceilf).float() * torch.where(puddle > 0, ((L[:, 8] - 0.52) * 6.0).clamp_(0, 1), torch.ones_like(gloss))
        gi = (gl > 0.02).nonzero(as_tuple=True)
        if gi[0].numel():
            bq = gi[0]
            sg, tg, glg, wat = s[gi], t[gi], gl[gi], water[gi]
            wxg, wzg, dxg, dzg = wx[gi], wz[gi], dirx[gi], dirz[gi]
            tim = self.t * self.dt
            iw = (wat > 0).float()
            rx_ = 0.05 * torch.sin(wxg * 3.1 + tim * 1.7) + 0.04 * torch.sin(wzg * 4.3 - tim * 1.3 + wxg * 1.3)
            rz_ = 0.05 * torch.sin(wzg * 2.7 - tim * 1.5) + 0.04 * torch.sin(wxg * 3.9 + tim * 1.1 - wzg * 1.7)
            dr = (self.Hc[bq] - ysurf[gi]) / sg.abs().clamp(min=0.03)
            rxp = wxg + dxg * dr + rx_ * dr * iw
            rzp = wzg + dzg * dr + rz_ * dr * iw
            hitc = ((tg + dr) < tw[gi]).float()
            refl = self.emit_now[bq, torch.floor(rzp).long() & M, torch.floor(rxp).long() & M] * hitc
            fx_, fz_ = rxp - torch.floor(rxp), rzp - torch.floor(rzp)
            refl = refl * (1 - ((torch.sqrt((fx_ - 0.5) ** 2 + (fz_ - 0.5) ** 2) - 0.2) / 0.3).clamp(0, 1))
            cosr = sg.abs() / torch.sqrt(1 + sg * sg)
            fres = (0.03 + 0.97 * (1 - cosr) ** 5) * glg
            c = rgb[gi]
            pool = (wat == 1).float()[:, None]
            dark = (wat == 2).float()[:, None]
            ca = torch.sin(wxg * 5.0 + torch.sin(wzg * 3.0 + tim * 1.1) * 1.5 + tim * 0.9) * torch.sin(wzg * 5.3 + torch.sin(wxg * 2.7 - tim * 0.8) * 1.5 - tim * 1.2)
            caust = (0.5 + 0.5 * ca) ** 3
            c = c * (1 - pool) + c * pool * torch.tensor([0.42, 0.88, 0.92], device=dev) * (0.75 + 1.1 * caust[:, None])
            c = c * (1 - dark) + c * dark * torch.tensor([0.10, 0.17, 0.2], device=dev)
            c = c * (1 - 0.35 * glg * (1 - iw))[:, None]                    # a wet floor is darker
            spec = (refl * 3.0)[:, None] * self.lcol_now[bq] + 0.5 * (self.amb[bq] + self.fogcol[bq])
            rgb[gi] = c * (1 - fres)[:, None] + spec * fres[:, None]
        del mp_, gl
        # distance
        fogd = v3(self.fog)
        f = torch.exp(-fogd * t3) * (1 - ((t3 - 34.0) / 12.0).clamp_(0, 1))
        rgb = rgb * f[..., None] + self.fogcol[:, None, None, :] * (1 - f)[..., None]
        if bool(self.outdoor.any()):
            self._sky(rgb, plane & isceil & v3(self.outdoor), s, az)
        self._tick("glow, water, fog")
        # sprites
        S = self._spr
        st0 = torch.gather(S["t"][:, 0], 1, ai.view(B, -1)).view(B, H, W)
        qi = (st0 < t).nonzero(as_tuple=True)
        if qi[0].numel():
            bq, hq, wq = qi
            aq, sq, tq = ai[qi], s[qi], t[qi]
            eyeq = self.eye_now[bq]
            c = rgb[qi]
            fq = self.light[bq].float() * self.spot[hq, wq]
            s2q = torch.sqrt(1 + sq * sq)
            for layer in (1, 0):
                ts = S["t"][bq, layer, aq]
                uu = S["u"][bq, layer, aq]
                sid = S["id"][bq, layer, aq]
                base, hh, txi = S["base"][bq, sid], S["h"][bq, sid], S["tex"][bq, sid]
                v = 1 - (eyeq + sq * ts - base) / hh
                ok = (ts < tq) & (v >= 0) & (v < 1) & (uu >= 0) & (uu < 1)
                t3s = ts * s2q
                lod = torch.log2((t3s * (self.pixang * TS) / hh).clamp(min=1.0)).add_(0.3).long().clamp_(max=NMIP - 1)
                size = self.mipsize[lod]
                iu = torch.minimum((uu.clamp(0, 0.9999) * size).long(), size - 1)
                iv = torch.minimum((v.clamp(0, 0.9999) * size).long(), size - 1)
                tx = self.atlas[self.base[txi * NMIP + lod] + iv * size + iu].float() * (1.0 / 255)
                al = S["alpha"][bq, sid] * ok.float()
                a = tx[:, 3] * al
                emis = S["emis"][bq, sid][:, None]
                litq = (S["light"][bq, sid] + (fq * FL_P / (1.0 + t3s * t3s * 0.11))[:, None]) * (1 - emis) + emis * 1.6
                col = tx[:, :3] ** 2 * litq * al[:, None]
                fs = (torch.exp(-self.fog[bq] * t3s) * (1 - ((t3s - 34.0) / 12.0).clamp(0, 1)))[:, None]
                col = col * fs + self.fogcol[bq] * a[:, None] * (1 - fs)
                c = c * (1 - a)[:, None] + col
            rgb[qi] = c
        self._tick("sprites")
        # glare around what is brighter than the picture can show
        hi = F.avg_pool2d((rgb - 1.0).clamp_(min=0).permute(0, 3, 1, 2), 8)
        for _ in range(2):
            hi = F.avg_pool2d(F.pad(hi, (2, 2, 2, 2), mode="replicate"), 5, 1)
        rgb = rgb + F.interpolate(hi, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1) * 0.55
        # the picture itself: fades, fear, tone curve
        dg = self.danger
        pulse = 0.75 + 0.25 * math.sin(self.t * self.dt * 2 * math.pi * 1.6)
        mul = v3(self.fade) * self.vig[None] * (1 - v3(dg * pulse) * 0.85 * self.dvig[None])
        rgb = rgb * mul[..., None]
        di = ((dg > 0.01) | (self.red > 0)).nonzero().flatten()
        if di.numel():
            c = rgb[di]
            grey = c.mean(-1, keepdim=True)
            k = (dg[di] * 0.7).view(-1, 1, 1, 1)
            c = c + (grey - c) * k
            r_ = self.red[di].view(-1, 1, 1, 1)
            c = c * (1 - r_ * torch.tensor([0.0, 0.8, 0.85], device=dev)) + r_ * torch.tensor([0.25, 0.0, 0.0], device=dev) * self.dvig[None, :, :, None] * self.fade[di].view(-1, 1, 1, 1)
            rgb[di] = c
        out = torch.sqrt(1 - torch.exp(rgb.clamp_(min=0) * (-EXPO))).mul_(255.0).add_(0.5).clamp_(0, 255).to(torch.uint8)
        gi_ = (self.glitch > 0).nonzero().flatten()
        if gi_.numel():
            out[gi_] = self._glitch(out[gi_], self.glitch[gi_])
        self._overlay(out)
        return out

    def _tex(self, ti, u, v, lod=1):
        """Texture ti [n] at (u, v) [n, H, W] -> [n, H, W, 4] in 0..1, nothing outside the texture."""
        size = TS >> lod
        ok = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        iu, iv = (u.clamp(0, 0.9999) * size).long(), (v.clamp(0, 0.9999) * size).long()
        t = self.atlas[self.base[ti * NMIP + lod][:, None, None] + iv * size + iu].float() * (1.0 / 255)
        return t * ok[..., None].float()

    def _overlay(self, out):
        """Over the finished picture: the face of whatever struck (SCR frames, coming at the camera), the words after the
        last blow, and an E where a door is within reach. Health itself is not shown."""
        si = (self.scr > 0).nonzero().flatten()
        if si.numel():
            k = SCR - self.scr[si]
            zoom = (0.5 + 0.45 * k.float()).clamp(max=1.25)[:, None, None]
            jit = (self._rand(si.numel(), 2) - 0.5) * 0.08 * (k >= 2).float()[:, None]
            tx = self._tex(self.scr_tex[si], 0.5 + self.ov_x[None] / zoom + jit[:, 0, None, None], 0.5 + self.ov_y[None] / zoom + jit[:, 1, None, None])
            cov, fl = self._tab["scr_cover"][k][:, None, None, None], self._tab["scr_flash"][k][:, None, None, None]
            out[si] = (out[si].float() * (1 - cov) + tx[..., :3] * (255 * cov) + 255 * fl).clamp_(0, 255).to(torch.uint8)
        di = ((self.trans_kind == 1) & (self.trans >= 4) & (self.trans <= 14)).nonzero().flatten()
        if di.numel():
            ti = torch.full_like(di, self.tex_names["t_died"])
            tx = self._tex(ti, 0.5 + self.ov_x[None].expand(di.numel(), -1, -1) * (1.2 * self.H / self.W), 0.5 + self.ov_y[None].expand(di.numel(), -1, -1) * (1.2 * self.H / self.W))
            out[di] = torch.maximum(out[di], (tx[..., :3] * 255).to(torch.uint8))
        d = (self.door_now > 0).nonzero().flatten()
        if d.numel():
            (gy, gx), (gh, gw) = self.glyph_at, self.glyph.shape
            out[d, gy:gy + gh, gx:gx + gw] = self.glyph_col[self.glyph]

    def _sky(self, rgb, m, s, az):
        """Outdoors, what is above everything: a gradient from the horizon up, clouds, stars, haze near the ground."""
        ei = m.nonzero(as_tuple=True)
        if not ei[0].numel():
            return
        bq = ei[0]
        el = torch.atan(s[ei]) * (2 / math.pi)                      # 0 at the horizon, 1 overhead
        tu = (az[ei] + self.yaw[bq]) * (1 / math.pi)
        iu = ((tu - torch.floor(tu)) * TS).long().clamp_(max=TS - 1)
        iv = ((1 - el).clamp(0, 0.9999) * TS).long()
        tx = self.atlas[self.base[self.sky_tex[bq] * NMIP] + iv * TS + iu].float() * (1.0 / 255)
        e = el.clamp(0, 1)[:, None]
        col = self.sky_h[bq] + (self.sky_z[bq] - self.sky_h[bq]) * e.sqrt()
        col = col * (1 + self.sky_c[bq][:, None] * (tx[:, 0:1] - 0.5) * 2 * (0.25 + 0.75 * e))
        col = col + tx[:, 3:4] * self.sky_s[bq][:, None] * (e * 3).clamp(max=1)
        cl = ((tx[:, 0:1] - 0.52) * 6.0).clamp(0, 1) * self.sky_w[bq][:, None] * (0.3 + 0.7 * e)       # white clouds in a clear sky
        col = col * (1 - cl) + 1.05 * cl
        hz = torch.exp(-e * 9.0) * 0.9
        col = col * (1 - hz) + self.fogcol[bq] * hz
        bg = self.sky_bg[bq]
        bi = ((bg >= 0) & (el < SKY_BG * 2 / math.pi)).nonzero().flatten()      # what stands on the horizon, hazy with distance
        if bi.numel():
            ivb = ((1 - el[bi] * (math.pi / 2 / SKY_BG)).clamp(0, 0.9999) * TS).long()
            tb = self.atlas[self.base[bg[bi] * NMIP] + ivb * TS + iu[bi]].float() * (1.0 / 255)
            land = tb[:, :3] ** 2 * self.amb[bq[bi]] * 1.1
            col[bi] = col[bi] + (land * 0.7 + self.fogcol[bq[bi]] * 0.3 - col[bi]) * tb[:, 3:4]
        rgb[ei] = col

    def _glitch(self, fr, strength):
        """Tear the picture into bands shifted sideways, with the colours pulled apart."""
        n, H, W = fr.shape[0], self.H, self.W
        dev = self.dev
        nb = 18
        edges = torch.sort(self._rand(n, nb), dim=1)[0]
        rows = (torch.arange(H, device=dev) + 0.5) / H
        bi = (rows.view(1, H, 1) > edges.view(n, 1, nb)).sum(-1)                          # band of each row
        amt = (self._rand(n, nb + 1) - 0.5) * (self._rand(n, nb + 1) < 0.55).float()
        shift = (amt.gather(1, bi) * strength.view(n, 1) * W * 0.5).long()               # [n, H]
        xs = torch.arange(W, device=dev).view(1, 1, W)
        outc = []
        for c, extra in enumerate((7, 0, -7)):
            ex = (extra * strength).long().view(n, 1, 1)
            idx = (xs + shift.view(n, H, 1) + ex) % W
            outc.append(torch.gather(fr[..., c], 2, idx))
        o = torch.stack(outc, -1)
        inv = (self._rand(n, nb + 1) < 0.08 * strength.view(n, 1)).gather(1, bi)          # a few bands in negative
        inv = inv.view(n, H, 1, 1) & (o.float().mean(dim=(1, 2, 3), keepdim=True) > 20)   # not on a black screen
        o = torch.where(inv, 255 - o, o)
        line = (self._rand(n, nb + 1) < 0.12 * strength.view(n, 1)).gather(1, bi) & (torch.arange(H, device=dev).view(1, H) % 7 == 0)
        return torch.where(line.view(n, H, 1, 1), torch.full_like(o, 150), o)
