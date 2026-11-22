"""A scripted player. It is not good at the game and is not meant to be: it produces the spread of behaviour a
person at the keyboard might show, so that the world model sees every key and mouse movement in many situations."""
import math

import torch

from .core import NTR, PITCH_LIM, PITCH_MAX, TURN_MAX
from .levels import G

WANDER, SPRINT, LOOK, SPIN, BACK, STRAFE, RANDOM, BUMP, FLEE, APPROACH, STARE, RETURN = range(12)
#            wander sprint look  spin  back  strafe random bump
_P = [0.22, 0.07, 0.14, 0.07, 0.12, 0.18, 0.15, 0.05]
NOSE, FLAT = 1.3, 2.5      # a wall nearer than NOSE straight ahead, or than FLAT across 45 degrees, leaves little to see
_DUR = [(20, 100), (20, 80), (10, 60), (5, 28), (10, 40), (10, 50), (15, 80), (10, 40), (25, 70), (20, 60), (10, 40), (30, 120)]


def _wr(d):
    return (d + G / 2) % G - G / 2


def _wa(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Explorer:
    def __init__(self, B, device="cuda", seed=0):
        self.B, self.dev = B, torch.device(device)
        dev = self.dev
        self.g = torch.Generator(device=dev)
        self.g.manual_seed(int(seed) * 104729 + 71)
        self.rel = torch.arange(16, device=dev) * (math.pi / 8) - math.pi          # index 8 looks straight ahead
        self.wfront = 0.55 + 0.45 * torch.cos(self.rel[4:13])
        z = lambda *s: torch.zeros(*s, device=dev)
        self.mode = torch.zeros(B, dtype=torch.long, device=dev)
        self.timer = z(B)
        self.bias = z(B)
        self.noise = z(B, 16)
        self.spin = z(B)
        self.keys = z(B, 3)
        self.side = z(B)
        self.yr, self.pr, self.ptgt = z(B), z(B), z(B)
        self.light = torch.zeros(B, dtype=torch.bool, device=dev)
        self.nc = 300 + 900 * self._r(B)
        self.stuck = z(B)
        self.brave = self._r(B)
        self.first = True
        self.bt = torch.zeros(B, dtype=torch.long, device=dev)      # RETURN: the point of the trail being walked back to
        # doors: 0 nothing, 1 to the spot in front of one, 2 face it and press E, 3 walk in, 4 inside (doing whatever),
        # 5 back to the doorway, 6 out to the spot in front, 7 turn round and shut it
        self.dm = torch.zeros(B, dtype=torch.long, device=dev)
        self.di = torch.zeros(B, dtype=torch.long, device=dev)
        self.dtm = z(B)
        self.curious = self._r(B)                                   # how much this player cares for doors
        self.nose = z(B)                                            # frames in a row with little but a wall to see
        self.patience = 4 + 10 * self._r(B)                         # ... and how many of them this player stands

    def _r(self, *s):
        return torch.rand(*s, generator=self.g, device=self.dev)

    def _n(self, *s):
        return torch.randn(*s, generator=self.g, device=self.dev)

    def act(self, sim):
        B, dev = self.B, self.dev
        free = sim.probe(self.rel)                                   # [B, 16]
        run = sim.axis != 0                                          # Level !
        alive = sim.trans == 0
        if self.first:
            self.light = sim.light.clone()
            self.first = False
        # a new thing to do when the current one has run its time
        self.timer = self.timer - 1
        new = self.timer <= 0
        if bool(new.any()):
            p = torch.tensor(_P, device=dev).expand(B, -1).clone()
            p[run] = torch.tensor([0.22, 0.56, 0.03, 0.08, 0.02, 0.03, 0.04, 0.02], device=dev)   # runs, and glances back
            m = torch.multinomial(p, 1, generator=self.g)[:, 0]
            self.mode = torch.where(new, m, self.mode)
            back = new & ~run & (self._r(B) < 0.12) & (sim.tr_count > 14)       # walk back the way one came
            self.mode = torch.where(back, torch.full_like(self.mode, RETURN), self.mode)
            self.bt = torch.where(back, sim.tr_count - 1 - (6 + 24 * self._r(B)).long(), self.bt)
            lo = torch.tensor([d[0] for d in _DUR], device=dev, dtype=torch.float32)[self.mode]
            hi = torch.tensor([d[1] for d in _DUR], device=dev, dtype=torch.float32)[self.mode]
            self.timer = torch.where(new, lo + (hi - lo) * self._r(B), self.timer)
            self.bias = torch.where(new, self._n(B) * 1.5, self.bias)
            sg = torch.where(self._r(B) < 0.5, -1.0, 1.0)
            self.spin = torch.where(new, sg * (0.12 + 0.5 * self._r(B) ** 2), self.spin)
            self.side = torch.where(new, sg, self.side)
            self.brave = torch.where(new & (self._r(B) < 0.2), self._r(B), self.brave)
            pat = torch.where(self._r(B) < 0.08, 30 + 40 * self._r(B), 4 + 10 * self._r(B))     # now and then a long stare
            self.patience = torch.where(new, pat, self.patience)
        # reacting to what is there
        seen = (sim.ent_vis > 0) & alive
        react = seen & (self.mode < FLEE) & (self._r(B) < 0.12)
        if bool(react.any()):
            u = self._r(B)
            m = torch.where(u < 0.55 - 0.3 * self.brave, FLEE, torch.where(u < 0.75, STARE, torch.where(u < 0.9 * self.brave + 0.1, APPROACH, WANDER)))
            m = torch.where(m == WANDER, self.mode, m)
            self.mode = torch.where(react, m, self.mode)
            self.timer = torch.where(react, 20 + 50 * self._r(B), self.timer)
        panic = (sim.danger > 0.25) & (self.mode != FLEE) & (self._r(B) < 0.3 * (1.2 - self.brave)) & alive
        self.mode = torch.where(panic, torch.full_like(self.mode, FLEE), self.mode)
        self.timer = torch.where(panic, 30 + 40 * self._r(B), self.timer)
        md = self.mode
        # where is the thing, relative to the heading
        e_rel = sim.e_pos - sim.pos[:, None]
        e_rel = (e_rel + 32) % 64 - 32
        e_d = torch.where(sim.e_type > 0, e_rel.norm(dim=2), torch.full((B, e_rel.shape[1]), 99.0, device=dev))
        ie = e_d.argmin(1)
        er = e_rel[torch.arange(B, device=dev), ie]
        e_ang = (torch.atan2(er[:, 1], er[:, 0]) - sim.yaw + math.pi) % (2 * math.pi) - math.pi
        has = e_d.min(1)[0] < 98
        # steering towards open space
        resample = self._r(B) < 0.08
        self.noise = torch.where(resample[:, None], self._n(B, 16) * 1.6, self.noise)
        want = torch.zeros(B, device=dev)                                         # a heading the mode asks for
        pull = torch.zeros(B, device=dev)
        axis_rel = (torch.where(sim.axis > 0, 0.0, math.pi) - sim.yaw + math.pi) % (2 * math.pi) - math.pi
        want = torch.where(run, axis_rel, want)
        pull = torch.where(run, torch.full_like(pull, 4.0), pull)
        away = (e_ang + 2 * math.pi) % (2 * math.pi) - math.pi                    # the opposite direction
        fl = (md == FLEE) & has
        want = torch.where(fl, away, want)
        pull = torch.where(fl, torch.full_like(pull, 6.0), pull)
        rt = md == RETURN
        if bool(rt.any()):
            lo_ = max(sim.tr_count - NTR + 1, 0)
            self.bt = self.bt.clamp(lo_, max(sim.tr_count - 1, lo_))
            rel_ = _wr(sim.trail[torch.arange(B, device=dev), self.bt % NTR] - sim.pos)
            self.bt = self.bt - (rt & (rel_.norm(dim=1) < 0.9)).long()
            self.timer = torch.where(rt & (self.bt < lo_), torch.zeros_like(self.timer), self.timer)
            want = torch.where(rt, _wa(torch.atan2(rel_[:, 1], rel_[:, 0]) - sim.yaw), want)
            pull = torch.where(rt, torch.full_like(pull, 5.0), pull)
        dd = (self.rel[None] - want[:, None] + math.pi) % (2 * math.pi) - math.pi
        score = free.clamp(max=14.0) + self.noise + pull[:, None] * torch.cos(dd)
        sf = score[:, 4:13] * self.wfront + self.bias[:, None] * torch.sign(self.rel[4:13])[None] * 0.8
        best_f = self.rel[4:13][sf.argmax(1)]
        best_a = self.rel[score.argmax(1)]
        tight = free[:, 8] < 0.8
        use_all = tight | ((fl | run | rt) & (dd[:, 8].abs() > 1.6))
        head = torch.where(use_all, best_a, best_f)
        steer = (head * 0.45 / TURN_MAX).clamp(-0.55, 0.55)
        fwd = torch.zeros(B, device=dev)
        right = torch.zeros(B, device=dev)
        sprint = torch.zeros(B, dtype=torch.bool, device=dev)
        tgt = torch.zeros(B, device=dev)                                          # wanted turn this frame
        mov = (md == WANDER) | (md == SPRINT) | (md == FLEE) | rt
        fwd = torch.where(mov, torch.where(tight & (head.abs() > 1.0), 0.0, 1.0), fwd)
        tgt = torch.where(mov, steer + 0.03 * torch.sin(sim.t * 0.21 + self.bias), tgt)
        sprint = sprint | (md == SPRINT) | (md == FLEE) | (run & mov & (self.brave > 0.08))
        lk = md == LOOK
        tgt = torch.where(lk, self.spin * 0.6 * torch.sin(sim.t * 0.35 + self.bias * 3), tgt)
        sn = md == SPIN
        tgt = torch.where(sn, self.spin, tgt)
        bk = md == BACK
        fwd = torch.where(bk, torch.full_like(fwd, -1.0), fwd)
        tgt = torch.where(bk, self.spin * 0.15, tgt)
        sf_ = md == STRAFE
        right = torch.where(sf_, self.side, right)
        fwd = torch.where(sf_ & (self.bias > 0.3), torch.ones_like(fwd), fwd)
        tgt = torch.where(sf_, -self.side * self.spin.abs() * (self.bias.abs() > 1.0).float() * 0.5, tgt)
        rd = md == RANDOM
        rk = (self._r(B) < 0.15) & rd
        if bool(rk.any()):
            u = self._r(B, 3)
            k0 = torch.where(u[:, 0] < 0.2, -1.0, torch.where(u[:, 0] < 0.5, 0.0, 1.0))
            k1 = torch.where(u[:, 1] < 0.2, -1.0, torch.where(u[:, 1] < 0.8, 0.0, 1.0))
            k2 = (u[:, 2] < 0.25).float()
            self.keys = torch.where(rk[:, None], torch.stack([k0, k1, k2], 1), self.keys)
        fwd = torch.where(rd, self.keys[:, 0], fwd)
        right = torch.where(rd, self.keys[:, 1], right)
        sprint = sprint | (rd & (self.keys[:, 2] > 0))
        tgt = torch.where(rd, self.yr * 0.9 + 0.25 * self._n(B), tgt)
        bp = md == BUMP
        fwd = torch.where(bp, torch.ones_like(fwd), fwd)
        tgt = torch.where(bp, self.spin * 0.05, tgt)
        ap = (md == APPROACH) & has
        st = (md == STARE) & has
        tgt = torch.where(ap | st, (e_ang * 0.5 / TURN_MAX).clamp(-0.6, 0.6), tgt)
        fwd = torch.where(ap, torch.ones_like(fwd), fwd)
        use = (self._r(B) < torch.where(rd, 0.02, 0.003)) & alive                 # E for no reason, now and then
        # doors
        ar = torch.arange(B, device=dev)
        dst = sim.k_door.gather(1, sim.wall.view(B, -1).gather(1, sim.d_cell.clamp(min=0)))
        dst = torch.where(sim.d_cell >= 0, dst, torch.zeros_like(dst))                    # [B, ND]: 1 shut, 3 open
        self.dm = torch.where(alive & (md != FLEE) & ~run, self.dm, torch.zeros_like(self.dm))
        frel = _wr(sim.d_front - sim.pos[:, None])
        fd = torch.where((dst == 1) | (dst == 3), frel.norm(dim=2), torch.full_like(dst, 99, dtype=torch.float32))
        near_d, near_i = fd.min(dim=1)
        fr1 = frel[ar, near_i]
        b16 = torch.round((_wa(torch.atan2(fr1[:, 1], fr1[:, 0]) - sim.yaw) + math.pi) / (math.pi / 8)).long() % 16
        clear = free[ar, b16] > near_d - 0.4                                            # nothing between here and that spot
        begin = (self.dm == 0) & alive & (md < FLEE) & ~run & (near_d < 7.0) & clear & \
            (self._r(B) < (0.01 + 0.07 * self.curious) * torch.where(near_d < 3.0, 2.5, 1.0))
        self.dm = torch.where(begin, torch.ones_like(self.dm), self.dm)
        self.di = torch.where(begin, near_i, self.di)
        self.dtm = torch.where(begin, torch.full_like(self.dtm, 70.0), self.dtm)
        if bool((self.dm > 0).any()):
            dm = self.dm
            cell = sim.d_cell[ar, self.di].clamp(min=0)
            dc = torch.stack([(cell % G).float() + 0.5, (cell // G).float() + 0.5], 1)     # the door
            dn = sim.d_nrm[ar, self.di]
            st = dst[ar, self.di]
            tp = sim.d_front[ar, self.di]                                                   # phases 1 and 6
            tp = torch.where(((dm == 2) | (dm == 7))[:, None], dc, tp)
            tp = torch.where((dm == 3)[:, None], sim.d_room[ar, self.di], tp)
            tp = torch.where((dm == 5)[:, None], dc - dn * 0.95, tp)
            rel_ = _wr(tp - sim.pos)
            dist = rel_.norm(dim=1)
            err = _wa(torch.atan2(rel_[:, 1], rel_[:, 0]) - sim.yaw)
            walk = (dm == 1) | ((dm == 3) & (st == 3)) | (dm == 5) | (dm == 6)
            ctl = (dm > 0) & (dm != 4)
            fwd = torch.where(ctl, (walk & (err.abs() < 0.7)).float(), fwd)
            right = torch.where(ctl, torch.zeros_like(right), right)
            sprint = sprint & ~ctl
            tgt = torch.where(ctl, (err * 0.5 / TURN_MAX).clamp(-0.6, 0.6), tgt)
            aimed = err.abs() < 0.25
            u = self._r(B, 3)
            press = (dm == 2) & aimed & (st == 1) & (sim.door_now == 1)
            shut = (dm == 7) & aimed & (st == 3) & (sim.door_now == 3)
            use = (use & ~ctl) | press | shut
            nd = dm.clone()
            nd = torch.where((dm == 1) & (dist < 0.35), torch.full_like(nd, 2), nd)
            nd = torch.where(press | ((dm == 2) & aimed & (st == 3)), torch.where(u[:, 0] < 0.25, 0, 3), nd)     # some only look in
            nd = torch.where((dm == 3) & (st == 3) & (dist < 0.8), torch.full_like(nd, 4), nd)
            nd = torch.where((dm == 5) & (dist < 0.4), torch.full_like(nd, 6), nd)
            nd = torch.where((dm == 6) & (dist < 0.35), torch.where(u[:, 1] < 0.35, 7, 0), nd)
            nd = torch.where(shut, torch.zeros_like(nd), nd)
            ch = nd != dm
            self.dtm = torch.where(ch, torch.where(nd == 4, 15 + 80 * u[:, 2], torch.full_like(self.dtm, 60.0)), self.dtm - 1)
            out = (self.dtm <= 0) & ~ch
            nd = torch.where(out, torch.where(dm == 4, 5, 0), nd)
            self.dtm = torch.where(out, torch.full_like(self.dtm, 60.0), self.dtm)
            look = ch & (nd == 4)                                                             # inside: look around
            self.mode = torch.where(look, torch.where(u[:, 1] < 0.5, LOOK, SPIN), self.mode)
            self.timer = torch.where(look, 15 + 30 * u[:, 2], self.timer)
            self.dm = nd
        # little but a wall to see: after a moment a person turns to where there is more, whatever else they were
        # doing (not while fleeing or watching a thing, and not at a door being opened or shut). In a room behind a
        # door every wall is near, so there only a wall right in the face counts.
        blank = torch.where(self.dm == 4, free[:, 8] < NOSE, free[:, 7:10].amax(1) < FLAT)
        self.nose = torch.where(blank & alive, self.nose + 1, torch.zeros_like(self.nose))
        turn_away = (self.nose > self.patience) & ~mov & ~ap & ~(md == STARE) & ((self.dm == 0) | (self.dm == 4))
        tgt = torch.where(turn_away, (best_a * 0.45 / TURN_MAX).clamp(-0.55, 0.55), tgt)
        fwd = torch.where(turn_away & (fwd > 0), torch.zeros_like(fwd), fwd)
        # walking into something for a while: do something else
        self.stuck = torch.where((fwd > 0) & (sim._speed < 0.25) & mov, self.stuck + 1, torch.zeros_like(self.stuck))
        self.timer = torch.where(self.stuck > 8, torch.zeros_like(self.timer), self.timer)
        # the mouse: follows the target smoothly, with noise and the odd flick
        self.yr = self.yr + (tgt - self.yr) * 0.5 + 0.012 * self._n(B)
        flick = (self._r(B) < 0.012).float() * torch.where(self._r(B) < 0.5, -1.0, 1.0) * (0.35 + 0.65 * self._r(B))
        turn = (self.yr + flick).clamp(-1, 1)
        np_ = self._r(B) < torch.where(lk, 0.08, 0.02)
        amp = torch.where(lk | rd, 0.5, 0.12) + 0.5 * (self._r(B) < 0.12).float()
        self.ptgt = torch.where(np_, (self._n(B) * amp).clamp(-PITCH_LIM, PITCH_LIM), self.ptgt)
        back = torch.where(self.ptgt.abs() > 0.3, 0.10, 0.01)          # a look far up or down is a glance, not a stare
        self.ptgt = torch.where(self._r(B) < back, torch.zeros_like(self.ptgt), self.ptgt)
        self.pr = self.pr + (((self.ptgt - sim.pitch) * 0.35 / PITCH_MAX).clamp(-1, 1) - self.pr) * 0.5 + 0.01 * self._n(B)
        pitch = self.pr.clamp(-1, 1)
        # flashlight: mostly on where it is dark, now and then toggled for no reason
        p_on = torch.where(sim.bright < 0.05, torch.maximum(sim.flashp, torch.full_like(sim.flashp, 0.85)), sim.flashp)
        tog = self._r(B) < 0.012
        self.light = torch.where(tog, self._r(B) < p_on, self.light)
        self.light = torch.where((sim.bright < 0.03) & (self._r(B) < 0.05) & (sim.flashp > 0.5), torch.ones_like(self.light), self.light)
        # noclip: every half minute to two minutes, and sometimes in a panic
        self.nc = self.nc - alive.float()
        press = ((self.nc <= 0) | ((sim.danger > 0.6) & (self._r(B) < 0.02))) & alive
        self.nc = torch.where(press, 300 + 900 * self._r(B), self.nc)
        self.timer = torch.where(press, torch.zeros_like(self.timer), self.timer)
        return torch.stack([fwd, right, turn, pitch, sprint.float(), self.light.float(), press.float(), use.float()], 1)
