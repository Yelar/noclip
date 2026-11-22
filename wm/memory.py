"""Memory of places: which earlier frames the model is shown besides the last few seconds.

The model's window covers a few seconds. So that a room is still the same room when the player comes back to it,
the model also gets up to R *memory frames*: frames from longer ago that were seen from nearly the same spot in
nearly the same direction as the frame about to be drawn. They are chosen by pose, here; the same rule serves
training (poses recorded by the simulator, wm/data.py) and playing (poses reckoned from the keys, `Pose`, and
frames the model drew itself, `Bank`).
"""
import math

import numpy as np

G = 64.0                    # the simulator's maps wrap around after this many metres
POS_S, YAW_S, PITCH_S = 1.5, 0.6, 0.5       # metres / radians that count as one unit of distance between two views
MAX_SCORE = 2.5             # views further apart than this are of no use to each other
GAP = 5                     # chosen frames are at least this many frames apart (neighbours show the same)
STRIDE = 4                  # the model's frame positions of the memory slots are this far apart


def wrap(d, period=G):
    return (d + period / 2) % period - period / 2


def score(x, z, yaw, pitch, x0, z0, yaw0, pitch0, period=G):
    """How far the views at (x, z, yaw, pitch) (arrays) are from the view at (x0, z0, yaw0, pitch0): 0 = the same."""
    d = np.hypot(wrap(x - x0, period), wrap(z - z0, period)) if period else np.hypot(x - x0, z - z0)
    return d / POS_S + np.abs(wrap(yaw - yaw0, 2 * math.pi)) / YAW_S + np.abs(pitch - pitch0) / PITCH_S


def choose(sc, when, n, gap=GAP):
    """Indices of up to n entries with the lowest scores below MAX_SCORE, no two closer in time than `gap`."""
    out = []
    for j in np.argsort(sc, kind="stable"):
        if sc[j] > MAX_SCORE or len(out) >= n:
            break
        if all(abs(int(when[j]) - int(when[c])) >= gap for c in out):
            out.append(int(j))
    return out


class Pose:
    """Where the player probably is, reckoned from the actions alone (the simulator's own rules of movement, without
    its walls: walking into a wall makes this drift, which the retrieval tolerates up to a point)."""

    def __init__(self, fps=10.0, turn_max=0.6, pitch_max=0.35, move=(1.4, 3.0, 1.1, 1.2), pitch_lim=0.8):
        self.dt, self.turn_max, self.pitch_max, self.move, self.pitch_lim = 1.0 / fps, turn_max, pitch_max, move, pitch_lim
        self.reset()

    def reset(self):
        self.x = self.z = self.yaw = self.pitch = self.vx = self.vz = 0.0

    def step(self, act):
        """act: forward, right, turn, pitch, sprint, ... as the model gets them."""
        f, r, turn, pit, sprint = (float(v) for v in act[:5])
        walk, run, back, strafe = self.move
        self.yaw += max(-1.0, min(1.0, turn)) * self.turn_max
        self.pitch = max(-self.pitch_lim, min(self.pitch_lim, self.pitch + max(-1.0, min(1.0, pit)) * self.pitch_max))
        sp = run if (sprint > 0.5 and f > 0) else walk
        vf, vr = f * (sp if f > 0 else back), r * strafe
        n = math.hypot(vf, vr)
        if n > sp:
            vf, vr = vf * sp / n, vr * sp / n
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.vx += (cy * vf - sy * vr - self.vx) * 0.7
        self.vz += (sy * vf + cy * vr - self.vz) * 0.7
        self.x += self.vx * self.dt
        self.z += self.vz * self.dt
        return self.x, self.z, self.yaw, self.pitch


class Bank:
    """The frames of a dream that can be recalled: every `every`-th frame with the pose it was drawn at."""

    def __init__(self, slots, every=3, cap=4000, period=0.0):
        """period: the length after which positions repeat (the simulator's own poses), 0 for reckoned poses."""
        self.R, self.every, self.cap, self.period = slots, every, cap, period
        self.clear()

    def clear(self):
        self.pose, self.when, self.items = [], [], []

    def add(self, i, pose, item):
        """Frame number i drawn at pose (x, z, yaw, pitch); item: whatever the caller wants back (the latent)."""
        if i % self.every == 0:
            self.pose.append(pose); self.when.append(i); self.items.append(item)
            if len(self.when) > self.cap:
                del self.pose[0], self.when[0], self.items[0]

    def recall(self, pose, now, older_than):
        """The stored frames to show for a frame at `pose`: those nearest in view among the frames more than
        `older_than` frames before `now`, oldest first. -> [(frame number, item)]"""
        when = np.asarray(self.when)
        n = int(np.searchsorted(when, now - older_than))
        if n == 0:
            return []
        p = np.asarray(self.pose[:n], np.float64)
        sc = score(p[:, 0], p[:, 1], p[:, 2], p[:, 3], *pose, period=self.period)
        return [(int(when[j]), self.items[j]) for j in sorted(choose(sc, when[:n], self.R))]
