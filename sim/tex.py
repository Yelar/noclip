"""Procedural textures and sprites, drawn once per process with torch and packed into one mip-mapped atlas.

Every texture is TS x TS, tileable, RGBA in display space. For surfaces the alpha channel marks what glows (a light
panel, a lamp, a sign); for sprites it is opacity. Coordinates: X to the right, Y downwards (Y = 1 is the floor end of
a wall texture and the feet of a sprite).
"""
import math

import torch
import torch.nn.functional as F

TS = 512
NMIP = 8  # 512 ... 4
PI = math.pi


class Ctx:
    def __init__(self, dev, seed=1234):
        self.dev = dev
        self.g = torch.Generator().manual_seed(seed)
        a = (torch.arange(TS, device=dev, dtype=torch.float32) + 0.5) / TS
        self.Y, self.X = torch.meshgrid(a, a, indexing="ij")
        fy = torch.fft.fftfreq(TS, d=1.0 / TS, device=dev)
        fx = torch.fft.rfftfreq(TS, d=1.0 / TS, device=dev)
        self.f = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.g).to(self.dev)

    def r(self, lo=0.0, hi=1.0):
        return lo + (hi - lo) * float(torch.rand((), generator=self.g))

    def fbm(self, beta=2.0, lo=1.0, hi=None, sx=1.0, sy=1.0):
        """Tileable noise, power spectrum 1/f^beta between lo and hi cycles per texture, mean 0 and std 1.
        sx, sy > 1 stretch it along that axis."""
        hi = hi or TS / 2
        if sx != 1.0 or sy != 1.0:
            fy = torch.fft.fftfreq(TS, d=1.0 / TS, device=self.dev) * sy
            fx = torch.fft.rfftfreq(TS, d=1.0 / TS, device=self.dev) * sx
            f = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        else:
            f = self.f
        amp = torch.where((f >= lo) & (f <= hi), f.clamp(min=1.0) ** (-beta / 2), torch.zeros_like(f))
        ph = self.rand(TS, TS // 2 + 1) * (2 * PI)
        x = torch.fft.irfft2(amp * torch.exp(1j * ph), s=(TS, TS))
        return (x - x.mean()) / (x.std() + 1e-8)

    def n01(self, beta=2.0, lo=1.0, hi=None, sx=1.0, sy=1.0):
        return 0.5 * (1 + torch.erf(self.fbm(beta, lo, hi, sx, sy) / 1.4142))

    def cellrand(self, ix, iy, salt=0):
        """A random number per integer cell (ix, iy), periodic with 64."""
        if not hasattr(self, "_tbl"):
            self._tbl = self.rand(8, 64, 64)
        return self._tbl[salt % 8][iy.long() % 64, ix.long() % 64]


def col(c, r, g, b):
    return torch.tensor([r, g, b], device=c.dev, dtype=torch.float32).view(3, 1, 1)


def lerp(a, b, t):
    return a + (b - a) * t


def sstep(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def lines(x, n, w):
    """1 on n evenly spaced lines per unit of x (one of them at 0), falling to 0 at distance w."""
    d = (x * n - torch.round(x * n)).abs() / n
    return (1 - d / w).clamp(0, 1)


def band(x, a, b, w=0.004):
    """1 inside [a, b], soft edge of width w."""
    return sstep(a - w, a + w, x) * (1 - sstep(b - w, b + w, x))


def box(c, x0, y0, x1, y1, w=0.004):
    return band(c.X, x0, x1, w) * band(c.Y, y0, y1, w)


def out(c, rgb, em=None):
    a = torch.zeros(1, TS, TS, device=c.dev) if em is None else em.view(1, TS, TS).clamp(0, 1)
    return torch.cat([rgb.clamp(0, 1).expand(3, TS, TS), a], 0)


def drips(c, n=40.0, length=0.5):
    """Streaks running down from the top edge."""
    colr = c.n01(1.5, 4, n, sy=1e3)          # varies along x only
    reach = sstep(0.55, 0.95, colr) * length
    return (sstep(0.0, 1.0, (reach - c.Y) / (reach + 1e-3)) * sstep(0.55, 0.8, colr)).clamp(0, 1)


# ---------------------------------------------------------------- surfaces

def wallpaper(c, base=(0.80, 0.70, 0.34), style=0, wear=0.5, baseboard=(0.34, 0.27, 0.15)):
    X, Y = c.X, c.Y
    grain = c.fbm(0.6, 30)
    blot = c.n01(3.0, 1, 10)
    if style == 0:      # narrow stripes
        pat = 0.5 + 0.5 * torch.cos(2 * PI * X * 24)
    elif style == 1:    # small diamonds
        pat = ((torch.sin(PI * X * 16).abs() + torch.sin(PI * Y * 20).abs()) > 1.25).float()
        pat = F.avg_pool2d(pat[None, None], 3, 1, 1)[0, 0]
    elif style == 2:    # chevrons
        pat = 0.5 + 0.5 * torch.cos(2 * PI * (Y * 22 + ((X * 8) % 1 - 0.5).abs() * 3))
    else:               # plain with broad faded stripes
        pat = 0.5 + 0.5 * torch.cos(2 * PI * X * 6)
    b = col(c, *base)
    rgb = b * (0.93 + 0.07 * pat) * (1 + 0.025 * grain)
    stain = sstep(0.62, 0.97, blot) * wear * 0.55
    rgb = lerp(rgb, rgb * col(c, 0.66, 0.56, 0.40), stain)
    rgb = lerp(rgb, rgb * col(c, 0.55, 0.46, 0.30), drips(c) * wear * 0.8)
    rgb = rgb * (1 - 0.25 * sstep(0.75, 0.96, Y) * wear)           # scuffed towards the floor
    bb = sstep(0.955, 0.96, Y)
    rgb = lerp(rgb, col(c, *baseboard) * (1 + 0.05 * grain) * (1 + 0.5 * band(Y, 0.958, 0.966, 0.002)), bb)
    return out(c, rgb)


def carpet(c, base=(0.52, 0.44, 0.26), blot=0.5, pattern=0, accent=(0.75, 0.62, 0.25)):
    X, Y = c.X, c.Y
    speck = c.fbm(0.2, 60)
    mid = c.n01(2.4, 2, 24)
    big = c.n01(3.0, 1, 5)
    rgb = col(c, *base) * (0.86 + 0.07 * speck + 0.16 * (mid - 0.5))
    rgb = lerp(rgb, rgb * col(c, 0.62, 0.56, 0.48), sstep(0.55, 0.95, big) * blot * 0.6)
    if pattern == 1:    # hotel: diamonds and dots
        d = ((X * 4) % 1 - 0.5).abs() + ((Y * 4) % 1 - 0.5).abs()
        ring = band(d, 0.36, 0.42, 0.01) + band(d, 0.0, 0.07, 0.01)
        e = ((X * 4 + 0.5) % 1 - 0.5).abs() + ((Y * 4 + 0.5) % 1 - 0.5).abs()
        ring = ring + band(e, 0.10, 0.14, 0.01)
        rgb = lerp(rgb, col(c, *accent) * (0.9 + 0.1 * speck), ring.clamp(0, 1) * 0.85)
    elif pattern == 2:  # office carpet tiles
        rgb = rgb * (1 - 0.35 * torch.maximum(lines(X, 4, 0.004), lines(Y, 4, 0.004)))
        rgb = rgb * (0.94 + 0.12 * c.cellrand(torch.floor(X * 4), torch.floor(Y * 4))[None])
    return out(c, rgb)


def ceil_tile(c, base=(0.80, 0.78, 0.70), n=4, stains=0.5):
    X, Y = c.X, c.Y
    pin = (c.fbm(0.0, 70) > 1.5).float()
    pin = F.avg_pool2d(pin[None, None], 3, 1, 1)[0, 0]
    tone = 0.94 + 0.08 * c.cellrand(torch.floor(X * n), torch.floor(Y * n), 1)
    rgb = col(c, *base) * tone * (1 - 0.35 * pin) * (1 + 0.02 * c.fbm(1.0, 8))
    blot = c.n01(3.0, 1, 6)
    ringm = band(blot, 0.80, 0.84, 0.01) * 0.6 + sstep(0.84, 0.95, blot) * 0.45
    rgb = lerp(rgb, rgb * col(c, 0.70, 0.56, 0.36), ringm * stains)
    grid = torch.maximum(lines(X, n, 0.006), lines(Y, n, 0.006))
    rgb = lerp(rgb, col(c, 0.55, 0.54, 0.50), grid)
    return out(c, rgb)


def ceil_panel(c, base=(0.80, 0.78, 0.70), kind=0):
    """One ceiling cell (1 m) with a light fitting in it; alpha marks the glowing part."""
    X, Y = c.X, c.Y
    t = ceil_tile(c, base, 2, 0.2)[:3]
    if kind == 0:       # recessed fluorescent troffer
        m = box(c, 0.06, 0.26, 0.94, 0.74, 0.006)
        ribs = 0.92 + 0.08 * torch.cos(2 * PI * Y * 40)
        glow = col(c, 1.0, 1.0, 0.96) * ribs * (0.86 + 0.14 * band(X, 0.1, 0.9, 0.1))
        frame = box(c, 0.04, 0.24, 0.96, 0.76, 0.004) - m
        rgb = lerp(t, col(c, 0.45, 0.45, 0.43), frame.clamp(0, 1))
        return out(c, lerp(rgb, glow, m), m)
    if kind == 1:       # two bare tubes
        m = (box(c, 0.08, 0.36, 0.92, 0.42, 0.006) + box(c, 0.08, 0.58, 0.92, 0.64, 0.006)).clamp(0, 1)
        house = box(c, 0.05, 0.30, 0.95, 0.70, 0.004)
        rgb = lerp(t, col(c, 0.60, 0.60, 0.58), house)
        return out(c, lerp(rgb, col(c, 1, 1, 1), m), m)
    if kind == 2:       # round industrial lamp
        d = torch.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        m = 1 - sstep(0.16, 0.18, d)
        rim = band(d, 0.18, 0.24, 0.006)
        rgb = lerp(t, col(c, 0.2, 0.2, 0.2), rim)
        return out(c, lerp(rgb, col(c, 1, 0.97, 0.9), m), m)
    # skylight: almost the whole cell
    m = box(c, 0.03, 0.03, 0.97, 0.97, 0.006)
    bars = torch.maximum(lines(X, 2, 0.012), lines(Y, 2, 0.012))
    glow = col(c, 0.92, 0.97, 1.0) * (1 - 0.5 * bars)
    return out(c, lerp(t, glow, m), m * (1 - 0.6 * bars))


def concrete(c, tone=0.42, warm=0.0, cracks=0.6, panels=2, stains=0.5, stripe=None):
    X, Y = c.X, c.Y
    a = c.n01(2.2, 1, 40)
    fine = c.fbm(0.4, 40)
    rgb = col(c, tone * (1 + warm), tone, tone * (1 - warm)) * (0.78 + 0.4 * (a - 0.5) + 0.05 * fine)
    ridge = c.fbm(2.6, 2, 60).abs()
    crack = (1 - sstep(0.0, 0.035, ridge)) * sstep(0.5, 0.7, c.n01(3.0, 1, 4))
    rgb = rgb * (1 - 0.6 * crack * cracks)
    if panels:
        rgb = rgb * (1 - 0.3 * torch.maximum(lines(X, panels, 0.004), lines(Y, panels, 0.004)))
    rgb = lerp(rgb, rgb * 0.6, drips(c, 30, 0.7) * stains)
    if stripe is not None:
        s = lines(X, 1, 0.035) * (c.n01(1.0, 8, 80) > 0.25).float()
        rgb = lerp(rgb, col(c, *stripe), s * 0.8)
    return out(c, rgb)


def brick(c, base=(0.42, 0.20, 0.15), mortar=(0.45, 0.43, 0.40), rows=14, cols=5, soot=0.5):
    X, Y = c.X, c.Y
    ry = torch.floor(Y * rows)
    xo = X * cols + 0.5 * (ry % 2)
    rx = torch.floor(xo)
    fx, fy = xo - rx, Y * rows - ry
    m = (band(fx, 0.04, 0.96, 0.02) * band(fy, 0.10, 0.90, 0.05))
    tone = 0.75 + 0.5 * c.cellrand(rx, ry, 2)[None]
    hue = c.cellrand(rx, ry, 3)[None]
    b = col(c, *base) * tone * lerp(col(c, 1, 1, 1), col(c, 1.15, 0.85, 0.8), hue)
    rgb = lerp(col(c, *mortar) * (0.9 + 0.1 * c.fbm(0.5, 30)), b * (1 + 0.08 * c.fbm(0.8, 20)), m)
    rgb = rgb * (1 - soot * 0.3 * sstep(0.3, 1.0, c.n01(3.0, 1, 4)))
    rgb = rgb * (1 - 0.3 * soot * drips(c, 20, 0.8))
    return out(c, rgb)


def pipe_wall(c, base=(0.20, 0.19, 0.18), rust=0.6, lamp=False, valve=False):
    """Dark service-tunnel wall with pipes running along it."""
    X, Y = c.X, c.Y
    a = c.n01(2.2, 1, 40)
    rgb = col(c, *base) * (0.7 + 0.6 * (a - 0.5) + 0.05 * c.fbm(0.4, 40))
    rgb = rgb * (1 - 0.3 * lines(X, 2, 0.004))
    for (yc, rad, tone) in ((0.16, 0.050, (0.36, 0.33, 0.30)), (0.30, 0.028, (0.50, 0.30, 0.18)),
                            (0.37, 0.020, (0.42, 0.42, 0.40)), (0.72, 0.040, (0.30, 0.31, 0.33)),
                            (0.86, 0.022, (0.48, 0.42, 0.20))):
        d = ((Y - yc) / rad).clamp(-1, 1)
        m = band(Y, yc - rad, yc + rad, 0.003)
        shade = (1 - d * d).sqrt() * (0.55 + 0.45 * (1 - (d + 0.5).clamp(0, 1)))
        flange = lines(X + 0.13 * yc, 2, 0.012)
        p = col(c, *tone) * (0.35 + 0.9 * shade) * (1 + 0.25 * flange)
        rgb = lerp(rgb * (1 - 0.5 * band(Y, yc + rad, yc + rad + 0.03, 0.01)), p, m)
    strap = lines(X + 0.25, 2, 0.014) * band(Y, 0.10, 0.92, 0.01)
    rgb = lerp(rgb, col(c, 0.50, 0.48, 0.44) * (0.7 + 0.5 * a), strap * 0.9)
    bolt = lines(X + 0.25, 2, 0.008) * (lines(Y, 12, 0.006) > 0.2).float()
    rgb = lerp(rgb, col(c, 0.10, 0.09, 0.08), bolt * 0.9)
    r = sstep(0.55, 0.85, c.n01(2.5, 2, 30)) * rust
    rgb = lerp(rgb, col(c, 0.40, 0.20, 0.08) * (0.7 + 0.5 * a), r)
    rgb = lerp(rgb, rgb * 0.6, drips(c, 30, 0.9) * 0.35)
    em = None
    if lamp:        # caged bulkhead lamp
        d = torch.sqrt(((X - 0.5) / 0.055) ** 2 + ((Y - 0.52) / 0.075) ** 2)
        m = 1 - sstep(0.85, 1.0, d)
        cage = torch.maximum(lines(X - 0.5, 28, 0.004), lines(Y - 0.52, 18, 0.004)) * m
        halo = (1 - sstep(1.0, 3.5, d)) * 0.5
        rgb = lerp(rgb + col(c, 1.0, 0.45, 0.2) * halo * 0.35, col(c, 1.0, 0.8, 0.6), m)
        rgb = lerp(rgb, col(c, 0.08, 0.08, 0.08), cage * 0.8)
        em = m * (1 - 0.7 * cage)
    if valve:
        d = torch.sqrt((X - 0.3) ** 2 + ((Y - 0.30) * 2.5) ** 2)
        wheel = band(d, 0.10, 0.13, 0.004) + lines(X - 0.3, 1, 0.012) * (d < 0.11) + lines(Y - 0.30, 1, 0.005) * (d < 0.11)
        rgb = lerp(rgb, col(c, 0.55, 0.10, 0.08), wheel.clamp(0, 1))
    return out(c, rgb, em)


def plaster(c, base=(0.80, 0.80, 0.78), skirting=(0.35, 0.36, 0.38), wear=0.4, window=False, top=None):
    """Painted wall (offices, hospital); `top` paints the part above a dado line differently."""
    X, Y = c.X, c.Y
    rgb = col(c, *base) * (0.96 + 0.03 * c.fbm(1.0, 6) + 0.012 * c.fbm(0.3, 60))
    if top is not None:
        up = 1 - sstep(0.585, 0.59, Y)
        rgb = lerp(rgb, col(c, *top) * (0.97 + 0.03 * c.fbm(1.0, 6)), up)
        rgb = lerp(rgb, col(c, 0.62, 0.60, 0.55), band(Y, 0.59, 0.615, 0.003))     # handrail
        rgb = rgb * (1 - 0.3 * band(Y, 0.615, 0.635, 0.008))
    scuff = sstep(0.6, 0.9, c.n01(2.5, 3, 40, sx=3.0)) * band(Y, 0.62, 0.95, 0.08)
    rgb = lerp(rgb, rgb * 0.72, scuff * wear)
    rgb = lerp(rgb, rgb * col(c, 0.75, 0.68, 0.55), drips(c, 24, 0.4) * wear * 0.6)
    rgb = lerp(rgb, col(c, *skirting), sstep(0.955, 0.96, Y))
    if window:      # a window that shows nothing
        m = box(c, 0.12, 0.25, 0.88, 0.62, 0.004)
        fr = box(c, 0.10, 0.23, 0.90, 0.64, 0.004)
        glass = col(c, 0.015, 0.02, 0.03) * (1 + 6 * sstep(0.2, 0.0, (X - Y * 0.6 - 0.1).abs()))
        mull = torch.maximum(lines(X - 0.5, 1, 0.008), lines(Y - 0.435, 1, 0.006))
        rgb = lerp(rgb, col(c, 0.5, 0.5, 0.5), fr)
        rgb = lerp(rgb, lerp(glass, col(c, 0.5, 0.5, 0.5), mull), m)
    return out(c, rgb)


def cubicle(c, base=(0.36, 0.42, 0.50)):
    X, Y = c.X, c.Y
    weave = 0.5 + 0.25 * torch.cos(2 * PI * X * 120) + 0.25 * torch.cos(2 * PI * Y * 120)
    rgb = col(c, *base) * (0.85 + 0.15 * weave) * (0.95 + 0.1 * c.n01(2.5, 2, 20))
    trim = torch.maximum(lines(X, 1, 0.02), torch.maximum(sstep(0.04, 0.03, Y), sstep(0.95, 0.96, Y)))
    return out(c, lerp(rgb, col(c, 0.55, 0.56, 0.58), trim))


def wood(c, base=(0.30, 0.17, 0.09), planks=0, panel=False, gloss=0.0):
    X, Y = c.X, c.Y
    w = c.fbm(2.0, 1, 12, sy=8.0)
    ringp = 0.5 + 0.5 * torch.cos(2 * PI * (X * 14 + 1.6 * w))
    rgb = col(c, *base) * (0.75 + 0.3 * ringp + 0.06 * c.fbm(0.5, 30, sy=6.0))
    if planks:
        px = torch.floor(X * planks)
        rgb = rgb * (0.85 + 0.3 * c.cellrand(px, px * 0, 4)[None]) * (1 - 0.6 * lines(X, planks, 0.003))
        rgb = rgb * (1 - 0.4 * lines(Y + 0.37 * (px % 3), 1, 0.003))
    if panel:       # raised panels
        fx, fy = (X * 2) % 1, Y
        inner = band(fx, 0.14, 0.86, 0.006) * band(fy, 0.12, 0.88, 0.006)
        edge = band(fx, 0.10, 0.90, 0.006) * band(fy, 0.08, 0.92, 0.006) - inner
        rgb = rgb * (1 - 0.35 * edge.clamp(0, 1)) * (1 + 0.12 * inner * (1 - fy))
    return out(c, rgb)


def hotel_wall(c, paper=(0.36, 0.08, 0.08), motif=(0.55, 0.40, 0.16), wood_c=(0.22, 0.12, 0.07), sconce=False):
    X, Y = c.X, c.Y
    # damask-like motif
    fx, fy = (X * 8) % 1 - 0.5, (Y * 10 + 0.5 * torch.floor(X * 8)) % 1 - 0.5
    r = torch.sqrt(fx * fx + fy * fy)
    th = torch.atan2(fy, fx)
    mot = (band(r, 0.0, 0.16 + 0.10 * torch.cos(4 * th), 0.02) * 0.6 + band(r, 0.30, 0.34, 0.01) * 0.4)
    up = col(c, *paper) * (0.9 + 0.08 * c.fbm(0.8, 10))
    up = lerp(up, col(c, *motif), mot * 0.55)
    up = lerp(up, up * 0.6, drips(c, 30, 0.5) * 0.5)
    yy = (Y - 0.64) / 0.36
    fxw = (X * 2) % 1
    inner = band(fxw, 0.14, 0.86, 0.006) * band(yy, 0.14, 0.84, 0.01)
    edge = band(fxw, 0.10, 0.90, 0.006) * band(yy, 0.09, 0.89, 0.01) - inner
    wd2 = wood(c, wood_c)[:3] * (1 - 0.4 * edge.clamp(0, 1)) * (1 + 0.1 * inner)
    lower = sstep(0.638, 0.642, Y)
    rgb = lerp(up, wd2, lower)
    rgb = lerp(rgb, col(c, *wood_c) * 1.5, band(Y, 0.625, 0.645, 0.003))            # chair rail
    rgb = lerp(rgb, col(c, *wood_c) * 1.3, sstep(0.045, 0.04, Y))                   # crown
    rgb = lerp(rgb, col(c, *wood_c) * 0.8, sstep(0.955, 0.96, Y))
    em = None
    if sconce:
        d = torch.sqrt(((X - 0.5) / 0.045) ** 2 + ((Y - 0.34) / 0.06) ** 2)
        m = 1 - sstep(0.8, 1.0, d)
        halo = (1 - sstep(0.8, 6.0, d))
        rgb = rgb + col(c, 1.0, 0.62, 0.25) * halo * halo * 0.55
        rgb = lerp(rgb, col(c, 0.5, 0.38, 0.12), box(c, 0.485, 0.39, 0.515, 0.47, 0.003))
        rgb = lerp(rgb, col(c, 1.0, 0.86, 0.6), m)
        em = m
    return out(c, rgb, em)


def door(c, wall, leaf=(0.25, 0.14, 0.08), frame=(0.16, 0.09, 0.05), style=0, top=0.18):
    """One cell (1 m) of `wall` with a door in it."""
    X, Y = c.X, c.Y
    rgb = wall[:3].clone()
    fr = box(c, 0.07, top - 0.02, 0.93, 1.0, 0.003)
    lf = box(c, 0.11, top + 0.01, 0.89, 0.995, 0.003)
    grain = 0.85 + 0.2 * (0.5 + 0.5 * torch.cos(2 * PI * (X * 22 + 1.5 * c.fbm(2.0, 1, 10, sy=8.0))))
    l = col(c, *leaf) * grain
    if style == 0:      # panelled wood
        for (y0, y1) in ((top + 0.06, top + 0.32), (top + 0.38, 0.93)):
            for (x0, x1) in ((0.19, 0.47), (0.53, 0.81)):
                inner = box(c, x0, y0, x1, y1, 0.004)
                rim = box(c, x0 - 0.015, y0 - 0.012, x1 + 0.015, y1 + 0.012, 0.004) - inner
                l = l * (1 - 0.4 * rim.clamp(0, 1)) * (1 + 0.1 * inner)
        l = lerp(l, col(c, 0.75, 0.6, 0.25), box(c, 0.44, top + 0.05, 0.56, top + 0.085, 0.002))   # number plate
    elif style == 1:    # flat office door with a narrow window
        l = col(c, *leaf) * (0.95 + 0.05 * c.fbm(1.0, 8))
        win = box(c, 0.62, top + 0.10, 0.76, top + 0.42, 0.003)
        l = lerp(l, col(c, 0.02, 0.025, 0.03), win)
    else:               # hospital double door with portholes
        l = col(c, *leaf) * (0.95 + 0.05 * c.fbm(1.0, 8)) * (1 - 0.6 * lines(X - 0.5, 1, 0.004))
        for xc in (0.31, 0.69):
            d = torch.sqrt((X - xc) ** 2 + ((Y - top - 0.16) * 2.6) ** 2)
            l = lerp(l, col(c, 0.02, 0.02, 0.025), 1 - sstep(0.105, 0.11, d))
            l = lerp(l, col(c, 0.6, 0.6, 0.6), band(d, 0.11, 0.13, 0.004))
        l = lerp(l, col(c, 0.55, 0.55, 0.55), box(c, 0.13, 0.90, 0.87, 0.985, 0.003))   # kick plate
    hx = 0.82 if style != 2 else 0.46
    hd = torch.sqrt((X - hx) ** 2 + ((Y - 0.60) * 2.6) ** 2)
    l = lerp(l, col(c, 0.8, 0.7, 0.4), 1 - sstep(0.024, 0.03, hd))
    rgb = lerp(rgb, col(c, *frame), fr)
    rgb = lerp(rgb, l, lf)
    rgb = lerp(rgb, col(c, 0.0, 0.0, 0.0), band(Y, 0.992, 1.0, 0.002) * lf * 0.8)   # gap under the door
    return out(c, rgb)


def tiles(c, base=(0.86, 0.90, 0.90), grout=(0.62, 0.66, 0.66), n=8, band_c=None, grime=0.3, gloss=True):
    X, Y = c.X, c.Y
    ix, iy = torch.floor(X * n), torch.floor(Y * n)
    fx, fy = X * n - ix, Y * n - iy
    tone = 0.95 + 0.07 * c.cellrand(ix, iy, 5)[None]
    b = col(c, *base)
    if band_c is not None:
        row = (iy == 2) | (iy == 5 if n >= 8 else iy == 2)
        b = lerp(b, col(c, *band_c), row.float()[None])
    rgb = b * tone * (1 + 0.06 * (0.5 - fy) * (1 if gloss else 0)) * (1 + 0.015 * c.fbm(0.5, 40))
    g = torch.maximum(lines(X, n, 0.0035), lines(Y, n, 0.0035))
    gr = col(c, *grout) * (1 - grime * sstep(0.4, 0.9, c.n01(2.5, 2, 30)))
    rgb = lerp(rgb, gr, g)
    rgb = lerp(rgb, rgb * col(c, 0.8, 0.85, 0.75), sstep(0.6, 0.95, c.n01(3.0, 1, 6)) * grime)
    return out(c, rgb)


def lino(c, a=(0.62, 0.66, 0.60), b=(0.56, 0.60, 0.55), n=4):
    X, Y = c.X, c.Y
    chk = ((torch.floor(X * n) + torch.floor(Y * n)) % 2)[None]
    rgb = lerp(col(c, *a), col(c, *b), chk) * (0.94 + 0.05 * c.fbm(0.3, 50) + 0.06 * (c.n01(2.5, 2, 20) - 0.5))
    scuff = sstep(0.75, 0.95, c.n01(2.0, 3, 60, sy=6.0))
    rgb = lerp(rgb, rgb * 0.7, scuff * 0.5)
    return out(c, rgb * (1 - 0.2 * torch.maximum(lines(X, n, 0.002), lines(Y, n, 0.002))))


def party_wall(c, style=0, face=False):
    X, Y = c.X, c.Y
    pal = [(0.90, 0.16, 0.20), (0.98, 0.82, 0.15), (0.15, 0.45, 0.90), (0.20, 0.75, 0.35), (0.95, 0.45, 0.75)]
    if style == 0:      # broad stripes
        i = torch.floor(X * 10) % 5
        rgb = sum(col(c, *pal[k]) * (i == k).float()[None] for k in range(5))
    elif style == 1:    # polka dots on yellow
        fx, fy = (X * 8) % 1 - 0.5, (Y * 10 + 0.5 * torch.floor(X * 8)) % 1 - 0.5
        ci = (torch.floor(X * 8) + torch.floor(Y * 10 + 0.5 * torch.floor(X * 8))) % 5
        d = torch.sqrt(fx * fx + fy * fy)
        dots = sum(col(c, *pal[k]) * (ci == k).float()[None] for k in range(5))
        rgb = lerp(col(c, 0.98, 0.90, 0.55), dots, (1 - sstep(0.26, 0.30, d)))
    else:               # confetti on pale blue
        n = c.fbm(0.0, 40, 90)
        m = sstep(1.6, 2.0, n)
        ci = torch.floor(c.n01(0.0, 20, 60) * 5).clamp(0, 4)
        conf = sum(col(c, *pal[k]) * (ci == k).float()[None] for k in range(5))
        rgb = lerp(col(c, 0.70, 0.86, 0.96), conf, m)
    rgb = rgb * (0.95 + 0.04 * c.fbm(1.0, 8))
    # bunting along the top
    t = (X * 8) % 1
    tri = ((Y - 0.03) < 0.09 * (1 - (2 * t - 1).abs())) & (Y > 0.03)
    bi = torch.floor(X * 8) % 5
    bun = sum(col(c, *pal[(k + 2) % 5]) * (bi == k).float()[None] for k in range(5))
    rgb = lerp(rgb, bun, tri.float()[None])
    if face:            # =)
        m = torch.zeros_like(X)
        for xc in (0.40, 0.60):
            m = torch.maximum(m, 1 - sstep(0.9, 1.0, torch.sqrt(((X - xc) / 0.045) ** 2 + ((Y - 0.40) / 0.03) ** 2)))
        d = torch.sqrt((X - 0.5) ** 2 + ((Y - 0.42) * 2.6) ** 2)
        m = torch.maximum(m, band(d, 0.22, 0.27, 0.005) * (Y > 0.45).float())
        rgb = lerp(rgb, col(c, 0.02, 0.02, 0.02), m)
    rgb = lerp(rgb, col(c, 0.95, 0.95, 0.9), sstep(0.955, 0.96, Y))
    rgb = lerp(rgb, rgb * 0.75, drips(c, 30, 0.5) * 0.35)
    return out(c, rgb)


def checker(c, a=(0.88, 0.88, 0.84), b=(0.10, 0.10, 0.11), n=4):
    X, Y = c.X, c.Y
    chk = ((torch.floor(X * n) + torch.floor(Y * n)) % 2)[None]
    rgb = lerp(col(c, *a), col(c, *b), chk) * (0.95 + 0.04 * c.fbm(0.5, 30))
    rgb = lerp(rgb, rgb * 0.75, sstep(0.7, 0.95, c.n01(2.5, 2, 30)) * 0.5)
    return out(c, rgb)


def bookshelf(c, wood_c=(0.20, 0.11, 0.06), shelves=6):
    X, Y = c.X, c.Y
    sy = Y * shelves
    row = torch.floor(sy)
    fy = sy - row
    # books of uneven width: two interleaved widths per row
    n1 = 22 + 6 * (row % 3)
    bx = torch.floor(X * n1 + 0.37 * c.cellrand(torch.floor(X * 9), row, 1) * 2)
    fx = (X * n1 + 0.0) % 1
    h = 0.45 + 0.5 * c.cellrand(bx, row, 2)
    hue = c.cellrand(bx, row, 3)
    val = 0.35 + 0.6 * c.cellrand(bx, row, 4)
    pal = [(0.45, 0.10, 0.08), (0.10, 0.25, 0.14), (0.10, 0.14, 0.32), (0.40, 0.30, 0.14), (0.25, 0.22, 0.20), (0.50, 0.42, 0.30)]
    hi = torch.floor(hue * 6).clamp(0, 5)
    bk = sum(col(c, *pal[k]) * (hi == k).float()[None] for k in range(6)) * val[None]
    bk = bk * (0.6 + 0.4 * (1 - (2 * fx - 1) ** 2)) * (1 + 0.5 * band(fy, 0.45, 0.50, 0.01) * (hue > 0.5).float())
    inbook = (fy > (1 - h * 0.86) - 0.0).float() * (fy < 0.90).float() * band(fx, 0.04, 0.96, 0.03)
    gap = (c.cellrand(bx, row, 5) > 0.9).float()
    back = col(c, *wood_c) * 0.25
    rgb = lerp(back, bk, inbook * (1 - gap))
    board = (fy > 0.90).float()
    rgb = lerp(rgb, wood(c, wood_c)[:3] * 1.3, board)
    rgb = lerp(rgb, wood(c, wood_c)[:3], lines(X, 1, 0.02))
    return out(c, rgb)


def chainlink(c):
    X, Y = c.X, c.Y
    a = lines(X * 1.0 + Y * 1.4, 40, 0.0035)
    b = lines(X * 1.0 - Y * 1.4, 40, 0.0035)
    wire = torch.maximum(a, b)
    rgb = lerp(col(c, 0.012, 0.012, 0.014), col(c, 0.42, 0.44, 0.45), wire)
    post = torch.maximum(lines(X, 1, 0.02), torch.maximum(sstep(0.03, 0.02, Y), band(Y, 0.49, 0.51, 0.003)))
    rgb = lerp(rgb, col(c, 0.30, 0.31, 0.32) * (0.8 + 0.2 * c.n01(2.0, 2, 30)), post)
    rgb = lerp(rgb, rgb * col(c, 1.2, 0.7, 0.4), sstep(0.6, 0.9, c.n01(2.5, 2, 20)) * 0.5)
    return out(c, rgb)


def elec_panel(c, wall):
    X, Y = c.X, c.Y
    rgb = wall[:3].clone()
    cab = box(c, 0.10, 0.22, 0.90, 0.88, 0.004)
    body = col(c, 0.42, 0.45, 0.44) * (0.9 + 0.1 * c.n01(2.0, 2, 30))
    body = body * (1 - 0.5 * lines(X - 0.5, 1, 0.004))
    body = body * (1 - 0.3 * box(c, 0.14, 0.26, 0.47, 0.84, 0.003) * (1 - box(c, 0.15, 0.27, 0.46, 0.83, 0.003)))
    body = body * (1 - 0.3 * box(c, 0.53, 0.26, 0.86, 0.84, 0.003) * (1 - box(c, 0.54, 0.27, 0.85, 0.83, 0.003)))
    warn = box(c, 0.24, 0.34, 0.37, 0.43, 0.003)
    stripes = (torch.floor((X + Y) * 60) % 2)[None]
    body = lerp(body, lerp(col(c, 0.9, 0.75, 0.05), col(c, 0.05, 0.05, 0.05), stripes), warn)
    em = torch.zeros_like(X)
    for (x, y, cc) in ((0.60, 0.33, (1.0, 0.1, 0.05)), (0.66, 0.33, (0.1, 1.0, 0.2)), (0.72, 0.33, (1.0, 0.7, 0.1))):
        d = torch.sqrt((X - x) ** 2 + ((Y - y) * 2.8) ** 2)
        m = 1 - sstep(0.012, 0.016, d)
        body = lerp(body, col(c, *cc), m)
        em = torch.maximum(em, m)
    rgb = lerp(rgb, body, cab)
    cable = band(X, 0.48, 0.52, 0.004) * (Y < 0.22).float()
    rgb = lerp(rgb, col(c, 0.05, 0.05, 0.05), cable)
    return out(c, rgb, em * cab)


def sign_wall(c, wall, kind=0):
    """A wall cell with something glowing on it: 0 exit sign, 1 red alarm lamp, 2 yellow warning beacon."""
    X, Y = c.X, c.Y
    rgb = wall[:3].clone()
    if kind == 0:
        m = box(c, 0.28, 0.11, 0.72, 0.19, 0.003)
        txt = (band((X * 12) % 1, 0.25, 0.75, 0.05) * band(Y, 0.13, 0.17, 0.003) * band(X, 0.34, 0.66, 0.01))
        g = lerp(col(c, 0.05, 0.75, 0.30), col(c, 0.95, 1.0, 0.95), txt)
        d = torch.sqrt(((X - 0.5) / 0.4) ** 2 + ((Y - 0.15) / 0.2) ** 2)
        rgb = rgb + col(c, 0.1, 0.9, 0.4) * (1 - sstep(0.3, 1.2, d)) * 0.25
        return out(c, lerp(rgb, g, m), m)
    cc = (1.0, 0.08, 0.05) if kind == 1 else (1.0, 0.65, 0.05)
    d = torch.sqrt(((X - 0.5) / 0.075) ** 2 + ((Y - 0.16) / 0.028) ** 2)
    m = 1 - sstep(0.85, 1.0, d)
    rgb = rgb + col(c, *cc) * (1 - sstep(0.8, 5.0, d)) ** 2 * 0.6
    rgb = lerp(rgb, col(c, 0.25, 0.25, 0.25), box(c, 0.45, 0.19, 0.55, 0.205, 0.003))
    return out(c, lerp(rgb, col(c, *cc) * 0.6 + 0.4, m), m)


def crate(c, base=(0.46, 0.33, 0.18)):
    X, Y = c.X, c.Y
    rgb = wood(c, base, planks=5)[:3]
    fr = 1 - box(c, 0.07, 0.07, 0.93, 0.93, 0.004)
    diag = lines((X - Y) * 0.7071, 1, 0.04) * (1 - fr)
    rgb = lerp(rgb, wood(c, (base[0] * 0.8, base[1] * 0.8, base[2] * 0.8), planks=0)[:3], torch.maximum(fr, diag))
    rgb = rgb * (1 - 0.4 * (band(X, 0.065, 0.075, 0.002) + band(X, 0.925, 0.935, 0.002) + band(Y, 0.065, 0.075, 0.002)
                              + band(Y, 0.925, 0.935, 0.002)).clamp(0, 1))
    return out(c, rgb)


def hazard(c):
    X, Y = c.X, c.Y
    conc = concrete(c, 0.50, panels=1)[:3]
    st = (torch.floor((X * 2 + Y) * 6) % 2)[None]
    hz = lerp(col(c, 0.85, 0.70, 0.05), col(c, 0.06, 0.06, 0.06), st) * (0.8 + 0.2 * c.n01(2.0, 2, 30))
    return out(c, lerp(conc, hz, band(Y, 0.15, 0.55, 0.004)))


def grate(c):
    X, Y = c.X, c.Y
    bars = torch.maximum(lines(X, 16, 0.012), lines(Y, 4, 0.012))
    glow = col(c, 1.0, 0.35, 0.08) * (0.5 + 0.5 * c.n01(2.5, 1, 6))
    metal = col(c, 0.16, 0.15, 0.14) * (0.8 + 0.3 * c.n01(2.0, 2, 30))
    return out(c, lerp(glow * 0.6, metal, bars), (1 - bars) * 0.8)


def metal_plate(c, base=(0.26, 0.26, 0.27)):
    X, Y = c.X, c.Y
    fx, fy = (X * 16) % 1 - 0.5, (Y * 16 + 0.5 * torch.floor(X * 16)) % 1 - 0.5
    tread = (1 - sstep(0.10, 0.14, (fx.abs() * 2.5 + fy.abs() * 0.7)))
    rgb = col(c, *base) * (0.8 + 0.3 * c.n01(2.2, 1, 30)) * (1 + 0.5 * tread)
    rgb = lerp(rgb, col(c, 0.35, 0.18, 0.08), sstep(0.6, 0.9, c.n01(2.5, 2, 20)) * 0.6)
    return out(c, rgb * (1 - 0.4 * torch.maximum(lines(X, 2, 0.004), lines(Y, 2, 0.004))))


def cloth(c, style=0):
    """Table cloth for the party rooms."""
    X, Y = c.X, c.Y
    chk = ((torch.floor(X * 12) + torch.floor(Y * 12)) % 2)[None]
    rgb = lerp(col(c, 0.92, 0.92, 0.9), col(c, 0.85, 0.15, 0.2) if style == 0 else col(c, 0.2, 0.4, 0.85), chk)
    return out(c, rgb * (0.9 + 0.1 * c.n01(2.5, 2, 12)))


def shelf_top(c):
    return wood(c, (0.20, 0.11, 0.06), planks=3)


_FONT = {
    "A": "01110 10001 10001 11111 10001 10001 10001", "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "11111 00100 00100 00100 00100 00100 11111", "K": "10001 10010 10100 11000 10100 10010 10001",
    "L": "10000 10000 10000 10000 10000 10000 11111", "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "R": "11110 10001 10001 11110 10100 10010 10001", "S": "01111 10000 10000 01110 00001 00001 11110",
    "T": "11111 00100 00100 00100 00100 00100 00100", "U": "10001 10001 10001 10001 10001 10001 01110",
    "X": "10001 10001 01010 00100 01010 10001 10001", "Y": "10001 10001 01010 00100 00100 00100 00100",
    "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 11011 10001", " ": "00000 00000 00000 00000 00000 00000 00000",
    "=": "00000 00000 11111 00000 11111 00000 00000", ")": "01000 00100 00010 00010 00010 00100 01000",
}


def scrawl(c, wall, text, colr=(0.20, 0.02, 0.02), y0=0.30):
    """Words painted by hand on a wall cell of 1 m x about 2.7 m. `text`: lines separated by /."""
    X, Y = c.X, c.Y
    rows = text.split("/")
    ncol = max(len(r) for r in rows) * 6 - 1
    bm = torch.zeros(len(rows) * 9, ncol)
    for j, r in enumerate(rows):
        x0 = (ncol - (len(r) * 6 - 1)) // 2
        for i, ch in enumerate(r):
            g = torch.tensor([[int(v) for v in row] for row in _FONT[ch].split()], dtype=torch.float32)
            bm[j * 9:j * 9 + 7, x0 + i * 6:x0 + i * 6 + 5] = g
    bm = bm.to(c.dev)
    pw = 0.84 / ncol
    wob = 0.012 * c.fbm(2.0, 1, 6)
    cx = torch.floor((X + wob - 0.08) / pw).long()
    cy = torch.floor((Y + 0.4 * wob - y0) / (pw / 2.6)).long()
    ok = (cx >= 0) & (cx < ncol) & (cy >= 0) & (cy < bm.shape[0])
    m = bm[cy.clamp(0, bm.shape[0] - 1), cx.clamp(0, ncol - 1)] * ok.float()
    m = F.avg_pool2d(m[None, None], 9, 1, 4)[0, 0]
    m = sstep(0.30, 0.55, m + 0.35 * (c.n01(1.0, 20, 120) - 0.5))
    run = F.max_pool2d(F.pad(m[None, None], (0, 0, 60, 0)), (61, 1), 1)[0, 0] * sstep(0.6, 0.9, c.n01(1.5, 20, 80, sy=1e3)) * 0.5   # paint that ran
    rgb = lerp(wall[:3], col(c, *colr), torch.maximum(m, run * (Y > y0).float()) * 0.92)
    return out(c, rgb, wall[3])


# ---------------------------------------------------------------- sprites (alpha = opacity)

def _cap(c, x0, y0, x1, y1, r0, r1=None):
    """Signed distance to a capsule from (x0, y0) to (x1, y1), radius r0 at the start and r1 at the end."""
    r1 = r0 if r1 is None else r1
    X, Y = c.X, c.Y
    dx, dy = x1 - x0, y1 - y0
    t = (((X - x0) * dx + (Y - y0) * dy) / (dx * dx + dy * dy + 1e-9)).clamp(0, 1)
    return torch.sqrt((X - x0 - t * dx) ** 2 + (Y - y0 - t * dy) ** 2) - (r0 + (r1 - r0) * t)


def _ell(c, cx, cy, rx, ry):
    return (torch.sqrt(((c.X - cx) / rx) ** 2 + ((c.Y - cy) / ry) ** 2) - 1) * min(rx, ry)


def _cov(d, w=0.004):
    return (0.5 - d / w).clamp(0, 1)


def _sprite(c, rgb, a):
    a = a.clamp(0, 1)
    return torch.cat([(rgb.clamp(0, 1) * a.sqrt()).expand(3, TS, TS), a.view(1, TS, TS)], 0)


def _union(*ds):
    d = ds[0]
    for e in ds[1:]:
        d = torch.minimum(d, e)
    return d


def s_figure(c, eyes=True, tone=0.012, lean=0.0, arms=1.0, eye_col=(0.95, 0.95, 0.9)):
    """A tall thin figure standing still. Canvas: 1.0 m wide, 2.3 m high."""
    hx = 0.5 + lean * 0.1
    d = _union(
        _ell(c, hx, 0.085, 0.085, 0.060),
        _cap(c, hx, 0.14, 0.5, 0.20, 0.03, 0.045),
        _cap(c, 0.5, 0.20, 0.5, 0.52, 0.115, 0.075),
        _cap(c, 0.385, 0.20, 0.33 - 0.05 * arms, 0.46, 0.035, 0.026), _cap(c, 0.33 - 0.05 * arms, 0.46, 0.34 - 0.07 * arms, 0.72, 0.026, 0.018),
        _cap(c, 0.615, 0.20, 0.67 + 0.05 * arms, 0.46, 0.035, 0.026), _cap(c, 0.67 + 0.05 * arms, 0.46, 0.66 + 0.07 * arms, 0.72, 0.026, 0.018),
        _cap(c, 0.455, 0.52, 0.44, 0.76, 0.050, 0.036), _cap(c, 0.44, 0.76, 0.43, 0.985, 0.036, 0.026),
        _cap(c, 0.545, 0.52, 0.56, 0.76, 0.050, 0.036), _cap(c, 0.56, 0.76, 0.57, 0.985, 0.036, 0.026),
    )
    a = _cov(d, 0.006)
    rim = _cov(d.abs() - 0.004, 0.006) * 0.03
    rgb = col(c, tone, tone, tone * 1.1) + rim
    if eyes:
        for ex in (hx - 0.032, hx + 0.032):
            e = _cov(_ell(c, ex, 0.080, 0.011, 0.006), 0.003)
            rgb = lerp(rgb, col(c, *eye_col), e)
    return _sprite(c, rgb, a)


def s_smiler(c):
    """Eyes and a grin, nothing else. Canvas 0.9 x 0.9 m."""
    X, Y = c.X, c.Y
    a = torch.zeros_like(X)
    for ex in (0.30, 0.70):
        e = _ell(c, ex, 0.34, 0.085, 0.060)
        pupil = _ell(c, ex + 0.01 * (1 if ex < 0.5 else -1), 0.35, 0.018, 0.030)
        a = torch.maximum(a, _cov(e, 0.006) * (1 - _cov(pupil, 0.004)))
    d = torch.sqrt((X - 0.5) ** 2 + ((Y - 0.30) * 0.8) ** 2)
    mouth = band(d, 0.30, 0.40, 0.006) * sstep(0.50, 0.56, Y)
    teeth = 1 - 0.85 * lines(X, 18, 0.006) - 0.6 * band(d, 0.345, 0.355, 0.003)
    a = torch.maximum(a, mouth * teeth.clamp(0, 1))
    halo = (1 - sstep(0.0, 0.5, torch.sqrt((X - 0.5) ** 2 + (Y - 0.45) ** 2))) * 0.10
    rgb = col(c, 1.0, 1.0, 0.92).expand(3, TS, TS) * 1.0
    al = torch.maximum(a, halo * 0.0)
    return _sprite(c, rgb, al)


def s_hound(c):
    """Something on all fours with a pale face. Canvas 1.6 x 1.1 m."""
    d = _union(
        _cap(c, 0.30, 0.42, 0.68, 0.36, 0.14, 0.11),                      # trunk
        _cap(c, 0.68, 0.36, 0.80, 0.30, 0.09, 0.07),                      # neck
        _cap(c, 0.33, 0.50, 0.25, 0.74, 0.055, 0.035), _cap(c, 0.25, 0.74, 0.30, 0.985, 0.035, 0.025),
        _cap(c, 0.40, 0.52, 0.42, 0.76, 0.050, 0.032), _cap(c, 0.42, 0.76, 0.38, 0.985, 0.032, 0.022),
        _cap(c, 0.66, 0.46, 0.72, 0.72, 0.045, 0.03), _cap(c, 0.72, 0.72, 0.68, 0.985, 0.03, 0.022),
        _cap(c, 0.72, 0.44, 0.82, 0.70, 0.042, 0.028), _cap(c, 0.82, 0.70, 0.80, 0.985, 0.028, 0.02),
    )
    hair = _ell(c, 0.84, 0.25, 0.12, 0.13)
    a = _cov(torch.minimum(d, hair), 0.006)
    rgb = col(c, 0.02, 0.018, 0.016) + 0.03 * c.n01(1.0, 20)[None] * 0
    face = _cov(_ell(c, 0.87, 0.28, 0.062, 0.085), 0.006)
    rgb = lerp(rgb, col(c, 0.70, 0.66, 0.58), face)
    for ex in (0.845, 0.895):
        rgb = lerp(rgb, col(c, 0, 0, 0), _cov(_ell(c, ex, 0.26, 0.014, 0.018), 0.003))
    rgb = lerp(rgb, col(c, 0, 0, 0), _cov(_ell(c, 0.872, 0.325, 0.03, 0.022), 0.003))
    return _sprite(c, rgb, a)


def s_party(c):
    """A partygoer: yellow, smiling, holding a balloon. Canvas 1.1 x 2.3 m."""
    X, Y = c.X, c.Y
    body = _union(
        _ell(c, 0.42, 0.19, 0.105, 0.075),
        _cap(c, 0.42, 0.27, 0.42, 0.56, 0.12, 0.10),
        _cap(c, 0.31, 0.30, 0.24, 0.52, 0.035, 0.028), _cap(c, 0.24, 0.52, 0.25, 0.70, 0.028, 0.02),
        _cap(c, 0.53, 0.30, 0.64, 0.42, 0.035, 0.028), _cap(c, 0.64, 0.42, 0.72, 0.30, 0.028, 0.02),
        _cap(c, 0.37, 0.56, 0.36, 0.985, 0.05, 0.03), _cap(c, 0.47, 0.56, 0.48, 0.985, 0.05, 0.03),
    )
    a = _cov(body, 0.006)
    rgb = col(c, 0.92, 0.80, 0.25) * (0.85 + 0.15 * (1 - ((X - 0.42) / 0.14) ** 2).clamp(0, 1))
    for ex in (0.385, 0.455):
        rgb = lerp(rgb, col(c, 0, 0, 0), _cov(_ell(c, ex, 0.175, 0.012, 0.020), 0.003))
    d = torch.sqrt((X - 0.42) ** 2 + ((Y - 0.175) * 1.0) ** 2)
    rgb = lerp(rgb, col(c, 0, 0, 0), band(d, 0.050, 0.060, 0.003) * (Y > 0.195).float())
    hat = (((Y - 0.03) > 0) & ((X - 0.42).abs() < (Y - 0.03) * 0.6) & (Y < 0.125)).float()
    rgb = lerp(rgb, col(c, 0.85, 0.15, 0.6), hat)
    a = torch.maximum(a, hat)
    bal = _cov(_ell(c, 0.76, 0.10, 0.085, 0.060), 0.006)
    rgb = lerp(rgb, col(c, 0.90, 0.10, 0.12) * (0.7 + 0.5 * (1 - sstep(0.0, 0.1, torch.sqrt((X - 0.73) ** 2 + (Y - 0.08) ** 2)))), bal)
    a = torch.maximum(a, bal)
    string = _cov(_cap(c, 0.76, 0.16, 0.72, 0.30, 0.004), 0.003)
    rgb = lerp(rgb, col(c, 0.8, 0.8, 0.8), string * (1 - bal))
    a = torch.maximum(a, string)
    return _sprite(c, rgb, a)


def s_faceless(c):
    """A person in grey with no face. Canvas 1.0 x 1.9 m."""
    d_body = _union(
        _cap(c, 0.5, 0.22, 0.5, 0.52, 0.14, 0.11),
        _cap(c, 0.365, 0.22, 0.32, 0.50, 0.045, 0.035), _cap(c, 0.32, 0.50, 0.33, 0.62, 0.035, 0.028),
        _cap(c, 0.635, 0.22, 0.68, 0.50, 0.045, 0.035), _cap(c, 0.68, 0.50, 0.67, 0.62, 0.035, 0.028),
        _cap(c, 0.445, 0.52, 0.44, 0.985, 0.065, 0.045), _cap(c, 0.555, 0.52, 0.56, 0.985, 0.065, 0.045),
    )
    d_head = _union(_ell(c, 0.5, 0.095, 0.075, 0.085), _cap(c, 0.5, 0.15, 0.5, 0.21, 0.04))
    a = _cov(torch.minimum(d_body, d_head), 0.006)
    rgb = lerp(col(c, 0.16, 0.16, 0.18), col(c, 0.07, 0.07, 0.09), sstep(0.5, 0.56, c.Y))
    rgb = rgb * (0.8 + 0.2 * c.n01(2.0, 3, 30))
    skin = _cov(d_head, 0.006)
    rgb = lerp(rgb, col(c, 0.62, 0.55, 0.50) * (0.75 + 0.25 * (1 - ((c.X - 0.5) / 0.08) ** 2).clamp(0, 1)), skin)
    hands = _cov(_union(_ell(c, 0.33, 0.635, 0.03, 0.035), _ell(c, 0.67, 0.635, 0.03, 0.035)), 0.005)
    rgb = lerp(rgb, col(c, 0.55, 0.48, 0.44), hands)
    return _sprite(c, rgb, torch.maximum(a, hands))


def s_horde(c):
    """A crowd of dark shapes with red eyes. Canvas 2.6 x 2.2 m."""
    X, Y = c.X, c.Y
    a = torch.zeros_like(X)
    eyes = torch.zeros_like(X)
    for (cx, top, w) in ((0.14, 0.30, 0.9), (0.33, 0.12, 1.1), (0.50, 0.24, 1.0), (0.68, 0.05, 1.15), (0.86, 0.28, 0.9), (0.24, 0.42, 0.8), (0.60, 0.40, 0.85)):
        d = _union(
            _ell(c, cx, top + 0.05, 0.040 * w, 0.045 * w),
            _cap(c, cx, top + 0.09, cx, top + 0.42, 0.065 * w, 0.05 * w),
            _cap(c, cx - 0.06 * w, top + 0.12, cx - 0.11 * w, top + 0.02 + 0.2 * (cx % 0.2), 0.02, 0.012),
            _cap(c, cx + 0.06 * w, top + 0.12, cx + 0.12 * w, top + 0.30, 0.02, 0.012),
            _cap(c, cx - 0.025, top + 0.42, cx - 0.03, 0.985, 0.03, 0.02),
            _cap(c, cx + 0.025, top + 0.42, cx + 0.035, 0.985, 0.03, 0.02),
        )
        a = torch.maximum(a, _cov(d, 0.006))
        for ex in (cx - 0.014 * w, cx + 0.014 * w):
            eyes = torch.maximum(eyes, _cov(_ell(c, ex, top + 0.045, 0.006, 0.005), 0.002))
    rgb = lerp(col(c, 0.012, 0.010, 0.010).expand(3, TS, TS), col(c, 1.0, 0.12, 0.08), eyes)
    return _sprite(c, rgb, a)


def s_balloon(c, cc=(0.9, 0.1, 0.12)):
    """Canvas 0.5 x 1.9 m: the balloon floats at the top, the string hangs to the floor."""
    X, Y = c.X, c.Y
    b = _cov(_ell(c, 0.5, 0.11, 0.36, 0.10), 0.008)
    knot = _cov(_cap(c, 0.5, 0.21, 0.5, 0.225, 0.03, 0.05), 0.006)
    string = _cov(_cap(c, 0.5, 0.22, 0.5 + 0.0, 0.99, 0.012), 0.008) * (0.7 + 0.3 * torch.cos(2 * PI * Y * 3))
    hl = 1 - sstep(0.0, 0.16, torch.sqrt(((X - 0.38) / 1.0) ** 2 + ((Y - 0.07) * 3.5) ** 2))
    shade = 0.55 + 0.45 * (1 - (((X - 0.42) / 0.4) ** 2 + ((Y - 0.09) / 0.12) ** 2).clamp(0, 1))
    rgb = col(c, *cc) * shade + hl * 0.6
    rgb = lerp(col(c, 0.85, 0.85, 0.85).expand(3, TS, TS), rgb, torch.maximum(b, knot))
    return _sprite(c, rgb, torch.maximum(torch.maximum(b, knot), string * 0.9))


def s_chair(c):
    """Office chair, canvas 0.7 x 1.1 m."""
    d = _union(
        _cap(c, 0.25, 0.08, 0.75, 0.08, 0.05), _cap(c, 0.25, 0.08, 0.25, 0.42, 0.05), _cap(c, 0.75, 0.08, 0.75, 0.42, 0.05),
        _cap(c, 0.25, 0.42, 0.75, 0.42, 0.05), _ell(c, 0.5, 0.25, 0.25, 0.17),
        _cap(c, 0.5, 0.46, 0.5, 0.56, 0.035),
        _cap(c, 0.17, 0.60, 0.83, 0.60, 0.055),
        _cap(c, 0.5, 0.64, 0.5, 0.86, 0.03),
        _cap(c, 0.5, 0.86, 0.14, 0.95, 0.025), _cap(c, 0.5, 0.86, 0.86, 0.95, 0.025), _cap(c, 0.5, 0.86, 0.5, 0.96, 0.025),
        _ell(c, 0.14, 0.965, 0.045, 0.03), _ell(c, 0.86, 0.965, 0.045, 0.03), _ell(c, 0.5, 0.972, 0.045, 0.025),
    )
    a = _cov(d, 0.008)
    rgb = col(c, 0.07, 0.07, 0.08) * (0.7 + 0.6 * c.n01(2.0, 3, 30)) + 0.05 * (1 - c.Y)
    return _sprite(c, rgb, a)


def s_plant(c):
    """Potted plant, canvas 0.9 x 1.5 m."""
    X, Y = c.X, c.Y
    pot = _cov(_cap(c, 0.5, 0.80, 0.5, 0.96, 0.17, 0.12), 0.008)
    leaves = torch.zeros_like(X)
    k = 0
    for ang in (-1.25, -0.95, -0.6, -0.3, 0.0, 0.3, 0.6, 0.95, 1.25, -0.75, 0.75, 0.15, -0.15):
        k += 1
        L = 0.42 + 0.22 * ((k * 37) % 10) / 10
        x1, y1 = 0.5 + L * math.sin(ang) * 0.9, 0.74 - L * math.cos(ang)
        leaves = torch.maximum(leaves, _cov(_cap(c, 0.5, 0.76, x1, y1, 0.012, 0.045) , 0.008))
    g = col(c, 0.07, 0.22, 0.08) * (0.5 + 0.9 * c.n01(2.0, 3, 20))
    rgb = lerp(g, col(c, 0.45, 0.22, 0.14) * (0.7 + 0.3 * (1 - ((X - 0.5) / 0.17) ** 2).clamp(0, 1)), pot)
    return _sprite(c, rgb, torch.maximum(pot, leaves))


def s_lamp(c):
    """Floor lamp with a glowing shade, canvas 0.6 x 1.7 m."""
    X, Y = c.X, c.Y
    shade = ((Y > 0.03) & (Y < 0.22) & ((X - 0.5).abs() < 0.16 + 0.75 * (Y - 0.03))).float()
    pole = _cov(_cap(c, 0.5, 0.22, 0.5, 0.96, 0.022), 0.008)
    base = _cov(_ell(c, 0.5, 0.975, 0.24, 0.022), 0.008)
    rgb = lerp(col(c, 0.10, 0.08, 0.06).expand(3, TS, TS), col(c, 1.0, 0.82, 0.50) * (0.75 + 0.25 * (1 - ((X - 0.5) / 0.3) ** 2)), shade)
    return _sprite(c, rgb, torch.maximum(shade, torch.maximum(pole, base)))


def s_sign(c):
    """Yellow wet-floor sign, canvas 0.6 x 0.7 m."""
    X, Y = c.X, c.Y
    m = ((Y > 0.04) & (Y < 0.985) & ((X - 0.5).abs() < 0.12 + 0.34 * Y)).float()
    rgb = col(c, 0.92, 0.75, 0.05) * (0.8 + 0.2 * c.n01(2.0, 2, 20))
    tri = ((Y > 0.30) & (Y < 0.62) & ((X - 0.5).abs() < (Y - 0.30) * 0.5)).float()
    rgb = lerp(rgb, col(c, 0.05, 0.05, 0.05), tri * (1 - ((Y > 0.36) & (Y < 0.585) & ((X - 0.5).abs() < (Y - 0.36) * 0.38)).float()))
    rgb = lerp(rgb, col(c, 0.05, 0.05, 0.05), band(Y, 0.70, 0.76, 0.004) * band(X, 0.3, 0.7, 0.004))
    return _sprite(c, rgb, m)


def s_cone(c):
    X, Y = c.X, c.Y
    m = (((Y > 0.05) & (Y < 0.93) & ((X - 0.5).abs() < 0.03 + 0.26 * (Y - 0.05))) | ((Y >= 0.93) & (Y < 0.99) & ((X - 0.5).abs() < 0.38))).float()
    rgb = col(c, 0.95, 0.35, 0.05) * (0.6 + 0.5 * (1 - ((X - 0.45) / 0.3) ** 2).clamp(0, 1))
    rgb = lerp(rgb, col(c, 0.9, 0.9, 0.9), band(Y, 0.35, 0.50, 0.004))
    return _sprite(c, rgb, m)


def s_barrel(c, cc=(0.10, 0.22, 0.40)):
    """Drum, canvas 0.65 x 0.95 m."""
    X, Y = c.X, c.Y
    m = box(c, 0.06, 0.03, 0.94, 0.985, 0.006)
    shade = (1 - ((X - 0.42) / 0.5) ** 2).clamp(0.15, 1)
    rgb = col(c, *cc) * shade * (0.7 + 0.5 * c.n01(2.0, 2, 20))
    rgb = rgb * (1 + 0.5 * (band(Y, 0.30, 0.33, 0.004) + band(Y, 0.66, 0.69, 0.004)))
    rgb = lerp(rgb, col(c, 0.35, 0.17, 0.07), sstep(0.55, 0.85, c.n01(2.5, 2, 20)) * 0.7)
    return _sprite(c, rgb, m)


def s_cooler(c):
    """Water cooler, canvas 0.45 x 1.35 m."""
    X, Y = c.X, c.Y
    body = box(c, 0.12, 0.36, 0.88, 0.985, 0.006)
    bottle = _cov(_union(_cap(c, 0.5, 0.10, 0.5, 0.28, 0.27, 0.27), _cap(c, 0.5, 0.30, 0.5, 0.36, 0.10)), 0.008)
    rgb = col(c, 0.82, 0.83, 0.82) * (0.6 + 0.4 * (1 - ((X - 0.42) / 0.45) ** 2).clamp(0, 1))
    rgb = lerp(rgb, col(c, 0.1, 0.1, 0.1), box(c, 0.30, 0.46, 0.70, 0.56, 0.004))
    rgb = lerp(rgb, col(c, 0.45, 0.68, 0.90) * (0.6 + 0.5 * (1 - ((X - 0.40) / 0.3) ** 2).clamp(0, 1)), bottle)
    return _sprite(c, rgb, torch.maximum(body, bottle))


def s_gurney(c):
    """Hospital trolley seen from the side, canvas 2.0 x 1.05 m."""
    X, Y = c.X, c.Y
    mat = box(c, 0.05, 0.30, 0.95, 0.44, 0.006)
    sheet = _cov(_union(_cap(c, 0.25, 0.27, 0.80, 0.25, 0.07, 0.05), _ell(c, 0.14, 0.26, 0.07, 0.06)), 0.006)
    frame = _cov(_union(_cap(c, 0.08, 0.46, 0.92, 0.46, 0.015), _cap(c, 0.14, 0.46, 0.14, 0.90, 0.014), _cap(c, 0.86, 0.46, 0.86, 0.90, 0.014),
                        _cap(c, 0.14, 0.72, 0.86, 0.72, 0.012), _cap(c, 0.14, 0.46, 0.86, 0.72, 0.008)), 0.006)
    wheels = _cov(_union(_ell(c, 0.14, 0.94, 0.035, 0.05), _ell(c, 0.86, 0.94, 0.035, 0.05)), 0.006)
    rgb = col(c, 0.55, 0.57, 0.58).expand(3, TS, TS)
    rgb = lerp(rgb, col(c, 0.20, 0.30, 0.28), mat)
    rgb = lerp(rgb, col(c, 0.80, 0.80, 0.76) * (0.75 + 0.25 * c.n01(2.5, 2, 12)), sheet)
    rgb = lerp(rgb, col(c, 0.05, 0.05, 0.05), wheels)
    return _sprite(c, rgb, torch.maximum(torch.maximum(mat, sheet), torch.maximum(frame, wheels)))


def s_float(c):
    """Swim ring lying on the water, canvas 0.9 x 0.3 m."""
    X, Y = c.X, c.Y
    a = _cov(_ell(c, 0.5, 0.55, 0.46, 0.38), 0.01) * (1 - _cov(_ell(c, 0.5, 0.45, 0.22, 0.14), 0.01))
    seg = (torch.floor((torch.atan2((Y - 0.5) * 3, X - 0.5) / PI + 1) * 4) % 2)[None]
    rgb = lerp(col(c, 0.92, 0.92, 0.92), col(c, 0.88, 0.12, 0.10), seg) * (0.7 + 0.4 * (1 - Y))
    return _sprite(c, rgb, a)


def s_bulb(c):
    """Bare bulb on a wire, canvas 0.3 x 0.9 m (hangs from the ceiling)."""
    X, Y = c.X, c.Y
    wire = _cov(_cap(c, 0.5, 0.0, 0.5, 0.80, 0.012), 0.01)
    bulb = _cov(_ell(c, 0.5, 0.88, 0.20, 0.085), 0.01)
    sock = box(c, 0.38, 0.74, 0.62, 0.82, 0.01)
    rgb = lerp(col(c, 0.03, 0.03, 0.03).expand(3, TS, TS), col(c, 1.0, 0.93, 0.75), bulb)
    return _sprite(c, rgb, torch.maximum(torch.maximum(wire, sock), bulb))


def s_column(c, base=(0.86, 0.88, 0.88), tiled=True):
    """Round column, floor to ceiling. Canvas 0.7 m wide."""
    X, Y = c.X, c.Y
    m = band(X, 0.06, 0.94, 0.006)
    th = ((X - 0.5) / 0.44).clamp(-1, 1)
    shade = (1 - th * th).sqrt() * 0.75 + 0.25
    rgb = col(c, *base) * shade
    if tiled:
        ang = torch.asin(th) / PI + 0.5
        g = torch.maximum(lines(ang, 10, 0.006), lines(Y, 24, 0.0025))
        rgb = lerp(rgb, rgb * 0.72, g)
    return _sprite(c, rgb, m)


def s_boxes(c):
    """Stack of cardboard boxes, canvas 0.9 x 1.1 m."""
    X, Y = c.X, c.Y
    m = torch.maximum(box(c, 0.04, 0.52, 0.96, 0.985, 0.006), box(c, 0.16, 0.08, 0.80, 0.52, 0.006))
    rgb = col(c, 0.52, 0.38, 0.22) * (0.75 + 0.3 * c.n01(2.2, 2, 20))
    rgb = lerp(rgb, col(c, 0.62, 0.50, 0.32), band(X, 0.46, 0.54, 0.004))
    rgb = rgb * (1 - 0.5 * band(Y, 0.515, 0.525, 0.002)) * (1 - 0.3 * band(X, 0.495, 0.505, 0.002))
    return _sprite(c, rgb, m)


def s_exit(c):
    """Exit sign hanging from the ceiling, canvas 0.7 x 0.5 m."""
    X, Y = c.X, c.Y
    rods = torch.maximum(box(c, 0.18, 0.0, 0.21, 0.5, 0.004), box(c, 0.79, 0.0, 0.82, 0.5, 0.004))
    sign = box(c, 0.04, 0.48, 0.96, 0.96, 0.006)
    txt = band((X * 6.5) % 1, 0.25, 0.75, 0.05) * band(Y, 0.60, 0.84, 0.006) * band(X, 0.14, 0.86, 0.01)
    rgb = lerp(col(c, 0.04, 0.60, 0.22).expand(3, TS, TS), col(c, 0.95, 1.0, 0.95), txt)
    rgb = lerp(col(c, 0.2, 0.2, 0.2).expand(3, TS, TS), rgb, sign)
    return _sprite(c, rgb, torch.maximum(rods, sign))


# ---------------------------------------------------------------- outdoors, and the newer indoor levels

def grass(c, base=(0.22, 0.36, 0.13), dry=0.3):
    speck = c.fbm(0.3, 50)
    mid = c.n01(2.2, 3, 30)
    big = c.n01(3.0, 1, 5)
    rgb = col(c, *base) * (0.84 + 0.06 * speck + 0.28 * (mid - 0.5))
    rgb = lerp(rgb, col(c, 0.42, 0.38, 0.16) * (0.9 + 0.07 * speck), sstep(0.55, 0.9, big) * dry)
    return out(c, rgb)


def crop_top(c, base=(0.62, 0.50, 0.20)):
    """A stand of grain seen from above."""
    speck = c.fbm(0.2, 70)
    rows = 0.5 + 0.5 * torch.cos(2 * PI * c.X * 12)
    wind = c.n01(2.5, 1, 8)
    rgb = col(c, *base) * (0.78 + 0.09 * speck + 0.10 * rows + 0.26 * (wind - 0.5))
    return out(c, rgb)


def crop_side(c, base=(0.62, 0.50, 0.19), stalk=(0.28, 0.25, 0.09), heads=True):
    """The edge of a stand of grain or maize: stalks, darker towards the ground, ears at the top."""
    Y = c.Y
    st = c.n01(0.6, 12, 120, sy=40.0)
    st2 = c.fbm(0.8, 20, sy=6.0)
    rgb = lerp(col(c, *stalk), col(c, *base), sstep(0.25, 0.75, st)) * (0.9 + 0.1 * st2)
    rgb = rgb * (0.50 + 0.50 * (1 - Y) ** 0.7)
    if heads:
        ear = col(c, min(1.0, base[0] * 1.25), min(1.0, base[1] * 1.2), base[2] * 1.1) * (0.85 + 0.2 * c.n01(0.5, 30))
        rgb = lerp(rgb, ear, (1 - sstep(0.0, 0.3, Y)) * sstep(0.35, 0.7, st))
    else:       # maize: broad leaves hanging across the stalks
        leaf = sstep(0.62, 0.8, c.n01(1.2, 6, 60, sx=3.0))
        rgb = lerp(rgb, col(c, base[0] * 0.7, base[1] * 0.85, base[2] * 0.6), leaf * 0.7)
    return out(c, rgb)


def dirt(c, base=(0.30, 0.22, 0.14)):
    speck = c.fbm(0.3, 40)
    mid = c.n01(2.4, 2, 20)
    rgb = col(c, *base) * (0.8 + 0.12 * speck + 0.3 * (mid - 0.5))
    peb = sstep(0.93, 0.97, c.n01(0.2, 30, 90))
    return out(c, lerp(rgb, col(c, 0.45, 0.42, 0.38), peb * 0.7))


def asphalt(c, tone=0.10, cracks=0.5):
    speck = c.fbm(0.2, 60)
    mid = c.n01(2.5, 2, 16)
    rgb = col(c, tone, tone, tone * 1.06) * (0.8 + 0.25 * speck + 0.3 * (mid - 0.5))
    cr = c.n01(1.6, 3, 40)
    rgb = rgb * (1 - cracks * 0.6 * band(cr, 0.495, 0.505, 0.004))
    rgb = lerp(rgb, col(c, tone * 0.5, tone * 0.5, tone * 0.5), sstep(0.7, 0.95, c.n01(3.0, 1, 5)) * 0.5)   # patches
    return out(c, rgb)


def siding(c, base=(0.72, 0.72, 0.68), boards=9, wear=0.5):
    """Clapboard: horizontal boards, each casting a line of shadow on the one below."""
    Y = c.Y
    fy = (Y * boards) % 1
    rgb = col(c, *base) * (0.92 + 0.05 * c.fbm(1.0, 6, sx=8.0)) * (0.84 + 0.16 * fy)
    rgb = rgb * (1 - 0.45 * band(fy, 0.0, 0.07, 0.02))
    rgb = lerp(rgb, rgb * col(c, 0.55, 0.52, 0.45), sstep(0.6, 0.95, c.n01(3.0, 1, 8)) * wear * 0.6)
    rgb = lerp(rgb, rgb * 0.7, drips(c, 30, 0.4) * wear * 0.5)
    return out(c, rgb)


def wall_window(c, wall, lit=False, y0=0.30, y1=0.68, frame=(0.85, 0.85, 0.82)):
    """One cell (1 m) of `wall` with a window: dark, or lit from inside behind a curtain (the glass then glows)."""
    X, Y = c.X, c.Y
    rgb = wall[:3].clone()
    fr = box(c, 0.14, y0 - 0.02, 0.86, y1 + 0.02, 0.003)
    gl = box(c, 0.18, y0, 0.82, y1, 0.003)
    if lit:
        cur = 0.55 + 0.45 * torch.cos(2 * PI * X * 9) ** 2
        glass = col(c, 1.0, 0.78, 0.42) * cur * (0.75 + 0.25 * c.n01(2.0, 2, 10))
    else:
        glass = col(c, 0.015, 0.02, 0.03) + col(c, 0.05, 0.06, 0.07) * sstep(0.3, 0.9, c.n01(2.5, 1, 6)) * 0.6
    ym = (y0 + y1) / 2
    cross = torch.maximum(band(X, 0.49, 0.51, 0.003), band(Y, ym - 0.008, ym + 0.008, 0.003)) * gl
    rgb = lerp(rgb, col(c, *frame), fr)
    rgb = lerp(rgb, glass, gl)
    rgb = lerp(rgb, col(c, *frame), cross)
    return out(c, rgb, gl * (1 - cross) if lit else None)


def fence(c, base=(0.40, 0.30, 0.20), planks=6):
    X, Y = c.X, c.Y
    px = torch.floor(X * planks)
    fx = X * planks - px
    grain = 0.85 + 0.2 * (0.5 + 0.5 * torch.cos(2 * PI * (X * 40 + 1.5 * c.fbm(2.0, 1, 10, sy=8.0))))
    rgb = col(c, *base) * grain * (0.8 + 0.35 * c.cellrand(px, px * 0, 5)[None])
    rgb = rgb * (0.2 + 0.8 * band(fx, 0.08, 0.92, 0.02))                               # gaps between the planks
    rgb = rgb * (0.25 + 0.75 * sstep(0.0, 0.06, Y - 0.05 * (fx - 0.5).abs() * 2))      # pointed tops
    rgb = rgb * (1 - 0.25 * (band(Y, 0.22, 0.27, 0.004) + band(Y, 0.72, 0.77, 0.004)))  # rails
    return out(c, rgb)


def hedge(c, base=(0.06, 0.16, 0.06)):
    leaf = c.n01(0.8, 20, 120)
    mid = c.n01(2.2, 3, 20)
    rgb = col(c, *base) * (0.5 + 1.0 * leaf) * (0.7 + 0.6 * mid) * (0.6 + 0.4 * (1 - c.Y))
    return out(c, rgb)


def facade(c, base=(0.42, 0.40, 0.38), lit=0.0, style=0):
    """One storey (3.2 m) and two windows (4 m) of a city building. lit: the share of its windows that glow."""
    X, Y = c.X, c.Y
    rgb = col(c, *base) * (0.9 + 0.06 * c.fbm(1.2, 4) + 0.04 * c.fbm(0.4, 40))
    if style == 1:
        rgb = rgb * (1 - 0.18 * lines(Y, 16, 0.006))
    rgb = lerp(rgb, rgb * 0.6, drips(c, 30, 0.4) * 0.5)
    rgb = rgb * (1 - 0.35 * band(Y, 0.0, 0.035, 0.004))
    fx = (X * 2) % 1
    wx = torch.floor(X * 2)
    gl = band(fx, 0.22, 0.78, 0.004) * band(Y, 0.22, 0.72, 0.004)
    fr = band(fx, 0.19, 0.81, 0.004) * band(Y, 0.19, 0.76, 0.004)
    glass = col(c, 0.03, 0.04, 0.05) + col(c, 0.10, 0.12, 0.14) * sstep(0.2, 0.9, c.n01(2.5, 1, 6)) * (1 - Y)
    rgb = lerp(rgb, rgb * 0.5, fr)
    em = None
    if lit > 0:
        on = (c.cellrand(wx, wx * 0 + 3, 6) < lit).float()[None]
        glass = lerp(glass, col(c, 1.0, 0.80, 0.45) * (0.7 + 0.3 * c.n01(2.0, 2, 8)), on)
        em = gl * on[0]
    rgb = lerp(rgb, glass, gl)
    rgb = rgb * (1 - 0.5 * gl * torch.maximum(band(fx, 0.49, 0.51, 0.003), band(Y, 0.46, 0.48, 0.003)))
    return out(c, rgb, em)


def shopfront(c, wall, sign=(0.9, 0.2, 0.2)):
    """Two metres of shop: a dark window with shapes behind the glass, and a sign above it that glows."""
    X, Y = c.X, c.Y
    rgb = wall[:3].clone()
    gl = box(c, 0.05, 0.27, 0.95, 0.93, 0.003)
    inside = col(c, 0.03, 0.03, 0.035) + col(c, 0.10, 0.10, 0.11) * sstep(0.4, 0.9, c.n01(3.0, 1, 5))
    for xc in (0.3, 0.68):
        d = _union(_ell(c, xc, 0.42, 0.035, 0.035), _cap(c, xc, 0.47, xc, 0.70, 0.055, 0.04), _cap(c, xc, 0.70, xc, 0.92, 0.035, 0.03))
        inside = lerp(inside, col(c, 0.09, 0.09, 0.10), _cov(d, 0.01) * 0.8)
    rgb = lerp(rgb, col(c, 0.12, 0.12, 0.13), box(c, 0.03, 0.25, 0.97, 0.95, 0.003))
    rgb = lerp(rgb, inside, gl)
    rgb = rgb + gl * 0.05 * (1 - sstep(0.0, 0.4, (X - Y * 0.6 - 0.1).abs()))
    sg = box(c, 0.10, 0.07, 0.90, 0.20, 0.004)
    txt = band((X * 9) % 1, 0.2, 0.8, 0.05) * band(Y, 0.105, 0.165, 0.004) * band(X, 0.18, 0.82, 0.01)
    rgb = lerp(rgb, col(c, *sign) * 0.5, sg)
    rgb = lerp(rgb, col(c, 1.0, 1.0, 1.0), txt * sg)
    return out(c, rgb, sg * (0.35 + 0.65 * txt))


def shutter(c, wall, base=(0.45, 0.46, 0.48), top=0.2):
    """A roller shutter set in `wall`."""
    Y = c.Y
    rgb = wall[:3].clone()
    lf = box(c, 0.06, top, 0.94, 1.0, 0.003)
    slat = 0.5 + 0.5 * torch.cos(2 * PI * Y * 30)
    l = col(c, *base) * (0.7 + 0.3 * slat) * (0.9 + 0.1 * c.fbm(1.5, 3, 20))
    l = lerp(l, l * col(c, 0.6, 0.45, 0.35), sstep(0.6, 0.95, c.n01(2.5, 2, 12)) * 0.5)
    rgb = lerp(rgb, col(c, 0.2, 0.2, 0.21), box(c, 0.03, top - 0.03, 0.97, 1.0, 0.003))
    return out(c, lerp(rgb, l, lf))


def corrugated(c, base=(0.35, 0.16, 0.12)):
    """Painted corrugated steel (shipping containers, sheds)."""
    X = c.X
    rib = 0.5 + 0.5 * torch.cos(2 * PI * X * 16)
    rgb = col(c, *base) * (0.7 + 0.35 * rib) * (0.9 + 0.12 * c.fbm(1.5, 3, 20))
    rgb = lerp(rgb, col(c, 0.30, 0.15, 0.07), sstep(0.62, 0.92, c.n01(2.5, 2, 14)) * 0.7)
    return out(c, lerp(rgb, rgb * 0.6, drips(c, 30, 0.5) * 0.5))


def sky(c, stars=False):
    """Clouds for the sky dome (brightness only; the simulator colours it). With stars: where they are, in alpha."""
    lum = 0.62 * c.n01(2.6, 1, 40, sy=2.5) + 0.38 * c.n01(2.0, 4, 80)
    a = None
    if stars:
        st = (c.rand(TS, TS) > 0.9975).float() * (0.4 + 0.6 * c.rand(TS, TS))
        a = st * (1 - sstep(0.45, 0.7, lum))
    return out(c, lum[None].expand(3, TS, TS), a)


def s_tree(c, leaves=False):
    """A tree: bare, or a dark conifer. Canvas 3.0 x 5.2 m."""
    X, Y = c.X, c.Y
    if leaves:
        d = _cap(c, 0.5, 0.80, 0.5, 0.99, 0.035, 0.045)
        a = _cov(d, 0.006)
        for k in range(6):
            y0 = 0.06 + 0.13 * k
            w = 0.10 + 0.065 * k
            tri = ((Y > y0) & (Y < y0 + 0.24) & ((X - 0.5).abs() < w * (Y - y0) / 0.24 + 0.01)).float()
            a = torch.maximum(a, tri)
        rgb = col(c, 0.035, 0.085, 0.045) * (0.45 + 1.1 * c.n01(1.0, 12, 90)) * (0.7 + 0.5 * (1 - Y))
        rgb = lerp(rgb, col(c, 0.10, 0.07, 0.045).expand(3, TS, TS), _cov(d, 0.006) * (Y > 0.84).float())
        return _sprite(c, rgb, a)
    segs = [(0.5, 0.99, 0.5, 0.52, 0.045, 0.03)]
    tips = [(0.5, 0.52, -PI / 2, 0.20, 0.03)]
    for depth in range(4):
        nxt = []
        for i, (x, y, ang, L, r) in enumerate(tips):
            for j, da in enumerate((-0.55, 0.42) if depth else (-0.6, 0.05, 0.62)):
                a2 = ang + da + 0.12 * math.sin(7.3 * i + 3.1 * j + depth)
                x1, y1 = x + L * math.cos(a2) * 0.9, y + L * math.sin(a2)
                segs.append((x, y, x1, y1, r, r * 0.6))
                nxt.append((x1, y1, a2, L * 0.68, r * 0.6))
        tips = nxt
    d = _union(*[_cap(c, *sg) for sg in segs])
    rgb = col(c, 0.06, 0.05, 0.04) * (0.7 + 0.6 * c.n01(1.5, 6, 60, sy=6.0))
    return _sprite(c, rgb, _cov(d, 0.004))


def s_lamppost(c):
    """Street lamp without its light (that is `s_lamphead`). Canvas 1.0 x 4.2 m."""
    d = _union(_cap(c, 0.5, 0.995, 0.5, 0.10, 0.035, 0.022), _cap(c, 0.5, 0.10, 0.62, 0.05, 0.02, 0.016),
               _cap(c, 0.62, 0.05, 0.80, 0.055, 0.016), _cap(c, 0.5, 0.995, 0.5, 0.93, 0.06, 0.045))
    rgb = col(c, 0.05, 0.055, 0.06) * (0.8 + 0.4 * c.n01(2.0, 2, 30))
    return _sprite(c, rgb, _cov(d, 0.004))


def s_lamphead(c):
    """The glowing head of a street lamp. Canvas 0.7 x 0.35 m."""
    a = _cov(_ell(c, 0.5, 0.5, 0.42, 0.30), 0.02)
    core = 1 - sstep(0.0, 0.5, torch.sqrt((c.X - 0.5) ** 2 + (c.Y - 0.5) ** 2))
    rgb = lerp(col(c, 1.0, 0.62, 0.22), col(c, 1.0, 0.92, 0.70), core)
    return _sprite(c, rgb, a)


def s_pole(c):
    """Utility pole with two crossbars. Canvas 1.6 x 7.0 m."""
    d = _union(_cap(c, 0.5, 0.995, 0.5, 0.02, 0.022, 0.016), _cap(c, 0.12, 0.07, 0.88, 0.07, 0.010), _cap(c, 0.22, 0.13, 0.78, 0.13, 0.009))
    rgb = col(c, 0.09, 0.07, 0.05) * (0.7 + 0.6 * c.n01(1.5, 6, 60, sy=8.0))
    return _sprite(c, rgb, _cov(d, 0.003))


def s_scarecrow(c):
    """A scarecrow on its pole. Canvas 1.8 x 2.3 m."""
    X, Y = c.X, c.Y
    pole = _union(_cap(c, 0.5, 0.995, 0.5, 0.10, 0.014), _cap(c, 0.10, 0.30, 0.90, 0.30, 0.012))
    body = _union(_cap(c, 0.5, 0.30, 0.5, 0.62, 0.10, 0.12), _cap(c, 0.40, 0.30, 0.14, 0.31, 0.045, 0.05), _cap(c, 0.60, 0.30, 0.86, 0.31, 0.045, 0.05),
                  _cap(c, 0.45, 0.62, 0.43, 0.82, 0.05, 0.06), _cap(c, 0.55, 0.62, 0.57, 0.82, 0.05, 0.06))
    head = _ell(c, 0.5, 0.185, 0.065, 0.075)
    hat = _union(_cap(c, 0.36, 0.125, 0.64, 0.125, 0.012), _cap(c, 0.5, 0.115, 0.5, 0.06, 0.05, 0.035))
    straw = torch.zeros_like(X)
    for (x0, y0, dx) in ((0.11, 0.31, -0.05), (0.89, 0.31, 0.05), (0.43, 0.83, -0.02), (0.57, 0.83, 0.02)):
        for k in range(5):
            straw = torch.maximum(straw, _cov(_cap(c, x0, y0, x0 + dx + 0.02 * (k - 2), y0 + 0.07 + 0.01 * (k % 2), 0.004), 0.003))
    a = torch.maximum(_cov(_union(pole, body, head, hat), 0.005), straw)
    rgb = col(c, 0.20, 0.16, 0.11) * (0.6 + 0.7 * c.n01(1.5, 8, 60))
    rgb = lerp(rgb, col(c, 0.16, 0.20, 0.30) * (0.6 + 0.6 * c.n01(1.5, 8, 60)), _cov(body, 0.005) * (Y > 0.60).float())
    rgb = lerp(rgb, col(c, 0.50, 0.42, 0.26) * (0.75 + 0.4 * c.n01(1.0, 20, 90)), _cov(head, 0.005))
    for ex in (0.475, 0.525):
        rgb = lerp(rgb, col(c, 0, 0, 0), _cov(_ell(c, ex, 0.175, 0.012, 0.012), 0.003))
    rgb = lerp(rgb, col(c, 0, 0, 0), _cov(_cap(c, 0.465, 0.215, 0.535, 0.215, 0.004), 0.002) + lines(X - 0.5, 60, 0.002) * band(Y, 0.205, 0.225, 0.002) * band(X, 0.465, 0.535, 0.002))
    rgb = lerp(rgb, col(c, 0.07, 0.06, 0.05).expand(3, TS, TS), _cov(hat, 0.005))
    rgb = lerp(rgb, col(c, 0.62, 0.52, 0.22).expand(3, TS, TS), straw)
    return _sprite(c, rgb, a)


def s_haybale(c):
    """A round bale seen end on. Canvas 1.5 x 1.4 m."""
    X, Y = c.X, c.Y
    r = torch.sqrt(((X - 0.5) / 0.47) ** 2 + ((Y - 0.52) / 0.47) ** 2)
    a = 1 - sstep(0.97, 1.0, r)
    rings = 0.5 + 0.5 * torch.cos(2 * PI * (r * 7 + 0.15 * c.fbm(2.0, 2, 20)))
    rgb = col(c, 0.60, 0.48, 0.20) * (0.6 + 0.3 * rings + 0.25 * c.n01(0.6, 20, 100)) * (0.65 + 0.35 * (1 - Y))
    return _sprite(c, rgb, a)


def s_bed(c):
    """A bed seen from the side. Canvas 2.1 x 1.0 m."""
    X, Y = c.X, c.Y
    frame = torch.maximum(box(c, 0.02, 0.62, 0.98, 0.80, 0.004), torch.maximum(box(c, 0.02, 0.18, 0.07, 0.99, 0.004), box(c, 0.93, 0.50, 0.98, 0.99, 0.004)))
    mat = box(c, 0.07, 0.48, 0.93, 0.64, 0.006)
    pil = _cov(_cap(c, 0.13, 0.43, 0.24, 0.43, 0.055), 0.006)
    blanket = box(c, 0.32, 0.45, 0.93, 0.74, 0.006)
    rgb = col(c, 0.20, 0.12, 0.07) * (0.8 + 0.3 * c.n01(2.0, 2, 20, sx=6.0))
    rgb = lerp(rgb, col(c, 0.78, 0.76, 0.70) * (0.8 + 0.2 * c.n01(2.5, 2, 12)), mat)
    rgb = lerp(rgb, col(c, 0.30, 0.10, 0.10) * (0.7 + 0.4 * c.n01(2.2, 2, 14)), blanket)
    rgb = lerp(rgb, col(c, 0.85, 0.84, 0.80).expand(3, TS, TS), pil)
    return _sprite(c, rgb, torch.maximum(torch.maximum(frame, mat), torch.maximum(pil, blanket)))


def s_tv(c):
    """A television showing static, on a low cabinet. Canvas 1.0 x 1.2 m."""
    X, Y = c.X, c.Y
    cab = box(c, 0.08, 0.62, 0.92, 0.99, 0.004)
    set_ = box(c, 0.14, 0.10, 0.86, 0.62, 0.006)
    scr = box(c, 0.20, 0.16, 0.74, 0.56, 0.006)
    static = (0.25 + 0.75 * c.rand(TS, TS)) * (0.8 + 0.2 * torch.cos(2 * PI * Y * 40))
    rgb = col(c, 0.05, 0.035, 0.025) * (0.8 + 0.4 * c.n01(2.0, 2, 20, sx=6.0))
    rgb = lerp(rgb, col(c, 0.03, 0.03, 0.03).expand(3, TS, TS), set_)
    rgb = lerp(rgb, col(c, 0.75, 0.82, 0.95) * static, scr)
    return _sprite(c, rgb, torch.maximum(cab, set_))


def s_dresser(c):
    """Chest of drawers. Canvas 1.2 x 1.0 m."""
    X, Y = c.X, c.Y
    m = box(c, 0.04, 0.06, 0.96, 0.99, 0.004)
    rgb = col(c, 0.24, 0.14, 0.08) * (0.8 + 0.3 * c.n01(2.0, 2, 20, sx=6.0))
    for y0 in (0.12, 0.42, 0.70):
        dr = box(c, 0.09, y0, 0.91, y0 + 0.24, 0.004)
        rgb = rgb * (1 - 0.35 * (box(c, 0.08, y0 - 0.01, 0.92, y0 + 0.25, 0.004) - dr).clamp(0, 1))
        rgb = lerp(rgb, col(c, 0.75, 0.65, 0.35).expand(3, TS, TS), box(c, 0.44, y0 + 0.10, 0.56, y0 + 0.13, 0.003))
    return _sprite(c, rgb, m)


def s_bench(c):
    """Park bench. Canvas 1.7 x 0.9 m."""
    slats = torch.zeros_like(c.X)
    for y0 in (0.10, 0.24, 0.38, 0.56):
        slats = torch.maximum(slats, box(c, 0.03, y0, 0.97, y0 + 0.09, 0.004))
    legs = torch.maximum(box(c, 0.10, 0.10, 0.15, 0.99, 0.004), box(c, 0.85, 0.10, 0.90, 0.99, 0.004))
    rgb = lerp(col(c, 0.30, 0.20, 0.11) * (0.75 + 0.4 * c.n01(2.0, 2, 20, sx=8.0)), col(c, 0.05, 0.05, 0.055).expand(3, TS, TS), legs * (1 - slats))
    return _sprite(c, rgb, torch.maximum(slats, legs))


def shingles(c, base=(0.36, 0.20, 0.13)):
    """A roof seen from the side: rows of shingles."""
    X, Y = c.X, c.Y
    row = torch.floor(Y * 8)
    fx = (X * 10 + 0.5 * (row % 2)) % 1
    fy = (Y * 8) % 1
    rgb = col(c, *base) * (0.75 + 0.35 * c.cellrand(torch.floor(X * 10 + 0.5 * (row % 2)), row, 3)[None]) * (0.8 + 0.2 * fy)
    rgb = rgb * (1 - 0.5 * band(fy, 0.0, 0.10, 0.03)) * (1 - 0.35 * band(fx, 0.0, 0.06, 0.02))
    return out(c, rgb * (0.92 + 0.08 * c.fbm(1.0, 6)))


def picket(c, behind=(0.20, 0.34, 0.12)):
    """A white picket fence; between the pickets, the colour of the grass behind it."""
    X, Y = c.X, c.Y
    fx = (X * 6) % 1
    pk = band(fx, 0.2, 0.8, 0.02) * sstep(0.0, 0.10, Y - 0.16 * (fx - 0.5).abs() * 2)
    rail = (band(Y, 0.30, 0.38, 0.004) + band(Y, 0.72, 0.80, 0.004)).clamp(0, 1)
    wood_ = col(c, 0.93, 0.93, 0.90) * (0.9 + 0.1 * c.n01(2.0, 2, 30, sy=6.0)) * (1 - 0.15 * (1 - band(fx, 0.25, 0.75, 0.03)))
    rgb = lerp(col(c, *behind) * (0.7 + 0.5 * c.n01(1.0, 10, 80)), wood_, torch.maximum(pk, rail * 0.9))
    return out(c, rgb)


def backdrop_hills(c):
    """What stands on the horizon of the calm level, all the way round: green hills, a few white houses, a water
    tower. The lower edge is the horizon; alpha is 1 where there is land."""
    X, Y = c.X, c.Y
    ridge = lambda lo, hi, k, a: lo + (hi - lo) * (0.5 + 0.5 * torch.cos(2 * PI * (X * k + a))) * 0.5 + (hi - lo) * 0.5 * c.n01(2.4, 1, 6, sy=1e3)
    far, near = 1 - ridge(0.45, 0.95, 2, 0.1), 1 - ridge(0.15, 0.55, 3, 0.4)        # the ridge lines, as Y
    a_far, a_near = sstep(-0.004, 0.004, Y - far), sstep(-0.004, 0.004, Y - near)
    g = c.n01(2.0, 2, 30)
    rgb = col(c, 0.36, 0.52, 0.40) * (0.85 + 0.25 * g) * (1 - 0.25 * (Y - far).clamp(0, 0.3) / 0.3)
    rgb = lerp(rgb, col(c, 0.26, 0.48, 0.20) * (0.8 + 0.35 * g) * (1 - 0.3 * (Y - near).clamp(0, 0.3) / 0.3), a_near)
    alpha = torch.maximum(a_far, a_near)
    for (xc, on_near) in ((0.08, 1), (0.21, 0), (0.37, 1), (0.52, 0), (0.66, 1), (0.83, 0), (0.93, 1)):       # houses on the ridges
        line = near if on_near else far
        j = min(TS - 1, int(xc * TS))
        yb = float(line[0, j]) + 0.01
        w, h = (0.0095, 0.16) if on_near else (0.006, 0.10)
        wall = box(c, xc - w, yb - h, xc + w, yb, 0.002)
        roof = ((Y < yb - h) & (Y > yb - h * 1.7) & ((X - xc).abs() < (w * 1.25) * (Y - (yb - h * 1.7)) / (h * 0.7))).float()
        rgb = lerp(rgb, col(c, 0.92, 0.92, 0.88).expand(3, TS, TS), wall)
        rgb = lerp(rgb, col(c, 0.10, 0.10, 0.12).expand(3, TS, TS), box(c, xc - w * 0.35, yb - h * 0.7, xc + w * 0.1, yb - h * 0.35, 0.001))
        rgb = lerp(rgb, col(c, 0.40, 0.22, 0.15).expand(3, TS, TS), roof)
        alpha = torch.maximum(alpha, torch.maximum(wall, roof))
    xt = 0.74                                                                                   # the water tower
    yb = float(far[0, int(xt * TS)]) + 0.01
    tw_ = torch.maximum(box(c, xt - 0.006, yb - 0.30, xt + 0.006, yb - 0.20, 0.001),
                        torch.maximum(box(c, xt - 0.005, yb - 0.20, xt - 0.003, yb, 0.0008), box(c, xt + 0.003, yb - 0.20, xt + 0.005, yb, 0.0008)))
    rgb = lerp(rgb, col(c, 0.80, 0.82, 0.84).expand(3, TS, TS), tw_)
    alpha = torch.maximum(alpha, tw_)
    return torch.cat([rgb.clamp(0, 1), alpha.view(1, TS, TS).clamp(0, 1)], 0)


def s_oak(c):
    """A tree in leaf, round. Canvas 3.4 x 4.6 m."""
    X, Y = c.X, c.Y
    trunk = _cov(_union(_cap(c, 0.5, 0.99, 0.5, 0.55, 0.05, 0.035), _cap(c, 0.5, 0.62, 0.38, 0.45, 0.025, 0.015), _cap(c, 0.5, 0.60, 0.63, 0.44, 0.025, 0.015)), 0.006)
    crown = torch.zeros_like(X)
    for (cx, cy, rx, ry) in ((0.5, 0.30, 0.34, 0.24), (0.30, 0.38, 0.22, 0.17), (0.70, 0.38, 0.22, 0.17), (0.5, 0.17, 0.24, 0.15), (0.38, 0.24, 0.2, 0.15), (0.63, 0.25, 0.2, 0.15)):
        crown = torch.maximum(crown, _cov(_ell(c, cx, cy, rx, ry) + 0.02 * (c.n01(1.0, 10, 60) - 0.5), 0.01))
    leaf = col(c, 0.16, 0.40, 0.12) * (0.55 + 0.9 * c.n01(1.2, 8, 80)) * (0.75 + 0.5 * (1 - Y))
    rgb = lerp(col(c, 0.20, 0.13, 0.08) * (0.7 + 0.5 * c.n01(1.5, 6, 60, sy=6.0)), leaf, crown)
    return _sprite(c, rgb, torch.maximum(trunk, crown))


def s_bush(c):
    """A bush. Canvas 1.2 x 0.9 m."""
    X, Y = c.X, c.Y
    a = torch.zeros_like(X)
    for (cx, cy, rx, ry) in ((0.5, 0.60, 0.42, 0.36), (0.28, 0.70, 0.25, 0.27), (0.72, 0.70, 0.25, 0.27)):
        a = torch.maximum(a, _cov(_ell(c, cx, cy, rx, ry) + 0.03 * (c.n01(1.0, 8, 50) - 0.5), 0.012))
    a = a * (Y < 0.985).float()
    rgb = col(c, 0.13, 0.36, 0.11) * (0.55 + 0.9 * c.n01(1.2, 8, 80)) * (0.7 + 0.5 * (1 - Y))
    return _sprite(c, rgb, a)


def s_watertower(c):
    """A water tower on four legs. Canvas 4.5 x 9.0 m."""
    X, Y = c.X, c.Y
    legs = _union(_cap(c, 0.30, 0.995, 0.38, 0.36, 0.012), _cap(c, 0.70, 0.995, 0.62, 0.36, 0.012), _cap(c, 0.45, 0.995, 0.46, 0.36, 0.010),
                  _cap(c, 0.55, 0.995, 0.54, 0.36, 0.010), _cap(c, 0.31, 0.80, 0.69, 0.80, 0.006), _cap(c, 0.34, 0.60, 0.66, 0.60, 0.006),
                  _cap(c, 0.31, 0.80, 0.66, 0.60, 0.004), _cap(c, 0.69, 0.80, 0.34, 0.60, 0.004), _cap(c, 0.5, 0.995, 0.5, 0.36, 0.008))
    tank = box(c, 0.30, 0.13, 0.70, 0.37, 0.004)
    cap_ = ((Y > 0.04) & (Y <= 0.13) & ((X - 0.5).abs() < 0.22 * (Y - 0.04) / 0.09 + 0.01)).float()
    shade = 0.55 + 0.45 * (1 - ((X - 0.42) / 0.24) ** 2).clamp(0, 1)
    rgb = lerp(col(c, 0.30, 0.30, 0.32).expand(3, TS, TS), col(c, 0.88, 0.90, 0.92) * shade * (1 - 0.2 * lines(Y - 0.25, 12, 0.003)), tank)
    rgb = lerp(rgb, col(c, 0.45, 0.47, 0.50) * shade, cap_)
    return _sprite(c, rgb, torch.maximum(torch.maximum(_cov(legs, 0.004), tank), cap_))


def s_scream(c):
    """What fills the picture when one of them has you: a pale face, far too close. Square."""
    X, Y = c.X, c.Y
    a = _cov(_ell(c, 0.5, 0.50, 0.30, 0.42), 0.01)
    r = torch.sqrt(((X - 0.5) / 0.30) ** 2 + ((Y - 0.5) / 0.42) ** 2)
    rgb = col(c, 0.82, 0.80, 0.74) * (0.75 + 0.35 * c.n01(2.2, 2, 30)) * (1 - 0.55 * sstep(0.45, 1.0, r))
    for ex in (0.38, 0.62):
        eye = _cov(_ell(c, ex, 0.38, 0.075, 0.095), 0.008)
        rgb = lerp(rgb, rgb * col(c, 0.5, 0.3, 0.3), _cov(_ell(c, ex, 0.38, 0.11, 0.13), 0.03) * 0.7)
        rgb = lerp(rgb, col(c, 0.0, 0.0, 0.0), eye)
        rgb = lerp(rgb, col(c, 1.0, 1.0, 0.95), _cov(_ell(c, ex + 0.005, 0.385, 0.010, 0.012), 0.004))
    mm = _cov(_ell(c, 0.5, 0.72, 0.13, 0.17), 0.008)
    rgb = lerp(rgb, col(c, 0.03, 0.0, 0.0), mm)
    teeth = mm * (band(Y, 0.555, 0.60, 0.004) + band(Y, 0.84, 0.885, 0.004)) * (1 - lines(X, 22, 0.006))
    rgb = lerp(rgb, col(c, 0.85, 0.82, 0.70), teeth.clamp(0, 1))
    return _sprite(c, rgb, a)


def t_text(c, text, colr=(0.78, 0.03, 0.03)):
    """Words in the middle of a square, in square pixels (alpha: where the letters are). They span 60% of its width."""
    X, Y = c.X, c.Y
    ncol = len(text) * 6 - 1
    bm = torch.zeros(7, ncol)
    for i, ch in enumerate(text):
        bm[:, i * 6:i * 6 + 5] = torch.tensor([[int(v) for v in row] for row in _FONT[ch].split()], dtype=torch.float32)
    bm = bm.to(c.dev)
    pw = 0.6 / ncol
    cx, cy = torch.floor((X - 0.2) / pw).long(), torch.floor((Y - 0.5 + 3.5 * pw) / pw).long()
    ok = (cx >= 0) & (cx < ncol) & (cy >= 0) & (cy < 7)
    m = bm[cy.clamp(0, 6), cx.clamp(0, ncol - 1)] * ok.float()
    m = sstep(0.3, 0.6, F.avg_pool2d(m[None, None], 5, 1, 2)[0, 0])
    return _sprite(c, col(c, *colr).expand(3, TS, TS), m)


# ---------------------------------------------------------------- the atlas and the material table

def build(dev, keep=False):
    """-> dict: atlas uint8 [N, 4] (all textures, all mip levels), base long [NT * NMIP], names (texture -> index)."""
    c = Ctx(dev)
    T, names = [], {}

    def add(name, t):
        names[name] = len(T)
        T.append(t)

    lobby_w = [wallpaper(c, (0.80, 0.70, 0.34), 0, 0.5), wallpaper(c, (0.76, 0.69, 0.40), 1, 0.7), wallpaper(c, (0.83, 0.72, 0.30), 2, 0.4),
               wallpaper(c, (0.70, 0.66, 0.42), 3, 0.9)]
    for i, t in enumerate(lobby_w):
        add(f"wallpaper{i}", t)
    add("carpet_tan", carpet(c, (0.50, 0.42, 0.24), 0.6))
    add("carpet_brown", carpet(c, (0.40, 0.32, 0.20), 0.8))
    add("carpet_grey", carpet(c, (0.30, 0.32, 0.36), 0.4, 2))
    add("carpet_red", carpet(c, (0.42, 0.06, 0.07), 0.4, 1))
    add("carpet_green", carpet(c, (0.10, 0.22, 0.14), 0.4, 1, (0.55, 0.45, 0.2)))
    add("ceil_tile", ceil_tile(c))
    add("ceil_tile_office", ceil_tile(c, (0.86, 0.86, 0.84), 4, 0.25))
    add("ceil_plaster", plaster(c, (0.78, 0.74, 0.66), (0.78, 0.74, 0.66), 0.2))
    add("panel0", ceil_panel(c, kind=0))
    add("panel1", ceil_panel(c, kind=1))
    add("panel2", ceil_panel(c, (0.40, 0.40, 0.40), kind=2))
    add("skylight", ceil_panel(c, (0.86, 0.90, 0.90), kind=3))
    add("concrete_wall", concrete(c, 0.42, 0.02, 0.6, 2, 0.6))
    add("concrete_floor", concrete(c, 0.36, 0.0, 0.8, 1, 0.0))
    add("concrete_lines", concrete(c, 0.34, 0.0, 0.6, 1, 0.0, stripe=(0.85, 0.72, 0.1)))
    add("concrete_ceil", concrete(c, 0.38, 0.0, 0.3, 4, 0.0))
    add("concrete_dark", concrete(c, 0.22, 0.03, 0.8, 2, 0.8))
    add("brick_red", brick(c))
    add("brick_dark", brick(c, (0.26, 0.20, 0.18), (0.30, 0.29, 0.28), soot=0.8))
    pw = pipe_wall(c)
    add("pipe_wall", pw)
    add("pipe_wall_valve", pipe_wall(c, valve=True))
    add("pipe_wall_lamp", pipe_wall(c, lamp=True))
    add("pipe_bundle", pipe_wall(c, (0.12, 0.12, 0.12), rust=0.8))
    add("grate", grate(c))
    add("metal_plate", metal_plate(c))
    off = plaster(c, (0.82, 0.82, 0.80))
    add("office_wall", off)
    add("office_wall2", plaster(c, (0.72, 0.76, 0.78), wear=0.7))
    add("office_window", plaster(c, (0.82, 0.82, 0.80), window=True))
    add("office_door", door(c, off, (0.45, 0.47, 0.50), (0.25, 0.26, 0.28), 1))
    add("cubicle", cubicle(c))
    add("cubicle_top", plaster(c, (0.5, 0.5, 0.52), (0.5, 0.5, 0.52), 0.0))
    add("desk_top", wood(c, (0.58, 0.46, 0.30)))
    add("desk_side", wood(c, (0.40, 0.31, 0.20), panel=True))
    hw = hotel_wall(c)
    add("hotel_wall", hw)
    add("hotel_wall_green", hotel_wall(c, (0.08, 0.20, 0.14), (0.50, 0.42, 0.18)))
    add("hotel_sconce", hotel_wall(c, sconce=True))
    add("hotel_door", door(c, hw, (0.24, 0.13, 0.07), (0.13, 0.07, 0.04), 0))
    add("wood_dark", wood(c, (0.18, 0.10, 0.06), planks=6))
    add("wood_floor", wood(c, (0.33, 0.20, 0.11), planks=8))
    add("tile_white", tiles(c))
    add("tile_band", tiles(c, band_c=(0.10, 0.45, 0.62)))
    add("tile_floor", tiles(c, (0.80, 0.86, 0.86), n=4))
    add("tile_small", tiles(c, (0.70, 0.84, 0.86), (0.50, 0.60, 0.62), n=16))
    hosp = plaster(c, (0.60, 0.72, 0.62), (0.30, 0.34, 0.32), 0.8, top=(0.80, 0.80, 0.74))
    add("hosp_wall", hosp)
    add("hosp_door", door(c, hosp, (0.62, 0.70, 0.64), (0.40, 0.44, 0.42), 2))
    add("hosp_alarm", sign_wall(c, hosp, 1))
    add("hosp_exit", sign_wall(c, hosp, 0))
    add("lino", lino(c))
    add("lino_dark", lino(c, (0.30, 0.33, 0.30), (0.26, 0.28, 0.26)))
    add("party0", party_wall(c, 0))
    add("party1", party_wall(c, 1))
    add("party2", party_wall(c, 2))
    add("party_face", party_wall(c, 1, face=True))
    add("checker", checker(c))
    add("checker_color", checker(c, (0.92, 0.80, 0.20), (0.80, 0.15, 0.20)))
    add("cloth0", cloth(c, 0))
    add("cloth1", cloth(c, 1))
    add("bookshelf", bookshelf(c))
    add("bookshelf2", bookshelf(c, (0.14, 0.08, 0.05), 7))
    add("shelf_top", shelf_top(c))
    add("chainlink", chainlink(c))
    cd = concrete(c, 0.22, 0.03, 0.8, 2, 0.8)
    add("elec_panel", elec_panel(c, cd))
    add("conc_beacon", sign_wall(c, cd, 2))
    add("conc_alarm", sign_wall(c, cd, 1))
    add("crate", crate(c))
    add("crate_top", wood(c, (0.44, 0.31, 0.17), planks=5))
    add("hazard", hazard(c))
    dark_w = wallpaper(c, (0.34, 0.33, 0.27), 0, 1.0, (0.12, 0.10, 0.08))
    add("wallpaper_dark", dark_w)
    blue_w = wallpaper(c, (0.30, 0.40, 0.46), 3, 1.0, (0.12, 0.12, 0.14))
    add("wallpaper_blue", blue_w)
    add("lobby_doorb", door(c, blue_w, (0.20, 0.24, 0.26), (0.10, 0.12, 0.14), 0))
    # doors for the walls of the other levels
    add("steel_door", door(c, cd, (0.22, 0.24, 0.26), (0.12, 0.12, 0.13), 1))
    add("conc_door", door(c, T[names["concrete_wall"]], (0.30, 0.32, 0.34), (0.16, 0.16, 0.17), 1))
    add("pipe_door", door(c, pw, (0.22, 0.13, 0.08), (0.10, 0.08, 0.07), 1))
    add("tile_door", door(c, T[names["tile_white"]], (0.72, 0.80, 0.80), (0.48, 0.55, 0.56), 1))
    add("party_door", door(c, T[names["party0"]], (0.85, 0.25, 0.45), (0.95, 0.85, 0.30), 0))
    add("wood_door", door(c, T[names["wood_dark"]], (0.30, 0.17, 0.09), (0.12, 0.07, 0.04), 0))
    for i, t in enumerate(lobby_w + [dark_w]):
        nm = f"{i}" if i < 4 else "d"
        add(f"lobby_door{nm}", door(c, t, (0.52, 0.42, 0.24) if i < 4 else (0.2, 0.18, 0.14), (0.30, 0.24, 0.14) if i < 4 else (0.1, 0.09, 0.07), 0))
        words = [("HELP", "NO/EXIT"), ("RUN", "IT/SEES/YOU"), ("DONT/LOOK", "=)"), ("LEVEL/O", "HELP"), ("RUN", "NO/EXIT")][i]
        for j, w in enumerate(words):
            add(f"scrawl{nm}{j}", scrawl(c, t, w, (0.20, 0.02, 0.02) if j == 0 else (0.03, 0.03, 0.03), 0.28 if "/" in w else 0.36))
    # outdoors and the newer indoor levels
    add("grass", grass(c))
    add("grass_dry", grass(c, (0.30, 0.31, 0.13), 0.8))
    add("dirt", dirt(c))
    add("asphalt", asphalt(c))
    add("sidewalk", concrete(c, 0.48, 0.01, 0.5, 2, 0.0))
    add("wheat_side", crop_side(c))
    add("wheat_top", crop_top(c))
    add("corn_side", crop_side(c, (0.36, 0.42, 0.14), (0.09, 0.15, 0.05), heads=False))
    add("corn_top", crop_top(c, (0.30, 0.38, 0.13)))
    for nm, base in (("siding", (0.72, 0.72, 0.68)), ("siding_dark", (0.36, 0.40, 0.42)), ("barn", (0.42, 0.13, 0.10))):
        add(nm, siding(c, base, 9, 0.5 if nm != "barn" else 0.9))
        cell = siding(c, base, 18, 0.5)
        if nm != "barn":
            add(nm + "_window", wall_window(c, cell))
            add(nm + "_window_lit", wall_window(c, cell, lit=True))
        add(nm + "_door", door(c, cell, (0.30, 0.18, 0.10) if nm != "barn" else (0.30, 0.10, 0.08), (0.80, 0.80, 0.76) if nm != "barn" else (0.75, 0.72, 0.65), 0, top=0.30))
    add("fence", fence(c))
    add("hedge", hedge(c))
    add("facade0", facade(c))
    add("facade1", facade(c, (0.46, 0.33, 0.26), style=1))
    add("facade2", facade(c, (0.30, 0.33, 0.37)))
    add("facade_lit", facade(c, (0.40, 0.38, 0.36), lit=0.5))
    city_w = concrete(c, 0.40, 0.01, 0.4, 0, 0.5)
    add("city_wall", city_w)
    add("city_door", door(c, city_w, (0.16, 0.18, 0.20), (0.10, 0.10, 0.11), 1, top=0.28))
    for i, sg in enumerate(((0.95, 0.20, 0.18), (0.20, 0.65, 1.0), (0.30, 1.0, 0.50))):
        add(f"shop{i}", shopfront(c, city_w, sg))
    mall_w = plaster(c, (0.84, 0.82, 0.76), (0.60, 0.58, 0.52), 0.3)
    add("mall_wall", mall_w)
    for i, sg in enumerate(((1.0, 0.25, 0.55), (0.25, 0.85, 1.0), (1.0, 0.75, 0.20))):
        add(f"mall_shop{i}", shopfront(c, mall_w, sg))
    add("mall_shutter", shutter(c, mall_w))
    add("mall_floor", tiles(c, (0.78, 0.74, 0.66), (0.50, 0.48, 0.44), n=4))
    apt = plaster(c, (0.50, 0.55, 0.43), (0.25, 0.22, 0.16), 0.9, top=(0.74, 0.70, 0.56))
    add("apt_wall", apt)
    add("apt_door", door(c, apt, (0.34, 0.22, 0.13), (0.20, 0.14, 0.09), 0))
    add("container", corrugated(c))
    add("container_blue", corrugated(c, (0.12, 0.22, 0.34)))
    white = siding(c, (0.93, 0.93, 0.90), 9, 0.15)
    add("white_wall", white)
    wcell = siding(c, (0.93, 0.93, 0.90), 18, 0.15)
    add("white_window", wall_window(c, wcell, y0=0.34, y1=0.66, frame=(0.30, 0.34, 0.40)))
    add("white_door", door(c, wcell, (0.30, 0.20, 0.14), (0.86, 0.86, 0.82), 0, top=0.24))
    add("roof", shingles(c))
    add("picket", picket(c))
    add("bg_hills", backdrop_hills(c))
    add("sky_clouds", sky(c))
    add("sky_stars", sky(c, stars=True))
    n_surf = len(T)
    # sprites
    add("s_figure", s_figure(c))
    add("s_figure2", s_figure(c, eyes=False, lean=1.0, arms=1.6))
    add("s_smiler", s_smiler(c))
    add("s_hound", s_hound(c))
    add("s_party", s_party(c))
    add("s_faceless", s_faceless(c))
    add("s_horde", s_horde(c))
    add("s_balloon_r", s_balloon(c, (0.90, 0.10, 0.12)))
    add("s_balloon_b", s_balloon(c, (0.12, 0.35, 0.90)))
    add("s_balloon_y", s_balloon(c, (0.95, 0.80, 0.10)))
    add("s_chair", s_chair(c))
    add("s_plant", s_plant(c))
    add("s_lamp", s_lamp(c))
    add("s_sign", s_sign(c))
    add("s_cone", s_cone(c))
    add("s_barrel", s_barrel(c))
    add("s_barrel_rust", s_barrel(c, (0.35, 0.18, 0.08)))
    add("s_cooler", s_cooler(c))
    add("s_gurney", s_gurney(c))
    add("s_float", s_float(c))
    add("s_bulb", s_bulb(c))
    add("s_column", s_column(c))
    add("s_column_conc", s_column(c, (0.42, 0.42, 0.42), tiled=False))
    add("s_pipe", s_column(c, (0.30, 0.20, 0.12), tiled=False))
    add("s_boxes", s_boxes(c))
    add("s_exit", s_exit(c))
    add("s_tree", s_tree(c))
    add("s_pine", s_tree(c, leaves=True))
    add("s_lamppost", s_lamppost(c))
    add("s_lamphead", s_lamphead(c))
    add("s_pole", s_pole(c))
    add("s_scarecrow", s_scarecrow(c))
    add("s_haybale", s_haybale(c))
    add("s_bed", s_bed(c))
    add("s_tv", s_tv(c))
    add("s_dresser", s_dresser(c))
    add("s_bench", s_bench(c))
    add("s_oak", s_oak(c))
    add("s_bush", s_bush(c))
    add("s_watertower", s_watertower(c))
    add("s_scream", s_scream(c))
    add("t_died", t_text(c, "YOU DIED"))

    flat, base, off_ = [], [], 0
    for t in T:
        m = t[None]
        for l in range(NMIP):
            n = TS >> l
            flat.append((m[0].permute(1, 2, 0).reshape(n * n, 4) * 255 + 0.5).clamp(0, 255).to(torch.uint8))
            base.append(off_)
            off_ += n * n
            m = F.avg_pool2d(m, 2)
    return {"atlas": torch.cat(flat, 0), "base": torch.tensor(base, device=dev, dtype=torch.long), "names": names,
            "n_surf": n_surf, "tex": T if keep else None}
