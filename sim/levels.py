"""The levels: materials, sprite types and one map generator per level.

A map is a G x G grid of 1 m cells that wraps around at its edges (walk far enough and you are back, but fog hides
it). A cell is open or holds a `kind`: a block with a vertical extent (a full wall, a low wall or box, a lintel or
beam that leaves room to walk under), a texture for its sides, top and underside. Floor and ceiling have a material
per cell. Generators run on the CPU with numpy; they are called once per episode.
"""
import math

import numpy as np

G = 64
NCH = 5         # light channels: 0 steady, 1 flicker, 2 dying, 3 pulse / blink, 4 sparks

LEVELS = ["lobby", "habitable", "pipes", "electrical", "office", "hotel", "lightsout", "poolrooms", "run", "fun",
          "flooded", "library", "fields", "suburbs", "city", "apartments", "mall", "hills"]
CALM = ("hills",)   # levels in which nothing hunts the player
TITLES = {
    "lobby": "Level 0, The Lobby", "habitable": "Level 1, Habitable Zone", "pipes": "Level 2, Pipe Dreams",
    "electrical": "Level 3, Electrical Station", "office": "Level 4, Abandoned Office", "hotel": "Level 5, Terror Hotel",
    "lightsout": "Level 6, Lights Out", "poolrooms": "Level 37, The Poolrooms", "run": "Level !, Run For Your Life",
    "fun": "Level Fun", "flooded": "Level 7, Thalassophobia", "library": "The Endless Library",
    "fields": "Level 10, The Bumper Crop", "suburbs": "Level 9, The Darkened Suburbs", "city": "Level 11, The Endless City",
    "apartments": "Level 13, The Infinite Apartments", "mall": "The Infinite Mall", "hills": "Level 94, Motion",
}
ENT_RATE = 5.0      # every level's spawn rates are multiplied by this (entities used to be rare)
HUNT_RATE = 1.6     # ... and once more those of the entities that come for the player

# ------------------------------------------------------------------ materials
# name: (texture, metres per repeat across, metres per repeat up (0: stretch from floor to ceiling), glow strength,
#        glow colour, glow follows the map's lamp colour, light channel of the glow, gloss, water (1 pool, 2 dark), puddles)
MATS = {}


def _m(name, tex=None, size=2.0, v=None, em=0.0, emcol=(1, 1, 1), env=False, ch=0, gloss=0.0, water=0, puddle=False):
    MATS[name] = dict(tex=tex or name, size=size, v=size if v is None else v, em=em, emcol=emcol, env=env, ch=ch,
                      gloss=gloss, water=water, puddle=puddle)


for _i in range(4):
    _m(f"wallpaper{_i}", v=0)
_m("wallpaper_dark", v=0); _m("wallpaper_blue", v=0)
for _i in ("0", "1", "2", "3", "d"):
    _m(f"lobby_door{_i}", size=1, v=0); _m(f"scrawl{_i}0", size=1, v=0); _m(f"scrawl{_i}1", size=1, v=0)
_m("carpet_tan"); _m("carpet_brown"); _m("carpet_grey"); _m("carpet_red"); _m("carpet_green")
_m("ceil_tile"); _m("ceil_tile_office"); _m("ceil_plaster", size=3)
_m("panel0", size=1, em=3.2, env=True); _m("panel1", size=1, em=3.2, env=True); _m("panel2", size=1, em=3.2, env=True)
_m("skylight", size=1, em=3.6, env=True)
_m("concrete_wall", size=3); _m("concrete_floor", size=4); _m("concrete_lines", size=4); _m("concrete_ceil", size=4)
_m("concrete_dark", size=3)
_m("concrete_wet", "concrete_floor", size=4, gloss=0.9, puddle=True)
_m("brick_red", size=1.1, v=0.95); _m("brick_dark", size=1.1, v=0.95)
_m("pipe_wall", v=0); _m("pipe_wall_valve", size=1, v=0)
_m("pipe_wall_lamp", size=1, v=0, em=2.6, emcol=(1.0, 0.5, 0.22))
_m("pipe_wall_lamp_f", "pipe_wall_lamp", size=1, v=0, em=2.6, emcol=(1.0, 0.5, 0.22), ch=1)
_m("pipe_bundle", size=2, v=1); _m("grate", size=1, em=0.9, emcol=(1.0, 0.4, 0.1)); _m("metal_plate")
_m("office_wall", size=3, v=0); _m("office_wall2", size=3, v=0); _m("office_window", size=1, v=0)
_m("office_door", size=1, v=0); _m("cubicle", size=1, v=1.45); _m("cubicle_top", size=1); _m("desk_top", size=1)
_m("desk_side", size=1, v=0.75)
_m("hotel_wall", v=0); _m("hotel_wall_green", v=0); _m("hotel_door", size=1, v=0)
_m("hotel_sconce", size=1, v=0, em=2.6, emcol=(1.0, 0.75, 0.45))
_m("hotel_sconce_f", "hotel_sconce", size=1, v=0, em=2.6, emcol=(1.0, 0.75, 0.45), ch=1)
_m("wood_dark"); _m("wood_floor", gloss=0.12)
_m("tile_white", size=1.6); _m("tile_band", size=1.6); _m("tile_floor", size=1.6, gloss=0.3); _m("tile_small", size=1.6)
_m("water", "tile_small", size=1.6, gloss=1.0, water=1)
_m("water_dark", "concrete_dark", size=3, gloss=1.0, water=2)
_m("hosp_wall", v=0); _m("hosp_door", size=1, v=0)
_m("hosp_alarm", size=1, v=0, em=3.0, emcol=(1.0, 0.10, 0.05), ch=3)
_m("hosp_exit", size=1, v=0, em=2.0, emcol=(0.3, 1.0, 0.5))
_m("lino", gloss=0.35); _m("lino_dark", gloss=0.35)
_m("party0", v=0); _m("party1", v=0); _m("party2", v=0); _m("party_face", size=1, v=0)
_m("checker", gloss=0.25); _m("checker_color", gloss=0.25); _m("cloth0", size=1); _m("cloth1", size=1)
_m("bookshelf", v=2.3); _m("bookshelf2", v=2.3); _m("shelf_top", size=1)
_m("chainlink", size=1, v=2); _m("elec_panel", size=1, v=0, em=1.5)
_m("conc_beacon", size=1, v=0, em=3.0, emcol=(1.0, 0.65, 0.05), ch=3)
_m("conc_alarm", size=1, v=0, em=3.0, emcol=(1.0, 0.10, 0.05), ch=3)
_m("crate", size=1, v=1); _m("crate_top", size=1); _m("hazard", size=2, v=0.9)
# doors that open (a cell wide, stretched to the height of their wall)
for _n in ("lobby_doorb", "steel_door", "conc_door", "pipe_door", "tile_door", "party_door", "wood_door", "apt_door",
           "city_door", "mall_shutter", "siding_door", "siding_dark_door", "barn_door"):
    _m(_n, size=1, v=0)
# outdoors
_m("grass", size=3); _m("grass_dry", size=3); _m("dirt", size=3); _m("asphalt", size=4); _m("sidewalk", size=2)
_m("wheat_side", size=1.5, v=1.1); _m("wheat_top", size=2); _m("corn_side", size=1.5, v=2.7); _m("corn_top", size=2)
_m("siding", size=1.5, v=1.5); _m("siding_dark", size=1.5, v=1.5); _m("barn", size=1.5, v=1.5)
_m("siding_window", size=1, v=0); _m("siding_dark_window", size=1, v=0)
_m("siding_window_lit", size=1, v=0, em=0.8, emcol=(1.0, 0.75, 0.42))
_m("siding_dark_window_lit", size=1, v=0, em=0.8, emcol=(1.0, 0.75, 0.42))
_m("fence", size=1, v=1.2); _m("hedge", size=2, v=1.5)
_m("facade0", size=4, v=3.2); _m("facade1", size=4, v=3.2); _m("facade2", size=4, v=3.2)
_m("facade_lit", size=4, v=3.2, em=0.8, emcol=(1.0, 0.80, 0.48))
_m("city_wall", size=3, v=3)
_m("shop0", size=2, v=0, em=1.3, emcol=(1.0, 0.22, 0.18)); _m("shop1", size=2, v=0, em=1.3, emcol=(0.22, 0.68, 1.0))
_m("shop2", size=2, v=0, em=1.3, emcol=(0.30, 1.0, 0.50))
_m("container", size=2.5, v=2.5); _m("container_blue", size=2.5, v=2.5)
_m("white_wall", size=1.5, v=1.35); _m("white_window", size=1, v=0); _m("white_door", size=1, v=0)
_m("roof", size=2, v=1.6); _m("picket", size=1, v=1.0)
# the apartments and the mall
_m("apt_wall", size=3, v=0); _m("mall_wall", size=3, v=0); _m("mall_floor", size=2, gloss=0.45)
_m("mall_shop0", size=2, v=0, em=1.3, emcol=(1.0, 0.25, 0.55)); _m("mall_shop1", size=2, v=0, em=1.3, emcol=(0.25, 0.85, 1.0))
_m("mall_shop2", size=2, v=0, em=1.3, emcol=(1.0, 0.75, 0.20))
MAT_ID = {n: i for i, n in enumerate(MATS)}

# ------------------------------------------------------------------ sprites
# name: (texture, width m, height m (0: floor to ceiling), base height m (-1: hangs from the ceiling), glow 0..1, blocks the cell)
SPRITES = {
    "figure": ("s_figure", 1.0, 2.3, 0, 0.0, False), "figure2": ("s_figure2", 1.0, 2.3, 0, 0.0, False),
    "smiler": ("s_smiler", 0.9, 0.9, 1.2, 1.0, False), "hound": ("s_hound", 1.6, 1.1, 0, 0.0, False),
    "party": ("s_party", 1.1, 2.3, 0, 0.2, False), "faceless": ("s_faceless", 1.0, 1.9, 0, 0.0, False),
    "horde": ("s_horde", 2.6, 2.2, 0, 0.35, False),
    "balloon_r": ("s_balloon_r", 0.5, 1.9, 0, 0.2, False), "balloon_b": ("s_balloon_b", 0.5, 1.9, 0, 0.2, False),
    "balloon_y": ("s_balloon_y", 0.5, 1.9, 0, 0.2, False),
    "chair": ("s_chair", 0.7, 1.1, 0, 0.0, True), "plant": ("s_plant", 0.9, 1.5, 0, 0.0, True),
    "lamp": ("s_lamp", 0.6, 1.7, 0, 0.85, True), "sign": ("s_sign", 0.6, 0.7, 0, 0.0, False),
    "cone": ("s_cone", 0.45, 0.7, 0, 0.0, False), "barrel": ("s_barrel", 0.65, 0.95, 0, 0.0, True),
    "barrel_rust": ("s_barrel_rust", 0.65, 0.95, 0, 0.0, True), "cooler": ("s_cooler", 0.45, 1.35, 0, 0.0, True),
    "gurney": ("s_gurney", 2.0, 1.05, 0, 0.0, True), "float": ("s_float", 0.9, 0.3, 0, 0.0, False),
    "bulb": ("s_bulb", 0.3, 0.9, -1, 1.0, False), "column": ("s_column", 0.7, 0, 0, 0.0, True),
    "column_conc": ("s_column_conc", 0.7, 0, 0, 0.0, True), "pipe": ("s_pipe", 0.22, 0, 0, 0.0, False),
    "boxes": ("s_boxes", 0.9, 1.1, 0, 0.0, True), "exit": ("s_exit", 0.7, 0.5, -1, 0.9, False),
    "tree": ("s_tree", 3.0, 5.2, 0, 0.0, True), "pine": ("s_pine", 2.6, 5.2, 0, 0.0, True),
    "lamppost": ("s_lamppost", 1.0, 4.2, 0, 0.0, False), "lamphead": ("s_lamphead", 0.7, 0.35, 3.82, 1.0, False),
    "pole": ("s_pole", 1.6, 7.0, 0, 0.0, False), "scarecrow": ("s_scarecrow", 1.8, 2.3, 0, 0.0, False),
    "haybale": ("s_haybale", 1.5, 1.4, 0, 0.0, True), "bed": ("s_bed", 2.1, 1.0, 0, 0.0, True),
    "tv": ("s_tv", 1.0, 1.2, 0, 0.7, True), "dresser": ("s_dresser", 1.2, 1.0, 0, 0.0, True),
    "bench": ("s_bench", 1.7, 0.9, 0, 0.0, True), "mannequin": ("s_faceless", 0.9, 1.8, 0, 0.0, False),
    "tall": ("s_figure", 0.9, 3.1, 0, 0.0, False),
    "oak": ("s_oak", 3.4, 4.6, 0, 0.0, True), "bush": ("s_bush", 1.2, 0.9, 0, 0.0, False),
    "watertower": ("s_watertower", 4.5, 9.0, 0, 0.0, True),
}
SPR_ID = {n: i + 1 for i, n in enumerate(SPRITES)}     # 0 = none

# entity types
E_NONE, E_WATCHER, E_STALKER, E_SMILER, E_HOUND, E_HORDE, E_PARTY = range(7)

NK = 32       # kinds per map, including 0 = open


class Map:
    def __init__(self, rng, name):
        self.rng, self.name = rng, name
        self.w = np.zeros((G, G), np.uint8)
        self.kinds = [dict(side=0, top=0, bot=0, y0=0.0, y1=-1.0, full=False, block=False, up=0, split=1e9, door=0, nxt=0, prv=0)]
        self.floor = np.zeros((G, G), np.int16)
        self.ceil = np.zeros((G, G), np.int16)
        self.emit = np.zeros((G, G), np.float32)
        self.echan = np.zeros((G, G), np.uint8)
        self.lights, self.props, self.ents = [], [], []
        self.H = 2.7
        self.fog, self.fogcol, self.amb = 0.02, (0.1, 0.1, 0.1), (0.03, 0.03, 0.03)
        self.lcol = (1.0, 1.0, 1.0)             # colour of panels and of the light they give
        self.grime, self.staincol = 0.5, (0.45, 0.38, 0.28)
        self.pulse = (1.2, 0.1, 1.0, 0)         # period s, low, high, square wave?
        self.spark = 0.02
        self.wave_rate = 0.5                    # blackout waves per minute
        self.rates = {}                         # entity type -> spawns per minute
        self.tint_j = 0.07
        self.flash = 0.08                       # how much of the time the explorer keeps the flashlight on
        self.axis = None                        # preferred direction of travel (Level !)
        self.n_party = 0
        self.red_rate, self.red_perm = 0.0, False    # the lamps turning red for a while (per minute); red from the start
        self.outdoor = False                    # no ceiling: a sky, and walls end at their own height
        self.sky = ("sky_clouds", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0)   # texture, colour at the horizon and overhead, cloud contrast, stars
        self.clouds = 0.0                       # white clouds in a clear sky (0: the sky is only shaded by its texture)
        self.backdrop = None                    # a texture of what stands on the horizon
        self.hmap = None                        # ground height per cell [G, G] where the ground is not flat
        self.sun = (1.0, 0.0, 0.0)              # where the light on outdoor walls comes from (x, z), and how strong it is
        self.doors = []                         # (x, z, normal x, z, the room's centre x, z, what waits inside)
        self.haunt = (0.22, 0.05, 0.06)         # share of rooms with a figure standing in them, something that runs out, eyes in the dark
        self.used = np.zeros((G, G), bool)      # cells taken by free-standing rooms and what is around them
        self.inroom = np.zeros((G, G), bool)    # cells inside rooms that have a door

    # kinds ---------------------------------------------------------------
    def kind(self, side, top=None, bot=None, y0=0.0, y1=-1.0, block=None, full=None, up=None, split=0.0, door=0):
        """A block. Full walls (from the floor up, nothing is seen past them) reach the ceiling, or y1 when full=True is
        given with it (outdoors). Above `split` a full wall shows `up` instead of `side` (a shop under a facade)."""
        if full is None:
            full = y0 <= 0.0 and y1 < 0
        self.kinds.append(dict(side=MAT_ID[side], top=MAT_ID[top or side], bot=MAT_ID[bot or side], y0=y0, y1=y1,
                               full=full, block=(y0 < 1.85) if block is None else block, up=MAT_ID[up or side],
                               split=split if split > 0 else 1e9, door=door, nxt=0, prv=0))
        assert len(self.kinds) <= NK, "too many kinds in one map"
        return len(self.kinds) - 1

    def door(self, tex, wall, h=-1.0, up=None, split=0.0):
        """A door that E opens and closes; returns the kind of the closed door (a full wall showing `tex`). Opening
        takes it through three kinds with the leaf part of the way up, to a doorway under a lintel of `wall`.
        h: the height of the wall it stands in (-1: to the ceiling); with split, the door's own part of a taller wall."""
        y1 = split if split > 0 else h
        ch = [self.kind(tex, y1=h, full=True, up=up, split=split, door=1)]
        ch += [self.kind(tex, y0=y, y1=y1, door=2) for y in (0.5, 1.0, 1.55)]
        ch += [self.kind(wall, y0=2.05, y1=y1, door=3)]
        for a, b in zip(ch[:-1], ch[1:]):
            self.kinds[a]["nxt"], self.kinds[b]["prv"] = b, a
        return ch[0]

    def open_of(self, k):
        """The open doorway of a door kind."""
        while self.kinds[k]["nxt"]:
            k = self.kinds[k]["nxt"]
        return k

    def put_door(self, x, z, k, nx, nz, cx, cz, opened=False):
        """Door of kind k at cell (x, z), facing (nx, nz); (cx, cz): the middle of the room behind it."""
        self.w[z % G, x % G] = self.open_of(k) if opened else k
        u, (p0, p1, p2) = self.rng.random(), self.haunt
        h = 0 if opened else (1 if u < p0 else 2 if u < p0 + p1 else 3 if u < p0 + p1 + p2 else 0)
        self.doors.append((x % G + 0.5, z % G + 0.5, float(nx), float(nz), float(cx) % G, float(cz) % G, h))

    # drawing (everything wraps) ------------------------------------------
    def rect(self, x, z, sx, sz, k, a=None):
        a = self.w if a is None else a
        xs = np.arange(x, x + sx) % G
        zs = np.arange(z, z + sz) % G
        a[np.ix_(zs, xs)] = k

    def seg(self, x, z, L, horiz, k):
        self.rect(x, z, L if horiz else 1, 1 if horiz else L, k)

    def get(self, x, z):
        return self.w[z % G, x % G]

    def room(self, x, z, sx, sz, k, gaps=2, gapk=0, gapw=(1, 2)):
        r = self.rng
        self.seg(x, z, sx, True, k); self.seg(x, z + sz - 1, sx, True, k)
        self.seg(x, z, sz, False, k); self.seg(x + sx - 1, z, sz, False, k)
        for _ in range(gaps):
            wd = int(r.integers(gapw[0], gapw[1] + 1))
            side = int(r.integers(4))
            if side < 2:
                gx = x + int(r.integers(1, max(2, sx - wd)))
                self.seg(gx, z if side == 0 else z + sz - 1, wd, True, gapk)
            else:
                gz = z + int(r.integers(1, max(2, sz - wd)))
                self.seg(x if side == 2 else x + sx - 1, gz, wd, False, gapk)

    def light(self, x, z, col, power, radius, ch=0, panel=None, emit=1.0):
        """A light above the centre of cell (x, z). `panel`: ceiling material that shows it."""
        x, z = x % G, z % G
        if panel is not None:
            self.ceil[z, x] = MAT_ID[panel]
            self.emit[z, x] = emit
            self.echan[z, x] = ch
        if power > 0:
            self.lights.append((x + 0.5, z + 0.5, col[0], col[1], col[2], power, radius, ch))

    def prop(self, x, z, name):
        self.props.append((float(x), float(z), SPR_ID[name]))

    def open_cells(self):
        blk = np.array([k["block"] for k in self.kinds])[self.w]
        return ~blk

    def lattice_lights(self, sx, sz, panel, col, power, radius, dead=0.1, f1=0.05, f2=0.03, blobs=0, only=None, jit=0):
        r = self.rng
        ox, oz = int(r.integers(sx)), int(r.integers(sz))
        dark = np.zeros((G, G), bool)
        zz, xx = np.mgrid[0:G, 0:G]
        for _ in range(blobs):
            cx, cz, rad = r.integers(G), r.integers(G), r.uniform(5, 12)
            dx = (xx - cx + G // 2) % G - G // 2
            dz = (zz - cz + G // 2) % G - G // 2
            dark |= dx * dx + dz * dz < rad * rad
        for z in range(oz, G, sz):
            for x in range(ox, G, sx):
                xj, zj = (x + int(r.integers(-jit, jit + 1))) % G, (z + int(r.integers(-jit, jit + 1))) % G
                if self.w[zj, xj] != 0 or self.inroom[zj, xj] or (only is not None and not only[zj, xj]):
                    continue
                u = r.random()
                if dark[zj, xj] or u < dead:
                    self.light(xj, zj, col, 0.0, radius, 0, panel, emit=0.0)
                elif u < dead + f1:
                    self.light(xj, zj, col, power, radius, 1, panel)
                elif u < dead + f1 + f2:
                    self.light(xj, zj, col, power, radius, 2, panel)
                else:
                    self.light(xj, zj, col, power * r.uniform(0.85, 1.1), radius, 0, panel)

    def scatter(self, names, n, free=None, clear=1):
        """Put n props on random open cells (`clear`: also keep the 4 neighbours open)."""
        r = self.rng
        free = ((self.w == 0) & ~self.inroom) if free is None else free
        if clear:
            free = free & np.roll(free, 1, 0) & np.roll(free, -1, 0) & np.roll(free, 1, 1) & np.roll(free, -1, 1)
        zs, xs = np.nonzero(free)
        if len(zs) == 0:
            return
        for i in r.choice(len(zs), size=min(n, len(zs)), replace=False):
            nm = names[int(r.integers(len(names)))]
            self.prop(xs[i] + 0.5 + r.uniform(-0.25, 0.25), zs[i] + 0.5 + r.uniform(-0.25, 0.25), nm)

    def faces(self, k_from, k_to, p):
        """Turn a share p of the cells of kind k_from that touch an open cell into k_to."""
        op = self.w == 0
        near = np.roll(op, 1, 0) | np.roll(op, -1, 0) | np.roll(op, 1, 1) | np.roll(op, -1, 1)
        m = (self.w == k_from) & near & (self.rng.random((G, G)) < p)
        self.w[m] = k_to
        return m


def _fogc(r, c, lo, hi):
    f = r.uniform(lo, hi)
    return tuple(v * f * r.uniform(0.93, 1.07) for v in c)


def partitions(mp, kinds, n_seg, lens, n_rooms, room, n_blocks, n_pillars, gapk=0, gap_p=0.3):
    """Irregular open-plan maze: loose wall segments, room outlines with gaps, solid blocks, pillars."""
    r = mp.rng
    for _ in range(n_rooms):
        sx, sz = int(r.integers(room[0], room[1] + 1)), int(r.integers(room[0], room[1] + 1))
        mp.room(int(r.integers(G)), int(r.integers(G)), sx, sz, kinds[int(r.integers(len(kinds)))],
                gaps=int(r.integers(1, 4)), gapk=gapk if r.random() < gap_p else 0)
    for _ in range(n_seg):
        L = int(r.integers(lens[0], lens[1] + 1))
        x, z, hz = int(r.integers(G)), int(r.integers(G)), r.random() < 0.5
        k = kinds[int(r.integers(len(kinds)))]
        mp.seg(x, z, L, hz, k)
        if L >= 5 and r.random() < 0.5:
            o = int(r.integers(1, L - 2))
            mp.seg(x + (o if hz else 0), z + (0 if hz else o), int(r.integers(1, 3)), hz, gapk if r.random() < gap_p else 0)
    for _ in range(n_blocks):
        mp.rect(int(r.integers(G)), int(r.integers(G)), int(r.integers(2, 6)), int(r.integers(2, 5)), kinds[0])
    for _ in range(n_pillars):
        mp.w[r.integers(G), r.integers(G)] = kinds[0]


def lattice_maze(rng, n, loops):
    """A maze on an n x n grid of nodes that wraps around: eh[i, j] joins (i, j) and (i, j+1), ev[i, j] joins (i, j) and (i+1, j)."""
    eh, ev = np.zeros((n, n), bool), np.zeros((n, n), bool)
    seen = np.zeros((n, n), bool)
    stack = [(int(rng.integers(n)), int(rng.integers(n)))]
    seen[stack[0]] = True
    while stack:
        i, j = stack[-1]
        nb = [(d, (i + di) % n, (j + dj) % n) for d, (di, dj) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0)))]
        nb = [(d, a, b) for d, a, b in nb if not seen[a, b]]
        if not nb:
            stack.pop()
            continue
        d, a, b = nb[int(rng.integers(len(nb)))]
        if d == 0: eh[i, j] = True
        elif d == 1: ev[i, j] = True
        elif d == 2: eh[a, b] = True
        else: ev[a, b] = True
        seen[a, b] = True
        stack.append((a, b))
    eh |= rng.random((n, n)) < loops
    ev |= rng.random((n, n)) < loops
    return eh, ev


def carve(mp, eh, ev, width=1, wide_p=0.0):
    n = eh.shape[0]
    p = G // n
    r = mp.rng
    for i in range(n):
        for j in range(n):
            cz, cx = i * p + p // 2, j * p + p // 2
            wd = width + (1 if r.random() < wide_p else 0)
            mp.rect(cx - wd // 2, cz - wd // 2, wd, wd, 0)
            if eh[i, j]:
                mp.rect(cx, cz - wd // 2, p + 1, wd, 0)
            if ev[i, j]:
                mp.rect(cx - wd // 2, cz, wd, p + 1, 0)


def corridor_runs(mp):
    """For every open cell: is it a corridor running along x (1), along z (2), or neither (0)."""
    op = mp.w == 0
    wn, ws = ~np.roll(op, 1, 0), ~np.roll(op, -1, 0)
    ww, we = ~np.roll(op, 1, 1), ~np.roll(op, -1, 1)
    out = np.zeros((G, G), np.uint8)
    out[op & wn & ws & ~ww & ~we] = 1
    out[op & ww & we & ~wn & ~ws] = 2
    return out


# ------------------------------------------------------------------ rooms behind doors

def _room(mp, cx, cz, x, z, nx, nz, front=()):
    """The record of a room: its cells, its door, the cell one steps into, its middle."""
    ex, ez = (x - nx) % G, (z - nz) % G
    mx = (ex + _wr(cx - ex).mean()) % G + 0.5
    mz = (ez + _wr(cz - ez).mean()) % G + 0.5
    mp.inroom[cz, cx] = True
    return dict(cx=cx, cz=cz, door=(x % G, z % G), n=(nx, nz), entry=(ex, ez), mid=(mx, mz), front=list(front), lit=False)


def _wr(d):
    return (np.asarray(d) + G // 2) % G - G // 2


def rooms_in_solid(mp, solid, dk, n, depth=(3, 4), width=(3, 5), keep=(), roof=0, open_p=0.0):
    """Carve up to n rooms into solid wall. Each lies behind a cell that faces an open cell; that cell becomes a door
    of kind dk (already open for a share open_p of them). `solid`: kinds that may be carved and may stand as a
    room's walls; `keep`: kinds that may stand as walls only. roof: the kind the room's cells get instead of
    nothing (outdoors: the ceiling of the building). Returns the rooms (see _room)."""
    r, w = mp.rng, mp.w
    sol = np.isin(w, list(solid))
    okw = sol | np.isin(w, list(keep))
    op = w == 0
    cand = []
    for dz, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        zs, xs = np.nonzero(sol & np.roll(op, (-dz, -dx), (0, 1)))
        cand += [(int(x), int(z), dx, dz) for z, x in zip(zs, xs)]
    rooms = []
    for i in r.permutation(len(cand)):
        if len(rooms) >= n:
            break
        x, z, nx, nz = cand[i]
        if not sol[z, x]:
            continue
        d, wd = int(r.integers(depth[0], depth[1] + 1)), int(r.integers(width[0], width[1] + 1))
        o = int(r.integers(wd))
        kk, jj = np.meshgrid(np.arange(0, d + 2), np.arange(-1, wd + 1), indexing="ij")
        cx = (x - nx * kk + abs(nz) * (jj - o)) % G
        cz = (z - nz * kk + abs(nx) * (jj - o)) % G
        inner = (kk >= 1) & (kk <= d) & (jj >= 0) & (jj < wd)
        isdoor = (kk == 0) & (jj == o)
        if not (sol[cz, cx][inner].all() and okw[cz, cx][~inner & ~isdoor].all()):
            continue
        ix, iz = cx[inner], cz[inner]
        w[iz, ix] = roof
        sol[iz, ix] = okw[iz, ix] = False
        sol[z, x] = False
        fr = (kk == 0) & (jj >= 0) & (jj < wd) & ~isdoor
        rm = _room(mp, ix, iz, x, z, nx, nz, zip(cx[fr].tolist(), cz[fr].tolist()))
        mp.put_door(x, z, dk, nx, nz, rm["mid"][0], rm["mid"][1], opened=r.random() < open_p)
        rooms.append(rm)
    return rooms


def closed_rooms(mp, wall, dk, n, size=(4, 7), roof=0, open_p=0.0, tries=300):
    """Up to n free-standing rooms: an outline of `wall` (outer size from `size`) with one door of kind dk, emptied
    inside, and room to stand in front of the door."""
    r, rooms = mp.rng, []
    for _ in range(tries):
        if len(rooms) >= n:
            break
        sx, sz = int(r.integers(size[0], size[1] + 1)), int(r.integers(size[0], size[1] + 1))
        x0, z0 = int(r.integers(G)), int(r.integers(G))
        box = np.ix_(np.arange(z0 - 2, z0 + sz + 2) % G, np.arange(x0 - 2, x0 + sx + 2) % G)
        if mp.used[box].any():
            continue
        side = int(r.integers(4))
        if side < 2:
            x, z = x0 + int(r.integers(1, sx - 1)), (z0 if side == 0 else z0 + sz - 1)
            nx, nz = 0, (-1 if side == 0 else 1)
        else:
            x, z = (x0 if side == 2 else x0 + sx - 1), z0 + int(r.integers(1, sz - 1))
            nx, nz = (-1 if side == 2 else 1), 0
        mp.used[box] = True
        mp.rect(x0, z0, sx, sz, wall)
        mp.rect(x0 + 1, z0 + 1, sx - 2, sz - 2, roof)
        for k in (1, 2):
            mp.w[(z + nz * k) % G, (x + nx * k) % G] = 0
        zz, xx = np.meshgrid(np.arange(z0 + 1, z0 + sz - 1) % G, np.arange(x0 + 1, x0 + sx - 1) % G, indexing="ij")
        rm = _room(mp, xx.ravel(), zz.ravel(), x, z, nx, nz)
        mp.put_door(x, z, dk, nx, nz, rm["mid"][0], rm["mid"][1], opened=r.random() < open_p)
        rooms.append(rm)
    return rooms


def furnish(mp, rm, props=(), n=(1, 3), floor=None, ceil=None, lamp=None, lamp_p=0.5, panel=None):
    """Fill a room: floor and ceiling, in a share lamp_p of rooms a light (lamp: colour, power, radius, sprite or
    None, light channel), and props on cells off the way in."""
    r = mp.rng
    cx, cz = rm["cx"], rm["cz"]
    if floor:
        mp.floor[cz, cx] = MAT_ID[floor]
    if ceil:
        mp.ceil[cz, cx] = MAT_ID[ceil]
    (ex, ez), (nx, nz) = rm["entry"], rm["n"]
    way = {(ex, ez), ((ex - nx) % G, (ez - nz) % G)}
    cells = [c for c in zip(cx.tolist(), cz.tolist()) if c not in way]
    cells = [cells[i] for i in r.permutation(len(cells))]
    k = 0
    if lamp is not None and cells and r.random() < lamp_p:
        colr, power, rad, spr, ch = lamp
        x, z = cells[k]
        k += 1
        if spr:
            mp.prop(x + 0.5, z + 0.5, spr)
        mp.light(x, z, colr, power, rad, ch, panel=panel)
        rm["lit"] = True
    for _ in range(int(r.integers(n[0], n[1] + 1)) if props else 0):
        if k >= len(cells):
            break
        x, z = cells[k]
        k += 1
        mp.prop(x + 0.5 + r.uniform(-0.15, 0.15), z + 0.5 + r.uniform(-0.15, 0.15), props[int(r.integers(len(props)))])


WARM = (1.0, 0.75, 0.45)


# ------------------------------------------------------------------ the levels

def gen_lobby(mp, dark=False, flooded=False):
    r = mp.rng
    mp.H = r.uniform(2.5, 3.0)
    var = None
    if dark:
        ws, var = ["wallpaper_dark", "wallpaper_dark"], "d"
    elif flooded:
        ws = ["wallpaper_blue", "wallpaper_dark"]
    else:
        a = int(r.integers(4))
        ws, var = [f"wallpaper{a}", f"wallpaper{(a + 1 + int(r.integers(3))) % 4}"], str(a)
    A, Bk = mp.kind(ws[0]), mp.kind(ws[1])
    Lt = mp.kind(ws[0], bot="ceil_tile", y0=2.05)
    D = mp.door(f"lobby_door{var or 'b'}", ws[0])
    S0 = mp.kind(f"scrawl{var}0") if var else A
    S1 = mp.kind(f"scrawl{var}1") if var else A
    d = r.uniform(0.3, 1.0) * (0.7 if flooded else 1.0)
    kinds = [A, A, A, Bk] if r.random() < 0.6 else [A]
    partitions(mp, kinds, int(35 + 85 * d), (2, 9), int(3 + 8 * d), (4, 11), int(r.integers(2, 8)),
               int(r.integers(0, 26)), gapk=Lt, gap_p=0.45)
    if r.random() < 0.35:       # a hall of pillars
        x0, z0, s = int(r.integers(G)), int(r.integers(G)), int(r.integers(3, 6))
        sx, sz = int(r.integers(10, 24)), int(r.integers(10, 24))
        mp.rect(x0, z0, sx, sz, 0)
        for z in range(z0, z0 + sz, s):
            for x in range(x0, x0 + sx, s):
                mp.w[z % G, x % G] = A
    rooms = closed_rooms(mp, A, D, int(r.integers(3, 9)), (4, 7), open_p=0.1)
    mp.faces(A, S0, 0.006)
    mp.faces(A, S1, 0.005)
    mp.floor[:] = MAT_ID["water_dark" if flooded else ("carpet_brown" if dark or r.random() < 0.3 else "carpet_tan")]
    mp.ceil[:] = MAT_ID["ceil_tile"]
    if dark:
        mp.lattice_lights(int(r.integers(2, 5)), int(r.integers(3, 5)), "panel0", (1, 1, 1), 0.0, 7, dead=1.0)
        mp.amb, mp.fog, mp.fogcol = (0.0035, 0.0035, 0.0045), r.uniform(0.07, 0.11), (0.0, 0.0, 0.0)
        mp.rates = {E_SMILER: 2.2, E_STALKER: 0.5, E_WATCHER: 0.3}
        mp.flash, mp.wave_rate, mp.grime = 0.93, 0.0, 0.9
    elif flooded:
        mp.lcol = (0.55, 0.78, 1.0)
        mp.lattice_lights(int(r.integers(4, 7)), int(r.integers(4, 7)), "panel0", mp.lcol, r.uniform(0.6, 0.9), 7.5,
                          dead=r.uniform(0.3, 0.6), f1=0.06, f2=0.12, blobs=int(r.integers(0, 3)))
        mp.amb, mp.fog, mp.fogcol = (0.004, 0.009, 0.016), r.uniform(0.04, 0.065), (0.0, 0.008, 0.016)
        mp.rates = {E_WATCHER: 0.5, E_SMILER: 0.5, E_STALKER: 0.3}
        mp.flash, mp.grime, mp.staincol = 0.45, 0.9, (0.25, 0.35, 0.32)
    else:
        mp.lcol = (1.0, r.uniform(0.9, 0.97), r.uniform(0.62, 0.8))
        mp.lattice_lights(int(r.integers(2, 5)), int(r.integers(3, 5)), "panel0", mp.lcol, r.uniform(0.75, 1.15), 7.0,
                          dead=r.uniform(0.04, 0.4), f1=0.05, f2=0.03, blobs=int(r.integers(0, 3)))
        b = r.uniform(0.5, 1.2)
        mp.amb, mp.fog = (0.030 * b, 0.026 * b, 0.016 * b), r.uniform(0.012, 0.03)
        mp.fogcol = _fogc(r, (0.13, 0.105, 0.05), 0.5, 1.0)
        mp.rates = {E_WATCHER: 0.5, E_STALKER: 0.25, E_HOUND: 0.2}
        mp.grime = r.uniform(0.4, 1.0)
        mp.red_rate, mp.red_perm = 0.25, r.random() < 0.06
        mp.scatter(["chair", "boxes", "sign"], int(r.integers(0, 8)))
    for rm in rooms:
        furnish(mp, rm, ["chair", "boxes", "sign"], (0, 2), lamp=(mp.lcol, 0.8, 5.0, None, int(r.integers(2))),
                lamp_p=0.0 if dark else (0.3 if flooded else 0.45), panel="panel0")


def gen_habitable(mp):
    r = mp.rng
    mp.H = r.uniform(3.2, 4.2)
    C = mp.kind("concrete_wall")
    B = mp.kind("hazard", top="concrete_floor", y1=0.9)
    K = mp.kind("crate", top="crate_top", y1=1.0)
    K2 = mp.kind("crate", top="crate_top", y1=2.0)
    BEAM = mp.kind("concrete_wall", bot="concrete_ceil", y0=mp.H - 0.5)
    CD_ = mp.door("conc_door", "concrete_wall")
    p = int(r.integers(5, 9))
    ox, oz = int(r.integers(p)), int(r.integers(p))
    round_cols = r.random() < 0.4
    for z in range(oz, G - p + 1, p):
        for x in range(ox, G - p + 1, p):
            if r.random() < 0.9:
                if round_cols:
                    mp.prop(x + 0.5, z + 0.5, "column_conc")
                else:
                    mp.w[z, x] = C
    for _ in range(int(r.integers(3, 9))):
        L, hz = int(r.integers(8, 30)), r.random() < 0.5
        x, z = int(r.integers(G)), int(r.integers(G))
        mp.seg(x, z, L, hz, C)
        for o in range(int(r.integers(3, 7)), L, int(r.integers(5, 9))):
            mp.seg(x + (o if hz else 0), z + (0 if hz else o), 2, hz, 0)
    for _ in range(int(r.integers(3, 10))):
        mp.seg(int(r.integers(G)), int(r.integers(G)), int(r.integers(2, 7)), r.random() < 0.5, B)
    for _ in range(int(r.integers(5, 14))):
        x, z = int(r.integers(G)), int(r.integers(G))
        for _ in range(int(r.integers(1, 6))):
            if mp.get(x, z) == 0:
                mp.w[z % G, x % G] = K if r.random() < 0.65 else K2
            x, z = x + int(r.integers(-1, 2)), z + int(r.integers(-1, 2))
    rooms = closed_rooms(mp, C, CD_, int(r.integers(2, 7)), (4, 8), open_p=0.1)
    mp.floor[:] = MAT_ID["concrete_wet" if r.random() < 0.7 else "concrete_lines"]
    mp.ceil[:] = MAT_ID["concrete_ceil"]
    mp.lcol = (1.0, r.uniform(0.9, 1.0), r.uniform(0.75, 0.95))
    bulbs = r.random() < 0.35
    n0 = len(mp.lights)
    mp.lattice_lights(p, p, None if bulbs else "panel2", mp.lcol, r.uniform(1.1, 1.6), 9.0, dead=r.uniform(0.15, 0.45),
                      f1=0.10, f2=0.08, blobs=int(r.integers(0, 3)), jit=1)
    lit = np.zeros((G, G), bool)
    for (x, z, *_rest) in mp.lights[n0:]:
        lit[int(z), int(x)] = True
        if bulbs:
            mp.prop(x, z, "bulb")
    lit |= mp.emit > 0
    free = (mp.w == 0) & ~lit & (mp.ceil == MAT_ID["concrete_ceil"])
    if r.random() < 0.75:
        for z in range(oz, G, p):
            if r.random() < 0.7:
                mp.w[z % G][free[z % G]] = BEAM
        if r.random() < 0.5:
            for x in range(ox, G, p):
                mp.w[:, x % G][free[:, x % G] & (mp.w[:, x % G] == 0)] = BEAM
    mp.amb, mp.fog = (0.016, 0.018, 0.021), r.uniform(0.04, 0.07)
    mp.fogcol = _fogc(r, (0.085, 0.09, 0.095), 0.4, 1.0)
    mp.rates = {E_WATCHER: 0.5, E_HOUND: 0.35, E_STALKER: 0.15}
    mp.grime, mp.staincol, mp.flash = 0.8, (0.3, 0.3, 0.3), 0.2
    mp.scatter(["barrel", "barrel_rust", "cone", "boxes", "cone"], int(r.integers(4, 20)))
    for rm in rooms:
        furnish(mp, rm, ["barrel", "boxes", "barrel_rust", "cone"], (1, 4), lamp=(mp.lcol, 1.0, 5.5, "bulb", int(r.integers(2))), lamp_p=0.5)


def gen_pipes(mp):
    r = mp.rng
    mp.H = r.uniform(2.3, 2.6)
    W = mp.kind("pipe_wall")
    V = mp.kind("pipe_wall_valve")
    Lm = mp.kind("pipe_wall_lamp")
    Lf = mp.kind("pipe_wall_lamp_f")
    OH = mp.kind("pipe_bundle", y0=1.9)
    PD = mp.door("pipe_door", "pipe_wall")
    mp.w[:] = W
    n = 16 if r.random() < 0.7 else 8
    eh, ev = lattice_maze(r, n, r.uniform(0.08, 0.2))
    carve(mp, eh, ev, 1, wide_p=0.15)
    p = G // n
    for _ in range(int(r.integers(4, 12))):         # boiler rooms
        i, j = int(r.integers(n)), int(r.integers(n))
        s = int(r.integers(3, 6))
        mp.rect(j * p + p // 2 - s // 2, i * p + p // 2 - s // 2, s, s, 0)
    rooms = rooms_in_solid(mp, {W}, PD, int(r.integers(6, 18)), (1, 3) if n == 16 else (2, 4), (1, 3) if n == 16 else (2, 4), open_p=0.1)
    mp.floor[:] = MAT_ID["metal_plate" if r.random() < 0.6 else "concrete_floor"]
    mp.ceil[:] = MAT_ID["concrete_dark"]
    op = mp.w == 0
    mp.w[(mp.w == W) & (r.random((G, G)) < 0.06)] = V
    mp.w[op & (r.random((G, G)) < 0.10)] = OH
    mp.floor[op & (r.random((G, G)) < 0.06)] = MAT_ID["grate"]
    col = (1.0, r.uniform(0.35, 0.5), r.uniform(0.1, 0.22))
    zs, xs = np.nonzero(mp.w == 0)
    for i in r.permutation(len(zs))[: int(len(zs) * r.uniform(0.12, 0.25))]:
        z, x = int(zs[i]), int(xs[i])
        for dz, dx in r.permutation([(0, 1), (0, -1), (1, 0), (-1, 0)]):
            if mp.get(x + dx, z + dz) == W:
                u = r.random()
                fl = u < 0.15
                mp.w[(z + dz) % G, (x + dx) % G] = Lf if fl else Lm
                mp.light(x, z, col, r.uniform(1.1, 1.6), 6.0, 1 if fl else 0)
                break
    gz, gx = np.nonzero(mp.floor == MAT_ID["grate"])
    for z, x in zip(gz, gx):
        mp.light(int(x), int(z), (1.0, 0.4, 0.1), 0.25, 3.0)
    mp.amb, mp.fog = (0.034, 0.014, 0.008), r.uniform(0.035, 0.065)
    mp.fogcol = _fogc(r, (0.16, 0.07, 0.03), 0.5, 1.0)
    mp.rates = {E_STALKER: 0.5, E_SMILER: 0.4, E_WATCHER: 0.4}
    mp.grime, mp.staincol, mp.flash, mp.wave_rate = 0.7, (0.4, 0.25, 0.15), 0.3, 0.3
    mp.scatter(["pipe", "barrel_rust", "pipe"], int(r.integers(5, 25)), clear=0, free=(mp.w == 0) & ~mp.inroom)
    for rm in rooms:
        furnish(mp, rm, ["barrel_rust", "pipe"], (0, 2), lamp=(col, 1.0, 4.5, None, int(r.integers(2))), lamp_p=0.4)


def gen_electrical(mp):
    r = mp.rng
    mp.H = r.uniform(3.0, 3.5)
    BR = mp.kind("brick_dark" if r.random() < 0.6 else "brick_red")
    CD = mp.kind("concrete_dark")
    EP = mp.kind("elec_panel")
    CG = mp.kind("chainlink")
    BE = mp.kind("conc_beacon")
    OH = mp.kind("pipe_bundle", y0=mp.H - 0.7)
    SD = mp.door("steel_door", "concrete_dark")
    mp.w[:] = BR
    n = 8
    eh, ev = lattice_maze(r, n, r.uniform(0.2, 0.45))
    carve(mp, eh, ev, 2, wide_p=0.2)
    p = G // n
    for i in range(n):
        for j in range(n):
            if r.random() < 0.5:
                sx, sz = int(r.integers(4, 7)), int(r.integers(4, 7))
                x0, z0 = j * p + p // 2 - sx // 2, i * p + p // 2 - sz // 2
                mp.rect(x0, z0, sx, sz, 0)
                if r.random() < 0.5:        # a cage in the corner
                    cx, cz = x0 + int(r.integers(0, 2)) * (sx - 3), z0 + int(r.integers(0, 2)) * (sz - 3)
                    mp.room(cx, cz, 3, 3, CG, gaps=int(r.integers(0, 2)))
    rooms = rooms_in_solid(mp, {BR}, SD, int(r.integers(5, 13)), (2, 4), (2, 5), open_p=0.1)
    mp.faces(BR, CD, 0.25)
    mp.faces(CD, EP, 0.35)
    op = mp.w == 0
    mp.w[op & (r.random((G, G)) < 0.05)] = OH
    mp.floor[:] = MAT_ID["concrete_floor" if r.random() < 0.6 else "metal_plate"]
    mp.ceil[:] = MAT_ID["concrete_dark"]
    mp.lcol = (0.85, 0.92, 1.0)
    mp.lattice_lights(int(r.integers(4, 7)), int(r.integers(4, 7)), "panel1", mp.lcol, r.uniform(1.0, 1.4), 7.5,
                      dead=r.uniform(0.2, 0.5), f1=0.12, f2=0.12, jit=1)
    bm = mp.faces(CD, BE, 0.08)
    by = (1.0, 0.6, 0.05)
    for z, x in zip(*np.nonzero(bm)):
        for dz, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if mp.get(x + dx, z + dz) == 0:
                mp.light(int(x + dx), int(z + dz), by, 1.0, 6.0, 3)
                break
    ez, ex = np.nonzero(mp.w == EP)
    for i in r.permutation(len(ez))[: int(r.integers(3, 12))]:
        for dz, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if mp.get(ex[i] + dx, ez[i] + dz) == 0:
                mp.light(int(ex[i] + dx), int(ez[i] + dz), (0.6, 0.75, 1.0), 2.2, 7.0, 4)
                break
    mp.amb, mp.fog, mp.fogcol = (0.020, 0.022, 0.028), r.uniform(0.03, 0.05), (0.008, 0.01, 0.014)
    mp.pulse = (r.uniform(0.9, 1.6), 0.0, 1.0, 1)
    mp.spark = r.uniform(0.01, 0.05)
    mp.rates = {E_STALKER: 0.4, E_HOUND: 0.3, E_WATCHER: 0.4}
    mp.grime, mp.staincol, mp.flash = 0.7, (0.3, 0.28, 0.26), 0.45
    mp.scatter(["barrel", "barrel_rust", "boxes", "cone"], int(r.integers(3, 14)))
    for rm in rooms:
        furnish(mp, rm, ["barrel", "boxes", "barrel_rust"], (0, 3), lamp=(mp.lcol, 1.0, 5.0, None, int(r.choice([0, 1, 4]))),
                lamp_p=0.5, panel="panel1")


def gen_office(mp):
    r = mp.rng
    mp.H = r.uniform(2.5, 2.8)
    ow = "office_wall" if r.random() < 0.6 else "office_wall2"
    OW = mp.kind(ow)
    WIN = mp.kind("office_window")
    OD = mp.door("office_door", ow)
    CU = mp.kind("cubicle", top="cubicle_top", y1=1.45)
    DK = mp.kind("desk_side", top="desk_top", y1=0.75)
    Lt = mp.kind("office_wall", bot="ceil_tile_office", y0=2.05)
    d = r.uniform(0.3, 1.0)
    partitions(mp, [OW], int(18 + 35 * d), (4, 14), int(4 + 6 * d), (6, 14), int(r.integers(1, 5)), int(r.integers(0, 8)),
               gapk=Lt, gap_p=0.5)
    for _ in range(int(r.integers(2, 6))):          # cubicle farms
        x0, z0 = int(r.integers(G)), int(r.integers(G))
        nx, nz = int(r.integers(3, 6)), int(r.integers(2, 5))
        mp.rect(x0 - 1, z0 - 1, nx * 3 + 3, nz * 6 + 1, 0)
        for a in range(nz):
            zz = z0 + a * 6
            mp.seg(x0, zz + 2, nx * 3 + 1, True, CU)                    # spine
            for b in range(nx + 1):
                mp.seg(x0 + b * 3, zz, 5, False, CU)                    # dividers on both sides of the spine
            for b in range(nx):
                for dz_ in (1, 3):
                    if r.random() < 0.85:
                        mp.w[(zz + dz_) % G, (x0 + b * 3 + 1 + int(r.integers(2))) % G] = DK
                        if r.random() < 0.6:
                            mp.prop(x0 + b * 3 + 1.5 + r.uniform(-0.3, 0.3), zz + (0.5 if dz_ == 1 else 4.5), "chair")
    rooms = closed_rooms(mp, OW, OD, int(r.integers(4, 11)), (4, 8), open_p=0.15)
    for rm in rooms:            # a desk in most of them
        if r.random() < 0.7 and len(rm["cx"]) > 4:
            j = int(r.integers(len(rm["cx"])))
            if (int(rm["cx"][j]), int(rm["cz"][j])) != rm["entry"]:
                mp.w[rm["cz"][j], rm["cx"][j]] = DK
    mp.faces(OW, WIN, 0.07)
    mp.floor[:] = MAT_ID["carpet_grey"]
    mp.ceil[:] = MAT_ID["ceil_tile_office"]
    mp.lcol = (r.uniform(0.9, 1.0), 0.98, 1.0)
    mp.lattice_lights(int(r.integers(2, 4)), int(r.integers(2, 4)), "panel0", mp.lcol, r.uniform(0.8, 1.2), 6.5,
                      dead=r.uniform(0.05, 0.5), f1=0.05, f2=0.03, blobs=int(r.integers(0, 4)))
    b = r.uniform(0.5, 1.1)
    mp.amb, mp.fog = (0.026 * b, 0.028 * b, 0.031 * b), r.uniform(0.012, 0.03)
    mp.fogcol = _fogc(r, (0.09, 0.10, 0.11), 0.3, 1.0)
    mp.rates = {E_WATCHER: 0.6, E_STALKER: 0.3}
    mp.red_rate = 0.15
    mp.grime, mp.staincol, mp.flash = r.uniform(0.2, 0.7), (0.5, 0.45, 0.38), 0.1
    mp.scatter(["cooler", "plant", "boxes", "chair", "chair"], int(r.integers(6, 24)), free=(mp.w == 0) & ~mp.inroom)
    for rm in rooms:
        furnish(mp, rm, ["chair", "cooler", "plant", "boxes"], (1, 3), lamp=(mp.lcol, 0.9, 5.0, None, int(r.integers(2))),
                lamp_p=0.6, panel="panel0")


def gen_hotel(mp):
    r = mp.rng
    mp.H = r.uniform(2.6, 3.0)
    green = r.random() < 0.3
    HW = mp.kind("hotel_wall_green" if green else "hotel_wall")
    HD = mp.door("hotel_door", "hotel_wall_green" if green else "hotel_wall")
    HS = mp.kind("hotel_sconce")
    HF = mp.kind("hotel_sconce_f")
    mp.w[:] = HW
    n = 8
    eh, ev = lattice_maze(r, n, r.uniform(0.25, 0.7))
    carve(mp, eh, ev, 2)
    p = G // n
    lobbies = []
    for i in range(n):
        for j in range(n):
            if r.random() < 0.14:
                s = int(r.integers(5, 8))
                mp.rect(j * p + p // 2 - s // 2 + 1, i * p + p // 2 - s // 2 + 1, s, s, 0)
                lobbies.append((j * p + p // 2 + 1, i * p + p // 2 + 1, s))
    op = mp.w == 0
    rooms = rooms_in_solid(mp, {HW}, HD, int(r.integers(20, 38)), (3, 4), (3, 5), open_p=0.08)
    col = (1.0, r.uniform(0.66, 0.78), r.uniform(0.34, 0.48))
    dead = r.uniform(0.1, 0.45)
    sp = int(r.integers(3, 6))
    for z in range(G):
        for x in range(G):
            if mp.w[z, x] != HW:
                continue
            for dz, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if op[(z + dz) % G, (x + dx) % G]:
                    a = (x if dz != 0 else z) + (7 if dx + dz > 0 else 0)
                    if a % sp == 1:
                        u = r.random()
                        if u > dead:
                            fl = u > 0.93
                            mp.w[z, x] = HF if fl else HS
                            mp.light((x + dx) % G, (z + dz) % G, col, r.uniform(0.55, 0.85), 5.5, 1 if fl else 0)
                    break
    mp.floor[:] = MAT_ID["carpet_green" if green else "carpet_red"]
    mp.ceil[:] = MAT_ID["ceil_plaster"]
    for (x, z, s) in lobbies:
        mp.prop(x + 0.5, z + 0.5, "lamp")
        mp.light(x, z, col, 1.0, 6.0)
        for _ in range(int(r.integers(1, 5))):
            mp.prop(x + r.uniform(-s / 2 + 1, s / 2 - 1), z + r.uniform(-s / 2 + 1, s / 2 - 1), ["plant", "chair", "boxes"][int(r.integers(3))])
    mp.lcol = col
    mp.amb, mp.fog, mp.fogcol = (0.014, 0.008, 0.005), r.uniform(0.025, 0.045), (0.022, 0.011, 0.006)
    mp.rates = {E_WATCHER: 0.9, E_STALKER: 0.35}
    mp.red_rate = 0.12
    mp.grime, mp.staincol, mp.flash = 0.5, (0.35, 0.25, 0.2), 0.15
    mp.scatter(["plant", "sign"], int(r.integers(0, 6)), free=(mp.w == 0) & ~mp.inroom)
    rf = "carpet_brown" if r.random() < 0.6 else "wood_floor"
    for rm in rooms:
        furnish(mp, rm, ["bed", "dresser", "tv", "chair", "plant", "bed"], (2, 4), floor=rf,
                lamp=(WARM, 0.9, 5.0, "lamp", int(r.random() < 0.2)), lamp_p=0.55)


def gen_poolrooms(mp):
    r = mp.rng
    mp.H = r.uniform(3.0, 4.6)
    TW = mp.kind("tile_white")
    TB = mp.kind("tile_band")
    AR = mp.kind("tile_band", bot="tile_white", y0=2.25)
    LO = mp.kind("tile_small", top="tile_small", y1=0.45)
    TD = mp.door("tile_door", "tile_white")
    d = r.uniform(0.2, 1.0)
    partitions(mp, [TW, TB], int(10 + 30 * d), (3, 10), int(2 + 5 * d), (7, 16), int(r.integers(1, 6)), 0, gapk=AR, gap_p=0.8)
    rooms = closed_rooms(mp, TW, TD, int(r.integers(2, 7)), (4, 8), open_p=0.2)
    mp.floor[:] = MAT_ID["tile_floor"]
    mp.ceil[:] = MAT_ID["tile_white"]
    if r.random() < 0.2:
        mp.floor[:] = MAT_ID["water"]
    else:
        for _ in range(int(r.integers(6, 16))):
            mp.rect(int(r.integers(G)), int(r.integers(G)), int(r.integers(3, 15)), int(r.integers(3, 15)), MAT_ID["water"], mp.floor)
    for _ in range(int(r.integers(0, 6))):          # low tiled walls to step around
        mp.seg(int(r.integers(G)), int(r.integers(G)), int(r.integers(2, 8)), r.random() < 0.5, LO)
    if r.random() < 0.7:                            # round columns
        s = int(r.integers(4, 8))
        ox, oz = int(r.integers(s)), int(r.integers(s))
        for z in range(oz, G - s + 1, s):
            for x in range(ox, G - s + 1, s):
                if mp.w[z, x] == 0 and r.random() < 0.8:
                    mp.prop(x + 0.5, z + 0.5, "column")
    mp.lcol = (r.uniform(0.82, 0.95), r.uniform(0.93, 1.0), 1.0)
    s = int(r.integers(5, 9))
    ox, oz = int(r.integers(s)), int(r.integers(s))
    pw = r.uniform(1.0, 1.5)
    for z in range(oz, G, s):
        for x in range(ox, G, s):
            if r.random() < 0.08:
                continue
            for dz in (0, 1):
                for dx in (0, 1):
                    if mp.get(x + dx, z + dz) == 0:
                        mp.light(x + dx, z + dz, mp.lcol, pw * 0.5, 10.0, 0, "skylight")
    b = r.uniform(0.6, 1.2)
    mp.amb, mp.fog = (0.05 * b, 0.065 * b, 0.07 * b), r.uniform(0.008, 0.02)
    mp.fogcol = _fogc(r, (0.45, 0.58, 0.60), 0.5, 1.0)
    mp.rates = {E_WATCHER: 0.25}
    mp.grime, mp.staincol, mp.flash, mp.wave_rate = 0.25, (0.6, 0.75, 0.65), 0.03, 0.1
    mp.scatter(["float"], int(r.integers(0, 14)), free=(mp.w == 0) & (mp.floor == MAT_ID["water"]), clear=0)
    for rm in rooms:
        furnish(mp, rm, ["float"], (0, 1), floor="water" if r.random() < 0.5 else "tile_floor",
                lamp=(mp.lcol, 0.8, 6.0, None, 0), lamp_p=0.7, panel="skylight")


def gen_run(mp):
    r = mp.rng
    mp.H = r.uniform(2.6, 2.9)
    HP = mp.kind("hosp_wall")
    HD = mp.door("hosp_door", "hosp_wall")
    HA = mp.kind("hosp_alarm")
    HE = mp.kind("hosp_exit")
    CT = mp.kind("metal_plate", top="metal_plate", y1=0.9)
    mp.w[:] = HP
    zs = [int(8 + r.integers(-2, 3)), int(30 + r.integers(-2, 3)), int(52 + r.integers(-2, 3))]
    for z in zs:
        mp.rect(0, z - 1, G, 3, 0)
    for x in range(int(r.integers(4, 10)), G, int(r.integers(14, 22))):
        mp.rect(x, 0, 2, G, 0)
    for _ in range(int(r.integers(8, 20))):         # side rooms
        z = zs[int(r.integers(3))]
        s = int(r.integers(2, 5))
        mp.rect(int(r.integers(G)), z + 2 if r.random() < 0.5 else z - 1 - s, int(r.integers(2, 5)), s, 0)
    op = mp.w == 0
    rooms = rooms_in_solid(mp, {HP}, HD, int(r.integers(14, 28)), (2, 4), (2, 5), open_p=0.25)
    red = (1.0, 0.07, 0.04)
    for z in range(G):
        for x in range(G):
            if mp.w[z, x] != HP:
                continue
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if op[(z + dz) % G, (x + dx) % G]:
                    a = x if dz != 0 else z
                    if a % 10 == 2:
                        mp.w[z, x] = HA
                        mp.light((x + dx) % G, (z + dz) % G, red, r.uniform(1.0, 1.5), 7.0, 3)
                    elif a % 20 == 7:
                        mp.w[z, x] = HE
                        mp.light((x + dx) % G, (z + dz) % G, (0.2, 1.0, 0.45), 0.25, 4.0, 0)
                    break
    mp.floor[:] = MAT_ID["lino" if r.random() < 0.6 else "lino_dark"]
    mp.ceil[:] = MAT_ID["ceil_tile"]
    mp.lcol = (1.0, 0.95, 0.9)
    mp.lattice_lights(int(r.integers(5, 8)), 1, "panel0", mp.lcol, r.uniform(0.4, 0.7), 6.0, dead=r.uniform(0.4, 0.7),
                      f1=0.15, f2=0.2, only=np.isin(np.arange(G), zs)[:, None] & np.ones((1, G), bool))
    for z in zs:                                    # things in the way
        for x in range(0, G, int(r.integers(5, 11))):
            u = r.random()
            if u < 0.35:
                mp.prop(x + 0.5, z + int(r.integers(-1, 2)) + 0.5, "gurney")
            elif u < 0.5:
                mp.w[(z + int(r.integers(-1, 2))) % G, x % G] = CT
            elif u < 0.6:
                mp.prop(x + 0.5, z + int(r.integers(-1, 2)) + 0.5, "sign")
    mp.amb, mp.fog, mp.fogcol = (0.022, 0.005, 0.004), r.uniform(0.03, 0.045), (0.05, 0.006, 0.004)
    mp.pulse = (r.uniform(0.8, 1.3), 0.12, 1.0, 0)
    mp.rates = {}
    mp.grime, mp.staincol, mp.flash, mp.wave_rate = 0.6, (0.4, 0.3, 0.25), 0.1, 0.0
    mp.axis = (zs, 1 if r.random() < 0.5 else -1)
    for rm in rooms:
        furnish(mp, rm, ["gurney", "chair", "boxes"], (0, 2), lamp=(mp.lcol, 0.6, 5.0, None, int(r.integers(1, 3))),
                lamp_p=0.35, panel="panel0")


def gen_fun(mp):
    r = mp.rng
    mp.H = r.uniform(2.8, 3.4)
    P = [mp.kind("party0"), mp.kind("party1"), mp.kind("party2")]
    PF = mp.kind("party_face")
    TB = mp.kind("cloth0" if r.random() < 0.5 else "cloth1", y1=0.8)
    Lt = mp.kind("party1", bot="ceil_tile", y0=2.1)
    PDo = mp.door("party_door", "party0")
    d = r.uniform(0.3, 1.0)
    partitions(mp, P, int(20 + 45 * d), (3, 10), int(4 + 6 * d), (5, 12), int(r.integers(1, 5)), int(r.integers(0, 10)),
               gapk=Lt, gap_p=0.4)
    mp.haunt = (0.0, 0.0, 0.0)
    rooms = closed_rooms(mp, P[0], PDo, int(r.integers(2, 7)), (4, 7), open_p=0.15)
    mp.doors = [d[:6] + (4 if r.random() < 0.45 else 0,) for d in mp.doors]        # a partygoer waits in some
    for k in P:
        mp.faces(k, PF, 0.06)
    for _ in range(int(r.integers(6, 20))):
        x, z = int(r.integers(G)), int(r.integers(G))
        if mp.get(x, z) == 0:
            mp.rect(x, z, int(r.integers(1, 3)), int(r.integers(1, 3)), TB)
    mp.floor[:] = MAT_ID["checker" if r.random() < 0.5 else "checker_color"]
    mp.ceil[:] = MAT_ID["ceil_tile"]
    mp.lcol = (1.0, 0.98, 0.92)
    mp.lattice_lights(int(r.integers(2, 4)), int(r.integers(2, 4)), "panel0", mp.lcol, r.uniform(1.0, 1.4), 7.0,
                      dead=r.uniform(0.0, 0.12), f1=0.03, f2=0.01)
    for _ in range(int(r.integers(4, 14))):         # coloured lights
        x, z = int(r.integers(G)), int(r.integers(G))
        if mp.get(x, z) == 0:
            h = r.random() * 6
            c = tuple(float(np.clip(abs((h + o) % 6 - 3) - 1, 0, 1)) for o in (0, 4, 2))
            mp.light(x, z, c, 1.6, 6.0)
    mp.amb, mp.fog = (0.045, 0.042, 0.038), r.uniform(0.008, 0.02)
    mp.fogcol = (0.16, 0.14, 0.12)
    mp.rates = {}
    mp.grime, mp.staincol, mp.flash, mp.wave_rate = 0.25, (0.6, 0.5, 0.45), 0.03, 0.5
    mp.scatter(["balloon_r", "balloon_b", "balloon_y"], int(r.integers(40, 110)), clear=0, free=(mp.w == 0) & ~mp.inroom)
    for rm in rooms:
        furnish(mp, rm, ["balloon_r", "balloon_b", "balloon_y", "chair"], (2, 5), lamp=(mp.lcol, 1.0, 5.0, None, 0),
                lamp_p=0.8, panel="panel0")
    mp.n_party = int(r.integers(3, 6))


def gen_library(mp):
    r = mp.rng
    mp.H = r.uniform(3.4, 4.6)
    tall = r.random() < 0.5
    BS = mp.kind("bookshelf") if tall else mp.kind("bookshelf2", top="shelf_top", y1=2.3)
    BS2 = mp.kind("bookshelf2") if tall else mp.kind("bookshelf", top="shelf_top", y1=2.3)
    WD = mp.kind("wood_dark")
    TB = mp.kind("desk_side", top="desk_top", y1=0.75)
    WDo = mp.door("wood_door", "wood_dark")
    s = int(r.integers(3, 5))
    cross = int(r.integers(8, 14))
    cw = int(r.integers(1, 3))
    for z in range(0, G - s + 1, s):
        if r.random() < 0.9:
            k = BS if r.random() < 0.7 else BS2
            for x in range(G):
                if x % cross >= cw and r.random() < 0.985:
                    mp.w[z, x] = k
    for _ in range(int(r.integers(2, 6))):          # reading rooms
        x0, z0, sx, sz = int(r.integers(G)), int(r.integers(G)), int(r.integers(6, 12)), int(r.integers(5, 10))
        mp.rect(x0, z0, sx, sz, 0)
        for _ in range(int(r.integers(1, 4))):
            mp.rect(x0 + int(r.integers(1, sx - 2)), z0 + int(r.integers(1, sz - 2)), int(r.integers(1, 3)), 1, TB)
        mp.prop(x0 + sx / 2, z0 + sz / 2, "lamp")
        mp.light(x0 + sx // 2, z0 + sz // 2, (1.0, 0.75, 0.45), 1.1, 7.0)
    for _ in range(int(r.integers(0, 6))):
        mp.rect(int(r.integers(G)), int(r.integers(G)), int(r.integers(2, 4)), int(r.integers(2, 4)), WD)
    if r.random() < 0.5:
        mp.w = np.ascontiguousarray(mp.w.T)
        mp.props = [(z, x, i) for (x, z, i) in mp.props]
        mp.lights = [(z, x, *rest) for (x, z, *rest) in mp.lights]
    rooms = closed_rooms(mp, WD, WDo, int(r.integers(2, 7)), (4, 8), open_p=0.15)
    mp.floor[:] = MAT_ID["wood_floor" if r.random() < 0.6 else "carpet_green"]
    mp.ceil[:] = MAT_ID["ceil_plaster"]
    mp.lcol = (1.0, r.uniform(0.74, 0.86), r.uniform(0.42, 0.6))
    mp.lattice_lights(int(r.integers(4, 8)), int(r.integers(4, 8)), "panel2", mp.lcol, r.uniform(1.0, 1.5), 8.0,
                      dead=r.uniform(0.15, 0.5), f1=0.06, f2=0.05, blobs=int(r.integers(0, 3)), jit=1)
    mp.amb, mp.fog, mp.fogcol = (0.018, 0.013, 0.008), r.uniform(0.03, 0.05), (0.012, 0.008, 0.005)
    mp.rates = {E_WATCHER: 0.8, E_STALKER: 0.35}
    mp.grime, mp.staincol, mp.flash = 0.4, (0.35, 0.28, 0.2), 0.25
    mp.scatter(["chair", "boxes", "plant"], int(r.integers(0, 8)), free=(mp.w == 0) & ~mp.inroom)
    for rm in rooms:
        furnish(mp, rm, ["chair", "boxes", "plant", "dresser"], (0, 3), floor="carpet_green",
                lamp=(WARM, 1.0, 5.5, "lamp", 0), lamp_p=0.5)

# ------------------------------------------------------------------ outdoors

def _sky(mp, kind):
    """Time of day: light, fog and sky of an outdoor map. kind: "day" (overcast), "dusk", "night"."""
    r = mp.rng
    a = r.uniform(0, 2 * math.pi)
    if kind == "day":
        b = r.uniform(0.8, 1.1)
        mp.amb = (0.80 * b, 0.82 * b, 0.86 * b)
        hz, zn = (0.74 * b, 0.76 * b, 0.78 * b), (0.50 * b, 0.54 * b, 0.60 * b)
        mp.fog, mp.fogcol = r.uniform(0.025, 0.06), tuple(0.92 * v for v in hz)
        mp.sky, mp.sun, mp.flash = ("sky_clouds", hz, zn, 0.5, 0.0), (math.cos(a), math.sin(a), 0.22), 0.02
    elif kind == "dusk":
        mp.amb = (0.26, 0.19, 0.16)
        hz, zn = (0.95, 0.42, 0.16), (0.09, 0.09, 0.20)
        mp.fog, mp.fogcol = r.uniform(0.02, 0.04), (0.42, 0.22, 0.12)
        mp.sky, mp.sun, mp.flash = ("sky_clouds", hz, zn, 0.6, 0.0), (math.cos(a), math.sin(a), 0.55), 0.25
    else:
        b = r.uniform(0.7, 1.5)
        mp.amb = (0.010 * b, 0.013 * b, 0.022 * b)
        hz, zn = (0.020, 0.026, 0.046), (0.003, 0.004, 0.010)
        mp.fog, mp.fogcol = r.uniform(0.04, 0.07), (0.006, 0.008, 0.014)
        mp.sky, mp.sun, mp.flash = ("sky_stars", hz, zn, 0.5, 0.9), (math.cos(a), math.sin(a), 0.0), 0.9
    mp.wave_rate, mp.red_rate, mp.spark = 0.0, 0.0, 0.0
    return kind


def _street_lamp(mp, x, z, col=(1.0, 0.62, 0.25), power=1.7, radius=10.0):
    """A lamp post at cell (x, z); now and then a dead or a failing one."""
    u = mp.rng.random()
    mp.prop(x + 0.5, z + 0.5, "lamppost")
    if u < 0.15:
        return
    mp.prop(x + 0.8, z + 0.5, "lamphead")
    mp.light(x, z, col, power, radius, 1 if u < 0.27 else (2 if u < 0.33 else 0))


def _building(mp, x0, z0, sx, sz, wall, dk, roof, side, floor, win=(), win_p=0.0, opened=0.1):
    """A house: outline of `wall` (outer size sx x sz) with a door of kind dk on `side` (0: -z, 1: +z, 2: -x, 3: +x),
    its inside under `roof`, some wall cells turned into windows. Returns the room."""
    r = mp.rng
    mp.rect(x0 - 1, z0 - 1, sx + 2, sz + 2, 0)
    mp.rect(x0, z0, sx, sz, wall)
    mp.rect(x0 + 1, z0 + 1, sx - 2, sz - 2, roof)
    mp.rect(x0, z0, sx, sz, MAT_ID[floor], mp.floor)
    if side < 2:
        x, z = x0 + int(r.integers(1, sx - 1)), (z0 if side == 0 else z0 + sz - 1)
        nx, nz = 0, (-1 if side == 0 else 1)
    else:
        x, z = (x0 if side == 2 else x0 + sx - 1), z0 + int(r.integers(1, sz - 1))
        nx, nz = (-1 if side == 2 else 1), 0
    if win:
        for xx in range(x0, x0 + sx):
            for zz in range(z0, z0 + sz):
                edge_x, edge_z = xx in (x0, x0 + sx - 1), zz in (z0, z0 + sz - 1)
                if (edge_x != edge_z) and (xx, zz) != (x, z) and r.random() < win_p:
                    lit = len(win) > 1 and r.random() < 0.3
                    mp.w[zz % G, xx % G] = win[1] if lit else win[0]
                    if lit:
                        ox, oz = (-1 if xx == x0 else 1) if edge_x else 0, (-1 if zz == z0 else 1) if edge_z else 0
                        mp.light(xx + ox, zz + oz, WARM, 0.45, 4.0)
    zz, xx = np.meshgrid(np.arange(z0 + 1, z0 + sz - 1) % G, np.arange(x0 + 1, x0 + sx - 1) % G, indexing="ij")
    rm = _room(mp, xx.ravel(), zz.ravel(), x, z, nx, nz)
    mp.put_door(x, z, dk, nx, nz, rm["mid"][0], rm["mid"][1], opened=r.random() < opened)
    mp.used[np.ix_(np.arange(z0 - 1, z0 + sz + 1) % G, np.arange(x0 - 1, x0 + sx + 1) % G)] = True
    return rm


def gen_fields(mp):
    r = mp.rng
    mp.outdoor, mp.H = True, 24.0
    tod = _sky(mp, str(r.choice(["day", "dusk", "night"], p=[0.45, 0.2, 0.35])))
    grass = "grass" if r.random() < 0.6 else "grass_dry"
    mp.floor[:] = MAT_ID[grass]
    WH = mp.kind("wheat_side", top="wheat_top", y1=r.uniform(0.85, 1.15), block=False)
    CO = mp.kind("corn_side", top="corn_top", y1=2.7, full=True)
    FE = mp.kind("fence", y1=1.2)
    barn = "barn" if r.random() < 0.6 else "siding"
    BW = mp.kind(barn, y1=3.0, full=True)
    WN = () if barn == "barn" else (mp.kind("siding_window", y1=3.0, full=True), mp.kind("siding_window_lit", y1=3.0, full=True))
    RF = mp.kind("wood_dark", bot="wood_dark", y0=2.7, y1=3.0)
    BD = mp.door(barn + "_door", barn, h=3.0)
    CT = mp.kind("container" if r.random() < 0.5 else "container_blue", y1=2.5, full=True)
    for _ in range(int(r.integers(4, 9))):          # stands of wheat
        mp.rect(int(r.integers(G)), int(r.integers(G)), int(r.integers(10, 28)), int(r.integers(8, 24)), WH)
    for _ in range(int(r.integers(0, 3))):          # maize, taller than the player, with lanes cut through it
        x0, z0, sx, sz = int(r.integers(G)), int(r.integers(G)), int(r.integers(12, 24)), int(r.integers(12, 24))
        mp.rect(x0, z0, sx, sz, CO)
        for z in range(z0 + 1, z0 + sz, 3):
            mp.seg(x0, z, sx, True, 0)
            for x in range(x0 + int(r.integers(1, 5)), x0 + sx, int(r.integers(4, 9))):
                mp.seg(x, z, 3, False, 0)
        mp.seg(x0 + int(r.integers(1, sx - 1)), z0, 2, False, 0)
    roads = []
    for _ in range(int(r.integers(1, 3))):          # dirt roads, some fenced
        hz, o, wd = r.random() < 0.5, int(r.integers(G)), int(r.integers(2, 4))
        roads.append((hz, o, wd))
        if hz:
            mp.rect(0, o, G, wd, 0); mp.rect(0, o, G, wd, MAT_ID["dirt"], mp.floor)
        else:
            mp.rect(o, 0, wd, G, 0); mp.rect(o, 0, wd, G, MAT_ID["dirt"], mp.floor)
        if r.random() < 0.7:
            for a in range(0, G, int(r.integers(7, 14))):
                L = int(r.integers(3, 7))
                mp.seg(a if hz else o - 1, o - 1 if hz else a, L, hz, FE)
        for a in range(int(r.integers(8)), G, 9):   # poles along it
            mp.prop((a if hz else o + wd) + 0.5, (o + wd if hz else a) + 0.5, "pole")
    for _ in range(int(r.integers(0, 3))):          # a container or two, left in a field
        mp.seg(int(r.integers(G)), int(r.integers(G)), 3 if r.random() < 0.6 else 5, r.random() < 0.5, CT)
    rooms = []
    for _ in range(int(r.integers(1, 4))):          # barns and sheds
        sx, sz = int(r.integers(6, 11)), int(r.integers(5, 9))
        for _try in range(20):
            x0, z0 = int(r.integers(G)), int(r.integers(G))
            if not mp.used[np.ix_(np.arange(z0 - 2, z0 + sz + 2) % G, np.arange(x0 - 2, x0 + sx + 2) % G)].any():
                break
        else:
            continue
        mp.rect(x0 - 2, z0 - 2, sx + 4, sz + 4, MAT_ID["dirt"], mp.floor)
        mp.rect(x0 - 2, z0 - 2, sx + 4, sz + 4, 0)
        rooms.append(_building(mp, x0, z0, sx, sz, BW, BD, RF, int(r.integers(4)), "concrete_floor" if r.random() < 0.5 else "wood_floor",
                               WN, 0.12, opened=0.2))
        if tod != "day" or r.random() < 0.3:
            _street_lamp(mp, x0 - 2, z0 - 2, WARM, 1.4, 9.0)
    op = (mp.w == 0) & ~mp.used
    mp.scatter(["haybale"], int(r.integers(4, 16)), free=op)
    mp.scatter(["tree", "tree", "pine"], int(r.integers(6, 26)), free=op)
    mp.scatter(["scarecrow"], int(r.integers(2, 7)), free=(mp.w == WH), clear=0)
    mp.scatter(["barrel_rust", "boxes"], int(r.integers(0, 6)), free=op)
    for rm in rooms:
        furnish(mp, rm, ["haybale", "boxes", "barrel", "barrel_rust", "chair"], (1, 4), lamp=(WARM, 1.0, 6.0, "lamp", int(r.random() < 0.3)),
                lamp_p=0.2 if tod == "day" else 0.45)
    mp.rates = {E_WATCHER: 0.9, E_STALKER: 0.35} if tod == "day" else {E_WATCHER: 0.6, E_STALKER: 0.6, E_HOUND: 0.35, E_SMILER: 0.5}
    mp.grime, mp.staincol = 0.3, (0.40, 0.36, 0.22)


def gen_suburbs(mp):
    r = mp.rng
    mp.outdoor, mp.H = True, 24.0
    _sky(mp, "night" if r.random() < 0.8 else "dusk")
    mp.floor[:] = MAT_ID["grass"]
    sid = "siding" if r.random() < 0.5 else "siding_dark"
    HW = mp.kind(sid, y1=3.0, full=True)
    WN = (mp.kind(sid + "_window", y1=3.0, full=True), mp.kind(sid + "_window_lit", y1=3.0, full=True))
    RF = mp.kind("ceil_plaster", bot="ceil_plaster", y0=2.7, y1=3.0)
    HD = mp.door(sid + "_door", sid, h=3.0)
    FE = mp.kind("fence", y1=1.2)
    HG = mp.kind("hedge", top="hedge", y1=1.5)
    s0, c0 = int(r.integers(16)), int(r.integers(32))
    streets = [(s0 + 16 * i) % G for i in range(4)]
    for z in streets:                               # streets with a pavement on both sides
        mp.rect(0, z, G, 6, MAT_ID["sidewalk"], mp.floor)
        mp.rect(0, z + 1, G, 4, MAT_ID["asphalt"], mp.floor)
    cross = [(c0 + 32 * i) % G for i in range(2)]
    for x in cross:
        mp.rect(x, 0, 6, G, MAT_ID["sidewalk"], mp.floor)
        mp.rect(x + 1, 0, 4, G, MAT_ID["asphalt"], mp.floor)
    for z in streets:
        mp.rect(0, z + 1, G, 4, MAT_ID["asphalt"], mp.floor)
    road = np.isin(mp.floor, [MAT_ID["sidewalk"], MAT_ID["asphalt"]])
    mp.used |= road
    rooms = []
    for z in streets:                               # a row of houses in the block behind each street
        up = r.random() < 0.5                       # they face this street, or the next one
        x = int(r.integers(3))
        while x < G - 5:
            sx, sz = int(r.integers(6, 9)), int(r.integers(5, 7))
            z0 = z + 6 + (2 if up else 10 - 2 - sz)
            if mp.used[np.ix_(np.arange(z0 - 1, z0 + sz + 1) % G, np.arange(x - 1, x + sx + 1) % G)].any() or r.random() < 0.12:
                x += 2
                continue
            rm = _building(mp, x, z0, sx, sz, HW, HD, RF, 0 if up else 1, "wood_floor" if r.random() < 0.5 else "carpet_brown", WN, 0.22, opened=0.12)
            rooms.append(rm)
            dx_, dzu = rm["door"][0], (z0 if up else z0 + sz - 1)
            fz = z + 5 if up else z + 16                 # a path from the pavement to the door
            for zz in range(min(fz, dzu), max(fz, dzu) + 1):
                if mp.get(dx_, zz) == 0:
                    mp.floor[zz % G, dx_ % G] = MAT_ID["sidewalk"]
            if r.random() < 0.5:                         # a fence or a hedge towards the neighbour
                mp.seg(x + sx + 1, z0 - 1, sz + 2, False, FE if r.random() < 0.6 else HG)
            x += sx + int(r.integers(3, 6))
    for z in streets:                               # lamps along the streets
        for x in range(int(r.integers(6)), G, int(r.integers(9, 14))):
            if not any(0 <= (x - c) % G < 6 for c in cross):
                _street_lamp(mp, x, z if r.random() < 0.5 else z + 5)
    op = (mp.w == 0) & ~mp.used
    mp.scatter(["tree", "pine", "tree"], int(r.integers(10, 28)), free=op)
    mp.scatter(["sign", "barrel", "boxes", "cone"], int(r.integers(0, 8)), free=(mp.w == 0) & road, clear=0)
    for rm in rooms:
        furnish(mp, rm, ["bed", "tv", "dresser", "chair", "plant", "lamp"], (2, 5), lamp=(WARM, 0.9, 5.5, "lamp", int(r.random() < 0.25)), lamp_p=0.35)
    mp.rates = {E_STALKER: 0.7, E_WATCHER: 0.8, E_HOUND: 0.45, E_SMILER: 0.5}
    mp.haunt = (0.26, 0.07, 0.08)
    mp.grime, mp.staincol = 0.3, (0.3, 0.3, 0.25)


def gen_city(mp):
    r = mp.rng
    mp.outdoor, mp.H = True, 60.0
    tod = _sky(mp, str(r.choice(["day", "dusk", "night"], p=[0.55, 0.3, 0.15])))
    mp.floor[:] = MAT_ID["sidewalk"]
    hs = sorted(r.uniform(9, 46, size=3).tolist())
    fac = [str(f) for f in r.permutation(["facade0", "facade1", "facade2"])]
    if tod != "day":
        fac[0] = "facade_lit"
    F = [mp.kind(fac[i], y1=hs[i], full=True) for i in range(3)]
    S = [mp.kind(f"shop{i}", y1=hs[i], full=True, up=fac[i], split=3.4) for i in range(3)]
    D = [mp.door("city_door", "city_wall", h=hs[i], up=fac[i], split=3.4) for i in range(2)]
    RF = mp.kind("city_wall", bot="ceil_tile_office", y0=3.2, y1=60.0)
    PL = mp.kind("hedge", top="hedge", y1=0.8)
    CT = mp.kind("container", y1=2.5, full=True)
    s0, c0 = int(r.integers(16)), int(r.integers(16))
    for i in range(4):                              # a grid of streets; what is between them is built up
        mp.rect(0, s0 + 16 * i + 1, G, 4, MAT_ID["asphalt"], mp.floor)
        mp.rect(c0 + 16 * i + 1, 0, 4, G, MAT_ID["asphalt"], mp.floor)
    rooms = []
    for i in range(4):
        for j in range(4):
            x0, z0 = c0 + 16 * j + 6, s0 + 16 * i + 6
            if r.random() < 0.08:                   # an empty lot
                mp.rect(x0, z0, 10, 10, MAT_ID["asphalt" if r.random() < 0.5 else "dirt"], mp.floor)
                if r.random() < 0.6:
                    mp.seg(x0 + 2, z0 + 3, 4, True, CT)
                continue
            t = int(r.integers(3))
            if r.random() < 0.35:                   # two buildings on the block
                t2 = int(r.integers(3))
                a = int(r.integers(4, 7))
                mp.rect(x0, z0, a, 10, F[t]); mp.rect(x0 + a, z0, 10 - a, 10, F[t2])
            else:
                mp.rect(x0, z0, 10, 10, F[t])
            mp.used[np.ix_(np.arange(z0, z0 + 10) % G, np.arange(x0, x0 + 10) % G)] = True
    for t in range(3):                              # shop windows at street level
        mp.faces(F[t], S[t], r.uniform(0.15, 0.5))
    for t in range(2):                              # ways in: a lobby behind a door
        for rm in rooms_in_solid(mp, {F[t]}, D[t], int(r.integers(2, 7)), (2, 4), (3, 5), keep=set(F) | set(S), roof=RF, open_p=0.15):
            rooms.append(rm)
    for i in range(4):                              # street furniture on the pavements
        for a in range(int(r.integers(8)), G, int(r.integers(8, 13))):
            z = s0 + 16 * i + (0 if r.random() < 0.5 else 5)
            if mp.get(a, z) == 0 and not mp.used[z % G, a % G]:
                _street_lamp(mp, a, z, (1.0, 0.85, 0.6), 1.5, 10.0) if tod != "day" or r.random() < 0.5 else mp.prop(a + 0.5, z + 0.5, "lamppost")
        for a in range(int(r.integers(8)), G, int(r.integers(6, 12))):
            x = c0 + 16 * i + (0 if r.random() < 0.5 else 5)
            if mp.get(x, a) == 0 and not mp.used[a % G, x % G]:
                u = r.random()
                if u < 0.3:
                    mp.w[a % G, x % G] = PL
                else:
                    mp.prop(x + 0.5, a + 0.5, ["bench", "tree", "cone", "sign", "barrel", "boxes"][int(r.integers(6))])
    for rm in rooms:
        furnish(mp, rm, ["plant", "chair", "boxes", "mannequin", "bench"], (0, 3), floor="lino_dark",
                lamp=((0.9, 0.95, 1.0), 0.9, 5.5, "lamp", int(r.random() < 0.3)), lamp_p=0.5)
    mp.rates = {E_WATCHER: 0.8, E_STALKER: 0.3} if tod == "day" else {E_WATCHER: 0.8, E_STALKER: 0.5, E_HOUND: 0.25, E_SMILER: 0.3}
    mp.grime, mp.staincol = 0.4, (0.3, 0.3, 0.3)


def _noise2(r, lo, hi, beta=2.4):
    """Smooth noise [G, G] that wraps around, mean 0 and spread 1, with lo..hi waves across the map."""
    n = r.standard_normal((G, G))
    fy, fx = np.meshgrid(np.fft.fftfreq(G, 1 / G), np.fft.rfftfreq(G, 1 / G), indexing="ij")
    f = np.sqrt(fy * fy + fx * fx)
    x = np.fft.irfft2(np.fft.rfft2(n) * np.where((f >= lo) & (f <= hi), np.maximum(f, 1) ** (-beta / 2), 0), s=(G, G))
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


def _smooth(m, n=3):
    """A mask [G, G] with its edges spread over a few cells (wrapping)."""
    m = m.astype(np.float32)
    for _ in range(n):
        m = (m + np.roll(m, 1, 0) + np.roll(m, -1, 0) + np.roll(m, 1, 1) + np.roll(m, -1, 1)) / 5
    return m


def gen_hills(mp):
    """Level 94: green hills under a clear sky that never darkens, white houses to walk into, nobody about."""
    r = mp.rng
    mp.outdoor, mp.H = True, 30.0
    b, a = r.uniform(0.9, 1.1), r.uniform(0, 2 * math.pi)
    mp.amb = (0.80 * b, 0.84 * b, 0.86 * b)
    mp.fog, mp.fogcol = r.uniform(0.006, 0.014), (0.58, 0.72, 0.72)
    mp.sky, mp.clouds, mp.backdrop = ("sky_clouds", (0.70, 0.84, 1.0), (0.22, 0.42, 0.90), 0.0, 0.0), r.uniform(0.6, 1.0), "bg_hills"
    mp.sun, mp.flash = (math.cos(a), math.sin(a), 0.45), 0.0
    mp.wave_rate, mp.red_rate, mp.spark = 0.0, 0.0, 0.0
    mp.floor[:] = MAT_ID["grass"]
    h = r.uniform(1.3, 2.0) * _noise2(r, 1, 3) + 0.4 * _noise2(r, 3, 7)
    HW = mp.kind("white_wall", y1=4.3, full=True, up="roof", split=2.7)
    WN = (mp.kind("white_window", y1=4.3, full=True, up="roof", split=2.7),)
    RF = mp.kind("roof", bot="ceil_plaster", y0=2.7, y1=4.3)
    HD = mp.door("white_door", "white_wall", h=4.3, up="roof", split=2.7)
    PF = mp.kind("picket", y1=1.0)
    rooms = []
    for _ in range(int(r.integers(4, 9))):          # houses, each on ground levelled for it
        sx, sz = int(r.integers(6, 10)), int(r.integers(5, 8))
        for _try in range(30):
            x0, z0 = int(r.integers(G)), int(r.integers(G))
            if not mp.used[np.ix_(np.arange(z0 - 5, z0 + sz + 5) % G, np.arange(x0 - 5, x0 + sx + 5) % G)].any():
                break
        else:
            continue
        rooms.append(_building(mp, x0, z0, sx, sz, HW, HD, RF, int(r.integers(4)), "wood_floor", WN, 0.2, opened=0.3))
        pad = np.zeros((G, G), bool)
        pad[np.ix_(np.arange(z0 - 3, z0 + sz + 3) % G, np.arange(x0 - 3, x0 + sx + 3) % G)] = True
        m = np.clip(_smooth(pad, 4) * 1.6, 0, 1)
        h = h * (1 - m) + float(h[np.ix_(np.arange(z0, z0 + sz) % G, np.arange(x0, x0 + sx) % G)].mean()) * m
        mp.used[np.ix_(np.arange(z0 - 3, z0 + sz + 3) % G, np.arange(x0 - 3, x0 + sx + 3) % G)] = True
        if r.random() < 0.6:                        # a picket fence round the garden, with a gate
            for (xa, za, L, hz) in ((x0 - 3, z0 - 3, sx + 6, True), (x0 - 3, z0 + sz + 2, sx + 6, True), (x0 - 3, z0 - 3, sz + 6, False), (x0 + sx + 2, z0 - 3, sz + 6, False)):
                mp.seg(xa, za, L, hz, PF)
                o = int(r.integers(1, L - 2))
                mp.seg(xa + (o if hz else 0), za + (0 if hz else o), 2, hz, 0)
    for _ in range(int(r.integers(2, 6))):          # stray lengths of fence
        mp.seg(int(r.integers(G)), int(r.integers(G)), int(r.integers(3, 9)), r.random() < 0.5, PF)
    op = (mp.w == 0) & ~mp.used
    top = np.unravel_index(int(np.argmax(np.where(op, h, -1e9))), h.shape)      # the water tower stands on the highest ground
    mp.prop(top[1] + 0.5, top[0] + 0.5, "watertower")
    mp.scatter(["oak", "oak", "pine"], int(r.integers(8, 22)), free=op)
    mp.scatter(["bush"], int(r.integers(8, 30)), free=op, clear=0)
    for rm in rooms:
        furnish(mp, rm, ["bed", "dresser", "chair", "plant", "tv", "lamp"], (2, 5), lamp=(WARM, 0.6, 5.0, "lamp", 0), lamp_p=0.5)
    mp.hmap = h - float(h.mean())
    mp.rates, mp.haunt = {}, (0.0, 0.0, 0.0)
    mp.doors = [d[:6] + (0,) for d in mp.doors]
    mp.grime, mp.staincol = 0.1, (0.5, 0.5, 0.4)


# ------------------------------------------------------------------ the apartments and the mall

def gen_apartments(mp):
    r = mp.rng
    mp.H = r.uniform(2.5, 2.8)
    AW = mp.kind("apt_wall")
    AD = mp.door("apt_door", "apt_wall")
    mp.w[:] = AW
    n = 8
    eh, ev = lattice_maze(r, n, r.uniform(0.15, 0.5))
    wd = 1 if r.random() < 0.45 else 2
    carve(mp, eh, ev, wd)
    rooms = rooms_in_solid(mp, {AW}, AD, int(r.integers(30, 60)), (3, 5), (3, 6), open_p=0.15)
    mp.floor[:] = MAT_ID["lino_dark" if r.random() < 0.6 else "carpet_brown"]
    mp.ceil[:] = MAT_ID["ceil_tile"]
    mp.lcol = (r.uniform(0.8, 0.95), 1.0, r.uniform(0.72, 0.9))
    mp.lattice_lights(int(r.integers(3, 6)), int(r.integers(3, 6)), "panel0", mp.lcol, r.uniform(0.7, 1.0), 6.0,
                      dead=r.uniform(0.2, 0.5), f1=0.15, f2=0.12, blobs=int(r.integers(0, 3)))
    for rm in rooms:
        u = r.random()
        if u < 0.4:
            furnish(mp, rm, ["bed", "dresser", "chair", "bed"], (2, 4), floor="carpet_brown", lamp=(WARM, 0.9, 5.0, "lamp", int(r.random() < 0.2)), lamp_p=0.5)
        elif u < 0.7:
            furnish(mp, rm, ["tv", "chair", "plant", "dresser", "boxes"], (2, 4), floor="wood_floor", lamp=(WARM, 0.9, 5.0, "lamp", int(r.random() < 0.2)), lamp_p=0.5)
        elif u < 0.85:
            furnish(mp, rm, ["cooler", "chair", "boxes"], (1, 3), floor="tile_floor", lamp=(mp.lcol, 0.9, 5.0, None, int(r.integers(3))), lamp_p=0.6, panel="panel0")
        else:
            furnish(mp, rm, ["boxes", "mannequin"], (0, 2), floor="concrete_floor")
    b = r.uniform(0.5, 1.0)
    mp.amb, mp.fog, mp.fogcol = (0.016 * b, 0.018 * b, 0.013 * b), r.uniform(0.025, 0.05), (0.012, 0.014, 0.010)
    mp.rates = {E_STALKER: 0.5, E_WATCHER: 0.6, E_SMILER: 0.25}
    mp.haunt = (0.30, 0.07, 0.08)
    mp.red_rate = 0.1
    mp.grime, mp.staincol, mp.flash = 0.9, (0.36, 0.30, 0.20), 0.3
    mp.scatter(["boxes", "chair", "sign"], int(r.integers(0, 7)))


def gen_mall(mp):
    r = mp.rng
    mp.H = r.uniform(4.2, 5.5)
    MW = mp.kind("mall_wall")
    S = [mp.kind(f"mall_shop{i}", up="mall_wall", split=3.2) for i in range(3)]
    SH = mp.door("mall_shutter", "mall_wall")
    KI = mp.kind("mall_wall", top="desk_top", y1=1.1)
    PL = mp.kind("hedge", top="hedge", y1=0.7)
    mp.w[:] = MW
    n = 4
    eh, ev = lattice_maze(r, n, r.uniform(0.4, 0.9))
    carve(mp, eh, ev, int(r.integers(4, 7)))
    p = G // n
    halls = []
    for i in range(n):
        for j in range(n):
            if r.random() < 0.3:                    # an atrium
                sz_ = int(r.integers(9, 13))
                mp.rect(j * p + p // 2 - sz_ // 2, i * p + p // 2 - sz_ // 2, sz_, sz_, 0)
                halls.append((j * p + p // 2, i * p + p // 2))
    op = mp.w == 0
    rooms = rooms_in_solid(mp, {MW}, SH, int(r.integers(18, 40)), (3, 6), (4, 7), open_p=0.4)
    sign = [MATS[f"mall_shop{i}"]["emcol"] for i in range(3)]
    for rm in rooms:                                # the shop's front, and the glow of its sign on the floor outside
        k = int(r.integers(3))
        for (x, z) in rm["front"]:
            mp.w[z, x] = S[k]
        (x, z), (nx, nz) = rm["door"], rm["n"]
        if r.random() < 0.7:
            mp.light(x + nx, z + nz, sign[k], 0.5, 4.5, 1 if r.random() < 0.15 else 0)
    mp.floor[:] = MAT_ID["mall_floor"]
    mp.ceil[:] = MAT_ID["ceil_tile_office"]
    mp.lcol = (1.0, r.uniform(0.92, 1.0), r.uniform(0.82, 0.95))
    mp.lattice_lights(int(r.integers(3, 6)), int(r.integers(3, 6)), "panel0", mp.lcol, r.uniform(0.9, 1.3), 8.0,
                      dead=r.uniform(0.15, 0.6), f1=0.06, f2=0.05, blobs=int(r.integers(0, 4)), only=op)
    for (x, z) in halls:                            # daylight panels over the atriums, something to sit on under them
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                mp.light(x + dx * 2, z + dz * 2, (0.85, 0.92, 1.0), 0.5, 9.0, 0, "skylight")
        mp.w[z % G, x % G] = PL
        mp.prop(x + 1.5, z + 0.5, "bench")
    for _ in range(int(r.integers(2, 9))):          # kiosks
        x, z = int(r.integers(G)), int(r.integers(G))
        if op[z, x] and op[z, (x + 1) % G] and mp.get(x, z) == 0:
            mp.seg(x, z, 2, True, KI)
    for rm in rooms:
        furnish(mp, rm, ["mannequin", "mannequin", "boxes", "chair", "plant", "dresser"], (1, 5), floor="carpet_grey" if r.random() < 0.6 else "tile_floor",
                lamp=(mp.lcol, 0.9, 6.0, None, int(r.integers(2))), lamp_p=0.45, panel="panel0")
    b = r.uniform(0.5, 1.1)
    mp.amb, mp.fog = (0.030 * b, 0.029 * b, 0.027 * b), r.uniform(0.01, 0.022)
    mp.fogcol = _fogc(r, (0.10, 0.095, 0.085), 0.3, 1.0)
    mp.rates = {E_WATCHER: 0.9, E_STALKER: 0.4}
    mp.wave_rate, mp.red_rate = 0.4, 0.1
    mp.grime, mp.staincol, mp.flash = r.uniform(0.2, 0.6), (0.5, 0.45, 0.4), 0.12
    mp.scatter(["plant", "bench", "sign", "cone", "plant"], int(r.integers(6, 22)), free=op & (mp.w == 0))


GEN = {
    "lobby": gen_lobby, "habitable": gen_habitable, "pipes": gen_pipes, "electrical": gen_electrical,
    "office": gen_office, "hotel": gen_hotel, "lightsout": lambda mp: gen_lobby(mp, dark=True),
    "poolrooms": gen_poolrooms, "run": gen_run, "fun": gen_fun, "flooded": lambda mp: gen_lobby(mp, flooded=True),
    "library": gen_library, "fields": gen_fields, "suburbs": gen_suburbs, "city": gen_city,
    "apartments": gen_apartments, "mall": gen_mall, "hills": gen_hills,
}


def _reach(open_, start):
    """Cells reachable from start by walking (4-neighbours, wrapping)."""
    seen = np.zeros((G, G), bool)
    seen[start] = True
    stack = [start]
    while stack:
        z, x = stack.pop()
        for dz, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            a, b = (z + dz) % G, (x + dx) % G
            if open_[a, b] and not seen[a, b]:
                seen[a, b] = True
                stack.append((a, b))
    return seen


def make(level, seed):
    """Generate one map of `level` (index into LEVELS). The same (level, seed) always gives the same map."""
    name = LEVELS[level]
    rng = np.random.default_rng([seed, level, 7919])
    mp = Map(rng, name)
    GEN[name](mp)
    r = mp.rng
    mp.rates = {k: v * ENT_RATE * (1.0 if k == E_WATCHER else HUNT_RATE) for k, v in mp.rates.items()}
    # sprites that block their cell
    blocks_spr = {SPR_ID[n] for n, s in SPRITES.items() if s[5]}
    pb = np.zeros((G, G), bool)
    for (x, z, i) in mp.props:
        if i in blocks_spr:
            pb[int(z) % G, int(x) % G] = True
    op = mp.open_cells() & ~pb
    best = None
    for _ in range(6):
        zs, xs = np.nonzero(op)
        i = int(r.integers(len(zs)))
        reach = _reach(op, (int(zs[i]), int(xs[i])))
        if best is None or reach.sum() > best.sum():
            best = reach
        if reach.sum() > 0.5 * op.sum():
            break
    mp.reach = best
    mp.pblock = pb
    # spawn: a reachable cell with room around it
    rz, rx = np.nonzero(best & np.roll(best, 1, 0) & np.roll(best, -1, 0) & np.roll(best, 1, 1) & np.roll(best, -1, 1))
    if len(rz) == 0:
        rz, rx = np.nonzero(best)
    if mp.axis is not None:
        zs_, d = mp.axis
        ok = np.isin(rz, zs_)
        if ok.any():
            rz, rx = rz[ok], rx[ok]
    i = int(r.integers(len(rz)))
    yaw = r.uniform(0, 2 * math.pi)
    if mp.axis is not None:
        yaw = 0.0 if mp.axis[1] > 0 else math.pi
    mp.spawn = (rx[i] + 0.5, rz[i] + 0.5, yaw)
    # per-material tint
    tj = mp.tint_j
    mp.tint = 1.0 + r.uniform(-tj, tj, size=(len(MATS), 3)) + r.uniform(-tj, tj, size=(len(MATS), 1))
    return mp
