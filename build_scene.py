"""QtPass strata: the canyon film, built and rendered in Blender.

blender -b --factory-startup -P build_scene.py -- [--stills 1,480,...]
        [--pct 25] [--samples 64] [--save scene.blend] [--anim [--start N]]

The wall is a time axis. A glowing front ("now", see timeline.py) moves
along it: behind the front, every month's layer is as thick as the code
written that month that still survives *now*, so the old layers wear down
as 2026 rewrites them. Ahead of the front there is only sand. The wall,
plateau, end faces, sparks, embers and the language river are rebuilt for
every frame with numpy before it is rendered.
"""
import json
import math
import os
import random
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector, noise

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timeline as TL  # noqa: E402

DATA = TL.DATA
COLS, BANDS, TAGS = DATA["cols"], DATA["bands"], DATA["tags"]
X0, X1 = TL.X0, TL.X1
DEPTH = 18.0
FPS, FRAMES = TL.FPS, TL.FRAMES
LOGO = os.path.join(HERE, "assets", "qtpass-logo.png")
LATO = os.path.expanduser("~/.local/share/fonts/Lato/Lato-Light.ttf")
FOSSIL = (2.8, 0.25, 0.34)  # x, z centre and size (m) of the logo in the 2014 rock

XK = np.array([c["x"] for c in COLS])
NK = len(XK)
NC = len(BANDS)
TS = np.array(DATA["ts"])
YM = np.array(DATA["ym"])  # (cohort, sample) metres
BORN = np.array(DATA["born_x"])
END = np.array(DATA["end_x"])
STATIC_TOP = np.array([c["tops"][-1] for c in COLS])
PR = 28  # plateau rows, front to back (the present face uses the same)
FJ = (np.arange(PR + 1) / PR) ** 1.7  # depth fraction of each row


def frame_t(f):
    return TL.t_of(f)


# ---------------------------------------------------------------- geometry

def disp(x, z):
    """Cliff relief (towards +Y is into the rock); depends on x, z only so
    neighbouring bands meet without cracks."""
    amp = 0.3 + 0.7 * min(z / 3.0, 1.0)
    big = noise.fractal(Vector((x * 0.22, 7.3, z * 0.3)), 1.0, 2.0, 4)
    small = noise.noise(Vector((x * 2.1, 3.1, z * 2.6)))
    grit = noise.noise(Vector((x * 7.0, 5.7, z * 9.0)))
    return big * 0.55 * amp + small * 0.09 + grit * 0.025


DZ = 0.02


def build_disp_grid():
    """disp() sampled per column every 2 cm up to the column's highest top."""
    nz = int((STATIC_TOP.max() + 1.2) / DZ) + 2
    grid = np.zeros((NK, nz), dtype=np.float32)
    for k in range(NK):
        x = float(XK[k])
        top = int((STATIC_TOP[k] + 1.0) / DZ) + 2
        for i in range(min(top, nz)):
            grid[k, i] = disp(x, i * DZ)
        grid[k, top:] = grid[k, top - 1]
    return grid


DG = None


def disp_at(k, z):
    """Vectorised disp() from the grid: k column indices, z heights."""
    iz = np.clip(z / DZ, 0, DG.shape[1] - 1.001)
    i0 = iz.astype(np.int64)
    f = iz - i0
    return DG[k, i0] * (1 - f) + DG[k, i0 + 1] * f


def sizes_at(ts):
    """Every cohort's surviving size (m) at unix time ts."""
    i = int(np.clip(np.searchsorted(TS, ts) - 1, 0, len(TS) - 2))
    f = float(np.clip((ts - TS[i]) / (TS[i + 1] - TS[i]), 0, 1))
    return YM[:, i] * (1 - f) + YM[:, i + 1] * f


def wall_state(frame):
    """Band bottoms and tops (cohort x column) at a frame, plus the front."""
    c = TL.cursor_x(frame)
    size = sizes_at(TL.ts_at_x(c))
    upto = np.minimum(END, c)
    span = np.maximum(upto - BORN, 1e-4)
    frac = np.clip((XK[None, :] - BORN[:, None]) / span[:, None], 0, 1)
    thick = size[:, None] * frac
    vis = XK <= c + 1e-9
    kc = int(np.nonzero(vis)[0].max()) if vis.any() else 0
    thick[:, ~vis] = thick[:, [kc]]
    tops = np.cumsum(thick, axis=0)
    return tops - thick, tops, vis, kc, c


class Wall:
    """The strata: one strip per cohort, rows by its greatest thickness."""

    def __init__(self, mat):
        rnd = random.Random(7)
        vb, vk, vr, vs, vrec, faces = [], [], [], [], [], []
        cols, glows = [], []
        base = 0
        for c, band in enumerate(BANDS):
            rows = max(1, min(8, math.ceil(float(YM[c].max()) / 0.25)))
            recess = rnd.uniform(-0.3, 0.12)
            r = np.arange(rows + 1)
            n = NK * (rows + 1)
            vb.append(np.full(n, c))
            vk.append(np.repeat(np.arange(NK), rows + 1))
            vr.append(np.tile(r / rows, NK))
            vs.append(np.tile(np.sin(np.pi * r / rows) ** 0.35, NK))
            vrec.append(np.full(n, 0.18 + recess))
            cols.append(np.tile(band["rgb"], (n, 1)))
            glows.append(np.full(n, band["glow"]))
            k = np.arange(NK - 1)[:, None]
            rr = np.arange(rows)[None, :]
            a = (base + k * (rows + 1) + rr).ravel()
            b = a + rows + 1
            faces.append(np.stack([a, b, b + 1, a + 1], axis=1))
            base += n
        self.vb = np.concatenate(vb)
        self.vk = np.concatenate(vk)
        self.vr = np.concatenate(vr)
        self.vs = np.concatenate(vs)
        self.vrec = np.concatenate(vrec)
        self.n = base
        order = np.argsort(self.vk, kind="stable")
        bounds = np.searchsorted(self.vk[order], np.arange(NK + 1))
        self.colidx = [order[bounds[k]:bounds[k + 1]] for k in range(NK)]
        faces = np.concatenate(faces)
        self.ob = new_mesh_object("Strata", np.zeros((base, 3)), faces,
                                  np.concatenate(cols), np.concatenate(glows), mat)
        self.co = np.zeros((base, 3), dtype=np.float32)

    def update(self, st):
        lo, hi, vis, kc, _ = st
        k = np.where(vis[self.vk], self.vk, kc)
        l = lo[self.vb, self.vk]
        h = hi[self.vb, self.vk]
        z = l + (h - l) * self.vr
        bulge = -self.vrec * self.vs * np.minimum((h - l) / 0.25, 1.0)
        self.co[:, 0] = XK[k]
        self.co[:, 1] = disp_at(k, z) + bulge
        self.co[:, 2] = z
        set_co(self.ob, self.co)

    def front_profile(self, k):
        idx = self.colidx[k]
        z, y = self.co[idx, 2], self.co[idx, 1]
        o = np.argsort(z, kind="stable")
        return z[o], y[o]


class Plateau:
    def __init__(self, mat):
        shape = np.ones((NK, PR + 1))
        dune = np.zeros((NK, PR + 1))
        cols, glows = [], []
        for k in range(NK):
            x, top = float(XK[k]), float(STATIC_TOP[k])
            y0 = disp(x, top)
            band = top_band(k)
            for j in range(PR + 1):
                f = FJ[j]
                y = y0 + (DEPTH - y0) * f
                if f >= 0.5:
                    u = min((f - 0.5) / 0.5, 1.0)
                    rough = noise.noise(Vector((x * 0.3, y * 0.3, 4.4))) * 0.08
                    shape[k, j] = max((1 - u * u * (3 - 2 * u)) * (1 + rough), 0)
                else:
                    dune[k, j] = noise.noise(Vector((x * 0.4, y * 0.4, 1.7))) \
                        * 0.12 * min(f * 6, 1)
                cols.append([min(1.0, v * 1.12 + 0.03) for v in band["rgb"]])
                glows.append(band["glow"] * 0.6)
        self.shape, self.dune = shape, dune
        k = np.arange(NK - 1)[:, None]
        j = np.arange(PR)[None, :]
        a = (k * (PR + 1) + j).ravel()
        b = a + PR + 1
        faces = np.stack([a, b, b + 1, a + 1], axis=1)
        self.ob = new_mesh_object("Plateau", np.zeros((NK * (PR + 1), 3)), faces,
                                  np.array(cols), np.array(glows), mat)
        self.co = np.zeros((NK, PR + 1, 3), dtype=np.float32)

    def update(self, st):
        _, hi, vis, kc, _ = st
        k = np.where(vis, np.arange(NK), kc)
        top = hi[-1, k]
        y0 = disp_at(k, top)
        self.co[:, :, 0] = XK[k][:, None]
        self.co[:, :, 1] = y0[:, None] + (DEPTH - y0[:, None]) * FJ[None, :]
        self.co[:, :, 2] = top[:, None] * self.shape[k] + \
            self.dune[k] * np.minimum(top / 0.5, 1.0)[:, None]
        set_co(self.ob, self.co.reshape(-1, 3))


def top_band(k):
    tops = COLS[k]["tops"]
    for c in range(len(BANDS) - 1, -1, -1):
        if tops[c] - (tops[c - 1] if c else 0) > 1e-4:
            return BANDS[c]
    return BANDS[0]


class Cap:
    """An end face: +1 is the glowing present, -1 the far end of 2014."""
    NZ = 120

    def __init__(self, mat, sign, glow, wall, plateau):
        self.sign, self.wall, self.plateau = sign, wall, plateau
        n = (PR + 1) * (self.NZ + 1)
        faces = []
        s = self.NZ + 1
        for j in range(PR):
            for r in range(self.NZ):
                a, b = j * s + r, (j + 1) * s + r
                faces.append((a, b, b + 1, a + 1) if sign > 0 else (a, a + 1, b + 1, b))
        rgb = BANDS[-1]["rgb"] if sign > 0 else BANDS[0]["rgb"]
        self.ob = new_mesh_object("Cap+" if sign > 0 else "Cap-", np.zeros((n, 3)),
                                  np.array(faces), np.tile(rgb, (n, 1)),
                                  np.full(n, glow), mat)
        self.co = np.zeros((PR + 1, self.NZ + 1, 3), dtype=np.float32)

    def update(self, st):
        _, hi, _, kc, _ = st
        k = kc if self.sign > 0 else 0
        top = float(hi[-1, k])
        zs, ys = self.wall.front_profile(k)
        pco = self.plateau.co[k]  # (PR+1, 3): the plateau row above this face
        r = np.arange(self.NZ + 1) / self.NZ
        for j in range(PR + 1):
            h = max(top * self.plateau.shape[k, j], 0.0)
            z = h * r
            yf = np.interp(z, zs, ys) if len(zs) else np.zeros_like(z)
            y = yf + (DEPTH - yf) * FJ[j]
            z = z.copy()
            z[-1], y[-1] = pco[j, 2], pco[j, 1]
            pin = min(FJ[j] * 8, 1) * np.minimum((self.NZ - np.arange(self.NZ + 1))
                                                 / (self.NZ * 0.1), 1)
            jag = np.array([noise.noise(Vector((yy * 0.5, zz * 0.5, 9.1)))
                            for yy, zz in zip(y, z)]) * 0.35 * self.sign * pin
            self.co[j, :, 0] = XK[k] + jag
            self.co[j, :, 1] = y
            self.co[j, :, 2] = z
        set_co(self.ob, self.co.reshape(-1, 3))


def new_mesh_object(name, verts, faces, cols=None, glows=None, mat=None):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.asarray(verts, dtype=np.float32).ravel())
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces):
        me.loops.add(faces.size)
        me.loops.foreach_set("vertex_index", faces.ravel())
        me.polygons.add(len(faces))
        me.polygons.foreach_set("loop_start", np.arange(0, faces.size, faces.shape[1]))
        me.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    me.update(calc_edges=True)
    if cols is not None:
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        c4 = np.ones((len(verts), 4), dtype=np.float32)
        c4[:, :3] = cols
        ca.data.foreach_set("color", c4.ravel())
    if glows is not None:
        ga = me.attributes.new("glow", "FLOAT", "POINT")
        ga.data.foreach_set("value", np.asarray(glows, dtype=np.float32))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if mat:
        me.materials.append(mat)
    return ob


def set_co(ob, co):
    ob.data.vertices.foreach_set("co", np.ascontiguousarray(co, dtype=np.float32).ravel())
    ob.data.update()


# ---------------------------------------------------------------- sparks

def unit_ico():
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
    v = np.array([p.co[:] for p in bm.verts])
    f = np.array([[p.index for p in face.verts] for face in bm.faces])
    bm.free()
    return v, f


def author_rgb(i):
    if i == 0:
        return (1.0, 0.7, 0.28)  # the maintainer: gold
    h = (i * 0.618034 + 0.53) % 1.0
    import colorsys
    return colorsys.hsv_to_rgb(h, 0.7, 1.0)


class Sparks:
    """Commits as sparks at the front, embers out of the 2026 eruption."""

    def __init__(self, mat):
        rnd = random.Random(5246)
        self.p = []  # (birth frame, life s, p0, v, gravity, radius, rgb, gain)
        for x, a in DATA["commits"]:
            k = int(np.clip(np.searchsorted(XK, x), 0, NK - 1))
            top = float(STATIC_TOP[k])
            f = TL.frame_of_x(x) + rnd.random()
            if f < 37:  # before the first sample: around the first spark
                f = 40 + rnd.uniform(0, 70)
            p0 = (x + rnd.uniform(-0.04, 0.04), -0.3 - rnd.uniform(0, 0.6),
                  max(0.03, rnd.uniform(0.05, 0.95) * top))
            v = (rnd.uniform(-0.03, 0.03), -rnd.uniform(0.0, 0.05),
                 rnd.uniform(0.0, 0.06))
            self.p.append((f, 2.0, p0, v, 0.0, 0.004 + 0.003 * min(top, 25),
                           author_rgb(a), 1.0))
        # the first commit, alone in the dark
        self.p.append((36, 7.0, (X0 + 0.05, -0.22, 0.04), (0, -0.01, 0.006), 0.0,
                       0.022, (1.0, 0.78, 0.4), 1.6))
        er = np.random.default_rng(2026)
        for f in range(1, FRAMES + 1):
            e = TL.eruption(frame_t(f))
            if e <= 0:
                continue
            c = TL.cursor_x(f)
            k = int(np.clip(np.searchsorted(XK, c), 0, NK - 1))
            top = float(STATIC_TOP[k])
            for _ in range(er.poisson(90 * e / FPS)):
                p0 = (c + er.uniform(-0.8, 0.3), er.uniform(-0.3, 5.0),
                      top * er.uniform(0.9, 1.0))
                v = (er.uniform(-1.5, 1.5), er.uniform(-1.2, 1.5), er.uniform(2.5, 8))
                self.p.append((f + er.random(), er.uniform(1.6, 3.4), p0, v, -5.0,
                               er.uniform(0.025, 0.07), (1.0, 0.36, 0.08), 1.3))
        self.birth = np.array([q[0] for q in self.p])
        self.life = np.array([q[1] for q in self.p]) * FPS
        self.ico_v, self.ico_f = unit_ico()
        me = bpy.data.meshes.new("Sparks")
        me.materials.append(mat)
        self.ob = bpy.data.objects.new("Sparks", me)
        self.ob.visible_shadow = False
        bpy.context.scene.collection.objects.link(self.ob)

    def update(self, frame):
        age = frame - self.birth
        live = np.nonzero((age >= 0) & (age < self.life))[0]
        nv = len(self.ico_v)
        verts = np.zeros((len(live) * nv, 3))
        faces = np.zeros((len(live) * len(self.ico_f), 3), dtype=np.int64)
        col = np.ones((len(live) * nv, 4), dtype=np.float32)
        heat = np.zeros(len(live) * nv, dtype=np.float32)
        for n, i in enumerate(live):
            _, life, p0, v, g, rad, rgb, gain = self.p[i]
            a = age[i] / FPS
            u = age[i] / self.life[i]
            hval = min(a / 0.12, 1.0) * (1 - u) ** 1.6 * gain
            c = np.array(p0) + np.array(v) * a
            c[2] += 0.5 * g * a * a
            c[2] = max(c[2], 0.01)
            s = slice(n * nv, (n + 1) * nv)
            verts[s] = c + self.ico_v * rad * (0.6 + 0.4 * (1 - u))
            faces[n * len(self.ico_f):(n + 1) * len(self.ico_f)] = self.ico_f + n * nv
            col[s, :3] = rgb
            heat[s] = hval
        me = self.ob.data
        me.clear_geometry()
        for name in ("Col", "heat"):
            a = me.attributes.get(name)
            if a:
                me.attributes.remove(a)
        me.from_pydata(verts.tolist(), [], faces.tolist())
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        ga = me.attributes.new("heat", "FLOAT", "POINT")
        if len(verts):
            ca.data.foreach_set("color", col.ravel())
            ga.data.foreach_set("value", heat)
        me.update()


class HighWater:
    """Where the rock once stood: a faint line along each column's old top,
    shown only where the wall has since worn down beneath it."""
    TH = 0.08

    def __init__(self, mat):
        n = NK
        verts = np.zeros((n * 2, 3))
        faces = [(2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1) for k in range(n - 1)]
        self.ob = new_mesh_object("HighWater", verts, np.array(faces), mat=mat)
        self.ob.visible_shadow = False
        me = self.ob.data
        self.a = me.attributes.new("heat", "FLOAT", "POINT")
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        ca.data.foreach_set("color", np.tile([1.0, 0.86, 0.66, 1.0], n * 2)
                            .astype(np.float32))
        self.co = np.zeros((n, 2, 3), dtype=np.float32)
        self.y = np.array([disp(float(XK[k]), float(STATIC_TOP[k])) - 0.06
                           for k in range(n)])

    def update(self, st):
        _, hi, vis, kc, _ = st
        k = np.where(vis, np.arange(NK), kc)
        top = STATIC_TOP[k]
        worn = np.clip((top - hi[-1, k]) / 0.4, 0, 1) * vis
        self.co[:, 0, 0] = self.co[:, 1, 0] = XK[k]
        self.co[:, :, 1] = self.y[k][:, None]
        self.co[:, 0, 2] = top
        self.co[:, 1, 2] = top + 0.03 + 0.012 * top
        set_co(self.ob, self.co.reshape(-1, 3))
        self.a.data.foreach_set("value", np.repeat(worn, 2).astype(np.float32))


# ---------------------------------------------------------------- river

class River:
    """The languages: a stream of light at the foot of the wall, wider with
    every locale, and a tributary where new ones joined."""

    def __init__(self, mat):
        loc = DATA["locales"]
        lx = np.array([p[0] for p in loc])
        ln = np.array([p[1] for p in loc], dtype=float)
        self.xs = np.arange(X0, X1 + 1e-6, 0.1)
        n = np.interp(self.xs, lx, ln)
        n = np.convolve(np.pad(n, 6, mode="edge"), np.ones(13) / 13, mode="valid")
        w = 0.05 + 0.011 * n
        inner = -0.5 - 0.06 * np.sin(0.7 * self.xs)
        verts, faces, segs = [], [], []
        m = len(self.xs)
        for i, x in enumerate(self.xs):
            for side in (0.0, 0.5, 1.0):
                verts.append((x, inner[i] - w[i] * side, 0.0))
        for i in range(m - 1):
            for s in range(2):
                a, b = i * 3 + s, (i + 1) * 3 + s
                faces.append((a, b, b + 1, a + 1))
        self.main = (0, m * 3)
        self.mid = inner - w / 2
        self.trib = []  # (vertex range, join x, start frame, along s per vertex)
        prev = ln[0]
        for x, cnt in zip(lx, ln):
            d = cnt - prev
            prev = cnt
            if d <= 0 or x <= X0:
                continue
            ym = float(np.interp(x, self.xs, self.mid))
            yo = float(np.interp(x, self.xs, inner - w))
            p0 = np.array([x - 3.0 - 0.1 * d, yo - 3.2 - 0.12 * d])
            p1 = np.array([x - 0.6, yo - 1.4])
            p2 = np.array([x, ym])
            tw = 0.025 + 0.016 * d
            start = len(verts)
            ss = []
            for i in range(25):
                s = i / 24
                p = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s * s * p2
                dp = 2 * (1 - s) * (p1 - p0) + 2 * s * (p2 - p1)
                nrm = np.array([-dp[1], dp[0]]) / (np.linalg.norm(dp) + 1e-9)
                for side in (-0.5, 0.5):
                    q = p + nrm * tw * side * (0.5 + 0.5 * s)
                    verts.append((q[0], q[1], 0.002))
                    ss.append(s)
            for i in range(24):
                a = start + i * 2
                faces.append((a, a + 2, a + 3, a + 1))
            self.trib.append(((start, len(verts)), x, TL.frame_of_x(x) - 30,
                              np.array(ss)))
        self.base = np.array(verts, dtype=np.float32)
        self.ob = new_mesh_object("River", self.base, np.array(faces), mat=mat)
        self.ob.visible_shadow = False
        self.co = self.base.copy()

    def update(self, frame):
        c = TL.cursor_x(frame)
        co = self.co
        co[:] = self.base
        a, b = self.main
        seg = co[a:b]
        ahead = seg[:, 0] > c
        if ahead.any():
            i = max(int(np.nonzero(~ahead)[0].max()) if (~ahead).any() else 0, 0)
            row = (i // 3) * 3
            for s in range(3):
                seg[(np.arange(len(seg)) % 3 == s) & ahead] = seg[row + s]
        for (ta, tb), _, f0, ss in self.trib:
            p = min(max((frame - f0) / (1.5 * FPS), 0.0), 1.0)
            part = co[ta:tb]
            hide = ss > p + 1e-6
            if hide.all():
                part[:] = (part[0][0], part[0][1], -0.5)
            elif hide.any():
                last = int(np.nonzero(~hide)[0].max())
                part[hide] = part[last - (last % 2) + (np.nonzero(hide)[0] % 2)]
        set_co(self.ob, co)


# ---------------------------------------------------------------- materials

def node(nt, kind, loc, **props):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def key_w(sock, rate=lambda t: 0.04):
    """Animate a 4D noise's W: lava that crawls, faster when 2026 erupts."""
    w = 0.0
    for f in range(1, FRAMES + 2, FPS):
        sock.default_value = w
        sock.keyframe_insert("default_value", frame=f)
        w += rate(frame_t(f))


def lava_rate(t):
    return 0.03 + 0.16 * TL.eruption(t)


def fissures(nt, tc, x, y, scale, width):
    """Meandering crack lines (ridged 4D noise); returns a 0..1 socket."""
    n = node(nt, "ShaderNodeTexNoise", (x, y), noise_dimensions="4D")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = 6
    n.inputs["Roughness"].default_value = 0.55
    n.inputs["Distortion"].default_value = 0.6
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    key_w(n.inputs["W"], lava_rate)
    sub = node(nt, "ShaderNodeMath", (x + 180, y), operation="SUBTRACT")
    sub.inputs[1].default_value = 0.5
    nt.links.new(n.outputs["Fac"], sub.inputs[0])
    ab = node(nt, "ShaderNodeMath", (x + 340, y), operation="ABSOLUTE")
    nt.links.new(sub.outputs["Value"], ab.inputs[0])
    mr = node(nt, "ShaderNodeMapRange", (x + 500, y))
    mr.inputs["From Min"].default_value = 0.0
    mr.inputs["From Max"].default_value = width
    mr.inputs["To Min"].default_value = 1.0
    mr.inputs["To Max"].default_value = 0.0
    nt.links.new(ab.outputs["Value"], mr.inputs["Value"])
    return mr.outputs["Result"]


def fossil_mask(nt, tc):
    """The QtPass logo pressed into the first layer like a fossil.

    Returns (relief, darken): relief is the raised shape (shackle highest,
    keyhole sunk), darken is where the rock is stained darker.
    """
    cx, cz, size = FOSSIL
    sep = node(nt, "ShaderNodeSeparateXYZ", (-1500, 600))
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    u = node(nt, "ShaderNodeMath", (-1300, 700), operation="MULTIPLY_ADD")
    nt.links.new(sep.outputs["X"], u.inputs[0])
    u.inputs[1].default_value = 1 / size
    u.inputs[2].default_value = 0.5 - cx / size
    v = node(nt, "ShaderNodeMath", (-1300, 550), operation="MULTIPLY_ADD")
    nt.links.new(sep.outputs["Z"], v.inputs[0])
    v.inputs[1].default_value = 1 / size
    v.inputs[2].default_value = 0.5 - cz / size
    uv = node(nt, "ShaderNodeCombineXYZ", (-1100, 600))
    nt.links.new(u.outputs["Value"], uv.inputs["X"])
    nt.links.new(v.outputs["Value"], uv.inputs["Y"])
    img = node(nt, "ShaderNodeTexImage", (-900, 600), extension="CLIP",
               interpolation="Cubic")
    img.image = bpy.data.images.load(LOGO)
    nt.links.new(uv.outputs["Vector"], img.inputs["Vector"])
    lum = node(nt, "ShaderNodeRGBToBW", (-600, 700))
    nt.links.new(img.outputs["Color"], lum.inputs["Color"])
    l2 = node(nt, "ShaderNodeMath", (-450, 700), operation="ADD")
    nt.links.new(lum.outputs["Val"], l2.inputs[0])
    l2.inputs[1].default_value = 0.25
    relief = node(nt, "ShaderNodeMath", (-300, 700), operation="MULTIPLY")
    nt.links.new(l2.outputs["Value"], relief.inputs[0])
    nt.links.new(img.outputs["Alpha"], relief.inputs[1])
    l3 = node(nt, "ShaderNodeMath", (-450, 550), operation="SUBTRACT")
    l3.inputs[0].default_value = 0.75
    nt.links.new(lum.outputs["Val"], l3.inputs[1])
    dark = node(nt, "ShaderNodeMath", (-300, 550), operation="MULTIPLY",
                use_clamp=True)
    nt.links.new(l3.outputs["Value"], dark.inputs[0])
    nt.links.new(img.outputs["Alpha"], dark.inputs[1])
    return relief.outputs["Value"], dark.outputs["Value"]


def rock_material():
    m = bpy.data.materials.new("Rock")
    nt = m.node_tree
    nt.nodes.clear()
    out = node(nt, "ShaderNodeOutputMaterial", (900, 0))
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", (600, 0))
    bsdf.inputs["Roughness"].default_value = 0.93
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    tc = node(nt, "ShaderNodeTexCoord", (-900, 0))
    col = node(nt, "ShaderNodeAttribute", (-600, 300), attribute_name="Col")
    glow = node(nt, "ShaderNodeAttribute", (-600, -500), attribute_name="glow")

    nz = node(nt, "ShaderNodeTexNoise", (-600, 100))
    nz.inputs["Scale"].default_value = 0.9
    nz.inputs["Detail"].default_value = 8
    nz.inputs["Roughness"].default_value = 0.62
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    mr = node(nt, "ShaderNodeMapRange", (-400, 100))
    mr.inputs["To Min"].default_value = 0.72
    mr.inputs["To Max"].default_value = 1.18
    nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])

    wave = node(nt, "ShaderNodeTexWave", (-600, -150), wave_type="BANDS",
                bands_direction="Z")
    wave.inputs["Scale"].default_value = 3.5
    wave.inputs["Distortion"].default_value = 4.0
    wave.inputs["Detail"].default_value = 3.0
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    wr = node(nt, "ShaderNodeMapRange", (-400, -150))
    wr.inputs["To Min"].default_value = 0.86
    wr.inputs["To Max"].default_value = 1.0
    nt.links.new(wave.outputs["Fac"], wr.inputs["Value"])

    m1 = node(nt, "ShaderNodeMix", (-150, 250), data_type="RGBA", blend_type="MULTIPLY")
    m1.inputs["Factor"].default_value = 1.0
    nt.links.new(col.outputs["Color"], m1.inputs["A"])
    nt.links.new(mr.outputs["Result"], m1.inputs["B"])
    m2 = node(nt, "ShaderNodeMix", (100, 250), data_type="RGBA", blend_type="MULTIPLY")
    m2.inputs["Factor"].default_value = 1.0
    nt.links.new(m1.outputs["Result"], m2.inputs["A"])
    nt.links.new(wr.outputs["Result"], m2.inputs["B"])
    fossil, fdark = fossil_mask(nt, tc)
    m3 = node(nt, "ShaderNodeMix", (350, 250), data_type="RGBA", blend_type="MULTIPLY")
    nt.links.new(fdark, m3.inputs["Factor"])
    nt.links.new(m2.outputs["Result"], m3.inputs["A"])
    m3.inputs["B"].default_value = (0.45, 0.4, 0.36, 1)
    nt.links.new(m3.outputs["Result"], bsdf.inputs["Base Color"])

    bn = node(nt, "ShaderNodeTexNoise", (-600, -350))
    bn.inputs["Scale"].default_value = 9.0
    bn.inputs["Detail"].default_value = 12
    nt.links.new(tc.outputs["Object"], bn.inputs["Vector"])
    bump = node(nt, "ShaderNodeBump", (300, -250))
    bump.inputs["Strength"].default_value = 0.8
    add = node(nt, "ShaderNodeMath", (100, -250), operation="ADD")
    nt.links.new(bn.outputs["Fac"], add.inputs[0])
    nt.links.new(wave.outputs["Fac"], add.inputs[1])
    relief = node(nt, "ShaderNodeMath", (200, -400), operation="MULTIPLY_ADD")
    nt.links.new(fossil, relief.inputs[0])
    relief.inputs[1].default_value = 2.5
    nt.links.new(add.outputs["Value"], relief.inputs[2])
    nt.links.new(relief.outputs["Value"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    cracks = fissures(nt, tc, -1000, -700, 0.9, 0.012)
    mask = node(nt, "ShaderNodeTexNoise", (-600, -950))
    mask.inputs["Scale"].default_value = 0.45
    mask.inputs["Detail"].default_value = 3
    nt.links.new(tc.outputs["Object"], mask.inputs["Vector"])
    mm = node(nt, "ShaderNodeMapRange", (-400, -950))
    mm.inputs["From Min"].default_value = 0.5
    mm.inputs["From Max"].default_value = 0.6
    nt.links.new(mask.outputs["Fac"], mm.inputs["Value"])
    vm = node(nt, "ShaderNodeMath", (-250, -800), operation="MULTIPLY")
    nt.links.new(cracks, vm.inputs[0])
    nt.links.new(mm.outputs["Result"], vm.inputs[1])
    gate = node(nt, "ShaderNodeMath", (-150, -600), operation="MULTIPLY")
    nt.links.new(vm.outputs["Value"], gate.inputs[0])
    nt.links.new(glow.outputs["Fac"], gate.inputs[1])
    strength = node(nt, "ShaderNodeMath", (100, -600), operation="MULTIPLY")
    strength.inputs[1].default_value = 45.0
    nt.links.new(gate.outputs["Value"], strength.inputs[0])
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.32, 0.07, 1)
    nt.links.new(strength.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m


def molten_material():
    m = bpy.data.materials.new("Molten")
    nt = m.node_tree
    nt.nodes.clear()
    out = node(nt, "ShaderNodeOutputMaterial", (700, 0))
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", (400, 0))
    bsdf.inputs["Base Color"].default_value = (0.03, 0.025, 0.025, 1)
    bsdf.inputs["Roughness"].default_value = 0.7
    tc = node(nt, "ShaderNodeTexCoord", (-700, 0))
    cracks = fissures(nt, tc, -900, 0, 0.7, 0.02)
    hot = node(nt, "ShaderNodeMix", (-200, 0), data_type="RGBA")
    hot.inputs["A"].default_value = (0, 0, 0, 1)
    hot.inputs["B"].default_value = (1.0, 0.5, 0.12, 1)
    nt.links.new(cracks, hot.inputs["Factor"])
    mask = node(nt, "ShaderNodeTexNoise", (-450, -250))
    mask.inputs["Scale"].default_value = 0.25
    mask.inputs["Detail"].default_value = 4
    nt.links.new(tc.outputs["Object"], mask.inputs["Vector"])
    mm = node(nt, "ShaderNodeMapRange", (-250, -250))
    mm.inputs["From Min"].default_value = 0.45
    mm.inputs["From Max"].default_value = 0.6
    mm.inputs["To Min"].default_value = 0.0
    nt.links.new(mask.outputs["Fac"], mm.inputs["Value"])
    mix = node(nt, "ShaderNodeMix", (0, 0), data_type="RGBA", blend_type="MULTIPLY")
    mix.inputs["Factor"].default_value = 1.0
    nt.links.new(hot.outputs["Result"], mix.inputs["A"])
    nt.links.new(mm.outputs["Result"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 40.0
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m


def emission_attr_material(name, strength):
    """Emission from the 'Col' and 'heat' point attributes (sparks)."""
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = node(nt, "ShaderNodeOutputMaterial", (500, 0))
    em = node(nt, "ShaderNodeEmission", (250, 0))
    col = node(nt, "ShaderNodeAttribute", (-200, 100), attribute_name="Col")
    heat = node(nt, "ShaderNodeAttribute", (-200, -100), attribute_name="heat")
    mul = node(nt, "ShaderNodeMath", (50, -100), operation="MULTIPLY")
    mul.inputs[1].default_value = strength
    nt.links.new(heat.outputs["Fac"], mul.inputs[0])
    nt.links.new(col.outputs["Color"], em.inputs["Color"])
    nt.links.new(mul.outputs["Value"], em.inputs["Strength"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def river_material():
    m = bpy.data.materials.new("River")
    nt = m.node_tree
    nt.nodes.clear()
    out = node(nt, "ShaderNodeOutputMaterial", (600, 0))
    em = node(nt, "ShaderNodeEmission", (350, 0))
    em.inputs["Color"].default_value = (0.13, 0.4, 1.0, 1)
    tc = node(nt, "ShaderNodeTexCoord", (-700, 0))
    wave = node(nt, "ShaderNodeTexWave", (-450, 0), wave_type="BANDS",
                bands_direction="X")
    wave.inputs["Scale"].default_value = 0.9
    wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 4.0
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    ph = wave.inputs["Phase Offset"]
    for f in (1, FRAMES + 1):
        ph.default_value = -(f / FPS) * 2.2
        ph.keyframe_insert("default_value", frame=f)
    mr = node(nt, "ShaderNodeMapRange", (-200, 0))
    mr.inputs["To Min"].default_value = 0.35
    mr.inputs["To Max"].default_value = 1.6
    nt.links.new(wave.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], em.inputs["Strength"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    linear_fcurves(m.node_tree)
    return m


def linear_fcurves(idb):
    ad = idb.animation_data
    if not (ad and ad.action):
        return
    for fc in iter_fcurves(ad.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def iter_fcurves(action):
    if hasattr(action, "fcurves"):
        try:
            yield from action.fcurves
            return
        except Exception:
            pass
    for layer in getattr(action, "layers", []):  # Blender 5 layered actions
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves


def simple_material(name, rgb, rough=0.95, emit=None, strength=0.0):
    m = bpy.data.materials.new(name)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = strength
    return m


def sand_material():
    m = bpy.data.materials.new("Sand")
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 1.0
    tc = node(nt, "ShaderNodeTexCoord", (-800, 0))
    nz = node(nt, "ShaderNodeTexNoise", (-600, 0))
    nz.inputs["Scale"].default_value = 0.08
    nz.inputs["Detail"].default_value = 6
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    ramp = node(nt, "ShaderNodeValToRGB", (-350, 0))
    ramp.color_ramp.elements[0].color = (0.32, 0.2, 0.12, 1)
    ramp.color_ramp.elements[1].color = (0.62, 0.45, 0.3, 1)
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    rip = node(nt, "ShaderNodeTexWave", (-600, -300), wave_type="BANDS",
               bands_direction="X")
    rip.inputs["Scale"].default_value = 0.6
    rip.inputs["Distortion"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], rip.inputs["Vector"])
    bump = node(nt, "ShaderNodeBump", (-200, -300))
    bump.inputs["Strength"].default_value = 0.25
    nt.links.new(rip.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


# ---------------------------------------------------------------- set

def build_ground(mat):
    bpy.ops.mesh.primitive_plane_add(size=1, location=(40, 40, -0.02))
    g = bpy.context.active_object
    g.name = "Ground"
    g.scale = (900, 700, 1)
    g.data.materials.append(mat)


def build_mesas(mat):
    rnd = random.Random(3)
    for i in range(11):
        bpy.ops.mesh.primitive_cube_add(size=2)
        o = bpy.context.active_object
        o.name = f"Mesa{i}"
        h = rnd.uniform(8, 30)
        o.location = (rnd.uniform(-150, 230), rnd.uniform(140, 330), h - 3)
        o.scale = (rnd.uniform(20, 55), rnd.uniform(12, 30), h)
        sub = o.modifiers.new("s", "SUBSURF")
        sub.subdivision_type = "SIMPLE"
        sub.levels = sub.render_levels = 5
        d = o.modifiers.new("d", "DISPLACE")
        tex = bpy.data.textures.new(f"mesa{i}", "CLOUDS")
        tex.noise_scale = 0.6
        d.texture = tex
        d.strength = 0.22
        o.rotation_euler = (0, 0, rnd.uniform(-0.4, 0.4))
        o.data.materials.append(mat)
        bpy.ops.object.shade_smooth()


def add_text(body, loc, size, mat, rot=(math.pi / 2, 0, 0), align="CENTER"):
    cu = bpy.data.curves.new(body, "FONT")
    cu.body = body
    cu.size = size
    cu.extrude = size * 0.04
    cu.font = bpy.data.fonts.load(LATO, check_existing=True)
    cu.align_x = align
    ob = bpy.data.objects.new("T " + body, cu)
    ob.location = loc
    ob.rotation_euler = rot
    bpy.context.scene.collection.objects.link(ob)
    cu.materials.append(mat)
    ob.visible_shadow = False
    return ob


def beacon_material(name):
    """A shaft of light, fading upwards; returns (material, fade socket)."""
    m = bpy.data.materials.new(name)
    m.surface_render_method = "BLENDED"
    nt = m.node_tree
    nt.nodes.clear()
    out = node(nt, "ShaderNodeOutputMaterial", (700, 0))
    mix = node(nt, "ShaderNodeMixShader", (500, 0))
    tr = node(nt, "ShaderNodeBsdfTransparent", (250, 100))
    em = node(nt, "ShaderNodeEmission", (250, -100))
    em.inputs["Color"].default_value = (1.0, 0.84, 0.6, 1)
    em.inputs["Strength"].default_value = 10.0
    tc = node(nt, "ShaderNodeTexCoord", (-600, 0))
    sep = node(nt, "ShaderNodeSeparateXYZ", (-400, 0))
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    inv = node(nt, "ShaderNodeMath", (-200, 0), operation="SUBTRACT")
    inv.inputs[0].default_value = 1.0
    nt.links.new(sep.outputs["Z"], inv.inputs[1])
    pw = node(nt, "ShaderNodeMath", (-50, 0), operation="POWER")
    pw.inputs[1].default_value = 2.2
    nt.links.new(inv.outputs["Value"], pw.inputs[0])
    fade = node(nt, "ShaderNodeValue", (-200, -200))
    fade.outputs[0].default_value = 0.0
    a = node(nt, "ShaderNodeMath", (100, 0), operation="MULTIPLY", use_clamp=True)
    nt.links.new(pw.outputs["Value"], a.inputs[0])
    nt.links.new(fade.outputs[0], a.inputs[1])
    nt.links.new(a.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m, fade.outputs[0]


def build_releases(river):
    """A beacon, a flash of light and a name at the foot of the wall for
    every release, lit as the front passes it."""
    tag_mat = simple_material("Tag", (0.9, 0.8, 0.55), 0.4, (1.0, 0.78, 0.45), 2.5)
    for t in TAGS:
        x, top = t["x"], t["top"]
        f0 = TL.frame_of_x(x)
        for r, depth, gain in ((0.08, 60, 1.0), (0.6, 40, 0.035)):
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=r, depth=depth,
                                                location=(x, 2.5, depth / 2))
            ob = bpy.context.active_object
            ob.name = f"Beacon {t['name']} {r}"
            ob.visible_shadow = False
            m, fade = beacon_material(ob.name)
            ob.data.materials.append(m)
            for f, v in ((f0 - 3, 0), (f0 + 4, 1), (f0 + int(2.5 * FPS), 0.25),
                         (f0 + 7 * FPS, 0)):
                fade.default_value = v * gain
                fade.keyframe_insert("default_value", frame=f)
        ld = bpy.data.lights.new(f"Flash {t['name']}", "POINT")
        ld.color = (1.0, 0.82, 0.55)
        ld.shadow_soft_size = 0.3
        lo = bpy.data.objects.new(ld.name, ld)
        lo.location = (x, -1.2 - top * 0.15, 0.6 + top * 0.4)
        bpy.context.scene.collection.objects.link(lo)
        peak = 60 + 40 * top * top
        for f, v in ((f0 - 3, 0), (f0 + 3, peak), (f0 + 3 * FPS, peak * 0.08),
                     (f0 + 6 * FPS, 0)):
            ld.energy = v
            ld.keyframe_insert("energy", frame=f)
        # on the wall face at half height, where the camera is looking; it
        # leaves again before the wall wears down beneath it
        size = min(max(top * 0.055, 0.13), 0.9)
        z = max(top * 0.45, 0.1)
        txt = add_text(t["name"], (x, -0.95 - size * 0.3, z), size, tag_mat)
        for f, s in ((f0 - 1, 0.001), (f0 + 8, 1.0), (f0 + 5 * FPS, 1.0),
                     (f0 + 6 * FPS, 0.001)):
            txt.scale = (s, s, s)
            txt.keyframe_insert("scale", frame=f)


# ---------------------------------------------------------------- light

def setup_world():
    s = bpy.context.scene
    w = bpy.data.worlds.new("Sky")
    s.world = w
    nt = w.node_tree
    nt.nodes.clear()
    out = node(nt, "ShaderNodeOutputWorld", (700, 0))
    bg = node(nt, "ShaderNodeBackground", (500, 0))
    sky = node(nt, "ShaderNodeTexSky", (-150, 150), sky_type="MULTIPLE_SCATTERING")
    for prop, val in (("sun_disc", True), ("air_density", 1.4),
                      ("aerosol_density", 2.5), ("dust_density", 2.5)):
        if hasattr(sky, prop):
            setattr(sky, prop, val)
    # stars: sparse Voronoi points on the view direction, above the horizon
    tc = node(nt, "ShaderNodeTexCoord", (-700, -200))
    vor = node(nt, "ShaderNodeTexVoronoi", (-450, -200))
    vor.inputs["Scale"].default_value = 420.0
    nt.links.new(tc.outputs["Generated"], vor.inputs["Vector"])
    pt = node(nt, "ShaderNodeMapRange", (-250, -200))
    pt.inputs["From Min"].default_value = 0.0
    pt.inputs["From Max"].default_value = 0.09
    pt.inputs["To Min"].default_value = 1.0
    pt.inputs["To Max"].default_value = 0.0
    nt.links.new(vor.outputs["Distance"], pt.inputs["Value"])
    bright = node(nt, "ShaderNodeMath", (-80, -200), operation="POWER")
    bright.inputs[1].default_value = 6.0
    nt.links.new(pt.outputs["Result"], bright.inputs[0])
    tw = node(nt, "ShaderNodeMath", (60, -260), operation="MULTIPLY")
    nt.links.new(bright.outputs["Value"], tw.inputs[0])
    nt.links.new(vor.outputs["Color"], tw.inputs[1])  # varied brightness
    sep = node(nt, "ShaderNodeSeparateXYZ", (-450, -420))
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    hz = node(nt, "ShaderNodeMapRange", (-250, -420))
    hz.inputs["From Min"].default_value = 0.02
    hz.inputs["From Max"].default_value = 0.2
    nt.links.new(sep.outputs["Z"], hz.inputs["Value"])
    star = node(nt, "ShaderNodeMath", (200, -300), operation="MULTIPLY")
    nt.links.new(tw.outputs["Value"], star.inputs[0])
    nt.links.new(hz.outputs["Result"], star.inputs[1])
    level = node(nt, "ShaderNodeValue", (0, -500))
    lv = node(nt, "ShaderNodeMath", (320, -380), operation="MULTIPLY")
    nt.links.new(star.outputs["Value"], lv.inputs[0])
    nt.links.new(level.outputs[0], lv.inputs[1])
    lv2 = node(nt, "ShaderNodeMath", (420, -380), operation="MULTIPLY")
    lv2.inputs[1].default_value = 30.0
    nt.links.new(lv.outputs["Value"], lv2.inputs[0])
    tint = node(nt, "ShaderNodeMix", (150, 100), data_type="RGBA", blend_type="ADD")
    nt.links.new(level.outputs[0], tint.inputs["Factor"])
    nt.links.new(sky.outputs["Color"], tint.inputs["A"])
    tint.inputs["B"].default_value = (0.012, 0.02, 0.05, 1)
    add = node(nt, "ShaderNodeMix", (330, 100), data_type="RGBA", blend_type="ADD")
    add.inputs["Factor"].default_value = 1.0
    nt.links.new(tint.outputs["Result"], add.inputs["A"])
    nt.links.new(lv2.outputs["Value"], add.inputs["B"])
    nt.links.new(add.outputs["Result"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.25
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    for f in range(1, FRAMES + 2, 6):
        level.outputs[0].default_value = TL.night(frame_t(f))
        level.outputs[0].keyframe_insert("default_value", frame=f)
    return sky


def build_mist():
    """Ground mist: a bounded volume, densest at the canyon floor."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(50, 130, 30))
    box = bpy.context.active_object
    box.name = "Mist"
    box.scale = (560, 560, 60)
    box.visible_shadow = False
    m = bpy.data.materials.new("Mist")
    nt = m.node_tree
    nt.nodes.clear()
    out = node(nt, "ShaderNodeOutputMaterial", (600, 0))
    vol = node(nt, "ShaderNodeVolumePrincipled", (350, 0))
    vol.inputs["Color"].default_value = (0.95, 0.8, 0.66, 1)
    vol.inputs["Anisotropy"].default_value = 0.45
    tc = node(nt, "ShaderNodeNewGeometry", (-500, 0))
    sep = node(nt, "ShaderNodeSeparateXYZ", (-300, 0))
    nt.links.new(tc.outputs["Position"], sep.inputs["Vector"])
    k = node(nt, "ShaderNodeMath", (-120, 0), operation="MULTIPLY")
    k.inputs[1].default_value = -0.35
    nt.links.new(sep.outputs["Z"], k.inputs[0])
    ex = node(nt, "ShaderNodeMath", (40, 0), operation="EXPONENT")
    nt.links.new(k.outputs["Value"], ex.inputs[0])
    dens = node(nt, "ShaderNodeMath", (200, 0), operation="MULTIPLY")
    nt.links.new(ex.outputs["Value"], dens.inputs[0])
    haze = node(nt, "ShaderNodeMath", (280, -150), operation="ADD")
    haze.inputs[1].default_value = 0.0012
    nt.links.new(dens.outputs["Value"], haze.inputs[0])
    nt.links.new(haze.outputs["Value"], vol.inputs["Density"])
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    box.data.materials.append(m)
    return dens.inputs[1]


class Plume:
    """Ash and glow over the 2026 front, as hard as the eruption."""

    def __init__(self):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        self.ob = bpy.context.active_object
        self.ob.name = "Plume"
        self.ob.scale = (30, 26, 48)
        self.ob.visible_shadow = False
        m = bpy.data.materials.new("Plume")
        nt = m.node_tree
        nt.nodes.clear()
        out = node(nt, "ShaderNodeOutputMaterial", (900, 0))
        vol = node(nt, "ShaderNodeVolumePrincipled", (650, 0))
        vol.inputs["Color"].default_value = (0.05, 0.045, 0.045, 1)
        vol.inputs["Emission Color"].default_value = (1.0, 0.33, 0.07, 1)
        tc = node(nt, "ShaderNodeTexCoord", (-900, 0))
        nz = node(nt, "ShaderNodeTexNoise", (-650, 150), noise_dimensions="4D")
        nz.inputs["Scale"].default_value = 3.0
        nz.inputs["Detail"].default_value = 6
        nz.inputs["Roughness"].default_value = 0.6
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        key_w(nz.inputs["W"], lambda t: 0.25)
        cl = node(nt, "ShaderNodeMapRange", (-450, 150))
        cl.inputs["From Min"].default_value = 0.45
        cl.inputs["From Max"].default_value = 0.75
        nt.links.new(nz.outputs["Fac"], cl.inputs["Value"])
        sep = node(nt, "ShaderNodeSeparateXYZ", (-650, -200))
        nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
        # allowed radius grows with height: a column that spreads
        rx = node(nt, "ShaderNodeMath", (-450, -150), operation="SUBTRACT")
        rx.inputs[1].default_value = 0.5
        nt.links.new(sep.outputs["X"], rx.inputs[0])
        ry = node(nt, "ShaderNodeMath", (-450, -250), operation="SUBTRACT")
        ry.inputs[1].default_value = 0.5
        nt.links.new(sep.outputs["Y"], ry.inputs[0])
        cmb = node(nt, "ShaderNodeCombineXYZ", (-300, -200))
        nt.links.new(rx.outputs["Value"], cmb.inputs["X"])
        nt.links.new(ry.outputs["Value"], cmb.inputs["Y"])
        ln = node(nt, "ShaderNodeVectorMath", (-150, -200), operation="LENGTH")
        nt.links.new(cmb.outputs["Vector"], ln.inputs[0])
        allow = node(nt, "ShaderNodeMath", (-300, -350), operation="MULTIPLY_ADD")
        nt.links.new(sep.outputs["Z"], allow.inputs[0])
        allow.inputs[1].default_value = 0.28
        allow.inputs[2].default_value = 0.06
        ratio = node(nt, "ShaderNodeMath", (0, -250), operation="DIVIDE")
        nt.links.new(ln.outputs["Value"], ratio.inputs[0])
        nt.links.new(allow.outputs["Value"], ratio.inputs[1])
        edge = node(nt, "ShaderNodeMapRange", (150, -250))
        edge.inputs["From Min"].default_value = 1.0
        edge.inputs["From Max"].default_value = 0.4
        nt.links.new(ratio.outputs["Value"], edge.inputs["Value"])
        topf = node(nt, "ShaderNodeMapRange", (150, -420))
        topf.inputs["From Min"].default_value = 1.0
        topf.inputs["From Max"].default_value = 0.6
        nt.links.new(sep.outputs["Z"], topf.inputs["Value"])
        d1 = node(nt, "ShaderNodeMath", (320, 0), operation="MULTIPLY")
        nt.links.new(cl.outputs["Result"], d1.inputs[0])
        nt.links.new(edge.outputs["Result"], d1.inputs[1])
        botf = node(nt, "ShaderNodeMapRange", (150, -560))
        botf.inputs["From Min"].default_value = 0.0
        botf.inputs["From Max"].default_value = 0.12
        nt.links.new(sep.outputs["Z"], botf.inputs["Value"])
        tb = node(nt, "ShaderNodeMath", (300, -480), operation="MULTIPLY")
        nt.links.new(topf.outputs["Result"], tb.inputs[0])
        nt.links.new(botf.outputs["Result"], tb.inputs[1])
        d2 = node(nt, "ShaderNodeMath", (420, -100), operation="MULTIPLY")
        nt.links.new(d1.outputs["Value"], d2.inputs[0])
        nt.links.new(tb.outputs["Value"], d2.inputs[1])
        self.level = node(nt, "ShaderNodeValue", (250, -550))
        d3 = node(nt, "ShaderNodeMath", (520, -100), operation="MULTIPLY")
        nt.links.new(d2.outputs["Value"], d3.inputs[0])
        nt.links.new(self.level.outputs[0], d3.inputs[1])
        nt.links.new(d3.outputs["Value"], vol.inputs["Density"])
        low = node(nt, "ShaderNodeMath", (300, -650), operation="POWER")
        inv = node(nt, "ShaderNodeMath", (150, -650), operation="SUBTRACT")
        inv.inputs[0].default_value = 1.0
        nt.links.new(sep.outputs["Z"], inv.inputs[1])
        nt.links.new(inv.outputs["Value"], low.inputs[0])
        low.inputs[1].default_value = 7.0
        es = node(nt, "ShaderNodeMath", (480, -650), operation="MULTIPLY")
        es.inputs[1].default_value = 3.0
        nt.links.new(low.outputs["Value"], es.inputs[0])
        ew = node(nt, "ShaderNodeMath", (620, -650), operation="MULTIPLY")
        nt.links.new(es.outputs["Value"], ew.inputs[0])
        nt.links.new(d3.outputs["Value"], ew.inputs[1])  # glow only where ash is
        ew2 = node(nt, "ShaderNodeMath", (740, -650), operation="MULTIPLY")
        ew2.inputs[1].default_value = 25.0
        nt.links.new(ew.outputs["Value"], ew2.inputs[0])
        nt.links.new(ew2.outputs["Value"], vol.inputs["Emission Strength"])
        nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
        self.ob.data.materials.append(m)
        for f in range(1, FRAMES + 2, 6):
            self.level.outputs[0].default_value = 0.14 * TL.eruption(frame_t(f))
            self.level.outputs[0].keyframe_insert("default_value", frame=f)

    def update(self, st, frame):
        _, hi, _, kc, c = st
        top = float(hi[-1, kc])
        self.ob.location = (c - 6, 9, top + 24 - 4)
        self.ob.hide_render = TL.eruption(frame_t(frame)) <= 0.001


def setup_sun(sky, fog_in):
    s = bpy.context.scene
    ld = bpy.data.lights.new("Sun", "SUN")
    ld.angle = math.radians(1.2)
    sun = bpy.data.objects.new("Sun", ld)
    s.collection.objects.link(sun)
    # second, elevation deg, azimuth deg, lamp strength, colour, fog density
    keys = [
        (0, -10, -75, 0.0, (1.0, 0.5, 0.28), 0.03),
        (5, -5, -74, 0.0, (1.0, 0.5, 0.28), 0.04),
        (9, 2, -72, 3.0, (1.0, 0.5, 0.28), 0.06),
        (14, 8, -66, 4.5, (1.0, 0.7, 0.5), 0.035),
        (22, 22, -50, 5.5, (1.0, 0.9, 0.78), 0.015),
        (28, 30, -20, 5.5, (1.0, 0.93, 0.86), 0.02),
        (31, 6, 10, 4.0, (1.0, 0.6, 0.38), 0.04),
        (33, -6, 15, 0.0, (1.0, 0.45, 0.25), 0.03),
        (39.5, -6, -70, 0.0, (1.0, 0.45, 0.25), 0.03),
        (43, 1, -70, 2.0, (1.0, 0.45, 0.25), 0.05),
        (48, 8, -64, 4.5, (1.0, 0.7, 0.5), 0.03),
        (60, 30, -20, 5.5, (1.0, 0.94, 0.88), 0.02),
        (69, 40, 15, 5.0, (1.0, 0.93, 0.86), 0.03),
        (80, 14, 50, 4.5, (1.0, 0.72, 0.5), 0.02),
        (95, 4, 62, 3.5, (1.0, 0.5, 0.28), 0.008),
        (TL.SECONDS, 1.5, 68, 2.8, (1.0, 0.42, 0.22), 0.006),
    ]
    for t, el, az, st, rgb, fog in keys:
        f = int(round(t * FPS)) + 1
        e, a = math.radians(el), math.radians(az)
        sun.rotation_euler = (math.pi / 2 - e, 0, a)
        sun.keyframe_insert("rotation_euler", frame=f)
        ld.energy = st
        ld.color = rgb
        ld.keyframe_insert("energy", frame=f)
        ld.keyframe_insert("color", frame=f)
        sky.sun_elevation = e
        sky.sun_rotation = a + math.pi
        sky.keyframe_insert("sun_elevation", frame=f)
        sky.keyframe_insert("sun_rotation", frame=f)
        fog_in.default_value = fog
        fog_in.keyframe_insert("default_value", frame=f)
    md = bpy.data.lights.new("Moon", "SUN")
    md.color = (0.55, 0.66, 1.0)
    md.angle = math.radians(0.6)
    moon = bpy.data.objects.new("Moon", md)
    moon.rotation_euler = (math.radians(55), 0, math.radians(-25))  # onto the face
    s.collection.objects.link(moon)
    for f in range(1, FRAMES + 2, 6):
        md.energy = 1.2 * TL.night(frame_t(f))
        md.keyframe_insert("energy", frame=f)


class FrontLight:
    """Orange light thrown by the front: a glow in the cold open, a
    flickering furnace in 2026; and the flare of the first commit."""

    def __init__(self):
        ld = bpy.data.lights.new("Front", "POINT")
        ld.color = (1.0, 0.42, 0.12)
        ld.shadow_soft_size = 1.0
        self.ld = ld
        self.ob = bpy.data.objects.new("Front", ld)
        bpy.context.scene.collection.objects.link(self.ob)
        fd = bpy.data.lights.new("Flare", "POINT")
        fd.color = (1.0, 0.75, 0.42)
        fd.shadow_soft_size = 0.02
        fo = bpy.data.objects.new("Flare", fd)
        fo.location = (X0 + 0.05, -0.22, 0.05)
        bpy.context.scene.collection.objects.link(fo)
        for f, v in ((1, 0), (35, 0), (38, 40), (60, 9), (130, 2.5), (260, 0.6)):
            fd.energy = v
            fd.keyframe_insert("energy", frame=f)

    def update(self, st, frame):
        _, hi, _, kc, c = st
        top = float(hi[-1, kc])
        t = frame_t(frame)
        e = TL.eruption(t)
        flick = 1 + 0.18 * math.sin(t * 11.3) * math.sin(t * 4.1 + 1) + \
            0.1 * math.sin(t * 23.7)
        self.ob.location = (c + 1.2 + 0.1 * top, -2.0 - 0.15 * top, 0.15 + 0.6 * top)
        self.ld.energy = (8 + 6 * top * top * (0.2 + e)) * flick


# ---------------------------------------------------------------- camera

class Camera:
    def __init__(self):
        s = bpy.context.scene
        cd = bpy.data.cameras.new("Cam")
        cd.clip_start = 0.02
        cd.clip_end = 2000
        cd.dof.use_dof = True
        self.cam = bpy.data.objects.new("Cam", cd)
        s.collection.objects.link(self.cam)
        s.camera = self.cam
        self.tgt = bpy.data.objects.new("Target", None)
        s.collection.objects.link(self.tgt)
        tr = self.cam.constraints.new("TRACK_TO")
        tr.target = self.tgt
        tr.track_axis = "TRACK_NEGATIVE_Z"
        tr.up_axis = "UP_Y"
        cd.dof.focus_object = self.tgt
        for f in range(1, FRAMES + 1, 2):
            self.set(f, key=True)

    def set(self, f, key=False):
        cl, tl, lens, fs = TL.camera(f)
        cd = self.cam.data
        self.cam.location = cl
        self.tgt.location = tl
        cd.lens = lens
        cd.dof.aperture_fstop = fs
        if key:
            self.cam.keyframe_insert("location", frame=f)
            self.tgt.keyframe_insert("location", frame=f)
            cd.keyframe_insert("lens", frame=f)
            cd.dof.keyframe_insert("aperture_fstop", frame=f)


# ---------------------------------------------------------------- render

def setup_render(pct):
    s = bpy.context.scene
    s.render.engine = "BLENDER_EEVEE"
    s.render.resolution_x, s.render.resolution_y = 3840, 1608
    s.render.resolution_percentage = pct
    s.render.fps = FPS
    s.frame_start, s.frame_end = 1, FRAMES
    ee = s.eevee
    for prop, val in (("taa_render_samples", 64), ("use_raytracing", True),
                      ("use_shadows", True), ("volumetric_end", 400.0),
                      ("volumetric_tile_size", "4"), ("volumetric_samples", 64),
                      ("use_volumetric_shadows", True)):
        if hasattr(ee, prop):
            try:
                setattr(ee, prop, val)
            except Exception as e:  # enum names shift between versions
                print("skip", prop, e)
    vs = s.view_settings
    vs.view_transform = "AgX"
    for look in ("AgX - Punchy", "Punchy"):
        try:
            vs.look = look
            break
        except TypeError:
            pass
    vs.exposure = -0.6


# ---------------------------------------------------------------- CVE

CVE_DAY = "2018-01-02"     # CVE-2017-18021, fixed (#342) and released in v1.2.1
CVE_LEN = 6 * FPS          # on screen this long, then gone without a trace


class Crack:
    """CVE-2017-18021: a red fault runs up the wall where the fix landed,
    then v1.2.1 seals it white-hot and it cools away. Hidden outside its
    window, so only those frames change."""

    def __init__(self, wall):
        self.wall = wall
        self.f0 = TL.frame_of_x(TL.X(CVE_DAY))
        self.k = int(np.argmin(np.abs(XK - TL.X(CVE_DAY))))
        rnd = random.Random(18021)
        # a main fault and three branches: (u along the height, dx, width)
        n, dx, pts = 36, 0.0, []
        for i in range(n):
            dx += rnd.uniform(-0.045, 0.045)
            pts.append((i / (n - 1), dx, 1.0))
        lines = [pts]
        for _ in range(3):
            j = rnd.randrange(6, n - 8)
            u0, x0 = pts[j][0], pts[j][1]
            side = rnd.choice((-1, 1))
            lines.append([(u0 + i * rnd.uniform(0.015, 0.03),
                           x0 + side * i * rnd.uniform(0.03, 0.06), 0.55 * (1 - i / 7))
                          for i in range(7)])
        self.lines = lines
        verts, faces, base = [], [], 0
        for ln in lines:
            for i in range(len(ln)):
                verts += [(0, 0, 0), (0, 0, 0)]
                if i:
                    a = base + 2 * (i - 1)
                    faces.append((a, a + 1, a + 3, a + 2))
            base += 2 * len(ln)
        m = bpy.data.materials.new("Crack")
        nt = m.node_tree
        nt.nodes.clear()
        out = node(nt, "ShaderNodeOutputMaterial", (300, 0))
        self.em = node(nt, "ShaderNodeEmission", (0, 0))
        nt.links.new(self.em.outputs["Emission"], out.inputs["Surface"])
        self.ob = new_mesh_object("Crack", verts, faces, mat=m)
        self.ob.visible_shadow = False
        self.co = np.zeros((len(verts), 3), dtype=np.float32)
        self.ld = bpy.data.lights.new("CrackLight", "POINT")
        self.ld.shadow_soft_size = 0.4
        self.lo = bpy.data.objects.new("CrackLight", self.ld)
        bpy.context.scene.collection.objects.link(self.lo)
        x = float(XK[self.k])
        h = float(wall_state(self.f0 + 12)[1][-1, self.k])
        red = simple_material("CVE", (0.9, 0.2, 0.15), 0.4, (1.0, 0.1, 0.06), 1.2)
        warm = simple_material("CVEsub", (0.9, 0.8, 0.6), 0.4, (1.0, 0.8, 0.55), 1.0)
        self.texts = [
            add_text("CVE-2017-18021", (x - 0.3, -1.05, 0.70 * h), 0.16, red,
                     align="RIGHT"),
            add_text("predictable passwords · fixed in v1.2.1",
                     (x - 0.31, -1.05, 0.70 * h - 0.17), 0.075, warm, align="RIGHT"),
        ]
        self.update(None, 1)  # start hidden

    def look(self, a):
        """(growth 0..1, rgb, strength, width scale) a frames in."""
        grow = min(max(a / 10, 0), 1)
        flick = 1 + 0.25 * math.sin(a * 1.7) * math.sin(a * 0.63 + 1)
        red, white = np.array((1.0, 0.07, 0.03)), np.array((1.0, 0.86, 0.62))
        if a < 36:                                   # the fault, glowing red
            return grow, red, 4.5 * flick, 1.0
        if a < 46:                                   # v1.2.1 seals it
            u = (a - 36) / 10
            return 1, red + (white - red) * u, 4.5 + 18 * u, 1.0 + 0.3 * u
        u = min((a - 46) / (CVE_LEN - 46), 1)        # and it cools away
        c = white + (np.array((1.0, 0.45, 0.15)) - white) * min(u * 2, 1)
        return 1, c, 22.5 * (1 - u) ** 2.5, 1.3 - u

    def update(self, st, frame):
        a = frame - self.f0
        on = 0 <= a < CVE_LEN
        for ob in [self.ob, self.lo] + self.texts:
            ob.hide_render = ob.hide_viewport = not on
        if not on:
            self.ld.energy = 0
            return
        s = 0.001 if a < 2 else min((a - 2) / 8, 1)
        if a > CVE_LEN - 24:
            s = max((CVE_LEN - a) / 24, 0.001)
        for t in self.texts:
            t.scale = (s, s, s)
        k = self.k
        top = float(st[1][-1, k])
        zs, ys = self.wall.front_profile(k)
        grow, rgb, strength, wscale = self.look(a)
        x0 = float(XK[k])
        i = 0
        for ln in self.lines:
            for u, dx, w in ln:
                z = 0.02 + min(u, grow) * top * 0.96
                y = float(np.interp(z, zs, ys)) - 0.06
                half = 0.0 if u > grow else \
                    0.5 * wscale * w * (0.02 + 0.03 * math.sin(math.pi * min(u, 1)))
                self.co[i] = (x0 + dx - half, y, z)
                self.co[i + 1] = (x0 + dx + half, y, z)
                i += 2
        set_co(self.ob, self.co)
        zl = 0.72 * top  # the name, on the face just left of the fault
        yl = float(np.interp(zl, zs, ys)) - 0.18
        self.texts[0].location = (x0 - 0.15, yl, zl)
        self.texts[1].location = (x0 - 0.16, yl, zl - 0.17)
        self.em.inputs["Color"].default_value = (*rgb, 1.0)
        self.em.inputs["Strength"].default_value = strength
        self.ld.color = tuple(rgb)
        self.ld.energy = strength * 25 * grow
        self.lo.location = (x0, -1.6 - 0.1 * top, 0.5 * top)


class Scene:
    def __init__(self):
        global DG
        DG = build_disp_grid()
        rock = rock_material()
        self.wall = Wall(rock)
        self.plateau = Plateau(rock)
        self.front = Cap(molten_material(), +1, 1.6, self.wall, self.plateau)
        self.back = Cap(rock, -1, 0.0, self.wall, self.plateau)
        build_ground(sand_material())
        build_mesas(simple_material("FarRock", (0.36, 0.22, 0.15)))
        self.river = River(river_material())
        build_releases(self.river)
        self.sparks = Sparks(emission_attr_material("Spark", 30.0))
        hw = emission_attr_material("HighWater", 2.0)
        level = hw.node_tree.nodes["Math"].inputs[1]
        for t, v in ((0, 2.0), (75, 2.0), (88, 0.0)):
            level.default_value = v
            level.keyframe_insert("default_value", frame=int(t * FPS) + 1)
        self.high = HighWater(hw)  # fades after the present: an open ending
        sky = setup_world()
        setup_sun(sky, build_mist())
        self.plume = Plume()
        self.light = FrontLight()
        self.camera = Camera()
        self.crack = Crack(self.wall)

    def update(self, f):
        st = wall_state(f)
        self.wall.update(st)
        self.plateau.update(st)
        self.front.update(st)
        self.back.update(st)
        self.river.update(f)
        self.high.update(st)
        self.sparks.update(f)
        self.plume.update(st, f)
        self.light.update(st, f)
        self.crack.update(st, f)
        self.camera.set(f)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {"pct": 25, "stills": "", "save": "", "anim": False, "samples": 64,
            "start": 1, "end": FRAMES, "out": "frames"}
    i = 0
    while i < len(argv):
        k = argv[i].lstrip("-")
        if k == "anim":
            opts["anim"] = True
            i += 1
        else:
            opts[k] = argv[i + 1]
            i += 2

    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_render(int(opts["pct"]))
    bpy.context.scene.eevee.taa_render_samples = int(opts["samples"])
    sc = Scene()
    s = bpy.context.scene

    def render(f, path):
        s.frame_set(f)
        sc.update(f)
        s.render.filepath = path
        bpy.ops.render.render(write_still=True)

    if opts["save"]:
        s.frame_set(1)
        sc.update(1)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, opts["save"]))
    for f in [int(v) for v in opts["stills"].split(",") if v]:
        render(f, os.path.join(HERE, "renders", f"still_{f:04d}.png"))
    if opts["anim"]:
        s.render.image_settings.file_format = "PNG"
        for f in range(int(opts["start"]), int(opts["end"]) + 1):
            render(f, os.path.join(HERE, "renders", opts["out"], f"f_{f:04d}.png"))


main()
