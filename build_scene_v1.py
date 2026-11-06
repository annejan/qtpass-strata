"""QtPass strata: build the canyon scene in Blender from scene_data.json.

blender -b -P build_scene.py -- [--stills 1,480,...] [--pct 25]
                                [--save scene.blend] [--anim [--start N]]
"""
import json
import math
import os
import random
import sys
from datetime import datetime

import bpy
from mathutils import Vector, noise

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "scene_data.json")))
COLS, BANDS, TAGS = DATA["cols"], DATA["bands"], DATA["tags"]
XPY = DATA["x_per_year"]
X0, X1 = DATA["x0"], DATA["x1"]
DEPTH = 18.0
FPS, FRAMES = 24, 2160
START = datetime(2014, 7, 30)
LOGO = os.path.join(HERE, "assets", "qtpass-logo.png")
LATO = os.path.expanduser("~/.local/share/fonts/Lato/Lato-Light.ttf")
FOSSIL = (2.8, 0.25, 0.34)  # x, z centre and size (m) of the logo in the 2014 rock


XMAP = DATA["xmap"]


def X(day):
    """x of a date on the wall's time axis (half calendar, half commits)."""
    t = datetime.fromisoformat(day).timestamp()
    i = 0
    while i < len(XMAP) - 2 and XMAP[i + 1][0] <= t:
        i += 1
    (t0, x0), (t1, x1) = XMAP[i], XMAP[i + 1]
    return x0 + (t - t0) / (t1 - t0) * (x1 - x0)


def mesa_height(top, x, y, y0):
    """Plateau height at depth y: flat, then a talus slope down at the back."""
    f = (y - y0) / (DEPTH - y0)
    if f < 0.5:
        return top
    u = min((f - 0.5) / 0.5, 1.0)
    fall = u * u * (3 - 2 * u)
    rough = noise.noise(Vector((x * 0.3, y * 0.3, 4.4))) * 0.08
    return max(top * (1 - fall) * (1 + rough), 0.0)


def top_at(x):
    k = min(max(int(round((x - COLS[0]["x"]) / 0.05)), 0), len(COLS) - 1)
    return COLS[k]["tops"][-1]


# ---------------------------------------------------------------- geometry

def disp(x, z):
    """Cliff relief (towards +Y is into the rock); depends on x, z only so
    neighbouring bands meet without cracks."""
    amp = 0.3 + 0.7 * min(z / 3.0, 1.0)
    big = noise.fractal(Vector((x * 0.22, 7.3, z * 0.3)), 1.0, 2.0, 4)
    small = noise.noise(Vector((x * 2.1, 3.1, z * 2.6)))
    grit = noise.noise(Vector((x * 7.0, 5.7, z * 9.0)))
    return big * 0.55 * amp + small * 0.09 + grit * 0.025


def new_mesh_object(name, verts, faces, cols=None, glows=None, mat=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    if cols is not None:
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        flat = [c for rgb in cols for c in (*rgb, 1.0)]
        ca.data.foreach_set("color", flat)
    if glows is not None:
        ga = me.attributes.new("glow", "FLOAT", "POINT")
        ga.data.foreach_set("value", glows)
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if mat:
        me.materials.append(mat)
    return ob


FRONT = {}  # column index -> [(z, y)] of the wall's front at the two ends


def front_y(k, z):
    """y of the wall's front surface at end column k and height z."""
    prof = FRONT.get(k)
    if not prof:
        return disp(COLS[k]["x"], z)
    for (z0, y0), (z1, y1) in zip(prof, prof[1:]):
        if z <= z1:
            t = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
            return y0 + (y1 - y0) * min(max(t, 0.0), 1.0)
    return prof[-1][1]


def build_wall(mat):
    rnd = random.Random(7)
    verts, faces, cols, glows = [], [], [], []
    n = len(COLS)
    for c, band in enumerate(BANDS):
        if band["max_thick"] < 2e-4:
            continue
        rows = max(1, min(8, math.ceil(band["max_thick"] / 0.25)))
        recess = rnd.uniform(-0.3, 0.12)
        base = len(verts)
        for k in range(n):
            lo = COLS[k]["tops"][c - 1] if c else 0.0
            hi = COLS[k]["tops"][c]
            x = COLS[k]["x"]
            for r in range(rows + 1):
                z = lo + (hi - lo) * r / rows
                bulge = (0.18 + recess) * math.sin(math.pi * r / rows) ** 0.35 * min((hi - lo) / 0.25, 1)
                bulge = -bulge
                verts.append((x, disp(x, z) + bulge, z))
                if k in (0, n - 1):
                    FRONT.setdefault(k, []).append((z, disp(x, z) + bulge))
                cols.append(band["rgb"])
                glows.append(band["glow"])
        stride = rows + 1
        for k in range(n - 1):
            t0 = COLS[k]["tops"][c] - (COLS[k]["tops"][c - 1] if c else 0)
            t1 = COLS[k + 1]["tops"][c] - (COLS[k + 1]["tops"][c - 1] if c else 0)
            if t0 < 1e-5 and t1 < 1e-5:
                continue
            for r in range(rows):
                a = base + k * stride + r
                b = base + (k + 1) * stride + r
                faces.append((a, b, b + 1, a + 1))
    for prof in FRONT.values():
        prof.sort()
    return new_mesh_object("Strata", verts, faces, cols, glows, mat)


def top_band(k):
    tops = COLS[k]["tops"]
    for c in range(len(BANDS) - 1, -1, -1):
        if tops[c] - (tops[c - 1] if c else 0) > 1e-4:
            return BANDS[c]
    return BANDS[0]


def build_plateau(mat):
    verts, faces, cols, glows = [], [], [], []
    rows = 28
    n = len(COLS)
    for k in range(n):
        x, z0 = COLS[k]["x"], COLS[k]["tops"][-1]
        band = top_band(k)
        rgb = [min(1.0, v * 1.12 + 0.03) for v in band["rgb"]]
        y0 = disp(x, z0)
        for j in range(rows + 1):
            f = (j / rows) ** 1.7
            y = y0 + (DEPTH - y0) * f
            dune = noise.noise(Vector((x * 0.4, y * 0.4, 1.7))) * 0.12 * min(f * 6, 1)
            verts.append((x, y, mesa_height(z0, x, y, y0) + dune * (1 if f < 0.5 else 0)))
            cols.append(rgb)
            glows.append(band["glow"] * 0.6)
    s = rows + 1
    for k in range(n - 1):
        for j in range(rows):
            a, b = k * s + j, (k + 1) * s + j
            faces.append((a, b, b + 1, a + 1))
    return new_mesh_object("Plateau", verts, faces, cols, glows, mat)


def band_at_height(k, z):
    tops = COLS[k]["tops"]
    for c, t in enumerate(tops):
        if z <= t + 1e-6:
            return BANDS[c]
    return BANDS[-1]


def build_cap(mat, k, sign, glow):
    """End face at column k; sign +1 faces +X (the present), -1 faces -X."""
    x, top = COLS[k]["x"], COLS[k]["tops"][-1]
    ny, nz = 30, max(2, math.ceil(top / 0.25))
    verts, faces, cols, glows = [], [], [], []
    yf = disp(x, top)
    for j in range(ny + 1):
        f = (j / ny) ** 1.7
        h = mesa_height(top, x, yf + (DEPTH - yf) * f, yf)
        for r in range(nz + 1):
            z = h * r / nz
            y0 = front_y(k, z)  # front edge meets the wall's own end profile
            y = y0 + (DEPTH - y0) * f
            if r == nz:  # top edge meets the plateau, dunes included
                z += noise.noise(Vector((x * 0.4, y * 0.4, 1.7))) * 0.12 \
                    * min(f * 6, 1) * (1 if f < 0.5 else 0)
            # rough face, pinned flush to the wall at the front and top edges
            pin = min(f * 8, 1) * min((nz - r) / max(nz * 0.1, 1), 1)
            jag = noise.noise(Vector((y * 0.5, z * 0.5, 9.1))) * 0.35 * sign * pin
            verts.append((x + jag, y, z))
            band = band_at_height(k, z)
            cols.append(band["rgb"])
            glows.append(glow)
    s = nz + 1
    for j in range(ny):
        for r in range(nz):
            a, b = j * s + r, (j + 1) * s + r
            faces.append((a, b, b + 1, a + 1) if sign > 0 else (a, a + 1, b + 1, b))
    return new_mesh_object("Cap+" if sign > 0 else "Cap-", verts, faces, cols, glows, mat)


# ---------------------------------------------------------------- materials

def node(nt, kind, loc, **props):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def fissures(nt, tc, x, y, scale, width):
    """Meandering crack lines (ridged noise); returns a 0..1 socket."""
    n = node(nt, "ShaderNodeTexNoise", (x, y))
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = 6
    n.inputs["Roughness"].default_value = 0.55
    n.inputs["Distortion"].default_value = 0.6
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
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
    # relief = alpha * (0.25 + lum): keyhole (black) lowest, shackle highest
    l2 = node(nt, "ShaderNodeMath", (-450, 700), operation="ADD")
    nt.links.new(lum.outputs["Val"], l2.inputs[0])
    l2.inputs[1].default_value = 0.25
    relief = node(nt, "ShaderNodeMath", (-300, 700), operation="MULTIPLY")
    nt.links.new(l2.outputs["Value"], relief.inputs[0])
    nt.links.new(img.outputs["Alpha"], relief.inputs[1])
    # stain = alpha * (0.75 - lum), so the keyhole reads darkest
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

    # Lava veins: Voronoi cell edges, gated by the glow attribute.
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


def build_labels():
    tag_mat = simple_material("Tag", (0.9, 0.8, 0.55), 0.4, (1.0, 0.78, 0.45), 3.0)
    for t in TAGS:
        if t["name"] == "v1.8.0":
            continue
        size = min(max(t["top"] * 0.055, 0.13), 0.9)
        z = max(t["top"] * 0.45, 0.1)
        add_text(t["name"], (t["x"], -0.55 - size * 0.3, z), size, tag_mat)


# ---------------------------------------------------------------- light

def setup_world():
    s = bpy.context.scene
    w = bpy.data.worlds.new("Sky")
    s.world = w
    nt = w.node_tree
    nt.nodes.clear()
    out = node(nt, "ShaderNodeOutputWorld", (400, 0))
    bg = node(nt, "ShaderNodeBackground", (150, 0))
    sky = node(nt, "ShaderNodeTexSky", (-150, 0), sky_type="MULTIPLE_SCATTERING")
    for prop, val in (("sun_disc", True), ("air_density", 1.4),
                      ("aerosol_density", 2.5), ("dust_density", 2.5)):
        if hasattr(sky, prop):
            setattr(sky, prop, val)
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.25
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
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
    haze.inputs[1].default_value = 0.0018
    nt.links.new(dens.outputs["Value"], haze.inputs[0])
    nt.links.new(haze.outputs["Value"], vol.inputs["Density"])
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    box.data.materials.append(m)
    return dens.inputs[1]


def setup_sun(sky, fog_in):
    s = bpy.context.scene
    ld = bpy.data.lights.new("Sun", "SUN")
    ld.angle = math.radians(1.2)
    sun = bpy.data.objects.new("Sun", ld)
    s.collection.objects.link(sun)
    # frame, elevation deg, azimuth deg, lamp strength, colour, fog density
    keys = [
        (1, 3.0, -72, 4.0, (1.0, 0.5, 0.28), 0.06),
        (480, 9, -66, 5.0, (1.0, 0.72, 0.52), 0.03),
        (960, 18, -60, 5.5, (1.0, 0.88, 0.76), 0.015),
        (1200, 40, -20, 5.0, (1.0, 0.95, 0.9), 0.05),
        (1380, 40, 20, 5.0, (1.0, 0.93, 0.86), 0.08),
        (1600, 14, 58, 4.5, (1.0, 0.7, 0.48), 0.03),
        (2160, 2.5, 68, 3.0, (1.0, 0.42, 0.22), 0.05),
    ]
    for f, el, az, st, rgb, fog in keys:
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


# ---------------------------------------------------------------- camera

def setup_camera():
    s = bpy.context.scene
    cd = bpy.data.cameras.new("Cam")
    cd.lens = 28
    cd.clip_end = 2000
    cam = bpy.data.objects.new("Cam", cd)
    s.collection.objects.link(cam)
    s.camera = cam
    tgt = bpy.data.objects.new("Target", None)
    s.collection.objects.link(tgt)
    tr = cam.constraints.new("TRACK_TO")
    tr.target = tgt
    tr.track_axis = "TRACK_NEGATIVE_Z"
    tr.up_axis = "UP_Y"
    cd.dof.use_dof = True
    cd.dof.focus_object = tgt
    x1 = X1
    # frame, camera (x, y, z), target (x, y, z), lens, f-stop
    keys = [
        (1, (0.9, -1.4, 0.15), (2.8, 0, 0.2), 24, 2.2),
        (240, (4.2, -2.0, 0.45), (8.0, 0, 0.8), 24, 2.8),
        (480, (12.0, -3.2, 1.0), (17.3, 0, 1.2), 26, 4),
        (720, (21.0, -4.5, 1.9), (26.0, 0, 1.8), 28, 5.6),
        (960, (30.0, -5.5, 2.4), (35.6, 0, 1.9), 28, 5.6),
        (1140, (40.0, -8.0, 3.5), (46.0, 0, 2.0), 30, 8),
        (1320, (51.0, -10.0, 3.0), (60.0, 2, 5.0), 26, 8),
        (1500, (58.5, -7.5, 1.6), (68.0, 0, 9.0), 18, 8),
        (1680, (80.0, -12.0, 7.0), (75.5, 4, 14.0), 20, 8),
        (1860, (93.0, 1.0, 14.0), (76.5, 8, 16.0), 22, 8),
        (2010, (95.0, -14.0, 9.0), (60.0, 2, 4.0), 28, 11),
        (2160, (90.0, -20.0, 4.5), (35.0, 0, 2.0), 32, 11),
    ]
    for f, cl, tl, lens, fs in keys:
        cam.location = cl
        tgt.location = tl
        cd.lens = lens
        cd.dof.aperture_fstop = fs
        cam.keyframe_insert("location", frame=f)
        tgt.keyframe_insert("location", frame=f)
        cd.keyframe_insert("lens", frame=f)
        cd.dof.keyframe_insert("aperture_fstop", frame=f)
    return cam


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


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {"pct": 25, "stills": "", "save": "", "anim": False, "samples": 64,
            "start": 1}
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
    rock = rock_material()
    build_wall(rock)
    build_plateau(rock)
    build_cap(molten_material(), len(COLS) - 1, +1, 1.6)
    build_cap(rock, 0, -1, 0.0)
    build_ground(sand_material())
    build_mesas(simple_material("FarRock", (0.36, 0.22, 0.15)))
    build_labels()
    sky = setup_world()
    setup_sun(sky, build_mist())
    setup_camera()
    setup_render(int(opts["pct"]))
    bpy.context.scene.eevee.taa_render_samples = int(opts["samples"])

    if opts["save"]:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, opts["save"]))
    s = bpy.context.scene
    for f in [int(v) for v in opts["stills"].split(",") if v]:
        s.frame_set(f)
        s.render.filepath = os.path.join(HERE, "renders", f"still_{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    if opts["anim"]:
        s.frame_start = int(opts["start"])
        s.render.filepath = os.path.join(HERE, "renders", "frames", "f_")
        s.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(animation=True)


main()
