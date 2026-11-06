"""One clock for the film: where "now" is on the wall at every frame.

Used by build_scene.py (inside Blender), cards.py and audio.py, so the
camera, the titles and the sound agree. Plain Python, no bpy.
"""
import bisect
import json
import math
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 24
SECONDS = 105
FRAMES = SECONDS * FPS

DATA = json.load(open(os.path.join(HERE, "scene_data.json")))
XMAP = DATA["xmap"]  # [unix time, x] on the first of every month
X0, X1 = DATA["x0"], DATA["x1"]
_XT = [t for t, _ in XMAP]
_XX = [x for _, x in XMAP]


def x_at_ts(ts):
    """x on the wall of a unix time."""
    i = min(max(bisect.bisect_right(_XT, ts) - 1, 0), len(_XT) - 2)
    f = (ts - _XT[i]) / (_XT[i + 1] - _XT[i])
    return _XX[i] + (_XX[i + 1] - _XX[i]) * f


def ts_at_x(x):
    """Unix time at x on the wall (x grows with time)."""
    i = min(max(bisect.bisect_right(_XX, x) - 1, 0), len(_XX) - 2)
    f = (x - _XX[i]) / (_XX[i + 1] - _XX[i])
    return _XT[i] + (_XT[i + 1] - _XT[i]) * f


def X(day):
    return x_at_ts(datetime.fromisoformat(day).timestamp())


# The "now" cursor: (second, x). Linear between, then smoothed.
CHAPTERS = [
    (0, X0), (5, X0),                  # cold open: the first commit
    (11, X("2015-01-01")),             # 2014
    (19, X("2016-01-01")),             # 2015
    (33, X("2021-01-01")),             # 2016-2020
    (41, X("2026-01-01")),             # the quiet years, by night
    (69, X1),                          # 2026, the eruption
    (SECONDS, X1),                     # the present, then the reveal
]


def _raw(t):
    for (t0, x0), (t1, x1) in zip(CHAPTERS, CHAPTERS[1:]):
        if t <= t1:
            return x0 + (x1 - x0) * min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return CHAPTERS[-1][1]


def _smooth():
    half = int(0.75 * FPS)
    raw = [_raw((f - 1) / FPS) for f in range(1 - half, FRAMES + 2 + half)]
    out = [X0]  # index 0 unused; frames are 1-based
    for f in range(1, FRAMES + 2):
        w = raw[f - 1:f + 2 * half]
        out.append(sum(w) / len(w))
    for f in range(2, len(out)):  # keep it monotone
        out[f] = max(out[f], out[f - 1])
    return out


CURSOR = _smooth()


def t_of(frame):
    return (frame - 1) / FPS


def cursor_x(frame):
    return CURSOR[min(max(int(frame), 1), FRAMES + 1)]


def now_ts(frame):
    return ts_at_x(cursor_x(frame))


def frame_of_x(x):
    """First frame at which the cursor has reached x (the present at most)."""
    return max(1, bisect.bisect_left(CURSOR, min(x, CURSOR[-1]), 1))


def eruption(t):
    """0..1: how hard 2026 is erupting at second t."""
    keys = [(0, 0), (41, 0), (47, 0.35), (60, 1.0), (69, 1.0), (80, 0.5),
            (SECONDS, 0.4)]
    return _interp(keys, t)


def night(t):
    """0..1: stars and moonlight."""
    keys = [(0, 1), (5, 1), (9, 0), (30, 0), (33, 1), (39.5, 1), (43, 0),
            (SECONDS, 0)]
    return _interp(keys, t)


def _interp(keys, t):
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            u = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
            u = u * u * (3 - 2 * u)
            return v0 + (v1 - v0) * u
    return keys[-1][1]


# ------------------------------------------------------------ camera
# (second, camera xyz, target xyz, lens, f-stop). An x given as a string
# "c+1.5" is relative to the cursor at that second, so the camera keeps
# pace with the front however fast time runs.
CAMERA = [
    (0.0, ("c-0.75", -0.85, 0.09), ("c+0.1", -0.15, 0.05), 40, 2.0),
    (4.5, ("c-0.95", -1.0, 0.12), ("c+0.05", -0.1, 0.07), 38, 2.0),
    (8.0, ("c-1.6", -1.3, 0.25), ("c-0.25", 0, 0.22), 30, 2.4),
    (11.0, ("c-2.2", -1.5, 0.35), ("c-0.4", 0, 0.25), 28, 2.8),
    (15.0, ("c-4.5", -2.6, 0.9), ("c-0.5", 0, 0.8), 26, 4),
    (19.0, ("c-6", -3.6, 1.5), ("c-0.8", 0, 1.3), 26, 5.6),
    (26.0, ("c-7", -5.0, 2.4), ("c-1", 0, 1.8), 28, 5.6),
    (33.0, ("c-8", -7.0, 3.4), ("c-1", 0, 1.8), 30, 8),
    (37.0, ("c-6", -6.0, 2.4), ("c+0", 0, 1.5), 24, 8),
    (41.0, ("c-5", -6.5, 1.6), ("c+1", 0, 3.0), 20, 8),
    (50.0, ("c-9", -11.0, 3.0), ("c-1", 1, 7.0), 20, 8),
    (60.0, ("c-12", -16.0, 5.0), ("c-1", 2, 12.0), 20, 8),
    (69.0, (X1 + 3, -14.0, 7.0), (X1 - 1, 3, 14.0), 20, 8),
    (73.0, (X1 + 14, -9.0, 9.0), (X1, 5, 15.0), 22, 8),
    (78.0, (X1 + 17, 1.0, 14.0), (X1, 8, 16.0), 22, 8),
    # out past the present face, then low, looking back along the canyon
    (89.0, (X1 + 18, -14.0, 9.0), (60.0, 2, 4.0), 28, 11),
    (SECONDS, (X1 + 13, -20.0, 4.5), (35.0, 0, 2.0), 32, 11),
]


def _cursor_t(t):
    return cursor_x(int(round(t * FPS)) + 1)


def _hermite(ts, vs, t):
    n = len(ts)
    i = min(max(bisect.bisect_right(ts, t) - 1, 0), n - 2)
    t0, t1 = ts[i], ts[i + 1]
    u = min(max((t - t0) / (t1 - t0), 0.0), 1.0)

    def slope(j):
        if j <= 0 or j >= n - 1:
            return 0.0
        return (vs[j + 1] - vs[j - 1]) / (ts[j + 1] - ts[j - 1])

    m0, m1 = slope(i) * (t1 - t0), slope(i + 1) * (t1 - t0)
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * vs[i] + h10 * m0 + h01 * vs[i + 1] + h11 * m1


def _tracks():
    ts = [k[0] for k in CAMERA]
    tracks = {}
    for name, idx in (("cam", 1), ("tgt", 2)):
        for axis in range(3):
            vals = []
            for k in CAMERA:
                v = k[idx][axis]
                if axis == 0:  # store x relative to the cursor
                    if isinstance(v, str):
                        v = float(v[1:])
                    else:
                        v = v - _cursor_t(k[0])
                vals.append(v)
            tracks[(name, axis)] = vals
    tracks["lens"] = [k[3] for k in CAMERA]
    tracks["fstop"] = [k[4] for k in CAMERA]
    return ts, tracks


_CT, _TRACKS = _tracks()


def camera(frame):
    """(camera xyz, target xyz, lens, f-stop) at a frame."""
    t = t_of(frame)
    c = cursor_x(frame)
    out = []
    for name in ("cam", "tgt"):
        p = [_hermite(_CT, _TRACKS[(name, a)], t) for a in range(3)]
        p[0] += c
        out.append(tuple(p))
    lens = _hermite(_CT, _TRACKS["lens"], t)
    fstop = _hermite(_CT, _TRACKS["fstop"], t)
    return out[0], out[1], lens, fstop


# ------------------------------------------------------------ titles
STATS = DATA.get("stats", {})


def cards():
    s = STATS
    return [
        (1.5, 7.0, "centre", "QtPass", "twelve years in stone"),
        (7.5, 11.2, "lower", "2014", "first commit, 30 July"),
        (11.8, 17.5, "lower", "2015",
         f"v1.0.0 · {s.get('commits_2015', 984):,} commits that year"),
        (20.0, 26.5, "lower", "2016 – 2020",
         "steady layers · 22 to 36 languages"),
        (33.5, 39.5, "lower", "2021 – 2025", "the quiet years"),
        (42.0, 48.0, "lower", "2026",
         f"{s.get('commits_2026', 2305):,} commits · "
         f"{s.get('share_2026', 44)}% of all history"),
        (53.0, 60.0, "lower", "the past, rewritten",
         f"{s.get('replaced_pct', 61)}% of the older code replaced this year"),
        (70.0, 77.0, "lower", "the present",
         f"{s.get('lines_now', 72701):,} lines of code, still cooling"),
        (91.0, 103.5, "centre", "QtPass",
         f"{s.get('commits', 5246):,} commits · "
         f"{s.get('authors', 179)} contributors · "
         f"{s.get('locales', 64)} languages"),
    ]
