#!/usr/bin/env python3
"""Turn git-of-theseus monthly cohorts into wall geometry data for Blender.

Writes scene_data.json: uniform columns along the time axis, each with the
cumulative top of every cohort band, plus per-band colours and release tags.
"""
import colorsys
import json
import random
from datetime import datetime

X_PER_YEAR = 3.0  # metres of canyon wall per calendar year
X_COMMITS = 40.0  # plus metres spread over all commits, by commit count
Z_PER_LINE = 1 / 2500  # metres of rock per surviving line
STEP = 0.05  # column spacing in metres
START = datetime(2014, 7, 30)

YEAR_COLOURS = {
    2014: "#5a1e1b", 2015: "#7a2f22", 2016: "#9a4a2a", 2017: "#b06a35",
    2018: "#c4873f", 2019: "#d2a35a", 2020: "#ddb878", 2021: "#e3c896",
    2022: "#e8d3a8", 2023: "#d9c7a5", 2024: "#b8ab95", 2025: "#8f8779",
    2026: "#2b2a2e",
}

TAGS = [
    ("v0.8.1", "2015-05-07"), ("v1.0.0", "2015-08-01"),
    ("v1.1.0", "2016-01-25"), ("v1.2.0", "2017-11-09"),
    ("v1.3.0", "2019-08-20"), ("v1.4.0", "2023-09-17"),
    ("v1.5.0", "2026-03-21"), ("v1.8.0", "2026-09-13"),
    ("v2.0.0-rc1", "2026-09-24"), ("v2.0.0-rc2", "2026-10-05"),
    ("v2.0.0-rc3", "2026-11-03"),  # tagged 2026-11-05: the present face
]


import bisect

COMMITS = sorted(int(l) for l in open("data/commit_times.txt") if l.strip())


def years(d):
    return (d - START).total_seconds() / (365.2425 * 86400)


def x_of(d):
    """Half calendar time, half commit count: busy months get room."""
    frac = bisect.bisect_right(COMMITS, d.timestamp()) / len(COMMITS)
    return years(d) * X_PER_YEAR + frac * X_COMMITS


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def main():
    d = json.load(open("data/monthly/cohorts.json"))
    ts = [datetime.fromisoformat(t) for t in d["ts"]]
    xs = [x_of(t) for t in ts]
    labels = [l.rsplit(" ", 1)[1] for l in d["labels"]]  # "2014-07"
    ys = d["y"]  # [cohort][sample]
    n_c = len(labels)

    # Cumulative tops per sample, oldest cohort at the bottom.
    cum = []
    for i in range(len(ts)):
        acc, tops = 0.0, []
        for c in range(n_c):
            acc += ys[c][i] * Z_PER_LINE
            tops.append(acc)
        cum.append(tops)

    x0, x1 = xs[0], xs[-1]
    cols = []
    j = 0
    x = x0
    while x <= x1 + 1e-9:
        while j + 1 < len(xs) - 1 and xs[j + 1] < x:
            j += 1
        a, b = xs[j], xs[min(j + 1, len(xs) - 1)]
        f = 0.0 if b == a else min(max((x - a) / (b - a), 0.0), 1.0)
        tops = [cum[j][c] * (1 - f) + cum[min(j + 1, len(xs) - 1)][c] * f
                for c in range(n_c)]
        cols.append({"x": round(x, 4), "tops": [round(t, 5) for t in tops]})
        x += STEP

    rnd = random.Random(2014)
    bands = []
    for c, lab in enumerate(labels):
        y, m = map(int, lab.split("-"))
        r, g, b = hex_rgb(YEAR_COLOURS[y])
        if y == 2026:  # volcanic months: charcoal, scoria, ash, basalt
            r, g, b = hex_rgb(["#2a2a2e", "#4b3530", "#5d5853", "#232529",
                               "#6a3a2a", "#3a3836", "#4e4a47", "#1f2024",
                               "#5a2f25", "#454240", "#2e2b2c"][m - 1])
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = min(max(l * (1 + (0.09 if m % 2 else -0.07) + rnd.uniform(-0.05, 0.05)), 0), 1)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        glow = 0.0
        if y == 2026:
            glow = 0.15 + 0.85 * max(m - 2, 0) / 9
        thick = max(cols[k]["tops"][c] - (cols[k]["tops"][c - 1] if c else 0)
                    for k in range(len(cols)))
        bands.append({"label": lab, "rgb": [r, g, b], "glow": glow,
                      "max_thick": thick})

    tags = []
    for name, day in TAGS:
        x = x_of(datetime.fromisoformat(day))
        k = min(range(len(cols)), key=lambda i: abs(cols[i]["x"] - x))
        x = min(x, cols[-1]["x"])
        tags.append({"name": name, "x": x, "top": cols[k]["tops"][-1]})

    years_x = {y: x_of(datetime(y, 1, 1)) for y in range(2015, 2027)}
    xmap = []
    for y in range(2014, 2027):
        for m in range(1, 13):
            d = datetime(y, m, 1)
            xmap.append([d.timestamp(), x_of(d)])
    # For the animated wall: every cohort's size (m) at every sample, and
    # where on the wall each cohort's month begins and ends.
    ym = [[round(v * Z_PER_LINE, 5) for v in ys[c]] for c in range(n_c)]
    born_x, end_x = [], []
    for lab in labels:
        y, m = map(int, lab.split("-"))
        born_x.append(x_of(datetime(y, m, 1)))
        end_x.append(x_of(datetime(y + m // 12, m % 12 + 1, 1)))

    authors, commits = {}, []
    for line in open("data/commits.tsv", encoding="utf-8"):
        if not line.strip():
            continue
        t, name = line.rstrip("\n").split("\t", 1)
        authors[name] = authors.get(name, 0) + 1
        commits.append((int(t), name))
    order = sorted(authors, key=lambda a: -authors[a])
    index = {a: i for i, a in enumerate(order)}
    commits.sort()
    commit_xy = [[round(x_of(datetime.fromtimestamp(t)), 4), index[a]]
                 for t, a in commits]

    locales = [[round(x_of(datetime.fromisoformat(d)), 4), n]
               for d, n in json.load(open("data/locales.json"))]

    def year_count(y):
        return sum(1 for t, _ in commits if datetime.fromtimestamp(t).year == y)

    i26 = min(range(len(ts)), key=lambda i: abs(ts[i] - datetime(2026, 1, 1)))
    before = sum(ys[c][i26] for c in range(n_c))
    survive = sum(ys[c][-1] for c, l in enumerate(labels) if l < "2026")
    stats = {
        "commits": len(commits), "authors": len(authors),
        "commits_2015": year_count(2015), "commits_2026": year_count(2026),
        "share_2026": round(100 * year_count(2026) / len(commits)),
        "replaced_pct": round(100 * (1 - survive / before)),
        "lines_now": int(round(sum(ys[c][-1] for c in range(n_c)))),
        "locales": locales[-1][1],
    }
    print("stats", stats)

    json.dump({"cols": cols, "bands": bands, "tags": tags,
               "years_x": years_x, "xmap": xmap, "x0": x0, "x1": x1,
               "x_per_year": X_PER_YEAR,
               "ts": [t.timestamp() for t in ts], "sample_x": xs, "ym": ym,
               "born_x": born_x, "end_x": end_x,
               "authors": order, "commits": commit_xy, "locales": locales,
               "stats": stats},
              open("scene_data.json", "w"))
    print(len(cols), "columns,", n_c, "bands, x", round(x0, 2), "->",
          round(x1, 2), "max top", round(max(c["tops"][-1] for c in cols), 2))


if __name__ == "__main__":
    main()
