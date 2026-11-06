# QtPass strata

Twelve years of QtPass history as a canyon wall. Left to right is time:
3 m per calendar year plus 40 m spread over all commits, so busy months get
room. Each layer of rock is the code written in one month (git-of-theseus
cohorts, translations excluded).

A glowing front, "now", travels along the wall and lays the rock down.
Behind it every layer is as thick as that month's code that still survives
*now*, so the old walls wear down as 2026 rewrites them; a faint high-water
line shows where they once stood. Commits are sparks at the front, coloured
by author; releases are beacons; the languages are a river of light at the
foot of the wall, with a tributary wherever new ones joined. 2014–2015 by
day, the quiet years by night, and 2026 as an eruption. The film ends on a
side view where the canyon is the survival chart.

## Files

- `prep.py`: cohorts, commits, locales and stats into `scene_data.json`
- `timeline.py`: the one clock (cursor, camera path, title times) shared by
  the scene, the cards and the sound
- `build_scene.py`: builds the scene in Blender and renders it, rebuilding
  the wall, plateau, end faces, sparks and river for every frame (numpy)
- `cards.py`: Lato title cards with the QtPass logo, as transparent PNGs
- `audio.py`: the soundtrack (wind, drone, commit pings, release booms,
  eruption), generated with numpy and scipy
- `finish.sh`: bloom, vignette, grain, fades, titles and sound into an MP4

## Pipeline

```sh
python3 -m venv .venv && .venv/bin/pip install git-of-theseus
.venv/bin/git-of-theseus-analyze --branch main --interval 604800 \
  --cohortfm '%Y-%m' --ignore 'localization/**' --outdir data/monthly ~/Projects/QtPass
git -C ~/Projects/QtPass log --format=%ct refs/heads/main -- > data/commit_times.txt
git -C ~/Projects/QtPass log --use-mailmap --format='%ct%x09%aN' refs/heads/main -- \
  > data/commits.tsv
# data/locales.json: [["YYYY-MM-01", number of .ts files], ...] per month
python3 prep.py                                   # -> scene_data.json
blender -b --factory-startup -P build_scene.py -- --stills 40,850,1500,2450 --pct 25
blender -b --factory-startup -P build_scene.py -- --pct 100 --samples 64 --anim
python3 audio.py audio.wav
./finish.sh renders/frames qtpass-strata.mp4
```

2520 frames at 24 fps (105 s), 3840x1608 (2.39:1) at 100 %. `--anim` takes
`--start N --end M` to render part of the film. `--save scene.blend` writes
the scene at frame 1; the per-frame geometry is driven by the script, not by
the .blend.
