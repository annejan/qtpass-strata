#!/usr/bin/env python3
"""The soundtrack, generated from the same clock as the picture.

audio.py [OUT.wav]   (default audio.wav; 48 kHz stereo, SECONDS long)

Wind over the canyon, a drone in D that changes chord with each chapter,
one soft ping per commit (pitch by author, on a pentatonic scale, so 2026
turns into a shimmer), a bell for the first commit, a boom and a bell for
every release, and the rumble and crackle of the 2026 eruption.
"""
import random
import sys
import wave

import numpy as np
from scipy import signal

import timeline as TL

SR = 48000
N = TL.SECONDS * SR
T = np.arange(N) / SR
rng = np.random.default_rng(2014)


def env(keys):
    """Smooth envelope over the whole film from (second, value) keys."""
    return np.array([TL._interp(keys, t) for t in np.arange(0, TL.SECONDS + 1, 0.05)])


def stretch(e):
    return np.interp(T, np.arange(len(e)) * 0.05, e)


def lowpass(x, hz, order=2):
    b, a = signal.butter(order, hz / (SR / 2))
    return signal.lfilter(b, a, x)


def bandpass(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band")
    return signal.lfilter(b, a, x)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


out = np.zeros((N, 2))
night = stretch(np.array([TL.night(t) for t in np.arange(0, TL.SECONDS + 1, 0.05)]))
erupt = stretch(np.array([TL.eruption(t) for t in np.arange(0, TL.SECONDS + 1, 0.05)]))

# ---------------------------------------------------------------- wind
for ch in range(2):
    w = rng.standard_normal(N)
    body = lowpass(w, 380)
    gust = bandpass(w, 500, 1800) * 0.25
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.07 + 0.01 * ch) * T + ch) * \
        np.sin(2 * np.pi * 0.023 * T + 1.3)
    level = (0.10 + 0.12 * night) * (1 - 0.6 * erupt)
    out[:, ch] += (body * 2.5 + gust) * lfo * level

# ---------------------------------------------------------------- drone
CHORDS = [  # (from second, MIDI notes)
    (0, [38, 45, 50, 53]),       # Dm
    (11, [34, 41, 46, 50]),      # Bb
    (19, [41, 48, 53, 57]),      # F
    (33, [38, 45, 52]),          # Dm9, sparse: the quiet years
    (41, [38, 45, 50, 53]),      # Dm: 2026 begins
    (55, [34, 41, 46, 50, 53]),  # Bb
    (62, [36, 43, 48, 52, 55]),  # C
    (69, [38, 45, 50, 54, 57]),  # D major: the present
    (80, [38, 45, 52, 54, 57]),  # Dsus2/6, open
]
drone = np.zeros(N)
fade = int(2.5 * SR)
for i, (t0, notes) in enumerate(CHORDS):
    t1 = CHORDS[i + 1][0] if i + 1 < len(CHORDS) else TL.SECONDS
    a, b = int(t0 * SR), min(int(t1 * SR) + fade, N)
    tt = T[a:b]
    g = np.ones(b - a)
    g[:fade] = np.linspace(0, 1, fade) ** 1.5
    if b < N:
        g[-fade:] *= np.linspace(1, 0, fade) ** 1.5
    for n in notes:
        f = hz(n)
        for h, amp in ((1, 1.0), (2, 0.35), (3, 0.12)):
            for det in (-0.6, 0.6):
                drone[a:b] += np.sin(2 * np.pi * (f * h + det) * tt
                                     + rng.uniform(0, 6.28)) * amp * g
drone = lowpass(drone, 1400) / 30
level = stretch(env([(0, 0), (4, 0.4), (33, 0.55), (35, 0.35), (41, 0.45),
                     (69, 0.9), (80, 0.75), (100, 0.6), (TL.SECONDS, 0)]))
out[:, 0] += drone * level
out[:, 1] += np.roll(drone, 240) * level

# ---------------------------------------------------------------- commits
PENTA = [62, 64, 66, 69, 71, 74, 76, 78, 81, 83]  # D major pentatonic
rnd = random.Random(5246)
times = []
for x, a in TL.DATA["commits"]:
    f = TL.frame_of_x(x) + rnd.random()
    if f < 37:
        f = 40 + rnd.uniform(0, 70)
    times.append(((f - 1) / TL.FPS, a))
counts = np.zeros(TL.SECONDS + 1)
for t, _ in times:
    counts[min(int(t), TL.SECONDS)] += 1
L = int(0.35 * SR)
tg = np.arange(L) / SR
for t, a in times:
    note = PENTA[(a * 7 + 3) % len(PENTA)] if a else PENTA[(int(t * 7) % 4) + 2]
    f = hz(note) * (1 + rnd.uniform(-0.003, 0.003))
    dens = counts[min(int(t), TL.SECONDS)]
    amp = 0.09 / np.sqrt(max(dens, 1))
    g = np.sin(2 * np.pi * f * tg) * np.exp(-tg * 14) * (1 - np.exp(-tg * 900))
    g += 0.3 * np.sin(2 * np.pi * f * 2.01 * tg) * np.exp(-tg * 22)
    i = int(t * SR)
    j = min(i + L, N)
    pan = rnd.uniform(0.15, 0.85)
    out[i:j, 0] += g[:j - i] * amp * (1 - pan)
    out[i:j, 1] += g[:j - i] * amp * pan


def bell(t, f0, amp, decay=3.0, partials=((1, 1), (2.76, 0.5), (5.4, 0.25), (8.9, 0.1))):
    i = int(t * SR)
    L = min(int(decay * 2.5 * SR), N - i)
    tt = np.arange(L) / SR
    g = sum(a * np.sin(2 * np.pi * f0 * p * tt) * np.exp(-tt * (1 + p) / decay)
            for p, a in partials)
    out[i:i + L, 0] += g * amp
    out[i:i + L, 1] += np.roll(g, 120) * amp


def boom(t, amp):
    i = int(t * SR)
    L = min(int(2.2 * SR), N - i)
    tt = np.arange(L) / SR
    f = 52 * np.exp(-tt * 0.6) + 34
    ph = 2 * np.pi * np.cumsum(f) / SR
    g = np.sin(ph) * np.exp(-tt * 2.2) * (1 - np.exp(-tt * 200))
    swell_l = int(0.7 * SR)
    a0 = max(i - swell_l, 0)
    sw = bandpass(rng.standard_normal(i - a0), 300, 4000) * \
        np.linspace(0, 1, i - a0) ** 3 * 0.25
    for ch in range(2):
        out[i:i + L, ch] += g * amp
        out[a0:i, ch] += sw * amp


# the first commit, alone in the dark
bell(35 / TL.FPS, hz(74), 0.22, decay=4.5)
for tag in TL.DATA["tags"]:
    t = (TL.frame_of_x(tag["x"]) - 1) / TL.FPS
    boom(t, 0.22)
    bell(t + 0.02, hz(81 if "rc" not in tag["name"] else 86), 0.08, decay=2.5)

# ---------------------------------------------------------------- eruption
rum = lowpass(rng.standard_normal(N), 90, 4) * 6 + 0.4 * np.sin(2 * np.pi * 33 * T)
rum *= 0.7 + 0.3 * np.sin(2 * np.pi * 0.31 * T) * np.sin(2 * np.pi * 0.13 * T)
crack = (rng.random(N) < 0.0009).astype(float) * rng.standard_normal(N)
crack = bandpass(crack, 900, 6000) * 9
for ch in range(2):
    out[:, ch] += rum * erupt * 0.33 + np.roll(crack, ch * 37) * erupt ** 1.5 * 0.12

# ---------------------------------------------------------------- room
ir_len = int(2.4 * SR)
ti = np.arange(ir_len) / SR
wet = np.zeros_like(out)
for ch in range(2):
    ir = rng.standard_normal(ir_len) * np.exp(-ti * 2.6)
    ir = lowpass(ir, 5000)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet[:, ch] = signal.fftconvolve(out[:, ch], ir)[:N]
mix = out * 0.8 + wet * 0.35
fade_in = np.minimum(T / 0.5, 1)
fade_out = np.clip((TL.SECONDS - T) / 3.0, 0, 1)
mix *= (fade_in * fade_out)[:, None]
mix = np.tanh(mix / np.max(np.abs(mix)) * 1.4) / np.tanh(1.4) * 0.9

path = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
with wave.open(path, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print("wrote", path, f"{TL.SECONDS} s,", len(times), "commit pings,",
      len(TL.DATA["tags"]), "releases")
