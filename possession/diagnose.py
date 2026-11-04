"""Label-free comparison of the challenge ATD with the simulated dev games. Run locally:

    python -m possession.diagnose

Prints aggregate statistics only (no frames, no labels, no ground truth), one line for the
challenge game and one per simulated dev game. If the challenge line differs a lot from the dev
lines, noise.py does not imitate the real tracking well and its presets need adjusting. Paste the
printout to Jarvis.
"""
from __future__ import annotations

import warnings

import numpy as np

from . import devgames, features, noise, run


def stats(m, label):
    graded = np.zeros(m.n_frames, bool)
    for s, e in m.runs:
        graded[s:e] = True
    miss = np.isnan(m.bx) & graded
    edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    gaps = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / m.fps
    f = features.compute(m)
    ms = f["mean_speed_4s"][graded]
    seen = graded & ~np.isnan(m.bx)
    bs = f["bspeed"][seen]
    out = f["out_margin"][seen] > 0.5
    minutes = graded.sum() / m.fps / 60
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
    print(f"{label:16s} ball gaps/min {len(gaps) / minutes:5.1f} | gap s q50 {q(gaps, 50):.2f} "
          f"q90 {q(gaps, 90):.1f} q99 {q(gaps, 99):.1f} | players speed q10 {q(ms, 10):.2f} "
          f"q50 {q(ms, 50):.2f} q90 {q(ms, 90):.2f} | ball speed q50 {q(bs, 50):.1f} "
          f"q90 {q(bs, 90):.1f} | ball outside {100 * out.mean():.1f} % | "
          f"players {np.mean(np.sum(~np.isnan(m.px[graded]), 1)):.1f}")


def main():
    warnings.simplefilter("ignore")
    stats(run.load_challenge(), "CHALLENGE")
    clean, gt = devgames.load("metrica3")
    stats(clean, "dev clean")
    for name, cfg in noise.PRESETS.items():
        stats(noise.degrade(clean, gt, cfg), "dev " + name)


if __name__ == "__main__":
    main()
