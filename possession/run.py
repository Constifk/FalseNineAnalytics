"""Produce the submission for the challenge game. Run locally, never in a cloud AI tool.

    python -m possession.run --team <teamname>

1. trains the frame classifier on the open dev games (devdata/, downloaded if missing)
2. reads the challenge ATD from data/ (MTAH_DATA_DIR moves it)
3. decodes, writes out/<teamname>.csv and checks it against the submission format

Deterministic: same code + same data bundle -> byte-identical file.

What it prints about the challenge data is aggregate only, and each number has a published
counterpart in DATA.md, so the printout doubles as a loader check:
  ball present on graded frames   DATA.md: 59.3 %
  tracked players per graded frame DATA.md: 20.5
  track slots                      DATA.md: 46
Paste the printout, not the data.

It deliberately does not score itself on the public 5-minute window: that window is not a tuning
set (RULES.md), and a number printed on every run invites tuning on it. The format is checked by
the official evaluation/validate_submission.py.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np

from . import bench, devgames, features, methods, noise, score, tagging
from .match import X, load_epts, write_submission

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chosen on the open dev games (python -m possession.bench), never on the challenge game.
DECODE = methods.DecodeParams(beta=1.0, switch=8.0, stop=32.0, start=32.0)
TRAIN_PRESETS = ("mid", "alive_hard", "dead_dark")


def train_model(tags):
    devgames.download_all()

    def samples():   # one game at a time, so only the sampled training rows stay in memory
        for g in devgames.DEV_GAMES:
            clean, gt = devgames.load(g)
            for p in TRAIN_PRESETS:
                m, f = bench.noisy(g, clean, gt, p, tags=tags)
                yield f, gt, m.runs

    return methods.V2(decode=DECODE).fit(samples())


def load_challenge():
    sys.path.insert(0, ROOT)
    from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames

    paths = asset_paths()
    for key in ("atd_meta", "atd_raw"):
        if not os.path.exists(paths[key]):
            raise SystemExit(f"missing {paths[key]} -- unpack the data bundle into data/ (DATA.md)")
    runs = [(start, stop + 1) for _, start, stop in half_video_blocks(GAME)]
    match = load_epts(paths["atd_meta"], paths["atd_raw"], GAMES[GAME]["pitch"], runs,
                      n_frames=source_frames("tactical", GAME), name=GAME)
    match.extra["smart_tagging"] = paths.get("smart_tagging")
    return match


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", required=True, help="team name, used as the file name")
    ap.add_argument("--method", default="v2", choices=("v0", "v1", "v2"))
    ap.add_argument("--use-tags", action="store_true", help="include Smart Tagging set-piece windows")
    args = ap.parse_args()
    t0 = time.time()

    match = load_challenge()
    graded = np.zeros(match.n_frames, bool)
    for s, e in match.runs:
        graded[s:e] = True
    ball = ~np.isnan(match.bx)
    players = np.sum(~np.isnan(match.px), axis=1)
    print(f"loaded {match.n_frames} frames, {match.px.shape[1]} track slots  (DATA.md: 199319, 46)")
    print(f"ball present on graded frames: {100 * ball[graded].mean():.1f} %  (DATA.md: 59.3 %)")
    print(f"tracked players per graded frame: {players[graded].mean():.1f}  (DATA.md: 20.5)")

    f = features.compute(match)
    if args.method == "v0":
        codes = methods.v0_baseline(match)
    elif args.method == "v1":
        codes = methods.v1_rules(match, f)
    else:
        tag_path = match.extra.get("smart_tagging")
        use_tags = args.use_tags and tag_path and os.path.exists(tag_path)
        if use_tags:
            tags = tagging.load_sportscode(tag_path, fps=match.fps)
            signs = bench.add_tags(match, f, tags)
            n_flip = sum(1 for v in signs.values() if v < 0)
            print(f"smart tagging: {len(tags)} instances, {len(signs)} phase codes, "
                  f"{n_flip} read as naming the defending side")
        codes = train_model(bool(use_tags)).predict(match, f)
    codes = np.where(graded, codes, X).astype(np.int8)

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    path = os.path.join(ROOT, "out", f"{args.team}.csv")
    write_submission(path, codes)
    n_spells = len(score.spells(codes, match.runs))
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"wrote {os.path.relpath(path, ROOT)}  ({n_spells} possessions, sha256 {digest})")
    print(f"done in {time.time() - t0:.0f}s")
    print("next: python evaluation/validate_submission.py", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
