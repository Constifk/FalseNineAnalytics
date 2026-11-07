"""Write the public 5-minute answer key as subtitles, to watch the tactical video with the labels.

    python -m possession.srt                      # -> data/tactical/BAR-ATH_tactical.srt

VLC (and most players) load `<video name>.srt` automatically when it sits next to the video.
Each label run becomes one subtitle, so you can see exactly where a possession or a dead ball
starts and ends. In VLC, `E` steps one frame forward.

This is for understanding what the labels mean (RULES.md allows watching). Do not turn what you
see into frame numbers, hand labels or game-specific constants.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUT = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical.srt")
FPS = 25
TEXT = {"H": "H  Barça in possession", "A": "A  Athletic in possession", "D": "D  dead ball"}


def stamp(frame: int) -> str:
    ms = round(frame * 1000 / FPS)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def runs(labels):
    """[(label, first frame, last frame)] for every run of equal labels, X skipped."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] != "X":
                out.append((labels[start], start, i - 1))
            start = i
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    labels = pd.read_csv(GT)["label"].astype(str).str.strip().str.upper().tolist()
    cues = runs(labels)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for n, (lab, a, b) in enumerate(cues, 1):
            secs = (b - a + 1) / FPS
            fh.write(f"{n}\n{stamp(a)} --> {stamp(b + 1)}\n"
                     f"{TEXT[lab]}\nframes {a}-{b}  ({secs:.1f} s)\n\n")
    spells = sum(1 for lab, _, _ in cues if lab in "HA")
    print(f"wrote {os.path.relpath(args.out, ROOT)}: {len(cues)} subtitles, {spells} H/A runs")


if __name__ == "__main__":
    main()
