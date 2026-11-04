"""Write a fake challenge bundle from an open dev game, to smoke-test possession.run without the
real data:

    python -m possession.fake_bundle            # -> devdata/fake_bundle/tactical/...
    MTAH_DATA_DIR=devdata/fake_bundle python -m possession.run --team test

The fake has the challenge's file names and format (46 slots, x,y,speed groups, 199319 frames)
but the content of Metrica Sample Game 3 with ATD-like noise. Scores printed for it mean nothing.
"""
from __future__ import annotations

import numpy as np

from . import devgames, noise

N_FRAMES = 199319
OFFSET = 11185 - 360   # put the dev game's kickoff on the challenge's first graded frame


def main():
    clean, gt = devgames.load("metrica3")
    m = noise.degrade(clean, gt, noise.PRESETS["mid"])
    folder = devgames.DEV_DIR / "fake_bundle" / "tactical"
    folder.mkdir(parents=True, exist_ok=True)
    length, width = m.pitch
    players = "".join(f'<Player id="S{i}" teamId="{"T1" if t == 0 else "T2"}"><Name>Track {i + 1}</Name>'
                      f'<ShirtNumber>{i + 1}</ShirtNumber></Player>' for i, t in enumerate(m.team))
    (folder / "BAR-ATH_tactical_atd.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?><main><Metadata><Teams>'
        '<Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>'
        f"<Players>{players}</Players></Metadata></main>")

    def fmt(x, y):
        return "NaN,NaN,NaN" if np.isnan(x) else f"{x / length:.5f},{y / width:.5f},0.00"

    with open(folder / "BAR-ATH_tactical_atd.txt", "w", newline="\n") as out:
        for row in range(N_FRAMES):
            src = row - OFFSET
            if 0 <= src < m.n_frames:
                groups = ";".join(fmt(m.px[src, j], m.py[src, j]) for j in range(m.px.shape[1]))
                ball = fmt(m.bx[src], m.by[src])
            else:
                groups = ";".join(["NaN,NaN,NaN"] * m.px.shape[1])
                ball = "NaN,NaN,NaN"
            out.write(f"{row + 1}:{groups}:{ball}\n")
    print("wrote", folder)


if __name__ == "__main__":
    main()
