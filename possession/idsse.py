"""Seven open DFL games (IDSSE) as extra development games, never the challenge game.

Bassek, Rein, Weber, Memmert (2025): "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data 12, 195 (doi:10.1038/s41597-025-04505-y). Two Bundesliga and five
2. Bundesliga games of 2022/23, 25 fps, licensed CC BY 4.0. We download the copy the PySport
project mirrors for kloppy (huggingface.co/datasets/pysport/idsse-data), pinned to a revision and
checked by sha256.

Why these games: the DFL position feed carries, on every ball frame, `BallStatus` (0 dead, 1 alive)
and `BallPossession` (1 home, 2 guest). The challenge ground truth is derived the same way --
mechanically from the tracking provider's ball status and possession flags, without smoothing
(CHALLENGE.md) -- so the labels here are much closer to the target than labels rebuilt from events.

Ground truth per frame: D where BallStatus is 0, otherwise H/A from BallPossession. Every frame of
a game section that has a ball frame is graded. Positions are the clean optical tracking; noise.py
degrades them like the Metrica games.

The XML is 350-420 MB per game, so the parsed arrays are cached in devdata/idsse/cache/.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np

from .match import AWAY, HOME, Match, A, D, H, X

REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
BASE_URL = f"https://huggingface.co/datasets/pysport/idsse-data/resolve/{REVISION}"
CACHE_VERSION = 1

# match id -> competition id (DFL-COM-000001 Bundesliga, -000002 2. Bundesliga)
GAMES = {
    "J03WMX": "DFL-COM-000001",
    "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}

SHA256 = {
    # positions: the LFS sha256 the mirror publishes; match information: computed on download
    "positions_J03WMX": "d3dd92e509421a6466acb0a083c009035f1751bc5fd9ef42ffa8f20d01cf7d49",
    "positions_J03WN1": "56ef948961a18e150dba0b7b16869d0008ff66f3050a0c355d31fa5b8ec2fcec",
    "positions_J03WOH": "ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f",
    "positions_J03WOY": "a228ef0e676d009cefff1580fcb721b53f598454eb120ec347e0a628a3a4646b",
    "positions_J03WPY": "845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1",
    "positions_J03WQQ": "cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f",
    "positions_J03WR9": "70b63069189019066fb7e3d9db891122edfe787f3a43ba522ebe45ed89ea0269",
    "info_J03WMX": "a884728647a564cf8b20e5223a1d04a5da6f6725e1e506c8651505327c197ba9",
    "info_J03WN1": "5490286436c86e6f80a538d2b1b379d0efd016e0defcaf594bc59d9ef5079f4c",
    "info_J03WOH": "f23b3ee13b988b1ff9a0bce438c416dac47e71355bbc7ee1dab21aaf3e33c18c",
    "info_J03WOY": "4c497fac683eee8292901c18ca02c0be1d9609d0b9d4bd8aa092c61f9e0b19e9",
    "info_J03WPY": "566dd5e4c86e124ab90fcdba51100881f0436e22afa71832945a451a2d95a420",
    "info_J03WQQ": "e976aa0c5dba862cfa898ea57ec226efce9d1bc04b23b8d775908db86aa6fa41",
    "info_J03WR9": "429b584e89df3e3e8b81f4bd0d004b38bd0b4298bb390b8fb43efccb60963e01",
}

SECTIONS = ("firstHalf", "secondHalf")   # extra time, if any, is left out
HALF_GAP = 250                           # empty rows between the halves on our timeline


def _dir():
    from .devgames import DEV_DIR
    return DEV_DIR / "idsse"


def file_name(kind, match_id):
    stem = {"info": "DFL_02_01_matchinformation", "positions": "DFL_04_03_positions_raw_observed"}
    return f"{stem[kind]}_{GAMES[match_id]}_DFL-MAT-{match_id}.xml"


def download():
    """Fetch what is missing (~2.6 GB) and check every file against its pinned hash."""
    from .devgames import fetch
    folder = _dir()
    folder.mkdir(parents=True, exist_ok=True)
    for mid in GAMES:
        for kind in ("info", "positions"):
            name = file_name(kind, mid)
            fetch(f"{BASE_URL}/{name}", folder / name, SHA256[f"{kind}_{mid}"])


# ----------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------

_FRAMESET = re.compile(rb'<FrameSet ([^>]*)>')
_ATTR = re.compile(rb'(\w+)="([^"]*)"')
_FRAME = re.compile(rb'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"')
_BALL = re.compile(rb'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"'
                   rb'[^>]*? BallPossession="(\d)" BallStatus="(\d)"')


def _home_guest(info_path):
    root = ET.parse(info_path).getroot()
    teams = {t.get("Role"): t.get("TeamId") for t in root.iter("Team")}
    env = next(root.iter("Environment"))
    return teams["home"], teams["guest"], (float(env.get("PitchX")), float(env.get("PitchY")))


def _parse(match_id):
    folder = _dir()
    home_id, guest_id, pitch = _home_guest(folder / file_name("info", match_id))
    raw = (folder / file_name("positions", match_id)).read_bytes()
    heads = list(_FRAMESET.finditer(raw))
    sets = []   # (section, team, person, N, X, Y, possession, status)
    for i, head in enumerate(heads):
        attrs = {k.decode(): v.decode() for k, v in _ATTR.findall(head.group(1))}
        if attrs["GameSection"] not in SECTIONS or attrs["TeamId"] == "referee":
            continue
        block = raw[head.end():heads[i + 1].start() if i + 1 < len(heads) else len(raw)]
        is_ball = attrs["TeamId"].upper() == "BALL"
        rows = np.array((_BALL if is_ball else _FRAME).findall(block))
        if not len(rows):
            continue
        if is_ball:
            team = -1
        elif attrs["TeamId"] == home_id:
            team = HOME
        elif attrs["TeamId"] == guest_id:
            team = AWAY
        else:
            continue
        sets.append((attrs["GameSection"], team, attrs["PersonId"], rows[:, 0].astype(np.int64),
                     rows[:, 1].astype(np.float64), rows[:, 2].astype(np.float64),
                     rows[:, 3].astype(np.int8) if is_ball else None,
                     rows[:, 4].astype(np.int8) if is_ball else None))
    del raw

    # timeline: each section from its first to its last ball frame, halves back to back
    offsets, runs, start = {}, [], 0
    for sec in SECTIONS:
        ball = [s for s in sets if s[0] == sec and s[1] == -1]
        if not ball:
            continue
        n0, n1 = int(ball[0][3].min()), int(ball[0][3].max()) + 1
        offsets[sec] = start - n0
        runs.append((start, start + n1 - n0))
        start += n1 - n0 + HALF_GAP
    n = runs[-1][1]

    persons = sorted({(s[1], s[2]) for s in sets if s[1] >= 0})   # home first, then by id
    slot = {p: j for j, p in enumerate(persons)}
    length, width = pitch
    px = np.full((n, len(persons)), np.nan, np.float32)
    py = np.full((n, len(persons)), np.nan, np.float32)
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    gt = np.full(n, X, np.int8)
    for sec, team, person, N, x, y, poss, status in sets:
        if sec not in offsets:
            continue
        rows = N + offsets[sec]
        ok = (rows >= 0) & (rows < n)
        rows, x, y = rows[ok], x[ok] + length / 2, y[ok] + width / 2
        if team < 0:
            bx[rows], by[rows] = x, y
            gt[rows] = np.where(status[ok] == 0, D, np.where(poss[ok] == 1, H, A))
        else:
            px[rows, slot[(team, person)]] = x
            py[rows, slot[(team, person)]] = y
    team = np.array([t for t, _ in persons], np.int8)
    return px, py, team, bx, by, pitch, runs, gt


def load(match_id):
    """match id -> (clean Match, ground-truth codes); parses once, then reads the cache."""
    cache = _dir() / "cache" / f"{match_id}.v{CACHE_VERSION}.npz"
    if not cache.exists():
        px, py, team, bx, by, pitch, runs, gt = _parse(match_id)
        cache.parent.mkdir(parents=True, exist_ok=True)
        import os
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.part.npz")
        np.savez(tmp, px=px, py=py, team=team, bx=bx, by=by, pitch=np.array(pitch),
                 runs=np.array(runs), gt=gt)
        try:
            tmp.replace(cache)
        except PermissionError:   # another process wrote the same cache first (Windows)
            tmp.unlink()
    z = np.load(cache)
    runs = [(int(a), int(b)) for a, b in z["runs"]]
    pitch = (float(z["pitch"][0]), float(z["pitch"][1]))
    return Match(z["px"], z["py"], z["team"], z["bx"], z["by"], pitch, runs,
                 f"idsse_{match_id}"), z["gt"]
