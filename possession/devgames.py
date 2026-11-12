"""Open development games with a ground truth we derive ourselves, never the challenge game.

The rules forbid tuning on the challenge game, including on its public five minutes. So every
choice in this package is made on Metrica's open sample games (github.com/metrica-sports/sample-data):

  metrica1, metrica2   CSV format
  metrica3             FIFA/EPTS format -- the same format as the challenge ATD

They are clean: one slot per real player and a ball on almost every live frame. noise.py turns
them into something that looks like automatic tracking before any model sees them.

Ground truth comes from the open event data with the challenge's rules (CHALLENGE.md):
  * D from a stoppage (ball out, goal, whistled foul, offside, hand ball) until the restart
  * otherwise the team of the last on-ball event; a change needs a controlling action of the
    other team (pass, carry, recovery, shot, set piece), so a challenge or a touch does not switch
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .match import AWAY, HOME, Match, D, X, load_epts

ROOT = Path(__file__).resolve().parents[1]
DEV_DIR = ROOT / "devdata"
PITCH = (105.0, 68.0)
# Pinned to a commit (the latest, 15.04.2021) and checked by sha256, so a fresh clone trains on
# exactly the bytes we trained on.
BASE_URL = ("https://raw.githubusercontent.com/metrica-sports/sample-data/"
            "e706dd506b360d69d9d123d5b8026e7294b13996/data")
SHA256 = {
    "Sample_Game_1_RawEventsData.csv": "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05",
    "Sample_Game_1_RawTrackingData_Away_Team.csv": "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618",
    "Sample_Game_1_RawTrackingData_Home_Team.csv": "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364",
    "Sample_Game_2_RawEventsData.csv": "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba",
    "Sample_Game_2_RawTrackingData_Away_Team.csv": "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b",
    "Sample_Game_2_RawTrackingData_Home_Team.csv": "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694",
    "Sample_Game_3_events.json": "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0",
    "Sample_Game_3_metadata.xml": "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1",
    "Sample_Game_3_tracking.txt": "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658",
}

OWNING = {"PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE"}


# ----------------------------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------------------------

def download():
    """Fetch the three Metrica games into devdata/ (about 200 MB, only what is missing)."""
    files = {
        "metrica1": [f"Sample_Game_1/Sample_Game_1_{p}.csv" for p in
                     ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team")],
        "metrica2": [f"Sample_Game_2/Sample_Game_2_{p}.csv" for p in
                     ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team")],
        "metrica3": [f"Sample_Game_3/Sample_Game_3_{p}" for p in
                     ("events.json", "metadata.xml", "tracking.txt")],
    }
    for game, names in files.items():
        folder = DEV_DIR / game
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            fetch(f"{BASE_URL}/{name}", folder / Path(name).name, SHA256[Path(name).name])


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, target, digest):
    """Download url to target unless it is already there; either way insist on the pinned hash."""
    from urllib.request import urlretrieve

    if not target.exists():
        print("downloading", target.name)
        tmp = target.with_name(target.name + ".download")
        urlretrieve(url, tmp)
        tmp.replace(target)
    got = sha256(target)
    if got != digest:
        raise SystemExit(f"{target} has sha256 {got}, expected {digest} -- delete it and rerun")


# ----------------------------------------------------------------------------------------------
# Events -> ground truth
# ----------------------------------------------------------------------------------------------

def _is_stoppage(kind, subtypes):
    if kind == "BALL OUT" or kind == "FAULT RECEIVED":
        return True
    if kind == "SHOT" and ("OUT" in subtypes or "GOAL" in subtypes):
        return True
    if kind == "CHALLENGE" and "FAULT" in subtypes and "ADVANTAGE" not in subtypes:
        return True
    if kind == "BALL LOST" and ({"OFFSIDE", "HAND BALL"} & subtypes):
        return True
    if kind in ("PASS", "BALL LOST") and "OUT" in subtypes:
        return True
    return False


def _is_restart(kind, subtypes):
    return kind == "SET PIECE" or (kind == "PASS" and "GOAL KICK" in subtypes)


def ground_truth(events, n_frames):
    """H/A/D/X per frame from a normalised event table, plus the graded runs.

    `events` columns: type, subtypes (frozenset), team (HOME/AWAY), start, end (0-based rows),
    period. Frames outside [first event, last event] of each period are X (never scored).
    """
    gt = np.full(n_frames, X, np.int8)
    runs = []
    for _, half in events.groupby("period", sort=True):
        half = half.sort_values(["start", "order"], kind="stable")
        marks = []           # (frame, code): from this frame on the label is code
        alive, prev_end = False, None
        for ev in half.itertuples(index=False):
            kind, sub, team = ev.type, ev.subtypes, ev.team
            start, end = int(ev.start), int(max(ev.end, ev.start))
            if _is_restart(kind, sub):
                if alive and prev_end is not None and prev_end + 1 < start:
                    marks.append((prev_end + 1, D))   # a restart implies a stoppage before it
                marks.append((start, team))
                alive = True
            elif kind in OWNING:
                if alive or marks:
                    marks.append((start, team))
                    alive = True
            if _is_stoppage(kind, sub):
                marks.append((end + 1 if kind != "FAULT RECEIVED" else start, D))
                alive = False
            prev_end = end if prev_end is None else max(prev_end, end)

        if not marks:
            continue
        first = int(half["start"].min())
        last = int(max(half["end"].max(), half["start"].max())) + 1
        marks.sort(key=lambda m: m[0])
        bounds = [m[0] for m in marks[1:]] + [last]
        for (frame, code), stop in zip(marks, bounds):
            gt[max(frame, first):min(stop, last)] = code
        runs.append((first, min(last, n_frames)))
    return gt, runs


# ----------------------------------------------------------------------------------------------
# Metrica CSV games (1, 2)
# ----------------------------------------------------------------------------------------------

def _read_csv_tracking(path):
    header = pd.read_csv(path, skiprows=2, nrows=0).columns
    names, players = ["period", "frame", "time"], []
    for col in header[3:]:
        if not col.startswith("Unnamed"):
            stem = "ball" if col == "Ball" else col.replace("Player", "").strip()
            names += [f"{stem}_x", f"{stem}_y"]
            if stem != "ball":
                players.append(stem)
    return pd.read_csv(path, skiprows=3, header=None, names=names), players


def _csv_events(path):
    raw = pd.read_csv(path)
    subtypes = raw["Subtype"].fillna("").astype(str)
    return pd.DataFrame({
        "type": raw["Type"].str.upper(),
        "subtypes": [frozenset(s.strip() for s in v.split("-") if s.strip()) for v in subtypes],
        "team": np.where(raw["Team"].str.lower() == "home", HOME, AWAY),
        "start": raw["Start Frame"].astype(int) - 1,
        "end": raw["End Frame"].astype(int) - 1,
        "period": raw["Period"].astype(int),
        "order": np.arange(len(raw)),
    })


def load_csv_game(game):
    folder = DEV_DIR / f"metrica{game}"
    home, home_players = _read_csv_tracking(folder / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
    away, away_players = _read_csv_tracking(folder / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv")
    n = len(home)
    length, width = PITCH
    cols_x = [f"{p}_x" for p in home_players] + [f"{p}_x" for p in away_players]
    cols_y = [f"{p}_y" for p in home_players] + [f"{p}_y" for p in away_players]
    both = pd.concat([home[[f"{p}_{a}" for p in home_players for a in "xy"]],
                      away[[f"{p}_{a}" for p in away_players for a in "xy"]]], axis=1)
    px = both[cols_x].to_numpy(np.float32) * length
    py = both[cols_y].to_numpy(np.float32) * width
    team = np.array([HOME] * len(home_players) + [AWAY] * len(away_players), np.int8)
    bx = home["ball_x"].to_numpy(np.float32) * length
    by = home["ball_y"].to_numpy(np.float32) * width
    gt, runs = ground_truth(_csv_events(folder / f"Sample_Game_{game}_RawEventsData.csv"), n)
    return Match(px, py, team, bx, by, PITCH, runs, f"metrica{game}"), gt


# ----------------------------------------------------------------------------------------------
# Metrica EPTS game (3)
# ----------------------------------------------------------------------------------------------

def _names(value):
    if not value:
        return frozenset()
    if isinstance(value, dict):
        value = [value]
    return frozenset(v["name"].upper() for v in value)


def _json_events(path, home_name):
    data = json.load(open(path))["data"]
    return pd.DataFrame({
        "type": [e["type"]["name"].upper() for e in data],
        "subtypes": [_names(e.get("subtypes")) for e in data],
        "team": [HOME if e["team"]["name"] == home_name else AWAY for e in data],
        "start": [int(e["start"]["frame"]) - 1 for e in data],
        "end": [int(e["end"]["frame"] or e["start"]["frame"]) - 1 for e in data],
        "period": [int(e["period"]) for e in data],
        "order": np.arange(len(data)),
    })


def load_epts_game():
    folder = DEV_DIR / "metrica3"
    match = load_epts(folder / "Sample_Game_3_metadata.xml", folder / "Sample_Game_3_tracking.txt",
                      PITCH, runs=[], name="metrica3")
    # The first <Team> in the metadata (FIFATMA) is home; in the event file it is "Team A".
    gt, runs = ground_truth(_json_events(folder / "Sample_Game_3_events.json", "Team A"),
                            match.n_frames)
    match.runs = runs
    return match, gt


def load(name):
    """('metrica1'..'metrica3' | 'idsse_<match id>') -> (clean Match, ground-truth codes)."""
    if name.startswith("idsse_"):
        from . import idsse
        return idsse.load(name[len("idsse_"):])
    if name == "metrica3":
        return load_epts_game()
    return load_csv_game(int(name[-1]))


METRICA = ("metrica1", "metrica2", "metrica3")
IDSSE = ("idsse_J03WMX", "idsse_J03WN1", "idsse_J03WOH", "idsse_J03WOY", "idsse_J03WPY",
         "idsse_J03WQQ", "idsse_J03WR9")
DEV_GAMES = METRICA + IDSSE   # the order fixes each game's noise seed (bench.prepare)


def download_all(games=None):
    games = DEV_GAMES if games is None else games
    download()
    if any(g.startswith("idsse_") for g in games):
        from . import idsse
        idsse.download()
