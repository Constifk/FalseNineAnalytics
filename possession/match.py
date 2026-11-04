"""One match as plain numpy arrays, and the loader for Metrica's FIFA/EPTS tracking export.

Every model in this package works on a `Match`, so the same code runs unchanged on the challenge
game and on the open development games (see devgames.py).

Coordinates are metres with the origin at a corner flag: x along the pitch in [0, length], y across
it in [0, width]. The EPTS export is normalised to 0-1, so a position is simply x * length.

Labels use the scorer's integer codes (H=0, A=1, D=2, X=3), so a prediction array can go straight
into evaluation/possession_metrics.py.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

FPS = 25
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
HOME, AWAY = 0, 1


@dataclass
class Match:
    """Tracking of one match on the video timeline. Row i is video frame i."""

    px: np.ndarray            # (n_frames, n_slots) float32 metres, NaN where the slot is not seen
    py: np.ndarray
    team: np.ndarray          # (n_slots,) int8, HOME or AWAY
    bx: np.ndarray            # (n_frames,) float32 metres, NaN where the ball is not seen
    by: np.ndarray
    pitch: tuple[float, float]  # (length, width) in metres
    runs: list[tuple[int, int]]  # graded half-open spans [start, stop), one per half
    name: str = "match"
    fps: int = FPS
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.bx)

    def copy(self) -> "Match":
        return Match(self.px.copy(), self.py.copy(), self.team.copy(), self.bx.copy(),
                     self.by.copy(), self.pitch, list(self.runs), self.name, self.fps,
                     dict(self.extra))


# ----------------------------------------------------------------------------------------------
# EPTS loader
# ----------------------------------------------------------------------------------------------

def _strip_ns(root):
    for element in root.iter():
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]
    return root


def read_epts_meta(meta_path):
    """Teams, players and the frame-range -> column layout of an EPTS metadata XML.

    Returns (slot_ids, slot_team, layouts):
      slot_ids   list of player ids, one slot per <Player>, in file order
      slot_team  int8 array, HOME for the first <Team> in the file (the EPTS convention), else AWAY
      layouts    [(first_frame_id, last_frame_id, [slot index per player group])]; empty when the
                 file has no DataFormatSpecification, in which case group i is slot i
    """
    root = _strip_ns(ET.parse(meta_path).getroot())
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    slot_ids = [pid for pid, _ in players]
    slot_team = np.array([HOME if tid == team_ids[0] else AWAY for _, tid in players], np.int8)
    index_of = {pid: i for i, pid in enumerate(slot_ids)}

    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        groups = []
        for ref in spec.iter("PlayerChannelRef"):
            player = channel_player.get(ref.get("playerChannelId"))
            if ref.get("playerChannelId", "").endswith("_x") and player in index_of:
                groups.append(index_of[player])
        if groups:
            layouts.append((int(spec.get("startFrame")), int(spec.get("endFrame")), groups))
    return slot_ids, slot_team, layouts


def _parse_pair(text):
    """First two numbers of 'x,y' or 'x,y,s'. NaN stays NaN."""
    first = text.find(",")
    second = text.find(",", first + 1)
    return float(text[:first]), float(text[first + 1:second if second > 0 else None])


def load_epts(meta_path, raw_path, pitch, runs, n_frames=None, name="match"):
    """Read an EPTS metadata + raw text pair into a Match.

    Raw lines are '<frame_id>:<group>;<group>;...:<ball>', each group 'x,y' or 'x,y,speed'. The
    frame_id counts from 1 and the video from 0, so row = frame_id - 1.
    """
    slot_ids, slot_team, layouts = read_epts_meta(meta_path)
    n_slots = len(slot_ids)
    length, width = pitch

    rows, xs, ys, ball = [], [], [], []
    default_layout = list(range(n_slots))
    layout_iter = iter(layouts)
    current = next(layout_iter, None)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame_id = int(parts[0])
            while current is not None and frame_id > current[1]:
                current = next(layout_iter, None)
            layout = current[2] if current is not None and frame_id >= current[0] \
                else default_layout
            fx = np.full(n_slots, np.nan, np.float32)
            fy = np.full(n_slots, np.nan, np.float32)
            for slot, group in zip(layout, parts[1].split(";")):
                if group[0] != "N":
                    fx[slot], fy[slot] = _parse_pair(group)
            rows.append(frame_id - 1)
            xs.append(fx)
            ys.append(fy)
            ball.append(_parse_pair(parts[2]) if parts[2][0] != "N" else (np.nan, np.nan))

    rows = np.asarray(rows)
    n = int(n_frames if n_frames is not None else rows.max() + 1)
    px = np.full((n, n_slots), np.nan, np.float32)
    py = np.full((n, n_slots), np.nan, np.float32)
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    keep = (rows >= 0) & (rows < n)
    px[rows[keep]] = np.asarray(xs, np.float32)[keep] * length
    py[rows[keep]] = np.asarray(ys, np.float32)[keep] * width
    b = np.asarray(ball, np.float32)
    bx[rows[keep]] = b[keep, 0] * length
    by[rows[keep]] = b[keep, 1] * width
    return Match(px, py, slot_team, bx, by, tuple(pitch), list(runs), name)


# ----------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------

def write_submission(path, codes):
    """frame,label CSV, one row per video frame. `codes` are H/A/D/X integers."""
    letters = LETTERS[np.asarray(codes, dtype=np.int64)]
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{c}\n" for i, c in enumerate(letters))
    return path
