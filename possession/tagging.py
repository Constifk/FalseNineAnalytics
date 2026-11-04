"""Smart Tagging: Metrica's automatic phases and set pieces, as per-frame evidence.

Real file: SportsCode XML, <instance><start/><end/><code/><label><group/><text/></label></instance>,
seconds from the first frame of the tactical video (DATA.md: frame = round(seconds * 25)).

Dev games have no Smart Tagging, so `simulate()` draws imperfect tags from their events: set-piece
windows around each restart, phase intervals along possessions, with jittered edges, misses,
false alarms, and throw-ins without a team, like the real file (DATA.md).

What a phase's Team tag means is not documented for every code: for "DEFENDING IN ..." it is
plausibly the defending side. `calibrate_signs()` decides that per code without any labels, by
agreement with a tracking-only estimate (allowed: pseudo-labels produced by an algorithm), so
neither the simulation nor the real file depends on our guess.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .match import A, D, H, X

SET_PIECE = "SET PIECES"


def load_sportscode(path, home_names=("barcelona",), away_names=("atletic club", "athletic club"),
                    fps=25):
    """[(code, start_frame, end_frame, team, type)] team in {+1 home, -1 away, 0 unknown}."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        code = (inst.findtext("code") or "").strip().upper()
        start = int(round(float(inst.findtext("start")) * fps))
        end = int(round(float(inst.findtext("end")) * fps))
        team, kind = 0, ""
        for label in inst.iter("label"):
            group = (label.findtext("group") or "").strip().lower()
            text = (label.findtext("text") or "").strip().lower()
            if group == "team":
                team = 1 if text in home_names else -1 if text in away_names else 0
            elif group == "type":
                kind = text
        out.append((code, start, end, team, kind))
    return out


def simulate(gt, runs, seed=0, fps=25, miss=0.15, false_alarm=0.05, edge_jitter_s=1.5):
    """Imperfect tags from a dev game's ground truth (see module docstring)."""
    rng = np.random.default_rng(seed)
    gt = np.asarray(gt)
    tags = []
    for s, e in runs:
        w = gt[s:e]
        breaks = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(w)]))
        for i in range(len(bounds) - 1):
            a, b, code = s + bounds[i], s + bounds[i + 1], w[bounds[i]]
            prev = w[bounds[i - 1]] if i > 0 else D
            if code in (H, A):
                team = 1 if code == H else -1
                if prev == D and rng.random() > miss:      # restart after a stoppage
                    kind = rng.choice(["throw-in", "free kick", "goal kick", "corner kick"],
                                      p=[0.45, 0.25, 0.2, 0.1])
                    c = a + rng.normal(0, 0.5) * fps
                    tags.append((SET_PIECE, int(c - 2.5 * fps), int(c + 2.5 * fps),
                                 0 if kind == "throw-in" else team, kind))
                if (b - a) > 5 * fps and rng.random() > miss:
                    j0, j1 = rng.normal(0, edge_jitter_s, 2) * fps
                    if rng.random() < 0.5:
                        tags.append(("BUILD UP", int(a + j0), int(b + j1), team, ""))
                    else:   # a defending phase, tagged with the defending side
                        tags.append(("DEFENDING IN MIDDLE THIRD", int(a + j0), int(b + j1), -team, ""))
        n_false = rng.poisson(false_alarm * len(bounds))
        for _ in range(n_false):
            c = rng.integers(s, e)
            tags.append((SET_PIECE, int(c - 2.5 * fps), int(c + 2.5 * fps),
                         int(rng.choice([-1, 0, 1])), "free kick"))
    return tags


def calibrate_signs(tags, estimate):
    """{code: +1 or -1}: does the Team tag name the side in possession (+1) or the other (-1)?
    `estimate` is a tracking-only H/A/D/X array; no labels are used."""
    signs = {}
    for code in {t[0] for t in tags if t[0] != SET_PIECE}:
        agree = disagree = 0
        for c, s, e, team, _ in tags:
            if c != code or team == 0:
                continue
            seg = estimate[max(s, 0):max(e, 0)]
            mine = np.sum(seg == (H if team == 1 else A))
            other = np.sum(seg == (A if team == 1 else H))
            agree += mine
            disagree += other
        signs[code] = 1 if agree >= disagree else -1
    return signs


def features(tags, n_frames, signs, fps=25):
    """Per-frame tag evidence: set-piece window, set-piece team, phase team (+1 home / -1 away)."""
    sp = np.zeros(n_frames)
    sp_team = np.zeros(n_frames)
    phase = np.zeros(n_frames)
    for code, s, e, team, _ in tags:
        s, e = max(0, s), min(n_frames, e)
        if e <= s:
            continue
        if code == SET_PIECE:
            sp[s:e] = 1
            sp_team[s:e] = team
        else:
            phase[s:e] += signs.get(code, 1) * team
    return {"tag_sp": sp, "tag_sp_team": sp_team, "tag_phase": np.clip(phase, -1, 1)}
