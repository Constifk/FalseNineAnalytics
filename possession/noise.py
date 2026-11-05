"""Make a clean open game look like automatic tracking (ATD).

What DATA.md says about the challenge ATD, and how it is imitated here:

  * no player identities, 46 slots that appear, vanish and swap  -> every player's track is cut
    into tracklets (exponential lengths) and each tracklet lands in a random free slot of its team
  * ~20.5 of 22 players visible on average                         -> random dropout bursts
  * ball missing on 41 % of graded frames, more often in crowds,
    duels and in the air                                            -> gaps whose start rate rises
    with crowding and ball speed, lognormal lengths
  * the ball can be visible while it is dead (carried to a throw-in,
    placed for a free kick)                                         -> the open games mostly have
    no ball while dead, so it is drawn on a straight line between the last and next sighting and
    then thinned with its own missing rate

The two missing rates are parameters on purpose: we do not know the real split between live and
dead frames, so the benchmark runs several settings and a model only counts as better if it is
better under all of them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .match import AWAY, D, HOME, Match


@dataclass
class NoiseConfig:
    ball_missing_alive: float = 0.25     # target share of live frames without a ball
    ball_missing_dead: float = 0.85      # target share of dead frames without a ball
    gap_median_frames: float = 250.0     # lognormal gap length, median (~10 s, matching ATD q50=10.7s)
    gap_sigma: float = 1.15
    crowd_boost: float = 3.0             # gap start rate x (1 + boost) when >= 2 players within 2 m
    air_boost: float = 2.0               # ... and x (1 + boost) when the ball moves > 15 m/s
    ball_jitter_m: float = 0.20
    player_jitter_m: float = 0.04        # keeps q10/q50/q90 player speed matching real ATD (0.71/1.54/2.86)
    tracklet_mean_s: float = 30.0
    player_dropout: float = 0.07         # share of player-frames lost in bursts
    slots_per_team: int = 23
    seed: int = 0


PRESETS = {
    "mid":        NoiseConfig(),
    "alive_hard": NoiseConfig(ball_missing_alive=0.38, ball_missing_dead=0.55, gap_median_frames=180.0),
    "dead_dark":  NoiseConfig(ball_missing_alive=0.15, ball_missing_dead=0.95, gap_median_frames=300.0),
}


def _gap_mask(n, rate_weight, target, cfg, rng):
    """Boolean mask of missing frames: gaps start with a rate proportional to rate_weight, and the
    base rate is bisected until the missing share hits `target`."""
    if n == 0 or target <= 0:
        return np.zeros(n, bool)
    u = rng.random(n)
    lengths = np.maximum(1, rng.lognormal(np.log(cfg.gap_median_frames), cfg.gap_sigma, n)
                         .astype(np.int64))

    def build(base):
        starts = np.flatnonzero(u < base * rate_weight)
        delta = np.zeros(n + 1, np.int64)
        np.add.at(delta, starts, 1)
        np.add.at(delta, np.minimum(starts + lengths[starts], n), -1)
        return np.cumsum(delta[:-1]) > 0

    lo, hi = 0.0, 1.0 / max(rate_weight.mean(), 1e-9)
    for _ in range(30):
        mid = (lo + hi) / 2
        if build(mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return build((lo + hi) / 2)


def _smooth_noise(rng, shape, sd, frames):
    """Gaussian noise low-passed over `frames` (moving average), rescaled to standard deviation
    `sd`. Real tracking error drifts; white per-frame noise would fake 5 m/s of running."""
    white = rng.normal(0, 1, shape)
    if frames > 1:
        kernel = np.ones(frames) / frames
        white = np.apply_along_axis(lambda c: np.convolve(c, kernel, "same"), 0, white)
        white /= max(white.std(), 1e-9)
    return sd * white


def _fill_ball_line(bx, by):
    """Straight-line fill of every NaN stretch between two sightings (edges stay NaN)."""
    out_x, out_y = bx.copy(), by.copy()
    seen = np.flatnonzero(~np.isnan(bx))
    if len(seen) < 2:
        return out_x, out_y
    idx = np.arange(len(bx))
    inside = (idx > seen[0]) & (idx < seen[-1])
    out_x[inside] = np.interp(idx[inside], seen, bx[seen])
    out_y[inside] = np.interp(idx[inside], seen, by[seen])
    return out_x, out_y


def degrade(match: Match, gt: np.ndarray, cfg: NoiseConfig = NoiseConfig()) -> Match:
    """A noisy copy of `match`. `gt` is only used to know which frames are live or dead."""
    rng = np.random.default_rng(cfg.seed)
    n, k = match.px.shape
    length, width = match.pitch
    dead = np.asarray(gt) == D
    alive = ~dead

    # ---- ball ------------------------------------------------------------------------------
    bx, by = _fill_ball_line(match.bx.astype(np.float64), match.by.astype(np.float64))
    near = np.hypot(match.px - bx[:, None], match.py - by[:, None])
    crowd = np.nansum(near < 2.0, axis=1) >= 2
    speed = np.hypot(np.diff(bx, prepend=bx[0]), np.diff(by, prepend=by[0])) * match.fps
    air = np.nan_to_num(speed) > 15.0
    weight = (1 + cfg.crowd_boost * crowd) * (1 + cfg.air_boost * air)

    missing = np.zeros(n, bool)
    a_idx, d_idx = np.flatnonzero(alive), np.flatnonzero(dead)
    missing[a_idx] = _gap_mask(len(a_idx), weight[a_idx], cfg.ball_missing_alive, cfg, rng)
    missing[d_idx] = _gap_mask(len(d_idx), np.ones(len(d_idx)), cfg.ball_missing_dead, cfg, rng)
    bx = bx + _smooth_noise(rng, n, cfg.ball_jitter_m, 5)
    by = by + _smooth_noise(rng, n, cfg.ball_jitter_m, 5)
    bx[missing] = np.nan
    by[missing] = np.nan

    # ---- players: tracklets in anonymous slots -----------------------------------------------
    n_slots = 2 * cfg.slots_per_team
    px = np.full((n, n_slots), np.nan, np.float32)
    py = np.full((n, n_slots), np.nan, np.float32)
    team = np.array([HOME] * cfg.slots_per_team + [AWAY] * cfg.slots_per_team, np.int8)
    busy_until = np.full(n_slots, -1, np.int64)
    mean_len = cfg.tracklet_mean_s * match.fps

    pieces = []
    for j in range(k):
        visible = ~np.isnan(match.px[:, j])
        # dropout bursts
        drop = _gap_mask(n, np.ones(n), cfg.player_dropout, cfg, rng) if visible.any() else visible
        visible &= ~drop
        edges = np.diff(np.concatenate(([0], visible.astype(np.int8), [0])))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            t = s
            while t < e:
                cut = min(e, t + max(25, int(rng.exponential(mean_len))))
                pieces.append((t, cut, j))
                t = cut
    pieces.sort()
    jitter_x = _smooth_noise(rng, (n, k), cfg.player_jitter_m, match.fps)
    jitter_y = _smooth_noise(rng, (n, k), cfg.player_jitter_m, match.fps)
    for s, e, j in pieces:
        side = match.team[j]
        candidates = np.flatnonzero((team == side) & (busy_until < s))
        if not len(candidates):
            continue  # more simultaneous tracklets than slots: the tracker drops one
        slot = rng.choice(candidates)
        busy_until[slot] = e
        px[s:e, slot] = match.px[s:e, j] + jitter_x[s:e, j]
        py[s:e, slot] = match.py[s:e, j] + jitter_y[s:e, j]

    out = Match(px, py, team, bx.astype(np.float32), by.astype(np.float32), match.pitch,
                list(match.runs), f"{match.name}+noise{cfg.seed}", match.fps)
    return out
