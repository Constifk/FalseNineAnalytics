"""Per-frame features from a Match. Collective only: slots are tracklets, not people.

Everything is computed inside each graded run separately so nothing leaks across half time.
"""
from __future__ import annotations

import warnings

import numpy as np

from .match import AWAY, HOME, Match


def _nan_rolling_mean(x, window):
    """Centred rolling mean ignoring NaN; NaN where the whole window is NaN."""
    x = np.asarray(x, np.float64)
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    half = window // 2
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(x), idx + half + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def interpolate_ball(match: Match, max_gap: int):
    """Ball with gaps up to `max_gap` frames filled linearly (inside a run only).
    Returns (bx, by, interpolated_flag)."""
    bx = match.bx.astype(np.float64).copy()
    by = match.by.astype(np.float64).copy()
    flag = np.zeros(match.n_frames, bool)
    for start, stop in match.runs:
        x, y = bx[start:stop], by[start:stop]
        miss = np.isnan(x)
        if miss.all():
            continue
        edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if s == 0 or e == len(x) or e - s > max_gap:
                continue
            t = np.arange(s, e)
            x[s:e] = np.interp(t, [s - 1, e], [x[s - 1], x[e]])
            y[s:e] = np.interp(t, [s - 1, e], [y[s - 1], y[e]])
            flag[start + s:start + e] = True
    return bx, by, flag


def attack_sign(match: Match):
    """+1 where the home team attacks towards +x, -1 where towards -x, per frame (by run).

    Each team's centroid sits nearer its own goal than the opponent's centroid on almost every
    frame (keeper and back line), so the sign of the median centroid difference over a run is the
    direction. No labels involved.
    """
    sign = np.ones(match.n_frames)
    home = match.team == HOME
    with np.errstate(invalid="ignore"):
        ch = np.nanmean(np.where(home, match.px, np.nan), axis=1)
        ca = np.nanmean(np.where(~home, match.px, np.nan), axis=1)
    for start, stop in match.runs:
        diff = np.nanmedian(ca[start:stop] - ch[start:stop])
        sign[start:stop] = 1.0 if diff > 0 else -1.0
    return sign


def slot_speeds(match: Match, smooth=5):
    """(n_frames, n_slots) speed in m/s from consecutive positions of the same slot."""
    dx = np.diff(match.px, axis=0, prepend=np.nan)
    dy = np.diff(match.py, axis=0, prepend=np.nan)
    v = np.hypot(dx, dy) * match.fps
    v[v > 12] = np.nan  # identity switch inside a slot shows up as a jump
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        ok = ~np.isnan(v)
        num = np.apply_along_axis(lambda c: np.convolve(c, kernel, "same"), 0, np.where(ok, v, 0))
        den = np.apply_along_axis(lambda c: np.convolve(c, kernel, "same"), 0, ok.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(den > 0.5, num / den, np.nan)
    return v


def compute(match: Match, max_interp=25):
    """dict of per-frame float arrays."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _compute(match, max_interp)


def _compute(match: Match, max_interp):
    length, width = match.pitch
    fps = match.fps
    bx, by, interp = interpolate_ball(match, max_interp)
    seen = ~np.isnan(match.bx)
    has_ball = ~np.isnan(bx)
    sign = attack_sign(match)

    home = match.team == HOME
    dist = np.hypot(match.px - bx[:, None], match.py - by[:, None])
    with np.errstate(invalid="ignore", all="ignore"):
        dh = np.nanmin(np.where(home, dist, np.nan), axis=1)
        da = np.nanmin(np.where(~home, dist, np.nan), axis=1)
        # second-nearest of each team: support / pressure around the ball
        n_h2 = np.nansum(np.where(home, dist, np.nan) < 5, axis=1)
        n_a2 = np.nansum(np.where(~home, dist, np.nan) < 5, axis=1)

    bvx = np.gradient(np.where(has_ball, bx, np.nan)) * fps
    bvy = np.gradient(np.where(has_ball, by, np.nan)) * fps
    bspeed = _nan_rolling_mean(np.hypot(bvx, bvy), 5)

    out_margin = np.maximum.reduce([-bx, bx - length, -by, by - width])  # >0 outside the pitch

    v = slot_speeds(match)
    with np.errstate(invalid="ignore", all="ignore"):
        mean_speed = np.nanmean(v, axis=1)
        n_players = np.sum(~np.isnan(match.px), axis=1).astype(float)

        # Team shape, home-attacking frame: x' = sign * (x - L/2)
        xs = (match.px - length / 2) * sign[:, None]
        def team_stats(mask):
            x = np.where(mask, xs, np.nan)
            y = np.where(mask, match.py, np.nan)
            return (np.nanmean(x, axis=1), np.nanstd(x, axis=1), np.nanstd(y, axis=1),
                    np.nanmean(np.where(mask, v, np.nan), axis=1))
        hx, hlen, hwid, hv = team_stats(home)
        ax, alen, awid, av = team_stats(~home)
        # home attacks +x' and away attacks -x': both expressed as "how far up the pitch"
        bxs = (bx - length / 2) * sign

    # time since the ball was last seen (s), and how long it stays missing, inside runs
    since = np.zeros(match.n_frames)
    last = -10 ** 9
    for i in range(match.n_frames):
        if seen[i]:
            last = i
        since[i] = (i - last) / fps
    since = np.minimum(since, 30.0)

    feats = {
        "seen": seen.astype(float),
        "has_ball": has_ball.astype(float),
        "dh": np.clip(np.nan_to_num(dh, nan=40.0), 0, 40),
        "da": np.clip(np.nan_to_num(da, nan=40.0), 0, 40),
        "n_h5": np.nan_to_num(n_h2), "n_a5": np.nan_to_num(n_a2),
        "bspeed": np.clip(np.nan_to_num(bspeed, nan=0.0), 0, 40),
        "out_margin": np.clip(np.nan_to_num(out_margin, nan=-5.0), -10, 10),
        "ball_xs": np.nan_to_num(bxs, nan=0.0),
        "since_seen": since,
        "mean_speed": np.nan_to_num(mean_speed, nan=0.0),
        "mean_speed_1s": np.nan_to_num(_nan_rolling_mean(mean_speed, fps), nan=0.0),
        "mean_speed_4s": np.nan_to_num(_nan_rolling_mean(mean_speed, 4 * fps), nan=0.0),
        "n_players": n_players,
        "home_x": np.nan_to_num(hx), "away_x": np.nan_to_num(-ax),
        "home_len": np.nan_to_num(hlen), "away_len": np.nan_to_num(alen),
        "home_wid": np.nan_to_num(hwid), "away_wid": np.nan_to_num(awid),
        "home_v": np.nan_to_num(hv), "away_v": np.nan_to_num(av),
    }
    # smoothed shape differences: possession shows in shape over seconds, not frames
    for key in ("home_x", "away_x", "home_wid", "away_wid", "home_len", "away_len"):
        feats[key + "_3s"] = _nan_rolling_mean(feats[key], 3 * fps)
    feats["wid_diff_3s"] = feats["home_wid_3s"] - feats["away_wid_3s"]
    feats["x_diff_3s"] = feats["home_x_3s"] - feats["away_x_3s"]
    feats["len_diff_3s"] = feats["home_len_3s"] - feats["away_len_3s"]
    feats["d_diff"] = feats["da"] - feats["dh"]
    return feats
