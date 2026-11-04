"""Possession estimators. Each takes a Match (and its features) and returns H/A/D/X codes.

  v0_baseline   the official baseline, re-implemented: nearest player, D without ball, majority
                vote over +/- window frames
  v1_rules      control with memory (from symmetrica), made robust to a missing ball: the team
                keeps the ball while it is unseen, D comes from the ball leaving the pitch or
                everybody slowing down
  v2_hmm        frame classifier (multinomial logistic regression on collective features) as the
                emission of a three-state H/A/D model, decoded with Viterbi; switching costs are
                what keeps the number of possessions realistic
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import features as F
from .match import A, D, H, X, Match


def _outside_runs(match, codes):
    out = np.full(match.n_frames, X, np.int8)
    for s, e in match.runs:
        out[s:e] = codes[s:e]
    return out


# ----------------------------------------------------------------------------------------------
# v0
# ----------------------------------------------------------------------------------------------

def v0_baseline(match: Match, feats=None, window=50):
    feats = feats or F.compute(match, max_interp=0)
    seen = feats["seen"] > 0
    raw = np.where(feats["dh"] <= feats["da"], H, A)
    home = np.cumsum(np.concatenate(([0], seen & (raw == H))))
    away = np.cumsum(np.concatenate(([0], seen & (raw == A))))
    i = np.arange(match.n_frames)
    lo, hi = np.maximum(0, i - window), np.minimum(match.n_frames, i + window + 1)
    nh, na = home[hi] - home[lo], away[hi] - away[lo]
    codes = np.where(nh + na == 0, D, np.where(nh >= na, H, A)).astype(np.int8)
    return codes  # the official baseline writes H/A/D on every frame, X nowhere


# ----------------------------------------------------------------------------------------------
# v1
# ----------------------------------------------------------------------------------------------

@dataclass
class RuleParams:
    radius: float = 1.5          # m: a player this close to a slow ball controls it
    max_ball_speed: float = 8.0  # m/s
    min_frames: int = 5          # consecutive control frames before possession switches
    dead_speed: float = 1.2      # m/s: mean player speed below this (4 s mean) -> stoppage
    out_margin: float = 0.5      # m beyond the line -> ball out


def v1_rules(match: Match, feats=None, p: RuleParams = RuleParams()):
    f = feats or F.compute(match)
    n = match.n_frames
    closest = np.minimum(f["dh"], f["da"])
    controller = np.where(f["dh"] < f["da"], H, A)
    control = (f["has_ball"] > 0) & (closest < p.radius) & (f["bspeed"] < p.max_ball_speed)
    out = (f["has_ball"] > 0) & (f["out_margin"] > p.out_margin)
    slow = f["mean_speed_4s"] < p.dead_speed

    codes = np.full(n, X, np.int8)
    for s, e in match.runs:
        state, run_team, run_len = D, -1, 0
        for i in range(s, e):
            if control[i] and not out[i]:
                c = controller[i]
                run_len = run_len + 1 if c == run_team else 1
                run_team = c
                if run_len >= p.min_frames and state != c:
                    state = c
            else:
                run_len = 0
            if out[i] or (slow[i] and not control[i]):
                state = D
            codes[i] = state
    return codes


# ----------------------------------------------------------------------------------------------
# v2
# ----------------------------------------------------------------------------------------------

def design(f):
    """Feature matrix for the frame classifier: raw collective features plus a few nonlinear
    terms a linear model cannot build itself."""
    seen = f["has_ball"]
    cols = [
        seen, f["seen"],
        seen * np.tanh(f["d_diff"] / 3.0), seen * np.log1p(f["dh"]), seen * np.log1p(f["da"]),
        seen * (np.minimum(f["dh"], f["da"]) < 1.5), seen * f["n_h5"], seen * f["n_a5"],
        seen * np.minimum(f["bspeed"], 30) / 10,
        seen * np.clip(f["out_margin"], -3, 5), seen * (f["out_margin"] > 0.5),
        seen * f["ball_xs"] / 50,
        np.log1p(f["since_seen"]),
        f["mean_speed"], f["mean_speed_1s"], f["mean_speed_4s"],
        (f["mean_speed_4s"] < 1.2).astype(float), (f["mean_speed_4s"] < 1.6).astype(float),
        f["n_players"] / 22,
        f["wid_diff_3s"] / 5, f["x_diff_3s"] / 10, f["len_diff_3s"] / 5,
        f["home_x_3s"] / 20, f["away_x_3s"] / 20,
        f["home_wid_3s"] / 10, f["away_wid_3s"] / 10,
        f["home_v"] - f["away_v"],
        (1 - seen) * f["wid_diff_3s"] / 5, (1 - seen) * f["x_diff_3s"] / 10,
    ]
    if "tag_sp" in f:   # Smart Tagging evidence (tagging.py), when available
        cols += [f["tag_sp"], f["tag_sp_team"], f["tag_phase"], (1 - seen) * f["tag_phase"]]
    return np.column_stack([np.nan_to_num(np.asarray(c, float)) for c in cols])


@dataclass
class LogReg:
    """Multinomial logistic regression, numpy only, full-batch gradient descent with L2."""
    l2: float = 1e-3
    steps: int = 400
    lr: float = 0.5
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)
    W: np.ndarray = field(default=None, repr=False)
    prior: np.ndarray = field(default=None, repr=False)

    def fit(self, Xm, y, k=3):
        self.mean_, self.std_ = Xm.mean(0), Xm.std(0) + 1e-6
        Z = np.column_stack([(Xm - self.mean_) / self.std_, np.ones(len(Xm))])
        Y = np.eye(k)[y]
        W = np.zeros((Z.shape[1], k))
        # Adam
        m, v = np.zeros_like(W), np.zeros_like(W)
        for t in range(1, self.steps + 1):
            logits = Z @ W
            logits -= logits.max(1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(1, keepdims=True)
            g = Z.T @ (P - Y) / len(Z) + self.l2 * W
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            W -= 0.05 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        self.W = W
        self.prior = Y.mean(0)
        return self

    def log_proba(self, Xm):
        Z = np.column_stack([(Xm - self.mean_) / self.std_, np.ones(len(Xm))])
        logits = Z @ self.W
        logits -= logits.max(1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(1, keepdims=True))


@dataclass
class DecodeParams:
    beta: float = 0.2         # emission temperature: frames are correlated, so < 1
    switch: float = 8.0       # cost H <-> A
    stop: float = 6.0         # cost alive -> D
    start: float = 6.0        # cost D -> alive


def viterbi(log_em, p: DecodeParams):
    """Most likely H/A/D path under per-frame emission scores and fixed transition costs."""
    n = len(log_em)
    cost = np.array([[0, p.switch, p.stop],
                     [p.switch, 0, p.stop],
                     [p.start, p.start, 0]], float)
    back = np.zeros((n, 3), np.int8)
    score = log_em[0].copy()
    for t in range(1, n):
        cand = score[:, None] - cost          # from i (rows) to j (cols)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + log_em[t]
    path = np.zeros(n, np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


class V2:
    def __init__(self, decode: DecodeParams = DecodeParams(), clf: LogReg | None = None):
        self.decode = decode
        self.clf = clf

    def fit(self, samples, every=4):
        """samples: [(features dict, gt codes, runs)]"""
        Xs, ys = [], []
        for f, gt, runs in samples:
            Xm = design(f)
            for s, e in runs:
                idx = np.arange(s, e, every)
                idx = idx[gt[idx] != X]
                Xs.append(Xm[idx])
                ys.append(gt[idx])
        self.clf = LogReg().fit(np.vstack(Xs), np.concatenate(ys))
        return self

    def emissions(self, f):
        lp = self.clf.log_proba(design(f))
        return lp - np.log(self.clf.prior)

    def predict(self, match: Match, f=None, em=None):
        f = f or F.compute(match)
        em = self.emissions(f) if em is None else em
        codes = np.full(match.n_frames, X, np.int8)
        for s, e in match.runs:
            codes[s:e] = viterbi(self.decode.beta * em[s:e], self.decode)
        return codes
