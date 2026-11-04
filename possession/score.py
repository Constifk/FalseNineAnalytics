"""Our own implementation of the two headline metrics, written from EVALUATION.md and the
docstrings of evaluation/possession_metrics.py.

Used for the development benchmark only. The official file stays the authority: before any
submission, score with `python evaluation/run_evaluation.py` and check that both agree on the
public window (tests_team/test_score.py has the hand-checked cases from EVALUATION.md).

  spell_f1   possessions = maximal H or A runs inside a graded run, X forward-filled (back-filled
             at the head of a run); one-to-one greedy match by temporal IoU, same team only;
             F1 averaged over IoU 0.3 / 0.4 / 0.5 / 0.6 / 0.7
  macro_f1   mean per-class F1 over H, A, D on graded frames; a predicted X counts as a miss
"""
from __future__ import annotations

import numpy as np

from .match import A, D, H, X

SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def valid_runs(gt):
    graded = (np.asarray(gt) != X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def fill_abstentions(pred, runs):
    filled = np.asarray(pred).copy()
    for start, stop in runs:
        window = filled[start:stop]
        known = np.flatnonzero(window != X)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != X, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        filled[start:stop] = window[carry]
    return filled


def segments(codes, runs):
    out = []
    for start, stop in runs:
        window = codes[start:stop]
        if not len(window):
            continue
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(len(bounds) - 1):
            out.append((start + int(bounds[i]), start + int(bounds[i + 1]), int(window[bounds[i]])))
    return out


def spells(codes, runs):
    return [(a, b, k) for a, b, k in segments(codes, runs) if k in (H, A)]


def _overlaps(gt_spells, pred_spells):
    """Same-team overlapping pairs, IoU descending. Sweep instead of all pairs: both lists are
    sorted and non-overlapping within themselves."""
    out, j0 = [], 0
    for i, (gs, ge, gk) in enumerate(gt_spells):
        while j0 < len(pred_spells) and pred_spells[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < len(pred_spells) and pred_spells[j][0] < ge:
            ps, pe, pk = pred_spells[j]
            if pk == gk:
                inter = min(ge, pe) - max(gs, ps)
                union = max(ge, pe) - min(gs, ps)
                out.append((inter / union, i, j))
            j += 1
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    return out


def spell_f1(gt, pred, runs=None, ious=SPELL_IOUS, detail=False):
    gt = np.asarray(gt)
    runs = valid_runs(gt) if runs is None else runs
    filled = fill_abstentions(pred, runs)
    g, p = spells(gt, runs), spells(filled, runs)
    overlaps = _overlaps(g, p)
    f1s = []
    for threshold in ious:
        used_g, used_p = set(), set()
        for iou, i, j in overlaps:
            if iou < threshold:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
        tp = len(used_g)
        precision = tp / len(p) if p else 0.0
        recall = tp / len(g) if g else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    mean = float(np.mean(f1s))
    if detail:
        return {"spell_f1": mean, "per_iou": dict(zip(ious, f1s)), "n_gt": len(g), "n_pred": len(p)}
    return mean


def macro_f1(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    graded = gt != X
    g, p = gt[graded], pred[graded]
    scores = []
    for k in (H, A, D):
        tp = np.sum((g == k) & (p == k))
        fp = np.sum((g != k) & (p == k))
        fn = np.sum((g == k) & (p != k))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores))


def evaluate(gt, pred):
    d = spell_f1(gt, pred, detail=True)
    d["macro_f1"] = macro_f1(gt, pred)
    return d
