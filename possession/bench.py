"""Development benchmark: leave one open game out, train and tune on the others, score the
held-out one under every noise preset. Nothing here ever touches the challenge data.

    python -m possession.bench                 # all methods, all folds
    python -m possession.bench --quick         # one preset, smaller tuning grid
    python -m possession.bench --quick --train metrica   # learn from the 3 Metrica games only
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from . import devgames, features, methods, noise, score, tagging

GRID = {
    "switch": (4.0, 8.0, 16.0, 32.0, 64.0),
    "stop": (4.0, 8.0, 16.0, 32.0),
    "start": (4.0, 8.0, 16.0, 32.0),
}


def add_tags(m, f, tags):
    """Calibrate tag signs against the tracking-only v1 estimate, then add tag features."""
    estimate = methods.v1_rules(m, f)
    signs = tagging.calibrate_signs(tags, estimate)
    f.update(tagging.features(tags, m.n_frames, signs))
    return signs


def noisy(game, clean, gt, preset, s=0, tags=False):
    """One degraded copy of an open game and its features; the seed depends only on the game's
    place in DEV_GAMES, the preset's settings and s."""
    seed = s + 17 * devgames.DEV_GAMES.index(game)
    cfg = noise.NoiseConfig(**{**noise.PRESETS[preset].__dict__, "seed": seed})
    m = noise.degrade(clean, gt, cfg)
    f = features.compute(m)
    if tags:
        add_tags(m, f, tagging.simulate(gt, m.runs, seed=seed))
    return m, f


def prepare(presets, seeds=(0,), tags=False, games=devgames.DEV_GAMES):
    """{(game, preset, seed): (noisy match, gt, features)}"""
    out = {}
    for g in games:
        clean, gt = devgames.load(g)
        for p in presets:
            for s in seeds:
                m, f = noisy(g, clean, gt, p, s, tags)
                out[(g, p, s)] = (m, gt, f)
    return out


def tune_decode(model, train, grid=GRID):
    """Grid search of the switching costs on the training games (spellF1 mean)."""
    ems = [(m, gt, model.emissions(f)) for m, gt, f in train]
    best, best_p = -1.0, None
    for sw, st, sa in itertools.product(grid["switch"], grid["stop"], grid["start"]):
        p = methods.DecodeParams(beta=1.0, switch=sw, stop=st, start=sa)
        model.decode = p
        s = np.mean([score.spell_f1(gt, model.predict(m, em=em)) for m, gt, em in ems])
        if s > best:
            best, best_p = s, p
    model.decode = best_p
    return best_p, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="out/bench.json")
    ap.add_argument("--train", default="all", choices=("all", "metrica"),
                    help="games the model may learn from; every game is still held out once")
    args = ap.parse_args()

    presets = ["mid"] if args.quick else list(noise.PRESETS)
    grid = {k: v[1::2] for k, v in GRID.items()} if args.quick else GRID
    t0 = time.time()
    devgames.download_all()
    data = prepare(presets)
    pool = devgames.DEV_GAMES if args.train == "all" else devgames.METRICA
    print(f"prepared {len(data)} noisy games in {time.time() - t0:.0f}s")

    rows = []
    for test in devgames.DEV_GAMES:
        train = [v for (g, p, s), v in data.items() if g != test and g in pool]
        v2 = methods.V2().fit([(f, gt, m.runs) for m, gt, f in train])
        p, train_score = tune_decode(v2, train, grid)
        print(f"[{test}] tuned on train: {p}  spellF1 {train_score:.4f}")
        for (g, preset, s), (m, gt, f) in data.items():
            if g != test:
                continue
            for name, pred in (("v0", methods.v0_baseline(m)), ("v1", methods.v1_rules(m, f)),
                               ("v2", v2.predict(m, f))):
                r = score.evaluate(gt, pred)
                rows.append({"game": g, "preset": preset, "method": name,
                             "spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"],
                             "n_pred": r["n_pred"], "n_gt": r["n_gt"]})
                print(f"  {g:9s} {preset:10s} {name}  spellF1 {r['spell_f1']:.4f}  "
                      f"macroF1 {r['macro_f1']:.4f}  spells {r['n_pred']}/{r['n_gt']}")

    print("\nmean over folds and presets")
    for name in ("v0", "v1", "v2"):
        sel = [r for r in rows if r["method"] == name]
        print(f"  {name}  spellF1 {np.mean([r['spell_f1'] for r in sel]):.4f}  "
              f"(min {np.min([r['spell_f1'] for r in sel]):.4f})  "
              f"macroF1 {np.mean([r['macro_f1'] for r in sel]):.4f}")
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
