"""
crossplay.py  —  Round-robin evaluation of saved policies
=========================================================

Plays every checkpoint against every other (and itself), in both seats, and
optionally against scripted opponents. Output: a score matrix with 95%
confidence intervals, Elo ratings, a heatmap, and the raw counts.

    python crossplay.py models_bvr/archive                  # every .zip in it
    python crossplay.py models_bvr/archive/best_*.zip models_bvr/latest.zip
    python crossplay.py models_bvr/archive --episodes 60 --workers 20 \\
                        --scripted shooter adaptive_shooter

Wildcards are expanded here, so they work in Windows cmd too.

How games are played and scored
-------------------------------
* The row policy flies AC1, the column policy AC2. AC2 is a self-play
  opponent: its own radar and track, its own missile guidance, no truth.
* Episode i of every pairing starts from the same geometry (seed + i), so
  differences between cells come from the policies, not from luck of the
  draw. Mirrored pairings (A v B, B v A) replay the same geometries with the
  seats swapped, which cancels any advantage of the AC1 seat.
* Score for the row player: win 1, draw 0.5, loss 0.
    win  = KILL or BANDIT_CRASH
    loss = SHOT_DOWN or CRASH
    draw = MUTUAL_KILL, TIMEOUT, ESCAPE
* Diagonal cells (a policy against itself) should sit near 0.5; the gap is
  the AC1 seat's advantage, also reported as "seat bias".
* Both sides fly the same doctrine (--doctrine). Actions are the policy's
  most likely choice unless --stochastic.
* Elo: Bradley-Terry fit over all games (draws count half), one virtual draw
  per pairing as a prior so a 100% score stays finite; mean rating 1000.
"""

import argparse
import glob
import sys
import json
import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

WIN = ("KILL", "BANDIT_CRASH")
LOSS = ("SHOT_DOWN", "CRASH")
CHUNK = 10                      # episodes per worker task, for load balance
SCRIPTED_PREFIX = "scripted:"


# ── game play (runs in worker processes) ──────────────────────────────
_ENVS = {}


def _reseed(env, seed):
    """Put every generator that shapes an episode back to `seed`: the start
    geometry, the world (missile hand-off ranges), and both radars' noise."""
    gens = [env._rng, env._world._rng] + [o._rng for _, o in sorted(env._sp_observers.items())]
    for off, g in enumerate(gens):
        g.bit_generator.state = np.random.default_rng(seed * 8 + off).bit_generator.state


def _play(task):
    import torch
    torch.set_num_threads(1)
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType
    from bvr_selfplay import FrameStacker, N_STACK, PolicyOpponent, load_policy

    row, col, first, count, seed, doctrine, deterministic, envelope = task
    model_a, priv_a = load_policy(row)

    key = (priv_a, doctrine)
    env = _ENVS.get(key)
    if env is None:
        env = BvrEnv(opponent_type=BvrOpponentType.SELF_PLAY, privileged_critic=priv_a,
                     envelope_table=envelope, doctrine=doctrine, seed=seed)
        _ENVS[key] = env

    if col.startswith(SCRIPTED_PREFIX):
        env._opponent_factory = None
        env._opponent_type = BvrOpponentType[col[len(SCRIPTED_PREFIX):]]
    else:
        model_b, priv_b = load_policy(col)
        env._opponent_type = BvrOpponentType.SELF_PLAY
        env._opponent_factory = lambda e: PolicyOpponent(
            e, model_b, priv_b, label="policy", deterministic=deterministic, doctrine=doctrine)
        env.selfplay_observer(priv_b)       # exists before _reseed, so it is reseeded too

    out = []
    for i in range(first, first + count):
        _reseed(env, seed + i)
        obs, _ = env.reset()
        stack = FrameStacker(N_STACK)
        sobs = stack.reset(obs)
        while True:
            a, _ = model_a.predict(sobs, deterministic=deterministic,
                                   action_masks=env.action_masks())
            obs, _, term, trunc, info = env.step(a)
            sobs = stack.update(obs)
            if term or trunc:
                break
        out.append({"i": i, "outcome": info["terminal_outcome"],
                    "shots_a": int(info.get("shots_fired", 0)),
                    "shots_b": int(env._world.WPN_COUNT - env._world.wpn2),
                    "t": round(float(info.get("flight_time", env._t_sim)), 1)})
    return row, col, out


# ── statistics ────────────────────────────────────────────────────────
def score_of(outcome):
    return 1.0 if outcome in WIN else 0.0 if outcome in LOSS else 0.5


def cell_stats(games):
    s = np.array([score_of(g["outcome"]) for g in games])
    n = len(s)
    w = int(sum(g["outcome"] in WIN for g in games))
    l = int(sum(g["outcome"] in LOSS for g in games))
    mean = float(s.mean()) if n else float("nan")
    # 95% interval on the mean score (t-free normal approximation; n >= 20 in practice)
    half = 1.96 * float(s.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "win": w, "draw": n - w - l, "loss": l, "score": mean, "ci95": half,
            "shots_row": float(np.mean([g["shots_a"] for g in games])) if n else 0.0,
            "shots_col": float(np.mean([g["shots_b"] for g in games])) if n else 0.0}


def fit_elo(players, results, iters=2000):
    """Bradley-Terry by minorise-maximise; results[(a, b)] = list of scores for a."""
    idx = {p: k for k, p in enumerate(players)}
    P = len(players)
    wins = np.zeros((P, P))
    games = np.zeros((P, P))
    for (a, b), sc in results.items():
        if a == b or not sc:
            continue
        i, j = idx[a], idx[b]
        wins[i, j] += sum(sc)
        wins[j, i] += len(sc) - sum(sc)
        games[i, j] += len(sc)
        games[j, i] += len(sc)
    played = games > 0
    wins += 0.5 * played            # one virtual draw per pairing
    games += 1.0 * played
    r = np.ones(P)
    for _ in range(iters):
        denom = (games / (r[:, None] + r[None, :])).sum(axis=1)
        new = np.where(denom > 0, wins.sum(axis=1) / np.maximum(denom, 1e-12), r)
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - r)) < 1e-10:
            r = new
            break
        r = new
    elo = 400.0 * np.log10(r)
    return {p: float(1000.0 + elo[idx[p]] - elo.mean()) for p in players}


# ── driver ────────────────────────────────────────────────────────────
def expand(specs):
    paths = []
    for s in specs:
        if os.path.isdir(s):
            found = sorted(glob.glob(os.path.join(s, "*.zip")), key=os.path.getmtime)
        elif any(c in s for c in "*?["):
            found = sorted(glob.glob(s), key=os.path.getmtime)
        else:
            found = [s if s.endswith(".zip") or not os.path.exists(s + ".zip") else s + ".zip"]
        paths += [os.path.normpath(p) for p in found if p not in paths]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"not found: {', '.join(missing)}")
    return paths


_LABELS = {}


def label(path):
    if path in _LABELS:
        return _LABELS[path]
    return path[len(SCRIPTED_PREFIX):] if path.startswith(SCRIPTED_PREFIX) \
        else os.path.splitext(os.path.basename(path))[0]


def assign_labels(paths):
    """File stem, or parent/stem where two checkpoints share a file name."""
    stems = [label(p) for p in paths]
    for p, st in zip(paths, stems):
        _LABELS[p] = (os.path.basename(os.path.dirname(p)) + "/" + st) if stems.count(st) > 1 else st


def heatmap(rows, cols, cells, elo, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array([[cells[(r, c)]["score"] for c in cols] for r in rows])
    fig_w = max(6.0, 1.05 * len(cols) + 3.2)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fig_w + 3.0, max(3.6, 0.62 * len(rows) + 2.2)),
                                  gridspec_kw={"width_ratios": [fig_w, 3.0]})
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            st = cells[(r, c)]
            ax.text(j, i, f"{st['score']:.2f}\n±{st['ci95']:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black")
    ax.set_xticks(range(len(cols)), [label(c) for c in cols], rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(rows)), [label(r) for r in rows], fontsize=7.5)
    ax.set_xlabel("AC2 (column player)")
    ax.set_ylabel("AC1 (row player)")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="row player's score")
    order = sorted(elo, key=elo.get)
    ax2.barh(range(len(order)), [elo[p] for p in order], color="#3a78c2")
    ax2.set_yticks(range(len(order)), [label(p) for p in order], fontsize=7.5)
    ax2.axvline(1000, color="#888", lw=0.8, ls=":")
    lo = min(elo.values()) - 60
    ax2.set_xlim(lo, max(elo.values()) + 60)
    for k, p in enumerate(order):
        ax2.text(elo[p] + 5, k, f"{elo[p]:.0f}", va="center", fontsize=7.5)
    ax2.set_title("Elo (mean 1000)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="checkpoint files, wildcards or directories")
    ap.add_argument("--episodes", type=int, default=40, help="games per ordered pairing")
    ap.add_argument("--scripted", nargs="*", default=[],
                    help="scripted opponents to add as columns, e.g. shooter adaptive_shooter")
    ap.add_argument("--doctrine", default="balanced",
                    choices=["aggressive", "balanced", "conservative"])
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the most likely")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--envelope-table", default="envelope.npz")
    ap.add_argument("--out", default="crossplay_results")
    args = ap.parse_args()

    from bvr_opponents import BvrOpponentType, CURRICULUM
    models = expand(args.models)
    if not models:
        raise SystemExit("no checkpoints given")
    scripted = []
    for s in args.scripted:
        t = BvrOpponentType[s.upper()]
        if t not in CURRICULUM or t == BvrOpponentType.SELF_PLAY:
            raise SystemExit(f"{s}: not a scripted opponent")
        scripted.append(SCRIPTED_PREFIX + t.name)
    envelope = args.envelope_table if os.path.exists(args.envelope_table) else None
    if envelope is None:
        print(f"[crossplay] WARNING: {args.envelope_table} not found; using the analytic envelope")

    assign_labels(models)
    rows, cols = models, models + scripted
    pairs = [(r, c) for r in rows for c in cols]
    tasks = [(r, c, k, min(CHUNK, args.episodes - k), args.seed, args.doctrine.upper(),
              not args.stochastic, envelope)
             for r, c in pairs for k in range(0, args.episodes, CHUNK)]
    total = len(pairs) * args.episodes
    print(f"[crossplay] {len(models)} checkpoints, {len(scripted)} scripted | {len(pairs)} pairings "
          f"× {args.episodes} = {total} games | {args.workers} workers | doctrine {args.doctrine.upper()}")
    for m in models:
        print(f"             {label(m)}  ({m})")

    games = defaultdict(list)
    t0 = time.time()
    done = 0
    # One line per update when piped (the GUI reads lines); rewrite in place on a terminal.
    tty = sys.stdout.isatty()
    with Pool(args.workers) as pool:
        for r, c, out in pool.imap_unordered(_play, tasks):
            games[(r, c)] += out
            done += len(out)
            el = time.time() - t0
            eta = el / done * (total - done)
            print(f"{chr(13) if tty else ''}[crossplay] {done}/{total} games  {el/60:5.1f} min elapsed, "
                  f"~{eta/60:5.1f} min left", end="" if tty else "\n", flush=True)
    if tty:
        print()

    cells = {k: cell_stats(sorted(v, key=lambda g: g["i"])) for k, v in games.items()}
    results = {k: [score_of(g["outcome"]) for g in v] for k, v in games.items()}
    elo = fit_elo(models + scripted, results)
    # Seat bias: how far AC1 scores above 0.5 on the same matchups played both ways.
    mirrored = [(cells[(a, b)]["score"] + cells[(b, a)]["score"]) / 2 - 0.5
                for a in models for b in models]
    seat_bias = float(np.mean(mirrored))

    os.makedirs(args.out, exist_ok=True)
    w = max(len(label(c)) for c in cols)
    print(f"\nRow player's score (win 1, draw 0.5) — rows fly AC1, columns AC2, {args.episodes} games each")
    print(" " * (w + 2) + "".join(f"{label(c)[:12]:>13s}" for c in cols))
    for r in rows:
        print(f"{label(r):>{w}s}  " + "".join(
            f"{cells[(r, c)]['score']:7.2f}±{cells[(r, c)]['ci95']:.2f}".rjust(13) for c in cols))
    print(f"\nSeat bias (AC1 advantage on mirrored matchups): {seat_bias:+.3f}")
    print("\nElo (mean 1000):")
    for p in sorted(elo, key=elo.get, reverse=True):
        print(f"  {elo[p]:7.0f}  {label(p)}")

    with open(os.path.join(args.out, "crossplay.json"), "w") as f:
        json.dump({"config": vars(args), "players": models + scripted,
                   "rows": [label(r) for r in rows], "cols": [label(c) for c in cols],
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "cells": {f"{label(r)} | {label(c)}": {**cells[(r, c)], "row": r, "col": c}
                             for r, c in pairs},
                   "elo": {label(p): round(v, 1) for p, v in elo.items()},
                   "seat_bias": round(seat_bias, 4),
                   "games": {f"{label(r)} | {label(c)}": games[(r, c)] for r, c in pairs}}, f, indent=1)
    with open(os.path.join(args.out, "crossplay_matrix.csv"), "w") as f:
        f.write("row\\col," + ",".join(label(c) for c in cols) + "\n")
        for r in rows:
            f.write(label(r) + "," + ",".join(f"{cells[(r, c)]['score']:.3f}" for c in cols) + "\n")
    with open(os.path.join(args.out, "crossplay_elo.csv"), "w") as f:
        f.write("player,elo\n")
        for p in sorted(elo, key=elo.get, reverse=True):
            f.write(f"{label(p)},{elo[p]:.1f}\n")
    try:
        heatmap(rows, cols, cells, elo, os.path.join(args.out, "crossplay_heatmap.png"),
                f"Cross-play, doctrine {args.doctrine.upper()}, {args.episodes} games per cell")
        print(f"\n[crossplay] wrote {args.out}/crossplay_heatmap.png, crossplay_matrix.csv, "
              f"crossplay_elo.csv, crossplay.json")
    except ImportError:
        print(f"\n[crossplay] matplotlib not installed: no heatmap; wrote CSV and JSON to {args.out}/")


if __name__ == "__main__":
    main()
