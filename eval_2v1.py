"""
eval_2v1.py  —  2v1 checkpoints against fixed opponents
=======================================================

The 2v1 dashboard's win rate is measured against whatever the curriculum is
flying at the time, so it cannot compare checkpoints. This plays every 2v1
checkpoint against the same fixed reds, from the same starts:

    python eval_2v1.py models_bvr/latest_2v1.zip
    python eval_2v1.py models_bvr/archive/*2v1*.zip --episodes 100 \\
        --scripted shooter adaptive_shooter --red-policies models_bvr/dcs_v11.zip

Red columns: scripted opponents (--scripted, default SHOOTER and
ADAPTIVE_SHOOTER) and 1v1 checkpoints flying red (--red-policies; see
bvr_team.TeamPolicyRed). Wildcards are expanded here, so they work in
Windows cmd too.

How games are played and scored
-------------------------------
* Episode i of every cell starts from the same geometry, red opening rule
  and radar noise (seed + i), so differences between cells come from the
  policies, not from the draw.
* Both blue aircraft fly the checkpoint, on the platforms it was trained on
  (its scenario record); red flies the checkpoint's opponent platform unless
  --red-platform names one.
* Score for blue: 1 for KILL or BANDIT_CRASH, 0.5 for MUTUAL_KILL, TIMEOUT or
  ESCAPE, 0 for SHOT_DOWN or CRASH, with a 95% interval. Also per cell: clean
  kills, mutual kills, losses, timeouts, blue aircraft lost per episode, the
  exchange ratio (red aircraft destroyed / blue aircraft lost, losses floored
  at 1), blue's Pk (red kills / blue missiles) and red's shots per episode.
* Actions are each policy's most likely choice unless --stochastic. Blue
  draws its doctrine per episode unless --doctrine names one; a red policy
  draws its own.
"""

import argparse
import csv
import glob
import math
import multiprocessing as mp
import os

import numpy as np

WIN = ("KILL", "BANDIT_CRASH")
LOSS = ("SHOT_DOWN", "CRASH")
OUTCOMES = ["KILL", "MUTUAL_KILL", "SHOT_DOWN", "CRASH", "BANDIT_CRASH", "ESCAPE", "TIMEOUT"]
CHUNK = 10                       # episodes per worker task
SCRIPTED = "scripted:"


def score_of(outcome):
    return 1.0 if outcome in WIN else (0.0 if outcome in LOSS else 0.5)


def expand(specs):
    out = []
    for s in specs:
        hits = sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s]
        if os.path.isdir(s):
            hits = sorted(glob.glob(os.path.join(s, "*.zip")))
        for h in hits:
            if not os.path.exists(h) and os.path.exists(h + ".zip"):
                h += ".zip"
            if not os.path.exists(h):
                raise SystemExit(f"not found: {s}")
            if h not in out:
                out.append(h)
    return out


def _reseed(env, seed):
    """Every generator that shapes an episode back to `seed`: the start and
    red's opening rule, the world, the blue radars and red's radars."""
    gens = [env._rng, env._world._rng] + [o._rng for o in env._obs] + \
           [r._rng for r in env._red_obs]
    for j, g in enumerate(gens):
        g.bit_generator.state = np.random.default_rng([seed, j]).bit_generator.state


_ENVS = {}


def _play(task):
    import torch
    torch.set_num_threads(1)
    from gymnasium import spaces
    from bvr_compat import load_model
    from bvr_library import scenario_of
    from bvr_opponents import BvrOpponentType
    from bvr_selfplay import FrameStacker, N_STACK, load_policy
    from bvr_team import TeamBvrEnv, make_policy_red

    ckpt, col, first, count, seed, doctrine, deterministic, red_platform, max_steps = task
    model, _ = load_policy(ckpt)
    privileged = isinstance(model.observation_space, spaces.Dict)
    sc = scenario_of(model)
    blue = (sc["platform"], sc.get("wingman_platform") or sc["platform"])
    red = red_platform or sc["opponent_platform"]
    key = (ckpt, col, doctrine, red)
    env = _ENVS.get(key)
    if env is None:
        scripted = col.startswith(SCRIPTED)
        opp = BvrOpponentType[col[len(SCRIPTED):]] if scripted else BvrOpponentType.ADAPTIVE_SHOOTER
        env = TeamBvrEnv(opponent_type=opp, seed=seed, privileged_critic=privileged,
                         doctrine=doctrine, blue_platforms=blue, red_platform=red)
        if max_steps:
            env.MAX_STEPS = int(max_steps)
            for o in env._obs:
                o.MAX_STEPS = int(max_steps)
        if not scripted:
            env._opponent_factory = lambda e, col=col: make_policy_red(
                e, col, deterministic=deterministic)
        _ENVS[key] = env

    out = []
    for i in range(first, first + count):
        _reseed(env, seed + i)
        obs, _ = env.reset()
        stackers = [FrameStacker(N_STACK) for _ in obs]
        sobs = [st.reset(o) for st, o in zip(stackers, obs)]
        while True:
            masks = env.action_masks()
            acts = [model.predict(sobs[k], deterministic=deterministic, action_masks=masks[k])[0]
                    for k in range(env.n_agents)]
            obs, _, term, trunc, infos = env.step(acts)
            sobs = [st.update(o) for st, o in zip(stackers, obs)]
            if term or trunc:
                break
        fin = infos[0]
        out.append({"episode": i, "outcome": fin["terminal_outcome"],
                    "blue_losses": int(fin.get("blue_losses", 0)),
                    "blue_shots": int(fin.get("shots_fired", 0)),
                    "red_shots": int(env._red_wpn0 - env._world.wpn[env.RED - 1]),
                    "red_open_rule": fin.get("red_open_rule", ""),
                    "t_end": round(float(env._t_sim), 1)})
    return ckpt, col, out


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def cell_stats(rows):
    n = len(rows)
    sc = [score_of(r["outcome"]) for r in rows]
    mean = float(np.mean(sc)) if n else 0.0
    half = 1.96 * float(np.std(sc, ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    cnt = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
    red_down = cnt["KILL"] + cnt["MUTUAL_KILL"] + cnt["BANDIT_CRASH"]
    lost = sum(r["blue_losses"] for r in rows)
    shots = sum(r["blue_shots"] for r in rows)
    return {"n": n, "score": mean, "ci": half, "counts": cnt,
            "clean": cnt["KILL"] / n if n else 0.0,
            "lost_per_ep": lost / n if n else 0.0,
            "exchange": (cnt["KILL"] + cnt["MUTUAL_KILL"]) / max(lost, 1),
            "pk": (cnt["KILL"] + cnt["MUTUAL_KILL"]) / shots if shots else 0.0,
            "red_shots": sum(r["red_shots"] for r in rows) / n if n else 0.0,
            "red_down": red_down}


def label(path):
    if path.startswith(SCRIPTED):
        return path[len(SCRIPTED):]
    b = os.path.basename(path)
    return b[:-4] if b.endswith(".zip") else b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="2v1 checkpoints (.zip, folders or wildcards)")
    ap.add_argument("--scripted", nargs="*", default=["shooter", "adaptive_shooter"],
                    help="scripted reds (default: shooter adaptive_shooter; none: --scripted)")
    ap.add_argument("--red-policies", nargs="*", default=[],
                    help="1v1 checkpoints that fly red")
    ap.add_argument("--episodes", type=int, default=60, help="episodes per cell (default 60)")
    ap.add_argument("--seed", type=int, default=0, help="first episode's seed")
    ap.add_argument("--doctrine", default="MIXED",
                    help="blue's doctrine: MIXED (drawn per episode, default) or one of them")
    ap.add_argument("--red-platform", help="red's platform (default: each checkpoint's opponent)")
    ap.add_argument("--max-steps", type=int, help="episode length in decisions")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the most likely")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel worker processes (default: CPU cores - 1)")
    ap.add_argument("--csv", help="write one row per episode here")
    args = ap.parse_args()

    from bvr_selfplay import load_policy
    from bvr_library import scenario_of
    from bvr_opponents import BvrOpponentType
    from bvr_team import check_red_policy
    ckpts = expand(args.checkpoints)
    for c in ckpts:
        sc = scenario_of(load_policy(c)[0])
        if sc.get("format") != "2v1":
            print(f"note: {label(c)} was trained as {sc.get('format')}; flying it in 2v1")
    cols = []
    for s in args.scripted:
        try:
            cols.append(SCRIPTED + BvrOpponentType[s.upper()].name)
        except KeyError:
            raise SystemExit(f"unknown scripted opponent {s!r}")
    for p in expand(args.red_policies):
        red = args.red_platform or scenario_of(load_policy(ckpts[0])[0])["opponent_platform"]
        for n in check_red_policy(p, red):
            print(f"note: {n}")
        cols.append(p)
    if not cols:
        raise SystemExit("no red to play against: give --scripted and/or --red-policies")

    tasks = [(c, col, first, min(CHUNK, args.episodes - first), args.seed, args.doctrine.upper(),
              not args.stochastic, args.red_platform, args.max_steps)
             for c in ckpts for col in cols for first in range(0, args.episodes, CHUNK)]
    workers = max(1, min(args.workers, len(tasks)))
    print(f"{len(ckpts)} checkpoint(s) x {len(cols)} red(s) x {args.episodes} episodes "
          f"on {workers} worker process(es) ...", flush=True)
    rows = {(c, col): [] for c in ckpts for col in cols}
    if workers == 1:
        results = map(_play, tasks)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers)
        results = pool.imap_unordered(_play, tasks)
    for j, (c, col, out) in enumerate(results, start=1):
        rows[(c, col)].extend(out)
        if j % 10 == 0 or j == len(tasks):
            print(f"  {j}/{len(tasks)} tasks", flush=True)
    if pool is not None:
        pool.close(); pool.join()

    stats = {k: cell_stats(sorted(v, key=lambda r: r["episode"])) for k, v in rows.items()}
    w = max(12, *(len(label(c)) for c in ckpts))
    cw = max(16, *(len(label(col)) + 2 for col in cols))
    print("\nScore for blue (win 1, mutual kill / timeout / escape 0.5, loss 0) with a 95% interval")
    print(" " * w + "".join(f"{label(col):>{cw}}" for col in cols))
    for c in ckpts:
        print(f"{label(c):<{w}}" + "".join(
            f"{stats[(c, col)]['score']:>{cw - 7}.2f} ±{stats[(c, col)]['ci']:.2f}" for col in cols))
    print()
    for c in ckpts:
        for col in cols:
            s = stats[(c, col)]
            n, k = s["n"], s["counts"]
            lo, hi = _wilson(k["KILL"], n)
            print(f"{label(c)} v {label(col)}  ({n} episodes)")
            print("  " + "  ".join(f"{o} {100 * k[o] / n:.0f}%" for o in OUTCOMES if k[o]))
            print(f"  clean kills {s['clean']:.0%} (95% {lo:.0%}-{hi:.0%})   blue lost/ep "
                  f"{s['lost_per_ep']:.2f}   exchange {s['exchange']:.2f}   blue Pk {s['pk']:.2f}   "
                  f"red shots/ep {s['red_shots']:.1f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["checkpoint", "red", "episode", "outcome",
                                               "blue_losses", "blue_shots", "red_shots",
                                               "red_open_rule", "t_end"])
            wr.writeheader()
            for (c, col), rs in rows.items():
                for r in sorted(rs, key=lambda r: r["episode"]):
                    wr.writerow({"checkpoint": label(c), "red": label(col), **r})
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
