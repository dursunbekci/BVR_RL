"""
team_roles.py  —  2v1: how does the team do when red opens on the rear aircraft?
================================================================================

Red opens on the nearer blue aircraft, and the start formations put the lead
in front almost every time, so in training red's first missile nearly always
goes at the front aircraft while the rear one is free to close and shoot.
This flies a 2v1 checkpoint twice over the same starts:

  natural   red opens on the nearer aircraft (training mixes this with
            other openings: TeamBvrEnv.OPEN_RULES)
  farther   red opens on the farther (rear) aircraft and holds that target
            until its first launch; after that it picks targets as usual

and reports, for each, the outcomes and what happened to the aircraft red
shot at first ("targeted") and to the other ("free"). Episode i uses the
same start in both runs, so differences come from red's choice of target,
not from different starts.

    python team_roles.py models_bvr/latest_2v1.zip
    python team_roles.py models_bvr/archive/best_SHOOTER_0.700 --opponent SHOOTER --episodes 300
    python team_roles.py ckpt.zip --red-opens natural --csv roles.csv

The policy flies deterministically, as in the GUI's WATCH page. Writes one
row per episode to --csv when given.
"""

import argparse
import csv
import math
import multiprocessing as mp
import os

import numpy as np

from bvr_compat import load_model
from bvr_library import scenario_of
from bvr_opponents import BvrOpponentType
from bvr_selfplay import FrameStacker, N_STACK
from bvr_team import TeamBvrEnv

OUTCOMES = ["KILL", "MUTUAL_KILL", "SHOT_DOWN", "CRASH", "BANDIT_CRASH", "ESCAPE", "TIMEOUT"]


class NearTargetTeamEnv(TeamBvrEnv):
    """Red always opens on the nearer blue aircraft (the pre-SIM_REV 16 rule)."""
    OPEN_RULES = (("nearer", 1.0),)


class FarTargetTeamEnv(TeamBvrEnv):
    """Red opens on the blue aircraft farther from it at the start, and keeps
    that target until it has launched once (or the target is lost)."""
    OPEN_RULES = (("nearer", 1.0),)

    def reset(self, seed=None, options=None):
        self._forced_tgt = None
        return super().reset(seed=seed, options=options)

    def _retarget(self):
        w = self._world
        if self._forced_tgt is None:
            red = w.pos(self.RED)
            self._forced_tgt = max((1, 2), key=lambda i: float(np.linalg.norm(w.pos(i) - red)))
        red_launched = any(m.owner == self.RED for m in w.missiles)
        if not red_launched and self._obs[self._forced_tgt - 1]._alive:
            self._red_tgt = self._forced_tgt
            return
        super()._retarget()


def _dist(w, i, j):
    return float(np.linalg.norm(w.pos(i) - w.pos(j)))


def run_episode(env, model, seed):
    """One episode; returns its facts. Blue aircraft are numbered 1 (lead) and 2."""
    # Same start (and the same noise until the runs diverge) in every run and
    # on any worker: reseed every generator in place, since the world shares
    # its generator with the aircraft models.
    gens = [env._rng, env._world._rng] + [o._rng for o in env._obs]
    for j, g in enumerate(gens):
        g.bit_generator.state = np.random.default_rng([seed, j]).bit_generator.state
    obs, info = env.reset()
    w, RED = env._world, env.RED
    ic = info["ic"]
    d0 = {i: _dist(w, i, RED) for i in (1, 2)}
    opened_on = env._red_tgt
    stackers = [FrameStacker(N_STACK) for _ in obs]
    sobs = [st.reset(o) for st, o in zip(stackers, obs)]
    first_red = None                               # (target, t, nearer at launch?)
    first_blue = None                              # (shooter, t)
    lost_t = {}
    while True:
        masks = env.action_masks()
        acts = [model.predict(sobs[k], deterministic=True, action_masks=masks[k])[0]
                for k in range(env.n_agents)]
        alive_before = [o._alive for o in env._obs]
        obs, _, term, trunc, infos = env.step(acts)
        sobs = [st.update(o) for st, o in zip(stackers, obs)]
        if first_red is None:
            reds = [m for m in w.missiles if m.owner == RED]
            if reds:
                m = min(reds, key=lambda m: m.t_launch)
                other = 3 - m.target
                nearer = (not env._obs[other - 1]._alive and not alive_before[other - 1]) or \
                    _dist(w, m.target, RED) <= _dist(w, other, RED)
                first_red = (int(m.target), float(m.t_launch), bool(nearer))
        if first_blue is None:
            blues = [m for m in w.missiles if m.owner in (1, 2)]
            if blues:
                m = min(blues, key=lambda m: m.t_launch)
                first_blue = (int(m.owner), float(m.t_launch))
        for k, o in enumerate(env._obs, start=1):
            if alive_before[k - 1] and not o._alive:
                lost_t[k] = env._t_sim
        if term or trunc:
            break
    fin = infos[0]
    tgt = first_red[0] if first_red else opened_on
    free = 3 - tgt
    lost = [i + 1 for i, _ in env._lost]
    killer = env._killer if env._red_dead == "KILL" else None
    return {
        "seed": seed, "outcome": fin["terminal_outcome"],
        "scenario": ic.get("scenario", ""), "mirrored": bool(ic.get("mirrored", False)),
        "formation": ic.get("formation", ""),
        "front_at_start": min((1, 2), key=d0.get),
        "range_lead_km": round(d0[1] / 1000, 1), "range_wing_km": round(d0[2] / 1000, 1),
        "red_opened_on": opened_on,
        "red_first_shot_at": first_red[0] if first_red else 0,
        "red_first_shot_t": round(first_red[1], 1) if first_red else "",
        "red_shot_nearer": int(first_red[2]) if first_red else "",
        "blue_first_shot_by": first_blue[0] if first_blue else 0,
        "blue_first_shot_t": round(first_blue[1], 1) if first_blue else "",
        "targeted": tgt, "free": free,
        "targeted_lost": int(tgt in lost), "free_lost": int(free in lost),
        "first_lost": lost[0] if lost else 0,
        "killer": killer or 0,
        "kill_by": ("targeted" if killer == tgt else "free") if killer else "",
        "shots": fin.get("shots_fired", 0),
        "t_end": round(env._t_sim, 1),
    }


_W = {}


def _make_env(mode, args, privileged, blue, red):
    cls = NearTargetTeamEnv if mode == "natural" else FarTargetTeamEnv
    env = cls(opponent_type=BvrOpponentType[args.opponent], seed=args.seed,
              privileged_critic=privileged, doctrine=args.doctrine,
              blue_platforms=blue, red_platform=red)
    if args.max_steps:
        env.MAX_STEPS = args.max_steps
        for o in env._obs:
            o.MAX_STEPS = args.max_steps
    return env


def _init_worker(args, privileged, blue, red):
    _W["model"] = load_model(args.checkpoint, env=None, log=lambda m: None)
    _W["setup"] = (args, privileged, blue, red)
    _W["envs"] = {}


def _work(job):
    mode, seed = job
    envs = _W["envs"]
    if mode not in envs:
        envs[mode] = _make_env(mode, *_W["setup"])
    return dict(run_episode(envs[mode], _W["model"], seed), red_opens=mode)


def _pct(k, n):
    return f"{100 * k / n:5.1f}%" if n else "    -"


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def report(name, rows):
    n = len(rows)
    print(f"\n── red opens: {name}  ({n} episodes) " + "─" * 30)
    fired = [r for r in rows if r["red_first_shot_at"]]
    on_front = sum(r["red_shot_nearer"] == 1 for r in fired)
    on_lead = sum(r["red_first_shot_at"] == 1 for r in fired)
    print(f"red fired in {len(fired)}/{n}; its first missile went at the nearer aircraft "
          f"{_pct(on_front, len(fired))}, at the lead {_pct(on_lead, len(fired))}")
    print("outcomes:  " + "  ".join(f"{o} {_pct(sum(r['outcome'] == o for r in rows), n)}"
                                     for o in OUTCOMES if any(r["outcome"] == o for r in rows)))
    k = sum(r["outcome"] == "KILL" for r in rows)
    lo, hi = _wilson(k, n)
    print(f"clean kills {k}/{n} = {k / n:.1%}  (95% interval {lo:.0%}-{hi:.0%})")
    print(f"aircraft red shot at first: lost {_pct(sum(r['targeted_lost'] for r in rows), n)}"
          f"   the other aircraft: lost {_pct(sum(r['free_lost'] for r in rows), n)}")
    kills = [r for r in rows if r["kill_by"]]
    print(f"red killed by the targeted aircraft {_pct(sum(r['kill_by'] == 'targeted' for r in kills), len(kills))}"
          f", by the free one {_pct(sum(r['kill_by'] == 'free' for r in kills), len(kills))}"
          f"  (of {len(kills)} kills)")
    bf = [r for r in rows if r["blue_first_shot_by"]]
    print(f"blue's first missile fired by the free aircraft "
          f"{_pct(sum(r['blue_first_shot_by'] == r['free'] for r in bf), len(bf))} (of {len(bf)})")
    t_red = [r["red_first_shot_t"] for r in fired]
    t_blue = [r["blue_first_shot_t"] for r in bf]
    if t_red and t_blue:
        print(f"median first launch: red {np.median(t_red):.0f} s, blue {np.median(t_blue):.0f} s")
    return k, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="a 2v1 checkpoint (.zip)")
    ap.add_argument("--opponent", default="SHOOTER",
                    choices=[t.name for t in BvrOpponentType if t != BvrOpponentType.SELF_PLAY])
    ap.add_argument("--episodes", type=int, default=200, help="episodes per run (default 200)")
    ap.add_argument("--red-opens", default="both", choices=["both", "natural", "farther"])
    ap.add_argument("--doctrine", default="BALANCED")
    ap.add_argument("--opp-platform", help="red's platform (default: the checkpoint's)")
    ap.add_argument("--max-steps", type=int, help=f"episode length (default {TeamBvrEnv.MAX_STEPS})")
    ap.add_argument("--seed", type=int, default=0, help="first episode's seed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel worker processes (default: CPU cores - 1)")
    ap.add_argument("--csv", help="write one row per episode here")
    args = ap.parse_args()

    from gymnasium import spaces
    model = load_model(args.checkpoint, env=None, log=print)
    privileged = isinstance(model.observation_space, spaces.Dict)
    scen = scenario_of(model)
    blue = (scen["platform"], scen.get("wingman_platform") or scen["platform"])
    red = args.opp_platform or scen["opponent_platform"]
    if scen.get("format", "2v1") != "2v1":
        print(f"note: this checkpoint was trained as {scen.get('format')}; flying it in 2v1")
    print(f"{blue[0]} + {blue[1]} v {red} ({args.opponent}), {args.doctrine} doctrine, "
          f"{args.episodes} episodes per run")

    modes = ["natural", "farther"] if args.red_opens == "both" else [args.red_opens]
    jobs = [(m, args.seed + i) for m in modes for i in range(args.episodes)]
    setup = (args, privileged, blue, red)
    workers = max(1, min(args.workers, len(jobs)))
    print(f"{len(jobs)} episodes on {workers} worker process(es) ...", flush=True)
    rows_by = {m: [] for m in modes}
    if workers == 1:
        _init_worker(*setup)
        results = map(_work, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=setup)
        results = pool.imap_unordered(_work, jobs)
    for j, r in enumerate(results, start=1):
        rows_by[r["red_opens"]].append(r)
        if j % 25 == 0 or j == len(jobs):
            print(f"  {j}/{len(jobs)}", flush=True)
    if pool is not None:
        pool.close(); pool.join()
    all_rows, summary = [], {}
    for mode in modes:
        rows = sorted(rows_by[mode], key=lambda r: r["seed"])
        summary[mode] = report(mode, rows)
        all_rows += rows

    if len(summary) == 2:
        (k1, n1), (k2, n2) = summary["natural"], summary["farther"]
        p1, p2 = k1 / n1, k2 / n2
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        print(f"\nclean kills: natural {p1:.1%}, red opens on the rear aircraft {p2:.1%}; "
              f"difference {100 * (p2 - p1):+.1f} points (±{196 * se:.1f} at 95%)")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["red_opens"] + [k for k in all_rows[0] if k != "red_opens"])
            wr.writeheader()
            wr.writerows(all_rows)
        print(f"written: {args.csv}")


if __name__ == "__main__":
    main()
