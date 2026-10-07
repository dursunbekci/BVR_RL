"""
dcs_obs_check.py  —  do DCS engagements look like training to a policy?
=========================================================================

Replays DCS recordings (dcs/bvr_logger.lua) through the environment's
observation code and compares each of the policy's inputs with the same
input in simulator episodes. An input DCS drives outside the range the
simulator produces, or into the normalisation clip, is one a trained policy
has never seen: fix the bridge (a unit, sign or axis mistake) or accept it
as a real difference between DCS and the simulator before flying the
policy in DCS.

    python dcs_obs_check.py bvr_rl_43200_804789.jsonl
    python dcs_obs_check.py a.jsonl b.jsonl --blue Viper-1 --red Flanker-1 \\
        --platform F-16C --opp-platform F-16C --out dcs_check

The simulator reference is flown by a scripted policy that wanders
(random heading, altitude and speed choices held for 5-15 s) and fires
inside 60-90% of R-max. It shows the simulator's ranges, not a trained
policy's habits. Writes, in --out:
    summary.csv     one row per input: DCS and simulator percentiles, flags
    dcs_inputs.csv  every replayed decision step: time, inferred action, inputs
    report.txt      the table printed below
"""

import argparse
import csv
import os

import numpy as np

from bvr_env import (BvrEnv, OBS_LABELS, OBS_PHYS_LOW, OBS_PHYS_HIGH, HDG_OFFSETS_DEG,
                     ALT_DELTAS_M)
from bvr_library import DEFAULT_PLATFORM
from bvr_opponents import BvrOpponentType
from dcs_world import DcsRecording, DcsReplayEnv, capture_raw_obs

# An input is flagged when more than this share of DCS steps is outside the
# simulator's 0.5-99.5 percentile range, or when this much more of them than
# of the simulator's is beyond the normalisation limits (and so clipped).
OUTSIDE_FLAG = 0.20
CLIP_FLAG = 0.05


def replay(path, args):
    rec = DcsRecording(path)
    env = DcsReplayEnv(rec, blue=args.blue, red=args.red, platform=args.platform,
                       opponent_platform=args.opp_platform, doctrine=args.doctrine,
                       max_steps=args.max_steps)
    raw = capture_raw_obs(env)
    env.reset()
    rows, info = [], {}
    while True:
        action = env.infer_action()
        _, _, term, trunc, info = env.step(action)
        rows.append((round(env._t_sim, 2), action, raw["obs"].copy()))
        if term or trunc:
            break
    return env._world.names, rows, info.get("terminal_outcome")


class Wanderer:
    """Scripted reference: random choices held for a while, shots inside 60-90% of R-max."""

    def __init__(self, rng):
        self.rng, self.until, self.a = rng, -1, [0, 2, 2]
        self.frac = 0.75

    def __call__(self, env):
        r = self.rng
        if env._step_num >= self.until:
            self.until = env._step_num + int(r.integers(5, 16))
            hdg = int(r.choice(len(HDG_OFFSETS_DEG), p=_hdg_weights()))
            self.a = [hdg, int(r.integers(len(ALT_DELTAS_M))),
                      int(r.integers(len(env._plat.speed_cmds)))]
            self.frac = float(r.uniform(0.6, 0.9))
        fire = 0
        if env._can_fire():
            est = env._est()
            rmax, _ = env._own_envelope(est)
            fire = int(env._est_range(est) <= self.frac * rmax)
        return self.a + [fire]


def _hdg_weights():
    # Mostly toward the bandit, sometimes cranking or turning away.
    w = np.array([6.0 if o == 0 else (2.0 if abs(o) <= 50 else 1.0) for o in HDG_OFFSETS_DEG])
    return w / w.sum()


def reference(args):
    rows = []
    opp = BvrOpponentType[args.ref_opponent]
    for i in range(args.ref_episodes):
        env = BvrEnv(opponent_type=opp, seed=1000 + i, platform=args.platform,
                     opponent_platform=args.opp_platform, doctrine=args.doctrine)
        if args.max_steps is not None:
            env.MAX_STEPS = args.max_steps
        raw = capture_raw_obs(env)
        pol = Wanderer(np.random.default_rng(i))
        env.reset()
        while True:
            _, _, term, trunc, _ = env.step(pol(env))
            rows.append(raw["obs"].copy())
            if term or trunc:
                break
    return np.array(rows)


def compare(dcs, sim):
    out = []
    for j, lab in enumerate(OBS_LABELS):
        d, s = dcs[:, j], sim[:, j]
        lo, hi = np.percentile(s, [0.5, 99.5])
        span = max(hi - lo, 1e-9)
        outside = float(np.mean((d < lo - 1e-3 * span) | (d > hi + 1e-3 * span)))
        beyond = lambda x: float(np.mean((x < OBS_PHYS_LOW[j]) | (x > OBS_PHYS_HIGH[j])))
        clip, clip_sim = beyond(d), beyond(s)
        const = np.ptp(d) < 1e-9 and np.ptp(s) < 1e-9
        flag = "" if const else ("CLIP" if clip - clip_sim > CLIP_FLAG else
                                 ("OUTSIDE" if outside > OUTSIDE_FLAG else ""))
        out.append({"input": lab, "flag": flag,
                    "dcs_p1": np.percentile(d, 1), "dcs_p50": np.median(d),
                    "dcs_p99": np.percentile(d, 99),
                    "sim_p0.5": lo, "sim_p50": np.median(s), "sim_p99.5": hi,
                    "outside": outside, "clipped": clip, "clipped_sim": clip_sim,
                    "constant": const})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+", help="bvr_logger.lua .jsonl files")
    ap.add_argument("--blue", help="blue aircraft to replay (default: the first blue one)")
    ap.add_argument("--red", help="red aircraft (default: the first red one)")
    ap.add_argument("--platform", default=DEFAULT_PLATFORM, help="library platform for blue")
    ap.add_argument("--opp-platform", default=DEFAULT_PLATFORM, help="library platform for red")
    ap.add_argument("--doctrine", default="AGGRESSIVE",
                    help="doctrine whose shot cost the policy is told (default AGGRESSIVE)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help=f"episode length the policy was trained with (default {BvrEnv.MAX_STEPS})")
    ap.add_argument("--ref-episodes", type=int, default=20, help="simulator episodes for the reference")
    ap.add_argument("--ref-opponent", default="SHOOTER",
                    choices=[t.name for t in BvrOpponentType if t != BvrOpponentType.SELF_PLAY])
    ap.add_argument("--out", default="dcs_check", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_rows = []
    with open(os.path.join(args.out, "dcs_inputs.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["recording", "blue", "red", "t", "hdg", "alt", "spd", "fire", *OBS_LABELS])
        for path in args.recordings:
            names, rows, outcome = replay(path, args)
            print(f"{os.path.basename(path)}: {names[1]} v {names[2]}, "
                  f"{len(rows)} decisions, outcome {outcome}")
            for t, a, o in rows:
                wr.writerow([os.path.basename(path), names[1], names[2], t, *a,
                             *[f"{x:.6g}" for x in o]])
                all_rows.append(o)
    dcs = np.array(all_rows)

    print(f"simulator reference: {args.ref_episodes} episodes v {args.ref_opponent} ...")
    sim = reference(args)
    res = compare(dcs, sim)

    lines = [f"{len(dcs)} DCS decisions, {len(sim)} simulator decisions",
             f"flag OUTSIDE: >{OUTSIDE_FLAG:.0%} of DCS steps outside the simulator's "
             f"0.5-99.5 percentile range;  CLIP: >{CLIP_FLAG:.0%} more DCS steps than "
             f"simulator steps beyond the normalisation limits", "",
             f"{'input':22s} {'flag':8s} {'DCS p1':>10s} {'p50':>10s} {'p99':>10s}   "
             f"{'sim p0.5':>10s} {'p50':>10s} {'p99.5':>10s}  {'outside':>7s} {'clip DCS/sim':>12s}"]
    order = sorted(res, key=lambda r: (r["flag"] == "", r["constant"]))
    for r in order:
        if r["constant"]:
            continue
        lines.append(f"{r['input']:22s} {r['flag']:8s} {r['dcs_p1']:10.4g} {r['dcs_p50']:10.4g} "
                     f"{r['dcs_p99']:10.4g}   {r['sim_p0.5']:10.4g} {r['sim_p50']:10.4g} "
                     f"{r['sim_p99.5']:10.4g}  {r['outside']:7.0%} {r['clipped']:6.0%}/{r['clipped_sim']:<5.0%}")
    const = [r["input"] for r in res if r["constant"]]
    if const:
        lines += ["", "constant in both (not compared): " + ", ".join(const)]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(res[0]))
        wr.writeheader()
        wr.writerows(res)
    print(f"\nwritten to {args.out}/: report.txt, summary.csv, dcs_inputs.csv")


if __name__ == "__main__":
    main()
