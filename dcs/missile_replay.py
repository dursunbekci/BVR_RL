"""
missile_replay.py  —  does the simulator's missile hit where DCS's did?
=======================================================================

Each missile in dcs_live.py recordings is flown again with a library missile:
fired from the recorded launch (shooter position and velocity), guided on the
target's recorded flight path, which is what the target really flew in DCS
(the policy's jet, flown by the stick pilot, or the DCS AI). The model's
outcome is set beside DCS's, with the DCS missile's own closest approach to
its target, so a gap between the two worlds shows up shot by shot:

  DCS hit, model miss    the model's missile is weaker than DCS's (energy,
                         guidance or fuze) against the same evasion
  DCS miss, model hit    the reverse (or chaff, which the model does not have)

The model is guided on the target's true position all the way (perfect
support), so a model miss is the missile's own: energy or manoeuvre.

    python dcs\\missile_replay.py dcs_runs\\live_*.jsonl
    python dcs\\missile_replay.py dcs_runs\\live_*.jsonl --missile AIM-120C-DCS --json out.json
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from bvr_library import missile_config                  # noqa: E402
from dcs_world import DcsRecording                       # noqa: E402
from missile_fit import Shot, fly                        # noqa: E402

HANDOFFS = (8_000.0, 12_000.0, 16_000.0)                 # the model's seeker hand-off is drawn from 8-16 km
EXTRA_S = 20.0          # the model flies this long past the DCS missile's end, the target going on at its last
                        # velocity: a model a second slower than DCS's would otherwise read "still flying"


def load(paths, min_samples=10):
    """[(shot, dcs closest approach m, time of it s)] for every missile with a shooter and a target in the
    recording; a missile DCS recorded with no target is taken to be aimed at the other aircraft of the pair."""
    out, seen = [], set()
    for p in paths:
        rec = DcsRecording(p)
        name = os.path.basename(p).split("-", 1)[-1] if "-" in os.path.basename(p)[:9] else os.path.basename(p)
        aircraft = [u for u in rec.units if u not in rec.weapons]
        for d in sorted(rec.weapons.values(), key=lambda d: d["t_shot"] or 0):
            if d["t_shot"] is None or len(d["t"]) < min_samples or d["shooter"] not in rec.units:
                continue
            if d["target"] not in rec.units:
                others = [u for u in aircraft if u != d["shooter"]]
                if len(others) != 1:
                    continue
                d = dict(d, target=others[0])
            s = Shot(name, rec, d)
            n = int(EXTRA_S / 0.02)
            lp, lv = s.tgt_pos[-1], s.tgt_vel[-1]
            s.tgt_pos = np.vstack([s.tgt_pos, lp + np.outer(0.02 * np.arange(1, n + 1), lv)])
            s.tgt_vel = np.vstack([s.tgt_vel, np.repeat(lv[None, :], n, axis=0)])
            key = (s.shooter, round(s.t0 - rec.t_start), round(s.range0 / 200))
            if key in seen:
                continue
            seen.add(key)
            # DCS's own closest approach: missile samples against the target at the same times
            ts = rec.units[s.target]["t"]
            k = (d["t"] >= ts[0]) & (d["t"] <= ts[-1])
            if k.sum():
                tp = np.array([rec.state(s.target, float(t))["pos"] for t in d["t"][k]])
                dist = np.linalg.norm(d["pos"][k] - tp, axis=1)
                i = int(np.argmin(dist))
                ca, t_ca = float(dist[i]), float(d["t"][k][i] - s.t0)
            else:
                ca, t_ca = float("nan"), float("nan")
            out.append((s, ca, t_ca))
    return out


def replay(shot, cfg):
    """The model's outcome at each hand-off range: [(outcome, closest approach m, time s)]."""
    res = []
    for h in HANDOFFS:
        T, V, Z, R, outcome = fly(shot, cfg, handoff=h)
        i = int(np.argmin(R)) if len(R) else 0
        res.append((outcome, float(R[i]) if len(R) else float("nan"), float(T[i]) if len(T) else float("nan"),
                    float(V[i]) if len(V) else float("nan")))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--missile", default="AIM-120C-DCS", help="library missile to fly (default AIM-120C-DCS)")
    ap.add_argument("--json", help="write every shot's numbers here")
    args = ap.parse_args(argv)
    paths = [p for pat in args.recordings for p in (glob.glob(pat) or [pat])]
    cfg = missile_config(args.missile)
    rows = []
    for s, ca, t_ca in load(paths):
        r = replay(s, cfg)
        hits = sum(o.startswith("hit") for o, *_ in r)
        rows.append({"rec": s.name, "id": s.id, "shooter": s.shooter, "target": s.target,
                     "range_km": round(s.range0 / 1000, 1), "dcs_hit": bool(s.hit),
                     "dcs_closest_m": round(ca, 1), "dcs_t_closest": round(t_ca, 1),
                     "model_hits": hits, "model": [{"outcome": o, "closest_m": round(c, 1), "t": round(t, 1),
                                                     "speed": round(v)} for o, c, t, v in r]})
    if not rows:
        raise SystemExit("no missiles with a shooter and a target in those recordings")
    print(f"{len(rows)} missiles from {len(paths)} recordings; model {args.missile}, guided on the target's "
          f"recorded path, seeker hand-off {', '.join(f'{h / 1000:.0f}' for h in HANDOFFS)} km\n")
    print(f"{'recording':34s} {'id':>3s} {'shooter':8s} {'km':>5s}  {'DCS':9s} {'DCS closest':>11s}  model")
    for x in rows:
        m = x["model"][1]
        print(f"{x['rec'][:34]:34s} {x['id']:3d} {x['shooter']:8s} {x['range_km']:5.1f}  "
              f"{'HIT' if x['dcs_hit'] else 'miss':9s} {x['dcs_closest_m']:9.0f} m  "
              f"{x['model_hits']}/{len(HANDOFFS)} hit; at 12 km hand-off: {m['outcome']}, closest {m['closest_m']:.0f} m "
              f"at {m['t']:.0f} s, {m['speed']} m/s")
    print()
    for who in sorted({x["shooter"] for x in rows}):
        b = [x for x in rows if x["shooter"] == who]
        dh = sum(x["dcs_hit"] for x in b)
        mh = sum(x["model_hits"] for x in b) / len(HANDOFFS)
        both = sum(1 for x in b if x["dcs_hit"] and x["model_hits"] >= 2)
        dcs_only = sum(1 for x in b if x["dcs_hit"] and x["model_hits"] < 2)
        model_only = sum(1 for x in b if not x["dcs_hit"] and x["model_hits"] >= 2)
        print(f"{who}: {len(b)} missiles; DCS hit {dh}, model hit {mh:.1f} (mean over hand-offs); "
              f"both {both}, DCS only {dcs_only}, model only {model_only}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
