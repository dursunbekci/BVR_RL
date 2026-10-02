"""
inspect_run.py  —  what happened in a DCS recording, as a timeline
===================================================================

    python dcs\\inspect_run.py dcs_runs\\live_20261002_174959_ep1_shadow.jsonl

Prints, from a dcs_live.py or bvr_logger.lua recording:
  * each aircraft: when it was seen, its altitude and speed ranges
  * every event DCS sent, in mission time from the first sample
  * every missile: who shot it at whom, when, and how it ended
  * each death, and how dcs_live.py / dcs_obs_check.py count it
    (a missile kill, destroyed by something else, or a crash)

Short enough to paste into a message.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcs_world import DcsRecording, COAL_BLUE, COAL_RED      # noqa: E402

COAL = {COAL_BLUE: "blue", COAL_RED: "red"}


def main(path):
    rec = DcsRecording(path)
    t0 = rec.t_start
    rel = lambda t: f"{float(t) - t0:7.1f}"
    h = rec.header
    print(f"{os.path.basename(path)}: theatre {h.get('theatre', '?')}"
          + (f", agent {h['agent']}, red {h['red']}" if h.get("agent") else "")
          + f", {rec.t_end - t0:.1f} s of samples")

    print("\nAIRCRAFT")
    for name, d in rec.units.items():
        alt = d["pos"][:, 2]
        spd = np.linalg.norm(d["vel"], axis=1)
        print(f"  {name:12s} {COAL.get(d['coal'], d['coal']):4s} {d['type']:12s} "
              f"seen {float(d['t'][0]) - t0:6.1f} to {float(d['t'][-1]) - t0:6.1f} s   "
              f"alt {alt.min():6.0f} to {alt.max():6.0f} m   speed {spd.min():4.0f} to {spd.max():4.0f} m/s")

    print("\nEVENTS (s from the first sample)")
    for e in rec.events:
        k = e["ev"]
        if k == "ammo":
            continue
        if k == "shot":
            what = f"{e.get('shooter')} -> {e.get('target')}  {e.get('type')}  (missile {e.get('id')})"
        elif k == "hit":
            what = f"{e.get('target')} hit by {e.get('shooter')}, {e.get('type')} (missile {e.get('id')})"
        elif k == "kill":
            what = f"{e.get('unit')} destroyed by {e.get('killer')} (missile {e.get('id')})"
        elif k == "dead":
            what = f"{e.get('unit')}: {e.get('cause')}"
        elif k == "weapon_gone":
            what = f"missile {e.get('id')} gone"
        elif k == "fire":
            what = f"fire request {e.get('status')}" + (f": {e['reason']}" if e.get("reason") else "")
        elif k == "bridge":
            what = f"{e.get('agent')}: {e.get('status')}"
        else:
            what = str({x: y for x, y in e.items() if x not in ("ev", "t")})
        print(f"  {rel(e['t'])}  {k:12s} {what}")

    print("\nMISSILES")
    for d in sorted(rec.weapons.values(), key=lambda d: d["t_shot"] or 0):
        end = ("hit " + str(d["hit_target"]) + " at " + rel(d["t_hit"]).strip()) if d["t_hit"] is not None \
            else (f"gone at {rel(d['t_end']).strip()}" if d["t_end"] is not None else "?")
        print(f"  {d['id']:3d}  {d['shooter']} -> {d['target']}  {d['type']}  "
              f"shot at {rel(d['t_shot']).strip() if d['t_shot'] is not None else '?'}  {end}")

    print("\nDEATHS (as dcs_live.py counts them)")
    if not rec.dead:
        print("  none")
    for name in rec.dead:
        how = rec.death_cause(name)
        if how["weapon"] is not None:
            verdict = f"missile kill by {how['weapon']['shooter']} (missile {how['weapon']['id']})"
        elif how["hit"]:
            verdict = "destroyed (hit by something other than a tracked missile)"
        else:
            verdict = "CRASH (no hit recorded)"
        said = [f"{e['ev'] if e['ev'] == 'kill' else e.get('cause')} at {rel(e['t']).strip()}"
                for e in rec.events if e["ev"] in ("dead", "kill") and e.get("unit") == name]
        print(f"  {name:12s} out of the fight at {rel(how['t']).strip()} s ({how['cause']}): {verdict}")
        print(f"  {'':12s} DCS reported: {', '.join(said) if said else 'nothing (it left the samples)'}")
    low = [n for n, d in rec.units.items() if d["pos"][:, 2].min() < 300.0]
    if low:
        print(f"\n  below 300 m at some point (counted as a crash if alive then): {', '.join(low)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python dcs\\inspect_run.py <recording.jsonl>")
    main(sys.argv[1])
