"""
bvr_perf.py  —  Performance card for a library airframe or platform
===================================================================

Flies the real flight model (f16_sim.F16Aircraft with the item's parameters)
through a few standard manoeuvres and reports what the numbers produce:

    top speed          level flight at full throttle, by altitude
    max-bank turn      turn rate over the last 10 s of a 40 s maximum-bank
                       level turn, and the speed and altitude it costs
    climb              time and speed change for a 3 km climb, flown with
                       the platform's climb schedule
    ceiling            highest altitude still climbing at 0.5 m/s
    autopilot          overshoot and settling time of a 90° heading change
                       and of a 1 km altitude change

A number that is off by a factor of two shows up here in seconds, not as a
strange training run a week later. Nothing here is learned; it is a flight
test of the parameters as entered.

    python bvr_perf.py F-16C                  # platform id
    python bvr_perf.py --airframe F-16C
"""

import argparse
import json
import math

import numpy as np

from f16_sim import F16Aircraft, isa
from bvr_library import airframe_config, load_platform

DT = 0.02
D2R = math.pi / 180.0


def _new(cfg, alt, V, chi=0.0):
    ac = F16Aircraft(np.random.default_rng(0), cfg=cfg)
    ac.reset_state(0.0, 0.0, alt, chi, V, fuel_frac=0.6)
    ac.throttle = 1.0
    return ac


def _a(alt):
    return isa(alt)[3]


def top_speed(cfg, alt, t_max=400.0):
    """Level flight, speed command far above reach: where acceleration stops."""
    ac = _new(cfg, alt, 0.8 * _a(alt))
    cmd = {"hdgCmd": 0.0, "altTarget": alt, "V": 2000.0, "altFPA": 10 * D2R}
    t, v_prev = 0.0, ac.V
    while t < t_max:
        for _ in range(int(5.0 / DT)):
            ac.step(DT, cmd)
        t += 5.0
        if abs(ac.V - v_prev) < 0.25:       # < 0.05 m/s² over 5 s: settled
            break
        v_prev = ac.V
    return ac.V, ac.V / _a(alt), abs(ac.z - alt)


def max_bank_turn(cfg, alt, V):
    """Maximum-bank level turn at a commanded speed for 40 s."""
    ac = _new(cfg, alt, V)
    chi0 = ac.chi
    rates = []
    for k in range(int(40.0 / DT)):
        cmd = {"hdgCmd": (ac.chi + 170 * D2R) % (2 * math.pi), "altTarget": alt, "V": V,
               "altFPA": 25 * D2R}
        c0 = ac.chi
        ac.step(DT, cmd)
        if k * DT > 30.0:                   # the last 10 s
            rates.append(((ac.chi - c0 + math.pi) % (2 * math.pi) - math.pi) / DT)
    return {"turn_rate_deg_s": float(np.mean(rates)) / D2R, "speed_end": ac.V,
            "speed_loss": V - ac.V, "alt_error": ac.z - alt, "bank_deg": abs(ac.phi) / D2R,
            "load_factor": ac.nz}


def climb(cfg, alt0, V, climb_fpa=None, dz=3000.0):
    """Climb dz from alt0 at a commanded speed, with an optional climb-angle schedule."""
    ac = _new(cfg, alt0, V)
    t = 0.0
    while ac.z < alt0 + dz - 5.0 and t < 600.0:
        cmd = {"hdgCmd": 0.0, "altTarget": alt0 + dz, "V": V, "altFPA": 25 * D2R}
        if climb_fpa is not None:
            cmd["climbFPA"] = climb_fpa(ac.z)
        ac.step(DT, cmd)
        t += DT
    return {"time_s": t, "rate_m_s": (ac.z - alt0) / max(t, 1e-6), "speed_start": V,
            "speed_end": ac.V, "reached": ac.z >= alt0 + dz - 5.0}


def ceiling(cfg, t_max=900.0):
    """Climb at full throttle and a shallow angle until vertical speed drops below 0.5 m/s."""
    ac = _new(cfg, 1000.0, 0.7 * _a(1000.0))
    t, z_prev = 0.0, ac.z
    while t < t_max:
        for _ in range(int(10.0 / DT)):
            v_best = 0.8 * _a(ac.z)
            ac.step(DT, {"hdgCmd": 0.0, "altTarget": 30_000.0, "V": v_best, "altFPA": 6 * D2R})
        t += 10.0
        if (ac.z - z_prev) / 10.0 < 0.5 or ac.z >= cfg.ALT_CEIL - 1.0:
            break
        z_prev = ac.z
    return min(ac.z, cfg.ALT_CEIL), ac.z >= cfg.ALT_CEIL - 1.0


def _settle_time(err, band, t_total):
    """First time after which |err| stays inside band; None if it never does."""
    outside = [k for k, e in enumerate(err) if abs(e) > band]
    if not outside:
        return 0.0
    k = outside[-1] + 1
    return None if k >= len(err) else k * DT


def autopilot_check(cfg, alt, V, t_total=90.0):
    """Step responses: a 90° heading change and a +1 km altitude change."""
    ac = _new(cfg, alt, V)
    hdg_err = []
    for _ in range(int(t_total / DT)):
        ac.step(DT, {"hdgCmd": 90 * D2R, "altTarget": alt, "V": V, "altFPA": 25 * D2R})
        hdg_err.append(((ac.chi - 90 * D2R + math.pi) % (2 * math.pi) - math.pi) / D2R)
    ac = _new(cfg, alt, V)
    alt_err = []
    for _ in range(int(t_total / DT)):
        ac.step(DT, {"hdgCmd": 0.0, "altTarget": alt + 1000.0, "V": V, "altFPA": 25 * D2R})
        alt_err.append(ac.z - (alt + 1000.0))
    return {"heading_overshoot_deg": max(0.0, max(hdg_err)),
            "heading_settle_s": _settle_time(hdg_err, 2.0, t_total),
            "altitude_overshoot_m": max(0.0, max(alt_err)),
            "altitude_settle_s": _settle_time(alt_err, 50.0, t_total)}


def card(airframe_cfg, platform=None) -> dict:
    """The full card, with plain-language warnings."""
    cfg = airframe_cfg
    warn = []
    speeds = {}
    for alt in (0.0, 3000.0, 6000.0, 9000.0, 12000.0):
        if alt >= cfg.ALT_CEIL:
            continue
        v, m, dz = top_speed(cfg, alt)
        speeds[int(alt)] = {"speed": round(v, 1), "mach": round(m, 3)}
    if not speeds:
        warn.append("ceiling is below every test altitude")
    v_ref = platform.speed_cmds[len(platform.speed_cmds) // 2] if platform else 0.8 * _a(9000.0)
    turns = {int(alt): max_bank_turn(cfg, alt, v_ref) for alt in (3000.0, 9000.0)
             if alt < cfg.ALT_CEIL}
    if 0 in speeds and speeds[0]["mach"] > 1.3:
        warn.append(f"top speed at sea level is Mach {speeds[0]['mach']:.2f}: the transonic drag "
                    f"rise ends at Mach {cfg.DRAG_RISE_M2:g} and there is no supersonic wave drag "
                    f"beyond it (real fighters are limited to about Mach 1.2 at sea level)")
    for alt, t in turns.items():
        if abs(t["alt_error"]) > 150:
            warn.append(f"a hard turn at {alt/1000:.0f} km loses {abs(t['alt_error']):.0f} m of "
                        f"altitude: turns do not hold altitude")
    cl = climb(cfg, 5000.0, v_ref, platform.climb_fpa if platform else None)
    if not cl["reached"]:
        warn.append("cannot climb 3 km from 5 km within 10 minutes")
    elif cl["speed_start"] - cl["speed_end"] > 50:
        warn.append(f"the climb schedule bleeds {cl['speed_start']-cl['speed_end']:.0f} m/s; "
                    f"lower the climb angles")
    ceil, capped = ceiling(cfg)
    ap = autopilot_check(cfg, 6000.0, v_ref)
    if ap["heading_settle_s"] is None or ap["heading_overshoot_deg"] > 10:
        warn.append(f"heading hold overshoots a 90° turn by {ap['heading_overshoot_deg']:.0f}° "
                    f"or does not settle: lower 'Bank per heading error' or raise 'Roll rate "
                    f"per bank error'")
    if ap["altitude_settle_s"] is None or ap["altitude_overshoot_m"] > 150:
        settle = ("never settles" if ap["altitude_settle_s"] is None
                  else f"takes {ap['altitude_settle_s']:.0f} s to settle within 50 m")
        warn.append(f"altitude hold overshoots a 1 km step by {ap['altitude_overshoot_m']:.0f} m "
                    f"and {settle} (underdamped): lower 'Flight-path angle per altitude error' "
                    f"or raise 'Load factor per flight-path error'")
    if platform:
        for alt_k in (3000, 9000):
            if alt_k in speeds:
                top = speeds[alt_k]["speed"]
                over = [c for c in platform.speed_cmds if c > top + 1.0]
                if over:
                    warn.append(f"speed choices {', '.join(f'{c:.0f}' for c in over)} m/s are above "
                                f"the top speed at {alt_k/1000:.0f} km ({top:.0f} m/s)")
        if platform.alt_max > ceil + 1:
            warn.append(f"highest commanded altitude {platform.alt_max:.0f} m is above the "
                        f"ceiling found ({ceil:.0f} m)")
    return {"top_speed": speeds, "turn_speed_ref": v_ref,
            "max_bank_turn_40s": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in turns.items()},
            "climb_3km_from_5km": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in cl.items()},
            "ceiling_m": round(ceil, 0), "ceiling_at_model_cap": capped,
            "autopilot": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in ap.items()},
            "warnings": warn}


def card_for(kind: str, item_id: str) -> dict:
    if kind == "platform":
        p = load_platform(item_id)
        return card(p.airframe, p)
    if kind == "airframe":
        return card(airframe_config(item_id))
    raise ValueError("performance cards exist for airframes and platforms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", nargs="?", default=None)
    ap.add_argument("--airframe", default=None)
    args = ap.parse_args()
    kind, iid = ("airframe", args.airframe) if args.airframe else ("platform", args.platform or "F-16C")
    print(json.dumps(card_for(kind, iid), indent=2))


if __name__ == "__main__":
    main()
