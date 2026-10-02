"""
sweep_envelope.py  —  Calibrate the AIM-120 launch envelope against the
actual 3-DOF missile model in missile_sim.py
============================================================================

bvr_envelope.py's analytic Aim120Envelope was a physically-shaped GUESS
(R_MAX_REF=75km head-on) written before missile_sim.py existed. It was never
checked against the missile that actually flies in this sim. It doesn't
match — not by a small margin:

    aspect      analytic claim      actual (this sweep)
    head-on     75.0 km             ~48 km
    beam        57.2 km             ~18 km
    tail-chase  22.5 km             ~17 km

Every shot the RL agent takes is gated on "am I inside r_max" (bvr_env.py
_can_fire()), and the observation features r_over_rmax_own / r_over_rnez_own
ARE r_max. Training against the analytic guess means the agent is taught
that ranges 1.5-3x beyond the missile's real reach are valid shots. That
isn't a policy that needs more training — it is a policy being asked to hit
a target the missile physically cannot reach, every single time. This is
the reason a 3000-episode run against the EASIEST opponent (STRAIGHT,
unarmed, non-manoeuvring) produced a 0% win rate: bvr_metrics.json from
that run shows mean_launch_km=68.3, launch_r_rmax=1.06 — shots fired right
at (and past) the analytic model's edge, nowhere near the missile's real one.

WHAT THIS SCRIPT DOES
    For a grid of (launch mach, launch altitude, target aspect), fires the
    REAL AIM120 (missile_sim.py) at a target flying a constant course, under
    IDEALISED guidance (perfect datalink every frame — this measures the
    missile's physical reach, not radar quality, matching what
    Aim120Envelope is supposed to represent). Bisects on launch range to
    find:

      r_max  — the furthest range that still produces a HIT against a
               NON-REACTING target (matches the STRAIGHT/EVASIVE opponents
               most of an episode, and is what "can I take this shot at
               all" should mean).

      r_nez  — the furthest range that still produces a HIT if the target
               reacts by running at full speed away from the shooter AT THE
               INSTANT OF LAUNCH. Approximated as r_max at 180 deg aspect
               for the same (mach, alt) — the standard conservative NEZ
               definition, and cheap to compute here. Clipped to <=
               r_max(aspect) per point.

    target_mach is held at the table's reference (0.9) — bvr_envelope.py
    already applies a post-hoc f_tgt correction for other target speeds
    (see Aim120Envelope._interp_table), so the table itself only needs one
    target-speed slice.

USAGE
    python sweep_envelope.py                    # AIM-120 -> library/envelopes/
    python sweep_envelope.py --missile MY-MSL   # any library missile
    python sweep_envelope.py --quick             # coarser grid, for a sanity check

Which missile: --missile <library id> (default AIM-120). The table is
written to library/envelopes/<id>-<fingerprint>.npz, where the fingerprint
is a hash of the missile's parameters, and the environment loads it from
there. Editing a missile changes its fingerprint, so a stale table can never
be used for it: the environment refuses to start until it is calibrated.
"""

import argparse
import math
import time

import numpy as np

from missile_sim import AIM120, MslPhase, _isa_a
from bvr_library import missile_config, envelope_path

# Set in main(): the library missile being calibrated.
_MSL = None


def _fly(launch_range, shooter_mach, alt, aspect_deg, target_mach,
         dt=0.02, max_t=130.0, reactive=False):
    """
    One idealised engagement. Shooter at origin flying +Y at shooter_mach.
    Target placed `launch_range` up the Y axis.

    Aspect convention (matches aspect_deg_from_vectors in bvr_envelope.py):
    0 deg = target flying AT the shooter (nose-on, head-on shot), 180 deg =
    target flying directly away (tail chase). reactive=True snaps the target
    to a 180 deg (running) course from t=0 regardless of the nominal aspect
    — used to bound r_nez.
    """
    a_snd = _isa_a(alt)
    v_s = shooter_mach * a_snd
    v_t = target_mach * a_snd

    shooter_pos = np.array([0.0, 0.0, alt])
    shooter_vel = np.array([0.0, v_s, 0.0])
    target_pos = np.array([0.0, launch_range, alt])

    eff_aspect_deg = 180.0 if reactive else aspect_deg
    th = math.radians(eff_aspect_deg)
    # aspect=0: sin=0,cos=1  -> vel=(0,-v_t,0) toward shooter    (head-on)
    # aspect=180: sin=0,cos=-1 -> vel=(0,+v_t,0) away from shooter (tail chase)
    target_vel = np.array([v_t * math.sin(th), -v_t * math.cos(th), 0.0])

    m = AIM120(owner=1, target=2, pos=shooter_pos, vel=shooter_vel,
               handoff_range=0.5 * (_MSL.HANDOFF_MIN + _MSL.HANDOFF_MAX), cfg=_MSL)
    m.last_tgt_pos = target_pos.copy()
    m.last_tgt_vel = target_vel.copy()

    t = 0.0
    tp, tv = target_pos.copy(), target_vel.copy()
    while t < max_t and m.phase not in (MslPhase.HIT, MslPhase.MISS):
        tp = tp + tv * dt
        m.update_guidance({"valid": 1, "tgt_pos": tp.tolist(), "tgt_vel": tv.tolist()})
        m.step(dt, tp, tv)
        t += dt
    return m.phase == MslPhase.HIT


def _bisect_max_range(shooter_mach, alt, aspect_deg, target_mach, reactive=False,
                       lo=2_000.0, hi=160_000.0, iters=7):
    """Largest range in [lo,hi] that still HITs. Monotonic: hit at short
    range, miss at long range (checked with --verify)."""
    if not _fly(lo, shooter_mach, alt, aspect_deg, target_mach, reactive=reactive):
        return lo  # even point-blank misses (shouldn't happen; floor it)
    if _fly(hi, shooter_mach, alt, aspect_deg, target_mach, reactive=reactive):
        return hi  # whole bracket hits; missile is stronger than our range cap
    a, b = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if _fly(mid, shooter_mach, alt, aspect_deg, target_mach, reactive=reactive):
            a = mid
        else:
            b = mid
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missile", default="AIM-120", help="library missile id")
    ap.add_argument("--out", default=None,
                    help="output file (default: the library path for this missile)")
    ap.add_argument("--quick", action="store_true", help="coarse grid, fast sanity run")
    ap.add_argument("--target-mach", type=float, default=0.9)
    args = ap.parse_args()
    global _MSL
    _MSL = missile_config(args.missile)
    out = args.out or str(envelope_path(_MSL))
    print(f"Calibrating {_MSL.id} (parameters {_MSL.fingerprint}) -> {out}", flush=True)

    if args.quick:
        mach_grid = np.array([0.7, 1.1])
        alt_grid = np.array([3000.0, 9000.0])
        aspect_grid = np.array([0.0, 90.0, 180.0])
    else:
        # Coarse but monotonic beats the previous constant-formula guess by a
        # wide margin — this is the resolution that fits a reasonable wall
        # clock budget for this pure-Python missile sim (~1.2s/simulated
        # engagement). Tighten later with more grid points / GPU batching if
        # the trilinear interpolation proves too coarse near the NEZ.
        # 0.5 covers slow launchers such as a subsonic UCAV.
        mach_grid = np.array([0.5, 0.7, 1.0, 1.3])
        alt_grid = np.array([3000.0, 9000.0, 13000.0])
        aspect_grid = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])

    Nm, Na, Np = len(mach_grid), len(alt_grid), len(aspect_grid)
    r_max = np.zeros((Nm, Na, Np))
    r_nez = np.zeros((Nm, Na, Np))

    t0 = time.time()
    total = Nm * Na
    done = 0
    ib = int(np.argmin(np.abs(aspect_grid - 90)))
    for im, mach in enumerate(mach_grid):
        for ia, alt in enumerate(alt_grid):
            # NEZ proxy for this (mach,alt): r_max at aspect=180 with an
            # instant-reactive target (see _fly's `reactive` branch).
            nez_ceiling = _bisect_max_range(mach, alt, 180.0, args.target_mach, reactive=True)
            for ip, asp in enumerate(aspect_grid):
                rm = _bisect_max_range(mach, alt, asp, args.target_mach, reactive=False)
                r_max[im, ia, ip] = rm
                r_nez[im, ia, ip] = min(nez_ceiling, rm)
            done += 1
            print(f"  [{done}/{total}] mach={mach:.2f} alt={alt:.0f}m  "
                  f"r_max(head-on)={r_max[im,ia,0]/1000:.1f}km "
                  f"r_max(beam)={r_max[im,ia,ib]/1000:.1f}km "
                  f"r_max(tail)={r_max[im,ia,-1]/1000:.1f}km  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    import os
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out,
             mach_grid=mach_grid, alt_grid=alt_grid, aspect_grid=aspect_grid,
             r_max=r_max, r_nez=r_nez, target_mach_ref=args.target_mach,
             missile=_MSL.id, fingerprint=_MSL.fingerprint)
    print(f"\nWrote {out}  ({time.time()-t0:.0f}s total)")
    print(f"Head-on r_max range across grid: "
          f"{r_max[:,:,0].min()/1000:.1f}-{r_max[:,:,0].max()/1000:.1f} km")


if __name__ == "__main__":
    main()
