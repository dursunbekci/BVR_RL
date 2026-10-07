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
    For a grid of (launch mach, launch altitude, target altitude, target
    aspect), fires the REAL AIM120 (missile_sim.py) at a target flying a
    constant, level course, under
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

    Both are tabulated over target altitude too: against a target far below,
    the missile's last miles are in thick air (SIM_REV 11). At launch from
    9 km, Mach 0.9, head-on, AIM-120C-DCS reaches 84 km against a level
    target and 58 km against one at 1 km.

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
import multiprocessing as mp
import os
import time

import numpy as np

from missile_sim import AIM120, MslPhase, _isa_a
from bvr_library import missile_config, envelope_path

# Set in main(): the library missile being calibrated.
_MSL = None
# When the shortest range misses (a target steeply above or below), look for
# a hit further out in these steps, this far.
MIN_RANGE_STEP = 2_000.0
MIN_RANGE_SCAN = 30_000.0


def _fly(launch_range, shooter_mach, alt, aspect_deg, target_mach,
         dt=0.02, max_t=130.0, reactive=False, target_alt=None):
    """
    One idealised engagement. Shooter at origin flying +Y at shooter_mach.
    Target placed `launch_range` (slant range) up the Y axis, at target_alt
    (default: the shooter's altitude), flying level at target_mach.

    Aspect convention (matches aspect_deg_from_vectors in bvr_envelope.py):
    0 deg = target flying AT the shooter (nose-on, head-on shot), 180 deg =
    target flying directly away (tail chase). reactive=True snaps the target
    to a 180 deg (running) course from t=0 regardless of the nominal aspect
    — used to bound r_nez.
    """
    t_alt = alt if target_alt is None else float(target_alt)
    v_s = shooter_mach * _isa_a(alt)
    v_t = target_mach * _isa_a(t_alt)

    shooter_pos = np.array([0.0, 0.0, alt])
    shooter_vel = np.array([0.0, v_s, 0.0])
    dz = t_alt - alt
    target_pos = np.array([0.0, math.sqrt(max(launch_range ** 2 - dz ** 2, 0.0)), t_alt])

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
                       lo=2_000.0, hi=160_000.0, iters=7, target_alt=None):
    """Largest range in [lo,hi] that still HITs. Monotonic: hit at short
    range, miss at long range (checked with --verify)."""
    if target_alt is not None:            # a slant range can't be below the height gap
        lo = max(lo, abs(float(target_alt) - alt) + 500.0)
    fly = lambda r: _fly(r, shooter_mach, alt, aspect_deg, target_mach,
                         reactive=reactive, target_alt=target_alt)
    if not fly(lo):
        # Close in, a target far above or below is too steep an angle for the
        # missile to turn onto, so it can miss there and hit further out:
        # step out to the first hit before bisecting.
        r = lo + MIN_RANGE_STEP
        while r <= min(hi, lo + MIN_RANGE_SCAN) and not fly(r):
            r += MIN_RANGE_STEP
        if r > min(hi, lo + MIN_RANGE_SCAN):
            return lo  # no hit anywhere close in: floor it
        lo = r
    if fly(hi):
        return hi  # whole bracket hits; missile is stronger than our range cap
    a, b = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if fly(mid):
            a = mid
        else:
            b = mid
    return a


def _cell(job):
    """One (launch Mach, launch altitude) cell: r_max and r_nez over every
    target altitude and aspect. Runs in a worker process."""
    missile_id, mach, alt, talt_grid, aspect_grid, target_mach = job
    global _MSL
    if _MSL is None or _MSL.id != missile_id:
        _MSL = missile_config(missile_id)
    r_max = np.zeros((len(talt_grid), len(aspect_grid)))
    r_nez = np.zeros_like(r_max)
    for it, talt in enumerate(talt_grid):
        # NEZ proxy for this (mach, alt, target alt): r_max at aspect=180 with
        # an instant-reactive target (see _fly's `reactive` branch).
        nez_ceiling = _bisect_max_range(mach, alt, 180.0, target_mach, reactive=True,
                                        target_alt=talt)
        for ip, asp in enumerate(aspect_grid):
            rm = _bisect_max_range(mach, alt, asp, target_mach, target_alt=talt)
            r_max[it, ip] = rm
            r_nez[it, ip] = min(nez_ceiling, rm)
    return r_max, r_nez


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missile", default="AIM-120", help="library missile id")
    ap.add_argument("--out", default=None,
                    help="output file (default: the library path for this missile)")
    ap.add_argument("--quick", action="store_true", help="coarse grid, fast sanity run")
    ap.add_argument("--target-mach", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="processes to fly the grid with")
    args = ap.parse_args()
    global _MSL
    _MSL = missile_config(args.missile)
    out = args.out or str(envelope_path(_MSL))
    print(f"Calibrating {_MSL.id} (parameters {_MSL.fingerprint}) -> {out}", flush=True)

    if args.quick:
        mach_grid = np.array([0.7, 1.1])
        alt_grid = np.array([3000.0, 9000.0])
        talt_grid = np.array([3000.0, 9000.0])
        aspect_grid = np.array([0.0, 90.0, 180.0])
    else:
        # 0.5 covers slow launchers such as a subsonic UCAV. Launch and target
        # altitudes share one grid: 1 km is where a defending target ends up
        # after a deep dive (SIM_REV 10), and where it may shoot back from.
        mach_grid = np.array([0.5, 0.7, 1.0, 1.3])
        alt_grid = np.array([1000.0, 3000.0, 6000.0, 9000.0, 13000.0])
        talt_grid = alt_grid.copy()
        aspect_grid = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])

    Nm, Na, Nt, Np = len(mach_grid), len(alt_grid), len(talt_grid), len(aspect_grid)
    r_max = np.zeros((Nm, Na, Nt, Np))
    r_nez = np.zeros((Nm, Na, Nt, Np))

    t0 = time.time()
    cells = [(im, ia) for im in range(Nm) for ia in range(Na)]
    jobs = [(_MSL.id, float(mach_grid[im]), float(alt_grid[ia]), talt_grid, aspect_grid,
             args.target_mach) for im, ia in cells]
    ib = int(np.argmin(np.abs(aspect_grid - 90)))
    it_lvl = lambda ia: int(np.argmin(np.abs(talt_grid - alt_grid[ia])))
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        for done, ((im, ia), (rm, rn)) in enumerate(zip(cells, pool.imap(_cell, jobs)), 1):
            r_max[im, ia], r_nez[im, ia] = rm, rn
            lv = it_lvl(ia)
            print(f"  [{done}/{len(cells)}] mach={mach_grid[im]:.2f} alt={alt_grid[ia]:.0f}m  "
                  f"level target: r_max(head-on)={rm[lv, 0]/1000:.1f}km "
                  f"r_max(beam)={rm[lv, ib]/1000:.1f}km r_max(tail)={rm[lv, -1]/1000:.1f}km  "
                  f"head-on vs {talt_grid[0]:.0f}-{talt_grid[-1]:.0f}m: "
                  f"{rm[0, 0]/1000:.1f}-{rm[-1, 0]/1000:.1f}km  [{time.time()-t0:.0f}s]", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out,
             mach_grid=mach_grid, alt_grid=alt_grid, target_alt_grid=talt_grid,
             aspect_grid=aspect_grid, r_max=r_max, r_nez=r_nez,
             target_mach_ref=args.target_mach, missile=_MSL.id, fingerprint=_MSL.fingerprint)
    print(f"\nWrote {out}  ({time.time()-t0:.0f}s total)")
    lv = np.array([it_lvl(ia) for ia in range(Na)])
    level = r_max[:, np.arange(Na), lv, 0]
    print(f"Head-on r_max range across grid (level target): "
          f"{level.min()/1000:.1f}-{level.max()/1000:.1f} km")


if __name__ == "__main__":
    main()
