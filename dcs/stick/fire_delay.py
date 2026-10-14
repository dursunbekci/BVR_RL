"""
fire_delay.py  —  the launch delay in DCS: measured, and set on a platform for training
=========================================================================================

In DCS a fire decision is cockpit work before a missile leaves: the stick pilot (pilot.py, the live runner's
own code) puts the radar cursor on the target's bearing, presses target management right until the TWS track
is bugged, sets master arm and holds the weapon release 1.5 s. Training models this as the platform's
LAUNCH_DELAY_MIN / LAUNCH_DELAY_MAX (bvr_env: the launch goes a uniform draw between them after the decision).

1. measure (in DCS, the stick-test mission, the target ahead, hands off):

       python dcs/stick/fire_delay.py measure [--trials 10] [--fire 1]

   Flies so the target sits 0, +20, -20, +35, -35 deg off the nose in turn and, each time, drops the bug and
   times a new one exactly as a fire job does it (no missile spent). --fire N ends with N real launches,
   which also time master arm, the held release and the launch seen. Saved: dcs_runs/fire_delay_<time>.json.

2. set (no DCS): the delays from those files, and from the request-to-launch times dcs_live.py --stick writes
   to results.csv (launch_delays_s, end to end, the best evidence), are summarised, and the platform's
   LAUNCH_DELAY_MIN / MAX set to their 10th and 90th percentiles:

       python dcs/stick/fire_delay.py set F-16C-DCSFIT dcs_runs/fire_delay_<time>.json [dcs_runs/results.csv ...]
"""

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

D2R = math.pi / 180.0
OFFSETS = (0.0, 20.0, -20.0, 35.0, -35.0)     # deg, target right of the nose; TWS scans 25 deg either side of the cursor
RELEASE_EST_S = 2.5     # master arm (first shot), the release held 1.5 s, the launch seen: until a real launch times it


def _now(pilot):
    fr = pilot.r.frames
    return fr[-1].t if fr else None


def _wait(pilot, secs, until=None, wall_max=None, log=None):
    """Wait `secs` of DCS time (or until `until()` is true), draining the pilot's events into log."""
    t0, w0 = _now(pilot), time.time()
    wall_max = wall_max if wall_max is not None else max(30.0, 3.0 * secs)
    while True:
        for kind, status, why in pilot.drain_events():
            if log:
                log(f"    {kind} {status}: {why}")
        t = _now(pilot)
        if t is not None and t0 is None:
            t0 = t
        if until is not None and until():
            return True
        if t is not None and t0 is not None and t - t0 >= secs:
            return until is None
        if time.time() - w0 > wall_max:
            return False
        time.sleep(0.01)


def _idle(pilot):
    return pilot.busy is None and pilot.jobs.empty()


def _target(pilot):
    """(frame, nearest aircraft) from the latest frame with the world objects, or (None, None)."""
    import cockpit
    f = cockpit.last_with(pilot.r.frames[-300:], "obj")
    nr = cockpit.nearest(f) if f is not None else None
    return f, nr


def measure(link, gains=None, trials=10, fire=0, offsets=OFFSETS, settle_s=15.0, out_dir="dcs_runs",
            platform="F-16C-DCSFIT", log=print, pilot=None):
    """Fly the trials with a StickPilot on `link`; returns the record (also saved to out_dir)."""
    from pilot import StickPilot
    airframe = None
    try:
        from bvr_library import load_platform
        airframe = load_platform(platform).airframe
    except Exception as e:
        log(f"  WARNING: platform {platform} not loaded ({e}): no g and bank limits from its airframe")
    if pilot is None:
        pilot = StickPilot(link, gains=gains, log=log, out_dir=None, airframe=airframe)
    pilot.start()
    try:
        if not _wait(pilot, 30.0, until=lambda: _target(pilot)[1] is not None, wall_max=60.0):
            raise SystemExit("no other aircraft in the world objects: fly the stick-test mission (a target ahead)")
        f, nr = _target(pilot)
        alt0, v0 = f.alt, f.tas or 250.0
        log(f"  start: {alt0:.0f} m, {v0:.0f} m/s; target {nr[0]} at {nr[1] / 1000:.0f} km, {nr[2]:+.0f} deg")
        pilot.command(f.crs, alt0, v0)
        pilot.radar()
        _wait(pilot, 60.0, until=lambda: _idle(pilot), log=log)
        _wait(pilot, 6.0, log=log)                     # the track file builds in TWS
        for i in range(trials + fire):
            off = offsets[i % len(offsets)]
            f, nr = _target(pilot)
            if nr is None:
                log("  the target is gone: stopping")
                break
            pilot.command((f.hdg + nr[2] * D2R - off * D2R) % (2 * math.pi), alt0, v0)

            def settled():
                fo, n2 = _target(pilot)
                fl = pilot.r.frames[-1]
                return (n2 is not None and abs(n2[2] - off) < 5.0 and fl.bank is not None
                        and abs(fl.bank) < 10 * D2R and _idle(pilot))
            _wait(pilot, settle_s + 20.0, until=settled, log=log)
            _wait(pilot, 3.0, log=log)
            what = "bug" if i < trials else "FIRE"
            log(f"  trial {i + 1}/{trials + fire}: target {off:+.0f} deg off the nose, {what}")
            n_b, n_f = len(pilot.bug_times), len(pilot.fire_times)
            pilot.time_bug() if i < trials else pilot.fire()
            _wait(pilot, 40.0, until=lambda: len(pilot.bug_times) > n_b or len(pilot.fire_times) > n_f, log=log)
            if i >= trials:
                _wait(pilot, 3.0, log=log)
    finally:
        pilot.stop()
    rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "platform": platform,
           "offsets": list(offsets), "bug_times": pilot.bug_times, "fire_times": pilot.fire_times}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"fire_delay_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)
        rec["path"] = path
    return rec


def delays_from(paths):
    """Launch delays (s) from fire_delay records and results.csv files: {"live": [...], "measured": [...],
    "release_s": float, "release_measured": bool, "bug_fail": (failed, tried), "notes": [...]}."""
    import statistics
    live, recs = [], []
    for p in paths:
        if p.lower().endswith(".csv"):
            with open(p, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("mode") not in (None, "", "stick"):
                        continue
                    live += [float(x) for x in (row.get("launch_delays_s") or "").split(";") if x.strip()]
        else:
            with open(p) as fh:
                recs.append(json.load(fh))
    fires = [ft for r in recs for ft in r.get("fire_times", []) if ft.get("ok") and ft.get("bug_s") is not None]
    rel = [ft["total_s"] - ft["bug_s"] for ft in fires]
    release = statistics.median(rel) if rel else RELEASE_EST_S
    bugs = [b for r in recs for b in r.get("bug_times", [])]
    measured = [b["s"] + release for b in bugs if b.get("ok")] + [ft["total_s"] for ft in fires]
    failed = sum(1 for b in bugs if not b.get("ok")) + sum(1 for r in recs for ft in r.get("fire_times", [])
                                                         if not ft.get("ok"))
    tried = len(bugs) + sum(len(r.get("fire_times", [])) for r in recs)
    return {"live": live, "measured": measured, "release_s": release, "release_measured": bool(rel),
            "bug_fail": (failed, tried)}


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q / 100.0
    i = int(math.floor(k))
    return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (k - i)


def summarise(d, log=print):
    """Which delays decide, and the (min, max) they give: the 10th and 90th percentiles, to 0.5 s. The live
    request-to-launch times decide once there are 5 (they include waiting for a radar job, as a fight does)."""
    use = d["live"] if len(d["live"]) >= 5 else d["live"] + d["measured"]
    src = "dcs_live.py --stick runs" if len(d["live"]) >= 5 else "timed bugs and launches (and any live runs)"
    for name, xs in (("live request to launch", d["live"]), ("measured (bug + release)", d["measured"])):
        if xs:
            log(f"  {name}: {len(xs)} samples, median {_pct(xs, 50):.1f} s, 10th-90th {_pct(xs, 10):.1f}-"
                f"{_pct(xs, 90):.1f} s, {min(xs):.1f}-{max(xs):.1f} s in all")
    log(f"  release (master arm, the held release, the launch seen): {d['release_s']:.1f} s "
        f"({'measured' if d['release_measured'] else 'estimate: no launch timed yet'})")
    f, n = d["bug_fail"]
    if n:
        log(f"  bugs that failed in 10 s: {f} of {n}")
    if len(use) < 3:
        return None, src
    lo = math.floor(_pct(use, 10) * 2) / 2
    hi = math.ceil(_pct(use, 90) * 2) / 2
    return (lo, max(hi, lo)), src


def set_platform(pid, lo, hi, note):
    """Write LAUNCH_DELAY_MIN / MAX into a user library platform."""
    from bvr_library import get_item, save_item, load_platform
    it = get_item("platform", pid)
    if it.get("builtin"):
        raise SystemExit(f"{pid} is a built-in platform: copy it to the user library first")
    it.pop("builtin", None)
    it["params"] = dict(it["params"], LAUNCH_DELAY_MIN=float(lo), LAUNCH_DELAY_MAX=float(hi))
    src = dict(it.get("sources") or {})
    src["LAUNCH_DELAY_MIN"] = src["LAUNCH_DELAY_MAX"] = note
    it["sources"] = src
    save_item(it)
    p = load_platform(pid)
    return p.launch_delay_min, p.launch_delay_max


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="in DCS: time bugs (and --fire launches) with the live runner's pilot")
    m.add_argument("--trials", type=int, default=10, help="timed bugs, no missile spent (default 10)")
    m.add_argument("--fire", type=int, default=0, help="real launches after them, to time the release (default 0)")
    m.add_argument("--platform", default="F-16C-DCSFIT", help="whose airframe limits the pilot flies with")
    m.add_argument("--out", default="dcs_runs")
    s = sub.add_parser("set", help="no DCS: set a platform's LAUNCH_DELAY_MIN/MAX from measured files")
    s.add_argument("platform", help="a user library platform, e.g. F-16C-DCSFIT")
    s.add_argument("files", nargs="+", help="dcs_runs/fire_delay_<time>.json and/or dcs_runs/results.csv")
    s.add_argument("--dry-run", action="store_true", help="show the values, do not write them")
    a = ap.parse_args(argv)
    if a.cmd == "measure":
        from stick_test import UdpLink
        link = UdpLink()
        try:
            rec = measure(link, trials=a.trials, fire=a.fire, out_dir=a.out, platform=a.platform)
        finally:
            link.release()
            link.close()
        print()
        rng, _ = summarise(delays_from([rec["path"]]))
        print(f"\n  saved: {rec['path']}")
        if rng:
            print(f"  set it: python dcs/stick/fire_delay.py set {a.platform} {rec['path']}"
                  f"   (LAUNCH_DELAY {rng[0]:.1f}-{rng[1]:.1f} s)")
        return rec
    d = delays_from(a.files)
    rng, src = summarise(d)
    if rng is None:
        raise SystemExit("fewer than 3 delays in those files: nothing set")
    note = (f"measured {datetime.date.today().isoformat()} from {src} "
            f"({', '.join(os.path.basename(f) for f in a.files)}): the 10th and 90th percentiles, to 0.5 s")
    if a.dry_run:
        print(f"\n  {a.platform}: LAUNCH_DELAY_MIN {rng[0]:.1f}, LAUNCH_DELAY_MAX {rng[1]:.1f} s (dry run: not written)")
        return rng
    got = set_platform(a.platform, rng[0], rng[1], note)
    print(f"\n  {a.platform}: LAUNCH_DELAY_MIN {got[0]:.1f}, LAUNCH_DELAY_MAX {got[1]:.1f} s written "
          f"(library/user/platform/{a.platform}.json; commit it so training elsewhere gets it)")
    return got


if __name__ == "__main__":
    main()
