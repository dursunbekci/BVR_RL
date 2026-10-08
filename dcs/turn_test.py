"""
turn_test.py  —  how hard does the DCS AI turn when the bridge steers it?
=========================================================================

The policy steers BLUE-1 with a heading; bvr_bridge.lua turns that into a
route for the DCS AI, which flew it at about 45 deg of bank and 1.5 g: about
2 deg/s, against 7-8 deg/s for the simulator's autopilot. This flies a fixed
series of heading changes through the bridge, no policy, and measures the
turns, for several ways of building the route:

    base         first route point 3 km ahead (as dcs_live.py flies)
    near1000     first point 1 km ahead
    far_only     no first point: only the far one, 60 km ahead
    flyover1000  first point 1 km ahead, both points "Fly Over Point"

Each variant flies three turns from the heading it starts on: 90 deg right,
back 90 left, then 170 right. Red holds its fire throughout. All four
variants run in one mission, about 15 minutes of mission time (less if the
turns are fast); use DCS time acceleration if you like.

    python dcs/turn_test.py                          # every variant, in DCS
    python dcs/turn_test.py --variants base far_only
    python dcs/turn_test.py --speed 280 --alt 6000   # another flight condition
    python dcs/turn_test.py --sim                    # the same turns in BVR_RL's simulator

Needs a mission built with bridge version 3 or later (python dcs/make_mission.py).
Writes dcs_runs/turn_<time>.jsonl (everything DCS sent) and appends one row
per turn to dcs_runs/turn_test.csv; prints a table.
"""

import argparse
import csv
import json
import math
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np                                        # noqa: E402

G = 9.80665
OPT_BRIDGE = 3            # bridge version with OPT near / wpt / redhold

VARIANTS = {
    "base":        {"near": 3000, "wpt": "turn"},
    "near1000":    {"near": 1000, "wpt": "turn"},
    "far_only":    {"near": 0,    "wpt": "turn"},
    "flyover1000": {"near": 1000, "wpt": "flyover"},
}
TURNS = [+90.0, -90.0, +170.0]    # each from the heading the previous one aimed at
SETTLE_S = 10.0                   # straight and level before a variant's first turn
DONE_DEG = 5.0                    # a turn is done within this of its heading ...
DONE_HOLD_S = 3.0                 # ... held this long
TURN_TIMEOUT = lambda deg: 15.0 + abs(deg) / 1.5      # s; generous for 1.5 deg/s
BULK_DEG = 30.0                   # turn rate measured while more than this is still to go
BOOST_ON, BOOST_OFF = 20.0, 5.0   # as dcs_live.py --speed-boost


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


# ── one sample of BLUE-1 from a bvr_logger line ───────────────────────
def sample(t, u):
    """t, heading (deg, map north clockwise), speed, altitude, bank (deg, right
    wing down positive), velocity, up vector. DCS axes: x north, y up, z east."""
    v = np.array([u["vx"], u["vy"], u["vz"]], dtype=float)
    f = np.array([u["fx"], u["fy"], u["fz"]], dtype=float)
    up = np.array([u["ux"], u["uy"], u["uz"]], dtype=float)
    right = np.cross(f, up)                      # x north, y up, z east: f x up points right
    bank = math.degrees(math.atan2(-right[1], up[1]))
    return {"t": float(t), "hdg": math.degrees(math.atan2(v[2], v[0])) % 360.0,
            "spd": float(np.linalg.norm(v)), "alt": float(u["y"]), "bank": bank,
            "v": v, "up": up}


def load_factor(samples):
    """nz from the change in velocity: (dv/dt + g up) . body up / g, centred over ±0.2 s."""
    out = []
    for i, s in enumerate(samples):
        j0, j1 = i, i
        while j0 > 0 and s["t"] - samples[j0]["t"] < 0.2:
            j0 -= 1
        while j1 < len(samples) - 1 and samples[j1]["t"] - s["t"] < 0.2:
            j1 += 1
        dt = samples[j1]["t"] - samples[j0]["t"]
        if dt <= 0:
            out.append(1.0)
            continue
        a = (samples[j1]["v"] - samples[j0]["v"]) / dt + np.array([0.0, G, 0.0])
        out.append(float(np.dot(a, s["up"]) / G))
    return out


def measure(samples, t_cmd, t_end, h_from, delta):
    """One commanded turn of `delta` deg from h_from, flown between t_cmd and t_end."""
    seg = [s for s in samples if t_cmd <= s["t"] <= t_end]
    if len(seg) < 3:
        return None
    nz = load_factor(seg)
    sign = 1.0 if delta > 0 else -1.0
    prog = [sign * wrap(s["hdg"] - h_from) for s in seg]      # degrees turned so far
    full = abs(delta)

    def first(frac):
        for s, p in zip(seg, prog):
            if p >= frac * full:
                return s["t"]
        return None
    t50, t90 = first(0.5), first(0.9)
    # The rate that matters: while more than BULK_DEG is still to go. Near the
    # end every autopilot eases in (the simulator's bank is proportional to
    # the heading error), so a mean over the whole turn mostly measures that.
    k = next((i for i, p in enumerate(prog) if full - p <= BULK_DEG), len(prog) - 1)
    bulk = prog[k] / (seg[k]["t"] - t_cmd) if seg[k]["t"] > t_cmd else 0.0
    peak = 0.0
    for i, s in enumerate(seg):                                # best 1 s of turning
        j = next((k for k in range(i, len(seg)) if seg[k]["t"] - s["t"] >= 1.0), None)
        if j is not None:
            peak = max(peak, (prog[j] - prog[i]) / (seg[j]["t"] - s["t"]))
    return {
        "turn_deg": delta,
        "speed_mps": round(seg[0]["spd"], 0), "alt_m": round(seg[0]["alt"], 0),
        "t50_s": round(t50 - t_cmd, 1) if t50 is not None else "",
        "t90_s": round(t90 - t_cmd, 1) if t90 is not None else "",
        "rate_bulk_dps": round(bulk, 2),
        "rate_peak_dps": round(peak, 2),
        "bank_max_deg": round(max(abs(s["bank"]) for s in seg), 0),
        "nz_max": round(max(nz), 2),
        # The g the turn was flown at; nz_max also catches a brief pull at the start.
        "nz_median": round(float(np.median(nz[:k + 1])), 2),
        "alt_change_m": round(max(abs(s["alt"] - seg[0]["alt"]) for s in seg), 0),
        "reached": t90 is not None,
    }


# ── the test ──────────────────────────────────────────────────────────
class TurnTest:
    def __init__(self, link, agent, t0, variants, speed, alt, boost, log=print):
        self.link, self.agent, self.log = link, agent, log
        self.variants, self.speed, self.alt, self.boost = variants, speed, alt, boost
        self.samples, self.acks = [], []
        self.t, self.seq, self.t_sent, self.boosting = t0, 0, -1e9, False
        self.ended = False

    def pump(self, timeout=0.2):
        for d in self.link.poll(timeout):
            if d.get("ev") == "bridge" and d.get("status") == "opt":
                self.acks.append(d)
            elif d.get("ev") == "mission_end":
                self.ended = True
            elif d.get("ev") == "dead" and d.get("unit") == self.agent:
                self.ended = True
            elif "units" in d:
                u = next((u for u in d["units"] if u.get("name") == self.agent), None)
                if u is not None:
                    s = sample(d["t"], u)
                    if not self.samples or s["t"] > self.samples[-1]["t"]:
                        self.samples.append(s)
                        self.t = s["t"]

    def command(self, hdg):
        """The heading, the altitude and the speed (boosted like dcs_live.py), once a second."""
        if self.t - self.t_sent < 1.0:
            return
        spd = self.speed
        if self.boost and self.samples:
            v = self.samples[-1]["spd"]
            if v < self.speed - BOOST_ON:
                self.boosting = True
            elif v >= self.speed - BOOST_OFF:
                self.boosting = False
            if self.boosting:
                spd = max(self.speed, self.boost)
        self.seq += 1
        self.link.send(f"CMD {self.seq} {hdg % 360.0:.2f} {self.alt:.1f} {spd:.1f} 0 {self.t:.1f}")
        self.t_sent = self.t

    def wait(self, until, hdg):
        while self.t < until and not self.ended:
            self.command(hdg)
            self.pump()

    def options(self, near, wpt, redhold):
        """Send the variant's options until the bridge confirms them (≤ 10 s)."""
        n0, t_end = len(self.acks), self.t + 10.0
        want = (float(near), wpt, bool(redhold))
        while self.t < t_end and not self.ended:
            for line in (f"OPT near {near}", f"OPT wpt {wpt}", f"OPT redhold {int(redhold)}"):
                self.link.send(line)
            for _ in range(5):
                self.pump()
            a = self.acks[-1] if len(self.acks) > n0 else None
            if a and (float(a["near_m"]), a["wpt"], bool(a["redhold"])) == want:
                return True
        return False

    def run(self):
        rows = []
        while not self.samples and not self.ended:
            self.pump()
        hdg = self.samples[-1]["hdg"] if self.samples else 0.0
        for name in self.variants:
            var = VARIANTS[name]
            if not self.options(var["near"], var["wpt"], True):
                self.log(f"  {name}: the bridge did not confirm its options; skipped")
                continue
            self.log(f"variant {name}: first point {var['near']} m, {var['wpt']} points")
            self.wait(self.t + SETTLE_S, hdg)
            for delta in TURNS:
                h_from = self.samples[-1]["hdg"]
                target = (hdg + delta) % 360.0
                t_cmd, held = self.t, None
                limit = t_cmd + TURN_TIMEOUT(delta)
                while self.t < limit and not self.ended:
                    self.command(target)
                    self.pump()
                    if abs(wrap(self.samples[-1]["hdg"] - target)) <= DONE_DEG:
                        held = held if held is not None else self.t
                        if self.t - held >= DONE_HOLD_S:
                            break
                    else:
                        held = None
                m = measure(self.samples, t_cmd, self.t, h_from, wrap(target - h_from))
                hdg = target
                if m:
                    m.update(variant=name, near_m=var["near"], wpt=var["wpt"])
                    rows.append(m)
                    self.log(f"  turn {m['turn_deg']:+5.0f} deg at {m['speed_mps']:.0f} m/s: "
                             f"{m['rate_bulk_dps']} deg/s (peak {m['rate_peak_dps']}), half in "
                             f"{m['t50_s'] or 'never'} s, 90% in {m['t90_s'] or 'never'} s, "
                             f"bank {m['bank_max_deg']:.0f} deg, {m['nz_median']:.1f} g "
                             f"(peak {m['nz_max']:.1f})")
                if self.ended:
                    break
            if self.ended:
                self.log("  the mission ended (or BLUE-1 was lost): stopping")
                break
        for _ in range(3):
            self.link.send("OPT redhold 0")
            self.link.send("STOP")
        return rows


FIELDS = ["run", "mode", "variant", "near_m", "wpt", "turn_deg", "speed_mps", "alt_m", "t50_s",
          "t90_s", "rate_bulk_dps", "rate_peak_dps", "bank_max_deg", "nz_median", "nz_max", "alt_change_m",
          "reached"]


def summary(rows, log=print):
    log("")
    log(f"turn rate while more than {BULK_DEG:.0f} deg is still to go (the simulator's F-16: "
        f"about 8 deg/s at 340 m/s, 9 km)")
    log(f"{'variant':12s} {'turn':>6s} {'deg/s':>6s} {'peak':>6s} {'half, s':>8s} {'90%, s':>7s} "
        f"{'bank':>5s} {'g':>5s} {'peak g':>7s}")
    for r in rows:
        log(f"{r['variant']:12s} {r['turn_deg']:+6.0f} {r['rate_bulk_dps']:6.2f} "
            f"{r['rate_peak_dps']:6.2f} {str(r['t50_s']):>8s} {str(r['t90_s']):>7s} "
            f"{r['bank_max_deg']:5.0f} {r['nz_median']:5.1f} {r['nz_max']:7.1f}")
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r["rate_bulk_dps"])
    log("")
    for v, rates in by.items():
        log(f"  {v:12s} {np.mean(rates):.1f} deg/s over {len(rates)} turns")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--speed", type=float, default=340.0, help="commanded speed, m/s (default 340)")
    ap.add_argument("--alt", type=float, default=None,
                    help="commanded altitude, m (default: BLUE-1's at the start)")
    ap.add_argument("--boost", type=float, default=550.0,
                    help="speed asked for while 20 m/s slow, as dcs_live.py --speed-boost (0: off)")
    ap.add_argument("--sim", action="store_true",
                    help="fly the turns in BVR_RL's simulator (dcs/fake_dcs.py) instead of DCS")
    ap.add_argument("--platform", default="F-16C-DCS", help="--sim: the simulator platform")
    ap.add_argument("--out", default="dcs_runs")
    args = ap.parse_args(argv)

    import dcs_live
    link = dcs_live.UdpLink()
    fake = th = None
    if args.sim:
        from fake_dcs import FakeDcs
        fake = FakeDcs(opponent="STRAIGHT", seed=3, speed=0.0, platform=args.platform,
                       opp_platform=args.platform, lockstep=True, max_time=3000.0, log=lambda m: None)
        stop = threading.Event()
        th = threading.Thread(target=fake.run, args=(stop,), daemon=True)
        th.start()
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw = open(os.path.join(args.out, f"turn_{stamp}{'_sim' if args.sim else ''}.jsonl"), "w",
               encoding="utf-8")
    rows = []
    try:
        print("waiting for " + ("the simulator" if args.sim else "DCS (start the mission)") + " ...")
        rec, agent, red = dcs_live.wait_for_fight(link, log=print)
        raw.write(json.dumps(rec.header) + "\n")
        bridge = int(rec.header.get("bridge", 1) or 1)
        if bridge < OPT_BRIDGE:
            raise SystemExit(f"this mission's bridge is version {bridge}; the turn test needs "
                             f"{OPT_BRIDGE}. Rebuild it (python dcs\\make_mission.py) and open the new "
                             f"mission in DCS.")
        link.sink = lambda line: raw.write(line + "\n")
        u = rec.units[agent]
        alt = args.alt if args.alt is not None else round(float(u.get("y", 9000.0)) / 100.0) * 100.0
        print(f"{agent}: turns at {args.speed:.0f} m/s and {alt:.0f} m"
              + (f", boost {args.boost:.0f} m/s" if args.boost else "") + f"; {red} holds its fire")
        test = TurnTest(link, agent, rec.t_end, args.variants, args.speed, alt, args.boost or None)
        rows = test.run()
    except KeyboardInterrupt:
        print("\nstopped")
        for _ in range(3):
            link.send("OPT redhold 0")
            link.send("STOP")
    finally:
        link.sink = None
        raw.close()
        link.close()
        if fake is not None:
            stop.set()
            th.join(timeout=10)
            fake.close()
    if rows:
        path = os.path.join(args.out, "turn_test.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow({**{k: r.get(k, "") for k in FIELDS}, "run": stamp,
                            "mode": "sim" if args.sim else "dcs"})
        summary(rows)
        print(f"\nwritten: {path} and {raw.name}")
    return rows


if __name__ == "__main__":
    main()
