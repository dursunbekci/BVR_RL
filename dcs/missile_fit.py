"""
missile_fit.py  —  fit BVR_RL's missile model to DCS missile flights
====================================================================

dcs_live.py and bvr_logger.lua recordings hold every DCS missile's position
and velocity ten times a second. This fits a library missile to them:

  1. Drag. In the coast (after the motor burns out) the only forces along
     the flight path are drag and gravity, so each sample gives the drag
     coefficient directly (with the model's own cross-section and burnt-out
     mass). The medians per Mach band become the drag table.
  2. Motor and loft. Thrust, burn time and the loft (climb angle, and how
     far below the target must be before the dive) are
     searched (Nelder-Mead) so that the model, fired from each recorded
     launch at the recorded target's actual flight path, matches the DCS
     missile's speed and altitude over time.

Then each recorded shot is flown with the starting missile and with the
fitted one, and compared with DCS: speed and altitude at 10 s steps, and
where each model is when the DCS missile hit.

    python dcs\\missile_fit.py dcs_runs\\live_*.jsonl
    python dcs\\missile_fit.py dcs_runs\\*.jsonl --save AIM-120C-DCS

--save writes the fitted missile to library/user/missile/<id>.json (and a
copy of --platform carrying it, see --platform-id), which the trainer, the GUI
and the envelope calibration can then use. Nothing else changes.

Only what the recordings cover is fitted: below the lowest Mach the DCS
missiles flew (about 1.5), the drag table keeps the starting missile's
shape, scaled to meet the fitted curve.
"""

import argparse
import copy
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bvr_library import get_item, load_platform, missile_config, save_item, validate  # noqa: E402
from dcs_world import DcsRecording                                                # noqa: E402
from missile_sim import AIM120, MslPhase, _isa_rho, _isa_a, G                    # noqa: E402

DT = 0.02
MACH_BANDS = np.arange(1.5, 4.51, 0.25)          # lower edges of the drag bands
FIT_KEYS = ("THRUST", "BOOST_TIME", "LOFT_ANGLE", "LOFT_DIVE")       # loft angles in degrees


# ── recorded shots ───────────────────────────────────────────────────
class Shot:
    """One recorded missile flight, with the target's track on the model's time grid."""

    def __init__(self, name, rec, d):
        self.name, self.id = name, d["id"]
        self.shooter, self.target = d["shooter"], d["target"]
        self.t0 = d["t_shot"]
        self.hit = d["t_hit"] is not None and d["hit_target"] == d["target"]
        self.t_end = (d["t_hit"] if self.hit else float(d["t"][-1])) - self.t0
        sh = rec.state(d["shooter"], self.t0)
        self.pos0, self.vel0 = sh["pos"], sh["vel"]
        self.range0 = float(np.linalg.norm(rec.state(d["target"], self.t0)["pos"] - sh["pos"]))
        # DCS missile samples, launch-relative, up to the hit
        k = (d["t"] - self.t0) <= self.t_end + 1e-6
        self.t = d["t"][k] - self.t0
        self.vel = d["vel"][k]
        self.pos = d["pos"][k]
        self.v = np.linalg.norm(self.vel, axis=1)
        # target on the model grid, held after its last sample
        n = int(math.ceil(self.t_end / DT)) + 2
        tt = self.t0 + DT * np.arange(1, n + 1)
        ts = rec.units[d["target"]]["t"]
        tt_c = np.minimum(tt, ts[-1])
        st = [rec.state(d["target"], float(x)) for x in tt_c]
        self.tgt_pos = np.array([s["pos"] for s in st])
        self.tgt_vel = np.array([s["vel"] for s in st])

    def label(self):
        return (f"{self.name} missile {self.id} ({self.shooter} -> {self.target}, "
                f"{self.range0 / 1000:.1f} km, {'hit after %.1f s' % self.t_end if self.hit else 'no hit recorded'})")


def load_shots(paths, min_samples=30):
    shots, seen = [], set()
    for p in paths:
        rec = DcsRecording(p)
        name = os.path.basename(p)
        for d in sorted(rec.weapons.values(), key=lambda d: d["t_shot"] or 0):
            if d["t_shot"] is None or len(d["t"]) < min_samples:
                continue
            if d["shooter"] not in rec.units or d["target"] not in rec.units:
                continue
            s = Shot(name, rec, d)
            # The same mission replayed gives the same shot: count it once.
            key = (s.shooter, round(s.t0 - rec.t_start), round(s.range0 / 200), round(s.t_end))
            if key in seen:
                continue
            seen.add(key)
            shots.append(s)
    return shots


# ── the model ────────────────────────────────────────────────────────
def config(base, **over):
    c = copy.copy(base)
    for k, v in over.items():
        setattr(c, k, v)
    if "DRAG_TABLE" in over:
        t = np.asarray(over["DRAG_TABLE"], dtype=float)
        c.drag_cd = lambda m, xs=t[:, 0], ys=t[:, 1]: float(np.interp(float(m), xs, ys))
    return c


def fly(shot, cfg, handoff=12_000.0):
    """The model fired from the recorded launch at the recorded target. Returns
    (t, speed, altitude, distance to the target at each step, outcome)."""
    m = AIM120(1, 2, shot.pos0.copy(), shot.vel0.copy(), handoff, cfg=cfg)
    T, V, Z, R = [], [], [], []
    out = "flying"
    for i in range(len(shot.tgt_pos)):
        tp, tv = shot.tgt_pos[i], shot.tgt_vel[i]
        m.update_guidance({"valid": 1, "tgt_pos": tp, "tgt_vel": tv})
        ev = m.step(DT, tp, tv)
        T.append((i + 1) * DT); V.append(float(np.linalg.norm(m.vel))); Z.append(float(m.pos[2]))
        R.append(float(np.linalg.norm(tp - m.pos)))
        if m.phase in (MslPhase.HIT, MslPhase.MISS):
            out = ev[0]["type"].replace("MISSILE_", "").lower() if ev else m.phase.value.lower()
            if ev and ev[0].get("cause"):
                out += " (" + ev[0]["cause"].lower() + ")"
            break
    return np.array(T), np.array(V), np.array(Z), np.array(R), out


def errors(shot, run):
    """RMS speed and altitude error against the DCS samples, over the model's flight."""
    T, V, Z = run[0], run[1], run[2]
    k = shot.t <= T[-1]
    if k.sum() < 3:
        return 1e3, 1e4
    v = np.interp(shot.t[k], T, V)
    z = np.interp(shot.t[k], T, Z)
    return (float(np.sqrt(np.mean((v - shot.v[k]) ** 2))),
            float(np.sqrt(np.mean((z - shot.pos[k, 2]) ** 2))))


# ── step 1: drag from the coast ──────────────────────────────────────
def coast_drag(shots, base):
    m_dry = base.MASS0 - base.MASS_PROP
    rows = []
    for s in shots:
        t, v, vel, z = s.t, s.v, s.vel, s.pos[:, 2]
        for i in range(3, len(t) - 3):
            if t[i] < base.BOOST_TIME + 3.0 or t[i] > s.t_end - 1.0:
                continue
            dv = (v[i + 3] - v[i - 3]) / (t[i + 3] - t[i - 3])
            q = 0.5 * _isa_rho(z[i]) * v[i] ** 2
            rows.append((v[i] / _isa_a(z[i]), -(dv + G * vel[i, 2] / v[i]) * m_dry / (q * base.S_REF)))
    rows = np.array(rows)
    bands = []
    for lo in MACH_BANDS:
        sel = (rows[:, 0] >= lo) & (rows[:, 0] < lo + 0.25)
        if sel.sum() >= 10:
            bands.append((float(np.median(rows[sel, 0])), float(np.median(rows[sel, 1])), int(sel.sum())))
    return bands


def drag_table(bands, base):
    """The fitted bands, the starting table below them scaled to meet the lowest
    band, and the last fitted value held above them."""
    lo_m, lo_cd, _ = bands[0]
    scale = lo_cd / base.drag_cd(lo_m)
    below = [[m, round(cd * scale, 4)] for m, cd in base.DRAG_TABLE if m < lo_m - 0.05]
    fitted = [[round(m, 3), round(cd, 4)] for m, cd, _ in bands]
    top = [[6.0, fitted[-1][1]]] if fitted[-1][0] < 6.0 else []
    return below + fitted + top, scale


# ── step 2: motor and loft ───────────────────────────────────────────
def cost(cfg, shots):
    c = 0.0
    for s in shots:
        ev, ez = errors(s, fly(s, cfg))
        c += (ev / 40.0) ** 2 + (ez / 600.0) ** 2
    return c / len(shots)


def nelder_mead(f, x0, step, iters=160, log=None):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for it in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]; vals = [vals[i] for i in order]
        if log and it % 20 == 0:
            log(f"    iteration {it:3d}: cost {vals[0]:.4f}")
        if abs(vals[-1] - vals[0]) < 1e-5:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit(shots, base, log=print):
    bands = coast_drag(shots, base)
    table, scale = drag_table(bands, base)
    log("Drag in the coast, per Mach band (DCS / starting model):")
    for m, cd, n in bands:
        log(f"  Mach {m:4.2f}: {cd:5.3f} / {base.drag_cd(m):5.3f}   ({n} samples)")
    log(f"  below Mach {bands[0][0]:.2f} (not flown in the recordings): starting table x {scale:.2f}")

    lo = {"THRUST": 2000.0, "BOOST_TIME": 1.0, "LOFT_ANGLE": 0.5, "LOFT_DIVE": 0.5}

    def params(x):
        return {k: max(float(v), lo[k]) for k, v in zip(FIT_KEYS, x)}

    def make(p):                             # the model takes the loft angles in radians
        return config(base, DRAG_TABLE=table, THRUST=p["THRUST"], BOOST_TIME=p["BOOST_TIME"],
                      LOFT_ANGLE=math.radians(p["LOFT_ANGLE"]), LOFT_DIVE=math.radians(p["LOFT_DIVE"]))

    x0 = [base.THRUST, base.BOOST_TIME, 20.0, 10.0]
    log("Motor and loft (Nelder-Mead on speed and altitude over time):")
    x, c = nelder_mead(lambda x: cost(make(params(x)), shots), x0,
                       step=[2000.0, 0.8, 8.0, 6.0], log=log)
    p = params(x)
    p = {"THRUST": round(p["THRUST"], 0), "BOOST_TIME": round(p["BOOST_TIME"], 2),
         "LOFT_ANGLE": round(p["LOFT_ANGLE"], 1), "LOFT_DIVE": round(p["LOFT_DIVE"], 1)}
    log(f"  thrust {p['THRUST']:.0f} N, burn {p['BOOST_TIME']:.2f} s, climb at {p['LOFT_ANGLE']:.1f} deg "
        f"until the target is {p['LOFT_DIVE']:.1f} deg below (cost {c:.4f})")
    return make(p), dict(p, DRAG_TABLE=table)


# ── report ───────────────────────────────────────────────────────────
def compare(shots, models, log=print):
    for s in shots:
        log("\n" + s.label())
        runs = {name: fly(s, cfg) for name, cfg in models.items()}
        head = "   time   DCS speed  alt  " + "".join(f"| {n:>10s} speed  alt  " for n in runs)
        log(head)
        for tt in range(5, int(s.t_end) + 1, 10):
            i = min(np.searchsorted(s.t, tt), len(s.t) - 1)
            row = f"  {tt:4d} s  {s.v[i]:6.0f} {s.pos[i, 2]:6.0f}  "
            for name, (T, V, Z, R, out) in runs.items():
                k = np.searchsorted(T, tt)
                row += f"| {'':>10s} {V[k]:5.0f} {Z[k]:6.0f}  " if k < len(T) else f"| {'':>10s} {'-':>5s} {'-':>6s}  "
            log(row)
        for name, (T, V, Z, R, out) in runs.items():
            ev, ez = errors(s, (T, V, Z))
            k = min(np.searchsorted(T, s.t_end), len(T) - 1)
            where = ("ended: " + out) if out != "flying" else "still flying"
            log(f"  {name:>10s}: speed error {ev:4.0f} m/s, altitude error {ez:5.0f} m (RMS); "
                f"{R[k] / 1000:6.2f} km from the target when DCS's {'hit' if s.hit else 'record ended'}; {where} "
                f"after {T[-1]:.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+", help="dcs_live.py / bvr_logger.lua .jsonl files (wildcards work)")
    ap.add_argument("--base", default="AIM-120", help="library missile to start from (default AIM-120)")
    ap.add_argument("--save", metavar="ID", help="save the fitted missile as library/user/missile/ID.json")
    ap.add_argument("--platform", default="F-16C",
                    help="with --save, also save a copy of this platform carrying the fitted missile")
    ap.add_argument("--platform-id", help="its id (default <platform>-<missile id>)")
    args = ap.parse_args()
    paths = [p for pat in args.recordings for p in (glob.glob(pat) or [pat])]
    shots = load_shots(paths)
    if not shots:
        raise SystemExit("no missile flights with enough samples in those recordings")
    print(f"{len(shots)} missile flights from {len(paths)} recordings")
    base = missile_config(args.base)
    cfg, params = fit(shots, base)
    compare(shots, {"start": base, "fitted": cfg})

    if args.save:
        item = copy.deepcopy(get_item("missile", args.base))
        item.pop("builtin", None)
        item["id"] = args.save
        item["name"] = f"{item['name']}, fitted to DCS"
        item["description"] = (f"{args.base} with drag, motor and loft fitted by dcs/missile_fit.py "
                               f"to {len(shots)} DCS missile flights.")
        item["params"].update(params)
        item.setdefault("sources", {}).update({
            k: f"fitted to {len(shots)} DCS flights (dcs/missile_fit.py)" for k in params})
        problems = validate(item)
        if problems:
            raise SystemExit("fitted missile is not valid: " + "; ".join(problems))
        print(f"\nsaved {save_item(item)}")
        if args.platform:
            plat = copy.deepcopy(get_item("platform", args.platform))
            plat.pop("builtin", None)
            plat["id"] = args.platform_id or f"{args.platform}-{args.save}"
            plat["name"] = f"{plat['name']} ({args.save})"
            plat["description"] = f"{args.platform} carrying {args.save}."
            plat["params"]["missile"] = args.save
            print(f"saved {save_item(plat)}")
            print("Its launch envelope is not calibrated yet: calibrate it on the LIBRARY page "
                  "(or with the envelope tool) before training with this platform.")


if __name__ == "__main__":
    main()
