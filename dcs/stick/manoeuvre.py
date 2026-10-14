"""
manoeuvre.py  —  the same manoeuvres in DCS and in the simulator, compared
=========================================================================

The policy flies heading, altitude and speed targets; in training the simulator's autopilot (f16_sim.py)
turns them into bank, g and throttle, in DCS the stick pilot (pilot.py, stick_test.Hold) does. For the
policy to meet in DCS the aircraft it was trained on, the two must fly the same targets the same way:
the same turn rate, climb rate, acceleration and speed bleed.

1. In DCS (stick_test.py manoeuvre): the Viper flies a fixed program of targets with the live runner's
   pilot settings and the simulator's altitude law (flight-path limits of the platform), and every
   tenth of a second is saved: dcs_runs/manoeuvre_<time>.json.

       python dcs/stick/stick_test.py manoeuvre

2. Here (no DCS needed): the simulator's aircraft (a library platform's airframe, F-16C-DCS by default)
   flies the same targets from the same start state, and the two are compared segment by segment, with a
   plot of both.

       python dcs/stick/manoeuvre.py dcs_runs/manoeuvre_<time>.json [--platform F-16C-DCS]

3. Fit (no DCS needed): the airframe's thrust, CD0, K_IND, CL_MAX and transonic drag are adjusted until the
   simulator flies the recorded program closest to DCS, and the result is written to the library as a new
   airframe and platform (F-16C-DCSFIT by default) to train with:

       python dcs/stick/manoeuvre.py dcs_runs/manoeuvre_<time>.json --fit

The program (targets relative to the start): hold 10 s; turn right 90 deg; turn left 180 deg (a long hard
turn: speed bleed); back to the start heading; climb 2000 m; descend 2000 m; accelerate 100 m/s (at most
460); decelerate to 80 m/s below the start (at least 220); turn 170 deg at that speed. About 7 minutes.
"""

import argparse
import datetime
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

D2R = math.pi / 180.0
DIVE_FPA = 25.0 * D2R           # bvr_env's altFPA

# (name, seconds, what changes: dhdg deg from the start heading, dalt m from the start, spd "hi"/"lo"/"start")
PROGRAM = [
    ("hold", 10.0, {}),
    ("turn right 90", 35.0, {"dhdg": 90.0}),
    ("turn left 180", 45.0, {"dhdg": -90.0}),
    ("back to the start heading", 30.0, {"dhdg": 0.0}),
    ("climb 2000 m", 60.0, {"dalt": 2000.0}),
    ("descend 2000 m", 60.0, {"dalt": 0.0}),
    ("accelerate", 70.0, {"spd": "hi"}),
    ("decelerate", 70.0, {"spd": "lo"}),
    ("turn 170 at low speed", 40.0, {"dhdg": 170.0}),
]


def climb_limit(platform_id="F-16C-DCS"):
    """The platform's climb-angle limit (rad) as a function of altitude, as bvr_env sends it (climbFPA)."""
    try:
        from bvr_library import load_platform
        p = load_platform(platform_id)
        return p.climb_fpa
    except Exception:
        def fallback(alt):               # F-16C-DCS's own numbers: 20 deg below 5 km to 10 deg above 11 km
            f = min(max((alt - 5000.0) / 6000.0, 0.0), 1.0)
            return (20.0 - 10.0 * f) * D2R
        return fallback


def targets(start, program=PROGRAM):
    """[(t0, t1, name, hdg rad, alt m, spd m/s)] for a start state {hdg, alt, tas}."""
    hi = min(start["tas"] + 100.0, 460.0)
    lo = max(start["tas"] - 80.0, 220.0)
    out, t = [], 0.0
    hdg, alt, spd = start["hdg"], start["alt"], start["tas"]
    for name, secs, ch in program:
        if "dhdg" in ch:
            hdg = (start["hdg"] + ch["dhdg"] * D2R) % (2 * math.pi)
        if "dalt" in ch:
            alt = start["alt"] + ch["dalt"]
        if "spd" in ch:
            spd = {"hi": hi, "lo": lo, "start": start["tas"]}[ch["spd"]]
        out.append((t, t + secs, name, hdg, alt, spd))
        t += secs
    return out


def target_at(segs, t):
    for s in segs:
        if s[0] <= t < s[1]:
            return s
    return segs[-1]


# ── 1. DCS ────────────────────────────────────────────────────────────

def stage_manoeuvre(run, gains, out_dir="dcs_runs", platform_id="F-16C-DCS", program=PROGRAM):
    """Fly the program in DCS with the live runner's pilot settings; save every 0.1 s."""
    from stick_test import Hold, drive
    from pilot import BANK_MAX_DEG, NZ_MAX, VS_MAX
    climb = climb_limit(platform_id)
    try:
        from bvr_library import load_platform
        airframe = load_platform(platform_id).airframe
    except Exception as e:
        airframe = None
        print(f"  WARNING: platform {platform_id} not loaded ({e}): no g and bank limits from its airframe")
    hold = Hold(gains, bank_max=BANK_MAX_DEG * D2R, nz_max=NZ_MAX, vs_max=VS_MAX, airframe=airframe)
    f0 = run.next()
    start = {"hdg": f0.hdg, "alt": f0.alt, "tas": f0.tas, "mach": f0.mach, "fuel": _fuel(f0)}
    segs = targets(start, program)
    total = segs[-1][1]
    print(f"  start: heading {start['hdg'] / D2R:.0f}, {start['alt']:.0f} m, {start['tas']:.0f} m/s "
          f"(Mach {start['mach']:.2f}); {len(segs)} segments, {total:.0f} s. Hands off; the stick moves.")
    samples, state = [], {"last": -1.0, "seg": None}

    def fn(f, tau):
        s = target_at(segs, tau)
        if s[2] != state["seg"]:
            state["seg"] = s[2]
            print(f"    {tau:5.0f} s  {s[2]}: heading {s[3] / D2R:.0f}, {s[4]:.0f} m, {s[5]:.0f} m/s")
        ax = hold.axes(f, s[3], s[4], v=s[5], dive=DIVE_FPA, climb=climb(f.alt))
        if tau - state["last"] >= 0.1 - 1e-6:
            state["last"] = tau
            samples.append({"t": round(tau, 3), "hdg": f.hdg, "alt": f.alt, "tas": f.tas, "mach": f.mach,
                            "bank": f.bank, "nz": f.nz, "vs": f.vs, "pitch": f.pitch, "fuel": _fuel(f),
                            "thr": ax.get("throttle") if ax else None, "seg": s[2]})
        return ax
    drive(run, total, fn)
    run.link.release()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"manoeuvre_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as fh:
        json.dump({"start": start, "platform": platform_id, "program": program,
                   "pilot": {"bank_max_deg": BANK_MAX_DEG, "nz_max": NZ_MAX, "dive_fpa_deg": DIVE_FPA / D2R},
                   "samples": samples}, fh)
    print(f"\n  saved: {path}\n  compare: python dcs/stick/manoeuvre.py {path}")
    return {"path": path, "samples": len(samples)}


def _fuel(f):
    """Internal fuel from the engine table, as DCS gives it (kg, or a fraction), or None."""
    eng = f.raw.get("eng") or {}
    v = eng.get("fuel_internal")
    return float(v) if isinstance(v, (int, float)) else None


# ── 2. the simulator ──────────────────────────────────────────────────

def simulate(rec, platform_id="F-16C-DCS", fuel_frac=None, cfg=None):
    """The simulator's aircraft flying the recorded program from the recorded start: samples like DCS's."""
    import numpy as np
    from f16_sim import F16Aircraft
    if cfg is None:
        from bvr_library import load_platform
        cfg = load_platform(platform_id).airframe
    climb = climb_limit(platform_id)
    st = rec["start"]
    if fuel_frac is None:
        fu = st.get("fuel")
        fuel_frac = (fu if fu <= 1.0 else fu / cfg.FUEL_FULL) if fu else 0.6
    segs = targets(st, [tuple(p) for p in rec["program"]])
    ac = F16Aircraft(np.random.default_rng(0), cfg=cfg)
    ac.reset_state(0.0, 0.0, st["alt"], st["hdg"], st["tas"], fuel_frac=fuel_frac)
    dt, t, out, last = 0.02, 0.0, [], -1.0
    total = segs[-1][1]
    while t < total:
        s = target_at(segs, t)
        ac.step(dt, {"hdgCmd": s[3], "altTarget": s[4], "V": s[5], "altFPA": DIVE_FPA, "climbFPA": climb(ac.z)})
        t = round(t + dt, 6)
        if t - last >= 0.1 - 1e-6:
            last = t
            out.append({"t": t, "hdg": ac.chi, "alt": ac.z, "tas": ac.V, "mach": ac.mach, "bank": ac.phi,
                        "nz": ac.nz, "vs": ac.V * math.sin(ac.gamma), "thr": ac.throttle, "seg": s[2]})
    return out, fuel_frac


# ── 3. comparison ─────────────────────────────────────────────────────

def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def segment_metrics(samples, seg, prev_target):
    """What one segment shows: by its kind (turn, climb or descent, speed change)."""
    t0, t1, name, hdg, alt, spd = seg
    ss = [s for s in samples if t0 <= s["t"] < t1 and s["tas"] is not None]
    if len(ss) < 5:
        return {}
    m = {}
    first = ss[0]
    d_hdg = _wrap(hdg - first["hdg"])
    if abs(d_hdg) > 20 * D2R:
        go = [s for s in ss if abs(_wrap(hdg - s["hdg"])) > 30 * D2R]
        if len(go) > 2:
            turned = abs(sum(_wrap(b["hdg"] - a["hdg"]) for a, b in zip(go, go[1:])))
            m["turn rate deg/s"] = turned / D2R / max(go[-1]["t"] - go[0]["t"], 0.1)
        hit = next((s["t"] - t0 for s in ss if abs(_wrap(hdg - s["hdg"])) < 0.1 * abs(d_hdg)), None)
        m["time to 90% of the turn s"] = hit
        m["peak bank deg"] = max(abs(s["bank"]) for s in ss if s["bank"] is not None) / D2R
        m["peak g"] = max(s["nz"] for s in ss if s["nz"] is not None)
        m["speed lost m/s"] = first["tas"] - min(s["tas"] for s in ss)
        m["altitude change m"] = max((abs(s["alt"] - first["alt"]) for s in ss), default=0.0)
    d_alt = alt - first["alt"]
    if abs(d_alt) > 300:
        hit = next((s["t"] - t0 for s in ss if abs(alt - s["alt"]) < 0.1 * abs(d_alt)), None)
        m["time to 90% of the height s"] = hit
        mid = [s for s in ss if 0.2 * abs(d_alt) < abs(s["alt"] - first["alt"]) < 0.8 * abs(d_alt)]
        if len(mid) > 2:
            m["vertical speed m/s"] = (mid[-1]["alt"] - mid[0]["alt"]) / max(mid[-1]["t"] - mid[0]["t"], 0.1)
        m["speed change m/s"] = ss[-1]["tas"] - first["tas"]
        over = max((s["alt"] - alt) * (1 if d_alt > 0 else -1) for s in ss)
        m["overshoot m"] = max(over, 0.0)
    d_v = spd - first["tas"]
    if abs(d_v) > 30 and abs(d_hdg) <= 20 * D2R:
        hit = next((s["t"] - t0 for s in ss if abs(spd - s["tas"]) < 0.1 * abs(d_v)), None)
        m["time to 90% of the speed s"] = hit
        mid = [s for s in ss if 0.1 * abs(d_v) < abs(s["tas"] - first["tas"]) < 0.9 * abs(d_v)]
        if len(mid) > 2:
            m["acceleration m/s2"] = (mid[-1]["tas"] - mid[0]["tas"]) / max(mid[-1]["t"] - mid[0]["t"], 0.1)
        m["final speed m/s"] = ss[-1]["tas"]
        m["mach at the end"] = ss[-1]["mach"]
    return m


def compare(rec, sim, out=print):
    segs = targets(rec["start"], [tuple(p) for p in rec["program"]])
    rows = []
    out(f"\n  {'segment / measure':44s} {'DCS':>9s} {'sim':>9s} {'sim/DCS':>8s}")
    for i, seg in enumerate(segs):
        md = segment_metrics(rec["samples"], seg, segs[i - 1] if i else None)
        ms = segment_metrics(sim, seg, segs[i - 1] if i else None)
        if not md and not ms:
            continue
        out(f"  {seg[2]}")
        for k in md.keys() | ms.keys():
            a, b = md.get(k), ms.get(k)
            r = (b / a) if (a not in (None, 0) and b is not None) else None
            rows.append({"segment": seg[2], "measure": k, "dcs": a, "sim": b, "ratio": r})
            fmt = lambda x: "-" if x is None else f"{x:9.2f}"
            out(f"    {k:42s} {fmt(a)} {fmt(b)} {'' if r is None else f'{r:8.2f}'}")
    return rows


def advice(rows):
    """Which simulator parameters the differences point at (the fitting itself is the next step)."""
    def ratio(seg_part, measure):
        rs = [r["ratio"] for r in rows if seg_part in r["segment"] and r["measure"] == measure and r["ratio"]
              and abs(r["dcs"]) > 1.0]           # a ratio to almost nothing says nothing
        return sum(rs) / len(rs) if rs else None
    notes = []
    a = ratio("accelerate", "acceleration m/s2")
    if a and abs(a - 1) > 0.15:
        notes.append(f"acceleration in the simulator is {a:.2f}x DCS's: thrust (T_MIL_SL, T_AB_SL, "
                     f"THRUST_MACH_TABLE) or drag (CD0, DRAG_RISE_*) of the airframe")
    d = ratio("decelerate", "acceleration m/s2")
    if d and abs(d - 1) > 0.15:
        notes.append(f"deceleration in the simulator is {d:.2f}x DCS's: drag at idle (CD0) and idle thrust")
    tr = ratio("turn", "turn rate deg/s")
    if tr and abs(tr - 1) > 0.10:
        notes.append(f"turn rate in the simulator is {tr:.2f}x DCS's: the g the wing gives (CL_MAX, NZ_POS_MAX) "
                     f"or the bank (PHI_MAX against the pilot's 80 deg)")
    sl = ratio("turn left 180", "speed lost m/s")
    if sl and abs(sl - 1) > 0.2:
        notes.append(f"speed lost in a long turn is {sl:.2f}x DCS's: induced drag (K_IND) against thrust")
    vs = ratio("climb", "vertical speed m/s")
    if vs and abs(vs - 1) > 0.15:
        notes.append(f"climb rate in the simulator is {vs:.2f}x DCS's: the climb-angle limit of the platform "
                     f"(CLIMB_FPA_*) or excess thrust")
    return notes or ["no difference above the thresholds (15% for speed changes and climbs, 10% for turn rate)"]


# ── 4. fitting the airframe to DCS ────────────────────────────────────

FIT_PARAMS = [                      # (name, how it is applied, bounds as a factor of the start value)
    ("thrust", "factor on T_MIL_SL and T_AB_SL", (0.6, 1.6)),
    ("CD0", "zero-lift drag", (0.5, 3.5)),
    ("K_IND", "induced drag", (0.5, 2.5)),
    ("CL_MAX", "maximum lift (the g the wing gives)", (0.6, 1.5)),
    ("DRAG_RISE_PEAK", "transonic drag", (0.3, 3.0)),
]


def _cfg_with(base, x):
    """A copy of the airframe config with fit vector x (factors on the base values)."""
    import copy
    c = copy.copy(base)
    c.T_MIL_SL = base.T_MIL_SL * x[0]
    c.T_AB_SL = base.T_AB_SL * x[0]
    c.CD0, c.K_IND, c.CL_MAX = base.CD0 * x[1], base.K_IND * x[2], base.CL_MAX * x[3]
    c.DRAG_RISE_PEAK = base.DRAG_RISE_PEAK * x[4]
    from bvr_library import _CdRise
    c.cd_rise = _CdRise(c.DRAG_RISE_M0, c.DRAG_RISE_M1, c.DRAG_RISE_M2, c.DRAG_RISE_PEAK, c.DRAG_RISE_DECAY)
    return c


def mismatch(rec, sim):
    """How far the simulator's flight is from DCS's: speed (per 5 m/s), heading (per 5 deg) and altitude (per
    50 m) RMS over the whole program, the three summed."""
    import numpy as np
    d = {round(s["t"], 1): s for s in rec["samples"] if s["tas"] is not None}
    pairs = [(d[round(m["t"], 1)], m) for m in sim if round(m["t"], 1) in d]
    if not pairs:
        return float("inf")
    dv = np.array([a["tas"] - b["tas"] for a, b in pairs])
    dh = np.array([_wrap(a["hdg"] - b["hdg"]) / D2R for a, b in pairs])
    da = np.array([a["alt"] - b["alt"] for a, b in pairs])
    return float(np.sqrt(np.mean(dv ** 2)) / 5.0 + np.sqrt(np.mean(dh ** 2)) / 5.0 + np.sqrt(np.mean(da ** 2)) / 50.0)


def nelder_mead(f, x0, lo, hi, step=0.15, iters=120, log=print):
    """A small Nelder-Mead in the box [lo, hi] (no scipy here)."""
    import numpy as np
    n = len(x0)
    clip = lambda x: np.minimum(np.maximum(x, lo), hi)
    pts = [clip(np.array(x0, float))]
    for i in range(n):
        p = np.array(x0, float)
        p[i] *= (1 + step)
        pts.append(clip(p))
    vals = [f(p) for p in pts]
    for k in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if k % 10 == 0:
            log(f"    fit {k:3d}: mismatch {vals[0]:.3f}  " + " ".join(f"{v:.3f}" for v in pts[0]))
        if vals[-1] - vals[0] < 1e-4:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = clip(c + (c - pts[-1]))
        fr = f(xr)
        if fr < vals[0]:
            xe = clip(c + 2 * (c - pts[-1]))
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = clip(c + 0.5 * (pts[-1] - c))
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [clip(pts[0] + 0.5 * (p - pts[0])) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit(rec, platform_id="F-16C-DCS", fuel_frac=None, iters=120, log=print):
    """Factors on the airframe's thrust, CD0, K_IND, CL_MAX and DRAG_RISE_PEAK that make the simulator fly
    the recorded program closest to DCS. Returns (factors, mismatch before, after, fitted config)."""
    import numpy as np
    from bvr_library import load_platform
    base = load_platform(platform_id).airframe
    lo = np.array([b[0] for *_, b in FIT_PARAMS])
    hi = np.array([b[1] for *_, b in FIT_PARAMS])
    f = lambda x: mismatch(rec, simulate(rec, platform_id, fuel_frac, cfg=_cfg_with(base, x))[0])
    x0 = np.ones(len(FIT_PARAMS))
    m0 = f(x0)
    log(f"  fitting {', '.join(n for n, *_ in FIT_PARAMS)} of {platform_id}'s airframe ({base.id}); "
        f"mismatch now {m0:.3f}")
    x, m1 = nelder_mead(f, x0, lo, hi, iters=iters, log=log)
    return x, m0, m1, _cfg_with(base, x)


def save_fit(platform_id, x, new_id, note=""):
    """The fitted airframe and a platform using it, as user library items (new ids)."""
    from bvr_library import get_item, save_item, load_platform
    base_pf = get_item("platform", platform_id)
    base_af = get_item("airframe", base_pf["params"]["airframe"])
    af = dict(base_af, id=new_id, name=f"{base_af.get('name', base_af['id'])}, fitted to DCS", based_on=base_af["id"])
    pr = dict(af["params"])
    pr["T_MIL_SL"] = round(float(pr["T_MIL_SL"]) * x[0], 1)
    pr["T_AB_SL"] = round(float(pr["T_AB_SL"]) * x[0], 1)
    for i, k in ((1, "CD0"), (2, "K_IND"), (3, "CL_MAX"), (4, "DRAG_RISE_PEAK")):
        pr[k] = round(float(pr[k]) * x[i], 5)
    af["params"] = pr
    af["description"] = (f"{base_af['id']} with thrust x{x[0]:.3f}, CD0 x{x[1]:.3f}, K_IND x{x[2]:.3f}, "
                         f"CL_MAX x{x[3]:.3f}, DRAG_RISE_PEAK x{x[4]:.3f}, fitted by dcs/stick/manoeuvre.py to the "
                         f"player's F-16C in DCS flown by the stick pilot. {note}").strip()
    af.pop("builtin", None)
    save_item(af)
    pf = dict(base_pf, id=new_id, name=f"{base_pf.get('name', platform_id)} (airframe fitted to DCS)",
              based_on=platform_id)
    pf["params"] = dict(base_pf["params"], airframe=new_id)
    pf.pop("builtin", None)
    save_item(pf)
    load_platform(new_id)                # it loads
    return new_id


def plot(rec, sim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    segs = targets(rec["start"], [tuple(p) for p in rec["program"]])
    d, s = rec["samples"], sim
    fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    unwrap = lambda xs: [x / D2R for x in __import__("numpy").unwrap([v for v in xs])]
    for ax, key, lab, f in ((axs[0], "hdg", "heading (deg)", unwrap), (axs[1], "alt", "altitude (m)", None),
                            (axs[2], "tas", "true airspeed (m/s)", None), (axs[3], "bank", "bank (deg)", None)):
        for data, style, who in ((d, "-", "DCS"), (s, "--", "simulator")):
            ys = [x[key] for x in data]
            if any(y is None for y in ys):
                continue
            ys = f(ys) if f else ([y / D2R for y in ys] if key == "bank" else ys)
            ax.plot([x["t"] for x in data], ys, style, label=who)
        ax.set_ylabel(lab)
        for seg in segs:
            ax.axvline(seg[0], color="0.85", lw=0.8)
        ax.grid(alpha=0.3)
    tgt_t = [x["t"] for x in d]
    axs[1].plot(tgt_t, [target_at(segs, t)[4] for t in tgt_t], ":", color="0.4", label="target")
    axs[2].plot(tgt_t, [target_at(segs, t)[5] for t in tgt_t], ":", color="0.4", label="target")
    axs[0].legend(loc="best")
    axs[1].legend(loc="best")
    axs[3].set_xlabel("time (s)")
    fig.suptitle(f"DCS F-16C against the simulator ({rec.get('platform')}): the same targets")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("record", help="dcs_runs/manoeuvre_<time>.json from stick_test.py manoeuvre")
    ap.add_argument("--platform", default=None, help="library platform to compare with (default: the record's)")
    ap.add_argument("--fit", action="store_true",
                    help="fit the airframe's thrust, CD0, K_IND, CL_MAX and transonic drag to the record, compare again "
                         "with the fit, and write it to the library as --save-as (a new airframe and platform)")
    ap.add_argument("--save-as", default="F-16C-DCSFIT", help="--fit: id of the new airframe and platform")
    ap.add_argument("--iters", type=int, default=150, help="--fit: optimiser iterations (default 150; about 5 minutes)")
    ap.add_argument("--fuel-frac", type=float, default=None,
                    help="the simulator's fuel at the start, a fraction of full (default: DCS's, else 0.6)")
    a = ap.parse_args(argv)
    with open(a.record) as fh:
        rec = json.load(fh)
    pid = a.platform or rec.get("platform") or "F-16C-DCS"
    sim, ff = simulate(rec, pid, a.fuel_frac)
    st = rec["start"]
    print(f"  start: heading {st['hdg'] / D2R:.0f}, {st['alt']:.0f} m, {st['tas']:.0f} m/s; DCS fuel "
          f"{st.get('fuel')}, simulator fuel {ff:.2f} of full; platform {pid}")
    rows = compare(rec, sim)
    print(f"\n  mismatch (speed RMS / 5 m/s + heading RMS / 5 deg + altitude RMS / 50 m): {mismatch(rec, sim):.3f}")
    print("\n  what the differences point at:")
    for n in advice(rows):
        print("    - " + n)
    base = os.path.splitext(a.record)[0]
    plot(rec, sim, base + "_compare.png")
    out = {"platform": pid, "fuel_frac": ff, "rows": rows}
    if a.fit:
        x, m0, m1, cfg = fit(rec, pid, a.fuel_frac, iters=a.iters)
        print(f"\n  fitted: mismatch {m0:.3f} -> {m1:.3f}")
        for (name, what, _), v in zip(FIT_PARAMS, x):
            print(f"    {name:15s} x{v:.3f}   ({what})")
        sim2, _ = simulate(rec, pid, a.fuel_frac, cfg=cfg)
        print("\n  with the fit:")
        rows2 = compare(rec, sim2)
        plot(rec, sim2, base + "_fit.png")
        new = save_fit(pid, x, a.save_as, note=f"Record {os.path.basename(a.record)}; mismatch {m0:.3f} -> {m1:.3f}.")
        print(f"\n  saved to the library: airframe and platform {new} (train with --platform {new})")
        out.update(fit={n: float(v) for (n, *_), v in zip(FIT_PARAMS, x)}, mismatch=[m0, m1], rows_fit=rows2,
                   saved_as=new)
    with open(base + "_compare.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n  written: {base}_compare.png{', ' + base + '_fit.png' if a.fit else ''}, {base}_compare.json")
    return out


if __name__ == "__main__":
    main()
