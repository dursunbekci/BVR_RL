"""
stick_test.py  --  test telemetry and stick control of the player's F-16C through Export.lua
===========================================================================================

Talks to bvr_stick_export.lua (installed with dcs/stick/install_export.py) over
UDP on this computer: telemetry arrives on 15401, commands leave on 15402.

Fly the F-16C yourself first: take off, climb to 5-8 km, level out at about
300 m/s (Mach 0.9) over flat ground or sea, trim the aircraft, then take your
hands OFF the stick and throttle (or unassign the physical axes in DCS: a
joystick that is still sending can fight the commands). Then:

    python dcs/stick/stick_test.py info                see what Export.lua found
    python dcs/stick/stick_test.py rate                telemetry rate and fields (no control)
    python dcs/stick/stick_test.py pulses              small stick pulses: signs, gains, lag
    python dcs/stick/stick_test.py throttle            throttle code against RPM and acceleration
    python dcs/stick/stick_test.py hold                heading/altitude/speed steps, closed loop
    python dcs/stick/stick_test.py all                 rate, pulses, throttle, hold
    python dcs/stick/stick_test.py release             panic: axes to 0 (the F-16 holds 1 g, the bank)
    python dcs/stick/stick_test.py raw 2004 0.5        one LoSetCommand(2004, 0.5)

    python dcs/stick/stick_test.py all --fake          the same against a synthetic aircraft

Safety. The tool stops, zeroes the axes and tells you why if: the bank passes
85 deg, the pitch 50 deg, the height above ground drops below --min-agl, the
speed drops under 110 m/s, the load factor leaves -3..8.5 g, or telemetry stops
for 3 s. bvr_stick_export.lua zeroes the axes on its own half a second after the
last command, so a crashed or Ctrl-C'd script releases the stick too. You can
always take the stick back by moving your own.

Output: dcs_runs/stick_<time>_<test>.jsonl (every frame), dcs_runs/stick_gains.json
(what `pulses` and `throttle` measured: `hold` reads it), dcs_runs/stick_summary_<time>.json.
"""

import argparse
import datetime
import json
import math
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TEL_PORT, CMD_PORT = 15401, 15402
G = 9.80665
DEG = 180.0 / math.pi


class Abort(Exception):
    pass


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _first(*vals):
    for v in vals:
        v = _num(v)
        if v is not None:
            return v
    return None


def _rpm(eng):
    if not isinstance(eng, dict):
        return None
    r = eng.get("RPM")
    if isinstance(r, dict):
        vals = [_num(r.get(k)) for k in ("left", "right")]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    return _num(r)


class Frame:
    """One telemetry line, in SI units and radians as DCS sends them."""

    def __init__(self, d):
        self.raw = d
        me, adi, g, ax = d.get("me") or {}, d.get("adi") or {}, d.get("g") or {}, d.get("ax") or {}
        self.n, self.rt = d.get("n"), _num(d.get("rt"))
        self.t = _first(d.get("t"), self.rt)
        self.alt = _first(d.get("asl"), me.get("alt"))
        self.agl = _num(d.get("agl"))
        self.tas, self.ias, self.mach = _num(d.get("tas")), _num(d.get("ias")), _num(d.get("mach"))
        self.hdg = _first(me.get("hdg"), adi.get("yaw"))
        self.pitch = _first(me.get("pitch"), adi.get("pitch"))
        self.bank = _first(me.get("bank"), adi.get("bank"))
        self.nz = _num(g.get("y"))
        self.aoa, self.beta, self.vs = _num(d.get("aoa")), _num(d.get("beta")), _num(d.get("vs"))
        self.rpm = _rpm(d.get("eng"))
        self.x, self.z = _num(me.get("x")), _num(me.get("z"))
        self.vel = d.get("vel") or {}
        self.ax = ax
        self.ax_seq = ax.get("seq")


class UdpLink:
    def __init__(self, tel_port=TEL_PORT, cmd_port=CMD_PORT, host="127.0.0.1"):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.rx.bind((host, tel_port))
        except OSError as e:
            raise SystemExit(f"cannot listen on {host}:{tel_port} ({e}). Another stick_test.py running?")
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, cmd_port)
        self.events, self.sent, self.latencies = {}, {}, []
        self.seq = self.commands_sent = 0

    def wall(self):
        return time.time()

    def next(self, timeout=1.0):
        end = time.time() + timeout
        while True:
            left = end - time.time()
            if left <= 0:
                return None
            self.rx.settimeout(left)
            try:
                data, _ = self.rx.recvfrom(65535)
            except socket.timeout:
                return None
            try:
                d = json.loads(data)
            except ValueError:
                continue
            if d.get("ev") == "tel":
                f = Frame(d)
                if f.ax_seq in self.sent and f.rt is not None:
                    self.latencies.append(f.rt - self.sent.pop(f.ax_seq))
                return f
            self.events[d.get("ev")] = d

    def _send(self, text):
        self.tx.sendto(text.encode(), self.addr)

    def axes(self, pitch=None, roll=None, rudder=None, throttle=None):
        self.seq += 1
        self.commands_sent += 1
        self.sent[self.seq] = time.time()
        if len(self.sent) > 500:
            self.sent.pop(min(self.sent))

        def s(v):
            return "-" if v is None else f"{max(-1.0, min(1.0, v)):.4f}"
        self._send(f"AXES {self.seq} {s(pitch)} {s(roll)} {s(rudder)} {s(throttle)}")

    def release(self):
        for _ in range(3):
            self._send("RELEASE")

    def raw(self, code, value):
        self._send(f"CMD {code} {value}")

    def info(self):
        self._send("INFO")

    def close(self):
        self.rx.close()
        self.tx.close()


# ── the run: frames in, limits checked, everything saved ────────────────

class Limits:
    def __init__(self, min_agl=1500.0, min_alt=2500.0):
        self.bank, self.pitch = math.radians(85), math.radians(50)
        self.min_agl, self.min_alt, self.tas_min = min_agl, min_alt, 110.0
        self.nz_lo, self.nz_hi = -3.0, 8.5


class Run:
    def __init__(self, link, out_dir, tag, limits):
        self.link, self.limits, self.tag = link, limits, tag
        self.frames = []
        self.fh = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            self.path = os.path.join(out_dir, f"stick_{datetime.datetime.now():%Y%m%d_%H%M%S}_{tag}.jsonl")
            self.fh = open(self.path, "w")

    def next(self):
        f = self.link.next(timeout=3.0)
        if f is None:
            raise Abort("no telemetry for 3 s (sim paused? DCS closed? Export.lua not loaded?)")
        self.frames.append(f)
        if self.fh:
            self.fh.write(json.dumps(f.raw, separators=(",", ":")) + "\n")
        self.check(f)
        return f

    def check(self, f):
        L = self.limits
        if f.bank is not None and abs(f.bank) > L.bank:
            raise Abort(f"bank {f.bank * DEG:.0f} deg")
        if f.pitch is not None and abs(f.pitch) > L.pitch:
            raise Abort(f"pitch {f.pitch * DEG:.0f} deg")
        if f.agl is not None and f.agl < L.min_agl:
            raise Abort(f"{f.agl:.0f} m above ground (limit {L.min_agl:.0f})")
        if f.agl is None and f.alt is not None and f.alt < L.min_alt:
            raise Abort(f"altitude {f.alt:.0f} m (limit {L.min_alt:.0f}; no height above ground sent)")
        if f.tas is not None and f.tas < L.tas_min:
            raise Abort(f"speed {f.tas:.0f} m/s")
        if f.nz is not None and not (L.nz_lo <= f.nz <= L.nz_hi):
            raise Abort(f"load factor {f.nz:.1f} g")

    def close(self):
        if self.fh:
            self.fh.close()


def drive(run, duration, fn):
    """Each frame for `duration` s of model time: send the axes fn(frame, tau) returns.
    Returns the frames of that period."""
    f = run.next()
    k0, t0 = len(run.frames) - 1, f.t
    while f.t - t0 < duration:
        ax = fn(f, f.t - t0)
        if ax is not None:
            run.link.axes(**ax)
        f = run.next()
    return run.frames[k0:]


ZERO = {"pitch": 0.0, "roll": 0.0, "rudder": 0.0}


def series(frames, attr):
    return np.array([getattr(f, attr) if getattr(f, attr) is not None else np.nan for f in frames], float)


def deriv(t, y, wrap=False, smooth=5):
    """dy/dt by central differences over `smooth` frames (angles unwrapped when `wrap`)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if wrap:
        y = np.unwrap(y)
    k = max(1, smooth // 2)
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        a, b = max(0, i - k), min(len(y) - 1, i + k)
        if b > a and t[b] > t[a]:
            out[i] = (y[b] - y[a]) / (t[b] - t[a])
    return out


def pct(a, p):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, p)) if len(a) else float("nan")


# ── the control law: the simulator's autopilot, on stick axes ───────────

class Hold:
    """Heading, altitude and speed hold: heading -> bank -> roll rate -> roll axis; altitude ->
    flight-path angle -> load factor -> pitch axis; speed -> acceleration -> throttle axis.
    The same structure as f16_sim.py's autopilot, with the gains `pulses` and `throttle` measured."""
    K_HDG_BANK = 2.0       # bank per rad of heading error (the simulator's K_HDG_PHI)
    K_BANK_RATE = 2.5      # roll rate per rad of bank error, 1/s
    TAU_GAMMA = 3.0        # s to take out a flight-path error
    TAU_SPEED = 8.0        # s to take out a speed error

    def __init__(self, gains, bank_max=math.radians(45), nz_max=4.0):
        self.g, self.bank_max, self.nz_max = gains, bank_max, nz_max
        self.prev = None

    def axes(self, f, hdg, alt, v=None, thr=None):
        g = self.g
        vs = f.vs
        if vs is None:
            vs = 0.0 if self.prev is None or f.t <= self.prev[0] else (f.alt - self.prev[1]) / (f.t - self.prev[0])
        self.prev = (f.t, f.alt)
        tas = max(f.tas or 150.0, 80.0)
        bank_cmd = max(-self.bank_max, min(self.bank_max, self.K_HDG_BANK * wrap_pi(hdg - f.hdg)))
        rate_cmd = max(-3.0, min(3.0, self.K_BANK_RATE * (bank_cmd - f.bank)))
        roll = g["roll_sign"] * rate_cmd / g["roll_rate"]
        gam = math.asin(max(-0.9, min(0.9, vs / tas)))
        vs_cmd = max(-30.0, min(30.0, 0.12 * (alt - f.alt)))
        gam_cmd = math.asin(max(-0.9, min(0.9, vs_cmd / tas)))
        nz_cmd = (math.cos(gam) + tas / (G * self.TAU_GAMMA) * (gam_cmd - gam)) / max(math.cos(f.bank), 0.3)
        nz_cmd = max(-1.0, min(self.nz_max, nz_cmd))
        pitch = g["pitch_sign"] * (nz_cmd - 1.0) / g["nz_per_cmd"]
        if v is not None:
            thr = g.get("thr_trim", 0.0) + ((v - tas) / self.TAU_SPEED) / g.get("thr_slope", 6.0)
        out = {"pitch": max(-1.0, min(1.0, pitch)), "roll": max(-1.0, min(1.0, roll)), "rudder": 0.0}
        if thr is not None:
            out["throttle"] = max(-1.0, min(1.0, thr))
        return out


# ── stages ──────────────────────────────────────────────────────────────

def stage_info(link, secs=3.0):
    link.info()
    end = time.time() + secs
    while time.time() < end and "hello" not in link.events:
        link.next(timeout=0.5)
    h = link.events.get("hello")
    if not h:
        print("no hello from Export.lua: is DCS running a mission, with the player in the aircraft, "
              "and is the line in Export.lua? (python dcs/stick/install_export.py --check)")
        return None
    print(f"bvr_stick_export.lua version {h.get('version')}, telemetry source '{h.get('source')}' "
          f"(per frame up to {h.get('max_rate')}/s, or {h.get('rate')}/s by event), watchdog {h.get('watchdog')} s")
    print(f"axis commands: {h.get('codes')}  (from {h.get('codes_from')})")
    api = h.get("api") or {}
    gone = sorted(k for k, v in api.items() if not v)
    print("DCS API missing: " + (", ".join(gone) if gone else "none"))
    return h


def stage_rate(run, secs=10.0):
    link = run.link
    link.info()
    f0 = run.next()
    t_end = link.wall() + secs
    while link.wall() < t_end:
        run.next()
    fr = run.frames
    rt = series(fr, "rt")
    tm = series(fr, "t")
    n = len(fr)
    dt = np.diff(rt)
    real_hz = (n - 1) / (rt[-1] - rt[0]) if rt[-1] > rt[0] else float("nan")
    model_hz = (n - 1) / (tm[-1] - tm[0]) if tm[-1] > tm[0] else float("nan")
    out = {"frames": n, "real_hz": real_hz, "model_hz": model_hz,
           "dt_ms_p50": 1e3 * pct(dt, 50), "dt_ms_p95": 1e3 * pct(dt, 95), "dt_ms_max": 1e3 * float(np.nanmax(dt)),
           "gaps_over_150ms": int(np.sum(dt > 0.15)),
           "model_over_real": (tm[-1] - tm[0]) / (rt[-1] - rt[0]) if rt[-1] > rt[0] else float("nan")}
    cover = {k: float(np.mean(~np.isnan(series(fr, k)))) for k in
             ("alt", "agl", "tas", "mach", "hdg", "pitch", "bank", "nz", "vs", "aoa", "rpm")}
    out["coverage"] = cover
    checks = []

    def chk(ok, text):
        checks.append((ok, text))

    h0 = link.events.get("hello") or {}
    chk(real_hz >= 20, f"telemetry {real_hz:.1f} lines/s real time ({model_hz:.1f}/s of model time, source "
                       f"'{h0.get('source')}'); a 20 Hz inner loop needs at least 20"
                       + ("" if real_hz >= 20 else " -- try the other source (see the comment in Export.lua)"))
    chk(out["dt_ms_p95"] < 2.5e3 / max(real_hz, 1), f"frame spacing p50 {out['dt_ms_p50']:.1f} ms, "
        f"p95 {out['dt_ms_p95']:.1f}, max {out['dt_ms_max']:.1f}; {out['gaps_over_150ms']} gaps over 150 ms")
    chk(abs(out["model_over_real"] - 1.0) < 0.05, f"model time runs {out['model_over_real']:.2f} x real time "
                                                  f"(1.00 = normal; pause or time acceleration breaks the controller)")
    miss = [k for k, v in cover.items() if v < 0.99]
    chk(not miss, "fields present: " + ("all" if not miss else "MISSING " + ", ".join(
        f"{k} ({cover[k] * 100:.0f}%)" for k in miss)))
    hdg, bank, nz = series(fr, "hdg"), series(fr, "bank"), series(fr, "nz")
    if not np.all(np.isnan(hdg)):
        chk(np.nanmax(np.abs(hdg)) <= 2 * math.pi + 0.1, f"heading looks like radians (max {np.nanmax(hdg):.2f})")
    if not np.all(np.isnan(bank)):
        chk(np.nanmax(np.abs(bank)) <= math.pi + 0.1, f"bank looks like radians (max |bank| {np.nanmax(np.abs(bank)):.2f})")
    lvl = np.abs(np.nan_to_num(bank, nan=9.0)) < math.radians(15)
    if np.any(lvl) and not np.all(np.isnan(nz)):
        m = float(np.nanmedian(nz[lvl]))
        chk(0.7 < m < 1.4, f"load factor in level flight {m:.2f} g (expected about 1)")
    f = fr[-1]
    if f.tas and f.mach:
        a = f.tas / f.mach
        chk(250 < a < 350, f"TAS / Mach = {a:.0f} m/s (speed of sound; TAS in m/s as expected)")
    v = f.vel
    if f.tas and all(isinstance(v.get(k), (int, float)) for k in ("x", "y", "z")):
        spd = math.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
        chk(abs(spd - f.tas) < 0.15 * f.tas + 15, f"|velocity vector| {spd:.0f} vs TAS {f.tas:.0f} m/s")
    st = link.events.get("stats")
    if st:
        out["lua_stats"] = {k: st.get(k) for k in ("tel_hz", "frame_hz", "event_hz", "frame_dt_max", "event_dt_max", "missing")}
        chk(True, f"in DCS: {st.get('tel_hz', 0):.1f} telemetry lines/s, {st.get('event_hz', 0):.1f} activity events/s, "
                  f"{st.get('frame_hz', 0):.1f} rendered frames/s; longest frame gap {1e3 * (st.get('frame_dt_max') or 0):.0f} ms; "
                  f"API errors: {st.get('missing') or 'none'}")
    h = link.events.get("hello")
    if h:
        out["codes"] = h.get("codes")
        chk(True, f"axis commands {h.get('codes')} from {h.get('codes_from')}")
    print()
    for ok, text in checks:
        print(("  ok    " if ok else "  WARN  ") + text)
    out["checks_failed"] = [t for ok, t in checks if not ok]
    print(f"\n  now: {f.alt:.0f} m, {f.tas:.0f} m/s (Mach {f.mach:.2f}), heading {f.hdg * DEG:.0f} deg, "
          f"bank {f.bank * DEG:+.1f}, nz {f.nz:.2f}, RPM {f.rpm}")
    return out


def _window(frames, t0, t1):
    return [f for f in frames if t0 <= f.t < t1]


def stage_pulses(run, size=0.1, on=0.5):
    """Pulse each axis both ways from level flight; measure the response and the lag."""
    link = run.link
    seq = [("pitch", 1), ("pitch", -1), ("roll", 1), ("roll", -1), ("rudder", 1), ("rudder", -1)]
    amp = {"pitch": size, "roll": size, "rudder": 2 * size}
    resp = {}
    print(f"\n  pulses of {size:.2f} (rudder {2 * size:.2f}) for {on} s, then 3 s of zero")
    for axis, sgn in seq:
        pre = drive(run, 2.5, lambda f, tau: dict(ZERO))
        t_on = pre[-1].t
        drive(run, on, lambda f, tau: {**ZERO, axis: sgn * amp[axis]})
        drive(run, 3.0, lambda f, tau: dict(ZERO))
        fr = run.frames
        base = _window(fr, t_on - 0.6, t_on)
        win = _window(fr, t_on, t_on + on + 0.8)
        ts = np.array([f.t for f in fr])
        if axis == "pitch":
            y = series(fr, "nz")
            sig = np.array([y[i] for i, f in enumerate(fr) if f in win])
            b = np.nanmean(series(base, "nz"))
            d = sig - b
        elif axis == "roll":
            r = deriv(ts, series(fr, "bank"), wrap=True)
            sig = np.array([r[i] for i, f in enumerate(fr) if f in win])
            b = np.nanmean([r[i] for i, f in enumerate(fr) if f in base])
            d = sig - b
        else:
            r = deriv(ts, series(fr, "hdg"), wrap=True)
            sig = np.array([r[i] for i, f in enumerate(fr) if f in win])
            b = np.nanmean([r[i] for i, f in enumerate(fr) if f in base])
            d = sig - b
        peak = float(d[np.nanargmax(np.abs(d))])
        hit = np.where(np.abs(d) > 0.1 * abs(peak))[0]
        lag = float(win[hit[0]].t - t_on) if len(hit) and abs(peak) > 0 else float("nan")
        resp[(axis, sgn)] = (peak, lag)
    g = {"size": size}
    lines = []
    for axis in ("pitch", "roll", "rudder"):
        (pp, lp), (pm, lm) = resp[(axis, 1)], resp[(axis, -1)]
        gain = (pp - pm) / (2 * amp[axis])
        lin = -pp / pm if pm else float("nan")
        lag = np.nanmean([lp, lm])
        unit = {"pitch": "g", "roll": "deg/s", "rudder": "deg/s"}[axis]
        k = 1.0 if axis == "pitch" else DEG
        lines.append((axis, amp[axis], pp * k, pm * k, unit, lin, lag, gain))
        g[f"{axis}_sign"] = 1 if gain >= 0 else -1
        g[f"{axis}_lag_s"] = float(lag)
        if axis == "pitch":
            g["nz_per_cmd"] = abs(gain)
        elif axis == "roll":
            g["roll_rate"] = abs(gain)
        else:
            g["rudder_yaw_rate"] = abs(gain)
    print(f"\n  {'axis':<7}{'cmd':>6}{'+ pulse':>14}{'- pulse':>14}{'+/-':>5}  {'lag':>6}   result")
    for axis, a, pp, pm, unit, lin, lag, gain in lines:
        what = {"pitch": "more g" if gain > 0 else "LESS g", "roll": "bank up" if gain > 0 else "bank DOWN",
                "rudder": "heading up" if gain > 0 else "heading DOWN"}[axis]
        per = f"{abs(gain):.2f} {'g' if axis == 'pitch' else 'rad/s'} per unit"
        print(f"  {axis:<7}{a:>6.2f}{pp:>+8.2f} {unit:<5}{pm:>+8.2f} {unit:<5}{lin:>5.2f}  {1e3 * lag:>4.0f}ms   "
              f"positive = {what}; {per}")
    if g["pitch_sign"] < 0:
        print("  note: a positive pitch command gave LESS g: the pitch axis is stick-forward-positive")
    return g


def stage_throttle(run, gains, hold_s=7.0, bank_max=math.radians(30)):
    """Throttle code against RPM and acceleration, holding heading and altitude on the stick."""
    link = run.link
    need = ("roll_sign", "roll_rate", "pitch_sign", "nz_per_cmd")
    if not all(k in gains for k in need):
        raise SystemExit("throttle needs the stick gains: run `pulses` first (or `all`).")
    f0 = run.next()
    hdg, alt = f0.hdg, f0.alt
    h = Hold(gains, bank_max=bank_max)
    levels = [0.0, -1.0, -0.5, 0.0, 0.5, 1.0]

    def law(thr):
        return lambda f, tau: h.axes(f, hdg, alt, thr=thr)
    print(f"\n  holding heading {hdg * DEG:.0f} deg and {alt:.0f} m on the stick; throttle code for {hold_s:.0f} s each")
    rows = []
    for thr in levels:
        fr = drive(run, hold_s, law(thr))
        tail = _window(fr, fr[0].t + hold_s - 5.0, fr[-1].t + 1e-6)
        ts, tas = series(tail, "t"), series(tail, "tas")
        acc = float(np.polyfit(ts - ts[0], tas, 1)[0]) if len(tail) > 5 else float("nan")
        rpm = float(np.nanmean(series(_window(fr, fr[-1].t - 2.0, fr[-1].t + 1e-6), "rpm")))
        rows.append((thr, rpm, acc, float(tail[-1].tas), float(tail[-1].alt - alt)))
    print(f"\n  {'code':>6}{'RPM %':>9}{'accel m/s2':>12}{'TAS':>8}{'alt err':>9}")
    for thr, rpm, acc, tas, da in rows[1:]:
        print(f"  {thr:>+6.2f}{rpm:>9.1f}{acc:>+12.2f}{tas:>8.0f}{da:>+9.0f}")
    xs, ys, rp = (np.array([r[i] for r in rows[1:]]) for i in (0, 2, 1))
    slope, icpt = np.polyfit(xs, ys, 1)
    out = {"thr_slope": float(slope), "thr_trim": float(np.clip(-icpt / slope, -1, 1)) if slope > 0 else 0.0,
           "rpm_by_code": {f"{r[0]:+.2f}": r[1] for r in rows[1:]}}
    if slope <= 0 or rp[-1] <= rp[0]:
        print("  WARN: more throttle code did not give more RPM / acceleration: code 2004 may not be the throttle, "
              "or a physical throttle is overriding it")
    else:
        print(f"\n  acceleration {slope:.2f} m/s2 per code unit; level flight at code {out['thr_trim']:+.2f} "
              f"(RPM {np.interp(out['thr_trim'], xs, rp):.0f}%)")
    drive(run, 3.0, lambda f, tau: h.axes(f, hdg, alt, thr=out["thr_trim"]))
    return out


def _steps(fr, t_cmd, hdg0, delta, span=22.0):
    """Response of a heading step commanded at model time t_cmd, `delta` rad from hdg0."""
    seg = [f for f in fr if t_cmd <= f.t < t_cmd + span]
    ts = np.array([f.t for f in seg]) - t_cmd
    h = np.unwrap([f.hdg for f in seg])
    d = (h - h[0]) * np.sign(delta)
    tgt = abs(delta)
    half = ts[np.argmax(d >= 0.5 * tgt)] if np.any(d >= 0.5 * tgt) else float("nan")
    nin = ts[np.argmax(d >= 0.9 * tgt)] if np.any(d >= 0.9 * tgt) else float("nan")
    rate = deriv(ts, h * np.sign(delta))
    bulk = (tgt - d > math.radians(30)) & (ts > 1.5)
    bank = np.array([f.bank for f in seg]) * np.sign(delta)
    return {"delta_deg": delta * DEG, "half_s": float(half), "p90_s": float(nin),
            "rate_bulk_dps": float(np.nanmean(rate[bulk]) * DEG) if np.any(bulk) else float("nan"),
            "peak_bank_deg": float(np.nanmax(bank) * DEG), "peak_nz": float(np.nanmax([f.nz for f in seg]))}


def stage_hold(run, gains, bank_deg=60.0, nz_max=5.0, turn=60.0, speed_step=40.0, alt_step=300.0):
    need = ("roll_sign", "roll_rate", "pitch_sign", "nz_per_cmd")
    if not all(k in gains for k in need):
        raise SystemExit("hold needs the stick gains: run `pulses` first (or `all`).")
    link = run.link
    f0 = run.next()
    hdg0, alt0, v0 = f0.hdg, f0.alt, f0.tas
    h = Hold(gains, bank_max=math.radians(bank_deg), nz_max=nz_max)
    d = math.radians(turn)
    T1, T2, T3, T4, T5 = 8.0, 38.0, 68.0, 96.0, 120.0
    plan = [(0.0, hdg0, alt0, v0), (T1, hdg0 + d, alt0, v0), (T2, hdg0, alt0, v0),
            (T3, hdg0, alt0 + alt_step, v0), (T4, hdg0, alt0 + alt_step, v0 + speed_step),
            (T5, hdg0, alt0 + alt_step, v0 + speed_step)]
    print(f"\n  start: heading {hdg0 * DEG:.0f}, {alt0:.0f} m, {v0:.0f} m/s. Plan: +{turn:.0f} deg at {T1:.0f} s, back at "
          f"{T2:.0f} s, +{alt_step:.0f} m at {T3:.0f} s, +{speed_step:.0f} m/s at {T4:.0f} s. "
          f"Bank limit {bank_deg:.0f} deg, {nz_max:.1f} g.")
    t_start = f0.t
    cur = {"i": 0}

    def law(f, tau):
        while cur["i"] + 1 < len(plan) - 1 and tau >= plan[cur["i"] + 1][0]:
            cur["i"] += 1
        _, hd, al, v = plan[cur["i"]]
        return h.axes(f, hd, al, v=v)
    fr = drive(run, plan[-1][0], law)
    res = {"heading_steps": [_steps(fr, t_start + T1, hdg0, d, span=T2 - T1),
                             _steps(fr, t_start + T2, hdg0 + d, -d, span=T3 - T2)]}
    bank = series(fr, "bank")
    hdg = np.unwrap(series(fr, "hdg"))
    ts = series(fr, "t")
    mid = (ts - t_start > T1 + 4) & (ts - t_start < T2 - 2)
    if np.any(mid) and abs(np.nanmean(bank[mid])) > math.radians(10):
        turning = np.nanmean(deriv(ts, hdg)[mid])
        if turning * np.nanmean(bank[mid]) < 0:
            raise Abort("the heading turned the opposite way to the bank: telemetry bank sign is opposite to this "
                        "tool's assumption (positive bank = right wing down)")
    seg = [f for f in fr if t_start + T3 <= f.t < t_start + T4]
    alt_t = np.array([f.t for f in seg]) - (t_start + T3)
    alt_a = np.array([f.alt for f in seg]) - alt0
    res["alt_step"] = {"step_m": alt_step, "t90_s": float(alt_t[np.argmax(alt_a >= 0.9 * alt_step)])
                       if np.any(alt_a >= 0.9 * alt_step) else float("nan"),
                       "overshoot_m": float(max(0.0, np.nanmax(alt_a) - alt_step))}
    seg = [f for f in fr if t_start + T4 <= f.t < t_start + T5]
    sv = np.array([f.tas for f in seg]) - v0
    res["speed_step"] = {"step_ms": speed_step, "end_error": float(sv[-1] - speed_step),
                         "t90_s": float(np.array([f.t for f in seg])[np.argmax(sv >= 0.9 * speed_step)] - (t_start + T4))
                         if np.any(sv >= 0.9 * speed_step) else float("nan")}
    quiet = [f for f in fr if f.t - t_start < 8.0]
    res["hold_error"] = {"alt_m": float(np.nanmax(np.abs(series(quiet, "alt") - alt0))),
                         "hdg_deg": float(np.nanmax(np.abs([wrap_pi(f.hdg - hdg0) for f in quiet])) * DEG),
                         "speed_ms": float(np.nanmax(np.abs(series(quiet, "tas") - v0)))}
    lat = link.latencies
    res["loop"] = {"commands": link.commands_sent, "command_to_frame_ms_p50": 1e3 * pct(lat, 50),
                   "p95": 1e3 * pct(lat, 95)}
    print(f"\n  {'step':<10}{'turn':>7}{'half':>8}{'90%':>8}{'deg/s':>8}{'bank':>7}{'nz':>6}")
    for i, s in enumerate(res["heading_steps"]):
        print(f"  {'turn ' + str(i + 1):<10}{s['delta_deg']:>+6.0f}d{s['half_s']:>7.1f}s{s['p90_s']:>7.1f}s"
              f"{s['rate_bulk_dps']:>8.1f}{s['peak_bank_deg']:>6.0f}d{s['peak_nz']:>6.1f}")
    a, s_, e, lp = res["alt_step"], res["speed_step"], res["hold_error"], res["loop"]
    print(f"  altitude +{a['step_m']:.0f} m: 90% in {a['t90_s']:.1f} s, overshoot {a['overshoot_m']:.0f} m")
    print(f"  speed +{s_['step_ms']:.0f} m/s: 90% in {s_['t90_s']:.1f} s, final error {s_['end_error']:+.1f} m/s")
    print(f"  steady hold, first 8 s: altitude {e['alt_m']:.0f} m, heading {e['hdg_deg']:.1f} deg, speed {e['speed_ms']:.1f} m/s")
    print(f"  commands sent {lp['commands']}; command to the frame that shows it: p50 "
          f"{lp['command_to_frame_ms_p50']:.0f} ms, p95 {lp['p95']:.0f} ms")
    print("  the simulator's F-16C-DCS turns at about 8 deg/s (80 deg of bank) at 340 m/s")
    return res


# ── command line ────────────────────────────────────────────────────────

def load_gains(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_gains(path, gains):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(gains, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=["info", "rate", "pulses", "throttle", "hold", "all", "release", "raw"])
    ap.add_argument("args", nargs="*", help="raw: CODE VALUE")
    ap.add_argument("--secs", type=float, default=10.0, help="rate: how long to listen (s)")
    ap.add_argument("--size", type=float, default=0.1, help="pulses: stick deflection (default 0.1)")
    ap.add_argument("--bank", type=float, default=60.0, help="hold: bank limit, deg (try 75 with --nz 6 once it works)")
    ap.add_argument("--nz", type=float, default=5.0, help="hold: load factor limit, g")
    ap.add_argument("--turn", type=float, default=60.0, help="hold: heading step, deg")
    ap.add_argument("--min-agl", type=float, default=1500.0, help="stop below this height above ground, m")
    ap.add_argument("--out", default="dcs_runs")
    ap.add_argument("--gains", default=None, help="gains file (default <out>/stick_gains.json)")
    ap.add_argument("--fake", action="store_true", help="fly a synthetic aircraft instead of DCS")
    ap.add_argument("--fake-roll-sign", type=int, default=1)
    ap.add_argument("--fake-pitch-sign", type=int, default=1)
    ap.add_argument("--port-tel", type=int, default=TEL_PORT)
    ap.add_argument("--port-cmd", type=int, default=CMD_PORT)
    a = ap.parse_args(argv)

    if a.fake:
        from fake_export import FakeLink
        link = FakeLink(roll_sign=a.fake_roll_sign, pitch_sign=a.fake_pitch_sign)
    else:
        link = UdpLink(a.port_tel, a.port_cmd)
    gains_path = a.gains or os.path.join(a.out, "stick_gains.json")
    summary = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "fake": a.fake}
    limits = Limits(min_agl=a.min_agl)
    if a.fake:
        limits.min_alt, limits.min_agl = 0.0, 0.0
    try:
        if a.test == "release":
            link.release()
            print("RELEASE sent: pitch, roll and rudder at 0, control off. The throttle stays where it was set.")
            return summary
        if a.test == "raw":
            if len(a.args) != 2:
                raise SystemExit("raw CODE VALUE")
            link.raw(int(a.args[0]), float(a.args[1]))
            print(f"sent LoSetCommand({a.args[0]}, {a.args[1]})")
            return summary
        if a.test == "info":
            stage_info(link)
            return summary
        todo = ["rate", "pulses", "throttle", "hold"] if a.test == "all" else [a.test]
        gains = load_gains(gains_path)
        for name in todo:
            print(f"\n=== {name} " + "=" * (60 - len(name)))
            run = Run(link, a.out, name, limits)
            try:
                if name == "rate":
                    summary["rate"] = stage_rate(run, a.secs)
                elif name == "pulses":
                    gains.update(stage_pulses(run, a.size))
                    save_gains(gains_path, gains)
                    summary["pulses"] = {k: v for k, v in gains.items()}
                elif name == "throttle":
                    gains.update(stage_throttle(run, gains))
                    save_gains(gains_path, gains)
                    summary["throttle"] = {k: gains[k] for k in ("thr_slope", "thr_trim", "rpm_by_code")}
                else:
                    summary["hold"] = stage_hold(run, gains, a.bank, a.nz, a.turn)
            finally:
                run.close()
                link.release()
            print(f"\n  frames: {run.path if run.fh else '(not saved)'}")
    except Abort as e:
        link.release()
        print(f"\nSTOPPED: {e}\nAxes released (pitch, roll, rudder 0). Take the stick and recover.")
        summary["aborted"] = str(e)
    except KeyboardInterrupt:
        link.release()
        print("\ninterrupted: axes released")
    finally:
        link.close()
    if a.test not in ("info", "release", "raw") and a.out:
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, f"stick_summary_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=1, default=lambda o: None)
        print(f"summary: {path}")
    return summary


if __name__ == "__main__":
    main()
