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
    python dcs/stick/stick_test.py curve               pulses up to full stick: response against command
    python dcs/stick/stick_test.py throttle            throttle code against RPM and acceleration
    python dcs/stick/stick_test.py hold                heading/altitude/speed steps, closed loop
    python dcs/stick/stick_test.py all                 rate, pulses, curve, throttle, hold
    python dcs/stick/stick_test.py click NAME          a cockpit switch or HOTAS control (see cockpit.py)
    python dcs/stick/stick_test.py radar               what the radar reports, once a second (lock by hand)
    python dcs/stick/stick_test.py ind --tag locked    the text of the cockpit displays (HUD, MFDs): run it with and without a lock
    python dcs/stick/stick_test.py manoeuvre           fly turns, climbs and speed changes to compare with the simulator (manoeuvre.py)
    python dcs/stick/stick_test.py cursormap           slew the radar cursor to each edge of the page and save the page (azimuth and range map)
    python dcs/stick/stick_test.py cursortry           try nine ways of moving the radar cursor, announced; you say which moved it
    python dcs/stick/stick_test.py cursor              slew the radar cursor and print which lines of the radar page follow it
    python dcs/stick/stick_test.py lock                try to lock the target (needs make_mission --stick-test --target)
    python dcs/stick/stick_test.py lock --tws          no cursor: TWS, bug the track, single-target track
    python dcs/stick/stick_test.py lock --aim          aim the cursor at the target (any azimuth within 60 deg, any altitude) and lock
    python dcs/stick/stick_test.py fire                master arm, release a missile, watch it go
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


def _both(eng, key):
    """A per-engine reading ({'left':, 'right':} or a number): the Viper has one engine and DCS
    reports 0 for the other, so the largest value is the engine's."""
    if not isinstance(eng, dict):
        return None
    r = eng.get(key)
    if isinstance(r, dict):
        vals = [_num(r.get(k)) for k in ("left", "right")]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    return _num(r)


def _rpm(eng):
    return _both(eng, "RPM")


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
        self.myaw = _num(d.get("myaw"))                 # magnetic heading, rad
        self.pitch = _first(me.get("pitch"), adi.get("pitch"))
        self.bank = _first(me.get("bank"), adi.get("bank"))
        self.nz = _num(g.get("y"))
        self.aoa, self.beta, self.vs = _num(d.get("aoa")), _num(d.get("beta")), _num(d.get("vs"))
        self.rpm = _rpm(d.get("eng"))
        self.fuel_flow = _both(d.get("eng"), "FuelConsumption")
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
        self.last_tel_wall = self.last_event_wall = 0.0
        self.ind_items = {}

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
                self.last_tel_wall = time.time()
                f = Frame(d)
                if f.ax_seq in self.sent and f.rt is not None:
                    self.latencies.append(f.rt - self.sent.pop(f.ax_seq))
                return f
            if d.get("ev") == "ind":
                self.ind_items[d.get("id")] = d.get("text", "")
            self.events[d.get("ev")] = d
            self.last_event_wall = time.time()

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

    def click(self, dev, cmd, value, mode=None):
        self._send(f"CLICK {int(dev)} {int(cmd)} {float(value):g}" + (f" {mode}" if mode else ""))

    def ind(self, first=0, last=49):
        self._send(f"IND {int(first)} {int(last)}")

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
            last = self.frames[-1] if self.frames else None
            where = (f"; last frame: model time {last.t:.1f} s, {last.alt:.0f} m, {last.tas:.0f} m/s "
                     f"(Mach {last.mach:.2f}), bank {last.bank * DEG:+.0f} deg") if last else ""
            alive = getattr(self.link, "last_event_wall", 0.0) > getattr(self.link, "last_tel_wall", 0.0)
            if alive:
                raise Abort("telemetry stopped but DCS is still running its frames: the sim is PAUSED (Esc menu? "
                            "Pause key? a dialog over DCS?)" + where)
            raise Abort("nothing at all from DCS for 3 s: DCS frozen, loading, minimised or closed?" + where
                        + ". If it came back, the mission may have stalled: look at dcs.log")
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

def inverse(curve, y):
    """Command (-1..1, odd symmetric) that gives the response y, from the measured curve
    [[command, response], ...] (responses positive for positive commands); beyond the last
    point the last slope is extended, and the command is capped at 1."""
    pts = sorted((float(a), float(r)) for a, r in curve if a > 0)
    xs, ys = [0.0] + [a for a, _ in pts], [0.0] + [r for _, r in pts]
    ys = list(np.maximum.accumulate(np.array(ys) + np.arange(len(ys)) * 1e-9))
    mag = abs(y)
    if mag >= ys[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if len(xs) > 1 and ys[-1] > ys[-2] else 1e-3
        x = xs[-1] + (mag - ys[-1]) / max(slope, 1e-6)
    else:
        x = float(np.interp(mag, ys, xs))
    return math.copysign(min(1.0, x), y)


class Hold:
    """Heading, altitude and speed hold: heading -> bank -> roll rate -> roll axis; altitude ->
    flight-path angle -> load factor -> pitch axis; speed -> acceleration -> throttle axis.
    The same structure as f16_sim.py's autopilot, with the gains `pulses` and `throttle` measured."""
    K_HDG_BANK = 2.0       # bank per rad of heading error (the simulator's K_HDG_PHI)
    K_BANK_RATE = 2.5      # roll rate per rad of bank error, 1/s
    TAU_GAMMA = 3.0        # s to take out a flight-path error
    TAU_SPEED = 8.0        # s to take out a speed error

    T_LEAD = 0.25          # s: the bank error is taken against the bank the roll rate will reach in this time
                           # (DCS rolls 140 deg/s at a 0.6 command and needs about 0.4 s to stop)
    KI_NZ = 1.5            # g of stick demand per g of error and second: the measured g closes the loop
    NZ_INT_MAX = 3.0       # g the correction may add or take away
    KP_V = 0.04            # throttle position per m/s of speed error
    KI_V = 0.002           # and per m/s of error and second

    def __init__(self, gains, bank_max=math.radians(45), nz_max=4.0, vs_max=30.0, airframe=None):
        """airframe: a library airframe config (bvr_library.airframe_config, or a platform's .airframe). Given,
        the g the wing can give and the bank that leaves enough of it to hold altitude are limited as the
        simulator's autopilot limits them (f16_sim.F16Aircraft.step: nz_avail, BANK_NZ_MARGIN)."""
        self.g, self.bank_max, self.nz_max, self.vs_max = gains, bank_max, nz_max, vs_max
        self.airframe = airframe
        self.last = {}
        self.prev = None
        self.integral, self.t_speed = 0.0, None
        self.prev_bank, self.roll_rate = None, 0.0
        self.nz_int, self.t_nz = 0.0, None     # the g correction: the stick curve was measured at one speed

    # the simulator's altitude law (f16_sim.F16Cfg), used when the command carries its flight-path limits
    K_ALT_GAM = 0.0003     # rad of flight-path angle per m of altitude error
    K_GAM_NZ = 7.0         # g per rad of flight-path error

    @staticmethod
    def _rho(alt):
        """ISA density, kg/m3 (as f16_sim.isa)."""
        h = max(0.0, min(20000.0, alt))
        if h <= 11000.0:
            return 1.225 * (1.0 - 2.25577e-5 * h) ** 4.2559
        return 0.36392 * math.exp(-(h - 11000.0) / 6341.62)

    def nz_avail(self, f):
        """The g the wing gives now: the simulator's nz_avail (dynamic pressure x wing area x CL_MAX / weight,
        at most NZ_POS_MAX), from the airframe and DCS's fuel; without an airframe, nz_max."""
        a = self.airframe
        if a is None or f.tas is None or f.alt is None:
            return self.nz_max
        fuel = (f.raw.get("eng") or {}).get("fuel_internal") if hasattr(f, "raw") else None
        frac = fuel if isinstance(fuel, (int, float)) and 0.0 <= fuel <= 1.0 else 0.6
        w = (a.MASS_EMPTY + a.MASS_WPNS + frac * a.FUEL_FULL) * G
        q = 0.5 * self._rho(f.alt) * f.tas ** 2
        nz = a.NZ_POS_MAX if q <= 100.0 else min(a.NZ_POS_MAX, q * a.S_REF * a.CL_MAX / max(w, 1.0))
        return min(self.nz_max, nz)

    def axes(self, f, hdg, alt, v=None, thr=None, dive=None, climb=None):
        """dive, climb (rad): the policy's flight-path limits (the simulator's altFPA and climbFPA). Given, the
        altitude is flown with the simulator's law; without, with the stick tests' climb-rate law."""
        g = self.g
        vs = f.vs
        if vs is None:
            vs = 0.0 if self.prev is None or f.t <= self.prev[0] else (f.alt - self.prev[1]) / (f.t - self.prev[0])
        self.prev = (f.t, f.alt)
        tas = max(f.tas or 150.0, 80.0)
        gam0 = math.asin(max(-0.9, min(0.9, vs / tas)))
        nz_av = self.nz_avail(f)
        # as the simulator: no more bank than leaves BANK_NZ_MARGIN of the available g to hold the flight path
        # (at 7 km the Viper bled from 312 to 110 m/s in a turn held at 80 deg while the wing gave less and less)
        margin = getattr(self.airframe, "BANK_NZ_MARGIN", 0.85) if self.airframe is not None else 0.85
        bank_lim = min(self.bank_max, math.acos(min(1.0, math.cos(gam0) / max(margin * nz_av, 1e-3))))
        self.last = {"nz_avail": nz_av, "bank_lim": bank_lim}
        bank_cmd = max(-bank_lim, min(bank_lim, self.K_HDG_BANK * wrap_pi(hdg - f.hdg)))
        if self.prev_bank is not None and 0.0 < f.t - self.prev_bank[0] < 0.1:
            inst = wrap_pi(f.bank - self.prev_bank[1]) / (f.t - self.prev_bank[0])
            self.roll_rate = 0.7 * self.roll_rate + 0.3 * inst        # smoothed roll rate, rad/s
        self.prev_bank = (f.t, f.bank)
        bank_pred = f.bank + self.T_LEAD * self.roll_rate
        rate_cmd = max(-3.0, min(3.0, self.K_BANK_RATE * (bank_cmd - bank_pred)))
        if g.get("roll_curve"):
            roll = g["roll_sign"] * inverse(g["roll_curve"], rate_cmd)
        else:
            roll = g["roll_sign"] * rate_cmd / g["roll_rate"]
        gam = math.asin(max(-0.9, min(0.9, vs / tas)))
        if dive is not None and climb is not None:
            gam_cmd = max(-abs(dive), min(abs(climb), self.K_ALT_GAM * (alt - f.alt)))
            nz_cmd = (math.cos(gam) + self.K_GAM_NZ * (gam_cmd - gam)) / max(math.cos(f.bank), 0.1)
        else:
            vs_cmd = max(-self.vs_max, min(self.vs_max, 0.12 * (alt - f.alt)))
            gam_cmd = math.asin(max(-0.9, min(0.9, vs_cmd / tas)))
            nz_cmd = (math.cos(gam) + tas / (G * self.TAU_GAMMA) * (gam_cmd - gam)) / max(math.cos(f.bank), 0.3)
        nz_cmd = max(-1.0, min(self.nz_max, nz_av, nz_cmd))
        # The curve gives the stick for a g, measured at one speed; at another the same stick gives another g (in
        # DCS, 7 g instead of 5.8 at 304 m/s and 80 deg of bank: the turn climbed and bled speed). The measured g
        # closes the loop: its error is integrated into the demand (not while the stick is at its stop).
        d_corr = 0.0
        if f.nz is not None:
            dt = 0.0 if self.t_nz is None else max(0.0, min(0.2, f.t - self.t_nz))
            self.t_nz = f.t
            err = nz_cmd - f.nz
            sat = getattr(self, "_pitch_sat", 0)
            if not (sat > 0 and err > 0) and not (sat < 0 and err < 0):
                self.nz_int = max(-self.NZ_INT_MAX, min(self.NZ_INT_MAX, self.nz_int + self.KI_NZ * err * dt))
            d_corr = self.nz_int
        if g.get("pitch_curve"):
            d = nz_cmd - 1.0 + d_corr            # pull and push have their own curves
            cmd = inverse(g["pitch_curve"], d) if d >= 0 else -inverse(g.get("pitch_push_curve") or g["pitch_curve"], -d)
            pitch = g["pitch_sign"] * cmd
        else:
            pitch = g["pitch_sign"] * (nz_cmd - 1.0 + d_corr) / g["nz_per_cmd"]
        self._pitch_sat = 1 if g["pitch_sign"] * pitch >= 1.0 else -1 if g["pitch_sign"] * pitch <= -1.0 else 0
        if v is not None:
            # speed -> throttle position u (+1 = most thrust) by PI; the axis code is sign * u
            # (DCS's throttle code is inverted: -1 is afterburner, +1 idle)
            err = v - (f.tas or tas)
            dt = 0.0 if self.t_speed is None else max(0.0, min(0.2, f.t - self.t_speed))
            self.t_speed = f.t
            u = g.get("thr_u_trim", 0.0) + self.KP_V * err + self.integral
            if abs(u) < 1.0 or err * u < 0:                 # no wind-up against the stop
                self.integral = max(-0.8, min(0.8, self.integral + self.KI_V * err * dt))
            thr = g.get("thr_sign", 1) * max(-1.0, min(1.0, u))
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
    chk(len(set(rt.tolist())) > 0.5 * n, f"real-time stamps distinct on {len(set(rt.tolist())) * 100 // n}% of frames "
        f"(low means the stamp is quantised: the command-to-response delay can't be measured)")
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
    if f.mach is not None:
        chk(0.6 < f.mach < 0.97, f"Mach {f.mach:.2f} at the end: fly the tests at Mach 0.8-0.95 "
                                 f"(transonic changes the pitch response and the drag)")
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
          f"bank {f.bank * DEG:+.1f}, nz {f.nz:.2f}, RPM {f.rpm:.1f}, fuel flow {f.fuel_flow}")
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


def drive_until(run, max_s, fn, done):
    f = run.next()
    t0 = f.t
    while f.t - t0 < max_s and not done(f):
        run.link.axes(**fn(f, f.t - t0))
        f = run.next()
    return f


CURVE_AMPS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def stage_curve(run, gains, amps=CURVE_AMPS, axes=("pitch", "roll"), max_bank_change=math.radians(20),
                max_nz_change=5.5, max_pitch_change=math.radians(12)):
    """Response of the pitch and roll axes against the command, up to full stick. Each pulse
    stops early (and the aircraft is brought back by the known sign) when the bank has
    changed by 20 deg, or the load factor up by max_nz_change g (or down to -2 g) or the pitch by 12 deg. A pulse
    stopped early on load factor or pitch only shows the response is *at least* that: such
    pitch points are marked * and left out of the stored curve."""
    for k in ("roll_sign", "pitch_sign"):
        if k not in gains:
            raise SystemExit("curve needs the signs: run `pulses` first (or `all`).")
    out = {}
    print(f"\n  pulses of {', '.join(f'{a:g}' for a in amps)} each way, stopped early at 20 deg of bank, "
          f"{max_nz_change:g} g or {math.degrees(max_pitch_change):.0f} deg of pitch")
    for axis in axes:
        rows, push = [], {}
        for a in amps:
            peak, held, early = {}, {}, {}
            for sgn in (1, -1):
                base = drive(run, 1.5, lambda f, tau: dict(ZERO))
                f0 = base[-1]
                bank0, gam0 = f0.bank, f0.pitch
                nz0 = float(np.nanmean(series(base[-25:], "nz")))
                rate0 = float(np.nanmean(deriv(series(base, "t"), series(base, "bank"), wrap=True)))
                st = {"stop": None, "why": None}
                on_max = 1.0 if axis == "pitch" else 0.7

                def fn(f, tau):
                    if st["stop"] is None:
                        hit = (axis == "roll" and abs(wrap_pi(f.bank - bank0)) > max_bank_change) \
                            or (axis == "pitch" and (f.nz - nz0 > max_nz_change or f.nz < -2.0
                                                     or abs(f.pitch - gam0) > max_pitch_change))
                        if tau >= on_max or hit:
                            st["stop"], st["why"] = tau, ("limit" if hit else "time")
                    return {**ZERO, axis: 0.0 if st["stop"] is not None else sgn * a}
                fr = drive(run, on_max + 0.5, fn)
                held[sgn] = st["stop"] if st["stop"] is not None else on_max
                early[sgn] = st["why"] == "limit"
                if axis == "roll":
                    r = deriv(series(fr, "t"), series(fr, "bank"), wrap=True) - rate0
                    peak[sgn] = float(r[np.nanargmax(np.abs(r))])
                else:
                    d = series(fr, "nz") - nz0
                    peak[sgn] = float(d[np.nanargmax(np.abs(d))])
                # bring it back with the known sign
                if axis == "roll":
                    drive_until(run, 8.0, lambda f, tau: {**ZERO, "roll": gains["roll_sign"] * max(-0.25, min(0.25, -0.8 * wrap_pi(f.bank - bank0)))},
                                lambda f: abs(wrap_pi(f.bank - bank0)) < math.radians(3))
                else:
                    drive_until(run, 10.0, lambda f, tau: {**ZERO, "pitch": gains["pitch_sign"] * max(-0.2, min(0.2, -2.0 * (f.pitch - gam0)))},
                                lambda f: abs(f.pitch - gam0) < math.radians(1.5) and abs(f.nz - nz0) < 0.25)
            resp = (peak[1] - peak[-1]) / 2.0
            lim = early[1] or early[-1]
            if axis == "pitch":
                # pull and push are different: the F-16 pushes over to about -1.7 g and no further
                ps = int(gains["pitch_sign"])
                resp = abs(peak[ps])                      # the pull side: what a turn needs
                lim = early[ps]
                push[a] = abs(peak[-ps])
            rows.append((a, peak[1], peak[-1], resp, 0.5 * (held[1] + held[-1]), lim))
        k = DEG if axis == "roll" else 1.0
        unit = "deg/s" if axis == "roll" else "g"
        print(f"\n  {axis}: {'rate' if axis == 'roll' else 'load factor change, pull side (push in the last column)'} "
              f"against command")
        print(f"  {'cmd':>6}{'+ pulse':>12}{'- pulse':>12}{'response':>12}{'per unit':>12}{'held s':>9}"
              + (f"{'push':>8}" if axis == "pitch" else ""))
        for a, pp, pm, resp, held, er in rows:
            mark = " *" if er and axis == "pitch" else "  "
            print(f"  {a:>6.2f}{pp * k:>+12.2f}{pm * k:>+12.2f}{abs(resp) * k:>9.2f} {unit:<4}{abs(resp) * k / a:>8.2f}"
                  f"{held:>9.2f}{mark}" + (f"{push[a]:>7.2f}g" if axis == "pitch" else ""))
        keep = [r for r in rows if not (axis == "pitch" and r[5])]
        if axis == "pitch" and len(keep) < len(rows):
            print(f"  * stopped early at the limit: at least this much; left out of the stored curve")
        if len(keep) < 3:
            keep = rows
        out[f"{axis}_curve"] = [[r[0], abs(r[3])] for r in keep]
        if axis == "pitch":
            out["pitch_push_curve"] = [[r[0], push[r[0]]] for r in rows]
        sgn_resp = [1 if r[3] >= 0 else -1 for r in rows]
        if len(set(sgn_resp)) > 1:
            print(f"  WARN: the {axis} response changes sign along the sweep")
        top = rows[-1]
        print(f"  at command {top[0]:g}: {abs(top[3]) * k:.1f} {unit}{' or more' if top[5] and axis == 'pitch' else ''}")
    return out


def stage_throttle(run, gains, hold_s=6.0, bank_max=math.radians(30)):
    """Which way the throttle code goes, and RPM and fuel flow against it, holding heading and
    altitude on the stick. DCS's code turned out inverted (-1 afterburner, +1 idle), so the
    sign is found first, from two moderate codes, and the levels are then run in thrust order
    u = -1 ... +1. A level ends early if the speed changes by 35 m/s during it, leaves the start speed by 120 m/s or passes Mach 1.15."""
    need = ("roll_sign", "pitch_sign")
    if not all(k in gains for k in need) or not (gains.get("roll_curve") or gains.get("roll_rate")):
        raise SystemExit("throttle needs the stick gains: run `pulses` and `curve` first (or `all`).")
    f0 = run.next()
    hdg, alt, tas0 = f0.hdg, f0.alt, f0.tas
    h = Hold(gains, bank_max=bank_max)

    def level(code, secs):
        k0, st = len(run.frames), {}

        def done(f):
            st.setdefault("v0", f.tas)
            return abs(f.tas - st["v0"]) > 35 or abs(f.tas - tas0) > 120 or (f.mach or 0) > 1.15
        drive_until(run, secs, lambda f, tau: h.axes(f, hdg, alt, thr=code), done)
        return run.frames[k0:]

    def tail(fr, secs, attr):
        t1 = fr[-1].t
        v = [getattr(f, attr) for f in fr if f.t >= t1 - secs and getattr(f, attr) is not None]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n  heading {hdg * DEG:.0f} deg, {alt:.0f} m, {tas0:.0f} m/s (Mach {f0.mach:.2f}) held on the stick")
    a_fr, b_fr = level(0.5, 4.0), level(-0.5, 4.0)
    ra, rb = tail(a_fr, 1.5, "rpm"), tail(b_fr, 1.5, "rpm")
    fa, fb = tail(a_fr, 1.5, "fuel_flow"), tail(b_fr, 1.5, "fuel_flow")
    more_at_minus = (rb > ra + 1.0) or (rb >= ra - 1.0 and fb > fa * 1.1)
    more_at_plus = (ra > rb + 1.0) or (ra >= rb - 1.0 and fa > fb * 1.1)
    sign = -1 if more_at_minus and not more_at_plus else 1 if more_at_plus and not more_at_minus else 0
    print(f"  probe: code +0.5 -> RPM {ra:.1f}, fuel {fa:.2f};  code -0.5 -> RPM {rb:.1f}, fuel {fb:.2f}")
    if sign == 0:
        print("  WARN: the probe codes gave no clear difference: is the throttle axis overridden (physical "
              "throttle assigned)? Assuming a positive code is more thrust.")
        sign = 1
    print(f"  throttle code is {'INVERTED: -1 is the most thrust' if sign < 0 else 'normal: +1 is the most thrust'}")
    rows = []
    for u in (-1.0, -0.5, 0.0, 0.5, 1.0):
        fr = level(sign * u, hold_s)
        dur = fr[-1].t - fr[0].t
        tl = [f for f in fr if f.t >= fr[-1].t - 4.0]
        acc = float(np.polyfit(series(tl, "t") - tl[0].t, series(tl, "tas"), 1)[0]) \
            if dur >= 3.5 and len(tl) > 5 else float("nan")
        rows.append((u, sign * u, tail(fr, 1.5, "rpm"), tail(fr, 1.5, "fuel_flow"), acc, fr[-1].tas, fr[-1].mach, dur))
    print(f"\n  {'u':>5}{'code':>7}{'RPM':>8}{'fuel':>8}{'accel m/s2':>12}{'TAS':>7}{'Mach':>7}{'held s':>8}")
    for u, c, rp, fu, acc, tas, m, dur in rows:
        print(f"  {u:>+5.1f}{c:>+7.1f}{rp:>8.1f}{fu:>8.2f}{acc:>+12.2f}{tas:>7.0f}{m:>7.2f}{dur:>8.1f}")
    ok = [(r[0], r[4]) for r in rows if not math.isnan(r[4])]
    out = {"thr_sign": sign, "thr_map": [[r[0], r[2], r[3]] for r in rows], "thr_u_trim": 0.0, "thr_slope": float("nan")}
    if len(ok) >= 3:
        xs, ys = np.array([o[0] for o in ok]), np.array([o[1] for o in ok])
        slope, icpt = np.polyfit(xs, ys, 1)
        if slope > 0:
            out["thr_slope"] = float(slope)
            out["thr_u_trim"] = float(np.clip(-icpt / slope, -0.5, 0.5))
        else:
            print("  WARN: acceleration did not rise with thrust (drag changes through the transonic range?): "
                  "the speed loop will find its own trim")
    r = [x[2] for x in rows]
    if not (r[-1] > r[0] + 5):
        print("  WARN: RPM did not rise from u = -1 to +1: the throttle code may not be reaching the engine")
    print(f"\n  thrust-neutral position (from the acceleration): u = {out['thr_u_trim']:+.2f}; "
          f"afterburner is where the fuel flow jumps (u = +1 here: {rows[-1][3]:.1f} against {rows[-2][3]:.1f})")
    level(sign * out["thr_u_trim"], 3.0)
    return out


def _steps(fr, t_cmd, hdg0, delta, span=22.0):
    """Response of a heading step commanded at model time t_cmd, `delta` rad from hdg0."""
    seg = [f for f in fr if t_cmd <= f.t < t_cmd + span]
    ts = np.array([f.t for f in seg]) - t_cmd
    h = np.unwrap([f.hdg for f in seg])
    d = (h - h[0]) * np.sign(delta)
    tgt = abs(delta)
    half = ts[np.argmax(d >= 0.5 * tgt)] if np.any(d >= 0.5 * tgt) else float("nan")
    near = tgt - math.radians(10.0)
    t10 = ts[np.argmax(d >= near)] if np.any(d >= near) else float("nan")
    nin = ts[np.argmax(d >= 0.9 * tgt)] if np.any(d >= 0.9 * tgt) else float("nan")
    rate = deriv(ts, h * np.sign(delta))
    bulk = (tgt - d > math.radians(30)) & (ts > 1.5)
    bank = np.array([f.bank for f in seg]) * np.sign(delta)
    peak_rate = float(np.nanmax(deriv(ts, h * np.sign(delta), smooth=41)) * DEG)
    return {"delta_deg": delta * DEG, "half_s": float(half), "p90_s": float(nin), "within10_s": float(t10),
            "peak_rate_dps": peak_rate,
            "rate_bulk_dps": float(np.nanmean(rate[bulk]) * DEG) if np.any(bulk) else float("nan"),
            "peak_bank_deg": float(np.nanmax(bank) * DEG), "peak_nz": float(np.nanmax([f.nz for f in seg]))}


def stage_hold(run, gains, bank_deg=60.0, nz_max=5.0, turn=60.0, speed_step=40.0, alt_step=300.0, speed=None):
    need = ("roll_sign", "pitch_sign", "thr_sign")
    if not all(k in gains for k in need) or not gains.get("roll_curve") or not gains.get("pitch_curve"):
        raise SystemExit("hold needs the stick gains and curves and the throttle sign: run `pulses`, `curve` and "
                         "`throttle` first (or `all`).")
    link = run.link
    f0 = run.next()
    h = Hold(gains, bank_max=math.radians(bank_deg), nz_max=nz_max)
    if speed is not None and abs(f0.tas - speed) > 6.0:
        # capture the speed first: the speed loop on the throttle, heading and altitude on the stick
        print(f"\n  capturing {speed:.0f} m/s from {f0.tas:.0f} (heading and altitude held)")
        hc, ac = f0.hdg, f0.alt
        st = {"n": 0}

        def settled(f):
            st["n"] = st["n"] + 1 if abs(f.tas - speed) < 4.0 else 0
            return st["n"] > 150                       # within 4 m/s for 3 s
        fl = drive_until(run, 60.0, lambda f, tau: h.axes(f, hc, ac, v=speed), settled)
        print(f"  speed {fl.tas:.0f} m/s (Mach {fl.mach:.2f}) after {fl.t - f0.t:.0f} s")
        if abs(fl.tas - speed) > 10.0:
            raise Abort(f"could not capture {speed:.0f} m/s in 60 s (at {fl.tas:.0f}): the speed loop does not work")
        f0 = run.next()
    hdg0, alt0, v0 = f0.hdg, f0.alt, f0.tas
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
    print(f"\n  {'step':<10}{'turn':>7}{'half':>8}{'to 10d':>8}{'90%':>8}{'mean':>8}{'peak':>7}{'bank':>7}{'nz':>6}")
    for i, s in enumerate(res["heading_steps"]):
        print(f"  {'turn ' + str(i + 1):<10}{s['delta_deg']:>+6.0f}d{s['half_s']:>7.1f}s{s['within10_s']:>7.1f}s"
              f"{s['p90_s']:>7.1f}s{s['rate_bulk_dps']:>8.1f}{s['peak_rate_dps']:>7.1f}{s['peak_bank_deg']:>6.0f}d{s['peak_nz']:>6.1f}")
    print(f"  (deg/s: mean while more than 30 deg to go, and peak; bank limit {bank_deg:.0f} deg sustains "
          f"{math.degrees(G * math.tan(math.radians(bank_deg)) / max(v0, 100.0)):.1f} deg/s at {v0:.0f} m/s)")
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
    ap.add_argument("test", choices=["info", "rate", "pulses", "curve", "throttle", "hold", "all", "release", "raw",
                                        "click", "radar", "lock", "fire", "ind", "inddiff", "cursor", "cursortry", "cursormap",
                                        "manoeuvre"])
    ap.add_argument("args", nargs="*", help="raw: CODE VALUE; click: NAME (group.CONTROL) or DEVICE COMMAND")
    ap.add_argument("--value", type=float, default=None, help="click: the value to send (a switch position, an axis)")
    ap.add_argument("--platform", default="F-16C-DCS",
                    help="manoeuvre: the library platform whose climb limits the program flies (default F-16C-DCS)")
    ap.add_argument("--mode", choices=["c", "s", "l"], default=None,
                    help="click: c = clickable element (mouse click), s = device SetCommand (key binding), l = LoSetCommand "
                         "(default: s for HOTAS controls, c for the rest)")
    ap.add_argument("--hold", type=float, default=0.2, help="click: seconds between press and release")
    ap.add_argument("--scan", action="store_true", help="lock: also slew the radar cursor between presses")
    ap.add_argument("--tws", action="store_true",
                    help="lock: no cursor: target management right held to TWS, right again until the track is bugged "
                         "(an AIM-120 can be fired at it; --stt also asks for a single-target track)")
    ap.add_argument("--stt", action="store_true", help="lock --tws: after the bug, target management up for a single-target track")
    ap.add_argument("--aim", action="store_true",
                    help="lock: put the cursor where the target is (bullseye readout) in range and azimuth, tilt the antenna "
                         "to its altitude, then designate: works off the nose and off the level")
    ap.add_argument("--bull", type=float, nargs=2, metavar=("X", "Z"), default=None,
                    help="lock --aim: the bullseye in DCS metres (x north, z east); by default it is read off the radar page")
    ap.add_argument("--ref", choices=["auto", "own", "bull"], default="auto",
                    help="lock --aim: what the radar page's cursor readout is measured from (auto: switch it to our own "
                         "aircraft through the DED if it reads from a bullseye)")
    ap.add_argument("--relock", action="store_true", help="lock --aim: drop an existing lock first (target management down)")
    ap.add_argument("--magvar", type=float, default=None,
                    help="lock --aim: magnetic variation, deg east (default: from the module's magnetic heading)")
    ap.add_argument("--force", action="store_true", help="fire: release even without a lock")
    ap.add_argument("--offset", type=float, default=None,
                    help="lock, fire, cursormap: first turn so the target is this many degrees right of the nose (negative: left)")
    ap.add_argument("--dalt", type=float, default=None,
                    help="lock, fire, cursormap: fly this many metres above the target (negative: below) instead of level with it")
    ap.add_argument("--release-hold", type=float, default=1.5,
                    help="fire: seconds the weapon release button is held (default 1.5; the missile leaves only while it is held)")
    ap.add_argument("--watch", type=int, default=15,
                    help="fire: seconds to watch for the missile (default 15; 90 shows the target destroyed from 60 km)")
    ap.add_argument("--aa", action="store_true",
                    help="lock, fire: always press the ICP A-A button (it toggles; by default it is pressed only if the HUD is not in A-A)")
    ap.add_argument("--no-aa", action="store_true", help="lock, fire: never press the ICP A-A button")
    ap.add_argument("--scan-steps", type=int, default=40, help="lock --scan: cursor steps (default 40)")
    ap.add_argument("--scan-hold", type=float, default=0.3, help="lock --scan: seconds of cursor slew per step (default 0.3)")
    ap.add_argument("--soi", choices=["left", "right", "none"], default="none",
                    help="lock: first make this MFD the sensor of interest (display management switch); default none: the radar MFD is normally the SOI already; a display management press can switch the MFD to its menu page")
    ap.add_argument("--release", type=float, default=None,
                    help="click: the value to send after --hold (for axes such as hotas.THROTTLE_RDR_CURSOR_Y_AXIS --value 0.5 --release 0)")
    ap.add_argument("--match-alt", action="store_true",
                    help="radar, lock, fire: fly to the nearest aircraft's altitude (a radar sees it best there)")
    ap.add_argument("--alt", type=float, default=None, help="radar, lock, fire: fly at this altitude, m")
    ap.add_argument("--show", type=int, nargs="*", default=None,
                    help="ind: print these displays in full (default: a summary of each); inddiff: only these displays")
    ap.add_argument("--tag", default="", help="ind: a name for the saved file (before_lock, locked)")
    ap.add_argument("--no-fly", action="store_true", help="radar, lock, fire: do not hold heading, altitude and speed")
    ap.add_argument("--secs", type=float, default=10.0, help="rate: how long to listen (s)")
    ap.add_argument("--size", type=float, default=0.1, help="pulses: stick deflection (default 0.1)")
    ap.add_argument("--bank", type=float, default=60.0, help="hold: bank limit, deg (try 75 with --nz 6 once it works)")
    ap.add_argument("--nz", type=float, default=5.0, help="hold: load factor limit, g")
    ap.add_argument("--turn", type=float, default=60.0, help="hold: heading step, deg")
    ap.add_argument("--speed", type=float, default=None,
                    help="hold: capture this true airspeed (m/s) first; default: hold the speed you are at")
    ap.add_argument("--axes", default="pitch,roll", help="curve: which axes (default pitch,roll)")
    ap.add_argument("--max-nz", type=float, default=5.5, help="curve: stop a pitch pulse at this load factor change, g")
    ap.add_argument("--min-agl", type=float, default=1500.0, help="stop below this height above ground, m")
    ap.add_argument("--out", default="dcs_runs")
    ap.add_argument("--gains", default=None, help="gains file (default <out>/stick_gains.json)")
    ap.add_argument("--fake", action="store_true", help="fly a synthetic aircraft instead of DCS")
    ap.add_argument("--fake-roll-sign", type=int, default=1)
    ap.add_argument("--fake-pitch-sign", type=int, default=1)
    ap.add_argument("--fake-roll-tau", type=float, default=0.12, help="fake aircraft: roll rate lag, s")
    ap.add_argument("--fake-thr-inverted", action="store_true", help="fake aircraft: throttle code -1 is most thrust")
    ap.add_argument("--fake-curve", type=float, default=1.0, help="fake aircraft: response ~ command ** this")
    ap.add_argument("--port-tel", type=int, default=TEL_PORT)
    ap.add_argument("--port-cmd", type=int, default=CMD_PORT)
    a = ap.parse_args(argv)

    if a.fake:
        from fake_export import FakeLink
        link = FakeLink(roll_sign=a.fake_roll_sign, pitch_sign=a.fake_pitch_sign, curve_pow=a.fake_curve,
                        thr_inverted=a.fake_thr_inverted, roll_tau=a.fake_roll_tau)
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
        if a.test == "ind":
            import cockpit
            cockpit.stage_ind(link, a.tag, a.out, show=a.show)
            return summary
        if a.test == "inddiff":
            import cockpit
            if len(a.args) != 2:
                raise SystemExit("inddiff FILE_A FILE_B (two files saved by `ind`)")
            cockpit.ind_diff(a.args[0], a.args[1], ids=a.show)
            return summary
        if a.test in ("click", "radar", "lock", "fire", "cursor", "cursortry", "cursormap"):
            import cockpit
            defs = cockpit.load_defs()
            cockpit.HOLD_OPTS.update(offset=a.offset, dalt=a.dalt)
            gains = load_gains(gains_path)
            have = all(k in gains for k in ("roll_sign", "pitch_sign", "thr_sign")) and gains.get("roll_curve") \
                and gains.get("pitch_curve")
            if a.test != "click" and not a.no_fly and not have:
                raise SystemExit("heading, altitude and speed are held on the stick while this runs, which needs the gains "
                                 "from pulses, curve and throttle (or add --no-fly and hold the aircraft yourself).")
            run = Run(link, a.out, a.test, limits)
            try:
                first = run.next()
                fgains = None if a.no_fly else gains
                if a.test == "click":
                    if not a.args:
                        raise SystemExit("click NAME (e.g. hotas.STICK_TGT_MANAGE_UP, ufc.AA, left_osb.4) or click DEVICE COMMAND")
                    spec = a.args if len(a.args) == 2 and all(x.lstrip("-").isdigit() for x in a.args) else a.args[0]
                    try:
                        cockpit.click(run, spec, defs, value=a.value, hold=a.hold, release=a.release, mode=a.mode)
                    except ValueError as e:
                        raise SystemExit(str(e))
                elif a.test == "cursormap":
                    summary["cursormap"] = cockpit.stage_cursormap(run, fgains, defs, out_dir=a.out, alt=a.alt,
                                                                   match_alt=a.match_alt)
                elif a.test == "cursortry":
                    summary["cursortry"] = cockpit.stage_cursortry(run, fgains, defs, alt=a.alt, match_alt=a.match_alt)
                elif a.test == "cursor":
                    summary["cursor"] = cockpit.stage_cursorprobe(run, fgains, defs, out_dir=a.out, alt=a.alt,
                                                                  match_alt=a.match_alt)
                elif a.test == "radar":
                    fn = cockpit.hold_fn(run, fgains, a.alt, a.match_alt) if fgains else None
                    summary["radar"] = cockpit.stage_radar(run, a.secs, a.out, fn=fn)
                elif a.test == "lock" and a.tws:
                    summary["lock"] = cockpit.stage_twslock(run, fgains, defs, alt=a.alt, match_alt=a.match_alt,
                                                            aa=True if a.aa else False if a.no_aa else None,
                                                            relock=a.relock, stt=a.stt)
                elif a.test == "lock" and a.aim:
                    summary["lock"] = cockpit.stage_aimlock(run, fgains, defs, alt=a.alt, match_alt=a.match_alt,
                                                            bull=a.bull, magvar=a.magvar, relock=a.relock, ref=a.ref,
                                                            aa=True if a.aa else False if a.no_aa else None)
                elif a.test == "lock":
                    summary["lock"] = cockpit.stage_lock(run, fgains, defs, scan=a.scan, alt=a.alt, match_alt=a.match_alt,
                                                         aa=True if a.aa else False if a.no_aa else None, soi=a.soi,
                                                         scan_steps=a.scan_steps, scan_hold=a.scan_hold)
                else:
                    summary["fire"] = cockpit.stage_fire(run, fgains, defs, force=a.force, alt=a.alt,
                                                         match_alt=a.match_alt, aa=True if a.aa else False if a.no_aa else None,
                                                         watch=a.watch, release_hold=a.release_hold)
            finally:
                run.close()
                link.release()
            if a.test == "click":
                return summary
            return summary
        todo = ["rate", "pulses", "curve", "throttle", "hold"] if a.test == "all" else [a.test]
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
                elif name == "curve":
                    gains.update(stage_curve(run, gains, axes=tuple(a.axes.split(",")),
                                             max_nz_change=a.max_nz))
                    save_gains(gains_path, gains)
                    summary["curve"] = {k: gains[k] for k in ("pitch_curve", "pitch_push_curve", "roll_curve") if k in gains}
                elif name == "throttle":
                    gains.update(stage_throttle(run, gains))
                    for stale in ("thr_trim", "rpm_by_code"):
                        gains.pop(stale, None)
                    save_gains(gains_path, gains)
                    summary["throttle"] = {k: gains[k] for k in ("thr_sign", "thr_u_trim", "thr_slope", "thr_map")}
                elif name == "manoeuvre":
                    import manoeuvre
                    if not all(gains.get(k) for k in ("roll_curve", "pitch_curve", "thr_sign")):
                        raise SystemExit("manoeuvre needs the measured gains: run stick_test.py all first")
                    summary["manoeuvre"] = manoeuvre.stage_manoeuvre(run, gains, a.out, a.platform)
                else:
                    summary["hold"] = stage_hold(run, gains, a.bank, a.nz, a.turn, speed=a.speed)
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
