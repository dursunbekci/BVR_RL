"""
fake_export.py  --  a stand-in for bvr_stick_export.lua, to try stick_test.py without DCS
=========================================================================================

FakeLink has the same methods as stick_test.UdpLink (next, axes, release, ...)
but flies a small synthetic aircraft in lock step: each next() advances 20 ms
with the axes last sent. It is not an F-16. It has the shape of one -- a roll
rate and a load factor proportional to the stick, speed set by the throttle --
with the signs and gains chosen by the caller, so the tests can check that
stick_test.py finds them again.

    python dcs/stick/stick_test.py all --fake
    python dcs/stick/stick_test.py all --fake --fake-roll-sign -1 --fake-pitch-sign -1
"""

import math

G = 9.80665
DT = 0.02


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class FakeAircraft:
    def __init__(self, roll_sign=1, pitch_sign=1, rudder_sign=1, roll_rate=2.6, nz_per_cmd=6.0,
                 thr_trim=0.1, thr_slope=4.0, v0=250.0, alt0=7000.0, hdg0=1.0, noise=0.0):
        self.roll_sign, self.pitch_sign, self.rudder_sign = roll_sign, pitch_sign, rudder_sign
        self.roll_rate, self.nz_per_cmd = roll_rate, nz_per_cmd
        self.thr_trim, self.thr_slope = thr_trim, thr_slope
        self.t = 100.0
        self.x, self.z, self.alt, self.V = 0.0, 0.0, alt0, v0
        self.hdg, self.gamma, self.bank, self.p, self.nz = hdg0, 0.0, 0.0, 0.0, 1.0
        self.thr, self.thr_f, self.beta = thr_trim, thr_trim, 0.0
        self.noise = noise

    def step(self, ax, dt=DT):
        roll, pitch, rud = ax.get("roll", 0.0), ax.get("pitch", 0.0), ax.get("rudder", 0.0)
        thr = ax.get("throttle")
        if thr is not None:
            self.thr = max(-1.0, min(1.0, thr))
        self.p += (self.roll_sign * self.roll_rate * roll - self.p) * dt / 0.12
        self.bank = wrap_pi(self.bank + self.p * dt)
        nz_t = 1.0 + self.pitch_sign * self.nz_per_cmd * pitch
        self.nz += (nz_t - self.nz) * dt / 0.10
        self.gamma += G * (self.nz * math.cos(self.bank) - math.cos(self.gamma)) / self.V * dt
        self.hdg = (self.hdg + (G * self.nz * math.sin(self.bank) / (self.V * max(math.cos(self.gamma), 0.2))
                                + self.rudder_sign * 0.08 * rud) * dt) % (2 * math.pi)
        self.thr_f += (self.thr - self.thr_f) * dt / 0.8
        acc = self.thr_slope * (self.thr_f - self.thr_trim) - G * math.sin(self.gamma) - 0.0005 * (self.V - 250.0)
        self.V = max(60.0, self.V + acc * dt)
        self.alt += self.V * math.sin(self.gamma) * dt
        self.x += self.V * math.cos(self.gamma) * math.cos(self.hdg) * dt
        self.z += self.V * math.cos(self.gamma) * math.sin(self.hdg) * dt
        self.beta = self.rudder_sign * 0.1 * rud
        self.t += dt

    def telemetry(self, n, ax, controlling, real_t):
        vs = self.V * math.sin(self.gamma)
        return {
            "ev": "tel", "n": n, "rt": real_t, "t": self.t,
            "me": {"name": "FAKE", "coal": 2, "l1": 1, "l2": 1, "lat": 42.0, "lon": 41.0, "alt": self.alt,
                   "hdg": self.hdg, "pitch": self.gamma, "bank": self.bank,
                   "x": self.x, "y": self.alt, "z": self.z},
            "tas": self.V, "ias": self.V * 0.8, "mach": self.V / 330.0, "aoa": 0.05, "beta": self.beta,
            "g": {"x": 0.0, "y": self.nz, "z": 0.0}, "vs": vs, "asl": self.alt, "agl": self.alt - 300.0,
            "adi": {"pitch": self.gamma, "bank": self.bank, "yaw": self.hdg},
            "vel": {"x": self.V * math.cos(self.gamma) * math.cos(self.hdg), "y": vs,
                    "z": self.V * math.cos(self.gamma) * math.sin(self.hdg)},
            "eng": {"RPM": {"left": 65.0 + 35.0 * (self.thr_f + 1) / 2, "right": 65.0 + 35.0 * (self.thr_f + 1) / 2}},
            "ax": {"pitch": ax["pitch"], "roll": ax["roll"], "rudder": ax["rudder"],
                   "throttle": ax.get("throttle"), "seq": ax["seq"], "on": controlling},
        }


class FakeLink:
    """Same interface as stick_test.UdpLink, in lock step with a FakeAircraft."""

    def __init__(self, **kw):
        self.ac = FakeAircraft(**kw)
        self.ax = {"pitch": 0.0, "roll": 0.0, "rudder": 0.0, "throttle": None, "seq": 0}
        self.controlling = False
        self.n = 0
        self.sent = {}
        self.latencies = []
        self.events = {"hello": {"ev": "hello", "source": "frame", "codes": {"pitch": 2001, "roll": 2002, "rudder": 2003,
                                                          "throttle": 2004}, "codes_from": "fake"},
                       "stats": {"ev": "stats", "tel_hz": 50.0, "frame_hz": 60.0, "event_hz": 1.0, "frame_dt_max": 0.02}}
        self.seq = 0
        self._wall = 1.0e9
        self.commands_sent = 0

    def wall(self):
        return self._wall

    def next(self, timeout=1.0):
        from stick_test import Frame
        self._wall += DT
        self.ac.step(self.ax if self.controlling else {"roll": 0.0, "pitch": 0.0, "rudder": 0.0})
        self.n += 1
        d = self.ac.telemetry(self.n, self.ax, self.controlling, self._wall)
        f = Frame(d)
        if f.ax_seq in self.sent:
            self.latencies.append(self._wall - self.sent.pop(f.ax_seq))
        return f

    def axes(self, pitch=None, roll=None, rudder=None, throttle=None):
        self.seq += 1
        self.commands_sent += 1
        for k, v in (("pitch", pitch), ("roll", roll), ("rudder", rudder), ("throttle", throttle)):
            if v is not None:
                self.ax[k] = max(-1.0, min(1.0, v))
        self.ax["seq"] = self.seq
        self.sent[self.seq] = self._wall
        self.controlling = True

    def release(self):
        self.ax.update({"pitch": 0.0, "roll": 0.0, "rudder": 0.0})
        self.controlling = False

    def raw(self, code, value):
        pass

    def info(self):
        pass

    def ping(self):
        return 0.0

    def close(self):
        pass
