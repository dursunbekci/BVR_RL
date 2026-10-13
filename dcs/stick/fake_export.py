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
                 thr_trim=0.1, thr_slope=4.0, v0=250.0, alt0=7000.0, hdg0=1.0, noise=0.0, curve_pow=1.0, push_limit_g=None, thr_inverted=False, roll_tau=0.12):
        self.roll_sign, self.pitch_sign, self.rudder_sign = roll_sign, pitch_sign, rudder_sign
        self.roll_rate, self.nz_per_cmd = roll_rate, nz_per_cmd
        self.thr_trim, self.thr_slope = thr_trim, thr_slope
        self.t = 100.0
        self.x, self.z, self.alt, self.V = 0.0, 0.0, alt0, v0
        self.hdg, self.gamma, self.bank, self.p, self.nz = hdg0, 0.0, 0.0, 0.0, 1.0
        self.thr, self.thr_f, self.beta = thr_trim, thr_trim, 0.0
        self.noise = noise
        self.push_limit_g = push_limit_g      # the push side stops here (DCS: about 2.7 g below 1 g)
        self.roll_tau = roll_tau              # roll rate lag, s
        self.thr_inverted = thr_inverted      # DCS: code -1 is the most thrust
        self.curve_pow = curve_pow        # > 1: small commands do little, as DCS's did at 0.1

    def step(self, ax, dt=DT):
        def shape(c):
            return math.copysign(abs(c) ** self.curve_pow, c)
        roll, pitch, rud = shape(ax.get("roll", 0.0)), shape(ax.get("pitch", 0.0)), ax.get("rudder", 0.0)
        thr = ax.get("throttle")
        if thr is not None:
            self.thr = max(-1.0, min(1.0, -thr if self.thr_inverted else thr))
        self.p += (self.roll_sign * self.roll_rate * roll - self.p) * dt / self.roll_tau
        self.bank = wrap_pi(self.bank + self.p * dt)
        nz_t = 1.0 + self.pitch_sign * self.nz_per_cmd * pitch
        if self.push_limit_g is not None:
            nz_t = max(nz_t, 1.0 - self.push_limit_g)
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
            "eng": {"RPM": {"left": 68.0 + 34.0 * (self.thr_f + 1) / 2, "right": 0},        # one engine
                    "FuelConsumption": {"left": 0.13 + 1.3 * (self.thr_f + 1) / 2, "right": 0}},
            "ax": {"pitch": ax["pitch"], "roll": ax["roll"], "rudder": ax["rudder"],
                   "throttle": ax.get("throttle"), "seq": ax["seq"], "on": controlling},
        }


class FakeCockpit:
    """Enough of the F-16C's radar and stores to test the click, radar, lock and fire logic: A-A mode
    (ICP), a lock by target management up (after `scan_needed` radar cursor presses), master arm and weapon
    release. The target is an unarmed aircraft 60 km ahead of the start, not moving."""

    def __init__(self, defs, ac, scan_needed=0, tgt_range=60000.0):
        h = defs["commands"]["hotas"]
        d = defs["devices"]
        self.code = {(d["UFC"], defs["commands"]["ufc"]["AA"]): "aa",
                     (d["HOTAS"], h["STICK_TGT_MANAGE_UP"]): "tms_up",
                     (d["HOTAS"], h["THROTTLE_RDR_CURSOR_FWD"]): "cursor",
                     (d["HOTAS"], h["THROTTLE_RDR_CURSOR_AFT"]): "cursor_aft",
                     (d["HOTAS"], h["STICK_DISP_MANAGE_LEFT"]): "dms_left",
                     (d["HOTAS"], h["STICK_DISP_MANAGE_RIGHT"]): "dms_right",
                     (d["HOTAS"], h["STICK_WEAPON_RELEASE"]): "release",
                     (d["MMC"], defs["commands"]["mmc"]["MasterArmSw"]): "arm"}
        self.aa = self.armed = False
        self.stage = 0                     # 0 searching, 1 designated (bugged), 2 single-target track
        self.soi_left = True               # which MFD the cursor and target management act on
        self.cursor, self.scan_needed = 0, scan_needed
        self.stores = {str(i): {"count": 1, "CLSID": "{AIM-120C}"} for i in (1, 2, 8, 9)}
        self.missile = None
        self.tx = ac.x + tgt_range * math.cos(ac.hdg)
        self.tz = ac.z + tgt_range * math.sin(ac.hdg)
        self.ac0 = ac

    def press(self, dev, cmd, value):
        name = self.code.get((dev, cmd))
        if name == "dms_left" and value != 0.0:            # the keymap sends -1.0 for left and down, +1.0 for right and up
            self.soi_left = True
        elif name == "dms_right" and value != 0.0:
            self.soi_left = False
        elif name == "aa" and value == 1.0:
            self.aa = True
            self.soi_left = False          # A-A brings the weapons page up on the right MFD and the SOI goes with it
        elif name == "cursor" and value == 1.0 and self.soi_left:
            self.cursor += 1
        elif name == "cursor_aft" and value == 1.0 and self.soi_left:
            self.cursor -= 1
        elif name == "tms_up" and value == 1.0 and self.aa and self.soi_left and self.cursor >= self.scan_needed:
            self.stage = (self.stage + 1) % 3          # a third press drops the track
        elif name == "arm":
            self.armed = value == 1.0
        elif name == "release" and value == 1.0 and self.aa and self.armed and self.missile is None:   # no lock: a maddog shot
            for k, v in self.stores.items():
                if v["count"]:
                    v["count"] = 0
                    break
            self.missile = {"id": 99, "name": "AIM_120C", "unit": "", "coal": 2, "l1": 4, "l2": 4,
                            "x": self.ac0.x, "y": self.ac0.alt, "z": self.ac0.z}
        return name is not None

    @property
    def locked(self):
        return self.stage >= 1

    @locked.setter
    def locked(self, v):
        self.stage = 2 if v else 0

    def extras(self, ac):
        rng = math.hypot(self.tx - ac.x, self.tz - ac.z)
        tgt = {"id": 10, "name": "F-16C_50", "unit": "TGT-1", "coal": 1, "l1": 1, "l2": 1,
               "x": self.tx, "y": ac.alt, "z": self.tz}
        info = {"1": {"ID": 1, "distance": rng}}
        out = {"obj": [tgt] + ([self.missile] if self.missile else []),
               "radar": {"targets": info if self.aa else {}, "locked": info if self.locked else {}},
               "rd": {"fcr": 4, "tracked": self.stage >= 1, "cursor": self.stage < 2, "sam": self.stage == 1,
                      "limits": self.stage == 1, "td": self.stage == 1, "tll": self.stage == 2, "threat": self.aa,
                      "aa": self.aa, "msl120": self.aa},
               "ball": [self.missile] if self.missile else [],
               "pay": {"CurrentStation": 7, "Stations": self.stores}}
        return out


class FakeLink:
    """Same interface as stick_test.UdpLink, in lock step with a FakeAircraft."""

    def __init__(self, cockpit_defs=None, scan_needed=0, **kw):
        self.ac = FakeAircraft(**kw)
        self.cockpit = FakeCockpit(cockpit_defs, self.ac, scan_needed) if cockpit_defs else None
        self.ax = {"pitch": 0.0, "roll": 0.0, "rudder": 0.0, "throttle": None, "seq": 0}
        self.controlling = False
        self.n = 0
        self.sent = {}
        self.latencies = []
        self.events = {"hello": {"ev": "hello", "source": "frame", "codes": {"pitch": 2001, "roll": 2002, "rudder": 2003,
                                                          "throttle": 2004}, "codes_from": "fake"},
                       "stats": {"ev": "stats", "tel_hz": 50.0, "frame_hz": 60.0, "event_hz": 1.0, "frame_dt_max": 0.02}}
        self.seq = 0
        self.last_tel_wall = self.last_event_wall = 0.0
        self.ind_items = {}
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
        if self.cockpit and self.n % 5 == 0:
            d.update(self.cockpit.extras(self.ac))
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

    def ind(self, first=0, last=49):
        stage = self.cockpit.stage if self.cockpit else 0
        n_amraam = sum(v["count"] for v in self.cockpit.stores.values()) if self.cockpit else 0
        self.ind_items = {1: "-----------\nHUD_glass\nHUD_Boresight_Cross\n" + ("HUD_TD_Box\nHUD_ASEC\n" if stage else "")
                             + "%d MRM\n" % n_amraam + "-----------",
                          4: "-----------\nFCR_Antenna_Azimuth_Tic_0\nFCR_Mode\n" + ("STT\n" if stage == 2 else "RWS\n")
                             + "Cursor_Range\n%d\n" % (20 + 3 * (self.cockpit.cursor if self.cockpit else 0))
                             + ("FCR_TUC_contactData\nFCR_TUC_contact_Altitude\n23\n" if self.cockpit and self.cockpit.aa
                                and self.cockpit.cursor == self.cockpit.scan_needed else "") + "-----------"}
        self.events["ind_end"] = {"ev": "ind_end", "count": 2}

    def click(self, dev, cmd, value, mode=None):
        ok = self.cockpit.press(dev, cmd, value) if self.cockpit else False
        self.events["click"] = {"ev": "click", "ok": True, "dev": dev, "cmd": cmd, "value": value, "known": ok}

    def info(self):
        pass

    def ping(self):
        return 0.0

    def close(self):
        pass
