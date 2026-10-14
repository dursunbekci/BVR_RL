"""
pilot.py  —  the policy's hands on the player's F-16C, through bvr_stick_export.lua
===================================================================================

dcs_live.py --stick flies the player's Viper instead of telling the DCS AI where to go. The mission's
bridge (bvr_bridge.lua) still reports both aircraft, shots, hits and kills and flies red; the policy's
1 Hz commands (heading, altitude, speed, fire) go to a StickPilot, a thread that:

  * flies them at the Export link's rate (50 Hz) with stick_test.Hold, the inner loop measured in DCS
    (heading -> bank with a roll-rate lead, altitude -> flight-path angle -> g, speed -> throttle),
    with the simulator's limits: 80 deg of bank, up to 8 g;
  * runs cockpit jobs between frames, the aircraft flown all the while:
      radar: A-A master mode, the radar to TWS (target management right held), the cursor to the
             target's bearing (TWS scans 25 deg either side of the cursor) - the track file is then
             built in the background, as measured in DCS (about 13 s from RWS);
      fire:  bug the TWS track (target management right until the AIM-120 launch zone is drawn),
             master arm on, weapon release held 1.5 s, and the launch confirmed by the HUD's
             missile count or a new AIM-120 in the world objects.

Its results come back as events (("fire", "launched"|"timeout", reason), ...) for dcs_live.py's fire
bookkeeping. Everything measured here is in README.md (results log).
"""

import math
import os
import queue
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import cockpit                                      # noqa: E402
from stick_test import Hold, Limits, Run, Abort, drive, load_gains   # noqa: E402

GAINS_PATH = os.path.join(HERE, "..", "..", "dcs_runs", "stick_gains.json")

BANK_MAX_DEG = 80.0        # the simulator's PHI_MAX: the policy was trained on its turn rates
NZ_MAX = 8.0               # g; the simulator allows 9 (and less where the wing cannot give it)
VS_MAX = 80.0              # m/s climb or descent while changing altitude


def live_limits():
    """Wider than the stick tests': a fight goes where a test does not. Still a floor and a stall guard."""
    L = Limits(min_agl=300.0, min_alt=300.0)
    L.bank, L.pitch = math.radians(179.0), math.radians(75.0)
    L.tas_min, L.nz_lo, L.nz_hi = 80.0, -3.5, 9.5
    return L


class StickPilot(threading.Thread):
    """Flies the latest command at the link's rate and runs cockpit jobs; see the module's text."""

    KEEP_FRAMES = 3000           # the run's frame list is trimmed to this many

    def __init__(self, link, gains=None, defs=None, log=print, limits=None, out_dir=None, airframe=None):
        super().__init__(daemon=True, name="StickPilot")
        self.link = link
        self.gains = gains if gains is not None else load_gains(GAINS_PATH)
        need = ("roll_sign", "pitch_sign", "thr_sign", "roll_curve", "pitch_curve")
        missing = [k for k in need if not self.gains.get(k)]
        if missing:
            raise SystemExit(f"stick gains are missing {missing}: run dcs/stick/stick_test.py all first "
                             f"(it writes {GAINS_PATH})")
        self.defs = defs if defs is not None else cockpit.load_defs()
        self.log = log
        self.r = Run(link, out_dir, "live", limits or live_limits())
        # airframe: the policy's platform's (its g and bank limits are the simulator's, as in training)
        self.hold = Hold(self.gains, bank_max=math.radians(BANK_MAX_DEG), nz_max=NZ_MAX, vs_max=VS_MAX,
                         airframe=airframe)
        self.target = None            # (heading rad, altitude m, speed m/s, dive rad, climb rad) from the policy
        self.jobs = queue.Queue()
        self.events = queue.Queue()   # ("fire", status, reason) / ("radar", status, text)
        self.busy = None              # the job running now ("radar", "fire") or None
        self._halt = threading.Event()
        self.aborts = 0
        self.armed = False
        self.center_az = None         # the bearing the cursor was last put on

    # ── what the main thread calls ──────────────────────────────────────
    def command(self, hdg, alt, spd, dive=None, climb=None):
        """The policy's command; dive and climb are its flight-path limits (the simulator's altFPA and climbFPA):
        with them the altitude is flown with the simulator's law."""
        self.target = (float(hdg), float(alt), float(spd), dive, climb)

    def radar(self):
        self.jobs.put("radar")

    def fire(self):
        self.jobs.put("fire")

    def stop(self, timeout=5.0):
        self._halt.set()
        self.join(timeout)

    def drain_events(self):
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    # ── the thread ──────────────────────────────────────────────────────
    def fn(self, f, tau):
        """The axes for frame f: the policy's command through the inner loop, or the present state held
        until the first one comes."""
        if self.target is None:
            if f.hdg is None or f.alt is None:
                return None
            self.target = (f.hdg, f.alt, f.tas or 250.0, None, None)
        hdg, alt, v, dive, climb = self.target
        return self.hold.axes(f, hdg, alt, v=v, dive=dive, climb=climb)

    def run(self):
        try:
            while not self._halt.is_set():
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    job = None
                try:
                    if job == "radar":
                        self.busy = "radar"
                        self._radar()
                    elif job == "fire":
                        self.busy = "fire"
                        self._fire()
                    else:
                        drive(self.r, 0.1, self.fn)
                except Abort as e:
                    # a pause, a stall guard, a frozen DCS: let go of the stick, say so, and try again
                    self.aborts += 1
                    self.log(f"  stick pilot: {e}: axes released; trying again")
                    self.link.release()
                    if job == "fire":
                        self.events.put(("fire", "timeout", f"interrupted: {e}"))
                    time.sleep(0.5)
                except Exception as e:                        # a bug must not leave the stick held
                    self.log(f"  stick pilot: {type(e).__name__}: {e}")
                    self.link.release()
                    if job == "fire":
                        self.events.put(("fire", "timeout", f"{type(e).__name__}: {e}"))
                    time.sleep(0.5)
                finally:
                    self.busy = None
                if len(self.r.frames) > self.KEEP_FRAMES:
                    del self.r.frames[:len(self.r.frames) - self.KEEP_FRAMES // 2]
        finally:
            try:
                if self.armed:
                    cockpit.click(self.r, "mmc.MasterArmSw", self.defs, value=0.0, verbose=False)
            except Exception:
                pass
            self.link.release()
            self.r.close()

    # ── jobs ────────────────────────────────────────────────────────────
    def _target_az(self):
        fo = cockpit.last_with(self.r.frames[-200:], "obj")
        nr = cockpit.nearest(fo) if fo else None
        return None if nr is None else nr[2]

    def _state(self):
        fd = cockpit.last_with(self.r.frames[-200:], "rd")
        return cockpit.lock_state(fd) if fd else None

    def _page(self):
        return cockpit.fcr_info(cockpit.ind_text(self.r, self.link, cockpit._fcr_id(self.r)))

    def _radar(self, force=False):
        """A-A, TWS, the cursor on the target's bearing. Quiet when all is already so."""
        r, d, fn = self.r, self.defs, self.fn
        cockpit.settle(r, 0.3, fn)
        if cockpit.master_mode(cockpit.last_with(r.frames[-200:], "rd")) is False:
            cockpit.click(r, "ufc.AA", d, fn=fn, verbose=False)
            cockpit.settle(r, 1.0, fn)
        info = self._page()
        if info["mode"] != "TWS":
            cockpit.click(r, "hotas.STICK_TGT_MANAGE_RIGHT", d, hold=1.5, fn=fn, verbose=False)
            cockpit.settle(r, 1.0, fn)
            info = self._page()
            self.events.put(("radar", "tws" if info["mode"] == "TWS" else "no_tws", f"radar {info['mode']}"))
        if self._state() not in ("designated", "stt"):
            az = self._target_az()
            # the cursor goes to the target's bearing, again only when that has moved by 10 deg: each move swings
            # the 25 deg scan away for a moment
            if az is not None and abs(az) < 60.0 and (force or self.center_az is None
                                                       or abs(az - self.center_az) > 10.0):
                cockpit.center_cursor(r, d, fn, az, verbose=False)
                self.center_az = az

    def _bug(self, max_secs=10.0):
        """Target management right every 2 s until the track is bugged (the launch zone is drawn or the
        markers say designated); the cursor goes to the target's bearing first."""
        r, d, fn = self.r, self.defs, self.fn
        if self._state() in ("designated", "stt"):
            return True
        self._radar(force=True)
        t0 = r.frames[-1].t
        while r.frames[-1].t - t0 < max_secs:
            cockpit.click(r, "hotas.STICK_TGT_MANAGE_RIGHT", d, fn=fn, verbose=False)
            for _ in range(4):
                cockpit.settle(r, 0.5, fn)
                if self._state() in ("designated", "stt"):
                    return True
                lines = cockpit.ind_text(r, self.link, cockpit._fcr_id(r))
                if any(x.strip().startswith("DLZ_AIM120") for x in lines):
                    return True
        return False

    def _fire(self):
        r, d, fn = self.r, self.defs, self.fn
        if not self._bug():
            self.events.put(("fire", "timeout", "no TWS track bugged in 10 s"))
            return
        mrm0 = cockpit.hud_mrm(r, self.link)
        fo = cockpit.last_with(r.frames[-200:], "obj")
        ids0 = cockpit._ids(fo, "obj")
        if not self.armed:
            cockpit.click(r, "mmc.MasterArmSw", d, value=1.0, fn=fn, verbose=False)
            cockpit.settle(r, 0.5, fn)
            self.armed = True
        # held until the missile leaves the rail: 0.4 s launched nothing, 1.5 s did (README)
        cockpit.click(r, "hotas.STICK_WEAPON_RELEASE", d, hold=1.5, fn=fn, verbose=False)
        for _ in range(6):
            cockpit.settle(r, 0.5, fn)
            fo2 = cockpit.last_with(r.frames[-60:], "obj")
            new = [o for o in ((fo2.raw.get("obj") or []) if fo2 else [])
                   if o.get("id") not in ids0 and o.get("l1") == 4]
            mrm1 = cockpit.hud_mrm(r, self.link)
            if new or (mrm0 is not None and mrm1 is not None and mrm1 < mrm0):
                self.events.put(("fire", "launched", f"HUD {mrm0} -> {mrm1} MRM"))
                return
        self.events.put(("fire", "timeout", f"no missile left the rail (HUD {mrm0} MRM)"))
