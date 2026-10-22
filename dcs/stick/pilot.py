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
        self.overridden = False       # the DOGFIGHT switch was put to MISSILE OVERRIDE (back to centre at the end)
        self.icp_aa_tried = False     # the ICP A-A button was pressed at least once
        self.aa_said = None           # the A-A state last reported
        self.center_az = None         # the bearing the cursor was last put on
        # chaff: the main thread says when (chaff()); the frame callback presses CMS forward every CHAFF_EVERY s
        # --point-before-fire (dcs_live.py): a fire job first turns the nose to within point_deg of the target
        # and holds it there through the release; None: it holds the nose where it was at the request
        self.point_deg = None
        self._point_hdg = None
        # --tws-scan (dcs_live.py): the TWS scan to set, (azimuth A-number, bars), e.g. (3, 4) for A3 4B; None leaves
        # DCS's A6 4B. A6 4B took 15-20 s to make a track of a target nose-on in the scan (22 October)
        self.tws_scan = None
        self.tws_scan_done = False
        self.chaff_on = False
        self.cmds_tried = False       # the countermeasures panel set to MAN, program 1, chaff on (once)
        self.cmds_ok = False
        self.cms_presses = 0
        self._cms_down = None         # DCS time the CMS switch went forward (released after CHAFF_HOLD)
        self._cms_last = -1e9
        self.chaff_first = self.chaff_last = None   # the jet's chaff count at the first press, and the latest
        self._chaff_said = False
        # timings, DCS seconds: each fire job {"bug_s", "total_s", "ok"}, each timed bug {"s", "ok", ...} (fire_delay.py)
        self.fire_times, self.bug_times = [], []
        # the cockpit through the fight (cockpit_summary): display samples, those with the HUD in A-A, and the first
        # bugged track (DCS time, range m)
        self.rd_frames = self.aa_frames = 0
        self.first_bug = None
        self.obj_full_said = False    # said once: the object list held only our chaff and flares
        self.bug_page_saved = False   # the whole radar page saved once, right after a bug

    # ── what the main thread calls ──────────────────────────────────────
    def command(self, hdg, alt, spd, dive=None, climb=None):
        """The policy's command; dive and climb are its flight-path limits (the simulator's altFPA and climbFPA):
        with them the altitude is flown with the simulator's law."""
        self.target = (float(hdg), float(alt), float(spd), dive, climb)

    def radar(self):
        self.jobs.put("radar")

    def fire(self):
        self.jobs.put("fire")

    def chaff(self, on):
        """Dispense chaff while on (a missile inbound: dcs_live.py decides)."""
        self.chaff_on = bool(on)

    def time_bug(self):
        """Drop the bug, then time a bug from there as a fire job does it, without a launch (fire_delay.py)."""
        self.jobs.put("bugtime")

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
        until the first one comes. Every frame also runs the chaff switch (_cms_tick), so chaff goes out
        during a radar or fire job too."""
        self._cms_tick(f)
        self._cockpit_tick(f)
        if self.target is None:
            if f.hdg is None or f.alt is None:
                return None
            self.target = (f.crs if f.crs is not None else f.hdg, f.alt, f.tas or 250.0, None, None)
        hdg, alt, v, dive, climb = self.target
        if self._point_hdg is not None:          # a fire job is pointing the nose at the target
            hdg = self._point_hdg
        return self.hold.axes(f, hdg, alt, v=v, dive=dive, climb=climb)

    def _cockpit_tick(self, f):
        rd = f.raw.get("rd")
        if rd is None:
            return
        self.rd_frames += 1
        self.aa_frames += bool(rd.get("aa"))
        if self.first_bug is None and rd.get("tracked"):
            fo = cockpit.last_with(self.r.frames[-200:], "obj")
            nr = cockpit.nearest(fo) if fo else None
            self.first_bug = (f.t, None if nr is None else nr[1])

    def cockpit_summary(self):
        """{"aa_pct": share of the display samples with the HUD in A-A, %; "first_bug_km", "first_bug_t": the range
        and DCS time of the first bugged track ("" if none)}."""
        fb = self.first_bug
        return {"aa_pct": round(100.0 * self.aa_frames / self.rd_frames) if self.rd_frames else "",
                "first_bug_km": "" if fb is None or fb[1] is None else round(fb[1] / 1000.0, 1),
                "first_bug_t": "" if fb is None else round(fb[0], 1)}

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
                    elif job == "bugtime":
                        self.busy = "bugtime"
                        self._bugtime()
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
                if self._cms_down is not None:
                    dev, cmd, _, up = cockpit.resolve("hotas.STICK_CMS_MANAGE_FWD", self.defs)
                    self.link.click(dev, cmd, up, "s")
                if self.overridden:
                    cockpit.click(self.r, "hotas.THROTTLE_DOG_FIGHT", self.defs, value=0.0, verbose=False)
            except Exception:
                pass
            self.link.release()
            self.r.close()

    # ── chaff ───────────────────────────────────────────────────────────
    CHAFF_EVERY = 1.5          # s between presses of CMS forward (each dispenses the MAN program once)
    CHAFF_HOLD = 0.3           # s the switch is held forward

    def _cmds_setup(self):
        """The countermeasures panel to MAN, PROGRAM 1 and the CH (chaff) switch ON, once: CMS forward then
        dispenses manual program 1. Switch positions, so a repeat does no harm. bvr_stick_export.lua version 2
        lets these three panel controls through (it refuses the rest of the panel)."""
        r, d, fn = self.r, self.defs, self.fn
        self.cmds_tried = True
        res = []
        for spec, value in (("cmds.Mode", 0.2), ("cmds.Prgm", 0.1), ("cmds.ChExp", 1.0)):
            res.append(cockpit.click(r, spec, d, value=value, fn=fn, verbose=False))
        refused = [x for x in res if isinstance(x, dict) and x.get("err") == "refused"]
        self.cmds_ok = not refused
        fc = cockpit.last_with(r.frames[-60:], "cm")
        cm = fc.raw.get("cm") if fc else None
        self.events.put(("chaff", "setup" if self.cmds_ok else "refused",
                         "countermeasures panel MAN, program 1, CH on"
                         + (f" ({cm.get('chaff')} chaff, {cm.get('flare')} flares)" if isinstance(cm, dict) else "")
                         if self.cmds_ok else
                         "the export refused the countermeasures panel: install bvr_stick_export.lua version 2 "
                         "(python dcs\\stick\\install_export.py); CMS forward is pressed anyway"))

    def _chaff_count(self):
        fc = cockpit.last_with(self.r.frames[-30:], "cm")
        cm = fc.raw.get("cm") if fc else None
        return cm.get("chaff") if isinstance(cm, dict) else None

    def _cms_tick(self, f):
        """CMS forward for CHAFF_HOLD s every CHAFF_EVERY s while chaff is on, straight on the link (no settle:
        it runs inside the frame callback); says once whether the jet's chaff count goes down."""
        t = f.t
        if t is None:
            return
        dev, cmd, down, up = cockpit.resolve("hotas.STICK_CMS_MANAGE_FWD", self.defs)
        if self._cms_down is not None:
            if t - self._cms_down >= self.CHAFF_HOLD:
                self.link.click(dev, cmd, up, "s")
                self._cms_down = None
            return
        if not self.chaff_on or t - self._cms_last < self.CHAFF_EVERY:
            return
        n = self._chaff_count()
        if self.chaff_first is None:
            self.chaff_first = n
        if n is not None:
            self.chaff_last = n
        if not self._chaff_said and self.cms_presses >= 1:
            if n is not None and self.chaff_first is not None and n < self.chaff_first:
                self._chaff_said = True
                self.events.put(("chaff", "ok", f"dispensing (chaff {self.chaff_first} -> {n})"))
            elif self.cms_presses >= 4:
                self._chaff_said = True
                self.events.put(("chaff", "none", f"{self.cms_presses} CMS presses and " + (
                    f"the chaff count is still {n}: the countermeasures panel did not dispense"
                    if n is not None else "no chaff count from the export (bvr_stick_export.lua before version 2)")))
        self.link.click(dev, cmd, down, "s")
        self._cms_down = self._cms_last = t
        self.cms_presses += 1

    # ── jobs ────────────────────────────────────────────────────────────
    def _target_az(self):
        fo = cockpit.last_with(self.r.frames[-200:], "obj")
        nr = cockpit.nearest(fo) if fo else None
        if nr is None and fo is not None and not self.obj_full_said and len(fo.raw.get("obj") or []) >= 12:
            # bvr_stick_export.lua before version 3 sends the 12 nearest objects, our chaff and flares included
            self.obj_full_said = True
            h = self.link.events.get("hello") or {}
            self.events.put(("radar", "no_target", "the export's object list is full of our own chaff and flares and "
                             "holds no aircraft" + ("" if (h.get("version") or 0) >= 3 else
                                                    ": install bvr_stick_export.lua version 3 "
                                                    "(python dcs\\stick\\install_export.py)")))
        return None if nr is None else nr[2]

    def _state(self):
        fd = cockpit.last_with(self.r.frames[-200:], "rd")
        return cockpit.lock_state(fd) if fd else None

    def _page(self):
        return cockpit.fcr_info(cockpit.ind_text(self.r, self.link, cockpit._fcr_id(self.r)))

    PREBUG_S = 6.0             # s of target management presses a radar job spends bugging the track

    def _radar(self, force=False, prebug=True):
        """A-A, TWS, the cursor on the target's bearing, and the track bugged (a pilot bugs the target before
        deciding to shoot: a fire request then waits only for the release, 7-14 s less in DCS). Quiet when all is
        already so."""
        r, d, fn = self.r, self.defs, self.fn
        cockpit.settle(r, 0.3, fn)
        self._aa()
        if not self.cmds_tried:
            self._cmds_setup()
        # Target management right held toggles RWS <-> TWS, so it is held only when the page says RWS: a read that
        # came back empty (mode None) once held it in TWS and the radar went to RWS for 29 s
        # (live_20261021_142940). An empty read is read once more; still nothing, the radar is left as it is.
        info = self._page()
        if info["mode"] is None:
            cockpit.settle(r, 0.3, fn)
            info = self._page()
        if info["mode"] == "RWS":
            cockpit.click(r, "hotas.STICK_TGT_MANAGE_RIGHT", d, hold=1.5, fn=fn, verbose=False)
            cockpit.settle(r, 1.0, fn)
            after = self._page()["mode"]
            self.events.put(("radar", "tws" if after == "TWS" else "no_tws", f"radar RWS -> {after}"))
        elif info["mode"] is None:
            self.events.put(("radar", "page_unread", "the radar page did not answer: the mode is left as it is"))
        if self.tws_scan and not self.tws_scan_done and (info["mode"] == "TWS" or self._page()["mode"] == "TWS"):
            self._set_scan()
        if self._state() not in ("designated", "stt"):
            az = self._target_az()
            # the cursor goes to the target's bearing, again only when that has moved by 10 deg: each move swings
            # the 25 deg scan away for a moment
            if az is not None and abs(az) < 60.0 and (force or self.center_az is None
                                                       or abs(az - self.center_az) > 10.0):
                cockpit.center_cursor(r, d, fn, az, verbose=False)
                self.center_az = az
                # where the cursor should now be, for the log: no readout says where it is (in two fights on 21 and
                # 22 October the target became buggable only as it drifted 5-7 deg further out than asked)
                back = max(0.0, (60.0 - min(58.0, abs(az))) / cockpit.CURSOR_AZ_RATE)
                r.note({"ev": "cursor", "az": round(az, 1), "back_s": round(back, 2)})
                self.events.put(("radar", "cursor", f"cursor to {az:+.0f} deg (edge, then {back:.2f} s back)"))
            if prebug and az is not None and abs(az) < 50.0 and self._press_bug(self.PREBUG_S):
                self.events.put(("radar", "bugged", f"track bugged ahead of a shot (target {az:+.0f} deg)"))
                if not self.armed:                   # and master arm on, as a pilot does before the shot
                    cockpit.click(r, "mmc.MasterArmSw", d, value=1.0, fn=fn, verbose=False)
                    cockpit.settle(r, 0.5, fn)
                    self.armed = True

    SCAN_OSB = {"az": 18, "bars": 17}   # the radar page's azimuth and bar-scan buttons (left MFD; each press steps)

    def _set_scan(self):
        """The TWS scan to tws_scan, by stepping the azimuth and bar buttons and reading the page back after each
        press; a press that changes nothing ends the try (the button is not the one thought), and it is not tried
        again in the fight."""
        r, d, fn = self.r, self.defs, self.fn
        self.tws_scan_done = True
        info = self._page()
        for key, want in (("az", self.tws_scan[0]), ("bars", self.tws_scan[1])):
            for _ in range(4):
                if info[key] == want or info[key] is None:
                    break
                before = info[key]
                cockpit.click(r, f"left_osb.{self.SCAN_OSB[key]}", d, fn=fn, verbose=False)
                cockpit.settle(r, 0.5, fn)
                info = self._page()
                if info[key] == before:
                    self.events.put(("radar", "scan_failed", f"OSB {self.SCAN_OSB[key]} did not change the scan's "
                                     f"{key} ({before}): left as it is"))
                    return
        r.note({"ev": "scan", "az": info["az"], "bars": info["bars"]})
        self.events.put(("radar", "scan", f"TWS scan A{info['az']} {info['bars']}B"))

    def _aa(self):
        """The A-A master mode, so that weapon release launches the AIM-120: the throttle's DOGFIGHT switch to MISSILE
        OVERRIDE (a switch position, through the HOTAS route that target management and weapon release take), and if
        the HUD still draws no A-A, the switch through centre to override again, then the ICP A-A button. In a live run
        (15 October) the ICP press alone left the HUD in NAV for the whole fight; on 21 October (live_..._090924) the
        override click left it in NAV for 125 s, and every fire request in that time timed out. So each call tries
        again while the HUD says NAV; the ICP button toggles, so it is pressed only when the HUD has just said NAV.
        True when the HUD shows A-A, False when it does not, None when there is no display data."""
        r, d, fn = self.r, self.defs, self.fn
        mm = cockpit.master_mode(cockpit.last_with(r.frames[-200:], "rd"))
        if mm is not False:
            return mm
        how = "missile override (DOGFIGHT switch aft)"
        cockpit.click(r, "hotas.THROTTLE_DOG_FIGHT", d, value=-1.0, fn=fn, verbose=False)
        self.overridden = True
        cockpit.settle(r, 1.0, fn)
        mm = cockpit.master_mode(cockpit.last_with(r.frames[-30:], "rd"))
        if mm is False:
            how = "missile override, the switch cycled through centre"
            cockpit.click(r, "hotas.THROTTLE_DOG_FIGHT", d, value=0.0, fn=fn, verbose=False)
            cockpit.settle(r, 0.5, fn)
            cockpit.click(r, "hotas.THROTTLE_DOG_FIGHT", d, value=-1.0, fn=fn, verbose=False)
            cockpit.settle(r, 1.0, fn)
            mm = cockpit.master_mode(cockpit.last_with(r.frames[-30:], "rd"))
        if mm is False:
            self.icp_aa_tried = True
            how = "ICP A-A button"
            cockpit.click(r, "ufc.AA", d, fn=fn, verbose=False)
            cockpit.settle(r, 1.0, fn)
            mm = cockpit.master_mode(cockpit.last_with(r.frames[-30:], "rd"))
        if mm != self.aa_said or mm is False:
            self.aa_said = mm
            self.events.put(("radar", "aa" if mm else "no_aa",
                             f"A-A master mode by the {how}" if mm else
                             "the HUD is still not in A-A after missile override (cycled) and the ICP A-A button: "
                             "weapon release will launch nothing; trying again at the next radar or fire job"))
        return mm

    def _bug(self, max_secs=10.0):
        """Target management right every 2 s until the track is bugged (the launch zone is drawn or the
        markers say designated); the cursor goes to the target's bearing first."""
        if self._state() in ("designated", "stt"):
            return True
        self._radar(force=True, prebug=False)
        return self._press_bug(max_secs)

    def _press_bug(self, max_secs):
        """Target management right every 2 s until the track is bugged; True once it is."""
        r, d, fn = self.r, self.defs, self.fn
        if self._state() in ("designated", "stt"):
            return True
        t0 = r.frames[-1].t
        while r.frames[-1].t - t0 < max_secs:
            cockpit.click(r, "hotas.STICK_TGT_MANAGE_RIGHT", d, fn=fn, verbose=False)
            for _ in range(4):
                cockpit.settle(r, 0.5, fn)
                if self._state() in ("designated", "stt"):
                    # the page once bugged: do its contact and track lists show the target? (they read 0 at every
                    # attempt so far, even 0.5 s before a bug: if they stay 0 here, they are not where tracks go)
                    page = cockpit.ind_text(r, self.link, cockpit._fcr_id(r))
                    info = cockpit.fcr_info(page)
                    # the whole page too, once: how DCS lists a real track (the counters read 0 so far)
                    r.note({"ev": "bug_ok", "mode": info["mode"], "contacts": info["contacts"],
                            "tracks": info["tracks"], **({} if self.bug_page_saved else {"lines": page})})
                    self.bug_page_saved = True
                    return True
                lines = cockpit.ind_text(r, self.link, cockpit._fcr_id(r))
                info = cockpit.fcr_info(lines)
                az = self._target_az()
                r.note({"ev": "bug_try", "mode": info["mode"], "contacts": info["contacts"], "tracks": info["tracks"],
                        "target_az": None if az is None else round(az, 1)})
                if any(x.strip().startswith("DLZ_AIM120") for x in lines):
                    r.note({"ev": "bug_ok", "mode": info["mode"], "contacts": info["contacts"],
                            "tracks": info["tracks"], "by": "DLZ", **({} if self.bug_page_saved else {"lines": lines})})
                    self.bug_page_saved = True
                    return True
        return False

    def _bugtime(self):
        r, d, fn = self.r, self.defs, self.fn
        for _ in range(2):                       # target management down drops the bug
            if self._state() not in ("designated", "stt"):
                break
            cockpit.click(r, "hotas.STICK_TGT_MANAGE_DOWN", d, fn=fn, verbose=False)
            cockpit.settle(r, 1.5, fn)
        mode = self._page()["mode"]
        az = self._target_az()
        t0 = r.frames[-1].t
        ok = self._bug()
        dt = r.frames[-1].t - t0
        self.bug_times.append({"s": round(dt, 2), "ok": ok, "mode_before": mode,
                               "az": None if az is None else round(az, 1)})
        self.events.put(("bug", "ok" if ok else "timeout", f"{dt:.1f} s (radar {mode} before, target at "
                                                            f"{'?' if az is None else f'{az:+.0f}'} deg)"))

    POINT_MAX_S = 10.0         # s a fire job turns the nose toward the target before it gives up and fires anyway

    def _point(self):
        """Turn the nose to within point_deg of the target (the nearest aircraft), up to POINT_MAX_S; the heading
        stays on it (_point_hdg, read by the frame callback) until the fire job ends."""
        r, fn = self.r, self.fn
        t0 = r.frames[-1].t
        az = self._target_az()
        az0 = az
        while az is not None and abs(az) > self.point_deg and r.frames[-1].t - t0 < self.POINT_MAX_S:
            f = r.frames[-1]
            self._point_hdg = (f.hdg or 0.0) + math.radians(az)
            cockpit.settle(r, 0.5, fn)
            az = self._target_az()
        if az is not None:
            self._point_hdg = (r.frames[-1].hdg or 0.0) + math.radians(az)
        if az0 is not None:
            self.events.put(("point", "ok" if az is not None and abs(az) <= self.point_deg else "timeout",
                             f"nose to the target before the release: {az0:+.0f} -> "
                             f"{'?' if az is None else f'{az:+.0f}'} deg in {r.frames[-1].t - t0:.1f} s"))

    def _fire(self):
        try:
            self._fire_job()
        finally:
            self._point_hdg = None

    def _fire_job(self):
        r, d, fn = self.r, self.defs, self.fn
        t0 = r.frames[-1].t
        aa = self._aa()
        if self.point_deg:
            self._point()
        else:
            # Hold the nose where it was at the request through the bug and the release (SIM_REV 20): the
            # policy kept cranking during dcsfit_v5's fire jobs in DCS, the target drifted from 32-36 to
            # 45-47 deg off the nose in the 10 s the bug may take, out of the TWS scan, and none was bugged.
            self._point_hdg = r.frames[-1].hdg
        if not self._bug():
            self.fire_times.append({"bug_s": None, "total_s": round(r.frames[-1].t - t0, 2), "ok": False})
            # the radar page as it was, whole, into the stick log (ev "fcr_page"), and its summary into the reason
            lines = cockpit.ind_text(r, self.link, cockpit._fcr_id(r))
            info = cockpit.fcr_info(lines)
            az = self._target_az()
            r.note({"ev": "fcr_page", "why": "bug timeout", "target_az": None if az is None else round(az, 1),
                    "cursor_az": self.center_az, "info": info, "lines": lines})
            page = (f"radar page {info['mode']} A{info['az']} {info['bars']}B {info['scale']} NM, "
                    f"{info['contacts']} contacts, {info['tracks']} tracks" if lines else "radar page did not answer")
            self.events.put(("fire", "timeout", "no TWS track bugged in 10 s"
                             + ("; the HUD was not in A-A (NAV)" if aa is False else "")
                             + f"; {page}; target {'?' if az is None else f'{az:+.0f}'} deg, cursor put at "
                             + ("?" if self.center_az is None else f"{self.center_az:+.0f}") + " deg"))
            return
        t_bug = r.frames[-1].t
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
            me = fo2.raw.get("me") if fo2 else None
            new = [o for o in ((fo2.raw.get("obj") or []) if fo2 else [])
                   if o.get("id") not in ids0 and self._own_missile(o, me)]
            mrm1 = cockpit.hud_mrm(r, self.link)
            if new or (mrm0 is not None and mrm1 is not None and mrm1 < mrm0):
                self.fire_times.append({"bug_s": round(t_bug - t0, 2), "total_s": round(r.frames[-1].t - t0, 2),
                                        "ok": True})
                self.events.put(("fire", "launched", f"HUD {mrm0} -> {mrm1} MRM"
                                 + (f", {new[0].get('name')} off the rail" if new else "")))
                return
        self.fire_times.append({"bug_s": round(t_bug - t0, 2), "total_s": round(r.frames[-1].t - t0, 2), "ok": False})
        rd = cockpit.last_with(r.frames[-200:], "rd")
        rd = rd.raw.get("rd") if rd else None
        hud = ("no HUD data" if not rd else
               f"HUD A-A {'yes' if rd.get('aa') else 'NO'}, AIM-120 selected {'yes' if rd.get('msl120') else 'NO'}")
        self.events.put(("fire", "timeout", f"no missile left the rail (HUD {mrm0} MRM; {hud}; radar {self._state()})"))

    @staticmethod
    def _own_missile(o, me, near=5000.0):
        """A missile (level 4 4) of our own side within `near` m of us: one just off our rail. Any other new
        weapon object (the bandit's missile, chaff, a flare) is not our launch: a live run once took one for it."""
        if o.get("l1") != 4 or o.get("l2") != 4:
            return False
        if me and o.get("coal") is not None and me.get("coal") is not None and o["coal"] != me["coal"]:
            return False
        if me and None not in (o.get("x"), o.get("z"), me.get("x"), me.get("z")):
            return math.hypot(o["x"] - me["x"], o["z"] - me["z"]) < near
        return True
