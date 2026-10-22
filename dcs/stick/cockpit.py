"""
cockpit.py  --  the F-16C's switches and HOTAS through Export.lua: click, radar, lock, fire
============================================================================================

The Viper's controls are (device, command, value) triples, which bvr_stick_export.lua passes to
the cockpit as `CLICK <device> <command> <value>` (the same call the module's own key bindings
make). f16c_defs.json holds the numbers and the value a key binding sends on press and release
(dcs/stick/extract_f16_defs.py made it from the module's own files).

    python dcs/stick/stick_test.py click hotas.STICK_TGT_MANAGE_UP        press and release
    python dcs/stick/stick_test.py click ufc.AA                           A-A master mode
    python dcs/stick/stick_test.py click mmc.MasterArmSw --value 1        a switch position
    python dcs/stick/stick_test.py click 16 3010 1.0                      or by numbers
    python dcs/stick/stick_test.py radar                                  what the radar reports (lock by hand)
    python dcs/stick/stick_test.py lock                                   try to lock the target
    python dcs/stick/stick_test.py fire                                   arm, release a missile, watch it go

lock and fire are experiments: what the radar and the stores system need is learned by trying, and
what happens after each step is printed. They work in the stick-test mission with a target
(`python dcs\\make_mission.py --stick-test --target`: one unarmed AI aircraft ahead).
"""

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEG = 180.0 / math.pi
GROUP_DEVICE = {"hotas": "HOTAS", "mmc": "MMC", "ufc": "UFC", "sms": "SMS", "fcr": "FCR",
                "control": "CONTROL_INTERFACE", "cmds": "CMDS", "rwr": "RWR"}


def load_defs(path=None):
    with open(path or os.path.join(HERE, "f16c_defs.json")) as fh:
        return json.load(fh)


def resolve(spec, defs, value=None):
    """'hotas.STICK_TGT_MANAGE_UP' (case of the group is free) or ['16', '3010'] -> (device, command, down, up).
    down is the value sent on press, up the value on release (None: no release, a switch position)."""
    if isinstance(spec, (list, tuple)):
        dev, cmd = int(spec[0]), int(spec[1])
        return dev, cmd, 1.0 if value is None else float(value), (0.0 if value is None else None)
    grp, _, name = spec.partition(".")
    grp = grp.lower()
    if grp == "mfd" or grp in ("mfd_left", "mfd_right"):
        raise ValueError("MFD buttons: click left_osb.N or right_osb.N (an MFD OSB number 1-20)")
    if grp in ("left_osb", "right_osb"):
        dev = defs["devices"]["MFD_LEFT" if grp == "left_osb" else "MFD_RIGHT"]
        cmd = defs["commands"]["mfd"][f"OSB_{int(name)}"]
        return dev, cmd, 1.0 if value is None else float(value), (0.0 if value is None else None)
    if grp not in GROUP_DEVICE or name not in defs["commands"].get(grp, {}):
        known = sorted(f"{g}.{n}" for g in GROUP_DEVICE for n in defs["commands"].get(g, {}))
        raise ValueError(f"unknown control {spec!r}; groups: {', '.join(GROUP_DEVICE)} (see f16c_defs.json); "
                         f"e.g. {', '.join(known[:4])}")
    dev, cmd = defs["devices"][GROUP_DEVICE[grp]], defs["commands"][grp][name]
    keys = defs["keys"].get(f"{GROUP_DEVICE[grp]}.{name}", [])
    if value is not None:
        return dev, cmd, float(value), None if any(k["up"] is None for k in keys) or not keys else 0.0
    if len(keys) == 1:
        return dev, cmd, keys[0]["down"], keys[0]["up"]
    if len(keys) > 1:
        opts = "; ".join(f"{k['down']:+.1f} = {k['name']}" for k in keys)
        raise ValueError(f"{spec} is a switch with several positions: give --value ({opts})")
    return dev, cmd, 1.0, 0.0


def settle(run, secs, fn=None):
    """Let `secs` of model time pass, flying `fn` (frame, tau) -> axes if given."""
    from stick_test import drive
    if fn is None:
        fn = lambda f, tau: None
    return drive(run, secs, fn)


def click(run, spec, defs, value=None, hold=0.2, fn=None, verbose=True, release=None, mode=None):
    """Press (and release after `hold` s) a control; `release` sets the value sent afterwards (for axes).
    Returns the module's answers."""
    dev, cmd, down, up = resolve(spec, defs, value)
    if release is not None:
        up = float(release)
    link = run.link
    if mode is None and dev == defs["devices"].get("HOTAS"):
        mode = "s"      # measured: the HOTAS cursor does not move when clicked like a mouse, only through SetCommand
    link.click(dev, cmd, down, mode)
    settle(run, max(hold, 0.02), fn)
    if up is not None:
        link.click(dev, cmd, up, mode)
        settle(run, 0.1, fn)
    if verbose:
        r = link.events.get("click")
        res = "no answer from the module" if r is None else ("ok" if r.get("ok") else f"NOT done: {r.get('err')}")
        print(f"  click {spec if isinstance(spec, str) else ' '.join(map(str, spec))}: device {dev}, command {cmd}, "
              f"value {down:+g}{'' if up is None else f' then {up:+g}'}  ->  {res}")
    return link.events.get("click")


# ── what the radar reports ────────────────────────────────────────────

def _count(x):
    """How many targets a LoGet*TargetInformation table holds (it is a list of targets, or one target)."""
    if not x:
        return 0
    if isinstance(x, dict):
        if any(k in x for k in ("ID", "distance", "position")):
            return 1
        return sum(1 for v in x.values() if isinstance(v, dict))
    if isinstance(x, list):
        return sum(1 for v in x if isinstance(v, dict))
    return 0


def last_with(frames, key):
    """The most recent frame that carries `key` (the radar and world objects come every 5th frame, the
    payload every 50th)."""
    for f in reversed(frames):
        if key in f.raw:
            return f
    return None


def lock_state(f):
    """What the radar is doing, from the display markers the module reads (`rd`): 'stt' (a single-target track),
    'designated' (a target bugged, search display still up), 'search', 'no radar page', or None (no data)."""
    rd = f.raw.get("rd") if f is not None else None
    if rd is None:
        return None
    if not rd.get("fcr"):
        return "no radar page"
    if rd.get("tracked") and not rd.get("cursor"):
        return "stt"
    if rd.get("tracked"):
        return "designated"
    return "search"


def master_mode(f):
    """True if the HUD draws the A-A master mode, False if not (NAV), None if there is no display data."""
    rd = f.raw.get("rd") if f is not None else None
    if rd is None or "aa" not in rd:
        return None
    return bool(rd["aa"])


def radar_state(f):
    r = f.raw.get("radar") or {}
    return {"targets": _count(r.get("targets")), "locked": _count(r.get("locked")), "present": "radar" in f.raw}


def nearest(f):
    """The nearest other aircraft in the world objects (weapons in flight left out): (name, range m, relative bearing
    deg, altitude difference m)."""
    objs = f.raw.get("obj") or []
    if not objs or f.x is None:
        return None
    best = None
    for o in objs:
        if o.get("x") is None or o.get("z") is None:
            continue
        if o.get("l1") not in (None, 1):          # aircraft only: a missile in flight (level1 4) is not the target
            continue
        dx, dz = o["x"] - f.x, o["z"] - f.z
        dy = (o.get("y") or 0.0) - (f.alt or 0.0)
        rng = math.sqrt(dx * dx + dz * dz + dy * dy)
        if best is None or rng < best[1]:
            brg = math.atan2(dz, dx)                              # DCS: x north, z east
            rel = (brg - (f.hdg or 0.0) + math.pi) % (2 * math.pi) - math.pi
            best = (o.get("unit") or o.get("name") or "?", rng, rel * DEG, dy, o.get("l1"), o.get("l2"))
    return best


def target_altitude(run):
    """Altitude of the nearest other aircraft, from the latest world objects (None if there are none)."""
    fo = last_with(run.frames[-300:], "obj")
    nr = nearest(fo) if fo else None
    if nr is None:
        return None
    return (fo.alt or 0.0) + nr[3]


# set from the command line: put the target this many degrees right of the nose (negative: left) and fly this
# many metres above it (negative: below) before lock and fire, to try the radar off the nose and off the level
HOLD_OPTS = {"offset": None, "dalt": None}


def hold_fn(run, gains, alt=None, match_alt=False):
    """A function (frame, tau) -> axes that keeps the heading and speed as they are now and the altitude as
    they are, or at `alt`, or at the nearest aircraft's altitude (match_alt): a radar sees a target best at
    its own height. HOLD_OPTS turns to put the target `offset` degrees off the nose and adds `dalt` to the
    target's altitude."""
    from stick_test import Hold
    f0 = run.next()
    want_obj = match_alt or HOLD_OPTS.get("offset") is not None or HOLD_OPTS.get("dalt") is not None
    for _ in range(12):                     # the world objects come every 5th frame: wait for one
        if "obj" in f0.raw or not want_obj:
            break
        f0 = run.next()
    off, dalt = HOLD_OPTS.get("offset"), HOLD_OPTS.get("dalt")
    h = Hold(gains, bank_max=math.radians(60 if off is not None else 30), vs_max=80.0 if dalt is not None else 30.0)
    hdg, v = f0.hdg, f0.tas
    if off is not None:
        nr = nearest(f0) if "obj" in f0.raw else None
        if nr is None:
            off = None
            print("  --offset: no other aircraft in the world objects; keeping the heading")
        else:
            hdg = (f0.hdg + math.radians(nr[2] - off)) % (2 * math.pi)
            print(f"  turning to keep the target {off:+.0f} deg off the nose (it is {nr[2]:+.0f} now), following it "
                  f"as it moves")
    if (match_alt or dalt is not None) and alt is None:
        alt = target_altitude(run)
        if alt is not None and dalt is not None:
            alt += dalt
        if alt is None:
            print("  no other aircraft in the world objects yet; keeping the altitude")
        elif dalt is None:
            print(f"  matching the target's altitude: {alt:.0f} m (now {f0.alt:.0f})")
        else:
            print(f"  flying {dalt:+.0f} m from the target's altitude: {alt:.0f} m (now {f0.alt:.0f})")
    alt = alt if alt is not None else f0.alt
    if off is None:
        return lambda f, tau: h.axes(f, hdg, alt, v=v)
    state = {"hdg": hdg}

    def follow(f, tau):
        # the heading that puts the target `off` deg off the nose, from each new world-objects frame
        if "obj" in f.raw and f.hdg is not None:
            nr = nearest(f)
            if nr is not None:
                state["hdg"] = (f.hdg + math.radians(nr[2] - off)) % (2 * math.pi)
        return h.axes(f, state["hdg"], alt, v=v)
    return follow


def stage_radar(run, secs=20.0, out_dir=None, fn=None):
    """Print the nearest aircraft and the radar's target counts every second; keep samples of the raw
    tables for the first moments there were targets and a lock."""
    import datetime
    print("\n  one line a second. The radar state comes from what the radar page draws: search, designated "
          "(target bugged) or stt (single-target track). Designate the target yourself now (A-A mode, cursor over the "
          "contact, target management up; again for stt) and watch it change.\n")
    f = run.next()
    t0, last = f.t, -1.0
    st, fo, ls, ma = {"targets": 0, "locked": 0}, None, None, None
    samples = {"targets": None, "locked": None, "pay": None, "obj": None}
    prev_rng, prev_t = None, None
    rows = 0
    while f.t - t0 < secs:
        if fn is not None:
            ax = fn(f, f.t - t0)
            if ax:
                run.link.axes(**ax)
        if "radar" in f.raw:
            st = radar_state(f)
        if "obj" in f.raw:
            fo = f
        if "rd" in f.raw:
            ls = lock_state(f)
            ma = master_mode(f)
        if st["targets"] and samples["targets"] is None:
            samples["targets"] = {"t": f.t, "radar": f.raw.get("radar")}
        if st["locked"] and samples["locked"] is None:
            samples["locked"] = {"t": f.t, "radar": f.raw.get("radar")}
        if f.raw.get("pay") and samples["pay"] is None:
            samples["pay"] = f.raw["pay"]
        if f.raw.get("obj") and samples["obj"] is None:
            samples["obj"] = f.raw["obj"]
        if f.t - last >= 1.0:
            last = f.t
            nr = nearest(fo) if fo else None
            if nr:
                name, rng, rel, dy, l1, l2 = nr
                clo = "" if prev_rng is None else f"  closure {(prev_rng - rng) / max(f.t - prev_t, 1e-3):+.0f} m/s"
                prev_rng, prev_t = rng, f.t
                near = f"{name} {rng / 1000:6.1f} km, {rel:+6.1f} deg off the nose, {dy:+6.0f} m{clo}"
            else:
                near = "no other aircraft in the world objects"
            mode = {True: "A-A", False: "NAV", None: "?"}[ma]
            print(f"  t {f.t - t0:5.1f}  {mode:3s}  radar: {ls or 'no display data'}   {near}")
            rows += 1
        f = run.next()
    print()
    if samples["targets"] is None:
        print("  the radar's target list (LoGetTargetInformation) stayed empty: this module may not export its "
              "radar through that call. Run `ind` (the cockpit display text) with and without a lock instead.")
    else:
        print("  radar target table, first sample:\n" + json.dumps(samples["targets"]["radar"], indent=1)[:1600])
    if samples["locked"] is None:
        print("  nothing was locked (LoGetLockedTargetInformation empty)")
    else:
        print("  locked target table, first sample:\n" + json.dumps(samples["locked"]["radar"], indent=1)[:1600])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"stick_radar_sample_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w") as fh:
            json.dump(samples, fh, indent=1)
        print(f"\n  samples: {path}")
    return {"targets_seen": samples["targets"] is not None, "locked_seen": samples["locked"] is not None}


def settle_setup(run, fn, max_secs=75.0):
    """With --offset or --dalt: fly until the target sits where asked (within 3 deg, 300 m) or max_secs."""
    off, dalt = HOLD_OPTS.get("offset"), HOLD_OPTS.get("dalt")
    if fn is None or (off is None and dalt is None):
        return
    t0 = None
    while True:
        fr = settle(run, 2.0, fn)
        f = fr[-1] if fr else run.next()
        t0 = f.t if t0 is None else t0
        fo = last_with(fr, "obj")
        nr = nearest(fo) if fo else None
        if nr is None:
            return
        ok_az = off is None or abs(nr[2] - off) < 3.0
        ok_alt = dalt is None or abs(-nr[3] - dalt) < 300.0
        print(f"    setting up: target {nr[1] / 1000:.1f} km, {nr[2]:+.0f} deg off the nose, we are {-nr[3]:+.0f} m "
              f"from its altitude")
        if (ok_az and ok_alt) or f.t - t0 > max_secs:
            return


def _observe(run, secs, fn, label):
    fr = settle(run, secs, fn)
    fr_r, fr_o, fr_d = last_with(fr, "radar"), last_with(fr, "obj"), last_with(fr, "rd")
    st = radar_state(fr_r) if fr_r else {"targets": 0, "locked": 0, "present": False}
    nr = nearest(fr_o) if fr_o else None
    near = f"{nr[0]} {nr[1] / 1000:.1f} km, {nr[2]:+.0f} deg" if nr else "no target in the world objects"
    ls = lock_state(fr_d) if fr_d else None
    st["state"] = ls
    print(f"    {label}: radar {ls or 'no display data'}   ({near})")
    return st


def stage_lock(run, gains, defs, scan=False, alt=None, match_alt=False, aa=None, soi="left",
               scan_steps=40, scan_hold=0.3):
    """Try to lock the nearest target: A-A mode, then the target management switch up; with `scan`
    also slew the radar cursor in range between presses. Prints what the radar reports after each step."""
    fn = hold_fn(run, gains, alt, match_alt) if gains else None
    settle_setup(run, fn)
    fr0 = settle(run, 1.5, fn)
    f_o = last_with(fr0, "obj")
    nr = nearest(f_o) if f_o else None
    if nr is None:
        print("  no other aircraft in the world objects: start the mission built with --stick-test --target")
    else:
        print(f"  target {nr[0]}: {nr[1] / 1000:.1f} km, {nr[2]:+.0f} deg off the nose, {nr[3]:+.0f} m")
        if abs(nr[2]) > 55 or nr[1] > 85000:
            print("  WARNING: outside the radar's gimbal (about 60 deg) or range (about 80 km): turn towards it first")
    print("  before:")
    st = _observe(run, 1.0, fn, "radar")
    if st["state"] is None:
        print("  no display data from the module (rd): reinstall it (python dcs\\stick\\install_export.py), "
              "restart the mission. Going on with the radar tables only, which are empty for this module.")
    log = []

    def done(st):
        return st.get("state") == "stt" or (st.get("state") is None and st["locked"])
    # The ICP's A-A button toggles (pressed in A-A mode it goes back to the previous mode), so it is pressed
    # only when the HUD says the cockpit is not in A-A (aa=None), or always (aa=True), or never (aa=False).
    steps = [("target management up: designate the target under the cursor", "hotas.STICK_TGT_MANAGE_UP"),
             ("target management up again: single-target track", "hotas.STICK_TGT_MANAGE_UP")]
    fr_m = last_with(run.frames[-300:], "rd")
    in_aa = master_mode(fr_m)
    if aa is None:
        aa = in_aa is False
    if soi in ("left", "right"):
        # target management and the radar cursor act on the sensor of interest; A-A mode brings the weapons page up on
        # the right MFD and the SOI can sit there: display management left makes the left MFD (the radar page) the SOI
        steps.insert(0, (f"display management {soi}: the {soi} MFD (radar page) becomes the sensor of interest",
                         f"hotas.STICK_DISP_MANAGE_{soi.upper()}"))
    if aa:
        steps.insert(0, ("A-A master mode (ICP)", "ufc.AA"))
        print("  the HUD is not in A-A: pressing the ICP A-A button first" if in_aa is False else
              "  pressing the ICP A-A button (asked for)")
    else:
        print("  A-A master mode: " + ("the HUD shows it" if in_aa else "not known (no display data), assumed selected"))
    for label, spec in steps:
        if done(st):
            break
        if spec == "hotas.STICK_TGT_MANAGE_UP" and st.get("state") in ("search", "no radar page") and log \
                and log[-1][0].startswith("target management") and st.get("state") != "designated":
            break                                # the first press designated nothing: more presses would not help
        print(f"  step: {label}")
        click(run, spec, defs, fn=fn)
        st = _observe(run, 2.0, fn, "after")
        log.append((label, st.get("state"), st["locked"]))
    if st.get("state") not in ("designated", "stt") and scan:
        # The radar page lists FCR_TUC_contactData while a contact is under the cursor (measured: it appears for one
        # cursor step as the cursor crosses the contact), so the cursor is slewed in range until that shows and
        # target management is pressed only then. The first press that designates names the delivery mode that works.
        fid = _fcr_id(run)
        link = run.link
        print(f"  scan: cursor in range (up to {scan_steps} steps out, then {2 * scan_steps} back, {scan_hold:g} s each) until "
              f"the radar page shows a target under the cursor, then target management up. Watch the cursor on the left MFD.")

        def tuc():
            return any("FCR_TUC_contactData" in x for x in ind_text(run, link, fid))

        def press():
            nonlocal st
            for mode in ("s", "c", "l"):
                click(run, "hotas.STICK_TGT_MANAGE_UP", defs, fn=fn, verbose=False, mode=mode)
                st = _observe(run, 1.0, fn, f"target management up (mode {mode})")
                if st.get("state") in ("designated", "stt"):
                    print(f"  designated: delivery mode {mode} works for target management")
                    return True
            return False
        moves = ["hotas.THROTTLE_RDR_CURSOR_FWD"] * scan_steps + ["hotas.THROTTLE_RDR_CURSOR_AFT"] * (2 * scan_steps)
        found = tuc()
        for i in range(len(moves) + 1):
            if found:
                print(f"  a target is under the cursor (after {i} cursor steps)")
                if not press():
                    print("  the contact is under the cursor but no delivery mode of target management up designated it: "
                          "is the radar the sensor of interest (left MFD, SOI box)? is it a mode that needs TMS held?")
                break
            if i == len(moves):
                print("  the cursor never showed a target under it: wrong range scale, or the radar does not see the target")
                break
            click(run, moves[i], defs, hold=scan_hold, fn=fn, verbose=False)
            settle(run, 0.15, fn)
            found = tuc()
            if i % 5 == 4:
                print(f"    scan {i + 1}: {'out' if i < scan_steps else 'back'}, still searching")
        if st.get("state") == "designated":
            print("  step: target management up again: single-target track")
            click(run, "hotas.STICK_TGT_MANAGE_UP", defs, fn=fn)
            st = _observe(run, 2.0, fn, "after")
    state = st.get("state")
    if state == "stt":
        print("  LOCKED: the radar page shows a single-target track")
    elif state == "designated":
        print("  DESIGNATED: the target is bugged (the search display is still up); press target management up once more "
              "for a single-target track, or fire from here")
    else:
        print("  not locked. Look at the left MFD: which radar mode and page is it in, where is the cursor? "
              "Designate the target by hand, run `radar`, and tell me what it prints.")
    return {"locked": state == "stt", "designated": state in ("designated", "stt"), "state": state, "steps": log}


# ── the aimed lock ─────────────────────────────────────────────────────

NM = 1852.0


def _children(lines, name):
    """How many direct children a display element lists (0 if none, None if the element is not on the page)."""
    for i, x in enumerate(lines):
        if x.strip() == name:
            if i + 1 >= len(lines) or lines[i + 1].strip() != "children are {":
                return 0
            depth, n = 0, 0
            for y in lines[i + 2:]:
                y = y.strip()
                if y == "}":
                    if depth == 0:
                        return n
                    depth -= 1
                elif y == "children are {":
                    depth += 1
                elif depth == 0:
                    n += 1
            return n
    return None


def fcr_info(lines):
    """What the radar page's text says: the cursor's bearing (magnetic deg) and range (NM) from the bullseye, the
    height coverage at the cursor's range (thousands of feet), the range scale, a target under the cursor (and its
    altitude), and whether the page is the sensor of interest."""
    import re
    out = {"br": None, "hi": None, "lo": None, "scale": None, "tuc": False, "tuc_alt": None, "soi": False,
           "mode": None, "az": None, "bars": None,
           # what the radar shows: AA_RadarTracksKeeper lists the radar's contacts, System_AirTracksKeeper the TWS
           # system tracks (a bug needs one); both empty on 22 October with the target inside the scan
           "contacts": _children(lines, "AA_RadarTracksKeeper"), "tracks": _children(lines, "System_AirTracksKeeper")}

    def nxt(i):
        return lines[i + 1].strip() if i + 1 < len(lines) else ""
    for i, x in enumerate(lines):
        x = x.strip()
        if x == "FCR_Bearing_range" and re.fullmatch(r"-?\d+ -?\d+", nxt(i)):
            a, b = nxt(i).split()
            out["br"] = (int(a), int(b))
        elif x == "Max_Search_Altitude" and re.fullmatch(r"-?\d+", nxt(i)):
            out["hi"] = int(nxt(i))
        elif x == "Min_Search_Altitude" and re.fullmatch(r"-?\d+", nxt(i)):
            out["lo"] = int(nxt(i))
        elif x == "Range_Scale_Value" and re.fullmatch(r"\d+", nxt(i)):
            out["scale"] = int(nxt(i))
        elif x == "FCR_TUC_contactData":
            out["tuc"] = True
        elif x == "FCR_TUC_contact_Altitude" and re.fullmatch(r"-?\d+", nxt(i)):
            out["tuc_alt"] = int(nxt(i))
        elif x == "SOI":
            out["soi"] = True
        elif x in ("RWS", "TWS", "VSR", "ACM", "STT") and out["mode"] is None:
            out["mode"] = x                       # the mode label's text (the other modes' tables are empty)
        elif x == "A" and re.fullmatch(r"\d", nxt(i)):
            out["az"] = int(nxt(i))               # A1 / A2 / A3 / A6: the scan is 10 / 25 / 30 / 60 deg either side of the cursor
        elif re.fullmatch(r"\d", x) and nxt(i) == "B":
            out["bars"] = int(x)
    return out


def _target_obj(f):
    """The nearest other aircraft's world-object entry."""
    nr = nearest(f) if f else None
    if nr is None:
        return None
    return next((o for o in (f.raw.get("obj") or []) if (o.get("unit") or o.get("name") or "?") == nr[0]), None)


def _polar(f, x, z):
    """A point from the aircraft: range NM, azimuth deg off the nose (+ right)."""
    dx, dz = x - f.x, z - f.z
    return math.hypot(dx, dz) / NM, math.degrees((math.atan2(dz, dx) - f.hdg + math.pi) % (2 * math.pi) - math.pi)


def stage_aimlock(run, gains, defs, alt=None, match_alt=False, bull=None, magvar=None, max_iter=200, aa=None,
                  relock=False, ref="auto"):
    """Lock the nearest aircraft wherever it is within the radar's limits: put the radar cursor on the point where
    the radar should show it (its bearing and range from the world objects), steering by the cursor's bullseye
    bearing and range on the radar page; tilt the antenna until the target's altitude is inside the height
    coverage at its range; search a little around the point until the page shows a target under the cursor, then
    target management up (designate) and up again (single-target track)."""
    fn = hold_fn(run, gains, alt, match_alt) if gains else None
    settle_setup(run, fn)
    link = run.link
    fr0 = settle(run, 1.0, fn)
    fid = _fcr_id(run)
    st = _observe(run, 0.5, fn, "radar")
    if relock and st.get("state") in ("designated", "stt"):
        print("  --relock: target management down drops the lock first")
        click(run, "hotas.STICK_TGT_MANAGE_DOWN", defs, fn=fn, verbose=False)
        st = _observe(run, 1.5, fn, "after")
    if st.get("state") == "stt":
        print("  already locked (--relock drops it first)")
        return {"locked": True, "designated": True, "state": "stt"}
    in_aa = master_mode(last_with(run.frames[-300:], "rd"))
    if aa or (aa is None and in_aa is False):
        print("  step: A-A master mode (ICP)")
        click(run, "ufc.AA", defs, fn=fn)
        settle(run, 1.5, fn)

    def page():
        return fcr_info(ind_text(run, link, fid))
    info = page()
    if info["br"] is None:
        print(f"  the radar page (display {fid}) shows no cursor bearing and range: is the radar page up on the left "
              f"MFD in A-A mode?")
        return {"locked": False, "designated": False, "state": st.get("state"), "why": "no readout"}
    if not info["soi"]:
        print("  WARNING: the radar page is not the sensor of interest (no SOI on it): the cursor will not move")
    f = run.frames[-1]
    if magvar is None and f.myaw is not None and f.hdg is not None:
        magvar = math.degrees((f.hdg - f.myaw + math.pi) % (2 * math.pi) - math.pi)
    if magvar is None:
        magvar = 0.0
        print("  WARNING: no magnetic heading from the module (reinstall it) and no --magvar: taking the variation as 0")
    print(f"  magnetic variation {magvar:+.1f} deg; range scale {info['scale']} NM; coverage at the cursor "
          f"{info['lo']}-{info['hi']} kft")

    def true_brg(b):
        return math.radians(b + magvar)
    def own_like(info, f):
        """Does the readout look like bearing and range from our own aircraft (inside the scale and the radar's
        60 deg), rather than from a bullseye far away?"""
        if info["br"] is None or f.hdg is None:
            return False
        b, r = info["br"]
        rel = math.degrees((true_brg(b) - f.hdg + math.pi) % (2 * math.pi) - math.pi)
        return r <= (info["scale"] or 160) + 1 and abs(rel) <= 62.0
    if bull is not None:
        ref = "bull"
    if ref == "auto":
        if own_like(info, run.frames[-1]):
            ref = "own"
        else:
            # The F-16 shows the cursor's bearing and range from the bullseye while BULLSEYE is selected in the DED,
            # from our own aircraft when it is not: whole degrees and NM from here are fine everywhere, and no edge of
            # the display (where the radar changes its range scale) has to be visited to find a bullseye.
            print("  the readout is from a bullseye: switching it to our own aircraft (ICP: LIST, 0 MISC, 8 BULL, "
                  "0 M-SEL, RTN)")
            for spec in ("ufc.LIST", "ufc.DIG0_M_SEL", "ufc.DIG8_FIX", "ufc.DIG0_M_SEL", "ufc.DCS_RTN"):
                click(run, spec, defs, fn=fn, verbose=False)
                settle(run, 0.35, fn)
            settle(run, 0.5, fn)
            ded = [x for x in ind_text(run, link, 6) if not x.startswith(("children", "}")) and len(x) < 40]
            print("    the DED (display 6) now shows: " + " | ".join(ded[:24]))
            info = page()
            if own_like(info, run.frames[-1]):
                ref = "own"
                print(f"    now {info['br'][0]:03d}/{info['br'][1]}: from our own aircraft")
            else:
                ref = "bull"
                print(f"    still {info['br'][0]:03d}/{info['br'][1]}: not switched; finding the bullseye instead "
                      f"(slower). Turn BULLSEYE off by hand in the DED (LIST, 0, 8, M-SEL) and run again")
    if ref == "bull" and bull is None:
        # the B-scope's bottom edge is range 0: with the cursor there, its bullseye bearing and range are the aircraft's
        print("  finding the bullseye: cursor to the bottom edge (it is then on our own aircraft)")
        prev = None
        for _ in range(10):
            click(run, "hotas.THROTTLE_RDR_CURSOR_AFT", defs, hold=1.0, fn=fn, verbose=False)
            settle(run, 0.4, fn)
            info = page()
            if info["br"] is not None and info["br"] == prev:
                break
            prev = info["br"]
        f = run.frames[-1]
        b, r = info["br"]
        bull = (f.x - r * NM * math.cos(true_brg(b)), f.z - r * NM * math.sin(true_brg(b)))
        rb, azb = _polar(f, *bull)
        print(f"    the cursor reads {b:03d}/{r} at our position: the bullseye is {rb:.0f} NM away, {azb:+.0f} deg off "
              f"the nose" + ("   (far: whole degrees and NM make the aim coarse; the missions from make_mission.py put "
                              "it near the fight)" if rb > 60 else ""))

    def cursor_at(info, fr):
        """Where the readout puts the cursor, in DCS metres."""
        b, r = info["br"]
        ox, oz = (fr.x, fr.z) if ref == "own" else bull
        return ox + r * NM * math.cos(true_brg(b)), oz + r * NM * math.sin(true_brg(b))
    f = run.frames[-1]
    tgt0 = _target_obj(last_with(run.frames[-100:], "obj"))
    r0 = _polar(f, tgt0["x"], tgt0["z"])[0] if tgt0 else 30.0
    # keep the target inside the range scale: at the top edge the radar steps the scale and the cursor jumps
    for _ in range(3):
        if not info["scale"] or r0 < 0.85 * info["scale"]:
            break
        s0 = info["scale"]
        click(run, "left_osb.19", defs, fn=fn, verbose=False)
        settle(run, 0.6, fn)
        info = page()
        print(f"    range scale {s0} -> {info['scale']} NM (target at {r0:.0f} NM)")
        if info["scale"] is None or info["scale"] <= s0:
            break
    rate = {"r": 6.0, "az": 40.0}                       # NM/s and deg/s while held, learnt as we go
    # The readout is whole degrees and NM from the bullseye: half a degree there is this far sideways at the cursor,
    # plus half a mile, twice (once placing the bullseye, once reading the cursor). Far from the bullseye that is
    # more than the target symbol is wide, so the tolerances and the search around the point grow with it.
    if ref == "own":
        rb_nm = 0.0
        unc_nm = r0 * math.tan(math.radians(0.5)) + 0.5
    else:
        rb_nm = math.hypot(f.x - bull[0], f.z - bull[1]) / NM
        unc_nm = 2 * (rb_nm * math.tan(math.radians(0.5)) + 0.5)
    unc_az = math.degrees(unc_nm / max(r0, 5.0))
    tol0 = {"range": 0.8, "azimuth": max(1.5, 0.6 * unc_az)}
    # First the cursor goes straight to the target's place. If nothing is under it there, rows are swept: from W deg
    # left of the place to W right, in steps of about 2 deg timed by the azimuth slew rate seen so far (the readout
    # cannot see steps that small when the bullseye is far), a row at the target's range and then 1, 2, ... NM nearer
    # and further, out to the readout's uncertainty.
    W = max(3.0, unc_az)
    n_rows = max(2, int(math.ceil(unc_nm)))
    offsets = [(0.0, 0.0)] + [(ro, -W) for i in range(n_rows + 1) for ro in ((0.0,) if i == 0 else (float(i), -float(i)))]
    print(f"  readout from {'our own aircraft' if ref == 'own' else 'the bullseye'}: it places the cursor to about {unc_nm:.1f} NM ({unc_az:.1f} deg at the target's range)"
          + ("" if rb_nm < 60 else f": coarse, because the bullseye is {rb_nm:.0f} NM away. Use the mission from "
                                   f"make_mission.py, which puts it near the fight"))
    k, elev_tries, found = 0, 0, False
    tol, last_dir = dict(tol0), {}                       # a tolerance widens each time its axis overshoots
    for it in range(max_iter):
        f = run.frames[-1]
        tgt = _target_obj(last_with(run.frames[-100:], "obj"))
        if tgt is None or info["br"] is None:
            print("  lost the target in the world objects, or the readout")
            break
        r_p, az_p = _polar(f, tgt["x"], tgt["z"])
        talt = (tgt.get("y") or 0.0) / 304.8
        if abs(az_p) > 58.0:
            print(f"  the target is {az_p:+.0f} deg off the nose: outside the radar's azimuth limits (60). Turn first.")
            break
        b, r = info["br"]
        cx, cz = cursor_at(info, f)
        r_c, az_c = _polar(f, cx, cz)
        if info["tuc"]:
            print(f"  a target is under the cursor (altitude {info['tuc_alt']} kft; ours {talt:.0f} kft): target "
                  f"management up")
            click(run, "hotas.STICK_TGT_MANAGE_UP", defs, fn=fn, verbose=False)
            st = _observe(run, 1.0, fn, "after")
            if st.get("state") == "designated":
                print("  step: target management up again: single-target track")
                click(run, "hotas.STICK_TGT_MANAGE_UP", defs, fn=fn, verbose=False)
                st = _observe(run, 1.5, fn, "after")
            if st.get("state") in ("designated", "stt"):
                break
        a_r, a_az = r_p + offsets[k][0], az_p + offsets[k][1]
        dr, daz = a_r - r_c, (a_az - az_c) if r_c > 2.0 else 0.0
        near = abs(r_p - r_c) < 3.0
        out_cov = None
        if near and info["hi"] is not None and info["lo"] is not None:
            if talt > info["hi"] - 1:
                out_cov = "up"
            elif talt < info["lo"] + 1:
                out_cov = "down"
        if out_cov and elev_tries < 25:
            spec, t, what = f"hotas.THROTTLE_ANT_ELEV_{out_cov.upper()}", 0.3, f"antenna {out_cov}"
            elev_tries += 1
        elif abs(dr) < tol["range"] and abs(daz) < tol["azimuth"]:
            if k > 0:                                    # at the start of a row: sweep it
                step = min(max(2.0 / rate["az"], 0.12), 0.3)       # shorter pulses did not move the cursor
                n = int(math.ceil(2 * W / max(2.0, step * rate["az"])))
                print(f"    sweeping {2 * W:.0f} deg to the right at {offsets[k][0]:+.0f} NM from the target's range "
                      f"({n} steps of {step:.2f} s)")
                for _ in range(n):
                    click(run, "hotas.THROTTLE_RDR_CURSOR_RIGHT", defs, hold=step, fn=fn, verbose=False)
                    settle(run, 0.25, fn)
                    info = page()
                    if info["tuc"]:
                        found = True
                        break
                if found:
                    found = False
                    continue
            if k + 1 >= len(offsets):
                print("  searched around the target's place without a target under the cursor: the radar does not "
                      "see it (range, or its altitude outside the coverage)")
                break
            k += 1
            tol, last_dir = dict(tol0), {}
            print(f"    nothing under the cursor: to the start of a row, {offsets[k][0]:+.1f} NM, {offsets[k][1]:+.1f} deg "
                  f"from the target")
            info = page()
            continue
        else:
            if abs(dr) / tol["range"] >= abs(daz) / tol["azimuth"]:
                spec = "hotas.THROTTLE_RDR_CURSOR_FWD" if dr > 0 else "hotas.THROTTLE_RDR_CURSOR_AFT"
                t, what, sgn = min(max(abs(dr) / rate["r"], 0.03), 0.6), "range", dr > 0
            else:
                spec = "hotas.THROTTLE_RDR_CURSOR_RIGHT" if daz > 0 else "hotas.THROTTLE_RDR_CURSOR_LEFT"
                t, what, sgn = min(max(abs(daz) / rate["az"], 0.03), 0.6), "azimuth", daz > 0
            if last_dir.get(what) is not None and last_dir[what] != sgn:
                tol[what] = min(tol[what] * 1.4, 2.0 if what == "range" else max(4.0, unc_az))
            last_dir[what] = sgn
        print(f"    aim {it + 1}: cursor {r_c:5.1f} NM {az_c:+5.1f} deg | target {r_p:5.1f} NM {az_p:+5.1f} deg "
              f"{talt:4.1f} kft | coverage {info['lo']}..{info['hi']} kft | {spec.split('_', 3)[-1].lower()} {t:.2f} s")
        click(run, spec, defs, hold=t, fn=fn, verbose=False)
        settle(run, 0.3, fn)
        info2 = page()
        if info2["br"] is not None and what in ("range", "azimuth"):
            b2, r2 = info2["br"]
            f2 = run.frames[-1]
            r_c2, az_c2 = _polar(f2, *cursor_at(info2, f2))
            moved = abs(r_c2 - r_c) if what == "range" else abs(az_c2 - az_c)
            key, lo_hi = ("r", (0.5, 60.0)) if what == "range" else ("az", (5.0, 400.0))
            if moved > (1.0 if what == "range" else max(1.5, unc_az)):      # above the readout's rounding
                rate[key] = min(max(math.sqrt(rate[key] * moved / t), lo_hi[0]), lo_hi[1])
        info = info2 if info2["br"] is not None else info
    st = _observe(run, 0.5, fn, "radar")
    state = st.get("state")
    print("  LOCKED: single-target track" if state == "stt" else "  DESIGNATED" if state == "designated" else
          "  not locked")
    return {"locked": state == "stt", "designated": state in ("designated", "stt"), "state": state}


CURSOR_AZ_RATE = 100.0     # deg/s the radar cursor moves sideways while held (DCS, an aim run: 39 deg in 0.37 s)


def center_cursor(run, defs, fn, az, verbose=True):
    """Put the radar cursor near azimuth `az` (deg off the nose) without reading it back: held toward that side it
    stops at the display's edge (60 deg), from where a timed move back leaves it at az. TWS scans 25 deg either side
    of the cursor, so 10 deg off is still a hit."""
    az = max(-58.0, min(58.0, az))
    side = "RIGHT" if az >= 0 else "LEFT"
    back = "LEFT" if az >= 0 else "RIGHT"
    click(run, f"hotas.THROTTLE_RDR_CURSOR_{side}", defs, hold=1.5, fn=fn, verbose=False)
    t = (60.0 - abs(az)) / CURSOR_AZ_RATE
    if t > 0.03:
        click(run, f"hotas.THROTTLE_RDR_CURSOR_{back}", defs, hold=t, fn=fn, verbose=False)
    if verbose:
        print(f"  cursor to the {side.lower()} edge, then {t:.2f} s back: about {az:+.0f} deg, where the target is")


def stage_twslock(run, gains, defs, alt=None, match_alt=False, aa=None, relock=False, stt=False, max_secs=35.0):
    """Lock without the cursor: target management right held 1.5 s puts the radar in TWS (track while scan), where it
    builds track files by itself; a short target management right bugs a track once there is one (measured in DCS:
    the fifth press, about 14 s after entering TWS with A6 and 4 bars; the page then draws the AIM-120 launch zone,
    DLZ_AIM120_root). It is pressed about once a second until the bug shows. An AIM-120 can be fired at the bugged
    track; stt=True also tries target management up for a single-target track (it stayed bugged in DCS).
    In a 1v1 the only track is the target."""
    fn = hold_fn(run, gains, alt, match_alt) if gains else None
    settle_setup(run, fn)
    link = run.link
    settle(run, 1.0, fn)
    fid = _fcr_id(run)
    st = _observe(run, 0.5, fn, "radar")
    if relock and st.get("state") in ("designated", "stt"):
        print("  --relock: target management down drops the lock first")
        click(run, "hotas.STICK_TGT_MANAGE_DOWN", defs, fn=fn, verbose=False)
        st = _observe(run, 1.5, fn, "after")
    in_aa = master_mode(last_with(run.frames[-300:], "rd"))
    if aa or (aa is None and in_aa is False):
        print("  step: A-A master mode (ICP)")
        click(run, "ufc.AA", defs, fn=fn)
        settle(run, 1.5, fn)

    def page():
        return fcr_info(ind_text(run, link, fid))

    def names():
        """The element names the radar page draws (no numbers or braces)."""
        import re
        return {x.strip() for x in ind_text(run, link, fid)
                if re.search(r"[A-Za-z_]{3}", x) and not x.startswith("children") and len(x) < 70}
    base = names()
    info = page()
    print(f"  radar page: {info['mode']}, A{info['az']} (scan {({1: 10, 2: 25, 3: 30, 6: 60}).get(info['az'], '?')} deg either "
          f"side of the cursor), {info['bars']} bars, range scale {info['scale']} NM")
    t0 = run.frames[-1].t
    if info["mode"] != "TWS" and st.get("state") not in ("designated", "stt"):
        for hold, how in ((1.5, "held 1.5 s"), (2.5, "held 2.5 s")):
            print(f"  step: target management right ({how}): RWS -> TWS")
            click(run, "hotas.STICK_TGT_MANAGE_RIGHT", defs, hold=hold, fn=fn, verbose=False)
            settle(run, 1.5, fn)
            info = page()
            print(f"    mode now {info['mode']}")
            if info["mode"] == "TWS":
                break
        if info["mode"] != "TWS":
            print("  the radar did not go to TWS with a short or a held press: tell me what the radar page shows after "
                  "target management right (Right Ctrl + Right) by hand, short and held")
            return {"locked": False, "designated": False, "state": st.get("state"), "why": "no TWS"}
    nr = nearest(last_with(run.frames[-100:], "obj"))
    if nr is not None and st.get("state") not in ("designated", "stt"):
        center_cursor(run, defs, fn, nr[2])          # the scan is 25 deg either side of the cursor, not the nose
    # The track file takes a scan or two, and the bug shows on the page a moment after the press: in DCS a press
    # every 1.2 s, looked at 0.8 s later, bugged nothing in 25 s, while presses 1.5 s apart after 8 s of scanning
    # bugged it on the fifth. So: 4 s of scanning, then a press every 2 s, watching the page for 2 s after each
    # (the AIM-120 launch zone is drawn, or the display markers say designated).
    def bugged(nm):
        return any(x.startswith("DLZ_AIM120") or "bug" in x.lower() for x in nm)
    t_tws = run.frames[-1].t
    settle(run, 4.0, fn)
    i = 0
    while st.get("state") not in ("designated", "stt") and run.frames[-1].t - t_tws < max_secs:
        i += 1
        click(run, "hotas.STICK_TGT_MANAGE_RIGHT", defs, fn=fn, verbose=False)
        for _ in range(4):
            settle(run, 0.5, fn)
            fd = last_with(run.frames[-60:], "rd")
            if fd is not None and lock_state(fd) in ("designated", "stt"):
                st["state"] = lock_state(fd)
                break
            if bugged(names() - base):
                st["state"] = "designated"
                break
        info = page()
        print(f"    target management right {i} ({run.frames[-1].t - t_tws:.0f} s in TWS): mode {info['mode']}, "
              f"radar {st.get('state')}")
    if st.get("state") == "designated" and stt:
        print("  step: target management up: single-target track")
        click(run, "hotas.STICK_TGT_MANAGE_UP", defs, fn=fn, verbose=False)
        st = _observe(run, 1.5, fn, "after")
    state = st.get("state")
    dt = run.frames[-1].t - t0
    print(f"  LOCKED: single-target track ({dt:.0f} s)" if state == "stt" else
          f"  DESIGNATED: the TWS track is bugged ({dt:.0f} s); the AIM-120 can be fired at it" if state == "designated"
          else "  not locked: no track was bugged. Is the target inside the scan (the A setting either side of the "
               "cursor) and the height coverage?")
    return {"locked": state == "stt", "designated": state in ("designated", "stt"), "state": state}


def payload_count(f):
    """Number of stores the payload table lists with a positive count, or None if there is none."""
    pay = f.raw.get("pay")
    if not isinstance(pay, dict):
        return None
    sts = pay.get("Stations")
    if not isinstance(sts, dict):
        return None
    return sum(int(v.get("count", 0)) for v in sts.values() if isinstance(v, dict))


def stations(f):
    """{station: (CLSID or weapon name, count)} from a payload table, or {}."""
    pay = f.raw.get("pay") if f else None
    sts = pay.get("Stations") if isinstance(pay, dict) else None
    out = {}
    if isinstance(sts, dict):
        for k, v in sts.items():
            if isinstance(v, dict):
                w = v.get("weapon") if isinstance(v.get("weapon"), dict) else {}
                out[str(k)] = (v.get("CLSID") or w.get("name") or "?", int(v.get("count", 0) or 0))
    return out


def hud_mrm(run, link, ident=1):
    """The AIM-120 count the HUD shows ('4 MRM'), or None."""
    import re
    for x in ind_text(run, link, ident):
        m = re.search(r"(\d+)\s*MRM", x)
        if m:
            return int(m.group(1))
    return None


def _ids(f, key):
    return {o.get("id") for o in ((f.raw.get(key) or []) if f else []) if isinstance(o, dict)}


def stage_fire(run, gains, defs, force=False, alt=None, match_alt=False, aa=None, watch=15, release_hold=1.5):
    """Arm, release, watch: master arm on, the weapon release button, then `watch` seconds of looking for the
    missile: the HUD's AIM-120 count ('4 MRM'), the weapons in flight (the module's `ball`), new world objects and
    the payload; master arm back to OFF at the end."""
    fn = hold_fn(run, gains, alt, match_alt) if gains else None
    link = run.link
    settle_setup(run, fn)
    fr0 = settle(run, 1.5, fn)
    f_r, f_o, f_p, f_d = last_with(fr0, "radar"), last_with(fr0, "obj"), last_with(fr0, "pay"), last_with(fr0, "rd")
    st = radar_state(f_r) if f_r else {"targets": 0, "locked": 0}
    state = lock_state(f_d) if f_d else None
    have = state in ("designated", "stt") or (state is None and st["locked"])
    print(f"  radar: {state or 'no display data'}")
    if not have and not force:
        print("  no target is designated or locked: run `lock` first (or designate by hand), or add --force to release anyway")
        return {"fired": False, "why": "no lock"}
    st["locked"] = 1 if have else 0
    pay0 = payload_count(f_p) if f_p else None
    objs0 = _ids(f_o, "obj")
    ball0 = _ids(last_with(fr0, "ball"), "ball")
    mrm0 = hud_mrm(run, link)
    sts0 = stations(f_p)
    print(f"  before: HUD shows {mrm0} AIM-120 (MRM); payload table {pay0} {sts0 or ''}; world objects {len(objs0)}; "
          f"weapons in flight {len(ball0)}")
    if not st["locked"]:
        print("  no lock: the AIM-120 will be released without a track (a 'maddog' shot: it flies along the nose "
              "and looks for a target with its own seeker). The target is dead ahead, so this can still work.")
    in_aa = master_mode(f_d)
    if aa is None:
        aa = in_aa is False
    if aa:
        print("  step: A-A master mode (the HUD is not in A-A)" if in_aa is False else "  step: A-A master mode (asked for)")
        click(run, "ufc.AA", defs, fn=fn)
    else:
        print("  A-A master mode: " + ("the HUD shows it" if in_aa else "not known (no display data), assumed selected"))
    if f_d is not None and f_d.raw.get("rd", {}).get("msl120") is False and in_aa:
        print("  WARNING: the HUD does not show the AIM-120 selected: step the missile selection "
              "(S in DCS, or click hotas.STICK_NWS_AR_DISC_MSL_STEP) and run fire again")
    def hud_arm():
        return any(x.strip() == "ARM" for x in ind_text(run, link, 1))

    print("  step: MASTER ARM")
    for mode in (None, "s", "l"):
        click(run, "mmc.MasterArmSw", defs, value=1.0, fn=fn, mode=mode, verbose=mode is None)
        settle(run, 1.0, fn)
        armed = hud_arm()
        print(f"    HUD shows ARM: {armed}" + ("" if armed else f"   (route {mode or 'c'} did not arm it)"))
        if armed:
            break
    if not armed:
        print("  WARNING: the HUD does not show ARM: continuing anyway, but the master arm switch may not have moved")
    new, pay1, mrm1, seen, worked = [], pay0, mrm0, False, None

    tgt0 = nearest(f_o) if f_o else None
    tgt_id = next((o.get("id") for o in (f_o.raw.get("obj") or []) if o.get("unit") == tgt0[0]), None) if tgt0 else None

    def miss_dist(fo2, msls):
        """Distance in km from each missile to the target, from the world objects' positions."""
        t = next((o for o in ((fo2.raw.get("obj") or []) if fo2 else []) if o.get("id") == tgt_id), None)
        if t is None or t.get("x") is None:
            return []
        return [round(math.dist((m["x"], m["y"], m["z"]), (t["x"], t["y"], t["z"])) / 1000, 1)
                for m in msls if m.get("x") is not None]

    def look(secs, label, watching=False):
        """Watch `secs` seconds. Looking for the launch (watching=False): returns True at the first sign of the
        missile. Watching it fly: goes on until the time is up or the target leaves the world objects."""
        nonlocal new, pay1, mrm1, seen
        for i in range(int(secs)):
            fr = settle(run, 1.0, fn)
            fp, fo2, fb = last_with(fr, "pay"), last_with(fr, "obj"), last_with(fr, "ball")
            pay1 = payload_count(fp) if fp else pay1
            new = [o for o in ((fo2.raw.get("obj") or []) if fo2 else []) if o.get("id") not in objs0]
            fly = [o for o in ((fb.raw.get("ball") or []) if fb else []) if o.get("id") not in ball0]
            if not watching and (i < 4 or i % 5 == 4):
                mrm1 = hud_mrm(run, link)
            gone = len(objs0 - _ids(fo2, "obj")) if fo2 else 0
            if watching:
                msls = [o for o in new + fly if o.get("l1") == 4]
                print(f"    {label} +{i + 1} s: {len(msls)} missile(s) in flight, to the target {miss_dist(fo2, msls)} km"
                      + (f"; {gone} world object(s) gone: the target is destroyed" if gone else ""))
                if gone:
                    return True
                continue
            print(f"    {label} +{i + 1} s: HUD MRM {mrm1}, payload {pay1}, weapons in flight "
                  f"{[(o.get('name'), round(o.get('alt') or 0)) for o in fly]}, new world objects "
                  f"{[(o.get('name'), o.get('l1'), o.get('l2')) for o in new]}"
                  + (f", {gone} world object(s) gone (target destroyed?)" if gone else ""))
            seen = seen or bool(fly or new)
            hud_down = mrm0 is not None and mrm1 is not None and mrm1 < mrm0
            if seen or hud_down or (pay0 is not None and pay1 is not None and pay1 < pay0):
                return True
        return False

    killed = False
    for mode in ("s", "c", "l", "g"):
        print(f"  step: weapon release button (route {mode}: " + {"s": "device SetCommand", "c": "clickable element",
              "l": "global command queue", "g": "DCS's generic pickle command, 350 and 351"}[mode] + ")")
        if mode == "g":                      # iCommandPlanePickleOn / Off of command_defs.lua
            link.raw(350, 1.0)
            settle(run, release_hold, fn)
            link.raw(351, 1.0)
        else:
            # the button has to stay pressed until the missile leaves the rail (a 0.4 s press launched nothing by any route)
            click(run, "hotas.STICK_WEAPON_RELEASE", defs, hold=release_hold, fn=fn, mode=mode)
        if look(4, f"route {mode}"):
            worked = mode
            print(f"  the release went through route {mode}")
            break
    if worked:
        # master arm goes OFF now: the missile in flight does not need it, and a second press cannot fire another
        print("  step: MASTER ARM back to OFF")
        click(run, "mmc.MasterArmSw", defs, value=0.0, fn=fn)
        killed = look(max(int(watch) - 1, 0), "watching", watching=True)
    fp_end = last_with(run.frames[-300:], "pay")
    sts1 = stations(fp_end)
    changed = {k: (sts0.get(k), v) for k, v in sts1.items() if sts0.get(k) != v}
    if changed:
        print("  stations that changed: " + "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items()))
    if not worked:
        print("  step: MASTER ARM back to OFF")
        click(run, "mmc.MasterArmSw", defs, value=0.0, fn=fn)
    by_hud = mrm0 is not None and mrm1 is not None and mrm1 < mrm0
    by_pay = pay0 is not None and pay1 is not None and pay1 < pay0
    fired = seen or by_hud or by_pay
    print("  a missile left the aircraft (" + ", ".join(w for w, ok in (("seen in flight", seen), ("HUD count down", by_hud),
          ("payload down", by_pay)) if ok) + ")" if fired else
          "  no missile seen by any of the four checks (HUD count, weapons in flight, new world objects, payload): "
          "tell me what the cockpit showed (right MFD SMS page, HUD, a launch sound, a missile on the external view)")
    if worked:
        print("  the target was destroyed" if killed else f"  the target was not seen destroyed in {watch} s (--watch longer?)")
    return {"fired": fired, "payload": [pay0, pay1], "hud_mrm": [mrm0, mrm1], "route": worked, "killed": killed}


def _lines(text):
    return [x.rstrip() for x in text.replace("\r", "").split("\n") if x.strip() and not set(x.strip()) <= {"-"}]


def stage_ind(link, tag="", out_dir=None, first=0, last=49, show=None):
    """Ask the module for the drawn elements (and the text) of the cockpit displays and print a summary. The
    HUD and the MFD pages list the elements they are drawing: what appears when a target is designated or
    locked shows in the difference between two saves (`inddiff`)."""
    import datetime
    link.events.pop("ind_end", None)
    link.ind_items.clear()
    link.ind(first, last)
    end = time.time() + 5.0
    while time.time() < end and "ind_end" not in link.events:
        link.next(timeout=0.3)
    if "ind_end" not in link.events:
        print("  no answer: reinstall the Export module (python dcs\\stick\\install_export.py) and restart the mission")
        return None
    items = [{"id": i, "text": t} for i, t in sorted(link.ind_items.items())]
    print(f"  {len(items)} displays (ids {first} to {last}); lines include names of drawn elements and values")
    for it in items:
        lines = _lines(it["text"])
        full = show is not None and (not show or it["id"] in show)
        print(f"\n  display {it['id']}: {len(it['text'])} characters, {len(lines)} lines")
        for x in (lines if full else lines[:6]):
            print("     " + x[:120])
        if not full and len(lines) > 6:
            print(f"     ... {len(lines) - 6} more (--show {it['id']} prints them, inddiff compares two saves)")
    r = {"ev": "ind", "items": items}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"stick_ind_{tag + '_' if tag else ''}{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w") as fh:
            json.dump(r, fh, indent=1)
        print(f"\n  saved: {path}")
    return r


def ind_diff(path_a, path_b, ids=None, out=None):
    """Lines present in one saved `ind` and not the other, per display: + only in B, - only in A."""
    import collections
    out = out or print
    docs = []
    for p in (path_a, path_b):
        with open(p) as fh:
            docs.append({it["id"]: _lines(it["text"]) for it in json.load(fh)["items"]})
    a, b = docs
    changed = 0
    for i in sorted(set(a) | set(b)):
        if ids and i not in ids:
            continue
        ca, cb = collections.Counter(a.get(i, [])), collections.Counter(b.get(i, []))
        plus, minus = list((cb - ca).elements()), list((ca - cb).elements())
        if not plus and not minus:
            continue
        changed += 1
        out(f"\n  display {i}: {len(minus)} lines only in A, {len(plus)} only in B")
        for x in minus[:40]:
            out("   - " + x[:120])
        for x in plus[:40]:
            out("   + " + x[:120])
    if not changed:
        out("  no difference between the two saves (in the displays compared)")
    return changed


def _fcr_id(run, default=4):
    fd = last_with(run.frames[-400:], "rd")
    return (fd.raw["rd"].get("fcr") if fd else None) or default


def ind_text(run, link, ident):
    """The lines of one display, asked for now."""
    link.events.pop("ind_end", None)
    link.ind(ident, ident)
    end = time.time() + 3.0
    while time.time() < end and "ind_end" not in link.events:
        link.next(timeout=0.2)
    return _lines(link.ind_items.get(ident, ""))


CURSOR_TRIES = [
    ("A", "throttle cursor switch fwd, then aft (what the keys ; and . press)", [],
     [("hotas.THROTTLE_RDR_CURSOR_FWD", None, None), ("hotas.THROTTLE_RDR_CURSOR_AFT", None, None)]),
    ("B", "the same after display management LEFT", ["hotas.STICK_DISP_MANAGE_LEFT"],
     [("hotas.THROTTLE_RDR_CURSOR_FWD", None, None), ("hotas.THROTTLE_RDR_CURSOR_AFT", None, None)]),
    ("C", "the same after display management RIGHT", ["hotas.STICK_DISP_MANAGE_RIGHT"],
     [("hotas.THROTTLE_RDR_CURSOR_FWD", None, None), ("hotas.THROTTLE_RDR_CURSOR_AFT", None, None)]),
    ("D", "cursor axis Y (3046) to +1 then -1", [],
     [("hotas.THROTTLE_RDR_CURSOR_Y_AXIS", 1.0, 0.0), ("hotas.THROTTLE_RDR_CURSOR_Y_AXIS", -1.0, 0.0)]),
    ("E", "cursor axis X (3047) to +1 then -1", [],
     [("hotas.THROTTLE_RDR_CURSOR_X_AXIS", 1.0, 0.0), ("hotas.THROTTLE_RDR_CURSOR_X_AXIS", -1.0, 0.0)]),
    ("F", "cursor ENABLE pressed, then the switch fwd and aft", ["hotas.THROTTLE_ENABLE"],
     [("hotas.THROTTLE_RDR_CURSOR_FWD", None, None), ("hotas.THROTTLE_RDR_CURSOR_AFT", None, None)]),
    ("G", "the switch fwd and aft through the device's SetCommand (how a key binding delivers it)", [],
     [("hotas.THROTTLE_RDR_CURSOR_FWD", None, None, "s"), ("hotas.THROTTLE_RDR_CURSOR_AFT", None, None, "s")]),
    ("H", "the switch fwd and aft through the global command queue (LoSetCommand)", [],
     [("hotas.THROTTLE_RDR_CURSOR_FWD", None, None, "l"), ("hotas.THROTTLE_RDR_CURSOR_AFT", None, None, "l")]),
    ("I", "the cursor axes through the device's SetCommand: Y then X, +1 and -1", [],
     [("hotas.THROTTLE_RDR_CURSOR_Y_AXIS", 1.0, 0.0, "s"), ("hotas.THROTTLE_RDR_CURSOR_Y_AXIS", -1.0, 0.0, "s"),
      ("hotas.THROTTLE_RDR_CURSOR_X_AXIS", 1.0, 0.0, "s"), ("hotas.THROTTLE_RDR_CURSOR_X_AXIS", -1.0, 0.0, "s")]),
]


def stage_cursortry(run, gains, defs, hold=2.5, only=None, alt=None, match_alt=False):
    """Try each way of moving the radar cursor in turn, announced, so the pilot can say which moved it.
    The cursor is on the radar page; the result is read by eye (its position is not in the display text)."""
    fn = hold_fn(run, gains, alt, match_alt) if gains else None
    print("  Watch the radar page (A-A mode, radar page up). Each try is announced; note the letters where the\n"
          "  cursor visibly moves. Hands off the keyboard and mouse.")
    settle(run, 1.0, fn)
    done = []
    for tag, text, pre, moves in CURSOR_TRIES:
        if only and tag not in only.upper():
            continue
        print(f"\n  try {tag}: {text}")
        settle(run, 1.5, fn)
        for spec in pre:
            r = click(run, spec, defs, fn=fn)
            if r is not None and not r.get("ok"):
                break
        for mv in moves:
            spec, down, up = mv[:3]
            r = click(run, spec, defs, value=down, hold=hold, fn=fn, release=up, mode=mv[3] if len(mv) > 3 else None)
            if r is not None and not r.get("ok"):
                break
            settle(run, 0.5, fn)
        done.append(tag)
    print("\n  Tell me the letters (A-I) where the cursor moved, or 'none'.")
    return {"tried": done}


def _pairs(lines):
    """The radar page's lines that are two integers ('174 248'): the cursor's place is one of them."""
    import re
    return [x.strip() for x in lines if re.fullmatch(r"-?\d+ -?\d+", x.strip())]


def stage_cursormap(run, gains, defs, hold=1.0, max_steps=15, out_dir=None, alt=None, match_alt=False):
    """Slew the radar cursor to each edge of the radar page (left, right, back to the middle, near, far) and save the
    page after every step, to learn how the cursor's readout maps to azimuth and range."""
    import datetime
    fn = hold_fn(run, gains, alt, match_alt) if gains else None
    settle_setup(run, fn)
    link = run.link
    fid = _fcr_id(run)
    settle(run, 1.0, fn)
    fo = last_with(run.frames[-300:], "obj")
    nr = nearest(fo) if fo else None
    page = ind_text(run, link, fid)
    if not page:
        print(f"  no text from the radar page (display {fid}): is the radar page up on the left MFD (A-A mode)?")
        return None
    caps = [{"step": "start", "pairs": _pairs(page), "lines": page,
             "target": None if nr is None else {"km": nr[1] / 1000, "az": nr[2], "dalt": nr[3]}}]
    print(f"  radar page is display {fid}; target {('%.1f km, %+.0f deg' % (nr[1] / 1000, nr[2])) if nr else '?'}")
    print(f"  start: number pairs on the page {caps[0]['pairs']}")

    def go(spec, label, until=None):
        prev, still = caps[-1]["pairs"], 0
        for i in range(max_steps):
            click(run, spec, defs, hold=hold, fn=fn, verbose=False)
            settle(run, 0.3, fn)
            cur = ind_text(run, link, fid)
            pr = _pairs(cur)
            fo2 = last_with(run.frames[-100:], "obj")
            n2 = nearest(fo2) if fo2 else None
            caps.append({"step": f"{label}{i + 1}", "pairs": pr, "lines": cur,
                         "target": None if n2 is None else {"km": n2[1] / 1000, "az": n2[2], "dalt": n2[3]}})
            changed = [(a, b) for a, b in zip(prev, pr) if a != b] if len(prev) == len(pr) else [("count", len(pr))]
            tuc = any("FCR_TUC_contactData" in x for x in cur)
            print(f"    {label} {i + 1}: {pr}{'  TARGET UNDER CURSOR' if tuc else ''}"
                  + ("" if changed else "  (no change)"))
            still = 0 if changed else still + 1
            prev = pr
            if still >= 2 or (until and until(pr)):
                return pr
        return prev
    start = caps[0]["pairs"]
    print("  left edge:")
    go("hotas.THROTTLE_RDR_CURSOR_LEFT", "left")
    print("  right edge:")
    go("hotas.THROTTLE_RDR_CURSOR_RIGHT", "right", )
    print("  back to the middle:")
    go("hotas.THROTTLE_RDR_CURSOR_LEFT", "middle", until=lambda pr: pr == start or (pr and start and pr[0] <= start[0]))
    print("  near edge (the range scale may step down):")
    go("hotas.THROTTLE_RDR_CURSOR_AFT", "near")
    print("  far edge (the range scale may step up):")
    go("hotas.THROTTLE_RDR_CURSOR_FWD", "far")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"stick_cursormap_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w") as fh:
            json.dump(caps, fh, indent=1)
        print(f"\n  saved: {path}  (send me this file)")
    return {"steps": len(caps) - 1}


def stage_cursorprobe(run, gains, defs, steps=8, hold=0.5, out_dir=None, alt=None, match_alt=False):
    """Slew the radar cursor out in range `steps` times and back, and print, after each move, which lines
    of the radar page changed: the ones that follow the cursor (its range, its altitude coverage) are
    what an automatic lock can steer by. The radar must be the page on the MFD (A-A mode)."""
    import datetime
    import collections
    link = run.link
    fn = hold_fn(run, gains, alt, match_alt) if gains else None
    fid = _fcr_id(run)
    fr0 = settle(run, 1.0, fn)
    f_o = last_with(fr0, "obj")
    nr = nearest(f_o) if f_o else None
    print(f"  radar page is display {fid}; target {nr[0] + ' at %.1f km (%.1f NM)' % (nr[1] / 1000, nr[1] / 1852) if nr else 'not in the world objects'}")
    prev = ind_text(run, link, fid)
    if not prev:
        print("  no text from the radar page: is the radar page up on an MFD? (display %d)" % fid)
        return None
    caps = [{"step": "start", "lines": prev}]
    for direction, spec in (("out", "hotas.THROTTLE_RDR_CURSOR_FWD"), ("back", "hotas.THROTTLE_RDR_CURSOR_AFT")):
        for i in range(steps):
            click(run, spec, defs, hold=hold, fn=fn, verbose=False)
            settle(run, 0.5, fn)
            cur = ind_text(run, link, fid)
            fr = settle(run, 0.1, fn)
            fo = last_with(fr, "obj")
            nr = nearest(fo) if fo else None
            ca, cb = collections.Counter(prev), collections.Counter(cur)
            minus, plus = list((ca - cb).elements()), list((cb - ca).elements())
            ls = lock_state(last_with(fr, "rd")) if last_with(fr, "rd") else None
            print(f"  cursor {direction} {i + 1}: radar {ls}, target {('%.1f km' % (nr[1] / 1000)) if nr else '?'}; "
                  f"{len(minus)} lines gone, {len(plus)} new")
            for x in minus[:8]:
                print("      - " + x[:100])
            for x in plus[:8]:
                print("      + " + x[:100])
            caps.append({"step": f"{direction}{i + 1}", "lines": cur})
            prev = cur
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"stick_cursor_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w") as fh:
            json.dump(caps, fh, indent=1)
        print(f"\n  saved: {path}")
    return {"steps": len(caps) - 1}
