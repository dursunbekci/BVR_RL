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


def click(run, spec, defs, value=None, hold=0.2, fn=None, verbose=True):
    """Press (and release after `hold` s) a control. Returns the module's answers."""
    dev, cmd, down, up = resolve(spec, defs, value)
    link = run.link
    n0 = link.clicks_seen if hasattr(link, "clicks_seen") else 0
    link.click(dev, cmd, down)
    settle(run, max(hold, 0.1), fn)
    if up is not None:
        link.click(dev, cmd, up)
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
    """The nearest other aircraft in the world objects: (name, range m, relative bearing deg, altitude difference m)."""
    objs = f.raw.get("obj") or []
    if not objs or f.x is None:
        return None
    best = None
    for o in objs:
        if o.get("x") is None or o.get("z") is None:
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


def hold_fn(run, gains, alt=None, match_alt=False):
    """A function (frame, tau) -> axes that keeps the heading and speed as they are now and the altitude as
    they are, or at `alt`, or at the nearest aircraft's altitude (match_alt): a radar sees a target best at
    its own height."""
    from stick_test import Hold
    f0 = run.next()
    for _ in range(12):                     # the world objects come every 5th frame: wait for one
        if "obj" in f0.raw or not match_alt:
            break
        f0 = run.next()
    h = Hold(gains, bank_max=math.radians(30))
    hdg, v = f0.hdg, f0.tas
    if match_alt and alt is None:
        alt = target_altitude(run)
        print(f"  matching the target's altitude: {alt:.0f} m (now {f0.alt:.0f})" if alt else
              "  --match-alt: no other aircraft in the world objects yet; keeping the altitude")
    alt = alt if alt is not None else f0.alt
    return lambda f, tau: h.axes(f, hdg, alt, v=v)


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


def stage_lock(run, gains, defs, scan=False, alt=None, match_alt=False, aa=None):
    """Try to lock the nearest target: A-A mode, then the target management switch up; with `scan`
    also slew the radar cursor in range between presses. Prints what the radar reports after each step."""
    fn = hold_fn(run, gains, alt, match_alt) if gains else None
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
        print("  scan: slewing the radar cursor in range and pressing target management up after each move")
        for i in range(10):
            click(run, "hotas.THROTTLE_RDR_CURSOR_FWD", defs, hold=0.4, fn=fn, verbose=False)
            click(run, "hotas.STICK_TGT_MANAGE_UP", defs, fn=fn, verbose=False)
            st = _observe(run, 1.5, fn, f"scan {i + 1}")
            if st.get("state") in ("designated", "stt"):
                break
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


def stage_fire(run, gains, defs, force=False, alt=None, match_alt=False, aa=None):
    """Arm, release, watch: master arm on, the weapon release button, then ten seconds of watching the
    payload and the world objects for a missile; master arm back to OFF at the end."""
    fn = hold_fn(run, gains, alt, match_alt) if gains else None
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
    objs0 = {o.get("id") for o in ((f_o.raw.get("obj") or []) if f_o else [])}
    sts0 = stations(f_p)
    print(f"  payload before: {pay0}  {sts0 or ''}; world objects: {len(objs0)}")
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
    print("  step: MASTER ARM")
    click(run, "mmc.MasterArmSw", defs, value=1.0, fn=fn)
    settle(run, 1.0, fn)
    print("  step: weapon release button")
    click(run, "hotas.STICK_WEAPON_RELEASE", defs, hold=0.4, fn=fn)
    new, pay1 = [], pay0
    for i in range(10):
        fr = settle(run, 1.0, fn)
        fp, fo2 = last_with(fr, "pay"), last_with(fr, "obj")
        pay1 = payload_count(fp) if fp else pay1
        new = [o for o in ((fo2.raw.get("obj") or []) if fo2 else []) if o.get("id") not in objs0]
        print(f"    +{i + 1} s: payload {pay1}, new world objects {[(o.get('name'), o.get('l1'), o.get('l2')) for o in new]}")
        if new and (pay0 is None or pay1 != pay0):
            break
    fp_end = last_with(run.frames[-300:], "pay")
    sts1 = stations(fp_end)
    changed = {k: (sts0.get(k), v) for k, v in sts1.items() if sts0.get(k) != v}
    if changed:
        print("  stations that changed: " + "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items()))
    print("  step: MASTER ARM back to OFF")
    click(run, "mmc.MasterArmSw", defs, value=0.0, fn=fn)
    fired = bool(new) or (pay0 is not None and pay1 is not None and pay1 < pay0)
    print("  a missile left the aircraft" if fired else
          "  no missile seen: check the SMS page on the right MFD (is AIM-120 selected, is the master arm switch up?) "
          "and tell me what the cockpit shows")
    return {"fired": fired, "payload": [pay0, pay1]}


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
