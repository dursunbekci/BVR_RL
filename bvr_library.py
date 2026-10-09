"""
bvr_library.py  —  Parameter library for airframes, missiles, radars, platforms
===============================================================================

Every number the simulation uses to model an aircraft, its radar and its
missile lives in a library item: a small JSON file with the values and, for
each value, a note saying where it came from.

    library/builtin/<kind>/<id>.json   shipped with the code, read-only
    library/user/<kind>/<id>.json      user copies, editable

Kinds:
    airframe   mass, aerodynamics, engine, limits, autopilot gains
    missile    motor, drag, guidance, seeker, datalink rules
    radar      detection range, gimbal limits, track accuracy
    platform   an airframe + radar + missile, with weapon count, radar
               cross-section and the command set the agent flies it with

The schema below (labels, units, groups, valid ranges, help) is the single
source of truth for what a parameter means; items only hold values. Angles
are stored in degrees for readability and converted to radians here.

Built-ins are never written by this module. To change one, copy it
(copy_item) and edit the copy: a user id can never shadow a built-in id.
"""

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LIB_DIR = HERE / "library"
BUILTIN_DIR = LIB_DIR / "builtin"
USER_DIR = LIB_DIR / "user"
ENVELOPE_DIR = LIB_DIR / "envelopes"

KINDS = ("airframe", "missile", "radar", "platform")
DEG2RAD = math.pi / 180.0
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$")


# ── schema ────────────────────────────────────────────────────────────
@dataclass
class P:
    key: str
    label: str
    unit: str
    group: str
    lo: float = None
    hi: float = None
    kind: str = "float"            # float | int | table | list | ref
    advanced: bool = False
    help: str = ""
    ref: str = None                # for kind="ref": which item kind it points to
    default: float = None          # used when an item predates this parameter


SCHEMA = {
    "airframe": [
        P("MASS_EMPTY", "Empty mass", "kg", "Mass", 200, 80_000),
        P("FUEL_FULL", "Internal fuel", "kg", "Mass", 20, 40_000),
        P("MASS_WPNS", "Stores mass", "kg", "Mass", 0, 15_000,
          help="Carried weapons and pylons, treated as constant."),
        P("S_REF", "Wing reference area", "m²", "Aerodynamics", 0.5, 300),
        P("CD0", "Zero-lift drag coefficient", "-", "Aerodynamics", 0.004, 0.1),
        P("K_IND", "Induced-drag factor", "-", "Aerodynamics", 0.01, 0.6,
          help="CD = CD0 + K·CL² + transonic rise. K = 1/(π·e·AR)."),
        P("CL_ALPHA", "Lift-curve slope", "1/rad", "Aerodynamics", 1, 8),
        P("CL_MAX", "Maximum lift coefficient", "-", "Aerodynamics", 0.4, 3.5,
          help="Caps the load factor the wing can produce at low speed."),
        P("CL_MIN", "Minimum lift coefficient", "-", "Aerodynamics", -3, 0),
        P("DRAG_RISE_M0", "Drag rise starts", "Mach", "Transonic drag rise", 0.4, 1.3),
        P("DRAG_RISE_M1", "Drag rise peaks", "Mach", "Transonic drag rise", 0.5, 1.6,
          help="Extra drag rises along a quarter sine from 'starts' to its peak here."),
        P("DRAG_RISE_M2", "Wave drag settles", "Mach", "Transonic drag rise", 0.6, 3.0,
          help="After the peak the extra drag decays exponentially until this Mach, then "
               "stays at that value: supersonic wave drag does not go away."),
        P("DRAG_RISE_PEAK", "Peak extra CD", "-", "Transonic drag rise", 0, 0.3),
        P("DRAG_RISE_DECAY", "Decay rate after peak", "1/Mach", "Transonic drag rise", 0, 30),
        P("T_MIL_SL", "Military thrust, sea level", "N", "Engine", 500, 500_000),
        P("T_AB_SL", "Afterburner thrust, sea level", "N", "Engine", 500, 800_000,
          help="Set equal to military thrust for an engine without afterburner."),
        P("ALT_LAPSE", "Thrust altitude exponent", "-", "Engine", 0.2, 2.0,
          help="Thrust scales with (air density ratio)^exponent."),
        P("THRUST_MACH_TABLE", "Thrust factor vs Mach", "[Mach, factor]", "Engine",
          kind="table", help="Multiplies thrust; linear between points, flat beyond the ends."),
        P("FF_IDLE", "Fuel flow, idle", "kg/s", "Engine", 0, 5),
        P("FF_MIL", "Fuel flow, military", "kg/s", "Engine", 0, 10),
        P("FF_AB", "Fuel flow, afterburner", "kg/s", "Engine", 0, 30),
        P("THROT_TC", "Engine lag", "s", "Engine", 0.1, 20),
        P("NZ_POS_MAX", "Positive g limit", "g", "Limits", 1, 20),
        P("NZ_NEG_MAX", "Negative g limit", "g", "Limits", -10, 0),
        P("V_STALL", "Stall speed", "m/s", "Limits", 15, 200),
        P("MACH_MAX", "Maximum Mach", "Mach", "Limits", 0.2, 3.5),
        P("ALT_CEIL", "Service ceiling", "m", "Limits", 500, 30_000),
        P("PHI_MAX", "Maximum bank", "deg", "Limits", 20, 90),
        P("ROLL_MAX", "Maximum roll rate", "deg/s", "Limits", 10, 500),
        P("GAMMA_MAX", "Maximum flight-path angle", "deg", "Limits", 5, 80),
        P("K_HDG_PHI", "Bank per heading error", "-", "Autopilot", 0.2, 10, advanced=True),
        P("K_PHI_ROLL", "Roll rate per bank error", "1/s", "Autopilot", 0.2, 20, advanced=True),
        P("K_ALT_GAM", "Flight-path angle per altitude error", "rad/m", "Autopilot",
          0.00005, 0.5, advanced=True,
          help="Small values ease the climb angle off early and avoid overshoot; "
               "check the performance card's altitude step response."),
        P("K_GAM_NZ", "Load factor per flight-path error", "1/rad", "Autopilot", 0.2, 20,
          advanced=True),
        P("BANK_NZ_MARGIN", "Load-factor share usable for turning", "-", "Autopilot",
          0.3, 1.0, advanced=True,
          help="The rest is kept for the altitude loop, so turns hold altitude."),
        P("K_V_INT", "Throttle per integrated speed error", "1/(m·s⁻¹·s)", "Autopilot",
          0.0, 0.05, advanced=True, default=0.0,
          help="Removes the steady speed error a proportional-only hold leaves. 0 turns "
               "it off, as in items saved before this parameter existed."),
        P("K_V_THROT", "Throttle per speed error", "1/(m/s)", "Autopilot", 0.001, 0.2,
          advanced=True),
    ],
    "missile": [
        P("MASS0", "Launch mass", "kg", "Mass and motor", 10, 2000),
        P("MASS_PROP", "Propellant mass", "kg", "Mass and motor", 1, 1500),
        P("BOOST_TIME", "Motor burn time", "s", "Mass and motor", 0.5, 60),
        P("THRUST", "Average thrust", "N", "Mass and motor", 500, 200_000),
        P("S_REF", "Body cross-section", "m²", "Aerodynamics", 0.001, 0.5),
        P("DRAG_TABLE", "Drag coefficient vs Mach", "[Mach, CD]", "Aerodynamics", kind="table"),
        P("K_INDUCED", "Induced drag factor", "", "Aerodynamics", 0, 2, advanced=True, default=0.0,
          help="Drag of the lift a turn takes: CD += K × CL², CL = lift / (q·S). The same pull "
               "costs more in thin air. 0: drag depends on Mach only (DRAG_TABLE then has to "
               "include the turning a typical flight does)."),
        P("N_PRO_NAV", "Proportional-navigation gain", "-", "Guidance", 2, 8),
        P("MAX_G", "Maximum manoeuvre", "g", "Guidance", 5, 80),
        P("MIN_MANOEUVRE_V", "Speed below which manoeuvre degrades", "m/s", "Guidance", 50, 800),
        P("FUZE_RADIUS", "Lethal radius", "m", "Guidance", 1, 100),
        P("SEEKER_FOV", "Seeker field of view (half-angle)", "deg", "Seeker", 2, 90),
        P("SEEKER_RANGE", "Seeker acquisition range", "m", "Seeker", 500, 100_000,
          help="Against a target of the reference RCS; scales with (RCS/ref)^¼."),
        P("SEEKER_REF_RCS_M2", "Seeker reference RCS", "m²", "Seeker", 0.0001, 100),
        P("HANDOFF_MIN", "Hand-off range, minimum", "m", "Datalink", 500, 100_000,
          help="The range to target at which the seeker is switched on is drawn "
               "uniformly between minimum and maximum for each missile."),
        P("HANDOFF_MAX", "Hand-off range, maximum", "m", "Datalink", 500, 100_000),
        P("SUPPORT_TIMEOUT", "Datalink support timeout", "s", "Datalink", 0.5, 60,
          help="Before hand-off, a missile left without datalink updates this long misses."),
        P("MAX_FLIGHT", "Maximum flight time", "s", "Datalink", 5, 600),
        P("LOFT_ANGLE", "Loft: climb angle", "deg", "Guidance", 0, 60, advanced=True, default=0.0,
          help="A long shot first climbs at this angle into thinner air, before guiding on "
               "the target. 0 (items saved before this parameter existed): no loft."),
        P("LOFT_DIVE", "Loft: dive when the target is this far below", "deg", "Guidance", 0, 60,
          advanced=True, default=0.0,
          help="The climb ends, for good, when the target is this many degrees below the "
               "missile, or within 5 km horizontally."),
    ],
    "radar": [
        P("MAX_RANGE", "Detection range", "m", "Detection", 1000, 400_000,
          help="Against a target of the reference RCS; scales with (RCS/ref)^¼."),
        P("REF_RCS_M2", "Reference target RCS", "m²", "Detection", 0.0001, 100),
        P("FOV_AZ", "Gimbal limit, azimuth", "deg", "Detection", 5, 180),
        P("FOV_EL", "Gimbal limit, elevation", "deg", "Detection", 5, 180),
        P("REF_RANGE", "Accuracy reference range", "m", "Accuracy", 1000, 300_000,
          help="The errors below apply at this range and grow with (range/ref)²."),
        P("SIG_AZ_REF", "Azimuth error", "mrad", "Accuracy", 0.01, 50),
        P("SIG_EL_REF", "Elevation error", "mrad", "Accuracy", 0.01, 50),
        P("SIG_RNG_REF", "Range error", "m", "Accuracy", 0.1, 2000),
        P("SIG_RDOT_REF", "Range-rate error", "m/s", "Accuracy", 0.01, 200),
        P("SIG_ADOT_REF", "Angle-rate error", "mrad/s", "Accuracy", 0.001, 10),
        P("TAU", "Error correlation time", "s", "Accuracy", 0.05, 30),
        P("LOWDOP_CLOSURE", "Low-Doppler threshold", "m/s", "Clutter", 0, 300,
          help="Below this closure speed the track errors grow (beam manoeuvre)."),
        P("LOWDOP_MAX_INFLATE", "Error growth at zero closure", "×", "Clutter", 1, 50),
    ],
    "platform": [
        P("airframe", "Airframe", "", "Components", kind="ref", ref="airframe"),
        P("radar", "Radar", "", "Components", kind="ref", ref="radar"),
        P("missile", "Missile", "", "Components", kind="ref", ref="missile"),
        P("WPN_COUNT", "Missiles carried", "", "Loadout", 0, 6, kind="int",
          help="At most 6: the agent's 'missiles left' input is scaled for 0 to 6."),
        P("RCS_M2", "Radar cross-section", "m²", "Signature", 0.0001, 100,
          help="How visible this aircraft is to enemy radar and missile seekers."),
        P("SPEED_CMDS", "Speed choices", "m/s", "Agent commands", kind="list",
          help="The five speeds the agent can command, slowest first (four before SIM_REV 14: "
               "a platform saved then needs a fifth, e.g. 460 m/s for an F-16)."),
        P("ALT_MIN_OP", "Lowest commanded altitude", "m", "Agent commands", 100, 20_000),
        P("ALT_MAX_OP", "Highest commanded altitude", "m", "Agent commands", 500, 25_000),
        P("CLIMB_FPA_LO", "Climb angle, low altitude", "deg", "Agent commands", 1, 60),
        P("CLIMB_FPA_HI", "Climb angle, high altitude", "deg", "Agent commands", 1, 60),
        P("CLIMB_FPA_ALT_LO", "Low-altitude band ends", "m", "Agent commands", 0, 25_000),
        P("CLIMB_FPA_ALT_HI", "High-altitude band starts", "m", "Agent commands", 0, 25_000,
          help="The climb angle eases linearly from the low to the high value between these."),
        P("CLIMB_MACH", "DCS-AI climb: Mach held", "", "Agent commands", 0, 2, advanced=True,
          default=0.0,
          help="Above 0: while climbing, fly a fixed throttle and hold this Mach with the nose, "
               "whatever speed is commanded, as the DCS AI does on a route. 0: climb at the "
               "commanded speed (the climb-angle limits above)."),
        P("CLIMB_THROTTLE", "DCS-AI climb: throttle", "", "Agent commands", 0, 1, advanced=True,
          default=1.0,
          help="The throttle of a DCS-AI climb: 0.9 is full military power, 1.0 full afterburner."),
        P("TURN_BANK_MAX", "DCS-AI turn: bank limit", "deg", "Agent commands", 0, 90, advanced=True,
          default=0.0,
          help="Above 0: heading changes are flown at no more than this bank, as the DCS AI on a "
               "route flies them (45 deg). 0: the airframe's own limits."),
        P("HOT_TURN_BANK", "Turns toward the enemy: bank", "deg", "Agent commands", 0, 90,
          advanced=True, default=0.0,
          help="Above 0: a heading command within 15 deg of the bearing to the enemy, with "
               "the aircraft more than 30 deg off it, is flown at this bank until within 10 deg, "
               "as the DCS bridge flies it (OPT hotturn: a guns-only attack, about 7 deg/s at "
               "4 g where a route turns at 1.6). Other turns keep TURN_BANK_MAX. 0: off."),
        P("AUTO_DEFEND", "Automatic missile defence", "", "Agent commands", 0, 1, kind="int",
          advanced=True, default=0,
          help="1: once an enemy missile is 12-15 km away the aircraft defends itself, as the "
               "DCS AI does when the DCS bridge hands it the aircraft (OPT autodefend), fitted "
               "to the DCS AI's recorded defences: a break toward the beam (70-110 deg off the "
               "missile) at 3-4.5 g, rolled past the vertical to dive 1.5-4 km at 15-35 deg, at "
               "its fastest speed, holding fire; each defence draws its numbers. Until then, "
               "and when the missile is gone, the agent flies. 0: the agent flies throughout."),
    ],
}

# Stored in degrees (or mrad), used in radians by the models.
_TO_SI = {("airframe", "PHI_MAX"): DEG2RAD, ("airframe", "ROLL_MAX"): DEG2RAD,
          ("airframe", "GAMMA_MAX"): DEG2RAD, ("missile", "SEEKER_FOV"): DEG2RAD,
          ("missile", "LOFT_ANGLE"): DEG2RAD, ("missile", "LOFT_DIVE"): DEG2RAD,
          ("radar", "FOV_AZ"): DEG2RAD, ("radar", "FOV_EL"): DEG2RAD,
          ("platform", "CLIMB_FPA_LO"): 1.0, ("platform", "CLIMB_FPA_HI"): 1.0}
_TO_SI_DIV = {("radar", "SIG_AZ_REF"): 1000.0, ("radar", "SIG_EL_REF"): 1000.0,
              ("radar", "SIG_ADOT_REF"): 1000.0}


def schema_json() -> dict:
    """The schema as plain data, for the GUI."""
    return {k: [p.__dict__ for p in v] for k, v in SCHEMA.items()}


# ── storage ───────────────────────────────────────────────────────────
class LibraryError(ValueError):
    pass


def _path(root, kind, item_id):
    return root / kind / f"{item_id}.json"


def list_items(kind: str = None) -> list:
    out = []
    for k in (KINDS if kind is None else (kind,)):
        for root, builtin in ((BUILTIN_DIR, True), (USER_DIR, False)):
            for f in sorted((root / k).glob("*.json")):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                out.append({"id": d.get("id", f.stem), "kind": k, "name": d.get("name", f.stem),
                            "builtin": builtin, "based_on": d.get("based_on"),
                            "description": d.get("description", "")})
    return out


def list_problems() -> list:
    """Why list_items() may be missing something, as readable strings.

    list_items() skips unreadable files so one bad file cannot empty the
    library; this reports what it skipped and any missing folder, so the GUI
    can say why a list is empty instead of showing nothing.
    """
    out = []
    for k in KINDS:
        if not (BUILTIN_DIR / k).is_dir():
            out.append(f"missing folder {BUILTIN_DIR / k}")
        for root in (BUILTIN_DIR, USER_DIR):
            for f in sorted((root / k).glob("*.json")):
                try:
                    json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    out.append(f"unreadable {f.relative_to(LIB_DIR)}: {e}")
    return out


def get_item(kind: str, item_id: str) -> dict:
    if kind not in KINDS:
        raise LibraryError(f"unknown kind {kind!r}")
    for root, builtin in ((BUILTIN_DIR, True), (USER_DIR, False)):
        f = _path(root, kind, item_id)
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            d["builtin"] = builtin
            d["kind"] = kind
            return d
    raise LibraryError(f"no {kind} named {item_id!r} in the library")


def validate(item: dict) -> list:
    """Problems with an item, as readable strings; empty if it is usable."""
    errs = []
    kind = item.get("kind")
    if kind not in KINDS:
        return [f"unknown kind {kind!r}"]
    if not _ID_RE.match(str(item.get("id", ""))):
        errs.append("id: letters, digits, '-', '_' or '.', up to 48 characters")
    params = item.get("params", {})
    for p in SCHEMA[kind]:
        v = params.get(p.key, p.default)
        if v is None:
            errs.append(f"{p.label}: missing")
            continue
        if p.kind == "ref":
            try:
                get_item(p.ref, v)
            except LibraryError:
                errs.append(f"{p.label}: no {p.ref} named {v!r}")
        elif p.kind == "table":
            try:
                t = np.asarray(v, dtype=float)
                if t.ndim != 2 or t.shape[1] != 2 or len(t) < 2:
                    raise ValueError
                if np.any(np.diff(t[:, 0]) < 0):
                    errs.append(f"{p.label}: Mach values must not decrease")
                if np.any(t[:, 1] < 0):
                    errs.append(f"{p.label}: values must not be negative")
            except (ValueError, TypeError):
                errs.append(f"{p.label}: must be at least two [Mach, value] pairs")
        elif p.kind == "list":
            try:
                lst = [float(x) for x in v]
                if len(lst) != N_SPEED_CMDS:
                    errs.append(f"{p.label}: exactly {N_SPEED_CMDS} values" + (
                        " (four before SIM_REV 14: add a fifth, faster one, e.g. 460 m/s "
                        "for an F-16)" if len(lst) == 4 else ""))
                elif any(b <= a for a, b in zip(lst, lst[1:])):
                    errs.append(f"{p.label}: must increase")
                elif lst[0] <= 0:
                    errs.append(f"{p.label}: must be positive")
            except (ValueError, TypeError):
                errs.append(f"{p.label}: must be {N_SPEED_CMDS} numbers")
        else:
            try:
                x = float(v)
            except (ValueError, TypeError):
                errs.append(f"{p.label}: not a number")
                continue
            if not math.isfinite(x):
                errs.append(f"{p.label}: not a number")
            elif (p.lo is not None and x < p.lo) or (p.hi is not None and x > p.hi):
                errs.append(f"{p.label}: {x:g} {p.unit} is outside {p.lo:g} to {p.hi:g}")
            if p.kind == "int" and x != int(x):
                errs.append(f"{p.label}: must be a whole number")
    # Cross-parameter rules.
    g = lambda k: params.get(k)
    try:
        if kind == "airframe":
            if float(g("T_AB_SL")) < float(g("T_MIL_SL")):
                errs.append("Afterburner thrust must be at least military thrust")
            if not float(g("DRAG_RISE_M0")) < float(g("DRAG_RISE_M1")) < float(g("DRAG_RISE_M2")):
                errs.append("Drag rise: starts < peaks < ends")
        elif kind == "missile":
            if float(g("MASS_PROP")) >= float(g("MASS0")):
                errs.append("Propellant must be less than launch mass")
            if float(g("HANDOFF_MIN")) > float(g("HANDOFF_MAX")):
                errs.append("Hand-off minimum must not exceed maximum")
        elif kind == "platform":
            if float(g("ALT_MIN_OP")) >= float(g("ALT_MAX_OP")):
                errs.append("Lowest commanded altitude must be below the highest")
            if float(g("CLIMB_FPA_ALT_LO")) > float(g("CLIMB_FPA_ALT_HI")):
                errs.append("Climb bands: low-altitude end must not be above high-altitude start")
    except (TypeError, ValueError):
        pass                                # already reported per parameter
    return errs


def save_item(item: dict) -> Path:
    """Write a user item. Built-ins, and ids that a built-in uses, are refused."""
    item = copy.deepcopy(item)
    kind, item_id = item.get("kind"), str(item.get("id", ""))
    if kind not in KINDS:
        raise LibraryError(f"unknown kind {kind!r}")
    if _path(BUILTIN_DIR, kind, item_id).exists():
        raise LibraryError(f"{item_id!r} is a built-in {kind}; save your changes under a new id")
    errs = validate(item)
    if errs:
        raise LibraryError("; ".join(errs))
    item.pop("builtin", None)
    f = _path(USER_DIR, kind, item_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return f


def copy_item(kind: str, src_id: str, new_id: str, new_name: str = None) -> dict:
    src = get_item(kind, src_id)
    if not _ID_RE.match(new_id):
        raise LibraryError("id: letters, digits, '-', '_' or '.', up to 48 characters")
    for root in (BUILTIN_DIR, USER_DIR):
        if _path(root, kind, new_id).exists():
            raise LibraryError(f"a {kind} named {new_id!r} already exists")
    item = copy.deepcopy(src)
    item.update({"id": new_id, "name": new_name or f"{src.get('name', src_id)} (copy)",
                 "based_on": src_id, "builtin": False})
    save_item(item)
    return get_item(kind, new_id)


def delete_item(kind: str, item_id: str) -> None:
    if _path(BUILTIN_DIR, kind, item_id).exists():
        raise LibraryError("built-in items cannot be deleted")
    f = _path(USER_DIR, kind, item_id)
    if not f.exists():
        raise LibraryError(f"no user {kind} named {item_id!r}")
    users = [i["id"] for i in list_items("platform")
             if kind != "platform" and get_item("platform", i["id"])["params"].get(kind) == item_id]
    if users:
        raise LibraryError(f"used by platform(s) {', '.join(users)}; change those first")
    f.unlink()


# Bumped when a kind's model equations change, so that unchanged parameters
# under changed equations do not keep their old fingerprint. Airframe 2: the
# transonic drag rise became a quarter sine peaking at DRAG_RISE_M1 (it was a
# half sine that fell back to zero there before jumping to the peak).
_MODEL_REV = {"airframe": 2}

# How many speed choices a platform gives the agent (bvr_env's action space).
N_SPEED_CMDS = 5

# Changes to the simulation that are not library parameters but alter what a
# model trained on: stored in each checkpoint's scenario record so a resume
# can say it is a warm start. 1 = before these revisions were recorded.
SIM_REV = 15
_SIM_REV_NOTES = {
    2: "missile time-to-go fixed (it read 999 s while a missile closed): the "
       "time-to-go inputs, the defence reward and the scripted opponents' "
       "last-ditch defence all changed",
    3: "the defence shaping term became -0.8*(1-tanh(tgo/25)), 0 with no "
       "missile inbound (it was +0.8*tanh(tgo/25)), so the critic's values shift",
    4: "without a track the heading choice is now an offset from the last known "
       "bearing to the bandit, held, not from the aircraft's nose; four inputs "
       "were added (the previous heading choice, the turn still to fly)",
    5: "2v1: red opens on the nearer blue aircraft (it always opened on the lead), "
       "SHOOTER and ADAPTIVE also fire at the other blue aircraft while cranking, "
       "and a lost blue aircraft costs -1.0 (was -0.7)",
    6: "a missile can only be fired at a target within 60 deg of the nose "
       "(the fire action is masked beyond it); in 1v1 the radar field of view "
       "mostly kept shots inside this already, in 2v1 datalink shots had no limit",
    7: "changing the heading choice costs 0.04 per 180 deg (was 0.015), to stop "
       "the policy switching choices every few seconds (wing-rocking)",
    8: "half the episodes now start mirrored: the opponent gets the start the "
       "agent always had (nose on while the other is offset, beaming or running), "
       "so the agent also practises the worse start; expect a lower win rate",
    9: "scripted shooters fight under the agent's rules: their missiles are guided "
       "on their own radar track (not the agent's true position), so the 3-second "
       "support rule binds them too, and they fire only when the agent's fire mask "
       "would allow it from their side; they crank at most 45 deg (SHOOTER 45, was "
       "50; ADAPTIVE 30-45, was 35-70) so they keep their own track. 1v1 only: "
       "2v1 red is unchanged",
    10: "ADAPTIVE defends deep in half its episodes: against an inbound missile it "
        "dives to 1-2.5 km, beaming (as the DCS AI does) or turning away, instead "
        "of beaming and descending 1.5 km (4 km in the last 12 s). SHOOTER and "
        "2v1 red are unchanged",
    11: "launch envelopes depend on the target's altitude: the tables add a target "
        "altitude axis (1-13 km) and launch altitudes 1 and 6 km, and the fire mask, "
        "the range/R-max inputs and scripted red's shot range use the target's "
        "estimated altitude. AIM-120C-DCS from 9 km head-on reaches 84 km against a "
        "level target but 58 km against one at 1 km, which the old tables allowed",
    12: "in a third of ADAPTIVE episodes red flies like the DCS AI: it shoots at "
        "0.95-1.0 of its R-max and presses on after its missile goes active instead "
        "of turning cold, so it often shoots first (against dcs_v7: first in 23 of 60 "
        "fights, was 13 of 40). 2v1 red is reset to the old defence explicitly: since "
        "SIM_REV 10 it had kept the deep defence its constructor drew",
    13: "fixes for what dcs_v8 did in DCS: the energy shaping weight is 0.07 (was "
        "0.15; the policy climbed to 11-14 km in every fight); in half of the starts "
        "the second aircraft is anywhere from 2 to 12.5 km high (was within 2.5 km of "
        "the first); in 1/6 of ADAPTIVE episodes red keeps its distance, low and slow, "
        "and only turns to fight inside 25-45 km, so waiting it out ends in a timeout. "
        "And scripted red's bearing to the agent is taken from the two positions: it "
        "was mirrored whenever the agent was on red's left (30-177 deg wrong in 15 of "
        "40 fights), so red went hot, cranked and dragged the wrong way. ADAPTIVE is "
        "much stronger for it: dcs_v8 won 68% of 120 fights against it before, 50% after. "
        "ADAPTIVE also flies at DCS speeds: hot and cranking at 380-450 m/s (was 330-340), "
        "running and defending deep at 420-500 (was 400): the DCS AI holds Mach 1.3 while "
        "climbing and goes on to Mach 1.5",
    14: "a wider action space: headings 0, ±15, ±30, ±40, ±50, ±90, ±135, 180 (were "
        "0, ±30, ±50, ±90, ±135, 180) and five speeds (the F-16s' 220, 280, 340, 400 "
        "plus 460 m/s, Mach 1.5 at 9 km; GENERIC-UCAV adds a slow 150). The new choices "
        "start just below their neighbours in the converted checkpoint, so it flies as "
        "before until training finds a use for them. Start speeds are scaled by the "
        "platform's fastest choice over the F-16's 460 (was 400): F-16 starts are "
        "unchanged, a UCAV's 13% slower. And for the DCS platforms: F-16C-DCSAI-AD "
        "defends itself (AUTO_DEFEND, fitted to the DCS AI's recorded defences: a late "
        "split-S break toward the beam once the missile is 12-15 km away) and turns "
        "toward the enemy at 77 deg of bank (HOT_TURN_BANK); while it defends, the "
        "heading, altitude and speed choices are masked to one (they are not flown)",
    15: "reward balance, from an audit of dcs_v8's returns against ADAPTIVE (400 steps): "
        "the heading-change cost, the one reward that is not potential-based, added up to "
        "-0.7 to -2.8 an episode, as large as the terminal rewards, which are discounted "
        "(0.997^400 = 0.30 at the end of a long episode); discounted, a kill came to about "
        "-0.2, a mutual kill -1.2, a loss -1.4 and a timeout -1.9, so dying beat surviving. "
        "Heading cost 0.04 -> 0.02 per 180 deg, timeout -0.5 -> -0.8 (2v1 too)",
}


def fingerprint(item: dict) -> str:
    """Short hash of an item's parameter values: equal hashes, equal physics."""
    params = item.get("params", {})
    rev = _MODEL_REV.get(item.get("kind"))
    if rev is not None:
        params = dict(params, __model_rev__=rev)
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


# ── model configs ─────────────────────────────────────────────────────
class _Cfg:
    """Attribute bag with the same names the models read from F16Cfg etc."""
    def __init__(self, kind, item):
        self.id = item["id"]
        self.item = item
        self.fingerprint = fingerprint(item)
        for p in SCHEMA[kind]:
            v = item["params"].get(p.key, p.default)
            if p.kind in ("float", "int"):
                v = float(v) if p.kind == "float" else int(v)
                if (kind, p.key) in _TO_SI_DIV:
                    v = v / _TO_SI_DIV[(kind, p.key)]
                elif (kind, p.key) in _TO_SI and _TO_SI[(kind, p.key)] != 1.0:
                    v = v * _TO_SI[(kind, p.key)]
            setattr(self, p.key, v)


# Curves are small classes, not closures, so an env (which holds them) can be
# pickled: older stable-baselines3 sends env attributes between processes.
class _InterpCurve:
    def __init__(self, table):
        t = np.asarray(table, dtype=float)
        self.xs, self.ys = t[:, 0].copy(), t[:, 1].copy()

    def __call__(self, m):
        return float(np.interp(float(m), self.xs, self.ys))


class _CdRise:
    def __init__(self, m0, m1, m2, peak, decay):
        self.m0, self.m1, self.m2, self.peak, self.decay = m0, m1, m2, peak, decay

    def __call__(self, mach):
        m0, m1, m2, peak, decay = self.m0, self.m1, self.m2, self.peak, self.decay
        if mach < m0: return 0.0
        if mach < m1: return peak * math.sin(0.5 * math.pi * (mach - m0) / (m1 - m0))
        if mach < m2: return peak * math.exp(-decay * (mach - m1))
        return peak * math.exp(-decay * (m2 - m1))          # supersonic wave drag


def _interp_curve(table):
    return _InterpCurve(table)


def airframe_config(item_id: str) -> _Cfg:
    it = get_item("airframe", item_id)
    c = _Cfg("airframe", it)
    c.thrust_mach_factor = _interp_curve(c.THRUST_MACH_TABLE)
    c.cd_rise = _CdRise(c.DRAG_RISE_M0, c.DRAG_RISE_M1, c.DRAG_RISE_M2,
                        c.DRAG_RISE_PEAK, c.DRAG_RISE_DECAY)
    return c


def missile_config(item_id: str) -> _Cfg:
    c = _Cfg("missile", get_item("missile", item_id))
    c.drag_cd = _interp_curve(c.DRAG_TABLE)
    return c


def radar_config(item_id: str) -> _Cfg:
    return _Cfg("radar", get_item("radar", item_id))


@dataclass
class Platform:
    id: str
    name: str
    airframe: object
    radar: object
    missile: object
    wpn_count: int
    rcs: float
    speed_cmds: list
    alt_min: float
    alt_max: float
    climb_fpa_lo: float            # deg
    climb_fpa_hi: float            # deg
    climb_alt_lo: float
    climb_alt_hi: float
    climb_mach: float = 0.0        # > 0: climbs like the DCS AI (f16_sim climbMach)
    climb_throttle: float = 1.0
    turn_bank_max: float = 0.0     # rad; > 0: heading changes at no more bank (f16_sim bankMax)
    auto_defend: bool = False      # defends itself from inbound missiles (sim_world)
    hot_turn_bank: float = 0.0     # rad; > 0: turns toward the enemy at this bank (sim_world)
    items: dict = field(default_factory=dict)   # every item used, for run snapshots

    def climb_fpa(self, alt: float) -> float:
        """Climb-angle limit (rad) at this altitude."""
        f = min(max((alt - self.climb_alt_lo) / (self.climb_alt_hi - self.climb_alt_lo), 0.0), 1.0) \
            if self.climb_alt_hi > self.climb_alt_lo else float(alt >= self.climb_alt_hi)
        return (self.climb_fpa_lo + f * (self.climb_fpa_hi - self.climb_fpa_lo)) * DEG2RAD


_PLATFORM_CACHE = {}


def load_platform(item_id: str) -> Platform:
    """A platform with its components resolved into model configs (cached by content)."""
    it = get_item("platform", item_id)
    errs = validate(it)
    if errs:
        raise LibraryError(f"platform {item_id!r} is not usable: " + "; ".join(errs))
    pr = it["params"]
    parts = {k: get_item(k, pr[k]) for k in ("airframe", "radar", "missile")}
    key = (item_id, fingerprint(it)) + tuple(fingerprint(v) for v in parts.values())
    if key in _PLATFORM_CACHE:
        return _PLATFORM_CACHE[key]
    p = Platform(
        id=item_id, name=it.get("name", item_id),
        airframe=airframe_config(pr["airframe"]), radar=radar_config(pr["radar"]),
        missile=missile_config(pr["missile"]),
        wpn_count=int(pr["WPN_COUNT"]), rcs=float(pr["RCS_M2"]),
        speed_cmds=[float(x) for x in pr["SPEED_CMDS"]],
        alt_min=float(pr["ALT_MIN_OP"]), alt_max=float(pr["ALT_MAX_OP"]),
        climb_fpa_lo=float(pr["CLIMB_FPA_LO"]), climb_fpa_hi=float(pr["CLIMB_FPA_HI"]),
        climb_alt_lo=float(pr["CLIMB_FPA_ALT_LO"]), climb_alt_hi=float(pr["CLIMB_FPA_ALT_HI"]),
        climb_mach=float(pr.get("CLIMB_MACH", 0.0)), climb_throttle=float(pr.get("CLIMB_THROTTLE", 1.0)),
        turn_bank_max=float(pr.get("TURN_BANK_MAX", 0.0)) * DEG2RAD,
        auto_defend=bool(int(pr.get("AUTO_DEFEND", 0))),
        hot_turn_bank=float(pr.get("HOT_TURN_BANK", 0.0)) * DEG2RAD,
        items={"platform": it, **parts})
    _PLATFORM_CACHE[key] = p
    return p


# How the aircraft is flown (a DCS-AI climb and turn, an automatic missile
# defence) and which four speeds its speed choices mean, which a self-play
# snapshot does not depend on: what it sees and the choices it has are the same.
_FLYING_ONLY = ("CLIMB_MACH", "CLIMB_THROTTLE", "TURN_BANK_MAX", "HOT_TURN_BANK", "AUTO_DEFEND",
                "SPEED_CMDS")


def selfplay_group(platform_id: str) -> str:
    """Platforms in one group can fly against each other in self-play: the same
    airframe, radar, missile, loadout, signature and commands, differing at most
    in how they climb, turn and defend and in their four speeds (F-16C-DCSAI-AD,
    F-16C-DCSAI and F-16C-DCS)."""
    return _selfplay_group_of(get_item("platform", platform_id))


def _as_value(v):
    """Numbers compared by value: a platform saved with 1000 and one with 1000.0
    (the GUI editor and hand-written files differ) are the same aircraft."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_as_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _as_value(x) for k, x in v.items()}
    return v


def _selfplay_group_of(it: dict) -> str:
    params = {k: _as_value(v) for k, v in it.get("params", {}).items() if k not in _FLYING_ONLY}
    parts = [fingerprint({"params": _as_value(get_item(k, params[k]).get("params", {}))})
             for k in ("airframe", "radar", "missile")]
    return fingerprint({"params": {**params, "_parts": parts}})


def selfplay_compatible(a: str, b: str) -> bool:
    return a == b or selfplay_group(a) == selfplay_group(b)


def rcs_scale(target_rcs: float, ref_rcs: float) -> float:
    """Detection-range factor from the radar range equation: (RCS / ref)^(1/4)."""
    if target_rcs == ref_rcs:
        return 1.0
    return (float(target_rcs) / float(ref_rcs)) ** 0.25


def envelope_path(missile_cfg) -> Path:
    """Where the calibrated envelope for this exact missile lives."""
    return ENVELOPE_DIR / f"{missile_cfg.id}-{missile_cfg.fingerprint}.npz"


DEFAULT_PLATFORM = "F-16C"


# ── scenarios recorded in checkpoints ─────────────────────────────────
def scenario_record(platform_id: str, opponent_platform_id: str,
                    wingman_platform_id: str = None, fmt: str = "1v1") -> dict:
    """
    What a model is trained with: the format, every platform and every
    library item they use, in full. Stored on the model (model.bvr_scenario),
    so it is saved inside each checkpoint and survives later edits to the
    library. In 2v1, "platform" is the lead's and "wingman_platform" the
    wingman's; "opponent_platform" is always the enemy's.
    """
    rec = {"format": fmt, "platform": platform_id, "opponent_platform": opponent_platform_id,
           "sim_rev": SIM_REV, "fingerprints": {}, "items": {}}
    sides = [("platform", platform_id), ("opponent_platform", opponent_platform_id)]
    if wingman_platform_id is not None:
        rec["wingman_platform"] = wingman_platform_id
        sides.append(("wingman_platform", wingman_platform_id))
    for side, pid in sides:
        p = load_platform(pid)
        rec["items"][side] = copy.deepcopy(p.items)
        rec["fingerprints"][side] = {k: fingerprint(v) for k, v in p.items.items()}
    return rec


def scenario_of(model) -> dict:
    """A checkpoint's scenario; checkpoints from before the library were F-16C v F-16C."""
    rec = getattr(model, "bvr_scenario", None)
    if isinstance(rec, dict) and rec.get("platform"):
        return {"format": "1v1", **rec}
    return {"format": "1v1", "platform": DEFAULT_PLATFORM, "opponent_platform": DEFAULT_PLATFORM,
            "legacy": True}


def scenario_drift(rec: dict) -> list:
    """What changed since the checkpoint was saved: the simulation, and
    library items whose parameters were edited."""
    out = [f"trained before the change: {_SIM_REV_NOTES[r]}; this run is a warm start"
           for r in range(int(rec.get("sim_rev", 1)) + 1, SIM_REV + 1)]
    for side, fps in (rec.get("fingerprints") or {}).items():
        pid = rec.get(side)
        try:
            now = load_platform(pid)
        except LibraryError as e:
            out.append(f"{side} {pid!r}: {e}")
            continue
        for k, fp in fps.items():
            if fingerprint(now.items[k]) != fp:
                out.append(f"{side} {pid!r}: its {k} {now.items[k]['id']!r} has changed since training")
    return out
