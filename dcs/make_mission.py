"""
make_mission.py  —  build the DCS mission for a 1v1 against a trained policy
=============================================================================

Writes a .miz file you can open in DCS World (MISSION from the main menu, or
the Mission Editor). It has two F-16Cs over the Black Sea on the Caucasus map
(free with DCS), head-on and apart like the training starts, each carrying
four AIM-120Cs as the training platform does, and a MISSION START trigger
that runs bvr_bridge.lua (the file is packed into the .miz).

    pip install pydcs
    python dcs/make_mission.py                          # -> dcs/bvr_rl_1v1.miz
    python dcs/make_mission.py --range-km 70 --red-player --out my.miz
    python dcs/make_mission.py --defender               # -> dcs/bvr_rl_defender.miz
    python dcs/make_mission.py --stick-test             # -> dcs/bvr_rl_sticktest.miz
    python dcs/make_mission.py --stick-test --target    # -> dcs/bvr_rl_sticktest_target.miz
    python dcs/make_mission.py --blue-player --seed 3   # -> dcs/bvr_rl_1v1_stick_s3.miz, a start drawn as in training

  BLUE-1   the aircraft the policy flies (an AI unit; without dcs_live.py it
           flies a CAP and fights on its own)
  RED-1    its opponent: DCS AI, ordered by the bridge to attack BLUE-1, or
           you, with --red-player (needs a module you can fly, e.g. the F-16C)

--defender: you fly RED-1 in the free Su-25T, which has no radar missiles,
so you can't shoot back beyond visual range: you defend, and the test is
whether the policy can shoot down a human who evades. It sets --red-type
Su-25T --red-player, starts red at 6000 m and 200 m/s (the Su-25T is slow),
and writes a briefing for the DCS mission screen.

--stick-test --target adds one unarmed AI F-16C 60 km ahead (same heading, 240 m/s) to lock and shoot at
with stick_test.py lock / fire.

--stick-test: only you, in the F-16C Viper module, in the air over the sea at
7000 m and 280 m/s heading east, with no other aircraft and no bridge, for
dcs/stick/stick_test.py (Export.lua stick control).

There is no player aircraft by default: DCS opens the mission in the map
view; F2 / F10 switch between external views of the aircraft.

pydcs prints "Couldn't detect any installed DCS World version" on a computer
where it cannot find DCS (or on Linux); the mission is still written.
"""

import argparse
import logging
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _skip_livery_scan():
    """
    pydcs reads the paint schemes of every aircraft in the DCS installation as
    soon as it is imported. On Python 3.13 that scan fails (KeyError:
    'country_list': pydcs relies on exec() writing into locals(), which 3.13
    no longer allows). The mission needs no paint schemes (DCS uses each
    type's default), so the scan is switched off: as soon as pydcs's scanner
    module loads, its initialize() becomes a no-op.
    """
    import importlib.abc
    import importlib.machinery
    import sys

    class SkipScan(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "dcs.liveries_scanner":
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            load = spec.loader.exec_module

            def exec_module(module):
                load(module)
                module.Liveries.initialize = staticmethod(lambda install="", saved_games="": None)
            spec.loader.exec_module = exec_module
            return spec

    if "dcs.liveries_scanner" not in sys.modules:
        sys.meta_path.insert(0, SkipScan())


def _no_payload_files():
    """
    Stop pydcs reading the payload files in the DCS installation. It fails on
    files it can't parse (KeyError on ...\\UnitPayloads\\F-100D.lua with
    current DCS), and the mission needs none of them: every weapon is loaded
    explicitly (missiles() below).
    """
    from pathlib import Path
    from dcs.payloads import PayloadDirectories
    from dcs.unittype import FlyingType
    PayloadDirectories.preferred = PayloadDirectories.fallback = None
    PayloadDirectories.dcs, PayloadDirectories.mod = [], []
    PayloadDirectories.user = Path(HERE) / "no_payload_files"
    FlyingType._payload_cache = None


# Medium-range air-to-air missiles, most wanted first: the first one a type can
# carry is the one it gets. Single-missile pylon stations only (no racks).
MISSILE_PREFS = ("AIM_120C", "AIM_120B", "R_77", "SD_10", "R_27ER", "AIM_7M", "AIM_7")
# For a type with none of those (the Su-25T): short-range heat-seekers.
SHORT_RANGE_PREFS = ("R_73", "AIM_9M", "AIM_9", "R_60M")


def missiles(typ, n=4):
    """[(pylon number, weapon)] for up to n of the best missile in MISSILE_PREFS.
    The F-16C gets AIM-120Cs on stations 1, 2, 8 and 9, as before; other types
    the outermost stations, alternating sides."""
    def scan(prefs):
        found = {}
        for i in range(1, 25):
            pyl = getattr(typ, f"Pylon{i}", None)
            if pyl is None:
                continue
            names = [k for k in dir(pyl) if not k.startswith("_")]
            for rank, pref in enumerate(prefs):
                hit = sorted(k for k in names if k.startswith(pref))
                if hit:
                    found[i] = (rank, getattr(pyl, hit[0]))
                    break
        return found

    stations = scan(MISSILE_PREFS) or scan(SHORT_RANGE_PREFS)
    if not stations:
        raise SystemExit(f"{typ.id}: no station takes an air-to-air missile")
    best = min(r for r, _ in stations.values())
    usable = sorted(i for i, (r, _) in stations.items() if r == best)
    if typ.id == "F-16C_50" and {1, 2, 8, 9} <= set(usable):
        order = [1, 2, 8, 9]
    else:
        lo, hi, order = 0, len(usable) - 1, []
        while lo <= hi:                          # outermost first, left and right in turn
            order.append(usable[lo]); lo += 1
            if lo <= hi:
                order.append(usable[hi]); hi -= 1
    return [(i, stations[i][1]) for i in sorted(order[:n])]


TARGET_BRIEFING = (
    "\n\nAn unarmed AI F-16C (TGT-1) flies the same way 60 km ahead of you at 240 m/s, for dcs/stick/stick_test.py "
    "radar, lock and fire. It carries no weapons, holds fire, ignores threats and flies straight on. "
    "You close on it at about 40 m/s: watch the range and turn away well before 5 km.\n\n"
    "stick_test.py aims the radar cursor by the cursor's bearing and range on the radar page, from your own aircraft: "
    "it turns BULLSEYE off in the DED (LIST, 0, 8, M-SEL) if the page reads from the bullseye.")

STICK_BRIEFING = (
    "You fly a Viper, in the air at 7000 m over the sea, heading east at 280 m/s. Nobody else is here.\n\n"
    "For dcs/stick/stick_test.py: trim for level flight (Mach 0.85-0.9, about 300 m/s), take your hands "
    "OFF the stick and throttle, then run the test on this computer. Keep the sim un-paused and time "
    "acceleration off. If anything looks wrong, move the stick: that takes the control back at once.")

DEFENDER_BRIEFING = (
    "You fly RED-1, a Su-25T. Your only air-to-air weapons are two short-range R-73s: you "
    "can't shoot back beyond visual range. BLUE-1, an F-16C with four AIM-120s, is flown by "
    "a trained BVR_RL policy (dcs_live.py on this computer). Your job: survive.\n\n"
    "BLUE-1 starts 90 km away, flying towards you. Your radar warning receiver shows its "
    "radar. It may warn of a missile only late, when the missile's own radar switches on.\n\n"
    "To defeat a missile: turn so it is at your 3 or 9 o'clock (beam it), release chaff, "
    "and descend; or run away early so it runs out of energy. Countermeasures: see "
    "Options > Controls, 'Countermeasures'.\n\n"
    "The fight ends when you are hit, or when dcs_live.py's time limit (300 s by default) "
    "runs out: that is a win for you. Flying more than 150 km from BLUE-1 also ends it.")


def build(args):
    logging.disable(logging.WARNING)           # pydcs's DCS-install search is noisy
    _skip_livery_scan()
    try:
        from dcs.mission import Mission
        from dcs.terrain import Caucasus
        from dcs import planes, task, countries, mapping
        from dcs.unit import Skill
        from dcs.triggers import TriggerStart
        from dcs.action import DoScriptFile
    except ImportError as e:
        raise SystemExit("pydcs is not installed: pip install pydcs") from e
    logging.disable(logging.NOTSET)
    _no_payload_files()

    types = {"F-16C": planes.F_16C_50, "F-15C": planes.F_15C, "Su-27": planes.Su_27,
             "MiG-29S": planes.MiG_29S, "FA-18C": planes.FA_18C_hornet, "Su-25T": planes.Su_25T}
    m = Mission(Caucasus())
    if args.stick_test:
        m.set_sortie_text("BVR_RL: stick control test (stick_test.py)")
        m.set_description_text(STICK_BRIEFING + (TARGET_BRIEFING if args.target else ""))
    elif args.defender:
        m.set_sortie_text("BVR_RL: survive the policy")
        m.set_description_text(DEFENDER_BRIEFING)
    elif getattr(args, "blue_player", False):
        m.set_sortie_text("BVR_RL: a trained policy flies your Viper (dcs_live.py --stick)")
        m.set_description_text(
            "You sit in BLUE-1, the F-16C; RED-1 is the DCS AI and will attack. Start dcs_live.py --stick on this "
            "computer (Export.lua must have bvr_stick_export.lua: dcs/stick/install_export.py), start the mission, "
            "trim for level flight and take your hands OFF the stick and throttle: the policy flies through "
            "Export.lua, puts the radar in TWS and fires through the cockpit. Ctrl+C in its window gives you the "
            "aircraft back.")
    else:
        m.set_sortie_text("BVR_RL: a trained policy flies BLUE-1 (dcs_live.py)")
        m.set_description_text(
            "BLUE-1 is flown by a BVR_RL policy through bvr_bridge.lua and dcs_live.py. "
            "RED-1 is its opponent. Start dcs_live.py on this computer, then start the mission.")
    # The combined task forces may fly any aircraft type.
    blue_c = m.country(countries.CombinedJointTaskForcesBlue.name)
    if blue_c is None:
        blue_c = countries.CombinedJointTaskForcesBlue()
        m.coalition["blue"].add_country(blue_c)
    red_c = m.country(countries.CombinedJointTaskForcesRed.name)
    if red_c is None:
        red_c = countries.CombinedJointTaskForcesRed()
        m.coalition["red"].add_country(red_c)

    # Over open sea west of Sukhumi, ~42.3 N 39.1 E: no terrain in the fight.
    cx, cy = -300_000.0, 420_000.0
    half = args.range_km * 500.0
    brg = math.radians(args.bearing)            # blue -> red, from map north
    dx, dy = half * math.cos(brg), half * math.sin(brg)
    p_blue = mapping.Point(cx - dx, cy - dy, m.terrain)
    p_red = mapping.Point(cx + dx, cy + dy, m.terrain)
    far = 4.0                                   # waypoints well past the other aircraft
    p_blue_wp = mapping.Point(cx + far * dx, cy + far * dy, m.terrain)
    p_red_wp = mapping.Point(cx - far * dx, cy - far * dy, m.terrain)
    def flight(country, name, typ, pos, wp, alt, heading, speed, armed=True, passive=False):
        kmh = speed * 3.6
        # a passive aircraft flies its route and ignores everything: no CAP (a CAP flight turns to intercept
        # whatever it sees, even unarmed, and the first radar test had the target closing head-on at 500 m/s)
        fg = m.flight_group_inflight(country, name, typ, pos, altitude=alt, speed=kmh,
                                     maintask=task.Nothing if passive else task.CAP)
        fg.add_waypoint(wp, alt, kmh)
        if passive:
            fg.points[0].tasks.append(task.OptROE(task.OptROE.Values.WeaponHold))
            fg.points[0].tasks.append(task.OptReactOnThreat(task.OptReactOnThreat.Values.NoReaction))
        u = fg.units[0]
        u.name = f"{name}-1"
        u.heading = heading
        u.skill = Skill.Excellent
        if armed:
            for pylon, weapon in missiles(typ, 4):
                fg.load_pylon(weapon, pylon)
        return fg

    def bullseye(p):
        # Far from the fight on purpose: stick_test.py's aimed lock reads the cursor's bearing and range from our own
        # aircraft (BULLSEYE off in the DED) and tells the two readouts apart by range; a bullseye this far away cannot
        # look like one from our own aircraft
        for side in ("blue", "red"):
            m.coalition[side].set_bullseye({"x": p.x, "y": p.y})

    hdg_blue = math.degrees(brg) % 360
    if args.stick_test:
        bullseye(mapping.Point(cx + 450_000.0, cy, m.terrain))
        me = flight(blue_c, "VIPER", types["F-16C"], mapping.Point(cx - 100_000.0, cy, m.terrain),
                    mapping.Point(cx + 400_000.0, cy, m.terrain), args.alt_blue, 90.0, args.speed)
        me.units[0].skill = Skill.Player
        if args.target:                         # an unarmed AI aircraft ahead, flying the same way
            tgt = flight(red_c, "TGT", types["F-16C"], mapping.Point(cx - 40_000.0, cy, m.terrain),
                         mapping.Point(cx + 400_000.0, cy, m.terrain), args.alt_blue, 90.0, 240.0, armed=False,
                         passive=True)
            tgt.units[0].skill = Skill.Excellent
        m.save(args.out)                        # no bridge: the Export.lua script does the work
        return me, None
    # --seed: each aircraft may start pointed off the other (head-on, offset, beam, running); its first waypoint is
    # far along that heading
    off_b, off_r = getattr(args, "blue_off", 0.0), getattr(args, "red_off", 0.0)
    h_b, h_r = (hdg_blue + off_b) % 360, (hdg_blue + 180 + off_r) % 360

    def ahead(p, hdg):
        h = math.radians(hdg)
        return mapping.Point(p.x + 4.0 * half * 2 * math.cos(h), p.y + 4.0 * half * 2 * math.sin(h), m.terrain)
    if off_b:
        p_blue_wp = ahead(p_blue, h_b)
    if off_r:
        p_red_wp = ahead(p_red, h_r)
    blue = flight(blue_c, "BLUE", types[args.blue_type], p_blue, p_blue_wp, args.alt_blue,
                  h_b, args.speed)
    red = flight(red_c, "RED", types[args.red_type], p_red, p_red_wp, args.alt_red,
                 h_r, args.speed_red)
    if args.red_player:
        red.units[0].skill = Skill.Player
    if getattr(args, "blue_player", False):
        # dcs_live.py --stick: you sit in BLUE-1 and the policy flies it through Export.lua; the bridge only
        # takes control of the agent on a CMD line, which --stick never sends
        blue.units[0].skill = Skill.Player
    bullseye(mapping.Point(cx + 450_000.0, cy, m.terrain))

    ts = TriggerStart(comment="BVR_RL bridge")
    ts.add_action(DoScriptFile(m.map_resource.add_resource_file(os.path.join(HERE, "bvr_bridge.lua"))))
    m.triggerrules.triggers.append(ts)

    m.save(args.out)
    return blue, red


# Training's start draw (bvr_env.BvrEnv._random_ic, F-16 platforms), for --seed. Kept here, not imported, so the
# mission builder needs only pydcs. The aircraft that starts nose-on is "first"; mirrored, that is red.
SCENARIOS = ("head_on", "offset_left", "offset_right", "beam", "stern_conversion")
MIRROR_START_FRAC = 0.5
WIDE_ALT_FRAC = 0.5


def draw_start(seed):
    """A start drawn as training draws them: range, altitudes, speeds, which side is nose-on and how the other is
    pointed. Returns the values for build() and a description."""
    import random
    rng = random.Random(seed)
    sc = rng.choice(SCENARIOS)
    rm = rng.uniform(70.0, 110.0)
    a1 = rng.uniform(6000.0, 11000.0)
    a2 = min(max(a1 + rng.uniform(-2500.0, 2500.0), 4000.0), 12500.0)
    wide = rng.uniform(2000.0, 12500.0)
    if rng.random() < WIDE_ALT_FRAC:
        a2 = wide
    s1, s2 = rng.uniform(250.0, 320.0), rng.uniform(250.0, 320.0)
    # the second aircraft's heading off pointing straight at the first, deg (training: a2p - (br + pi))
    if sc == "head_on":
        off = math.degrees(rng.uniform(-0.15, 0.15))
    elif sc == "offset_left":
        off = math.degrees(rng.uniform(0.25, 0.70))
    elif sc == "offset_right":
        off = -math.degrees(rng.uniform(0.25, 0.70))
    elif sc == "beam":
        off = -90.0 + math.degrees(rng.uniform(-0.30, 0.30))
    else:
        off = 180.0 + math.degrees(rng.uniform(-0.30, 0.30))
        rm = rng.uniform(45.0, 75.0)
    mirrored = rng.random() < MIRROR_START_FRAC
    bearing = rng.uniform(0.0, 360.0)
    if mirrored:     # red starts nose-on, blue gets the scenario's offset
        v = dict(alt_blue=a2, alt_red=a1, speed=s2, speed_red=s1, blue_off=off, red_off=0.0)
    else:
        v = dict(alt_blue=a1, alt_red=a2, speed=s1, speed_red=s2, blue_off=0.0, red_off=off)
    v.update(range_km=rm, bearing=bearing)
    desc = (f"seed {seed}: {sc}{' (mirrored: red nose-on, blue offset)' if mirrored else ''}, {rm:.0f} km, "
            f"blue {v['alt_blue']:.0f} m {v['speed']:.0f} m/s heading {v['blue_off']:+.0f} deg off red, "
            f"red {v['alt_red']:.0f} m {v['speed_red']:.0f} m/s heading {v['red_off']:+.0f} deg off blue")
    return v, desc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "bvr_rl_1v1.miz"))
    ap.add_argument("--range-km", type=float, default=90.0, help="start separation (default 90)")
    ap.add_argument("--bearing", type=float, default=90.0,
                    help="direction from blue to red, deg from map north (default 90: blue flies east)")
    ap.add_argument("--alt-blue", type=float, default=9000.0, help="m (default 9000)")
    ap.add_argument("--alt-red", type=float, help="m (default 9500; 6000 for the Su-25T)")
    ap.add_argument("--speed", type=float, default=280.0, help="start speed, m/s (default 280)")
    ap.add_argument("--speed-red", type=float,
                    help="red's start speed, m/s (default: --speed; 200 for the Su-25T)")
    ap.add_argument("--seed", type=int, default=None,
                    help="draw the start as training does (range, altitudes, speeds, head-on / offset / beam / stern, "
                         "half mirrored) from this seed; overrides --range-km, --bearing, --alt-*, --speed*. Each "
                         "seed is a different fight: dcs_live.py from one mission plays the same fight every time")
    ap.add_argument("--blue-type", default="F-16C", choices=["F-16C", "F-15C", "FA-18C"],
                    help="DCS aircraft for the policy (default F-16C, the training platform)")
    ap.add_argument("--red-type", default="F-16C", choices=["F-16C", "F-15C", "Su-27", "MiG-29S", "FA-18C", "Su-25T"],
                    help="DCS aircraft for red (default F-16C, as in F-16C v F-16C training)")
    ap.add_argument("--red-player", action="store_true", help="fly RED-1 yourself")
    ap.add_argument("--blue-player", action="store_true",
                    help="BLUE-1 is the player's aircraft, for dcs_live.py --stick (the policy flies it through "
                         "Export.lua; use --blue-type F-16C)")
    ap.add_argument("--stick-test", action="store_true",
                    help="only you, in the F-16C Viper, in the air (7000 m, 280 m/s); writes bvr_rl_sticktest.miz")
    ap.add_argument("--target", action="store_true",
                    help="with --stick-test: add one unarmed AI F-16C 60 km ahead to lock and shoot at")
    ap.add_argument("--defender", action="store_true",
                    help="you fly RED-1 in the free Su-25T and defend (sets --red-type Su-25T "
                         "--red-player; writes bvr_rl_defender.miz)")
    args = ap.parse_args()
    args.blue_off = args.red_off = 0.0
    start_desc = None
    if args.seed is not None:
        if args.stick_test or args.defender:
            raise SystemExit("--seed draws a 1v1 start: not with --stick-test or --defender")
        v, start_desc = draw_start(args.seed)
        for k, x in v.items():
            setattr(args, k, x)
        if args.out == ap.get_default("out") and not args.blue_player:
            args.out = os.path.join(HERE, f"bvr_rl_1v1_s{args.seed}.miz")
    if args.defender:
        args.red_type, args.red_player = "Su-25T", True
        if args.out == ap.get_default("out"):
            args.out = os.path.join(HERE, "bvr_rl_defender.miz")
    if args.blue_player:
        if args.blue_type != "F-16C":
            raise SystemExit("--blue-player: the stick link knows the F-16C only (--blue-type F-16C)")
        if args.out == ap.get_default("out"):
            args.out = os.path.join(HERE, "bvr_rl_1v1_stick.miz" if args.seed is None
                                    else f"bvr_rl_1v1_stick_s{args.seed}.miz")
    if args.stick_test:
        if args.out == ap.get_default("out"):
            args.out = os.path.join(HERE, "bvr_rl_sticktest_target.miz" if args.target else "bvr_rl_sticktest.miz")
        if args.alt_blue == ap.get_default("alt_blue"):
            args.alt_blue = 7000.0
        args.alt_red = args.speed_red = 0.0
    slow = args.red_type == "Su-25T"
    if args.alt_red is None:
        args.alt_red = 6000.0 if slow else 9500.0
    if args.speed_red is None:
        args.speed_red = 200.0 if slow else args.speed
    build(args)
    if args.stick_test:
        print(f"written {args.out}: you in the F-16C at {args.alt_blue:.0f} m, {args.speed:.0f} m/s, "
              f"{'with an unarmed target 60 km ahead' if args.target else 'alone'} over the sea")
        print("copy it to Saved Games\\DCS\\Missions\\ (or open it from anywhere in DCS)")
        return
    if start_desc:
        print(start_desc)
    print(f"written {args.out}: BLUE-1 ({args.blue_type}, policy) v RED-1 ({args.red_type}"
          f"{', you' if args.red_player else ', DCS AI'}), {args.range_km:.0f} km apart")
    print("copy it to Saved Games\\DCS\\Missions\\ (or open it from anywhere in DCS)")


if __name__ == "__main__":
    main()
