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

  BLUE-1   the aircraft the policy flies (an AI unit; without dcs_live.py it
           flies a CAP and fights on its own)
  RED-1    its opponent: DCS AI, ordered by the bridge to attack BLUE-1, or
           you, with --red-player (needs a module you can fly, e.g. the F-16C)

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

    types = {"F-16C": planes.F_16C_50, "F-15C": planes.F_15C, "Su-27": planes.Su_27,
             "MiG-29S": planes.MiG_29S, "FA-18C": planes.FA_18C_hornet}
    m = Mission(Caucasus())
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
    kmh = args.speed * 3.6

    def flight(country, name, typ, pos, wp, alt, heading):
        fg = m.flight_group_inflight(country, name, typ, pos, altitude=alt, speed=kmh,
                                     maintask=task.CAP)
        fg.add_waypoint(wp, alt, kmh)
        u = fg.units[0]
        u.name = f"{name}-1"
        u.heading = heading
        u.skill = Skill.Excellent
        aam = [n for n in dir(typ.Pylon1) if "AIM_120C" in n] if hasattr(typ, "Pylon1") else []
        if typ is planes.F_16C_50:
            for i in (1, 2, 8, 9):
                fg.load_pylon(getattr(getattr(typ, f"Pylon{i}"), aam[0]), i)
        else:
            fg.load_task_default_loadout(task.CAP)
        return fg

    hdg_blue = math.degrees(brg) % 360
    blue = flight(blue_c, "BLUE", types[args.blue_type], p_blue, p_blue_wp, args.alt_blue, hdg_blue)
    red = flight(red_c, "RED", types[args.red_type], p_red, p_red_wp, args.alt_red,
                 (hdg_blue + 180) % 360)
    if args.red_player:
        red.units[0].skill = Skill.Player

    ts = TriggerStart(comment="BVR_RL bridge")
    ts.add_action(DoScriptFile(m.map_resource.add_resource_file(os.path.join(HERE, "bvr_bridge.lua"))))
    m.triggerrules.triggers.append(ts)

    m.save(args.out)
    return blue, red


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "bvr_rl_1v1.miz"))
    ap.add_argument("--range-km", type=float, default=90.0, help="start separation (default 90)")
    ap.add_argument("--bearing", type=float, default=90.0,
                    help="direction from blue to red, deg from map north (default 90: blue flies east)")
    ap.add_argument("--alt-blue", type=float, default=9000.0, help="m (default 9000)")
    ap.add_argument("--alt-red", type=float, default=9500.0, help="m (default 9500)")
    ap.add_argument("--speed", type=float, default=280.0, help="start speed, m/s (default 280)")
    ap.add_argument("--blue-type", default="F-16C", choices=["F-16C", "F-15C", "FA-18C"],
                    help="DCS aircraft for the policy (default F-16C, the training platform)")
    ap.add_argument("--red-type", default="F-16C", choices=["F-16C", "F-15C", "Su-27", "MiG-29S", "FA-18C"],
                    help="DCS aircraft for red (default F-16C, as in F-16C v F-16C training)")
    ap.add_argument("--red-player", action="store_true", help="fly RED-1 yourself")
    args = ap.parse_args()
    build(args)
    print(f"written {args.out}: BLUE-1 ({args.blue_type}, policy) v RED-1 ({args.red_type}"
          f"{', you' if args.red_player else ', DCS AI'}), {args.range_km:.0f} km apart")
    print("copy it to Saved Games\\DCS\\Missions\\ (or open it from anywhere in DCS)")


if __name__ == "__main__":
    main()
