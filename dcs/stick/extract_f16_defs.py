"""
extract_f16_defs.py  --  the F-16C's cockpit command numbers, as a small JSON the tools can read
===============================================================================================

The Viper's switches, buttons and HOTAS controls are driven by (device, command, value) triples.
DCS defines them in the module's own files; running those files here gives the numbers, so
nothing in stick_test.py is a guess. The result, f16c_defs.json, holds only numbers and names
(the device ids, the command numbers of the groups the tools use, and the value a key binding
sends on press and on release).

    python dcs/stick/extract_f16_defs.py --dir "C:\\Program Files\\Eagle Dynamics\\DCS World\\Mods\\aircraft\\F-16C\\Cockpit\\Scripts" ^
        --keyboard "C:\\Program Files\\Eagle Dynamics\\DCS World\\Mods\\aircraft\\F-16C\\Input\\F-16C\\keyboard\\default.lua"

needs `pip install lupa`. The JSON is in the repository already; run this again after a DCS
update that changes the module (the command numbers are the ones DCS itself uses).
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {"hotas": "hotas_commands", "mmc": "mmc_commands", "ufc": "ufc_commands", "sms": "sms_commands",
          "fcr": "fcr_commands", "control": "control_commands", "cmds": "cmds_commands", "rwr": "rwr_commands"}
# group name -> the device it belongs to (devices.lua names)
GROUP_DEVICE = {"hotas": "HOTAS", "mmc": "MMC", "ufc": "UFC", "sms": "SMS", "fcr": "FCR",
                "control": "CONTROL_INTERFACE", "cmds": "CMDS", "rwr": "RWR"}


def lua():
    try:
        from lupa import lua51
    except ImportError:
        raise SystemExit("pip install lupa")
    return lua51.LuaRuntime(unpack_returned_tuples=True)


def table(t):
    return {str(k): v for k, v in t.items()}


def extract(scripts_dir, keyboard=None):
    L = lua()
    with open(os.path.join(scripts_dir, "command_defs.lua"), encoding="utf-8", errors="replace") as fh:
        L.execute(fh.read())
    with open(os.path.join(scripts_dir, "devices.lua"), encoding="utf-8", errors="replace") as fh:
        L.execute(fh.read())
    g = L.globals()
    out = {"devices": {k: int(v) for k, v in table(g.devices).items()}, "commands": {}, "keys": {}}
    for short, name in GROUPS.items():
        t = g[name]
        if t is not None:
            out["commands"][short] = {k: int(v) for k, v in table(t).items()}
    mfd = g.mfd_commands
    if mfd is not None:
        out["commands"]["mfd"] = {k: int(v) for k, v in table(mfd).items() if k.startswith("OSB_")}
    if keyboard and os.path.isfile(keyboard):
        # the values a key binding sends: press (value_down) and release (value_up)
        rx = re.compile(r"down\s*=\s*(\w+)\.(\w+)(?:.*?up\s*=\s*\w+\.\w+)?.*?cockpit_device_id\s*=\s*devices\.(\w+)"
                        r".*?value_down\s*=\s*(-?[\d.]+)(?:\s*,\s*value_up\s*=\s*(-?[\d.]+))?.*?name\s*=\s*_\('([^']*)'\)")
        with open(keyboard, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = rx.search(line)
                if not m:
                    continue
                grp, cmd, dev, down, up, label = m.groups()
                out["keys"].setdefault(f"{dev}.{cmd}", []).append(
                    {"down": float(down), "up": float(up) if up is not None else None, "name": label})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, help="the module's Cockpit\\Scripts folder (command_defs.lua, devices.lua)")
    ap.add_argument("--keyboard", help="Input\\F-16C\\keyboard\\default.lua of the module")
    ap.add_argument("--out", default=os.path.join(HERE, "f16c_defs.json"))
    a = ap.parse_args(argv)
    d = extract(a.dir, a.keyboard)
    with open(a.out, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    n = sum(len(v) for v in d["commands"].values())
    print(f"{a.out}: {len(d['devices'])} devices, {n} commands in {len(d['commands'])} groups, {len(d['keys'])} key values")


if __name__ == "__main__":
    main()
