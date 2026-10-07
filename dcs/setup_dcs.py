"""
setup_dcs.py  —  give DCS mission scripts a network socket for bvr_bridge.lua
=============================================================================

DCS removes networking and file access from mission scripts. bvr_bridge.lua
needs one thing back: LuaSocket, to talk to dcs_live.py over UDP on this
computer. This adds a few lines to <DCS install>\Scripts\MissionScripting.lua
that load LuaSocket *before* DCS's sanitizing runs and keep it only as the
global bvr_rl_socket. io, lfs, os, require and package stay removed, as DCS
ships them.

    python dcs/setup_dcs.py                 # find DCS, back up and patch
    python dcs/setup_dcs.py --check         # say what is installed, change nothing
    python dcs/setup_dcs.py --undo          # remove the patch
    python dcs/setup_dcs.py --dcs "D:\\Games\\DCS World"

DCS is usually installed under C:\\Program Files, which Windows protects: run
the command from a terminal opened with "Run as administrator".

Remember:
  * With the patch in place, any mission you load can open network
    connections. Undo it when you are not using BVR_RL, and don't join
    multiplayer servers with it (some check this file, and refuse you).
  * A DCS update replaces MissionScripting.lua: run this again afterwards.
"""

import argparse
import os
import shutil
import sys

BEGIN = "-- BEGIN bvr_rl (dcs/setup_dcs.py; remove with: python dcs/setup_dcs.py --undo)"
END = "-- END bvr_rl"
BLOCK = BEGIN + r"""
do
  local ok, sock = pcall(function()
    package.path  = package.path  .. ";.\\LuaSocket\\?.lua;" .. lfs.currentdir() .. "\\LuaSocket\\?.lua"
    package.cpath = package.cpath .. ";.\\LuaSocket\\?.dll;" .. lfs.currentdir() .. "\\LuaSocket\\?.dll"
    return require("socket")
  end)
  if ok then bvr_rl_socket = sock end
end
""" + END + "\n"

BACKUP_SUFFIX = ".bvr_rl_backup"


def candidates():
    """Likely DCS install folders, most likely first."""
    out = []
    if sys.platform == "win32":
        try:
            import winreg
            for key in (r"Software\Eagle Dynamics\DCS World", r"Software\Eagle Dynamics\DCS World OpenBeta"):
                try:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
                        out.append(winreg.QueryValueEx(k, "Path")[0])
                except OSError:
                    pass
        except ImportError:
            pass
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files", r"D:\Program Files"):
        out += [os.path.join(base, "Eagle Dynamics", "DCS World"),
                os.path.join(base, "Eagle Dynamics", "DCS World OpenBeta")]
    for steam in (r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam", r"D:\Steam",
                  r"D:\SteamLibrary", r"E:\SteamLibrary"):
        out.append(os.path.join(steam, "steamapps", "common", "DCSWorld"))
    seen, uniq = set(), []
    for p in out:
        if p and p.lower() not in seen:
            seen.add(p.lower()); uniq.append(p)
    return uniq


def find_dcs(path=None):
    for p in ([path] if path else candidates()):
        if p and os.path.isfile(os.path.join(p, "Scripts", "MissionScripting.lua")):
            return p
    return None


def _nl(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def patch_text(text: str) -> str:
    """MissionScripting.lua with the block inserted before DCS's sanitizing,
    in the file's own line endings."""
    if BEGIN in text:
        return text
    nl = _nl(text)
    block = BLOCK.replace("\n", nl)
    lines = text.splitlines(keepends=True)
    at = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("local function sanitizeModule") or "sanitizeModule(" in s \
                or s.startswith("_G['require']") or s.startswith('_G["require"]'):
            at = i
            break
    if at is None:
        # Nothing is sanitized in this file: the end is as good as anywhere.
        return text + ("" if text.endswith("\n") else nl) + block
    # Before a "do" that opens the sanitizing block, if that is where it starts.
    if at > 0 and lines[at - 1].strip() == "do":
        at -= 1
    return "".join(lines[:at]) + block + nl + "".join(lines[at:])


def unpatch_text(text: str) -> str:
    if BEGIN not in text:
        return text
    nl = _nl(text)
    a = text.index(BEGIN)
    b = text.index(END, a) + len(END)
    rest = text[b:]
    for _ in range(2):                    # the block's own line end and the blank line after it
        if rest.startswith(nl):
            rest = rest[len(nl):]
    return text[:a] + rest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dcs", help="DCS install folder (the one holding bin\\ and Scripts\\)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="report only")
    g.add_argument("--undo", action="store_true", help="remove the patch")
    args = ap.parse_args()

    root = find_dcs(args.dcs)
    if root is None:
        where = args.dcs or "any of:\n  " + "\n  ".join(candidates())
        raise SystemExit(f"DCS World not found in {where}\nGive the install folder with --dcs.")
    ms = os.path.join(root, "Scripts", "MissionScripting.lua")
    sock = os.path.join(root, "LuaSocket", "socket.lua")
    print(f"DCS World: {root}")
    print(f"LuaSocket: {'found' if os.path.isfile(sock) else 'NOT FOUND at ' + sock}")
    with open(ms, encoding="utf-8", errors="replace", newline="") as fh:
        text = fh.read()
    patched = BEGIN in text
    print(f"MissionScripting.lua: {'patched' if patched else 'not patched'}")
    if args.check:
        return
    if args.undo:
        if not patched:
            print("nothing to undo"); return
        new = unpatch_text(text)
    else:
        if patched:
            print("already patched: nothing to do"); return
        new = patch_text(text)
        backup = ms + BACKUP_SUFFIX
        try:
            if not os.path.exists(backup):
                shutil.copy2(ms, backup)
                print(f"backup: {backup}")
        except PermissionError:
            raise SystemExit("permission denied: open the terminal with 'Run as administrator' "
                             "and run this again")
    try:
        with open(ms, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
    except PermissionError:
        raise SystemExit("permission denied: open the terminal with 'Run as administrator' "
                         "and run this again")
    print("patch removed" if args.undo else
          "patched. Restart DCS if it is running. Run this again after every DCS update.")


if __name__ == "__main__":
    main()
