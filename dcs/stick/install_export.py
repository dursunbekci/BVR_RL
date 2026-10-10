"""
install_export.py  --  hook bvr_stick_export.lua into DCS's Export.lua
=====================================================================

DCS runs Saved Games\\DCS\\Scripts\\Export.lua when a mission starts. This copies
bvr_stick_export.lua to Saved Games\\DCS\\Scripts\\bvr_rl\\ and adds one guarded
block to Export.lua that loads it. Nothing in the DCS install folder changes,
and no administrator rights are needed.

    python dcs/stick/install_export.py                 # find Saved Games\\DCS, back up, install
    python dcs/stick/install_export.py --check         # say what is installed, change nothing
    python dcs/stick/install_export.py --undo          # remove the block and the copied file
    python dcs/stick/install_export.py --user-dir "D:\\Saved Games\\DCS.openbeta"

Run it again after editing bvr_stick_export.lua: the copy in Saved Games is
what DCS loads. Restart the mission (not DCS) to load a new copy.

Remember: while it is installed, any mission you start makes DCS send your
aircraft's data to UDP port 15401 on this computer and listen on 15402 for
stick commands -- from this computer only (127.0.0.1). It does nothing to
the aircraft until stick_test.py (or dcs_live.py) sends an AXES line.
Some multiplayer servers do not allow Export scripts; undo it for those.
"""

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "bvr_stick_export.lua")
BEGIN = "-- BEGIN bvr_rl_stick (dcs/stick/install_export.py; remove with: python dcs/stick/install_export.py --undo)"
END = "-- END bvr_rl_stick"
BLOCK = BEGIN + r"""
-- BVR_STICK_CFG = {source = "event", rate = 50}   -- uncomment to take telemetry from the activity event
do
  local ok, err = pcall(dofile, lfs.writedir() .. [[Scripts\bvr_rl\bvr_stick_export.lua]])
  if not ok then log.write("bvr_stick", log.ERROR, tostring(err)) end
end
""" + END + "\n"
BACKUP_SUFFIX = ".bvr_rl_backup"


def candidates():
    home = os.path.expanduser("~")
    roots = [os.path.join(home, "Saved Games")]
    if sys.platform == "win32":
        prof = os.environ.get("USERPROFILE")
        if prof:
            roots.append(os.path.join(prof, "Saved Games"))
    out, seen = [], set()
    for r in roots:
        for name in ("DCS", "DCS.openbeta", "DCS.openalpha"):
            p = os.path.join(r, name)
            if p.lower() not in seen:
                seen.add(p.lower())
                out.append(p)
    return out


def find_user_dirs(path=None):
    if path:
        return [path] if os.path.isdir(path) else []
    return [p for p in candidates() if os.path.isdir(p)]


def _nl(text):
    return "\r\n" if "\r\n" in text else "\n"


def patch_text(text):
    """Export.lua with the block at the end (so it chains whatever the file defined above it)."""
    if BEGIN in text:
        return text
    nl = _nl(text)
    sep = "" if not text or text.endswith("\n") else nl
    gap = nl if text.strip() else ""
    return text + sep + gap + BLOCK.replace("\n", nl)


def unpatch_text(text):
    if BEGIN not in text:
        return text
    a = text.index(BEGIN)
    b = text.index(END, a) + len(END)
    nl = _nl(text)
    rest = text[b:]
    if rest.startswith(nl):
        rest = rest[len(nl):]
    head = text[:a]
    if head.endswith(nl + nl):                      # the blank line patch_text put before the block
        head = head[:-len(nl)]
    return head + rest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-dir", help="the DCS folder under Saved Games (holds Config\\ and Scripts\\)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="report only")
    g.add_argument("--undo", action="store_true", help="remove the block and the copied file")
    args = ap.parse_args(argv)

    dirs = find_user_dirs(args.user_dir)
    if not dirs:
        where = args.user_dir or "any of:\n  " + "\n  ".join(candidates())
        raise SystemExit(f"no DCS folder under Saved Games found in {where}\nGive it with --user-dir.")
    if len(dirs) > 1 and not args.check:
        raise SystemExit("more than one DCS folder found:\n  " + "\n  ".join(dirs) +
                         "\nPick one with --user-dir (the one you start DCS from).")
    for user in dirs:
        scripts = os.path.join(user, "Scripts")
        export = os.path.join(scripts, "Export.lua")
        dest_dir = os.path.join(scripts, "bvr_rl")
        dest = os.path.join(dest_dir, "bvr_stick_export.lua")
        text = ""
        if os.path.isfile(export):
            with open(export, encoding="utf-8", errors="replace", newline="") as fh:
                text = fh.read()
        hooked = BEGIN in text
        copied = os.path.isfile(dest)
        same = copied and open(dest, "rb").read() == open(SRC, "rb").read()
        print(f"DCS user folder: {user}")
        print(f"  Export.lua: {'exists' if os.path.isfile(export) else 'missing'}, "
              f"bvr_rl block {'present' if hooked else 'absent'}")
        print(f"  {dest}: {'up to date' if same else 'differs from the repository copy' if copied else 'absent'}")
        if args.check:
            continue
        if args.undo:
            if hooked:
                with open(export, "w", encoding="utf-8", newline="") as fh:
                    fh.write(unpatch_text(text))
            if copied:
                os.remove(dest)
                try:
                    os.rmdir(dest_dir)
                except OSError:
                    pass
            print("  removed" if hooked or copied else "  nothing to undo")
            continue
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(SRC, dest)
        if not hooked:
            if os.path.isfile(export) and not os.path.exists(export + BACKUP_SUFFIX):
                shutil.copy2(export, export + BACKUP_SUFFIX)
                print(f"  backup: {export + BACKUP_SUFFIX}")
            with open(export, "w", encoding="utf-8", newline="") as fh:
                fh.write(patch_text(text))
        print("  installed. Start (or restart) the mission; the log line 'bvr_stick: started' in "
              "Saved Games\\DCS\\Logs\\dcs.log shows it loaded.")


if __name__ == "__main__":
    main()
