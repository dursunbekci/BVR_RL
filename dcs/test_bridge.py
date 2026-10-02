"""
test_bridge.py  —  run bvr_bridge.lua and the setup patch in Lua 5.1, without DCS
==================================================================================

    pip install lupa
    python dcs/test_bridge.py

1. bvr_bridge.lua runs against mock/dcs_api_mock.lua, a small stand-in for
   the DCS mission scripting API (a blue aircraft that follows its route
   task and launches when given an attack task), with LuaSocket replaced by
   in-memory queues. Checks: nothing changes before the first command; the
   first command takes control and routes the aircraft along the commanded
   heading; a fire command launches through an attack task and hands control
   back; a launch that never comes times out; STOP gives the aircraft back;
   red is ordered to attack; everything sent parses as a recording.
2. The block setup_dcs.py adds to MissionScripting.lua runs in a copy of the
   file DCS ships: bvr_rl_socket is set, and require, package, io, os and lfs
   are still removed.

DCS's real API may differ from the mock in ways it cannot show; the bridge
reports any error in Saved Games\\DCS\\Logs\\dcs.log as "bvr_bridge ... error".
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

MISSION_SCRIPTING = (
    "--Initialization script for the Mission lua Environment (SSE)\r\n\r\n"
    "dofile('Scripts/ScriptingSystem.lua')\r\n\r\n"
    "--Sanitize Mission Scripting environment\r\n\r\n"
    "local function sanitizeModule(name)\r\n\t_G[name] = nil\r\n\tpackage.loaded[name] = nil\r\nend\r\n\r\n"
    "do\r\n\tsanitizeModule('os')\r\n\tsanitizeModule('io')\r\n\tsanitizeModule('lfs')\r\n"
    "\t_G['require'] = nil\r\n\t_G['loadlib'] = nil\r\n\t_G['package'] = nil\r\nend\r\n")


def lua():
    try:
        from lupa import lua51
    except ImportError:
        return None
    return lua51.LuaRuntime(unpack_returned_tuples=True)


def test_bridge():
    from dcs_world import DcsRecording
    L = lua()
    sent, inbox, state = [], [], {}

    class Udp:
        def settimeout(self, t): pass
        def setsockname(self, h, p): return 1
        def sendto(self, data, h, p): sent.append((p, data)); return len(data)
        def receive(self, *a): return inbox.pop(0) if inbox else None

    class Sock:
        def udp(self): return Udp()

    plan = {5.0: ["CMD 1 90 10000 300 0"], 6.0: ["CMD 2 90 10000 300 0"],
            20.0: ["CMD 3 90 10000 300 1"], 21.0: ["CMD 3 90 10000 300 1"],
            40.0: ["CMD 9 90 10000 300 1"], 60.0: ["STOP"]}

    def on_tick(T):
        for t in list(plan):
            if T >= t - 1e-9:
                inbox.extend(plan.pop(t))
        if 39.9 < T < 40.1 and "ammo0" not in state:      # the second shot can't launch
            L.eval("function() blue.ammo = 0 end")()
            state["ammo0"] = 1
        if 19.9 < T < 20.0:
            state["hdg20"] = L.eval("blue.hdg")
        if 4.9 < T < 5.0:                                  # before the first command
            log = L.globals().LOG
            state["blue_calls_before"] = sum(1 for i in range(1, len(log) + 1)
                                             if log[i][1] == "Viper-1")

    with open(os.path.join(HERE, "mock", "dcs_api_mock.lua")) as fh:
        L.execute(fh.read())
    L.globals().bvr_rl_socket = Sock()
    L.globals().RUN(os.path.join(HERE, "bvr_bridge.lua"), 130.0, on_tick)

    log = L.globals().LOG
    LOG = [tuple(log[i].values()) for i in range(1, len(log) + 1)]
    errors = [x for x in LOG if x[0] == "envinfo"]
    assert not errors, errors

    def calls(who, what=None):
        return [x for x in LOG if x[0] == who and (what is None or x[1] == what)]

    lines = [json.loads(d) for p, d in sent if p == 15301]
    ra = calls("Bandit-1", "pushTask")
    assert ra and ra[0][2].id == "AttackUnit" and ra[0][2].params.unitId == L.eval("blue.uid")
    ctrl = [d["t"] for d in lines if d.get("ev") == "bridge" and d.get("status") == "control"]
    assert ctrl and ctrl[0] >= 5.0, ctrl
    assert state["blue_calls_before"] == 0, "the agent was changed before any command"
    opts = {x[2]: x[3] for x in calls("Viper-1", "opt")[:5]}
    assert opts[0] == 4 and opts[1] == 0 and opts[3] == 3 and opts[18] == 0, opts
    route = calls("Viper-1", "setTask")[0][2].params.route.points
    assert abs(route[2].x - (-300000 + 5 * 250)) < 2000 and route[2].y > 650000, "not routed east"
    assert route[2].alt == 10000 and route[2].speed == 300
    assert abs(math.degrees(state["hdg20"]) - 90) < 3, math.degrees(state["hdg20"])
    fires = [d["status"] for d in lines if d.get("ev") == "fire"]
    assert fires == ["requested", "launched", "requested", "timeout"], fires
    att = [x for x in calls("Viper-1", "pushTask") if x[2].id == "AttackUnit"]
    assert len(att) == 2 and att[0][2].params.weaponType == 134217728
    assert calls("Viper-1", "resetTask"), "STOP did not give the aircraft back"
    rec = DcsRecording([json.dumps(d) for d in lines])
    assert rec.header.get("agent") == "Viper-1" and rec.header.get("red") == "Bandit-1"
    assert "Bandit-1" in rec.dead and len(rec.weapons) == 1
    print(f"  bridge in Lua 5.1 ........... OK  ({len(lines)} lines sent, fire {fires})")


def test_setup_patch():
    import setup_dcs
    L = lua()
    patched = setup_dcs.patch_text(MISSION_SCRIPTING)
    assert setup_dcs.unpatch_text(patched) == MISSION_SCRIPTING
    L.execute('dofile = function(p) end\n'
              'lfs = {currentdir = function() return "C:\\\\DCS" end}\n'
              'package.preload["socket"] = function() return {udp = function() end, mock = true} end')
    L.execute(patched)
    g = L.globals()
    assert g.bvr_rl_socket is not None and g.bvr_rl_socket.mock
    for name in ("require", "package", "io", "os", "lfs"):
        assert g[name] is None, f"{name} is no longer sanitized"
    print("  MissionScripting patch ...... OK  (bvr_rl_socket only; undo exact)")


if __name__ == "__main__":
    if lua() is None:
        print("lupa is not installed (pip install lupa): skipped")
        sys.exit(0)
    test_bridge()
    test_setup_patch()
    print("2/2 passed")
