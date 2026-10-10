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
    opts = {x[2]: x[3] for x in calls("Viper-1", "opt")[:6]}
    assert opts[0] == 4 and opts[1] == 0 and opts[3] == 3 and opts[18] == 0, opts
    route = calls("Viper-1", "setTask")[0][2].params.route.points
    assert abs(route[2].x - (-300000 + 5 * 250)) < 2000 and route[2].y > 650000, "not routed east"
    assert route[2].alt == 10000 and route[2].speed == 300
    assert opts[16] is False, "afterburner not allowed"
    assert [x[2] for x in calls("Viper-1", "setSpeed")][:1] == [300], "speed not ordered"
    assert abs(math.degrees(state["hdg20"]) - 90) < 3, math.degrees(state["hdg20"])
    fires = [d["status"] for d in lines if d.get("ev") == "fire"]
    assert fires == ["requested", "launched", "requested", "timeout"], fires
    att = [x for x in calls("Viper-1", "pushTask") if x[2].id == "AttackUnit"]
    assert len(att) == 2 and att[0][2].params.weaponType == 134217728
    assert calls("Viper-1", "resetTask"), "STOP did not give the aircraft back"
    rec = DcsRecording([json.dumps(d) for d in lines])
    assert rec.header.get("agent") == "Viper-1" and rec.header.get("red") == "Bandit-1"
    assert "Bandit-1" in rec.dead and len(rec.weapons) == 1
    kills = [d for d in lines if d.get("ev") == "kill"]
    assert kills and kills[0]["unit"] == "Bandit-1" and kills[0]["killer"] == "Viper-1" \
        and kills[0]["id"] == 1, kills
    how = rec.death_cause("Bandit-1")
    assert how["weapon"] is not None and how["weapon"]["shooter"] == "Viper-1", how
    print(f"  bridge in Lua 5.1 ........... OK  ({len(lines)} lines sent, fire {fires})")


def test_bridge_destroy():
    """DESTROY <id> removes a missile in flight: no hit, and the bridge reports it."""
    L = lua()
    sent, inbox = [], []

    class Udp:
        def settimeout(self, t): pass
        def setsockname(self, h, p): return 1
        def sendto(self, data, h, p): sent.append((p, data)); return len(data)
        def receive(self, *a): return inbox.pop(0) if inbox else None

    class Sock:
        def udp(self): return Udp()

    plan = {5.0: ["CMD 1 90 10000 300 0"], 20.0: ["CMD 2 90 10000 300 1"],
            40.0: ["DESTROY 1"], 41.0: ["DESTROY 1"]}

    def on_tick(T):
        for t in list(plan):
            if T >= t - 1e-9:
                inbox.extend(plan.pop(t))

    with open(os.path.join(HERE, "mock", "dcs_api_mock.lua")) as fh:
        L.execute(fh.read())
    L.globals().bvr_rl_socket = Sock()
    L.globals().RUN(os.path.join(HERE, "bvr_bridge.lua"), 130.0, on_tick)
    lines = [json.loads(d) for p, d in sent if p == 15301]
    lost = [d for d in lines if d.get("ev") == "support_lost"]
    assert [d["status"] for d in lost] == ["destroyed", "gone"] and lost[0]["id"] == 1 \
        and lost[0]["shooter"] == "Viper-1", lost
    gone = [d for d in lines if d.get("ev") == "weapon_gone" and d["id"] == 1]
    assert gone and gone[0]["t"] >= lost[0]["t"], gone
    assert not any(d.get("ev") in ("hit", "kill", "dead") for d in lines), "the removed missile hit"
    from dcs_world import DcsRecording
    rec = DcsRecording([json.dumps(d) for d in lines])
    assert rec.weapons[1]["support_lost"] is not None and rec.weapons[1]["t_hit"] is None
    print(f"  bridge DESTROY .............. OK  (removed at {lost[0]['t']:.1f} s, no hit)")


def test_bridge_eta():
    """OPT eta: the route's points get locked arrival times from the commanded
    speed (speed not locked), on the mission or the day clock; OPT eta off ends it."""
    L = lua()
    sent, inbox = [], []

    class Udp:
        def settimeout(self, t): pass
        def setsockname(self, h, p): return 1
        def sendto(self, data, h, p): sent.append((p, data)); return len(data)
        def receive(self, *a): return inbox.pop(0) if inbox else None

    class Sock:
        def udp(self): return Udp()

    plan = {5.0: ["CMD 1 90 10000 300 0"],
            10.0: ["OPT eta mission", "CMD 2 90 10000 340 0"],
            20.0: ["OPT eta abs", "CMD 3 90 10000 340 0"],
            30.0: ["OPT eta off", "CMD 4 90 10000 340 0"]}

    def on_tick(T):
        for t in list(plan):
            if T >= t - 1e-9:
                inbox.extend(plan.pop(t))

    with open(os.path.join(HERE, "mock", "dcs_api_mock.lua")) as fh:
        L.execute(fh.read())
    L.globals().bvr_rl_socket = Sock()
    L.globals().RUN(os.path.join(HERE, "bvr_bridge.lua"), 40.0, on_tick)
    log = L.globals().LOG
    LOG = [tuple(log[i].values()) for i in range(1, len(log) + 1)]
    routes = [x[2].params.route.points for x in LOG if x[0] == "Viper-1" and x[1] == "setTask"]
    import re
    near = float(re.search(r"near_m\s*=\s*([\d.]+)", open(os.path.join(HERE, "bvr_bridge.lua")).read()).group(1))
    plain, mission, absolute, off = routes[0], routes[1], routes[2], routes[-1]
    assert plain[1].speed_locked and not plain[1].ETA_locked
    assert mission[1].ETA_locked and not mission[1].speed_locked
    assert abs(mission[1].ETA - (10.0 + near / 340.0)) < 0.2, mission[1].ETA
    assert absolute[1].ETA_locked and abs(absolute[1].ETA - (43200.0 + 20.0 + near / 340.0)) < 0.2
    assert off[1].speed_locked and not off[1].ETA_locked
    lines = [json.loads(d) for p, d in sent if p == 15301]
    assert lines[0].get("bridge") == 4, lines[0]           # dcs_live.py checks the version
    acks = [d["clock"] for d in lines if d.get("ev") == "bridge" and d.get("status") == "eta"]
    assert acks == ["mission", "abs", "off"], acks
    print(f"  bridge ETA lock ............. OK  (point 1 due {mission[1].ETA:.1f} s at 340 m/s)")


def test_bridge_opts():
    """OPT near / wpt / redhold (dcs/turn_test.py): the first route point moves
    (or goes, at 0), the points' type changes, an AI red holds its fire; every
    option is confirmed, and a change re-routes at once."""
    L = lua()
    sent, inbox = [], []

    class Udp:
        def settimeout(self, t): pass
        def setsockname(self, h, p): return 1
        def sendto(self, data, h, p): sent.append((p, data)); return len(data)
        def receive(self, *a): return inbox.pop(0) if inbox else None

    class Sock:
        def udp(self): return Udp()

    plan = {5.0: ["CMD 1 90 10000 300 0"],
            10.0: ["OPT near 1000", "OPT wpt flyover", "OPT redhold 1", "CMD 2 90 10000 300 0"],
            20.0: ["OPT near 0", "CMD 3 90 10000 300 0"],
            30.0: ["OPT near 0", "OPT wpt turn", "OPT redhold 0", "CMD 4 90 10000 300 0"]}

    def on_tick(T):
        for t in list(plan):
            if T >= t - 1e-9:
                inbox.extend(plan.pop(t))

    with open(os.path.join(HERE, "mock", "dcs_api_mock.lua")) as fh:
        L.execute(fh.read())
    L.globals().bvr_rl_socket = Sock()
    L.globals().RUN(os.path.join(HERE, "bvr_bridge.lua"), 40.0, on_tick)
    log = L.globals().LOG
    LOG = [tuple(log[i].values()) for i in range(1, len(log) + 1)]
    sets = [(x[2].params.route.points) for x in LOG if x[0] == "Viper-1" and x[1] == "setTask"]
    def first_d(pts):
        return len(pts), pts[1].action
    shapes = [first_d(p) for p in sets]
    assert shapes[0] == (2, "Turning Point"), shapes
    assert (2, "Fly Over Point") in shapes and (1, "Fly Over Point") in shapes and \
        shapes[-1] == (1, "Turning Point"), shapes
    near = next(p for p in sets if len(p) == 2 and p[1].action == "Fly Over Point")
    d = math.hypot(near[1].x - near[2].x, near[1].y - near[2].y)
    assert abs(d - 59000.0) < 300.0, d                      # first point 1 km out, far 60 km
    roe = [x[3] for x in LOG if x[0] != "Viper-1" and x[1] == "opt" and x[2] == 0]
    assert roe[-2:] == [4, 1], roe                           # hold, then weapons free again
    lines = [json.loads(d) for p, d in sent if p == 15301]
    acks = [(d["near_m"], d["wpt"], d["redhold"]) for d in lines
            if d.get("ev") == "bridge" and d.get("status") == "opt"]
    assert len(acks) == 7 and acks[0][0] == 1000 and acks[-1] == (0, "turn", False), acks
    print(f"  bridge turn-test options .... OK  ({len(acks)} confirmations)")


def test_bridge_autodefend():
    """OPT autodefend: while a red missile is inbound at the agent the DCS AI
    defends it (evade), routes wait and a shot is refused; when it is gone, the
    route is back. OPT hot pushes a guns-only attack task and pops it. Neither
    aircraft returns home at bingo fuel."""
    L = lua()
    sent, inbox = [], []

    class Udp:
        def settimeout(self, t): pass
        def setsockname(self, h, p): return 1
        def sendto(self, data, h, p): sent.append((p, data)); return len(data)
        def receive(self, *a): return inbox.pop(0) if inbox else None

    class Sock:
        def udp(self): return Udp()

    plan = {5.0: ["CMD 1 90 10000 300 0"], 6.0: ["OPT autodefend 1", "CMD 2 90 10000 300 0"],
            12.0: ["CMD 3 120 10000 300 1"], 25.0: ["CMD 4 120 10000 300 0"],
            30.0: ["OPT hot 1"], 35.0: ["OPT hot 0", "CMD 5 120 10000 300 0"]}
    calls = {10.0: "MOCK_RED_SHOT", 20.0: "MOCK_RED_GONE"}

    def on_tick(T):
        for t in list(plan):
            if T >= t - 1e-9:
                inbox.extend(plan.pop(t))
        for t in list(calls):
            if T >= t - 1e-9:
                L.globals()[calls.pop(t)]()

    with open(os.path.join(HERE, "mock", "dcs_api_mock.lua")) as fh:
        L.execute(fh.read())
    L.globals().bvr_rl_socket = Sock()
    L.globals().RUN(os.path.join(HERE, "bvr_bridge.lua"), 40.0, on_tick)
    log = L.globals().LOG
    LOG = [tuple(log[i].values()) for i in range(1, len(log) + 1)]
    lines = [json.loads(d) for p, d in sent if p == 15301]
    defend = [(round(d["t"], 1), d["on"]) for d in lines if d.get("ev") == "bridge" and d.get("status") == "defend"]
    assert [on for _, on in defend] == [True, False], defend
    assert 10.0 <= defend[0][0] <= 10.3 and 20.0 <= defend[1][0] <= 20.3, defend
    react = [x[3] for x in LOG if x[0] == "Viper-1" and x[1] == "opt" and x[2] == 1]
    assert react[-2:] == [2, 0], react                    # evade fire, then no reaction again
    refused = [d for d in lines if d.get("ev") == "fire" and d.get("status") == "refused"]
    assert refused and refused[0]["reason"] == "defending", refused
    bingo = {x[0] for x in LOG if x[1] == "opt" and x[2] == 6 and x[3] is False}
    assert bingo == {"Viper-1", "Bandit-1"}, bingo        # no return home at bingo fuel
    pushes = [x[2] for x in LOG if x[0] == "Viper-1" and x[1] == "pushTask"]
    assert any(t.id == "AttackUnit" and t.params.weaponType == 805306368 for t in pushes), pushes
    assert any(x[0] == "Viper-1" and x[1] == "popTask" for x in LOG)
    print(f"  bridge auto-defend .......... OK  (defending {defend[0][0]}-{defend[1][0]} s; "
          f"hot attack task; no bingo return)")


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
    test_bridge_destroy()
    test_bridge_eta()
    test_bridge_opts()
    test_bridge_autodefend()
    test_setup_patch()
    print("6/6 passed")
