"""
test_stick.py  --  test bvr_stick_export.lua, stick_test.py and install_export.py without DCS
==============================================================================================

    pip install lupa
    python dcs/stick/test_stick.py

1. bvr_stick_export.lua runs in Lua 5.1 against a mock of the Export API
   (LoGet*, LoSetCommand, LuaSocket as in-memory queues): every line it sends
   is valid JSON and parses as a stick_test.Frame; AXES lines reach
   LoSetCommand with the right numbers every frame; the watchdog zeroes the
   axes; RELEASE, PING, CMD work; callbacks Export.lua already had still run.
2. stick_test.UdpLink over real UDP on this computer.
3. stick_test.py end to end against a synthetic aircraft, with the roll and
   pitch signs both ways round: it finds the signs and gains again, and stops
   when a limit is crossed.
4. install_export.py: the block goes in and out of an Export.lua byte for
   byte, loads the right file, and installs into a folder.

The mock cannot show whether DCS's real API matches it: the first thing
to run in DCS is `stick_test.py info` and `rate`.
"""

import json
import math
import os
import socket
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def lua():
    try:
        from lupa import lua51
    except ImportError:
        return None
    return lua51.LuaRuntime(unpack_returned_tuples=True)


MOCK = r"""
CLOCK, MODEL_T = 1000.0, 50.0
INBOX, SENT, CMDS, CALLS = {}, {}, {}, {}
local mocksock = {gettime = function() return CLOCK end}
function mocksock.udp()
  local u = {}
  function u:settimeout() end
  function u:setsockname() return 1 end
  function u:sendto(data, host, port) SENT[#SENT + 1] = {port, data}; return #data end
  function u:receive() return table.remove(INBOX, 1) end
  function u:close() end
  return u
end
BVR_STICK_SOCKET = mocksock
BVR_STICK_CFG = {rate = 50, objects_every = 5, payload_every = 50, watchdog = 0.5, stats_every = 2.0}
log = {write = function() end, INFO = 1, ERROR = 2}

function LoGetModelTime() return MODEL_T end
function LoGetSelfData()
  return {Name = "F-16C_50", UnitName = "Viper-1", GroupName = "G", CoalitionID = 2, Type = {level1 = 1, level2 = 1},
          LatLongAlt = {Lat = 42.1, Long = 41.2, Alt = 7000.5}, Heading = 1.25, Pitch = 0.02, Bank = -0.3,
          Position = {x = 1000, y = 7000, z = 2000}}
end
function LoGetTrueAirSpeed() return 250.5 end
function LoGetIndicatedAirSpeed() return 200.0 end
function LoGetMachNumber() return 0.78 end
function LoGetAngleOfAttack() return 0.05 end
function LoGetAngleOfSideSlip() return 0.0 end
function LoGetAccelerationUnits() return {x = 0.0, y = 1.2, z = 0.0} end
function LoGetVerticalVelocity() return 1.5 end
function LoGetAltitudeAboveSeaLevel() return 7000.5 end
function LoGetAltitudeAboveGroundLevel() return 6800.0 end
function LoGetADIPitchBankYaw() return 0.02, -0.3, 1.25 end
function LoGetVectorVelocity() return {x = 80.0, y = 1.5, z = 237.0} end
function LoGetEngineInfo() return {RPM = {left = 88.0, right = 0}, FuelConsumption = {left = 1.2, right = 0}, fuel_internal = 2100.0} end
function LoGetWorldObjects()
  return {[10] = {Name = "Su-27", UnitName = "Bandit-1", GroupName = "R", CoalitionID = 1, Type = {level1 = 1, level2 = 1},
                  LatLongAlt = {Lat = 42.5, Long = 41.9, Alt = 9000}, Heading = 3.0, Pitch = 0, Bank = 0,
                  Position = {x = 60000, y = 9000, z = 50000}},
          [11] = {Name = "F-16C_50", UnitName = "Viper-1", GroupName = "G", CoalitionID = 2, Type = {level1 = 1, level2 = 1},
                  LatLongAlt = {Lat = 42.1, Long = 41.2, Alt = 7000.5}, Position = {x = 1000, y = 7000, z = 2000}}}
end
function LoGetPayloadInfo() return {CurrentStation = 7, Stations = {{count = 1, CLSID = "{AIM-120C}"}}} end
function LoSetCommand(c, v) CMDS[#CMDS + 1] = {c, v} end
CLICKS = {}
RD_TD, RD_TRACKED, RD_CURSOR = false, false, true
function list_indication(id)
  if id == 1 then return "HUD_glass\n" .. (RD_TD and "HUD_AA_TD_box\n" or "") end
  if id == 4 then return "FCR_Antenna_Azimuth_Tics_Root\n" .. (RD_CURSOR and "Cursor_Root\n" or "") .. (RD_TRACKED and "FCR_TrackedTgtPlaceholder\n" or "") end
  if id == 17 then return "CenterOrigin\nThreatSymbol1\n" end
  if id == 3 then return "-----------\nMODE\nCRM\n-----------" end
  if id == 5 then error("no such indicator") end
  if id == 7 then return "" end
  if id == 9 then return "-----------\nSTT\n-----------" end
  return nil
end
function GetDevice(id)
  return {performClickableAction = function(self, dev, c, v) CLICKS[#CLICKS + 1] = {dev, c, v} end}
end

-- what Export.lua already defined before our dofile
LuaExportStart = function() CALLS.start = (CALLS.start or 0) + 1 end
LuaExportStop = function() CALLS.stop = (CALLS.stop or 0) + 1 end
LuaExportBeforeNextFrame = function() CALLS.before = (CALLS.before or 0) + 1 end
LuaExportAfterNextFrame = function() CALLS.after = (CALLS.after or 0) + 1 end
LuaExportActivityNextEvent = function(t) CALLS.activity = (CALLS.activity or 0) + 1; return t + 1.0 end
"""


class Mock:
    """bvr_stick_export.lua loaded in Lua 5.1 against the mock, with helpers to run frames."""

    def __init__(self, source="frame"):
        self.L = lua()
        self.L.execute(MOCK)
        self.L.execute(f"BVR_STICK_CFG.source = '{source}'")
        with open(os.path.join(HERE, "bvr_stick_export.lua")) as fh:
            self.L.execute(fh.read())
        self.g = self.L.globals()
        self.next_event = 0.0

    def sent(self):
        s = self.g.SENT
        return [(s[i][1], s[i][2]) for i in range(1, len(s) + 1)]

    def lines(self):
        return [json.loads(d) for _, d in self.sent()]

    def cmds(self):
        c = self.g.CMDS
        return [(c[i][1], c[i][2]) for i in range(1, len(c) + 1)]

    def send(self, text):
        self.g.INBOX[len(self.g.INBOX) + 1] = text

    def run(self, seconds, hook=None, paused=False, fps=60):
        """DCS calls Before and After every rendered frame, and the activity event when it is due
        (here at frame boundaries, the worst case)."""
        g = self.g
        for i in range(int(seconds * fps)):
            g.CLOCK = float(g.CLOCK) + 1.0 / fps
            if not paused:
                g.MODEL_T = float(g.MODEL_T) + 1.0 / fps
            if hook:
                hook(i / fps)
            g.LuaExportBeforeNextFrame()
            if g.MODEL_T >= self.next_event:
                self.next_event = g.LuaExportActivityNextEvent(g.MODEL_T)
            g.LuaExportAfterNextFrame()


def test_lua_export():
    from stick_test import Frame
    m = Mock()
    g, sent, cmds, run = m.g, m.sent, m.cmds, m.run

    g.LuaExportStart()
    lines = [json.loads(d) for _, d in sent()]
    hello = [d for d in lines if d["ev"] == "hello"][0]
    assert hello["codes"] == {"pitch": 2001, "roll": 2002, "rudder": 2003, "throttle": 2004}, hello
    assert "default" in hello["codes_from"] and hello["api"]["LoSetCommand"] is True, hello
    assert hello["api"]["LoGetAngularVelocity"] is False, "a function the mock lacks must show as absent"

    # 4 s with nobody commanding: telemetry only, no axis commands
    run(4.0)
    tel = [json.loads(d) for p, d in sent() if p == 15401 and json.loads(d)["ev"] == "tel"]
    hz = len(tel) / 4.0
    assert 55 < hz < 62, hz            # a line per rendered frame; the previous callback's 1 s tick must not slow it
    assert not cmds(), "axes were touched before any command"
    f = Frame(tel[-1])
    assert abs(f.alt - 7000.5) < 1e-6 and abs(f.hdg - 1.25) < 1e-9 and abs(f.bank + 0.3) < 1e-9
    assert abs(f.tas - 250.5) < 1e-9 and abs(f.nz - 1.2) < 1e-9 and f.rpm == 88.0 and f.fuel_flow == 1.2 and f.agl == 6800.0
    assert f.t > 50 and f.rt > 1000
    objs = [d for d in tel if d.get("obj")]
    assert objs and len(objs[-1]["obj"]) == 1 and objs[-1]["obj"][0]["unit"] == "Bandit-1", "own aircraft not excluded"
    assert any("radar" in d for d in tel) and any("pay" in d for d in tel)
    assert any(json.loads(d)["ev"] == "stats" for _, d in sent()), "no stats line"

    # commands: applied every frame while they keep arriving
    g.INBOX[1] = "AXES 7 0.25 -0.5 0.1 0.75"
    n0 = len(cmds())
    run(0.2, hook=lambda t: g.INBOX.__setitem__(len(g.INBOX) + 1, "AXES 7 0.25 -0.5 0.1 0.75") if int(t * 60) % 6 == 5 else None)
    c = cmds()[n0:]
    got = {k: [v for kk, v in c if kk == k] for k in (2001, 2002, 2003, 2004)}
    assert all(len(v) >= 10 for v in got.values()), {k: len(v) for k, v in got.items()}
    assert set(got[2001]) == {0.25} and set(got[2002]) == {-0.5} and set(got[2003]) == {0.1} and set(got[2004]) == {0.75}
    last = [json.loads(d) for p, d in sent() if p == 15401 and json.loads(d)["ev"] == "tel"][-1]
    assert last["ax"]["seq"] == 7 and last["ax"]["on"] is True, last["ax"]

    # '-' leaves an axis alone; values are clamped to -1..1
    g.INBOX[len(g.INBOX) + 1] = "AXES 8 5 - - -"
    run(0.1, hook=lambda t: g.INBOX.__setitem__(len(g.INBOX) + 1, "AXES 8 5 - - -") if int(t * 60) % 6 == 0 else None)
    assert cmds()[-4:] == [(2001, 1.0), (2002, -0.5), (2003, 0.1), (2004, 0.75)] or \
        {(2001, 1.0), (2002, -0.5)} <= set(cmds()[-8:]), cmds()[-8:]

    # watchdog: silence for > 0.5 s zeroes pitch, roll, rudder once and leaves the throttle
    n0 = len(cmds())
    run(0.8)
    tail = cmds()[n0:]
    assert (2001, 0) in tail and (2002, 0) in tail and (2003, 0) in tail, tail[-6:]
    assert not [v for k, v in tail[-30:] if k == 2004 and v == 0], "the throttle was zeroed"
    ev = [json.loads(d) for _, d in sent()]
    assert any(d["ev"] == "released" and d["why"] == "watchdog" for d in ev), "no watchdog release"
    n1 = len(cmds())
    run(0.5)
    assert len(cmds()) == n1, "axes still being written after the watchdog"

    # RELEASE, PING, raw CMD
    g.INBOX[len(g.INBOX) + 1] = "AXES 9 0.3 0.3 0.3 -"
    run(0.1)
    g.INBOX[len(g.INBOX) + 1] = "RELEASE"
    g.INBOX[len(g.INBOX) + 1] = "PING abc"
    g.INBOX[len(g.INBOX) + 1] = "CMD 3011 0.5"
    run(0.2)
    ev = [json.loads(d) for _, d in sent()]
    assert any(d["ev"] == "released" and d["why"] == "release" for d in ev)
    assert any(d["ev"] == "pong" and d["id"] == "abc" for d in ev)
    assert any(d["ev"] == "raw" and d["code"] == 3011 and d["ok"] is True for d in ev)
    assert (3011, 0.5) in cmds()
    n2 = len(cmds())
    run(0.3)
    assert len(cmds()) == n2, "axes still being written after RELEASE"

    # garbage must not break the loop
    g.INBOX[len(g.INBOX) + 1] = "AXES x y z"
    g.INBOX[len(g.INBOX) + 1] = "\x00\x01 nonsense"
    run(0.2)

    # the chained callbacks ran
    calls = g.CALLS
    assert calls.start == 1 and calls.before > 100 and calls.after > 100 and calls.activity >= 6, \
        (calls.start, calls.before, calls.after, calls.activity)
    # pause: the model time stands still, so nothing is sent (and the axes are not re-sent either way)
    n = len([1 for _, d in sent() if json.loads(d)["ev"] == "tel"])
    run(0.5, paused=True)
    assert len([1 for _, d in sent() if json.loads(d)["ev"] == "tel"]) == n, "telemetry while paused"

    # while paused the module says so once a second, so a script can tell "paused" from "DCS gone"
    idle = [d for d in m.lines() if d["ev"] == "idle"]
    assert 0 < len(idle) <= 2, len(idle)

    g.LuaExportStop()
    assert g.CALLS.stop == 1
    assert any(json.loads(d)["ev"] == "bye" for _, d in sent())
    for _, d in sent():
        json.loads(d)
    # the other source: the activity event, which runs at frame boundaries here, so 30/s at 60 fps
    e = Mock("event")
    e.g.LuaExportStart()
    e.run(4.0)
    ehz = len([1 for d in e.lines() if d["ev"] == "tel"]) / 4.0
    assert 25 < ehz < 52, ehz
    assert e.lines()[0]["source"] == "event"
    print(f"  Lua export module ........... OK  ({len(sent())} lines, {hz:.0f} telemetry/s per frame, {ehz:.0f}/s by "
          f"event at 60 fps; axes every frame, watchdog + RELEASE + PING + CMD, pause, callbacks chained)")


def test_click():
    """CLICK reaches performClickableAction; the dangerous devices and commands are refused."""
    m = Mock()
    m.g.LuaExportStart()
    clicks = lambda: [(m.g.CLICKS[i][1], m.g.CLICKS[i][2], m.g.CLICKS[i][3]) for i in range(1, len(m.g.CLICKS) + 1)]
    for text in ("CLICK 16 3010 1", "CLICK 16 3010 0", "CLICK 17 3018 1", "CLICK 19 3002 1.0", "CLICK 31 3001 1"):
        m.send(text)
    m.run(0.2)
    assert clicks() == [(16, 3010, 1), (16, 3010, 0), (17, 3018, 1), (19, 3002, 1.0), (31, 3001, 1)], clicks()
    ev = [d for d in m.lines() if d["ev"] == "click"]
    assert len(ev) == 5 and all(d["ok"] for d in ev), ev
    n = len(clicks())
    refused = ["CLICK 10 3006 1",                           # CPT_MECH: ejection handle
               "CLICK 10 3003 1", "CLICK 7 3001 1",         # canopy, landing gear handle
               "CLICK 6 3003 1", "CLICK 4 3001 1", "CLICK 3 3001 1", "CLICK 2 3001 1",
               "CLICK 32 3004 1",                           # countermeasures jettison
               "CLICK 19 3003 1", "CLICK 19 3004 1", "CLICK 19 3005 1", "CLICK 19 3018 1"]   # stores jettison, ALT REL
    for text in refused:
        m.send(text)
    m.run(0.3)
    assert len(clicks()) == n, clicks()[n:]
    ev = [d for d in m.lines() if d["ev"] == "click"][5:]
    assert len(ev) == len(refused) and all(not d["ok"] and d["err"] == "refused" for d in ev), ev
    # the deny list agrees with the module's own numbers
    defs = json.load(open(os.path.join(HERE, "f16c_defs.json")))
    mm = defs["commands"]["mmc"]
    assert (mm["EmerStoresJett"], mm["GroundJett"], mm["AltRel"], mm["GroundJett_ITER"]) == (3003, 3004, 3005, 3018)
    dv = defs["devices"]
    assert [dv[k] for k in ("CONTROL_INTERFACE", "ELEC_INTERFACE", "FUEL_INTERFACE", "HYDRO_INTERFACE", "ENGINE_INTERFACE",
                            "GEAR_INTERFACE", "OXYGEN_INTERFACE", "CPT_MECH", "CMDS")] == [2, 3, 4, 5, 6, 7, 8, 10, 32]
    assert defs["devices"]["MMC"] == 19 and defs["commands"]["cmds"]["Jett"] == 3004
    # a failing DCS call is reported, not fatal
    m.L.execute("function GetDevice(id) error('no device') end")
    m.send("CLICK 16 3010 1")
    m.run(0.1)
    last = [d for d in m.lines() if d["ev"] == "click"][-1]
    assert last["ok"] is False and "no device" in last["err"], last
    print("  CLICK ....................... OK  (5 cockpit commands pass; 12 dangerous ones refused, the deny list "
          "matches the module's numbers; a DCS error is reported)")


def test_ind():
    m = Mock()
    m.g.LuaExportStart()
    m.send("IND 0 20")
    m.run(0.2)
    ev = [d for d in m.lines() if d["ev"] in ("ind", "ind_end")]
    assert [d["id"] for d in ev if d["ev"] == "ind"] == [1, 3, 4, 9, 17], ev   # nil, "" and an error are skipped
    assert ev[-1]["ev"] == "ind_end" and ev[-1]["count"] == 5
    assert "CRM" in ev[1]["text"] and "STT" in ev[3]["text"]
    m.L.execute("function list_indication(id) if id == 1 then return string.rep('HUD_Element_Name_x\\n', 3000) end end")
    m.send("IND 0 3")
    m.run(0.1)
    big = [d for d in m.lines() if d["ev"] == "ind" and d["id"] == 1][-1]
    assert big["len"] == 3000 * 19 and len(big["text"]) == 30000, (big["len"], len(big["text"]))   # a long display is cut at 30000
    m.send("IND 0 100000")                                                 # the range is capped
    m.run(0.1)
    print("  IND ......................... OK  (a message per display, then ind_end; long displays kept to 30000 characters)")


def test_radar_markers():
    """The module reads the radar state from what the displays draw."""
    import cockpit
    from stick_test import Frame
    m = Mock()
    m.g.LuaExportStart()
    states = []
    for td, tracked, cursor in ((False, False, True), (True, True, True), (False, True, False)):
        m.g.RD_TD, m.g.RD_TRACKED, m.g.RD_CURSOR = td, tracked, cursor
        n0 = len(m.lines())
        m.run(0.6)
        rds = [d for d in m.lines()[n0:] if d["ev"] == "tel" and "rd" in d]
        assert rds, "no rd in the telemetry"
        states.append((cockpit.lock_state(Frame(rds[-1])), rds[-1]["rd"]))
    assert [s[0] for s in states] == ["search", "designated", "stt"], states
    assert states[0][1]["fcr"] == 4 and states[1][1]["td"] is True and states[2][1]["cursor"] is False
    assert states[0][1]["threat"] is True
    # no radar page up: no state to read
    m.L.execute("function list_indication(id) if id == 1 then return 'HUD_glass' end end")
    n0 = len(m.lines())
    m.run(0.6)
    last = [d for d in m.lines()[n0:] if d["ev"] == "tel" and "rd" in d][-1]
    assert cockpit.lock_state(Frame(last)) == "no radar page", last["rd"]
    print("  radar markers ............... OK  (search, designated and single-target track read from the radar page; "
          "no page: no state)")


def test_precision_and_command_defs():
    """Epoch-sized times keep their decimals; the command numbers are read from command_defs.lua."""
    m = Mock()
    m.g.CLOCK = 1760351071.123456
    m.g.LuaExportStart()
    m.run(0.2)
    raw = [d for p, d in m.sent() if json.loads(d)["ev"] == "tel"]
    rts = [json.loads(d)["rt"] for d in raw]
    assert len(set(rts)) > 5 and all(abs(b - a - 1 / 60) < 1e-4 for a, b in zip(rts, rts[1:])), rts[:6]
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "Scripts"))
        with open(os.path.join(d, "Scripts", "command_defs.lua"), "w") as fh:
            fh.write("iCommandPlanePitch = 2001\niCommandPlaneRoll = 2002\niCommandPlaneRudder = 2003\n"
                     "iCommandPlaneThrustCommon = 2004\niCommandOther = 5\n")
        m = Mock()
        m.L.execute(f'lfs = {{currentdir = function() return [[{d}]] end}}')
        # lfs must exist before the module is loaded: boot a second copy
        m.L.execute(open(os.path.join(HERE, "bvr_stick_export.lua")).read())
        m.g.LuaExportStart()
        h = [x for x in m.lines() if x["ev"] == "hello"][-1]
        assert h["codes_from"] == "command_defs.lua", h["codes_from"]
        with open(os.path.join(d, "Scripts", "command_defs.lua"), "w") as fh:
            fh.write("iCommandPlanePitch = 3001\niCommandPlaneRoll = 3002\niCommandPlaneRudder = 3003\n"
                     "iCommandPlaneThrustCommon = 3004\n")
        m.L.execute(open(os.path.join(HERE, "bvr_stick_export.lua")).read())
        m.g.LuaExportStart()
        h = [x for x in m.lines() if x["ev"] == "hello"][-1]
        assert h["codes"] == {"pitch": 3001, "roll": 3002, "rudder": 3003, "throttle": 3004}, h
    m = Mock()
    m.L.execute('lfs = {currentdir = function() return "/nonexistent" end}')
    m.L.execute(open(os.path.join(HERE, "bvr_stick_export.lua")).read())
    m.g.LuaExportStart()
    h = [x for x in m.lines() if x["ev"] == "hello"][-1]
    assert h["codes_from"].startswith("default (") and "command_defs.lua" in h["codes_from"], h["codes_from"]
    print("  precision, command_defs ..... OK  (epoch time keeps its decimals; codes read from command_defs.lua; "
          "the reason is given when it can't be read)")


def test_udp_link():
    import stick_test
    tel_port, cmd_port = 25401, 25402
    cmd_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_rx.bind(("127.0.0.1", cmd_port))
    cmd_rx.settimeout(2.0)
    lua_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    link = stick_test.UdpLink(tel_port, cmd_port)

    def send(d):
        lua_tx.sendto(json.dumps(d).encode(), ("127.0.0.1", tel_port))
    try:
        send({"ev": "hello", "codes": {"pitch": 2001}, "codes_from": "test"})
        send({"ev": "stats", "frame_hz": 61.0, "event_hz": 50.0})
        base = {"ev": "tel", "n": 1, "t": 10.0, "rt": time.time(), "me": {"hdg": 1.0, "bank": 0.1, "pitch": 0.0, "alt": 5000},
                "tas": 240, "g": {"x": 0, "y": 1.0, "z": 0}, "ax": {"seq": 0}}
        send(base)
        f = link.next(timeout=2.0)
        assert f is not None and f.n == 1 and f.hdg == 1.0 and f.alt == 5000 and f.nz == 1.0
        assert link.events["hello"]["codes_from"] == "test" and link.events["stats"]["event_hz"] == 50.0
        link.axes(pitch=0.1, roll=-0.5)
        text = cmd_rx.recvfrom(1024)[0].decode()
        assert text == "AXES 1 0.1000 -0.5000 - -", text
        link.axes(rudder=2.0, throttle=-0.25)
        assert cmd_rx.recvfrom(1024)[0].decode() == "AXES 2 - - 1.0000 -0.2500"
        time.sleep(0.01)
        send({**base, "n": 2, "rt": time.time(), "ax": {"seq": 2}})
        f2 = link.next(timeout=2.0)
        assert f2.ax_seq == 2 and len(link.latencies) == 1 and 0.0 < link.latencies[0] < 1.0, link.latencies
        link.release()
        assert cmd_rx.recvfrom(1024)[0].decode() == "RELEASE"
        link.raw(3011, 0.5)
        while True:
            t = cmd_rx.recvfrom(1024)[0].decode()
            if t.startswith("CMD"):
                break
        assert t == "CMD 3011 0.5", t
        assert link.next(timeout=0.1) is None            # silence is a timeout, not an error
        lua_tx.sendto(b"{not json", ("127.0.0.1", tel_port))
        send({**base, "n": 3})
        assert link.next(timeout=2.0).n == 3             # and a bad line is skipped
    finally:
        link.close()
        cmd_rx.close()
        lua_tx.close()
    print("  UDP link .................... OK  (frames and events in, AXES/RELEASE/CMD lines out, latency measured)")


def _check_fake(roll_sign, pitch_sign, thr_inv=False, speed=None):
    import stick_test
    with tempfile.TemporaryDirectory() as d:
        s = stick_test.main(["all", "--fake", "--out", d, "--fake-roll-sign", str(roll_sign),
                             "--fake-pitch-sign", str(pitch_sign)] + (["--fake-thr-inverted"] if thr_inv else [])
                            + (["--speed", str(speed)] if speed else []))
        gains = json.load(open(os.path.join(d, "stick_gains.json")))
        files = os.listdir(d)
    assert "aborted" not in s, s
    assert gains["roll_sign"] == roll_sign and gains["pitch_sign"] == pitch_sign, gains
    assert abs(gains["roll_rate"] - 2.6) < 0.15, gains            # rad/s per unit
    assert abs(gains["nz_per_cmd"] - 6.0) < 0.4, gains            # g per unit
    assert gains["thr_sign"] == (-1 if thr_inv else 1), gains["thr_sign"]
    assert abs(gains["thr_u_trim"] - 0.1) < 0.2 and 2.5 < gains["thr_slope"] < 5.5, gains
    assert "thr_trim" not in gains and "rpm_by_code" not in gains
    assert s["rate"]["real_hz"] > 49 and not s["rate"]["checks_failed"], s["rate"]
    h = s["hold"]
    t1 = h["heading_steps"][0]
    assert 50 < t1["peak_bank_deg"] < 64 and t1["half_s"] < 15 and t1["rate_bulk_dps"] > 3.0, t1
    assert h["alt_step"]["t90_s"] < 25 and h["alt_step"]["overshoot_m"] < 60, h["alt_step"]
    assert abs(h["speed_step"]["end_error"]) < 8, h["speed_step"]
    assert h["hold_error"]["alt_m"] < 15 and h["hold_error"]["hdg_deg"] < 3, h["hold_error"]
    assert any(f.endswith("_hold.jsonl") for f in files) and any(f.startswith("stick_summary") for f in files)
    return gains, t1


def test_fake_nonlinear():
    """DCS's first pulses gave a tenth of a linear stick's response: the curve stage must measure it
    and hold must fly through the inverse."""
    import stick_test
    with tempfile.TemporaryDirectory() as d:
        s = stick_test.main(["all", "--fake", "--fake-curve", "2.0", "--out", d])
        gains = json.load(open(os.path.join(d, "stick_gains.json")))
    assert "aborted" not in s, s
    rc, pc = dict((round(a, 2), r) for a, r in gains["roll_curve"]), dict((round(a, 2), r) for a, r in gains["pitch_curve"])
    assert rc[0.1] < 0.05 and rc[0.5] > 0.5 and rc[0.5] > 20 * rc[0.1], rc            # rad/s: ~ 2.6 a^2
    assert abs(rc[0.5] - 2.6 * 0.25) < 0.1, rc
    assert pc[0.1] < 0.1 and abs(pc[0.5] - 6.0 * 0.25) < 0.2, pc                     # g: ~ 6 a^2
    assert stick_test.inverse(gains["roll_curve"], 0.65) > 0.45 and abs(stick_test.inverse(gains["roll_curve"], 0.65) - 0.5) < 0.05
    assert stick_test.inverse(gains["roll_curve"], -0.65) < -0.45 and stick_test.inverse(gains["roll_curve"], 100.0) == 1.0
    t1 = s["hold"]["heading_steps"][0]
    assert 50 < t1["peak_bank_deg"] < 64 and t1["half_s"] < 15 and t1["rate_bulk_dps"] > 3.0, t1
    assert s["hold"]["hold_error"]["alt_m"] < 15, s["hold"]["hold_error"]
    # a pitch pulse that hits the limit is a lower bound and stays out of the curve; --axes keeps the rest
    from fake_export import FakeLink
    with tempfile.TemporaryDirectory() as d:
        stick_test.main(["pulses", "--fake", "--out", d])
        before = json.load(open(os.path.join(d, "stick_gains.json")))
        link = FakeLink(curve_pow=1.0)                       # 6 g per unit: 1.0 would give 6 g > the 5.5 g stop
        run = stick_test.Run(link, None, "c", stick_test.Limits(min_agl=0, min_alt=0))
        out = stick_test.stage_curve(run, before, axes=("pitch",))
        assert "roll_curve" not in out and 0.75 in [a for a, _ in out["pitch_curve"]], out
        assert 1.0 not in [a for a, _ in out["pitch_curve"]], out["pitch_curve"]
        stick_test.main(["curve", "--fake", "--axes", "pitch", "--out", d])
        after = json.load(open(os.path.join(d, "stick_gains.json")))
        assert "pitch_curve" in after and after["roll_sign"] == before["roll_sign"], after.keys()
        # DCS pulls to +4.8 g but pushes only to -2.7 g: the curves must not be averaged
        link = FakeLink(curve_pow=2.0, push_limit_g=2.7)
        run = stick_test.Run(link, None, "c", stick_test.Limits(min_agl=0, min_alt=0))
        out = stick_test.stage_curve(run, before, axes=("pitch",))
        pull, push = dict(map(tuple, out["pitch_curve"])), dict(map(tuple, out["pitch_push_curve"]))
        assert abs(pull[0.75] - 6 * 0.5625) < 0.15 and abs(push[0.75] - 2.7) < 0.15, (pull, push)
        assert push[1.0] < 2.8 and pull[0.75] > push[0.75] + 0.4, (pull, push)
        g2 = {**before, **out}
        h = stick_test.Hold(g2)
        assert stick_test.inverse(out["pitch_push_curve"], 2.0) < 1.0 and stick_test.inverse(out["pitch_curve"], 3.0) < 1.0
    print(f"  stick_test.py, non-linear ... OK  (response ~ command squared measured; hold through the inverse "
          f"turns {t1['rate_bulk_dps']:.1f} deg/s at {t1['peak_bank_deg']:.0f} deg of bank)")


def test_fake_end_to_end():
    import stick_test
    from fake_export import FakeLink
    for rs, ps, ti, sp in ((1, 1, False, None), (-1, -1, True, 215), (1, -1, True, None)):
        gains, t1 = _check_fake(rs, ps, ti, sp)
    # a limit stops it and releases the axes
    link = FakeLink()
    run = stick_test.Run(link, None, "limit", stick_test.Limits(min_agl=1e5))
    try:
        run.next()
    except stick_test.Abort as e:
        assert "above ground" in str(e), e
    else:
        raise AssertionError("the height limit did not stop the run")
    # a bank beyond 85 deg stops a run that keeps rolling
    link = FakeLink()
    lim = stick_test.Limits(min_agl=0, min_alt=0)
    run = stick_test.Run(link, None, "bank", lim)
    try:
        stick_test.drive(run, 30.0, lambda f, tau: {"pitch": 0.0, "roll": 0.5, "rudder": 0.0})
    except stick_test.Abort as e:
        assert "bank" in str(e), e
    else:
        raise AssertionError("the bank limit did not stop the run")
    print("  stick_test.py, synthetic .... OK  (signs found both ways round, gains within 5%, "
          f"hold turn {t1['rate_bulk_dps']:.1f} deg/s at {t1['peak_bank_deg']:.0f} deg of bank; limits stop it)")


def test_roll_lead():
    """DCS rolls 140 deg/s and takes 0.4 s to stop: without the lead term the bank overshoots by 15 deg or more."""
    import stick_test
    peaks = {}
    for lead in (0.0, stick_test.Hold.T_LEAD):
        stick_test.Hold.T_LEAD = lead
        try:
            with tempfile.TemporaryDirectory() as d:
                s = stick_test.main(["all", "--fake", "--fake-roll-tau", "0.35", "--bank", "60", "--out", d])
        finally:
            stick_test.Hold.T_LEAD = 0.25
        assert "aborted" not in s, s
        peaks[lead] = s["hold"]["heading_steps"][0]["peak_bank_deg"]
    assert peaks[0.0] > 70 and peaks[0.25] < 66, peaks
    print(f"  bank lead ................... OK  (a 0.35 s roll lag: peak bank {peaks[0.0]:.0f} deg without the lead, "
          f"{peaks[0.25]:.0f} with, for a 60 deg limit)")


def test_g_loop():
    """At 80 deg of bank on the simulator's altitude law the pilot must fly the g that holds altitude (5.8), even
    when the stick gives more g than the measured curve says (DCS: 7 g at 304 m/s, the turn climbed 150 m)."""
    import stick_test
    from fake_export import FakeLink
    gains = {"roll_sign": 1, "pitch_sign": 1, "thr_sign": 1, "thr_u_trim": 0.1,
             "roll_curve": [[0.5, 1.3], [1.0, 2.6]], "pitch_curve": [[0.5, 3.0], [1.0, 6.0]]}
    link = FakeLink(nz_per_cmd=8.5, nz_tau=0.25, nz_rate_max=6.0)   # 40% more g per stick than the curve; DCS's lag
    run = stick_test.Run(link, None, "g", stick_test.Limits(min_agl=0, min_alt=0))
    f0 = run.next()
    hold = stick_test.Hold(gains, bank_max=math.radians(80), nz_max=8.0)
    hdg, alt = f0.hdg + math.radians(150), f0.alt
    frames = stick_test.drive(run, 14.0, lambda f, tau: hold.axes(f, hdg, alt, v=None, dive=math.radians(25),
                                                                     climb=math.radians(15)))
    # and out of the turn: no pitch oscillation (DCS: +3.4 to -1.5 g every 4 s with a fast correction)
    frames += stick_test.drive(run, 12.0, lambda f, tau: hold.axes(f, hdg, alt, v=None, dive=math.radians(25),
                                                                      climb=math.radians(15)))
    out = frames[-300:]
    assert min(f.nz for f in out) > 0.3 and max(f.nz for f in out) < 2.0, (min(f.nz for f in out), max(f.nz for f in out))
    turning = [f for f in frames if abs(f.bank) > math.radians(75)]
    climb = max(abs(f.alt - f0.alt) for f in frames)
    late = [f.nz for f in turning[len(turning) // 2:]]
    mean_late = sum(late) / len(late)
    # the factor learnt for the stick's g is the fake's true one (curve 6 g a unit, the stick 8.5: 0.71); the
    # altitude stays within 30 m through the turn and the roll-out
    assert len(turning) > 100 and 0.62 < hold.nz_gain < 0.80 and climb < 30.0 and 4.5 < mean_late < 6.5, \
        (hold.nz_gain, mean_late, climb, len(turning))
    # with the airframe, bank and g are limited as the simulator limits them: less bank as the wing gives less g
    sys.path.insert(0, os.path.join(HERE, "..", ".."))
    from bvr_library import load_platform
    h = stick_test.Hold(gains, bank_max=math.radians(80), nz_max=8.0, airframe=load_platform("F-16C-DCS").airframe)
    lims = {}
    for tas in (312.0, 200.0, 150.0):
        fr = stick_test.Frame({"me": {"alt": 6950.0, "hdg": 0.0, "bank": 0.0}, "tas": tas, "vs": 0.0, "t": 1.0,
                               "g": {"y": 1.0}, "eng": {"fuel_internal": 0.86}})
        h.axes(fr, math.radians(150), 6950.0, v=tas, dive=0.4, climb=0.3)
        lims[tas] = (h.last["nz_avail"], math.degrees(h.last["bank_lim"]))
    assert lims[312.0][1] == 80.0 and 65 < lims[200.0][1] < 78 and 50 < lims[150.0][1] < 66, lims
    # a stick that is right and a g as slow as DCS's (1.5 s): the lag must not be learnt as a stick too strong (in DCS
    # the factor fell to 0.54 rolling out of the 180 deg turn); it is, with no lag expected
    facs = {}
    for lag in (None, (0.01, 0.01)):
        link = FakeLink(nz_per_cmd=6.0, nz_tau=1.5)
        run = stick_test.Run(link, None, "g", stick_test.Limits(min_agl=0, min_alt=0))
        f0 = run.next()
        h = stick_test.Hold(gains, bank_max=math.radians(80), nz_max=8.0)
        if lag:
            h.lag = lag
        tgt = f0.hdg + math.radians(170)
        low = [1.0]

        def fly(f, tau, h=h, tgt=tgt, alt=f0.alt, low=low):
            a = h.axes(f, tgt, alt, v=None, dive=math.radians(25), climb=math.radians(15))
            low[0] = min(low[0], h.nz_gain)
            return a
        stick_test.drive(run, 40.0, fly)
        facs[lag] = low[0]
    assert facs[None] > 0.85 and facs[(0.01, 0.01)] < facs[None] - 0.03, facs
    # the factor is per speed: one learnt at 300 m/s does not reach 175 (in DCS 0.69 from a pull-up at 300 under-pulled
    # a turn at 176)
    h = stick_test.Hold(gains, bank_max=math.radians(80), nz_max=8.0)
    h.nz_gains[3] = 0.69                                   # the 300 m/s band
    fr = lambda tas: stick_test.Frame({"me": {"alt": 7000.0, "hdg": 0.0, "bank": 0.0}, "tas": tas, "vs": 0.0, "t": 1.0,
                                       "g": {"y": 1.0}})
    got = {}
    for tas in (300.0, 275.0, 175.0):
        h.axes(fr(tas), 0.0, 7000.0)
        got[tas] = h.nz_gain
    assert abs(got[300.0] - 0.69) < 1e-9 and abs(got[275.0] - 0.845) < 1e-9 and got[175.0] == 1.0, got
    print(f"  g loop ...................... OK  (80 deg of bank, the stick 40% stronger than its curve and g 0.25 s "
          f"late: factor {hold.nz_gain:.2f} learnt (0.71), altitude within {climb:.0f} m through turn and roll-out; bank limit 80/{lims[200.0][1]:.0f}/"
          f"{lims[150.0][1]:.0f} deg at 312/200/150 m/s; a right stick with DCS's slow g: factor at least "
          f"{facs[None]:.2f}, {facs[(0.01, 0.01)]:.2f} without the lag expected)")


def test_manoeuvre():
    """The manoeuvre program flown on the fake aircraft, saved, replayed through the simulator and compared; the
    fit recovers a known change of the airframe from a record the simulator itself made."""
    import io
    import contextlib
    import numpy as np
    import manoeuvre as M
    import stick_test
    from fake_export import FakeLink
    gains = {"roll_sign": 1, "pitch_sign": 1, "thr_sign": 1, "thr_u_trim": 0.1,
             "roll_curve": [[0.5, 1.3], [1.0, 2.6]], "pitch_curve": [[0.5, 3.0], [1.0, 6.0]]}
    short = [("hold", 3.0, {}), ("turn right 90", 20.0, {"dhdg": 90.0}), ("climb 2000 m", 30.0, {"dalt": 2000.0}),
             ("accelerate", 25.0, {"spd": "hi"})]
    with tempfile.TemporaryDirectory() as d:
        link = FakeLink()
        run = stick_test.Run(link, None, "m", stick_test.Limits(min_agl=0, min_alt=0))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = M.stage_manoeuvre(run, gains, d, program=short)
            rec = json.load(open(res["path"]))
            sim, _ = M.simulate(rec)
            rows = M.compare(rec, sim)
        assert res["samples"] > 700 and not link.controlling, (res, buf.getvalue()[-500:])
        meas = {(r["segment"], r["measure"]) for r in rows}
        assert ("turn right 90", "turn rate deg/s") in meas and ("climb 2000 m", "vertical speed m/s") in meas \
            and ("accelerate", "acceleration m/s2") in meas, meas
    # the fit: a record made by the simulator with thrust x0.85 and CL_MAX x0.9; a few iterations already halve
    # the mismatch and move both the right way
    from bvr_library import load_platform
    base = load_platform("F-16C-DCS").airframe
    rec = {"start": {"hdg": 1.0, "alt": 7000.0, "tas": 280.0, "mach": 0.9, "fuel": None}, "platform": "F-16C-DCS",
           "program": short}
    rec["samples"], _ = M.simulate(rec, "F-16C-DCS", 0.6, cfg=M._cfg_with(base, np.array([0.85, 1.0, 1.0, 0.9, 1.0])))
    with contextlib.redirect_stdout(io.StringIO()):
        x, m0, m1, _ = M.fit(rec, "F-16C-DCS", 0.6, iters=25)
    assert m1 < 0.5 * m0 and x[0] < 1.0 and x[3] < 1.0, (x, m0, m1)
    # the wing limit: samples at the stick's stop pulling CL 1.05 give back CL 1.05, and the fit then keeps it
    from f16_sim import isa
    w = (base.MASS_EMPTY + base.MASS_WPNS + 0.6 * base.FUEL_FULL) * 9.80665
    smp = []
    for i in range(40):
        tas, alt = 300.0 - 2 * i, 6700.0
        nz = 1.05 * 0.5 * isa(alt)[0] * tas ** 2 * base.S_REF / w
        smp.append({"t": i * 0.1, "stick": 1.0, "nz": nz, "tas": tas, "alt": alt, "fuel": 0.6})
    wl = M.wing_limit({"start": {"fuel": 0.6}, "samples": smp}, base)
    assert wl and abs(wl[0] - 1.05) < 0.01 and abs(wl[1] - 1.05) < 0.01 and wl[2] == 40, wl
    with contextlib.redirect_stdout(io.StringIO()):
        xc, *_ = M.fit(rec, "F-16C-DCS", 0.6, iters=3, cl_max=1.05)
    assert abs(xc[3] * base.CL_MAX - 1.05) < 1e-9, xc
    # the g lag: a demand turned into g by the simulator's lag (0.8 s building up, 1.6 s coming off) gives them back
    t = np.arange(0, 120, 0.1)
    cmd = 1.0 + 4.5 * (np.sin(t / 3.0) > 0.3) - 1.5 * (np.sin(t / 5.0) < -0.8)
    nz = M._lag_model(t, cmd, np.full_like(t, 1.0), np.zeros(len(t), bool), 0.8, 1.6)
    lg = M.g_lag({"samples": [{"t": a, "nz": b, "nz_cmd": c, "stick": 0.5} for a, b, c in zip(t, nz, cmd)]})
    assert lg and abs(lg[0] - 0.8) < 0.06 and abs(lg[1] - 1.6) < 0.06 and lg[2] < 0.05 < lg[3], lg
    # and the fit keeps a measured lag in the airframe it fits
    with contextlib.redirect_stdout(io.StringIO()):
        *_, cfg_l = M.fit(rec, "F-16C-DCS", 0.6, iters=2, lag=(0.8, 1.6))
    assert (cfg_l.NZ_TAU_ON, cfg_l.NZ_TAU_OFF) == (0.8, 1.6) and getattr(base, "NZ_TAU_ON", 0.0) == 0.0
    print(f"  manoeuvre compare and fit ... OK  (program flown and compared; the fit takes the mismatch from "
          f"{m0:.2f} to {m1:.2f}, thrust x{x[0]:.2f}, CL_MAX x{x[3]:.2f} for x0.85, x0.90; g lag "
          f"{lg[0]:.2f} / {lg[1]:.2f} s measured for 0.8 / 1.6)")


def test_sim_g_lag():
    """f16_sim's g response: instant with NZ_TAU_ON/OFF 0 (every library airframe before 14 October), lagged with them."""
    import copy
    import math
    import numpy as np
    from f16_sim import F16Aircraft
    from bvr_library import load_platform
    base = load_platform("F-16C-DCS").airframe
    out = {}
    for taus in ((0.0, 0.0), (0.9, 2.0)):
        c = copy.copy(base)
        c.NZ_TAU_ON, c.NZ_TAU_OFF = taus
        ac = F16Aircraft(np.random.default_rng(0), cfg=c)
        ac.reset_state(0.0, 0.0, 7000.0, 0.0, 280.0, fuel_frac=0.6)
        nz, alt = [], []
        for i in range(1000):                       # 20 s: a hard turn to 90 deg, then level
            ac.step(0.02, {"hdgCmd": math.pi / 2, "altTarget": 7000.0, "V": 280.0, "altFPA": 0.4, "climbFPA": 0.4})
            nz.append(ac.nz)
            alt.append(ac.z)
        out[taus] = (nz, alt)
    n0, n1 = out[(0.0, 0.0)][0], out[(0.9, 2.0)][0]
    k = next(i for i, v in enumerate(n0) if v > 4.0)
    assert n1[k] < n0[k] - 1.0, (k, n0[k], n1[k])            # when the instant g passes 4, the lagged one is building
    sink0 = 7000.0 - min(out[(0.0, 0.0)][1])
    sink1 = 7000.0 - min(out[(0.9, 2.0)][1])
    assert sink1 > sink0 + 5.0, (sink0, sink1)               # the lag lets the nose drop at the roll-in, as in DCS
    print(f"  simulator g lag ... OK  ({k * 0.02:.1f} s into the turn {n0[k]:.1f} g instant, {n1[k]:.1f} g lagged; "
          f"altitude lost {sink0:.0f} against {sink1:.0f} m)")


def test_fire_delay():
    """fire_delay.py: timed bugs (target 0, +-20 deg off the nose) and a launch with the live runner's pilot on the fake
    cockpit, and the platform range from them and from live request-to-launch times."""
    import io
    import contextlib
    import time as _time
    import fire_delay as FD
    import cockpit
    from fake_export import FakeLink
    gains = {"roll_sign": 1, "pitch_sign": 1, "thr_sign": 1, "thr_u_trim": 0.1,
             "roll_curve": [[0.5, 1.3], [1.0, 2.6]], "pitch_curve": [[0.5, 3.0], [1.0, 6.0]]}

    class Slow(FakeLink):
        def next(self, timeout=1.0):
            _time.sleep(0.001)
            return super().next(timeout)
    with tempfile.TemporaryDirectory() as d:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rec = FD.measure(Slow(cockpit_defs=cockpit.load_defs(), scan_needed=999), gains=gains, trials=3, fire=1,
                             out_dir=d, log=print)
        bugs, fires = rec["bug_times"], rec["fire_times"]
        assert len(bugs) == 3 and all(b["ok"] and b["s"] > 0.5 for b in bugs), (bugs, buf.getvalue()[-800:])
        assert len(fires) == 1 and fires[0]["ok"], (fires, buf.getvalue()[-800:])
        assert sorted(round(abs(b["az"]) / 10) for b in bugs) == [0, 2, 2], bugs     # 0, +20, -20 deg
        dd = FD.delays_from([rec["path"]])
        # the launch found the track bugged (the bug from the last timed one): its whole time is the release
        assert len(dd["release"]) == 1 and len(dd["bug"]) == 3, dd
        csvp = os.path.join(d, "results.csv")
        with open(csvp, "w") as fh:
            fh.write("episode,mode,launch_delays_s\n1,stick,4.0;5.5\n2,stick,6.0;4.5;7.5\n3,live,99\n")
        dd = FD.delays_from([rec["path"], csvp])
        assert dd["live"] == [4.0, 5.5, 6.0, 4.5, 7.5], dd["live"]
        with contextlib.redirect_stdout(io.StringIO()):
            rel, bug = FD.summarise(dd, log=print)
        r0 = dd["release"][0]
        assert rel[0] <= r0 <= rel[1] and rel[1] - rel[0] <= 2.0, (rel, r0)
        assert bug[0] <= min(dd["bug"]) + 0.5 and bug[1] >= max(dd["bug"]) - 0.5, (bug, dd["bug"])
        # set: written to a user platform (a copy of the user library here) and read back by the simulator
        import shutil
        import bvr_library
        lib = os.path.join(d, "user")
        shutil.copytree(str(bvr_library.USER_DIR), lib)
        keep = bvr_library.USER_DIR
        bvr_library.USER_DIR = type(keep)(lib)
        try:
            got = FD.set_platform("F-16C-DCSFIT", rel, bug, "test")
        finally:
            bvr_library.USER_DIR = keep
        assert got == (rel, (bug[0], bug[1], FD.BUG_READY_S)), got
    print(f"  fire delay .................. OK  (fake: bugs {', '.join(f'{b[chr(115)]:.1f}' for b in bugs)} s, a launch "
          f"{fires[0]['total_s']:.1f} s with the track bugged; set: release {rel[0]:.1f}-{rel[1]:.1f} s, bug "
          f"{bug[0]:.1f}-{bug[1]:.1f} s more)")


def test_course():
    """The pilot steers by the course (the velocity vector), as the simulator's autopilot, not by the nose: in a hard
    turn DCS's nose is 15-22 deg inside the flight path."""
    import math
    import stick_test
    gains = {"roll_sign": 1, "pitch_sign": 1, "thr_sign": 1, "roll_curve": [[0.5, 1.3], [1.0, 2.6]],
             "pitch_curve": [[0.5, 3.0], [1.0, 6.0]]}
    crs = math.radians(40.0)
    d = {"t": 1.0, "asl": 7000.0, "tas": 250.0, "mach": 0.8, "g": {"y": 5.0}, "vs": 0.0,
         "me": {"hdg": math.radians(60.0), "pitch": 0.1, "bank": math.radians(70.0)},
         "vel": {"x": 250.0 * math.cos(crs), "y": 0.0, "z": 250.0 * math.sin(crs)}}
    f = stick_test.Frame(d)
    assert abs(f.crs - crs) < 1e-9 and abs(f.hdg - math.radians(60.0)) < 1e-9, (f.crs, f.hdg)
    # aiming at 60: by the nose there is nothing left to turn (wings level asked), by the course 20 deg (40 of bank)
    h = stick_test.Hold(gains, bank_max=math.radians(80.0), nz_max=8.0)
    h.axes(f, math.radians(60.0), 7000.0)
    assert h.last["bank_lim"] > math.radians(40.0)
    f2 = stick_test.Frame(dict(d, vel={}))          # no velocity: the nose, as before
    assert f2.crs == f2.hdg
    roll_course = h.axes(f, math.radians(60.0), 7000.0)["roll"]
    h2 = stick_test.Hold(gains, bank_max=math.radians(80.0), nz_max=8.0)
    roll_nose = h2.axes(f2, math.radians(60.0), 7000.0)["roll"]
    # banked 70: by the course (40 deg asked) it rolls out less hard than by the nose (0 asked)
    assert roll_nose < roll_course < 0.0, (roll_nose, roll_course)
    print(f"  pilot steers by the course ... OK  (nose 60, course 40, aiming 60: roll {roll_course:+.2f} by the "
          f"course, {roll_nose:+.2f} by the nose)")


def test_cockpit():
    import io
    import contextlib
    import cockpit
    import stick_test
    from fake_export import FakeLink
    defs = cockpit.load_defs()
    # names resolve to the module's own numbers and key values
    assert cockpit.resolve("hotas.STICK_TGT_MANAGE_UP", defs) == (16, 3010, 1.0, 0.0)
    assert cockpit.resolve("HOTAS.STICK_WEAPON_RELEASE", defs) == (16, 3022, 1.0, 0.0)
    assert cockpit.resolve("ufc.AA", defs) == (17, 3018, 1.0, 0.0)
    assert cockpit.resolve("mmc.MasterArmSw", defs, value=1.0) == (19, 3002, 1.0, None)
    assert cockpit.resolve("left_osb.4", defs) == (24, 3004, 1.0, 0.0)
    assert cockpit.resolve("right_osb.20", defs)[0] == 25
    assert cockpit.resolve(["16", "3010"], defs) == (16, 3010, 1.0, 0.0)
    for bad in ("mmc.MasterArmSw", "nope.X", "hotas.NOT_A_COMMAND", "mfd.OSB_1"):
        try:
            cockpit.resolve(bad, defs)
        except ValueError as e:
            assert bad.split(".")[0] in str(e) or "switch" in str(e) or "MFD" in str(e), str(e)
        else:
            raise AssertionError(f"{bad} should not resolve")
    # the UDP line
    import socket as so
    rx = so.socket(so.AF_INET, so.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 25412))
    rx.settimeout(2.0)
    link = stick_test.UdpLink(25411, 25412)
    try:
        link.click(16, 3010, 1.0)
        assert rx.recvfrom(100)[0].decode() == "CLICK 16 3010 1", "wire format"
        link.click(19, 3002, -0.5)
        assert rx.recvfrom(100)[0].decode() == "CLICK 19 3002 -0.5"
    finally:
        link.close()
        rx.close()
    # gains the fake aircraft is flown with
    gains = {"roll_sign": 1, "pitch_sign": 1, "thr_sign": 1, "thr_u_trim": 0.1,
             "roll_curve": [[0.5, 1.3], [1.0, 2.6]], "pitch_curve": [[0.5, 3.0], [1.0, 6.0]]}
    lim = stick_test.Limits(min_agl=0, min_alt=0)

    def flow(scan_needed, scan):
        link = FakeLink(cockpit_defs=defs, scan_needed=scan_needed)
        run = stick_test.Run(link, None, "cp", lim)
        run.next()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = cockpit.stage_radar(run, 4.0)
            lk = cockpit.stage_lock(run, gains, defs, scan=scan, aa=True)
            fr = cockpit.stage_fire(run, gains, defs, aa=True)
        return r, lk, fr, buf.getvalue(), link
    r, lk, fr, out, link = flow(0, False)
    assert not r["locked_seen"] and not r["targets_seen"], r        # nothing set up before `lock`
    assert lk["locked"] and fr["fired"] and fr["payload"] == [4, 3], (lk, fr)
    assert "LOCKED" in out and "a missile left the aircraft" in out and "TGT-1" in out and "km" in out, out
    assert not link.cockpit.armed, "master arm must be back OFF after fire"
    # the A-A button toggles: by default it is pressed only if the HUD is not in A-A
    link = FakeLink(cockpit_defs=defs)
    link.cockpit.aa = True
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    for _ in range(25):
        run.next()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lk2 = cockpit.stage_lock(run, gains, defs)
    assert lk2["locked"] and "the HUD shows it" in buf.getvalue() and "ICP A-A button" not in buf.getvalue(), buf.getvalue()
    assert link.cockpit.aa, "A-A mode was toggled off"
    link = FakeLink(cockpit_defs=defs)                                  # NAV: the button is pressed by itself
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    for _ in range(25):
        run.next()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lk3 = cockpit.stage_lock(run, gains, defs)
    assert lk3["locked"] and "the HUD is not in A-A: pressing the ICP A-A button" in buf.getvalue(), buf.getvalue()
    assert link.cockpit.aa
    buf = io.StringIO()                                                  # and never when told not to
    link = FakeLink(cockpit_defs=defs)
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    for _ in range(25):
        run.next()
    with contextlib.redirect_stdout(buf):
        lk4 = cockpit.stage_lock(run, gains, defs, aa=False)
    assert not lk4["locked"] and not link.cockpit.aa, buf.getvalue()
    # the radar cursor has to be slewed first: lock fails without --scan, works with it
    r, lk, fr, out, link = flow(3, False)
    assert not lk["locked"] and not fr["fired"] and "run `radar`" in out, (lk, fr)
    r, lk, fr, out, link = flow(3, True)
    assert lk["locked"] and fr["fired"], (lk, fr)
    assert fr["hud_mrm"] == [4, 3], fr                # the HUD's AIM-120 count goes down with the release
    # ind through the fake: saved, and the difference between "not locked" and "locked" is what inddiff shows
    with tempfile.TemporaryDirectory() as d:
        link = FakeLink(cockpit_defs=defs)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cockpit.stage_ind(link, "before", d)
            link.cockpit.aa = True
            link.cockpit.locked = True
            cockpit.stage_ind(link, "locked", d)
        files = sorted(os.listdir(d))
        assert len(files) == 2 and "RWS" in out.getvalue(), (out.getvalue(), files)
        lines = []
        n = cockpit.ind_diff(os.path.join(d, [f for f in files if "before" in f][0]),
                             os.path.join(d, [f for f in files if "locked" in f][0]), out=lines.append)
        text = "\n".join(lines)
        assert n == 2 and "+ HUD_TD_Box" in text and "+ HUD_ASEC" in text and "- RWS" in text and "+ STT" in text, text
        lines = []
        assert cockpit.ind_diff(os.path.join(d, files[0]), os.path.join(d, files[0]), out=lines.append) == 0
    # match-alt: the script flies to the target's altitude, and fire without a lock says it is a maddog shot
    link = FakeLink(cockpit_defs=defs)
    link.cockpit.tx, link.cockpit.tz = link.cockpit.tx, link.cockpit.tz
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    for _ in range(12):
        run.next()
    link.ac.alt = 6000.0                                                # the target is at 6000 m + the fake's own height
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn = cockpit.hold_fn(run, gains, match_alt=True)
    assert "matching the target's altitude" in buf.getvalue(), buf.getvalue()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        frr = cockpit.stage_fire(run, gains, defs, force=True, aa=True)
    assert frr["fired"] and "maddog" in buf.getvalue() and "stations that changed" in buf.getvalue(), buf.getvalue()
    # the cursor probe shows the lines that follow the cursor
    link = FakeLink(cockpit_defs=defs)
    link.cockpit.aa = True
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = cockpit.stage_cursorprobe(run, gains, defs, steps=3, hold=0.2)
    text = buf.getvalue()
    assert res["steps"] == 6 and "+ 23" in text and "- 20" in text and "cursor back 3" in text and "+ 20" in text, text
    # cursortry announces every way and sends the clicks
    link = FakeLink(cockpit_defs=defs)
    link.cockpit.aa = True
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = cockpit.stage_cursortry(run, gains, defs, hold=0.2)
    assert res["tried"] == list("ABCDEFGHI") and "try D" in buf.getvalue(), buf.getvalue()
    # the sensor of interest: with it on the weapons page, target management does nothing until display management left
    link = FakeLink(cockpit_defs=defs)
    link.cockpit.aa, link.cockpit.soi_left = True, False
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    for _ in range(25):
        run.next()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lk5 = cockpit.stage_lock(run, gains, defs, soi="none")
    assert not lk5["locked"], buf.getvalue()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lk6 = cockpit.stage_lock(run, gains, defs, soi="left")                       # soi asked for
    assert lk6["locked"] and "sensor of interest" in buf.getvalue(), buf.getvalue()
    # an axis control is pressed, held, then set back with --release
    buf = io.StringIO()
    sent = []
    link.click = lambda dev, cmd, value, mode=None: (sent.append((dev, cmd, value)), link.events.__setitem__("click", {"ev": "click", "ok": True}))
    with contextlib.redirect_stdout(buf):
        cockpit.click(run, "hotas.THROTTLE_RDR_CURSOR_Y_AXIS", defs, value=0.5, hold=0.2, release=0.0)
    assert [x[2] for x in sent] == [0.5, 0.0], sent
    # a click the module refuses is reported
    link = FakeLink(cockpit_defs=defs)
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    link.click = lambda dev, cmd, value, mode=None: link.events.__setitem__("click", {"ev": "click", "ok": False, "err": "refused"})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cockpit.click(run, "hotas.STICK_TGT_MANAGE_UP", defs)
    assert "NOT done: refused" in buf.getvalue(), buf.getvalue()
    # the CLI
    with tempfile.TemporaryDirectory() as d:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            stick_test.main(["click", "ufc.AA", "--fake", "--out", d])
        assert "device 17, command 3018" in out.getvalue(), out.getvalue()
    # --offset and --dalt: the script turns to put the target 30 deg right and climbs 500 m above it; cursormap runs
    link = FakeLink(cockpit_defs=defs)
    link.cockpit.aa = True
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    cockpit.HOLD_OPTS.update(offset=30.0, dalt=500.0)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = cockpit.stage_cursormap(run, gains, defs, hold=0.2, max_steps=3)
    finally:
        cockpit.HOLD_OPTS.update(offset=None, dalt=None)
    nr = cockpit.nearest(cockpit.last_with(run.frames[-300:], "obj"))
    assert abs(nr[2] - 30.0) < 5.0 and abs(-nr[3] - 500.0) < 200.0 and res["steps"] > 0, (nr, buf.getvalue()[-800:])
    # the nearest aircraft, not the AIM-120 just fired (DCS: the next runs set up on the missile, 17 km away and lofting)
    fr = stick_test.Frame({"me": {"x": 0.0, "z": 0.0, "alt": 7000.0, "hdg": 0.0},
                           "obj": [{"name": "AIM_120C", "l1": 4, "l2": 4, "x": 17000.0, "y": 10000.0, "z": 0.0},
                                   {"unit": "TGT-1", "l1": 1, "l2": 1, "x": 40000.0, "y": 7000.0, "z": 0.0}]})
    assert cockpit.nearest(fr)[0] == "TGT-1", cockpit.nearest(fr)
    # the radar page's text as DCS wrote it (13 October): the cursor's bullseye bearing and range, coverage, scale, SOI
    real = ["Range_Scale_Root", "children are {", "Range_Scale_Value_Outlined", "40", "Range_Scale_Value", "40", "}",
            "FCR_Bearing_range", "174 245", "DBG_AA_RadarContactsKeeper", "Cursor_Root", "children are {",
            "Cursor_Line_Left", "Cursor_Line_Right", "Max_Search_Altitude", "34", "Min_Search_Altitude", "11", "}",
            "FCR_TUC_contactData", "FCR_TUC_contact_Altitude", "23",
            "  Table. Root. Unic ID: _id:8. Text", "4", "B", "children are {", "  Table. Root. Unic ID: _id:9. Text", "A",
            "6", "RWS Table. Root. Unic ID: _id:14", "children are {", "RWS Table. Root. Unic ID: _id:14. Text", "RWS",
            "TWS Table. Root. Unic ID: _id:17", "children are {", "}", "SOI"]
    assert cockpit.fcr_info(real) == {"br": (174, 245), "hi": 34, "lo": 11, "scale": 40, "tuc": True, "tuc_alt": 23,
                                      "soi": True, "mode": "RWS", "az": 6, "bars": 4}, cockpit.fcr_info(real)
    # the aimed lock: a target 30 deg right and 2 km above, which only aiming designates (scan_needed 999), is locked;
    # one 70 deg off the nose is refused
    # the aimed lock: the readout is switched from the bullseye (485 km ahead, like DCS's default) to our own aircraft
    # through the DED; with ref="bull" the bullseye is found at the bottom edge instead (slower, coarse); --relock
    # drops a lock that is already there; a target 70 deg off the nose is refused
    for az, dh, ok, ref, locked0 in ((30.0, 2000.0, True, "auto", False), (70.0, 0.0, False, "auto", False),
                                     (-20.0, 3000.0, True, "bull", False), (10.0, 6000.0, True, "auto", True)):
        link = FakeLink(cockpit_defs=defs, scan_needed=999)
        c, ac = link.cockpit, link.ac
        c.aa = True
        c.stage = 2 if locked0 else 0
        a = ac.hdg + math.radians(az)
        c.tx, c.tz, c.ty = ac.x + 55000 * math.cos(a), ac.z + 55000 * math.sin(a), ac.alt + dh
        run = stick_test.Run(link, None, "cp", lim)
        run.next()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = cockpit.stage_aimlock(run, gains, defs, relock=locked0, ref=ref)
        out = buf.getvalue()
        assert res["locked"] == ok, out[-1500:]
        assert ok or "outside the radar's azimuth limits" in out, out[-800:]
        assert "magnetic variation +6.0" in out, out[:800]
        assert ("readout from our own aircraft" in out) == (ref == "auto"), out[:1200]
        assert (ref == "bull") == ("coarse, because the bullseye is 261 NM away" in out), out[:1200]
        assert not locked0 or ("drops the lock first" in out and "after: radar search" in out), out[:1200]
        n_aim = out.count("    aim ")
        assert ref == "bull" or not ok or n_aim <= 12, (n_aim, out[-1500:])     # straight there, no hunting
    # the TWS lock: target management right to TWS, right again bugs the track, up gives the single-target track
    link = FakeLink(cockpit_defs=defs, scan_needed=999)
    c, ac = link.cockpit, link.ac
    c.aa = True
    a = ac.hdg + math.radians(-40.0)
    c.tx, c.tz = ac.x + 55000 * math.cos(a), ac.z + 55000 * math.sin(a)
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = cockpit.stage_twslock(run, gains, defs)
    out = buf.getvalue()
    assert res["designated"] and not res["locked"] and "RWS -> TWS" in out and "mode now TWS" in out \
        and "radar page: RWS, A6" in out and "the TWS track is bugged" in out, out[-1200:]
    with contextlib.redirect_stdout(io.StringIO()):          # stt=True: target management up after the bug
        res = cockpit.stage_twslock(run, gains, defs, stt=True)
    assert res["locked"], res
    # a target 30 deg off the nose is outside a scan centred on the nose: the cursor is moved to it first
    link = FakeLink(cockpit_defs=defs, scan_needed=999)
    c, ac = link.cockpit, link.ac
    c.aa = True
    a = ac.hdg + math.radians(30.0)
    c.tx, c.tz = ac.x + 55000 * math.cos(a), ac.z + 55000 * math.sin(a)
    run = stick_test.Run(link, None, "cp", lim)
    run.next()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = cockpit.stage_twslock(run, gains, defs)
    assert res["designated"] and "cursor to the right edge" in buf.getvalue() and abs(c.cur_az - 30.0) < 5.0, \
        (c.cur_az, buf.getvalue()[-800:])
    print("  cockpit, radar, lock, fire .. OK  (names resolve to the module's numbers; radar, lock, fire run end to "
          "end on a fake cockpit, with and without a cursor scan; the aimed lock off the nose and above; master arm ends OFF; refusals are shown)")


def test_stall_message():
    import stick_test
    from fake_export import FakeLink

    class Silent(FakeLink):
        def next(self, timeout=1.0):
            return None
    link = FakeLink()
    run = stick_test.Run(link, None, "stall", stick_test.Limits(min_agl=0, min_alt=0))
    for _ in range(5):
        run.next()
    run.link = Silent()
    try:
        run.next()
    except stick_test.Abort as e:
        assert "nothing at all" in str(e) and "model time" in str(e) and "Mach" in str(e), str(e)
    else:
        raise AssertionError("a silent link did not stop the run")
    run.link.last_tel_wall, run.link.last_event_wall = 10.0, 11.0          # an idle line came after the last frame
    try:
        run.next()
    except stick_test.Abort as e:
        assert "PAUSED" in str(e), str(e)
    else:
        raise AssertionError("a paused sim did not stop the run")
    print("  stall diagnosis ............. OK  (paused sim and silent DCS are told apart, with the last frame)")


def test_install():
    import install_export
    sample = "-- my export\r\nlocal x = 1\r\n"
    for text in ("", "dofile('a.lua')", sample, sample.replace("\r\n", "\n"), "x = 1\n\n"):
        p = install_export.patch_text(text)
        assert install_export.BEGIN in p and install_export.patch_text(p) == p, "not idempotent"
        back = install_export.unpatch_text(p)
        if text and not text.endswith("\n"):             # a file without a final newline gains one
            assert back == text + "\n", repr((text, p))
        else:
            assert back == text, repr((text, p))
    L = lua()
    L.execute('PATH = nil; dofile = function(p) PATH = p end\n'
              'lfs = {writedir = function() return "C:\\\\Users\\\\x\\\\Saved Games\\\\DCS\\\\" end}\n'
              'log = {write = function() end, ERROR = 1}')
    L.execute(install_export.patch_text(sample))
    assert L.globals().PATH == "C:\\Users\\x\\Saved Games\\DCS\\Scripts\\bvr_rl\\bvr_stick_export.lua", L.globals().PATH
    with tempfile.TemporaryDirectory() as d:
        user = os.path.join(d, "DCS")
        os.makedirs(os.path.join(user, "Scripts"))
        export = os.path.join(user, "Scripts", "Export.lua")
        with open(export, "w", newline="") as fh:
            fh.write(sample)
        install_export.main(["--user-dir", user])
        assert os.path.isfile(os.path.join(user, "Scripts", "bvr_rl", "bvr_stick_export.lua"))
        assert install_export.BEGIN in open(export, newline="").read()
        assert open(export + install_export.BACKUP_SUFFIX, newline="").read() == sample
        install_export.main(["--user-dir", user, "--check"])
        install_export.main(["--user-dir", user, "--undo"])
        assert open(export, newline="").read() == sample
        assert not os.path.exists(os.path.join(user, "Scripts", "bvr_rl"))
    print("  install_export.py ........... OK  (block in and out byte for byte, loads the right file, install/undo)")


if __name__ == "__main__":
    if lua() is None:
        print("lupa is not installed (pip install lupa): skipped")
        sys.exit(0)
    test_lua_export()
    test_precision_and_command_defs()
    test_click()
    test_ind()
    test_radar_markers()
    test_udp_link()
    test_fake_end_to_end()
    test_fake_nonlinear()
    test_roll_lead()
    test_cockpit()
    test_g_loop()
    test_manoeuvre()
    test_sim_g_lag()
    test_course()
    test_fire_delay()
    test_stall_message()
    test_install()
    print("17/17 passed")
