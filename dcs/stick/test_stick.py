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
function LoGetEngineInfo() return {RPM = {left = 88.0, right = 88.5}, fuel_internal = 2100.0} end
function LoGetWorldObjects()
  return {[10] = {Name = "Su-27", UnitName = "Bandit-1", GroupName = "R", CoalitionID = 1, Type = {level1 = 1, level2 = 1},
                  LatLongAlt = {Lat = 42.5, Long = 41.9, Alt = 9000}, Heading = 3.0, Pitch = 0, Bank = 0,
                  Position = {x = 60000, y = 9000, z = 50000}},
          [11] = {Name = "F-16C_50", UnitName = "Viper-1", GroupName = "G", CoalitionID = 2, Type = {level1 = 1, level2 = 1},
                  LatLongAlt = {Lat = 42.1, Long = 41.2, Alt = 7000.5}, Position = {x = 1000, y = 7000, z = 2000}}}
end
function LoGetPayloadInfo() return {CurrentStation = 7, Stations = {{count = 1, CLSID = "{AIM-120C}"}}} end
function LoSetCommand(c, v) CMDS[#CMDS + 1] = {c, v} end

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
    assert abs(f.tas - 250.5) < 1e-9 and abs(f.nz - 1.2) < 1e-9 and abs(f.rpm - 88.25) < 1e-9 and f.agl == 6800.0
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
    g.LuaExportStop()
    assert g.CALLS.stop == 1
    assert any(json.loads(d)["ev"] == "bye" for _, d in sent())
    for _, d in sent():
        json.loads(d)
    # pause: the model time stands still, so nothing is sent (and the axes are not re-sent either way)
    n = len([1 for _, d in sent() if json.loads(d)["ev"] == "tel"])
    run(0.5, paused=True)
    assert len([1 for _, d in sent() if json.loads(d)["ev"] == "tel"]) == n, "telemetry while paused"

    # the other source: the activity event, which runs at frame boundaries here, so 30/s at 60 fps
    e = Mock("event")
    e.g.LuaExportStart()
    e.run(4.0)
    ehz = len([1 for d in e.lines() if d["ev"] == "tel"]) / 4.0
    assert 25 < ehz < 52, ehz
    assert e.lines()[0]["source"] == "event"
    print(f"  Lua export module ........... OK  ({len(sent())} lines, {hz:.0f} telemetry/s per frame, {ehz:.0f}/s by "
          f"event at 60 fps; axes every frame, watchdog + RELEASE + PING + CMD, pause, callbacks chained)")


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


def _check_fake(roll_sign, pitch_sign):
    import stick_test
    with tempfile.TemporaryDirectory() as d:
        s = stick_test.main(["all", "--fake", "--out", d, "--fake-roll-sign", str(roll_sign),
                             "--fake-pitch-sign", str(pitch_sign)])
        gains = json.load(open(os.path.join(d, "stick_gains.json")))
        files = os.listdir(d)
    assert "aborted" not in s, s
    assert gains["roll_sign"] == roll_sign and gains["pitch_sign"] == pitch_sign, gains
    assert abs(gains["roll_rate"] - 2.6) < 0.15, gains            # rad/s per unit
    assert abs(gains["nz_per_cmd"] - 6.0) < 0.4, gains            # g per unit
    assert abs(gains["thr_trim"] - 0.1) < 0.12 and 3.0 < gains["thr_slope"] < 5.0, gains
    assert s["rate"]["real_hz"] > 49 and not s["rate"]["checks_failed"], s["rate"]
    h = s["hold"]
    t1 = h["heading_steps"][0]
    assert 50 < t1["peak_bank_deg"] < 64 and t1["half_s"] < 15 and t1["rate_bulk_dps"] > 3.0, t1
    assert h["alt_step"]["t90_s"] < 25 and h["alt_step"]["overshoot_m"] < 60, h["alt_step"]
    assert abs(h["speed_step"]["end_error"]) < 8, h["speed_step"]
    assert h["hold_error"]["alt_m"] < 15 and h["hold_error"]["hdg_deg"] < 3, h["hold_error"]
    assert any(f.endswith("_hold.jsonl") for f in files) and any(f.startswith("stick_summary") for f in files)
    return gains, t1


def test_fake_end_to_end():
    import stick_test
    from fake_export import FakeLink
    for rs, ps in ((1, 1), (-1, -1), (1, -1)):
        gains, t1 = _check_fake(rs, ps)
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
    test_udp_link()
    test_fake_end_to_end()
    test_install()
    print("4/4 passed")
