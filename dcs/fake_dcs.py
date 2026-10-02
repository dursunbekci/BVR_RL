"""
fake_dcs.py  —  a stand-in for DCS World and bvr_bridge.lua, for testing without DCS
=====================================================================================

Flies a 1v1 in BVR_RL's own simulator and talks to dcs_live.py exactly as
bvr_bridge.lua does: it sends the fight as bvr_logger.lua lines over UDP and
flies the blue aircraft on the CMD lines it receives. Red is a scripted
opponent. Use it to check the whole chain on a computer without DCS:

    terminal 1:  python dcs/fake_dcs.py --speed 4
    terminal 2:  python dcs_live.py models_bvr/latest.zip

What the real bridge and DCS add that this does not: DCS's flight models and
AI, the AI's delay before a requested launch, and DCS's own radar and missile
guidance. A policy that does well here and badly in DCS points at those.

--speed 4 runs four mission seconds per real second (DCS runs at 1, or
faster with time acceleration). dcs_live.py keeps up at 10 or more on most
computers; if it cannot, its commands arrive late, as they would in DCS.
"""

import argparse
import json
import math
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bvr_env import BvrEnv, DEG2RAD                      # noqa: E402
from bvr_opponents import BvrOpponentType                # noqa: E402
from bvr_library import DEFAULT_PLATFORM                 # noqa: E402
from dcs_world import FORMAT, sim_frame, sim_ammo, sim_events   # noqa: E402

PORT_TO_LIVE = 15301
PORT_FROM_LIVE = 15302


class FakeDcs:
    RATE = 0.1            # s of mission time between samples
    T_MISSION0 = 100.0    # mission time at the start
    AFTER_END_S = 20.0    # mission seconds flown on after the fight is decided

    def __init__(self, opponent="SHOOTER", seed=0, speed=4.0, platform=DEFAULT_PLATFORM,
                 opp_platform=DEFAULT_PLATFORM, names=("BLUE-1", "RED-1"), host="127.0.0.1",
                 port_out=PORT_TO_LIVE, port_in=PORT_FROM_LIVE, fire_delay=1.0,
                 max_time=900.0, log=print):
        self.env = BvrEnv(opponent_type=BvrOpponentType[opponent], seed=seed, platform=platform,
                          opponent_platform=opp_platform, doctrine="AGGRESSIVE")
        self.names, self.speed, self.fire_delay = names, float(speed), float(fire_delay)
        self.max_time, self.log = float(max_time), log
        self.host, self.port_out = host, port_out
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind((host, port_in))
        self.rx.setblocking(False)
        self.commands = []             # every CMD received: (mission t, seq, hdg, alt, spd, fire)

    def send(self, obj):
        self.tx.sendto(json.dumps(obj).encode(), (self.host, self.port_out))

    def T(self):
        return round(self.T_MISSION0 + self.env._world.t_sim, 3)

    def run(self, stop_evt=None):
        env = self.env
        w = env._world
        events = []
        orig_step = w.step

        def step(c1, c2):                 # collect each world step's events as DCS would send them
            tlm = orig_step(c1, c2)
            events.extend(sim_events(w, self.names, self.T()))
            return tlm
        w.step = step

        env.reset()
        events.clear()
        s = env._state
        hold = (float(s.get("psi", 0.0)), float(s.get("alt", 9000.0)), float(s.get("speed", 280.0)))
        cmd, controlled, last_seq, fire_at, fired_seq = hold, False, -1, None, None
        header = {"format": FORMAT, "rate": self.RATE, "t0": self.T(), "theatre": "fake",
                  "bridge": 1, "agent": self.names[0], "red": self.names[1]}
        self.send(header)
        for i in (1, 2):
            self.send(sim_ammo(w, self.names, i, self.T()))
        self.send(sim_frame(w, self.names, self.T()))
        t_end = None
        wall0, t0 = time.perf_counter(), w.t_sim
        t_resend = w.t_sim
        try:
            while stop_evt is None or not stop_evt.is_set():
                # commands
                while True:
                    try:
                        data, _ = self.rx.recvfrom(4096)
                    except (BlockingIOError, ConnectionResetError):
                        break
                    msg = data.decode("ascii", "replace").split()
                    if not msg:
                        continue
                    if msg[0] == "STOP":
                        controlled = False
                        st = env._state
                        hold = (float(st.get("psi", 0)), float(st.get("alt", 9000)),
                                float(st.get("speed", 280)))
                        continue
                    if msg[0] != "CMD" or len(msg) < 6:
                        continue
                    seq, hdg, alt, spd, fire = int(msg[1]), float(msg[2]), float(msg[3]), \
                        float(msg[4]), int(msg[5])
                    if seq <= last_seq:
                        continue
                    last_seq, controlled = seq, True
                    cmd = (hdg * DEG2RAD, alt, spd)
                    self.commands.append((self.T(), seq, hdg, alt, spd, fire))
                    if fire and fire_at is None and fired_seq != seq:
                        fired_seq, fire_at = seq, w.t_sim + self.fire_delay
                        self.send({"ev": "fire", "t": self.T(), "status": "requested", "seq": seq})
                hdg, alt, spd = cmd if controlled else hold
                pkt = env._encode_cmd(0, 2, 0, 0)
                pkt.update({"hdgCmd": hdg % (2 * math.pi), "altTarget": alt, "V": spd})
                fire = fire_at is not None and w.t_sim >= fire_at
                n_before = env._shots_fired
                env._advance(self.RATE, pkt, fire=fire and w.wpn[0] > 0 and w.alive[1])
                if fire:
                    status = "launched" if env._shots_fired > n_before else "refused"
                    self.send({"ev": "fire", "t": self.T(), "status": status, "seq": fired_seq})
                    fire_at = None
                if w.t_sim - t_resend >= 2.0:       # as the bridge: for a late listener
                    t_resend = w.t_sim
                    self.send(header)
                    for i in (1, 2):
                        if w.alive[i - 1]:
                            self.send(sim_ammo(w, self.names, i, self.T()))
                for e in events:
                    self.send(e)
                events.clear()
                self.send(sim_frame(w, self.names, self.T()))
                # The fight is decided: fly on a little, then end the mission.
                if t_end is None and (env._done or w.t_sim >= self.max_time):
                    t_end = w.t_sim + self.AFTER_END_S
                if t_end is not None and w.t_sim >= t_end:
                    break
                if self.speed > 0:
                    lag = (w.t_sim - t0) / self.speed - (time.perf_counter() - wall0)
                    if lag > 0:
                        time.sleep(lag)
            self.send({"ev": "mission_end", "t": self.T()})
        finally:
            w.step = orig_step
        return env._outcome

    def close(self):
        self.tx.close(); self.rx.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opponent", default="SHOOTER",
                    choices=[t.name for t in BvrOpponentType if t != BvrOpponentType.SELF_PLAY])
    ap.add_argument("--platform", default=DEFAULT_PLATFORM)
    ap.add_argument("--opp-platform", default=DEFAULT_PLATFORM)
    ap.add_argument("--speed", type=float, default=4.0, help="mission seconds per real second")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--missions", type=int, default=1, help="missions to fly one after another")
    args = ap.parse_args()
    for i in range(args.missions):
        fake = FakeDcs(opponent=args.opponent, seed=args.seed + i, speed=args.speed,
                       platform=args.platform, opp_platform=args.opp_platform)
        print(f"fake DCS mission {i + 1}: {fake.names[0]} v {fake.names[1]} ({args.opponent}), "
              f"sending to UDP {PORT_TO_LIVE}, listening on {PORT_FROM_LIVE}")
        out = fake.run()
        print(f"  simulator outcome: {out}, {len(fake.commands)} commands received")
        fake.close()
        time.sleep(2.0)


if __name__ == "__main__":
    main()
