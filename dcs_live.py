"""
dcs_live.py  —  fly a trained 1v1 policy in DCS World
=====================================================

Talks to dcs/bvr_bridge.lua over UDP on this computer. Each second of mission
time it builds the policy's observation from what DCS reports (with BVR_RL's
own radar and track model running on DCS's true positions, as in training),
asks the policy for an action and sends DCS the resulting heading, altitude,
speed and fire command. See dcs/README.md for the whole setup.

    python dcs_live.py models_bvr/latest.zip                 # fly
    python dcs_live.py models_bvr/latest.zip --shadow        # only watch: the DCS AI flies
    python dcs_live.py models_bvr/latest.zip --episodes 5    # restart the mission between them

Start it before or after the mission; it waits for DCS. When an episode ends
(a kill, a loss, the step limit) it gives the aircraft back to the DCS AI and
waits for the mission to be restarted, until --episodes are done.

Every episode leaves, in --out (default dcs_runs/):
    <name>.jsonl         everything DCS sent, in bvr_logger.lua's format
                         (dcs_obs_check.py reads it)
    <name>_steps.csv     each decision: time, action, command sent, all inputs
    results.csv          one row per episode
"""

import argparse
import csv
import json
import math
import os
import socket
import sys
import time

import numpy as np

from bvr_env import BvrEnv, OBS_LABELS, HDG_OFFSETS_DEG, ALT_DELTAS_M, RAD2DEG
from bvr_opponents import BvrOpponentType
from dcs_world import (DcsLiveRecording, DcsReplayWorld, capture_raw_obs, FORMAT)

PORT_FROM_DCS = 15301
PORT_TO_DCS = 15302


class UdpLink:
    """The UDP pair to bvr_bridge.lua: JSON lines in, command lines out."""

    def __init__(self, host="127.0.0.1", port_in=PORT_FROM_DCS, port_out=PORT_TO_DCS):
        self.host, self.port_out = host, port_out
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        try:
            self.rx.bind((host, port_in))
        except OSError as e:
            raise SystemExit(f"cannot listen on UDP port {port_in} ({e}). Is another "
                             f"dcs_live.py running?") from e
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sink = None              # callable(raw line) for every datagram

    def poll(self, timeout=0.2) -> list:
        """Everything waiting (up to `timeout` s for the first datagram), parsed."""
        out = []
        self.rx.settimeout(timeout)
        while True:
            try:
                data, _ = self.rx.recvfrom(65536)
            except (socket.timeout, BlockingIOError):
                break
            except ConnectionResetError:          # Windows: an earlier send found no listener
                continue
            line = data.decode("utf-8", "replace").strip()
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if self.sink:
                self.sink(line)
            self.rx.settimeout(0.0)
        return out

    def send(self, text: str):
        self.tx.sendto(text.encode("ascii"), (self.host, self.port_out))

    def close(self):
        self.rx.close(); self.tx.close()


class DcsLiveWorld(DcsReplayWorld):
    """DcsReplayWorld fed live: step() waits for DCS to reach the next frame's
    time, and passes the policy's commands on to the bridge."""

    RESEND_S = 1.0          # repeat the command this often (UDP can drop one)
    BEHIND_WARN_S = 2.0     # warn when DCS's data is this far ahead of the policy

    def __init__(self, link, rec, blue, red, blue_platform=None, red_platform=None,
                 shadow=False, log=print, support_rule="training", speed_boost=None,
                 eta_lock=None, auto_defend=False, hot_turn=False):
        super().__init__(rec, blue, red, blue_platform, red_platform, t_start=rec.t_end)
        self.t_end = math.inf
        self.link, self.shadow, self.log = link, bool(shadow), log
        self.seq = 0
        self.sent = None                     # (hdg deg, alt, spd) last sent
        self.t_sent = -math.inf
        self.fire_pending = None             # world time a fire request went out
        self.fire_log = []                   # (t_sim, status)
        self.mission_ended = False
        self.last_cmd = None
        self.t_behind_warned = -math.inf
        # The support rule (see _check_support): "training" applies the
        # simulator's, "dcs" leaves every missile to DCS.
        self.support_rule = support_rule
        # --speed-boost: while BLUE-1 is well below the policy's speed, ask DCS for
        # this much more (the DCS AI on a route will not light its afterburner for
        # the speed it is actually asked for); the real speed again once close.
        self.speed_boost = None if speed_boost is None else float(speed_boost)
        self._boosting = False
        self.boost_steps = 0     # decisions with the boost on (counted by run_episode)
        # --eta-lock: the bridge gives the route's points locked arrival times
        # from the commanded speed; a DCS AI uses afterburner to make a time.
        self.eta_lock = eta_lock
        self.eta_ack = None      # the bridge's confirmation of OPT eta
        # --auto-defend (or a platform with AUTO_DEFEND): while a missile is
        # inbound at BLUE-1 the bridge hands it to the DCS AI, which defends it
        # at its full agility; the policy's commands resume when it is gone.
        self.auto_defend = bool(auto_defend)
        self.defend_ack = None   # the bridge's confirmation of OPT autodefend
        self.t_defend_opt = -math.inf
        self.defending = False
        self.defences = 0        # times the DCS AI took over
        # --hot-turn (or a platform with HOT_TURN_BANK): a command toward red,
        # with BLUE-1 far off it, is flown as a guns-only attack (7 deg/s, not
        # the route's 1.6) until BLUE-1 is within 10 deg of it.
        self.hot_turn = bool(hot_turn)
        self.hotturn_ack = None
        self.t_hotturn_opt = -math.inf
        self.hot_turning = False
        self.hot_turns = 0       # turns flown as an attack
        self._supported = {}                 # missile id -> last time it had support
        self._destroy_sent = {}              # missile id -> last DESTROY sent
        self.support_log = []                # (t_sim, missile id, shooter, status)

    @property
    def finished(self) -> bool:
        return self.mission_ended and self.t >= self.rec.t_end - 1e-9

    def pump(self, until_t=None, timeout=0.2):
        """Read DCS until it has data up to until_t (or whatever is waiting, without it)."""
        if until_t is not None and self.rec.t_end >= until_t - 1e-9:
            return
        waited, notice = 0.0, 10.0
        while True:
            objs = self.link.poll(timeout if until_t is not None else 0.0)
            if objs:
                self._take(objs)
                waited = 0.0
            else:
                waited += timeout
            if until_t is None or self.rec.t_end >= until_t - 1e-9 or self.mission_ended:
                return
            if waited >= notice:
                self.log(f"waiting for DCS (mission time {self.rec.t_end - self.t0:.1f} s): "
                         f"is it paused?")
                notice += 30.0

    def _take(self, objs):
        for d in objs:
            if d.get("ev") == "fire":
                st = d.get("status")
                self.fire_log.append((round(float(d["t"]) - self.t0, 1), st))
                if st in ("launched", "timeout", "refused"):
                    self.fire_pending = None
                if st != "launched" and st != "requested":
                    self.log(f"  fire {st} at {float(d['t']) - self.t0:.1f} s"
                             + (f": {d.get('reason')}" if d.get("reason") else ""))
            elif d.get("ev") == "bridge" and d.get("status") == "eta":
                self.eta_ack = d.get("clock")
            elif d.get("ev") == "bridge" and d.get("status") == "opt":
                if "autodefend" in d:
                    self.defend_ack = bool(d.get("autodefend"))
                if "hotturn" in d:
                    self.hotturn_ack = bool(d.get("hotturn"))
            elif d.get("ev") == "bridge" and d.get("status") == "hotturn":
                on = bool(d.get("on"))
                if on and not self.hot_turning:
                    self.hot_turns += 1
                self.hot_turning = on
            elif d.get("ev") == "bridge" and d.get("status") == "defend":
                on = bool(d.get("on"))
                if on and not self.defending:
                    self.defences += 1
                if on != self.defending:
                    self.log(f"  {float(d['t']) - self.t0:.1f} s: " +
                             ("missile inbound: the DCS AI defends BLUE-1" if on else
                              "missile gone: the policy flies again"))
                self.defending = on
            elif d.get("ev") == "support_lost":
                self.support_log.append((round(float(d["t"]) - self.t0, 1), d.get("id"),
                                         d.get("shooter"), d.get("status")))
                if d.get("status") not in ("destroyed", "gone"):
                    self.log(f"  missile {d.get('id')}: could not be removed ({d.get('status')})")
            elif d.get("ev") == "mission_end":
                self.mission_ended = True
        # Frames from before this episode (a restarted mission) are not ours.
        objs = [d for d in objs if "units" not in d or float(d["t"]) >= self.t0 - 1.0]
        self.rec.extend(objs, reader_t=self.t - 1.0)
        behind = self.rec.t_end - self.t
        if behind > self.BEHIND_WARN_S and self.t - self.t_behind_warned > 30.0:
            self.t_behind_warned = self.t
            self.log(f"  this computer is {behind:.1f} s of mission time behind DCS: the policy's "
                     f"commands arrive late (lower DCS time acceleration)")

    BOOST_ON_MPS = 20.0      # start boosting this far below the policy's speed
    BOOST_OFF_MPS = 5.0      # back to the real speed once within this

    def _flown_speed(self):
        try:
            s = self.rec.state(self.names[1], min(self.t, self.rec.t_end))
            return float(np.linalg.norm(s["vel"]))
        except Exception:
            return None

    def _speed_to_send(self, spd):
        if self.speed_boost is None:
            return spd
        v = self._flown_speed()
        if v is not None:
            if v < spd - self.BOOST_ON_MPS:
                self._boosting = True
            elif v >= spd - self.BOOST_OFF_MPS:
                self._boosting = False
        return max(spd, self.speed_boost) if self._boosting else spd

    def send_command(self, cmd, fire):
        hdg = (math.degrees(float(cmd["hdgCmd"])) % 360.0)
        alt, spd = float(cmd["altTarget"]), self._speed_to_send(float(cmd["V"]))
        self.last_cmd = (round(hdg, 2), round(alt, 1), round(spd, 1))
        if self.shadow:
            return
        new = self.sent is None or self.last_cmd != self.sent
        if not (new or fire or self.t - self.t_sent >= self.RESEND_S):
            return
        self.seq += 1
        if self.eta_lock:            # sent with every command: UDP may lose one
            self.link.send(f"OPT eta {self.eta_lock}")
        # Until the bridge confirms it, with every command; then now and then.
        if self.auto_defend and (not self.defend_ack or self.t - self.t_defend_opt >= 10.0):
            self.link.send("OPT autodefend 1")
            self.t_defend_opt = self.t
        if self.hot_turn and (not self.hotturn_ack or self.t - self.t_hotturn_opt >= 10.0):
            self.link.send("OPT hotturn 1")
            self.t_hotturn_opt = self.t
        # The trailing mission time is ignored by the bridge; fake_dcs.py --lockstep
        # uses it to wait for us.
        self.link.send(f"CMD {self.seq} {hdg:.2f} {alt:.1f} {spd:.1f} {1 if fire else 0} {self.t:.1f}")
        self.sent, self.t_sent = self.last_cmd, self.t
        if fire:
            self.fire_pending = self.t

    def stop(self):
        if not self.shadow:
            for _ in range(3):
                self.link.send("STOP")

    def step(self, cmd1: dict, cmd2: dict) -> dict:
        if cmd1:
            self.send_command(cmd1, bool(cmd1.get("fire", 0)))
        if self.fire_pending is not None and self.t - self.fire_pending > 20.0:
            self.fire_pending = None            # the bridge never answered
        t_new = round(self.t + self.SIM_DT, 6)       # no drift (DcsReplayWorld.step)
        self.pump(until_t=t_new)
        self.t = min(t_new, self.rec.t_end) if self.mission_ended else t_new
        self.events = self._events_between(self._t_prev, self.t)
        self._t_prev = self.t
        tlm = self.telemetry()                 # also latches which seekers are active
        self._check_support(cmd1, cmd2)
        return tlm

    def telemetry(self, me: int = 1) -> dict:
        tlm = super().telemetry(me)
        if me == 1:
            tlm["defending"] = int(self.defending)    # holds the policy's fire (BvrEnv._can_fire)
        return tlm

    def _check_support(self, cmd1, cmd2=None):
        """The simulator's support rule, applied to DCS missiles: a missile whose
        own seeker has not taken over yet misses after the missile's support
        timeout (3 s) without guidance from its shooter. As in training:
          the policy's missiles are supported while its radar track gives a
            guidance estimate (BvrEnv._guidance_packet) and it is alive;
          the opponent's while its own radar track (the same model, run for it
            on DCS's positions) gives one and it is alive, as for a scripted
            opponent in training since SIM_REV 9.
        A missile past its timeout is removed in DCS (DESTROY), which counts as a
        SUPPORT_LOST miss. Seeker take-over is estimated, as for the inputs:
        inside the missile's mean hand-off range of its target."""
        if self.shadow or self.support_rule != "training":
            return
        blue, red = self.names[1], self.names[2]
        guided = bool(cmd1) and bool((cmd1.get("msl_guidance") or {}).get("valid", 0))
        # Red: its own radar and track (BvrEnv's opponent radar, run on DCS's
        # true positions) when the env set it up, as for a scripted opponent in
        # training since SIM_REV 9; without one, alive is enough.
        g2 = (cmd2 or {}).get("msl_guidance")
        red_guided = True if g2 is None else bool(g2.get("valid", 0))
        ok = {1: guided and self.rec.alive(blue, self.t),
              2: red_guided and self.rec.alive(red, self.t)}
        for d, _tgt, owner in self._in_flight():
            wid = d["id"]
            if wid in self._seeker:
                continue
            last = self._supported.setdefault(wid, self.t)
            if ok[owner]:
                self._supported[wid] = self.t
                continue
            plat = self.plats.get(d["shooter"])
            timeout = getattr(getattr(plat, "missile", None), "SUPPORT_TIMEOUT", 3.0)
            if self.t - last <= timeout or self.t - self._destroy_sent.get(wid, -math.inf) < 1.0:
                continue
            if wid not in self._destroy_sent:
                why = "the policy's radar lost the track" if owner == 1 and self.rec.alive(blue, self.t) \
                    else f"{d['shooter']} is down"
                self.log(f"  missile {wid} ({d['shooter']}): no support for {timeout:.0f} s "
                         f"({why}): removed in DCS")
            self.link.send(f"DESTROY {wid}")
            self._destroy_sent[wid] = self.t


class DcsLiveEnv(BvrEnv):
    """BvrEnv whose world is DCS, live. The observation code runs unchanged."""

    # No altitude floor: in training, below 300 m counts as hitting the ground.
    # In DCS an aircraft can fly lower (the AI dives to the sea to defend) and
    # DCS itself reports a real crash, which counts as one.
    MIN_ALT = -1e9

    def __init__(self, world_factory, platform, opponent_platform, max_steps=None,
                 privileged_critic=True, doctrine="BALANCED", seed=0):
        super().__init__(opponent_type=BvrOpponentType.STRAIGHT, seed=seed, platform=platform,
                         opponent_platform=opponent_platform, privileged_critic=privileged_critic,
                         doctrine=doctrine)
        if max_steps is not None:
            self.MAX_STEPS = int(max_steps)
        self._world = world_factory(self._plat, self._opp_plat)

    def _random_ic(self, blue_top_speed=None) -> dict:
        return self._world.initial_ic()

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        w = self._world
        # Red's own radar and track, for the support rule on its missiles.
        if w.support_rule == "training" and not w.shadow and self._radar_model == "sim":
            self._opp_radar = self._opponent_radar()
        return out

    def _can_fire(self) -> bool:
        # One shot at a time: the DCS AI takes a moment to launch, and the
        # launch (not the request) is what starts the 3 s between shots.
        if self._world.fire_pending is not None:
            return False
        return super()._can_fire()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if not (terminated or truncated) and self._world.finished:
            truncated = True
            info["terminal_outcome"] = self._outcome or "MISSION_END"
            self._ready = False
        return obs, reward, terminated, truncated, info


# ────────────────────────────────────────────────────────────────────
def wait_for_fight(link, agent=None, red=None, after_t=None, log=print):
    """Read DCS until a sample holds both aircraft. Returns (recording, agent, red).
    after_t: the last mission time of the previous episode. Its mission keeps
    sending after the episode; only a restarted one (time going back, or after
    a mission end) starts the next episode."""
    rec = DcsLiveRecording()
    header, pending = None, []
    need_restart = after_t is not None
    t_wait, notice = time.time(), 0.0
    while True:
        objs = link.poll(0.5)
        for d in objs:
            if d.get("format") == FORMAT:
                header = d
            elif d.get("ev") == "mission_end":
                need_restart, after_t, pending = False, None, []
        if need_restart:
            # Keep only what the restarted mission sent: earlier mission times.
            objs = [d for d in objs if "t" in d and float(d["t"]) < after_t - 1.0]
            if not any("units" in d for d in objs):
                objs = []
            else:
                need_restart = False
        if not need_restart and objs:
            pending += [d for d in objs if "format" not in d]
            if header is not None and any("units" in d for d in pending):
                rec.extend([header] + pending)
                pending = []
                units = rec.units
                a = agent or header.get("agent") or next(
                    (n for n, u in units.items() if u["coal"] == 2), None)
                r = red or header.get("red") or next(
                    (n for n, u in units.items() if u["coal"] == 1), None)
                if a in units and r in units:
                    return rec, a, r
        if time.time() - t_wait >= notice:
            what = ("the mission to be restarted" if need_restart else
                    "DCS (start the mission that runs bvr_bridge.lua)" if header is None else
                    "both aircraft")
            log(f"waiting for {what} ...")
            notice += 30.0


# Bridge version that understands OPT eta (--eta-lock).
ETA_BRIDGE = 2
# ... and OPT autodefend (--auto-defend) as trained: the DCS AI takes over
# once the missile is within 15 km (4 and 5 took over at launch).
DEFEND_BRIDGE = 6
# ... and OPT hotturn (--hot-turn).
HOTTURN_BRIDGE = 5

# A decision counts as "short" when BLUE-1 flies this much below its commanded speed.
SPEED_SHORT_MPS = 30.0


def _append_result(path, row):
    """Append a row to results.csv; an older file with fewer columns is rewritten
    with the new ones (left empty for its rows)."""
    rows, fields = [], list(row)
    if os.path.exists(path):
        with open(path, newline="") as fh:
            r = csv.DictReader(fh)
            old = r.fieldnames or []
            rows = list(r)
        fields = old + [k for k in row if k not in old]
        if old != fields:
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader(); w.writerows(rows)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def run_episode(link, model, args, scen, rec, agent, red, ep, log=print):
    from bvr_selfplay import FrameStacker, N_STACK
    from gymnasium import spaces
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"live_{stamp}_ep{ep}" + ("_shadow" if args.shadow else "")
    os.makedirs(args.out, exist_ok=True)
    raw_fh = open(os.path.join(args.out, name + ".jsonl"), "w", encoding="utf-8")
    raw_fh.write(json.dumps(rec.header) + "\n")
    for e in rec.events:
        raw_fh.write(json.dumps(e) + "\n")
    for f in rec._frames:
        raw_fh.write(json.dumps(f) + "\n")
    link.sink = lambda line: raw_fh.write(line + "\n")

    privileged = isinstance(model.observation_space, spaces.Dict)
    rule = getattr(args, "support_rule", "training")
    env = DcsLiveEnv(lambda bp, rp: DcsLiveWorld(link, rec, agent, red, bp, rp,
                                                 shadow=args.shadow, log=log, support_rule=rule,
                                                 speed_boost=getattr(args, "speed_boost", None),
                                                 eta_lock=getattr(args, "eta_lock", None),
                                                 auto_defend=bool(getattr(args, "auto_defend", False))
                                                 or bp.auto_defend,
                                                 hot_turn=bool(getattr(args, "hot_turn", False))
                                                 or bool(getattr(bp, "hot_turn_bank", 0.0))),
                     platform=scen["platform"], opponent_platform=args.opp_platform or scen["opponent_platform"],
                     max_steps=args.max_steps, privileged_critic=privileged, doctrine=args.doctrine)
    world = env._world
    raw = capture_raw_obs(env)
    stacker = FrameStacker(N_STACK)
    obs, info = env.reset()
    sobs = stacker.reset(obs)
    ic = info["ic"]
    log(f"episode {ep}: {agent} v {red}, start range {ic['start_range'] / 1000:.1f} km"
        + ("  [shadow: the DCS AI flies, the policy only watches]" if args.shadow else ""))

    steps_fh = open(os.path.join(args.out, name + "_steps.csv"), "w", newline="")
    wr = csv.writer(steps_fh)
    wr.writerow(["t", "hdg_off", "alt_delta", "speed", "fire", "cmd_hdg_deg", "cmd_alt_m",
                 "cmd_spd_mps", "track", "range_true_m", *OBS_LABELS])
    info = {}
    # The mission's bridge must be new enough for the options asked for.
    bridge_v = int(rec.header.get("bridge", 1) or 1)
    if getattr(args, "eta_lock", None) and bridge_v < ETA_BRIDGE and not args.shadow:
        log(f"  WARNING: this mission's bridge (version {bridge_v}) ignores --eta-lock; rebuild "
            f"the mission (python dcs\\make_mission.py) and open the new file in DCS")
    if world.auto_defend and bridge_v < DEFEND_BRIDGE and not args.shadow:
        log(f"  WARNING: this mission's bridge (version {bridge_v}) "
            + ("cannot defend BLUE-1" if bridge_v < 4 else
               "hands BLUE-1 to the DCS AI at the launch, not within 15 km as trained")
            + " (--auto-defend, or the platform's AUTO_DEFEND); rebuild the mission "
              "(python dcs\\make_mission.py) and open the new file in DCS")
    elif world.auto_defend and not args.shadow:
        log("  auto-defend: the DCS AI defends BLUE-1 while a missile is inbound")
    if world.hot_turn and bridge_v < HOTTURN_BRIDGE and not args.shadow:
        log(f"  WARNING: this mission's bridge (version {bridge_v}) flies every turn on the route "
            f"(--hot-turn, or the platform's HOT_TURN_BANK); rebuild the mission "
            f"(python dcs\\make_mission.py) and open the new file in DCS")
    elif world.hot_turn and not args.shadow:
        log("  hot turns: turns toward red are flown as an attack (about 7 deg/s)")
    defend_steps = hot_steps = 0
    spd = []    # (commanded, flown) m/s each decision: does DCS fly the speed asked for?
    try:
        while True:
            mask = env.action_masks()
            action, _ = model.predict(sobs, deterministic=True, action_masks=mask)
            obs, _, term, trunc, info = env.step(action)
            sobs = stacker.update(obs)
            spd.append((float(env._cmd_spd), float(env._state.get("speed", 0.0))))
            world.boost_steps += int(world._boosting)    # once per decision, not per frame
            defend_steps += int(world.defending)
            hot_steps += int(world.hot_turning)
            a = [int(x) for x in np.asarray(action).reshape(-1)]
            c = world.last_cmd or ("", "", "")
            wr.writerow([round(env._t_sim, 1), HDG_OFFSETS_DEG[a[0]], ALT_DELTAS_M[a[1]],
                         env._plat.speed_cmds[a[2]], int(info.get("fired", False)), *c,
                         info.get("track_state"), round(env._state.get("range", 0.0), 0),
                         *[f"{x:.6g}" for x in raw["obs"]]])
            if env._step_num % 10 == 0 or info.get("fired"):
                log(f"  t={env._t_sim:5.0f}s  range {env._state.get('range', 0) / 1000:5.1f} km  "
                    f"track {info.get('track_state')}  heading {HDG_OFFSETS_DEG[a[0]]:+4.0f}  "
                    f"alt {ALT_DELTAS_M[a[1]]:+5.0f}  speed {env._plat.speed_cmds[a[2]]:.0f}"
                    + ("  FIRE" if info.get("fired") else ""))
            if term or trunc:
                break
    finally:
        world.stop()
        link.sink = None
        raw_fh.close(); steps_fh.close()
    outcome = info.get("terminal_outcome", "?")
    row = {"episode": name, "mode": "shadow" if args.shadow else "live", "agent": agent,
           "red": red, "outcome": outcome, "flight_time_s": round(env._t_sim, 1),
           "start_range_km": round(ic["start_range"] / 1000, 1),
           "shots": info.get("shots_fired", 0),
           "fire_requests": sum(1 for _, s in world.fire_log if s == "requested"),
           "fire_timeouts": sum(1 for _, s in world.fire_log if s in ("timeout", "refused")),
           "missiles_removed": sum(1 for *_, st in world.support_log if st == "destroyed"),
           "support_losses": info.get("support_losses", 0)}
    # How often BLUE-1 flew well below the speed the policy asked for. The DCS
    # AI following a route may not use afterburner; the simulator always does.
    short = [c - f > SPEED_SHORT_MPS for c, f in spd]
    row.update(speed_cmd_mean=round(float(np.mean([c for c, _ in spd])), 1) if spd else "",
               speed_flown_mean=round(float(np.mean([f for _, f in spd])), 1) if spd else "",
               speed_short_pct=round(100.0 * sum(short) / len(short), 1) if short else "",
               speed_boost=world.speed_boost or "", eta_lock=world.eta_lock or "",
               eta_confirmed=(world.eta_ack == world.eta_lock) if world.eta_lock else "",
               bridge_version=bridge_v,
               auto_defend=int(world.auto_defend),
               defences=world.defences if world.auto_defend else "",
               defend_s=round(defend_steps / env.DECISION_HZ, 1) if world.auto_defend else "",
               hot_turn=int(world.hot_turn),
               hot_turns=world.hot_turns if world.hot_turn else "",
               hot_turn_s=round(hot_steps / env.DECISION_HZ, 1) if world.hot_turn else "",
               speed_boosted_pct=round(100.0 * world.boost_steps / len(spd), 1)
               if spd and world.speed_boost else "")
    _append_result(os.path.join(args.out, "results.csv"), row)
    log(f"episode {ep}: {outcome} after {env._t_sim:.0f} s, {row['shots']} shots "
        f"({row['fire_timeouts']} fire requests not answered by a launch)")
    if short and not args.shadow:
        log(f"  speed: asked for {row['speed_cmd_mean']:.0f} m/s on average, flew "
            f"{row['speed_flown_mean']:.0f}; more than {SPEED_SHORT_MPS:.0f} m/s short "
            f"{row['speed_short_pct']:.0f}% of the time"
            + ("  <- DCS is not flying the commanded speed" if row["speed_short_pct"] > 30 else ""))
        if world.eta_lock and world.eta_ack != world.eta_lock:
            log(f"  --eta-lock was NOT confirmed by the bridge (bridge version {bridge_v}): "
                f"DCS flew without locked arrival times; rebuild the mission")
        if world.auto_defend:
            log(f"  auto-defend: the DCS AI defended BLUE-1 {world.defences} time(s), "
                f"{row['defend_s']:.0f} s in all"
                + ("" if world.defend_ack else
                   "  <- the bridge never confirmed OPT autodefend: rebuild the mission"))
        if world.hot_turn:
            log(f"  hot turns: {world.hot_turns} turn(s) toward red flown as an attack, "
                f"{row['hot_turn_s']:.0f} s in all"
                + ("" if world.hotturn_ack else
                   "  <- the bridge never confirmed OPT hotturn: rebuild the mission"))
        if world.speed_boost:
            log(f"  speed boost: asked DCS for {world.speed_boost:.0f} m/s "
                f"{row['speed_boosted_pct']:.0f}% of the time")
    log(f"  written: {os.path.join(args.out, name)}.jsonl, _steps.csv")
    return row, rec.t_end


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="a 1v1 checkpoint (.zip) trained by train_bvr.py")
    ap.add_argument("--shadow", action="store_true",
                    help="never send commands: the DCS AI flies, the policy only watches")
    ap.add_argument("--episodes", type=int, default=1, help="episodes before quitting (default 1)")
    ap.add_argument("--agent", help="DCS unit name the policy flies (default: from the bridge)")
    ap.add_argument("--red", help="DCS unit name of the opponent (default: from the bridge)")
    ap.add_argument("--platform", help="library platform the policy is told it flies, for its own "
                                        "missile's envelopes (default: the checkpoint's); also red's, "
                                        "unless --opp-platform says otherwise")
    ap.add_argument("--opp-platform", help="library platform for red's envelopes "
                                            "(default: the checkpoint's training opponent)")
    ap.add_argument("--support-rule", choices=["training", "dcs"], default="training",
                    help="training (default): a missile without support from its shooter for 3 s "
                         "before its seeker takes over is removed, as in the simulator; dcs: DCS "
                         "decides (its missiles usually fly on and go active)")
    ap.add_argument("--speed-boost", type=float, nargs="?", const=550.0, default=None, metavar="M_S",
                    help="while BLUE-1 is more than 20 m/s below the policy's speed, ask DCS for "
                         "this speed instead (default 550 m/s), to make the DCS AI use afterburner")
    ap.add_argument("--eta-lock", nargs="?", const="mission", default=None, choices=["mission", "abs"],
                    help="give the route's points locked arrival times from the commanded speed, "
                         "so the DCS AI uses afterburner to make them (rebuild the mission first). "
                         "mission (default): times on the mission clock; abs: on the time of day")
    ap.add_argument("--auto-defend", action="store_true",
                    help="while a missile is inbound at BLUE-1 the DCS AI defends it at its full "
                         "agility (beam, dive, chaff), then hands it back to the policy; on by "
                         "default for a platform with AUTO_DEFEND (F-16C-DCSAI-AD). Bridge 4")
    ap.add_argument("--hot-turn", action="store_true",
                    help="a heading command toward red, with BLUE-1 more than 30 deg off it, is "
                         "flown as a guns-only attack (about 7 deg/s, where the route turns at "
                         "1.6) until within 10 deg; on by default for a platform with "
                         "HOT_TURN_BANK (F-16C-DCSAI-AD). Bridge 5")
    ap.add_argument("--doctrine", default="BALANCED")
    ap.add_argument("--max-steps", type=int, help=f"episode length, s (default {BvrEnv.MAX_STEPS})")
    ap.add_argument("--port-in", type=int, default=PORT_FROM_DCS)
    ap.add_argument("--port-out", type=int, default=PORT_TO_DCS)
    ap.add_argument("--out", default="dcs_runs", help="output directory")
    args = ap.parse_args(argv)

    from bvr_compat import load_model
    from bvr_library import scenario_of
    model = load_model(args.checkpoint, env=None, log=print)
    scen = scenario_of(model)
    if scen.get("format", "1v1") != "1v1":
        raise SystemExit("this checkpoint was trained for 2v1; the DCS link is 1v1 only")
    if args.platform:
        # e.g. F-16C-DCS: the same aircraft with the missile fitted to DCS, so
        # the policy's range inputs match the missiles DCS actually flies.
        scen = dict(scen, platform=args.platform)
        args.opp_platform = args.opp_platform or args.platform
    print(f"policy: {args.checkpoint} ({scen['platform']} v "
          f"{args.opp_platform or scen['opponent_platform']}), {args.doctrine} doctrine")
    link = UdpLink(port_in=args.port_in, port_out=args.port_out)
    print(f"listening for DCS on UDP {args.port_in}, commands to {args.port_out}")
    if not args.shadow:
        print("support rule: " + ("as in training (3 s without support before the seeker takes "
                                  "over: the missile is removed)" if args.support_rule == "training"
                                  else "DCS's own"))
    last_t, rows = None, []
    try:
        for ep in range(1, args.episodes + 1):
            rec, agent, red = wait_for_fight(link, args.agent, args.red, after_t=last_t)
            row, last_t = run_episode(link, model, args, scen, rec, agent, red, ep)
            rows.append(row)
            if ep < args.episodes:
                print("restart the mission in DCS for the next episode (Ctrl+C to stop)")
    except KeyboardInterrupt:
        print("\nstopped")
        link.send("STOP")
    finally:
        link.close()
    if rows:
        from collections import Counter
        c = Counter(r["outcome"] for r in rows)
        print("results: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    return rows


if __name__ == "__main__":
    main()
