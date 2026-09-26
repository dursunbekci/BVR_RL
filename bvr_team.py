"""
bvr_team.py  —  2v1 BVR: two blue aircraft on one shared policy
================================================================

One world (sim_world.SimWorld) with two blue aircraft and one red. Each blue
aircraft is an agent; one policy flies both (parameter sharing, as in MAPPO),
each on its own observation. Red is a scripted opponent from bvr_opponents.

Each blue aircraft keeps a BvrEnv as its "observer": that object's radar,
track filter, envelopes, observation, action mask, shaping potential and
command code are the 1v1 agent's, fed this aircraft's telemetry every frame
(the same way bvr_selfplay flies AC2). What this module adds is the team:

  * The datalink. Each observer's _wingman is the other, so it fights on its
    wingman's firm track whenever its own radar has none: one aircraft can
    fire and turn away while the other keeps its missile supported.
  * Team outcomes and rewards. A kill, a loss, a timeout or an escape is
    paid to both agents; each agent's shaping and its shot and heading costs
    stay its own.
  * Red's target. Red engages the nearest blue aircraft, switching only when
    the other is clearly closer, and sees the fight as a 1v1 against it with
    both blue aircraft's missiles counted as inbound.
  * Losing an aircraft. A blue aircraft shot down or crashed leaves the
    fight; its slot keeps running until the episode ends with one legal
    action (so it adds no policy gradient), no shaping, and the team's
    rewards (so its value keeps learning what the team goes on to do).

Episode ends: red destroyed or crashed, both blue aircraft lost, every live
blue aircraft beyond escape range, or the step limit. A kill, or the loss of
both blue aircraft, ends it only once no missile is left in flight at a live
aircraft: red's last missile can still take a blue aircraft with it, and a
lost blue aircraft's locked-on missile can still kill red. While blue flies
on after red's death, shaping stays at its value at the kill; when both blue
are down, the rest is flown out within the step.

Outcome labels reuse the 1v1 names, so trainer and GUI statistics work as
before: KILL red killed with no blue loss · MUTUAL_KILL red killed at the
cost of a blue aircraft · SHOT_DOWN both blue lost, at least one to a
missile · CRASH both blue lost to the ground · BANDIT_CRASH · ESCAPE ·
TIMEOUT. Blue losses are reported separately as well.

Interface (bvr_team_vec wraps it for Stable-Baselines3):
    obs_list, info          = env.reset()
    obs_list, rewards, terminated, truncated, infos = env.step(actions)   # one per agent
    masks                   = env.action_masks()                          # one row per agent
"""

import math

import numpy as np
from gymnasium import spaces

from bvr_env import (BvrEnv, OBS_DIM, PRIV_DIM, ACTION_NVEC, HDG_OFFSETS_DEG, FIRE_OPTIONS,
                     DOCTRINES, DOCTRINE_MIXED, PHI_TERMS, DEG2RAD, R_EARTH, REF_LAT,
                     _wrap_deg)
from bvr_opponents import BvrOpponent, BvrOpponentType
from bvr_track_adapter import TrackState
from bvr_library import DEFAULT_PLATFORM
from sim_world import SimWorld

BLUE, RED = 1, 2


class TeamBvrEnv:

    n_agents = 2
    metadata = {"render_modes": []}
    render_mode = None

    DECISION_HZ  = BvrEnv.DECISION_HZ
    MAX_STEPS    = BvrEnv.MAX_STEPS
    MIN_ALT      = BvrEnv.MIN_ALT
    ESCAPE_RANGE = BvrEnv.ESCAPE_RANGE

    # Team events, paid to both agents when they happen.
    R_KILL = +1.0            # red destroyed
    R_LOST = -0.7            # a blue aircraft shot down
    R_CRASH = -1.0           # a blue aircraft flown into the ground
    R_BANDIT_CRASH = +0.4
    R_TIMEOUT = -0.5
    R_ESCAPE = -0.6
    R_WASTED_MSL = BvrEnv.R_WASTED_MSL     # per own missile that did not kill

    # Red switches target only when the other blue aircraft is this much closer.
    RETARGET_FRAC = 0.8

    def __init__(self, opponent_type=BvrOpponentType.STRAIGHT, gamma_discount=0.997,
                 seed=42, instance_id=0, privileged_critic=True, envelope_table="library",
                 radar_model="sim", doctrine=DOCTRINE_MIXED,
                 blue_platforms=(DEFAULT_PLATFORM, DEFAULT_PLATFORM),
                 red_platform=DEFAULT_PLATFORM):
        if opponent_type == BvrOpponentType.SELF_PLAY:
            raise ValueError("2v1 has no self-play stage yet: red is a scripted opponent")
        if len(blue_platforms) != self.n_agents:
            raise ValueError(f"2v1 needs {self.n_agents} blue platforms, got {len(blue_platforms)}")
        self._opponent_type = opponent_type
        self._gamma = float(gamma_discount)
        self._privileged = bool(privileged_critic)
        self._rng = np.random.default_rng(seed + instance_id * 1000 + 500)
        self._instance = int(instance_id)
        doctrine = str(doctrine).upper()
        if doctrine != DOCTRINE_MIXED and doctrine not in DOCTRINES:
            raise ValueError(f"doctrine {doctrine!r}: choose {DOCTRINE_MIXED} or {', '.join(DOCTRINES)}")
        self._doctrine_cfg = doctrine
        self._blue_ids = tuple(blue_platforms)
        self._red_id = red_platform

        self._obs = [BvrEnv(opponent_type=BvrOpponentType.STRAIGHT, gamma_discount=gamma_discount,
                            seed=seed + 97 * k, instance_id=instance_id,
                            privileged_critic=privileged_critic, envelope_table=envelope_table,
                            radar_model=radar_model, doctrine="AGGRESSIVE",
                            platform=p, opponent_platform=red_platform)
                     for k, p in enumerate(blue_platforms)]
        a, b = self._obs
        a._wingman, b._wingman = b, a

        self._world = SimWorld(seed=seed + instance_id * 1000,
                               platforms=[o._plat for o in self._obs] + [a._opp_plat],
                               teams=[BLUE, BLUE, RED])
        self.RED = 3                              # red's aircraft number in the world
        self.observation_space = a.observation_space
        self.action_space = a.action_space
        self._opponent_factory = None             # callable(env) -> opponent, for evaluation
        self._opponent = None
        self._ready = False
        self._doctrine = "AGGRESSIVE"
        self.resolve_hook = None                  # callable(env), each second flown out

    # ── episode ───────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        w, (a, b) = self._world, self._obs
        self._step_num = 0
        self._t_sim = 0.0
        self._outcome = None
        self._lost = []                   # (agent index, "SHOT_DOWN" | "CRASH")
        self._red_dead = None             # None | "KILL" | "CRASH"
        self._killer = None
        self._events_seen = set()
        d = self._doctrine_cfg
        if d == DOCTRINE_MIXED:
            names = list(DOCTRINES)
            d = names[int(self._rng.integers(len(names)))]
        self._doctrine = d
        for o in self._obs:
            _observer_reset(o)
            o._doctrine, o._shot_cost = d, float(DOCTRINES[d])

        ic = self._random_ic()
        self._episode_id = int(self._rng.integers(1, 2_000_000_000))
        for o in self._obs:
            o._episode_id = self._episode_id
        if self._opponent_factory is not None:
            self._opponent = self._opponent_factory(self)
        else:
            self._opponent = BvrOpponent.create(self._opponent_type, rng=self._rng)
        # Scripted opponents read their start from the 1v1 key names.
        self._opponent.reset({**ic, "ac2_psi": ic["ac3_psi"], "ac2_alt": ic["ac3_alt"],
                              "ac2_spd": ic["ac3_spd"]})
        w.reset(ic, episode_id=self._episode_id)
        self._red_tgt = 1
        for k, o in enumerate(self._obs, start=1):
            o._cmd_hdg = float(ic[f"ac{k}_psi"]); o._cmd_alt = float(ic[f"ac{k}_alt"])
            o._cmd_spd = float(ic[f"ac{k}_spd"])

        cmds = [o._encode_cmd(0, 2, 2, 0) for o in self._obs]
        self._advance(0.5, cmds, [False, False])
        self._ready = True
        obs = [self._agent_obs(k) for k in range(self.n_agents)]
        self._prev_phi = [self._phi(k) for k in range(self.n_agents)]
        self._prev_terms = [dict(o._phi_terms) for o in self._obs]
        return obs, {"ic": ic, "episode_id": self._episode_id}

    def step(self, actions):
        if not self._ready:
            raise RuntimeError("step() before reset()")
        acts = np.asarray(actions, dtype=np.int64).reshape(self.n_agents, -1)
        self._step_num += 1
        cmds, fires, hdg_cost, shots_before = [], [], [], []
        for k, o in enumerate(self._obs):
            o._step_num = self._step_num
            i_hdg, i_alt, i_spd, i_fire = (int(x) for x in acts[k])
            if o._alive:
                fire = bool(FIRE_OPTIONS[i_fire]) and o._can_fire()
                cmds.append(o._encode_cmd(i_hdg, i_alt, i_spd, i_fire))
                off = HDG_OFFSETS_DEG[i_hdg]
                hdg_cost.append(o.W_HDG_CHANGE * abs(_wrap_deg(off - o._prev_hdg_off)) / 180.0)
                o._prev_hdg_off = off
            else:
                fire = False
                cmds.append(None)
                hdg_cost.append(0.0)
            fires.append(fire)
            shots_before.append(o._shots_fired)

        # After red's death blue flies on only to survive red's last missiles;
        # the potential stays at its value at the kill (see the module notes).
        flying = [o._alive and self._red_dead is None for o in self._obs]
        team_r = self._advance(1.0 / self.DECISION_HZ, cmds, fires)

        obs, rewards, infos = [], [], []
        terminated = self._outcome is not None
        truncated = (not terminated) and self._step_num >= self.MAX_STEPS
        if truncated:
            # A result waiting only on missiles still in flight stands.
            self._outcome = self._pending_outcome()
            if self._outcome is None:
                self._outcome = "TIMEOUT"
                team_r += self.R_TIMEOUT
        for k, o in enumerate(self._obs):
            obs.append(self._agent_obs(k))
            if flying[k]:
                # Lost during this step: shaped on its last real state, as a
                # 1v1 episode's final step is. Zeroing the potential instead
                # would pay an aircraft lost deep in the enemy's envelope
                # (negative potential) a bonus for dying.
                phi = o._potential()
                shaping = self._gamma * phi - self._prev_phi[k]
                terms = o._phi_terms
                shaping_terms = {t: self._gamma * terms[t] - self._prev_terms[k][t] for t in PHI_TERMS}
            else:
                phi, shaping = self._prev_phi[k], 0.0
                terms = self._prev_terms[k]
                shaping_terms = {t: 0.0 for t in PHI_TERMS}
            self._prev_phi[k] = phi
            self._prev_terms[k] = dict(terms)
            shot_cost = o._shot_cost * (o._shots_fired - shots_before[k])
            r = float(shaping) - hdg_cost[k] - shot_cost + team_r
            info = {"shaping": shaping, "phi": phi, "step": self._step_num,
                    "phi_terms": dict(terms), "shaping_terms": shaping_terms,
                    "cost_terms": {"heading": -hdg_cost[k], "shots": -shot_cost},
                    "doctrine": self._doctrine, "t_sim": self._t_sim,
                    "track_state": TrackState.NAMES[o._trk_state()] if o._alive else "NONE",
                    "fired": fires[k], "alive": o._alive, "agent": k,
                    "opponent": self._opponent_type.name,
                    "opponent_detail": getattr(self._opponent, "label", self._opponent_type.name)}
            if terminated or truncated:
                own_kill = 1 if self._killer == k + 1 else 0
                r += self.R_WASTED_MSL * float(max(o._shots_fired - own_kill, 0))
            rewards.append(r)
            infos.append(info)
        if terminated or truncated:
            # The episode's statistics ride on agent 0's info only, so the
            # trainer counts each episode once.
            launch_log = [dict(l, shooter=k) for k, o in enumerate(self._obs) for l in o._launch_log]
            infos[0].update({
                "terminal_outcome": self._outcome,
                "blue_losses": len(self._lost),
                "shots_fired": sum(o._shots_fired for o in self._obs),
                "misses": sum(o._misses for o in self._obs),
                "support_losses": sum(o._support_losses for o in self._obs),
                "launch_log": sorted(launch_log, key=lambda l: l["t_sim"]),
                "bank_reversals": sum(o._bank_revs for o in self._obs),
                "flight_time": sum(o._t_sim for o in self._obs),
                "wpn_remaining": sum(self._world.wpn[:self.n_agents]),
                "killer": self._killer})
            self._ready = False
        return obs, np.array(rewards, dtype=np.float32), terminated, truncated, infos

    def action_masks(self) -> np.ndarray:
        rows = []
        for o in self._obs:
            if o._alive:
                rows.append(o.action_masks())
            else:
                # One legal choice per dimension: straight ahead, same
                # altitude, first speed, no fire.
                m = np.zeros(sum(ACTION_NVEC), dtype=bool)
                off = 0
                for n, pick in zip(ACTION_NVEC, (0, 2, 0, 0)):
                    m[off + pick] = True
                    off += n
                rows.append(m)
        return np.stack(rows)

    # ── frame loop ────────────────────────────────────────────────────
    def _advance(self, duration, cmds, fires) -> float:
        """Fly `duration` seconds; returns the team reward from events on the way."""
        w = self._world
        sim_dt = w.SIM_DT
        n_frames = max(1, int(round(duration / sim_dt)))
        team_r = 0.0
        self._retarget()
        live = [o for o in self._obs if o._alive]
        self._opponent.rmax_t = self._obs[self._red_tgt - 1]._bandit_rmax() if live else 0.0

        f = 0
        ff_limit = n_frames + int(BvrEnv.RESOLVE_MAX_S / sim_dt)
        while f < n_frames or (self._outcome is None and not any(o._alive for o in self._obs)):
            if f >= ff_limit:
                self._outcome = self._pending_outcome()
                break
            if f >= n_frames and f % n_frames == 0 and self.resolve_hook is not None:
                self.resolve_hook(self)
            f += 1
            pkts = []
            for k, o in enumerate(self._obs):
                if o._alive and cmds[k] is not None:
                    p = dict(cmds[k])
                    p["fire"] = 1 if (fires[k] and f == 1) else 0
                    p["msl_guidance"] = o._guidance_packet()
                    p["target"] = self.RED
                else:
                    # A lost aircraft cannot support its missiles.
                    p = {"fire": 0, "msl_guidance": {"valid": 0}}
                pkts.append(p)
            if not self._obs[self._red_tgt - 1]._alive:
                self._retarget()
            red_view = w.telemetry(self._red_tgt, self.RED, view="team")
            try:
                opp = dict(self._opponent.act(red_view, self._t_sim))
            except Exception as e:
                from bvr_env import OpponentError
                raise OpponentError(self, e) from e
            opp["target"] = self._red_tgt
            pkts.append(opp)

            w.step_all(pkts)
            self._t_sim = w.t_sim
            for k, o in enumerate(self._obs, start=1):
                if not o._alive:
                    continue
                o._ingest(w.telemetry(k, self.RED))
                o._bandit_targets_me = (self._red_tgt == k)
                ph = o._state.get("phi", 0.0)
                if abs(ph) > o.BANK_REV_DEG * DEG2RAD:
                    sg = 1 if ph > 0 else -1
                    if o._bank_sign and sg != o._bank_sign:
                        o._bank_revs += 1
                    o._bank_sign = sg
                o._track.tick(sim_dt)
                if (o._track.t_now - o._last_convert_t) >= (1.0 / o.TRACK_CONVERT_HZ):
                    conv_dt = min(o._track.t_now - o._last_convert_t, 1.0)
                    o._last_convert_t = o._track.t_now
                    o._radar_update(dt=conv_dt)
            team_r += self._frame_events()
            if self._outcome is not None:
                break
        return team_r

    def _frame_events(self) -> float:
        """Kills, losses and crashes this frame; sets the outcome when the fight is over."""
        w = self._world
        r = 0.0
        for ev in w.events:
            key = (ev.get("type"), ev.get("id"), ev.get("ac"), round(float(ev.get("t_sim", w.t_sim)), 3))
            if key in self._events_seen:
                continue
            self._events_seen.add(key)
            if ev.get("type") == "MISSILE_HIT" and ev.get("killed", 1) and ev.get("target") == self.RED:
                self._killer = ev.get("owner")
            if ev.get("type") == "AC_DESTROYED":
                i = ev["ac"]
                if i == self.RED and self._red_dead is None:
                    self._red_dead = "KILL"
                    r += self.R_KILL
                elif i <= self.n_agents and self._obs[i - 1]._alive:
                    self._lose(i, "SHOT_DOWN")
                    r += self.R_LOST
        # The flight model does not crash on its own: apply the floor here.
        for i, o in enumerate(self._obs, start=1):
            if o._alive and w.acs[i - 1].z < self.MIN_ALT:
                w.kill(i, "CRASH")
                self._lose(i, "CRASH")
                r += self.R_CRASH
        if self._red_dead is None and w.acs[self.RED - 1].z < self.MIN_ALT:
            w.kill(self.RED, "CRASH")
            self._red_dead = "CRASH"
            r += self.R_BANDIT_CRASH

        alive = [o for o in self._obs if o._alive]
        if self._red_dead == "CRASH":
            self._outcome = "BANDIT_CRASH"
        elif self._red_dead == "KILL" or not alive:
            # Wait for missiles still in flight at a live aircraft.
            if not w.missiles_pending():
                self._outcome = self._pending_outcome()
        elif all(o._state.get("range", 0) > self.ESCAPE_RANGE for o in alive):
            self._outcome = "ESCAPE"
            r += self.R_ESCAPE
        return r

    def _pending_outcome(self):
        """The result the episode ends with once missiles in flight resolve."""
        if self._red_dead == "KILL":
            return "KILL" if not self._lost else "MUTUAL_KILL"
        if not any(o._alive for o in self._obs):
            return ("SHOT_DOWN" if any(c == "SHOT_DOWN" for _, c in self._lost)
                    else "CRASH")
        return None

    def _lose(self, i, cause):
        o = self._obs[i - 1]
        o._alive = False
        self._lost.append((i - 1, cause))

    def _retarget(self):
        """Red engages the nearest live blue aircraft, with some stickiness."""
        w = self._world
        live = [i for i in (1, 2) if self._obs[i - 1]._alive]
        if not live:
            return
        red = w.pos(self.RED)
        dist = {i: float(np.linalg.norm(w.pos(i) - red)) for i in live}
        cur = self._red_tgt if self._red_tgt in live else min(live, key=dist.get)
        best = min(live, key=dist.get)
        if best != cur and dist[best] < self.RETARGET_FRAC * dist[cur]:
            cur = best
        self._red_tgt = cur

    # ── per-agent views ───────────────────────────────────────────────
    def _phi(self, k) -> float:
        return self._obs[k]._potential()

    def _agent_obs(self, k):
        o = self._obs[k]
        if o._alive:
            return o._build_obs()
        # A lost aircraft sees nothing; the critic still sees its wingman.
        z = np.zeros(OBS_DIM, dtype=np.float32)
        if not self._privileged:
            return z
        pz = np.zeros(PRIV_DIM, dtype=np.float32)
        wm = o._wingman_priv()
        raw = np.concatenate([np.zeros(PRIV_DIM - len(wm)), wm]).astype(np.float32)
        pz[-len(wm):] = o._norm(raw, o._priv_lo, o._priv_hi)[-len(wm):]
        return {"obs": z, "priv": pz}

    # ── start geometry ────────────────────────────────────────────────
    FORMATIONS = ["abreast", "trail", "echelon"]

    def _random_ic(self) -> dict:
        """The 1v1 start for the lead against red, plus a wingman in formation."""
        lead = self._obs[0]._random_ic()
        rng = self._rng
        form = str(rng.choice(self.FORMATIONS))
        side = 1.0 if rng.random() < 0.5 else -1.0
        if form == "abreast":
            fwd, lat = float(rng.uniform(-1000, 1000)), side * float(rng.uniform(3000, 10000))
        elif form == "trail":
            fwd, lat = -float(rng.uniform(4000, 8000)), float(rng.uniform(-1000, 1000))
        else:
            fwd, lat = -float(rng.uniform(2000, 5000)), side * float(rng.uniform(3000, 6000))
        psi = lead["ac1_psi"]
        dx = fwd * math.sin(psi) + lat * math.cos(psi)       # east
        dy = fwd * math.cos(psi) - lat * math.sin(psi)       # north
        wing = self._obs[1]._plat
        ic = {
            "scenario": lead["scenario"], "formation": form, "start_range": lead["start_range"],
            "ac1_lat": lead["ac1_lat"], "ac1_lon": lead["ac1_lon"], "ac1_alt": lead["ac1_alt"],
            "ac1_psi": psi, "ac1_spd": lead["ac1_spd"],
            "ac2_lat": lead["ac1_lat"] + math.degrees(dy / R_EARTH),
            "ac2_lon": lead["ac1_lon"] + math.degrees(dx / (R_EARTH * math.cos(REF_LAT * DEG2RAD))),
            "ac2_alt": float(np.clip(lead["ac1_alt"] + rng.uniform(-1000, 1000), 4000, 12500)),
            "ac2_psi": psi,
            "ac2_spd": lead["ac1_spd"] / self._obs[0]._plat.speed_cmds[-1] * wing.speed_cmds[-1],
            "ac3_lat": lead["ac2_lat"], "ac3_lon": lead["ac2_lon"], "ac3_alt": lead["ac2_alt"],
            "ac3_psi": lead["ac2_psi"], "ac3_spd": lead["ac2_spd"],
            "wpn1": self._obs[0]._plat.wpn_count, "wpn2": wing.wpn_count,
            "wpn3": self._obs[0]._opp_plat.wpn_count,
        }
        return ic

    def close(self):
        pass


def _observer_reset(o: BvrEnv) -> None:
    """Start-of-episode state for a BvrEnv used as a team member's observer."""
    o._step_num = 0; o._t_sim = 0.0; o._ready = True; o._outcome = None
    o._last_shot_t = -999.0; o._shots_fired = 0; o._misses = 0
    o._support_losses = 0; o._events_seen.clear(); o._launch_log = []
    o._state = {}; o._last_convert_t = -1e9
    o._prev_hdg_off = 0.0; o._bank_revs = 0; o._bank_sign = 0
    o._alive = True; o._bandit_targets_me = False
    o._track.reset()
    if o._radar is not None:
        o._radar.reset()
