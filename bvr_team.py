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
  * Red's target. Red opens on a blue aircraft chosen by this episode's
    opening rule (OPEN_RULES: the nearer one, the farther one, or either at
    random) and holds a farther or random choice until its first launch.
    After that it engages the nearest blue aircraft, switching only when the
    other is clearly closer. It sees the fight as a 1v1 against its target
    with both blue aircraft's missiles counted as inbound.
  * Red's radar (RED_RADAR, SIM_REV 22). As in 1v1 since SIM_REV 9, red
    fights under the agent's rules: one track per blue aircraft, from the
    agent's own radar and track code run from red's side; it fires at an
    aircraft only when the agent's fire mask would allow it on that track,
    and each missile is guided on red's track of its target. No missile has
    the 3-second support rule (1v1's SIM_REV 18, as in DCS): one whose shooter
    loses the track flies on to its last estimate. Red still flies on truth,
    as in 1v1. Before, 2v1
    red guided on truth and shot without a track, and the old cranks and
    defence were kept for it (legacy_crank).
  * Losing an aircraft. A blue aircraft shot down or crashed leaves the
    fight; its slot keeps running until the episode ends with one legal
    action (so it adds no policy gradient), no shaping, and the team's
    rewards (so its value keeps learning what the team goes on to do).

Launch delay. A blue platform with LAUNCH_DELAY_MIN/MAX (F-16C-DCSFIT) fires
as in 1v1: the launch goes that many seconds after the decision, only if
the aircraft still tracks red within the fire cone then, and its fire mask
is shut meanwhile. A policy-flown red on such a platform does the same; a
scripted red (the DCS AI's stand-in) fires at once.

What red knows (red_awareness). "gci" (the default, as 1v1): scripted red
steers on the blue aircraft's true positions, as if a ground controller or
AWACS gave it the picture; only its shots and missile guidance depend on its
own radar. "radar": it steers on its own radar track of its target, coasting
its last estimate when the track drops, and before it first detects that
aircraft, on the position and velocity it had at the start, dead-reckoned (a
controller's initial vector, never updated). "radar_rwr": as "radar", and
while it has no track of an aircraft, its RWR's bearing to that aircraft's
radar (heard further away than the radar detects red) turns the belief onto
that bearing, keeping the believed range. Against a low-RCS blue team
(GENERIC-UCAV-DCS) that is the difference stealth makes to red's manoeuvring,
not only to when it can shoot.

The cost of losing a blue aircraft (blue_loss_cost, default 1.0): a smaller
cost for an attritable UCAV makes trading one for the kill worth more than
a timeout (see R_LOST).

Red flown by a policy (red_policies). In a share of the episodes at the last
scripted stage (ADAPTIVE_SHOOTER), red is a frozen 1v1 checkpoint instead
(TeamPolicyRed): it sees the fight as a 1v1 against its current target,
through red's own radar observer of that aircraft, with both blue aircraft's
missiles counted as inbound, and fires on its own fire mask. 2v1 has no
self-play stage; this is its stronger, less predictable opponent.

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
                     DOCTRINES, DOCTRINE_MIXED, PHI_TERMS, DEG2RAD, R_EARTH, REF_LAT, REF_LON,
                     _wrap_deg, _wrap_2pi)
from bvr_opponents import BvrOpponent, BvrOpponentType, ShooterOpponent
from bvr_track_adapter import TrackState
from bvr_library import DEFAULT_PLATFORM
from sim_world import SimWorld
from missile_sim import MslPhase, _isa_a

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
    # A blue aircraft shot down. At -0.7 a mutual kill (red killed, one blue
    # lost) netted +0.3, so trading the aircraft red shot first for a kill by
    # the other became the team's plan: 90% mutual kills against SHOOTER. At
    # -1.0 that trade nets 0, still above a timeout, below a clean kill by 1.
    R_LOST = -1.0
    R_CRASH = -1.0           # a blue aircraft flown into the ground
    R_BANDIT_CRASH = +0.4
    R_TIMEOUT = -0.8          # was -0.5 until SIM_REV 15 (see BvrEnv.R_TIMEOUT)
    R_ESCAPE = -0.6
    R_WASTED_MSL = BvrEnv.R_WASTED_MSL     # per own missile that did not kill

    # Red switches target only when the other blue aircraft is this much closer.
    RETARGET_FRAC = 0.8

    # Red's opening target, drawn per episode: (rule, probability). Red used
    # to open on the nearer aircraft only; with the usual formations that is
    # nearly always the lead, so the roles were fixed (team_roles.py showed
    # the rear aircraft free to close and shoot). "farther" and "random" hold
    # the choice until red's first launch, then red picks targets as usual.
    OPEN_RULES = (("nearer", 0.5), ("farther", 0.25), ("random", 0.25))

    RED_AWARENESS = ("gci", "radar", "radar_rwr")
    # radar_rwr (SIM_REV 22): red's RWR hears a blue aircraft's radar when red is inside that radar's scan
    # volume and within RWR_RANGE_FACTOR x the radar's detection range against its reference target (one-way
    # against two-way propagation: an emitter is heard further away than it detects). It gives a bearing
    # with RWR_BEARING_SIGMA_DEG of noise, no range, once per RWR_UPDATE_S. Illustrative values.
    RWR_RANGE_FACTOR = 1.5
    RWR_BEARING_SIGMA_DEG = 5.0
    RWR_UPDATE_S = 1.0

    # Scripted red fights on its own radar tracks (see the module notes).
    # False: the pre-SIM_REV 22 red, truth-guided with the old cranks.
    RED_RADAR = True

    def __init__(self, opponent_type=BvrOpponentType.STRAIGHT, gamma_discount=0.997,
                 seed=42, instance_id=0, privileged_critic=True, envelope_table="library",
                 radar_model="sim", doctrine=DOCTRINE_MIXED,
                 blue_platforms=(DEFAULT_PLATFORM, DEFAULT_PLATFORM),
                 red_platform=DEFAULT_PLATFORM, red_policies=(), red_policy_frac=0.5,
                 blue_loss_cost=None, red_awareness="gci"):
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
        # 1v1 checkpoints that may fly red (see the module notes), and the share
        # of ADAPTIVE_SHOOTER episodes they fly.
        self._red_policies = tuple(red_policies or ())
        self._red_policy_frac = float(red_policy_frac)
        if blue_loss_cost is not None:
            if not 0.0 <= float(blue_loss_cost) <= 2.0:
                raise ValueError(f"blue_loss_cost {blue_loss_cost}: choose 0-2")
            self.R_LOST = -float(blue_loss_cost)
        if red_awareness not in self.RED_AWARENESS:
            raise ValueError(f"red_awareness {red_awareness!r}: choose {' or '.join(self.RED_AWARENESS)}")
        self._red_awareness = red_awareness

        self._obs = [BvrEnv(opponent_type=BvrOpponentType.STRAIGHT, gamma_discount=gamma_discount,
                            seed=seed + 97 * k, instance_id=instance_id,
                            privileged_critic=privileged_critic, envelope_table=envelope_table,
                            radar_model=radar_model, doctrine="AGGRESSIVE",
                            platform=p, opponent_platform=red_platform)
                     for k, p in enumerate(blue_platforms)]
        a, b = self._obs
        a._wingman, b._wingman = b, a
        # Red's radar and track on each blue aircraft: the agent's own code run
        # from red's side (as BvrEnv._opponent_radar in 1v1), with the blue
        # observer's envelopes seen the other way round.
        self._red_obs = []
        for o, p in zip(self._obs, blue_platforms):
            r = BvrEnv(seed=seed + 7919 + 31 * len(self._red_obs), instance_id=instance_id,
                       privileged_critic=False, radar_model=radar_model, envelope_table=None,
                       platform=red_platform, opponent_platform=p)
            r._env_own, r._env_thr = o._env_thr, o._env_own
            self._red_obs.append(r)
        self._red_radar = False                   # this episode: red fights on its tracks
        self._open_rule, self._open_hold, self._red_wpn0 = "nearer", False, 0

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
        for o in self._obs + self._red_obs:
            # The run's episode length (train_bvr.py --max-steps sets it on the
            # team env): each observation's t_norm is step / MAX_STEPS, and
            # with the default 300 it went past 1 in a 400-step run.
            o.MAX_STEPS = self.MAX_STEPS
        for o in self._obs:
            _observer_reset(o)
            o._doctrine, o._shot_cost = d, float(DOCTRINES[d])

        ic = self._random_ic()
        self._episode_id = int(self._rng.integers(1, 2_000_000_000))
        for o in self._obs:
            o._episode_id = self._episode_id
        if self._opponent_factory is not None:
            self._opponent = self._opponent_factory(self)
        elif (self._red_policies and self._opponent_type == BvrOpponentType.ADAPTIVE_SHOOTER
              and self._rng.random() < self._red_policy_frac):
            path = self._red_policies[int(self._rng.integers(len(self._red_policies)))]
            self._opponent = make_policy_red(self, path)
        else:
            self._opponent = BvrOpponent.create(self._opponent_type, rng=self._rng)
        policy_red = isinstance(self._opponent, TeamPolicyRed)
        self._red_radar = policy_red or (self.RED_RADAR and isinstance(self._opponent, ShooterOpponent)
                                         and radar_ok(self._obs[0]))
        if policy_red:
            pass                                   # it fires on its own mask; reset() below
        elif self._red_radar:
            for r in self._red_obs:
                _red_observer_reset(r)
            # Bound methods, not lambdas: the env must pickle (SubprocVecEnv).
            self._opponent.fire_gate = self._red_gate_target
            self._opponent.fire_gate_other = self._red_gate_other
        else:
            # Truth-guided red keeps the pre-SIM_REV 9 cranks and old defence.
            self._opponent.legacy_crank = True
        # Scripted opponents read their start from the 1v1 key names.
        self._opponent.reset({**ic, "ac2_psi": ic["ac3_psi"], "ac2_alt": ic["ac3_alt"],
                              "ac2_spd": ic["ac3_spd"]})
        # No support rule for any missile (SIM_REV 18 in 1v1): a missile whose
        # shooter loses the track flies on to its last estimate and turns its
        # seeker on, as in DCS. BvrEnv.SUPPORT_RULE puts it back for all three.
        w.no_support_rule = set() if self._obs[0].SUPPORT_RULE else {1, 2, self.RED}
        w.reset(ic, episode_id=self._episode_id)
        # Red's opening target, by this episode's rule (OPEN_RULES).
        red = w.pos(self.RED)
        dist = {i: float(np.linalg.norm(w.pos(i) - red)) for i in (1, 2)}
        rules, probs = zip(*self.OPEN_RULES)
        p = np.asarray(probs, dtype=float)
        self._open_rule = str(rules[int(self._rng.choice(len(rules), p=p / p.sum()))])
        if self._open_rule == "farther":
            self._red_tgt = max((1, 2), key=dist.get)
        elif self._open_rule == "random":
            self._red_tgt = int(self._rng.integers(1, 3))
        else:
            self._red_tgt = min((1, 2), key=dist.get)
        self._open_hold = self._open_rule != "nearer"
        # red_awareness "radar": what red believes of each blue aircraft, as
        # (position, velocity, sim time); the start is the controller's vector.
        self._red_belief = {i: (w.pos(i).copy(), w.acs[i - 1].vel_enu.copy(), 0.0) for i in (1, 2)}
        self._rwr_next = {i: 0.0 for i in (1, 2)}
        self._rwr_heard = {i: 0 for i in (1, 2)}          # RWR bearings that corrected red's belief
        self._red_wpn0 = w.wpn[self.RED - 1]
        self._red_other = None
        for k, o in enumerate(self._obs, start=1):
            o._cmd_hdg = float(ic[f"ac{k}_psi"]); o._cmd_alt = float(ic[f"ac{k}_alt"])
            o._cmd_spd = float(ic[f"ac{k}_spd"])
            o._hdg_ref = self._start_ref(o, k, self.RED)
        if policy_red:
            for k, r in enumerate(self._red_obs, start=1):
                r._hdg_ref = self._start_ref(r, self.RED, k)

        cmds = [o._encode_cmd(0, 2, 2, 0) for o in self._obs]
        self._advance(0.5, cmds, [False, False])
        self._ready = True
        obs = [self._agent_obs(k) for k in range(self.n_agents)]
        self._prev_phi = [self._phi(k) for k in range(self.n_agents)]
        self._settled = [False] * self.n_agents       # shaping paid out (end state's potential 0)
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
                # Platform launch delay (as in 1v1): a delayed shot is queued and
                # goes, or is dropped, in the frame _due_fire() says.
                fire = fire and o._request_fire(self._t_sim)
                cmds.append(o._encode_cmd(i_hdg, i_alt, i_spd, i_fire))
                hdg_cost.append(o._heading_choice(HDG_OFFSETS_DEG[i_hdg]))
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
        # SIM_REV 16: out of time ends the episode (terminated), not a cut stable-baselines3 bootstraps through:
        # as "truncated", gamma * V(last state) was added to R_TIMEOUT and late states' values ran away (bvr_env).
        timeout = (not terminated) and self._step_num >= self.MAX_STEPS
        truncated = False
        if timeout:
            terminated = True
            # A result waiting only on missiles still in flight stands.
            self._outcome = self._pending_outcome()
            if self._outcome is None:
                self._outcome = "TIMEOUT"
                team_r += self.R_TIMEOUT
        for k, o in enumerate(self._obs):
            obs.append(self._agent_obs(k))
            if self._settled[k]:
                # Its shaping is paid out (lost earlier): nothing more.
                phi, shaping = 0.0, 0.0
                terms = {t: 0.0 for t in PHI_TERMS}
                shaping_terms = {t: 0.0 for t in PHI_TERMS}
            elif (flying[k] and not o._alive) or terminated:
                # SIM_REV 22, as 1v1 since SIM_REV 16: the potential of an
                # aircraft's end state is 0 (lost this step, or the episode
                # ends), so its shaping sums to -phi(start) whatever happens
                # and leaves the best policy as it was (Ng et al. 1999). Lost
                # deep in red's envelope (negative potential) this step pays
                # back exactly what the approach to it was charged.
                phi, shaping = 0.0, -self._prev_phi[k]
                terms = {t: 0.0 for t in PHI_TERMS}
                shaping_terms = {t: -self._prev_terms[k][t] for t in PHI_TERMS}
                self._settled[k] = True
            elif flying[k]:
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
                "fire_requests": sum(o._fire_requests for o in self._obs),
                "fire_dropped": sum(o._fire_dropped for o in self._obs),
                "misses": sum(o._misses for o in self._obs),
                "support_losses": sum(o._support_losses for o in self._obs),
                "launch_log": sorted(launch_log, key=lambda l: l["t_sim"]),
                "bank_reversals": sum(o._bank_revs for o in self._obs),
                "hdg_switches": sum(o._hdg_switches for o in self._obs),
                "hdg_pairs": _merge_counts(o._hdg_pairs for o in self._obs),
                "flight_time": sum(o._t_sim for o in self._obs),
                "wpn_remaining": sum(self._world.wpn[:self.n_agents]),
                "killer": self._killer,
                "red_open_rule": self._open_rule,
                "red_awareness": self._red_awareness,
                "red_rwr_bearings": sum(self._rwr_heard.values())})
            self._ready = False
        return obs, np.array(rewards, dtype=np.float32), terminated, truncated, infos

    def _start_ref(self, o, me, other):
        """The heading reference at the start (SIM_REV 22, 1v1's SIM_REV 21): the picture call's bearing
        from aircraft `me` to `other` until the radar has a track (o.GCI_START), else `me`'s own heading.
        Without it the reference was None and the first command read the heading before any state: 0, north."""
        w = self._world
        if not o.GCI_START:
            return float(w.acs[me - 1].chi)
        d = w.pos(other) - w.pos(me)
        return math.atan2(float(d[0]), float(d[1]))

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
        self._red_other = self._other_info()

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
                    p["fire"] = 1 if ((fires[k] and f == 1) or o._due_fire(self._t_sim)) else 0
                    p["msl_guidance"] = o._guidance_packet()
                    p["target"] = self.RED
                else:
                    # A lost aircraft cannot support its missiles.
                    p = {"fire": 0, "msl_guidance": {"valid": 0}}
                pkts.append(p)
            if not self._obs[self._red_tgt - 1]._alive:
                self._retarget()
                self._red_other = self._other_info()
            red_view = w.telemetry(self._red_tgt, self.RED, view="team")
            if self._red_awareness in ("radar", "radar_rwr") and self._red_radar:
                if self._red_awareness == "radar_rwr":
                    self._red_rwr_update()
                red_view = self._red_believed_view(red_view)
            if self._red_other is not None:
                red_view["other"] = self._red_other
            try:
                opp = dict(self._opponent.act(red_view, self._t_sim))
            except Exception as e:
                from bvr_env import OpponentError
                raise OpponentError(self, e) from e
            # A policy-flown red names the target of a delayed launch itself.
            opp.setdefault("target", self._red_tgt)
            if opp.pop("target_other", 0) and self._red_other is not None:
                opp["target"] = 3 - self._red_tgt
                self._red_other = dict(self._red_other, targeted=True)
            if self._red_radar:
                opp = self._red_radar_frame(opp)
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
        """Red engages the nearest live blue aircraft, with some stickiness,
        once any held opening target has been shot at."""
        w = self._world
        live = [i for i in (1, 2) if self._obs[i - 1]._alive]
        if not live:
            return
        if self._open_hold:
            # A farther or random opening is held until red's first launch.
            if self._red_tgt in live and w.wpn[self.RED - 1] >= self._red_wpn0:
                return
            self._open_hold = False
        red = w.pos(self.RED)
        dist = {i: float(np.linalg.norm(w.pos(i) - red)) for i in live}
        cur = self._red_tgt if self._red_tgt in live else min(live, key=dist.get)
        best = min(live, key=dist.get)
        if best != cur and dist[best] < self.RETARGET_FRAC * dist[cur]:
            cur = best
        self._red_tgt = cur

    # ── red's radar (RED_RADAR) ───────────────────────────────────────
    def _red_can_fire(self, k) -> bool:
        """Whether red may fire at blue aircraft k now: the agent's fire mask
        applied to red's own track of it."""
        return bool(self._obs[k - 1]._alive and self._red_obs[k - 1]._can_fire())

    def _rwr_bearing(self, k):
        """The bearing at which red's RWR hears blue aircraft k's radar, or None: k alive with a radar, red
        inside its scan volume and within RWR_RANGE_FACTOR x its detection range (against the radar's
        reference target, not red's RCS: the RWR hears the emitter's power)."""
        o, w = self._obs[k - 1], self._world
        rad = o._radar
        if not o._alive or rad is None:
            return None
        d = w.pos(self.RED) - w.pos(k)
        rng = float(np.linalg.norm(d))
        if rng > self.RWR_RANGE_FACTOR * float(o._plat.radar.MAX_RANGE):
            return None
        az = abs(_wrap_deg(math.degrees(math.atan2(d[0], d[1]) - w.acs[k - 1].chi)))
        el = abs(math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))))
        if az > math.degrees(rad.fov_az) or el > math.degrees(rad.fov_el):
            return None
        b = math.atan2(-d[0], -d[1])                       # red -> blue
        return b + math.radians(float(self._rng.normal(0.0, self.RWR_BEARING_SIGMA_DEG)))

    def _red_rwr_update(self):
        """red_awareness "radar_rwr": while red's radar has no track of a blue aircraft, an RWR bearing to it
        turns red's dead-reckoned belief onto that bearing, keeping the believed range and altitude (an RWR
        gives no range). Once per RWR_UPDATE_S for each aircraft."""
        red = self._world.pos(self.RED)
        for k in (1, 2):
            if self._t_sim < self._rwr_next[k]:
                continue
            self._rwr_next[k] = self._t_sim + self.RWR_UPDATE_S
            r = self._red_obs[k - 1]
            if r._track.state != TrackState.NONE and r._est()["valid"]:
                continue                                   # its own track is better
            b = self._rwr_bearing(k)
            if b is None:
                continue
            pos, vel, t0 = self._red_belief[k]
            rel = pos + vel * (self._t_sim - t0) - red
            rh = float(math.hypot(rel[0], rel[1]))
            new = red + np.array([rh * math.sin(b), rh * math.cos(b), rel[2]])
            self._red_belief[k] = (new, vel, self._t_sim)
            self._rwr_heard[k] += 1

    def _red_believed_view(self, view: dict) -> dict:
        """red_awareness "radar": red's packet with its target's position and
        range taken from red's own track (or its coasted / dead-reckoned
        belief) instead of the truth. Scripted red reads only these of the
        target (its bearing and range); its own state stays true."""
        from sim_world import _enu_to_latlon
        k = self._red_tgt
        r = self._red_obs[k - 1]
        est = r._est() if r._track.state != TrackState.NONE else {"valid": False}
        if est["valid"]:
            self._red_belief[k] = (np.asarray(est["pos"], float).copy(),
                                   np.asarray(est["vel"], float).copy(), self._t_sim)
        pos, vel, t0 = self._red_belief[k]
        pos = pos + vel * (self._t_sim - t0)
        lat, lon, alt = _enu_to_latlon(*pos)
        rng = float(np.linalg.norm(pos - self._world.pos(self.RED)))
        return {**view, "lat": lat, "lon": lon, "alt": alt, "range": max(rng, 1.0)}

    def _red_gate_target(self) -> bool:
        return self._red_can_fire(self._red_tgt)

    def _red_gate_other(self) -> bool:
        return self._red_can_fire(3 - self._red_tgt)

    def _red_radar_frame(self, opp: dict) -> dict:
        """One frame of red's sensors, as BvrEnv._opponent_radar_frame in 1v1,
        with one track per blue aircraft; each red missile is guided on red's
        track of its own target (the key must be present for every target, or
        the world falls back to truth)."""
        w = self._world
        for k, r in enumerate(self._red_obs, start=1):
            if not self._obs[k - 1]._alive:
                continue
            r._track.tick(w.SIM_DT)
            r._t_sim = self._t_sim
            if (r._track.t_now - r._last_convert_t) >= 1.0 / BvrEnv.TRACK_CONVERT_HZ:
                conv_dt = min(r._track.t_now - r._last_convert_t, 1.0)
                r._last_convert_t = r._track.t_now
                r._state = w.telemetry(self.RED, k)
                r._radar_update(dt=conv_dt)
        opp = dict(opp)
        if opp.get("fire"):
            for r in self._red_obs:              # one radar: the 3 s between shots binds both
                r._last_shot_t = self._t_sim
        opp["msl_guidance_by_target"] = {
            k: (r._guidance_packet() if self._obs[k - 1]._alive else {"valid": 0})
            for k, r in enumerate(self._red_obs, start=1)}
        opp["msl_guidance"] = {"valid": 0}
        return opp

    def _other_info(self):
        """The blue aircraft red is not engaging, as red sees it: the scripted
        shooter's second shot reads this. None when there is no such aircraft.
        Refreshed once per decision step, and when red changes target."""
        w = self._world
        k = 3 - self._red_tgt
        o = self._obs[k - 1]
        if not o._alive or not self._obs[self._red_tgt - 1]._alive:
            return None
        los = w.pos(k) - w.pos(self.RED)
        v = w.acs[self.RED - 1].vel_enu
        off = abs(_wrap_deg(math.degrees(math.atan2(los[0], los[1]) - math.atan2(v[0], v[1]))))
        targeted = any(m.owner == self.RED and m.target == k
                       and m.phase not in (MslPhase.HIT, MslPhase.MISS) for m in w.missiles)
        return {"range": float(np.linalg.norm(los)), "rmax": float(o._bandit_rmax()),
                "off_nose_deg": off, "targeted": targeted}

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

    # ── start geometry (SIM_REV 22) ───────────────────────────────────
    # 2v1's own start generator. CLASSIC_FRAC of the starts are 1v1's set-ups for the lead (head-on, offset,
    # beam, stern, half of them swapped), the rest FREE: range, the lead's angle off the line of sight and
    # red's aspect drawn independently, SHORT_FRAC of those at 30-60 km (late detection). 1v1's set-ups put
    # one aircraft within 10 deg of nose-on in 60% of the starts and almost never 40-60 deg off or quartering
    # away. A faster red keeps the stern starts where it is behind blue (swapped; the most dangerous set-up for
    # a slow UCAV) and loses only those where it runs from blue, which blue could never catch.
    CLASSIC_FRAC = 0.7
    SHORT_FRAC = 0.5                  # of the free starts: 30-60 km
    # Start speeds by Mach (0.75-0.95 at the start altitude), within each aircraft's own speed choices (a UCAV:
    # 85-100% of its fastest, 212-250 m/s); an
    # aircraft whose fastest choice reaches Mach 1.2 there starts supersonic (Mach 1.0-1.3) in SUPERSONIC_FRAC
    # of the starts. 1v1 scales an F-16's 250-320 m/s by the top speed: a UCAV started at 136-174 m/s.
    START_MACH = (0.75, 0.95)
    SUPERSONIC_MACH = (1.0, 1.3)
    SUPERSONIC_FRAC = 0.2
    # Wingman formations, (name, probability): the three tight ones, a wide split (15-30 km abeam, a pincer
    # from the start) and a long trail (10-20 km behind). In STACK_FRAC of the starts the wingman flies 3-4 km
    # above or below the lead (high-low), in HDG_OFF_FRAC 10-30 deg off the lead's heading.
    FORMATIONS = (("abreast", 0.25), ("trail", 0.2), ("echelon", 0.25), ("split", 0.15), ("long_trail", 0.15))
    STACK_FRAC = 0.25
    HDG_OFF_FRAC = 0.25
    ALT_RANGE = (2000.0, 12500.0)

    def blue_top_speed(self) -> float:
        return max(o._plat.speed_cmds[-1] for o in self._obs)

    def _start_speed(self, plat, alt, supersonic=None) -> float:
        cmds = plat.speed_cmds
        a = _isa_a(alt)
        if supersonic is None:
            supersonic = cmds[-1] >= 1.2 * a and self._rng.random() < self.SUPERSONIC_FRAC
        if supersonic:
            return float(np.clip(self._rng.uniform(*self.SUPERSONIC_MACH) * a, cmds[0], cmds[-1]))
        # Mach 0.75-0.95, the top end held to the fastest choice; a slow aircraft (a UCAV whose fastest
        # choice is below Mach 0.95) draws evenly from 85% of that to it rather than piling up at the cap.
        hi = min(self.START_MACH[1] * a, cmds[-1])
        lo = min(self.START_MACH[0] * a, 0.85 * hi)
        return float(np.clip(self._rng.uniform(lo, hi), cmds[0], cmds[-1]))

    def _lead_geometry(self) -> dict:
        """Lead (ac1) and red (ac3) positions, headings and altitudes, before speeds."""
        rng, lead_env = self._rng, self._obs[0]
        red_faster = lead_env._opp_plat.speed_cmds[-1] > self.blue_top_speed()
        if rng.random() < self.CLASSIC_FRAC:
            while True:
                g = lead_env._random_ic(blue_top_speed=1e9)          # every set-up, stern included
                # Red running from blue (unswapped stern) only if blue can catch it.
                if not (red_faster and g["scenario"] == "stern_conversion" and not g["mirrored"]):
                    break
            return {"geometry": "classic", "scenario": g["scenario"], "mirrored": g["mirrored"],
                    "start_range": g["start_range"],
                    "ac1_lat": g["ac1_lat"], "ac1_lon": g["ac1_lon"], "ac1_alt": g["ac1_alt"], "ac1_psi": g["ac1_psi"],
                    "ac3_lat": g["ac2_lat"], "ac3_lon": g["ac2_lon"], "ac3_alt": g["ac2_alt"], "ac3_psi": g["ac2_psi"]}
        short = rng.random() < self.SHORT_FRAC
        rm = float(rng.uniform(30_000, 60_000) if short else rng.uniform(60_000, 110_000))
        br = float(rng.uniform(0, 2 * math.pi))                       # lead -> red
        ata = float(rng.uniform(0, math.pi)) * (1 if rng.random() < 0.5 else -1)
        # Red's aspect: 0 nose-on to the lead. A faster red not running from blue (within 120 deg).
        asp = float(rng.uniform(0, (2 / 3 if red_faster else 1.0) * math.pi)) * (1 if rng.random() < 0.5 else -1)
        a1 = float(rng.uniform(*self.ALT_RANGE))
        a3 = float(np.clip(a1 + rng.uniform(-2500, 2500), *self.ALT_RANGE)) if rng.random() < 0.5 \
            else float(rng.uniform(*self.ALT_RANGE))
        dn, de = rm * math.cos(br), rm * math.sin(br)
        return {"geometry": "short" if short else "free", "scenario": "free_short" if short else "free",
                "mirrored": False, "start_range": rm,
                "ac1_lat": REF_LAT, "ac1_lon": REF_LON, "ac1_alt": a1, "ac1_psi": _wrap_2pi(br + ata),
                "ac3_lat": REF_LAT + math.degrees(dn / R_EARTH),
                "ac3_lon": REF_LON + math.degrees(de / (R_EARTH * math.cos(REF_LAT * DEG2RAD))),
                "ac3_alt": a3, "ac3_psi": _wrap_2pi(br + math.pi + asp)}

    def _random_ic(self) -> dict:
        """The lead against red (_lead_geometry), a wingman in formation, and start speeds."""
        rng = self._rng
        g = self._lead_geometry()
        names, probs = zip(*self.FORMATIONS)
        p = np.asarray(probs, dtype=float)
        form = str(names[int(rng.choice(len(names), p=p / p.sum()))])
        side = 1.0 if rng.random() < 0.5 else -1.0
        if form == "abreast":
            fwd, lat = float(rng.uniform(-1000, 1000)), side * float(rng.uniform(3000, 10000))
        elif form == "trail":
            fwd, lat = -float(rng.uniform(4000, 8000)), float(rng.uniform(-1000, 1000))
        elif form == "echelon":
            fwd, lat = -float(rng.uniform(2000, 5000)), side * float(rng.uniform(3000, 6000))
        elif form == "split":
            fwd, lat = float(rng.uniform(-3000, 3000)), side * float(rng.uniform(15000, 30000))
        else:                                                        # long_trail
            fwd, lat = -float(rng.uniform(10000, 20000)), float(rng.uniform(-2000, 2000))
        psi = g["ac1_psi"]
        dx = fwd * math.sin(psi) + lat * math.cos(psi)       # east
        dy = fwd * math.cos(psi) - lat * math.sin(psi)       # north
        stack = rng.random() < self.STACK_FRAC
        if stack:
            up = 1.0 if rng.random() < 0.5 else -1.0
            if not self.ALT_RANGE[0] <= g["ac1_alt"] + up * 3000 <= self.ALT_RANGE[1]:
                up = -up
            dz = up * float(rng.uniform(3000, 4000))
        else:
            dz = float(rng.uniform(-1000, 1000))
        hdg_off = float(rng.uniform(10, 30)) * (1 if rng.random() < 0.5 else -1) \
            if rng.random() < self.HDG_OFF_FRAC else 0.0
        lead_p, wing_p, red_p = self._obs[0]._plat, self._obs[1]._plat, self._obs[0]._opp_plat
        a2 = float(np.clip(g["ac1_alt"] + dz, *self.ALT_RANGE))
        v1 = self._start_speed(lead_p, g["ac1_alt"])
        sup = v1 > _isa_a(g["ac1_alt"])                       # the wingman flies the lead's regime
        ic = {
            "scenario": g["scenario"], "mirrored": g["mirrored"], "geometry": g["geometry"],
            "formation": form, "wing_stack": bool(stack), "wing_hdg_off": hdg_off,
            "start_range": g["start_range"],
            "ac1_lat": g["ac1_lat"], "ac1_lon": g["ac1_lon"], "ac1_alt": g["ac1_alt"],
            "ac1_psi": psi, "ac1_spd": v1,
            "ac2_lat": g["ac1_lat"] + math.degrees(dy / R_EARTH),
            "ac2_lon": g["ac1_lon"] + math.degrees(dx / (R_EARTH * math.cos(REF_LAT * DEG2RAD))),
            "ac2_alt": a2, "ac2_psi": _wrap_2pi(psi + hdg_off * DEG2RAD),
            "ac2_spd": self._start_speed(wing_p, a2, supersonic=sup and wing_p.speed_cmds[-1] >= 1.2 * _isa_a(a2)),
            "ac3_lat": g["ac3_lat"], "ac3_lon": g["ac3_lon"], "ac3_alt": g["ac3_alt"],
            "ac3_psi": g["ac3_psi"], "ac3_spd": self._start_speed(red_p, g["ac3_alt"]),
            "wpn1": lead_p.wpn_count, "wpn2": wing_p.wpn_count, "wpn3": red_p.wpn_count,
        }
        return ic

    def close(self):
        pass


def _merge_counts(dicts) -> dict:
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def radar_ok(o: BvrEnv) -> bool:
    """Red radar needs the simulated radar model (as OPPONENT_RADAR in 1v1)."""
    return getattr(o, "_radar_model", "sim") == "sim"


def _red_observer_reset(r: BvrEnv) -> None:
    """Start-of-episode state for one of red's radar observers."""
    r._track.reset()
    if r._radar is not None:
        r._radar.reset()
    r._state = {}; r._t_sim = 0.0
    r._last_shot_t = -999.0; r._last_convert_t = -1e9
    r._fire_due = None; r._fire_requests = 0; r._fire_dropped = 0


def _observer_reset(o: BvrEnv) -> None:
    """Start-of-episode state for a BvrEnv used as a team member's observer."""
    o._step_num = 0; o._t_sim = 0.0; o._ready = True; o._outcome = None
    o._last_shot_t = -999.0; o._shots_fired = 0; o._misses = 0
    o._fire_due = None; o._fire_requests = 0; o._fire_dropped = 0
    o._support_losses = 0; o._events_seen.clear(); o._launch_log = []
    o._state = {}; o._last_convert_t = -1e9
    o._prev_hdg_off = 0.0; o._bank_revs = 0; o._bank_sign = 0
    o._hdg_switches = 0; o._hdg_pairs = {}
    o._hdg_ref = None
    o._alive = True; o._bandit_targets_me = False
    o._track.reset()
    if o._radar is not None:
        o._radar.reset()


# ── red flown by a 1v1 policy ─────────────────────────────────────────
class TeamPolicyRed:
    """Red flown by a frozen 1v1 checkpoint (red_policies).

    The policy fights its current target as a 1v1: its observation, mask
    and commands are the agent's code run on red's radar observer of that
    aircraft (env._red_obs), which also guides its missiles, so it sees
    nothing a 1v1 opponent would not. Both blue aircraft's missiles aimed at
    red appear as inbound. When red changes target, the commanded heading,
    altitude, speed and doctrine carry over to the new target's observer.
    The frame history is kept across a change of target.
    """

    def __init__(self, env, model, privileged: bool, label: str,
                 deterministic: bool = False, doctrine: str = None):
        from bvr_selfplay import FrameStacker, N_STACK
        self._env = env
        self._model = model
        self._privileged = bool(privileged)
        self._deterministic = bool(deterministic)
        self._doctrine = doctrine
        self._stack = FrameStacker(N_STACK)
        self.label = label
        self.rmax_t = None          # set by the env; unused here

    def reset(self, ic):
        rs = self._env._red_obs
        for r in rs:
            _red_observer_reset(r)
            r._privileged = self._privileged
            r._step_num = 0
            r._prev_hdg_off = 0.0
            r._hdg_ref = None
            r._cmd_hdg = float(ic.get("ac2_psi", r._cmd_hdg))
            r._cmd_alt = float(ic.get("ac2_alt", r._cmd_alt))
            r._cmd_spd = float(ic.get("ac2_spd", r._cmd_spd))
        a = rs[0]
        if self._doctrine is not None:
            a._doctrine_cfg = str(self._doctrine).upper()
        a._pick_doctrine()          # red fights under its own doctrine
        for r in rs[1:]:
            r._doctrine, r._shot_cost = a._doctrine, a._shot_cost
        self._tgt = None
        self._pending = None        # (observer, target) of a launch waiting out its delay
        self._next_decision = -1.0
        self._started = False
        self._cmd = None

    def _handover(self, old, new):
        for k in ("_cmd_hdg", "_cmd_alt", "_cmd_spd", "_prev_hdg_off", "_hdg_ref",
                  "_doctrine", "_shot_cost", "_last_shot_t"):
            setattr(new, k, getattr(old, k))

    def act(self, state, t_sim):
        env = self._env
        w, tgt = env._world, env._red_tgt
        o = env._red_obs[tgt - 1]
        fire, fire_at = 0, None
        if self._pending is not None:
            po, pk = self._pending
            if not env._obs[pk - 1]._alive:
                po._fire_due = None; po._fire_dropped += 1; self._pending = None
            elif po._due_fire(t_sim):           # a launch decided earlier (launch delay)
                fire, fire_at, self._pending = 1, pk, None
            elif po._fire_due is None:          # dropped by _due_fire
                self._pending = None
        if t_sim >= self._next_decision:
            if self._tgt is not None and self._tgt != tgt:
                self._handover(env._red_obs[self._tgt - 1], o)
            self._tgt = tgt
            o._state = w.telemetry(env.RED, tgt)
            o._t_sim = t_sim
            o._step_num = env._step_num
            raw = o._build_obs()
            obs = self._stack.update(raw) if self._started else self._stack.reset(raw)
            self._started = True
            mask = o.action_masks()
            if self._pending is not None or fire:
                mask = mask.copy(); mask[-1] = False   # one launch on its way at a time
            a, _ = self._model.predict(obs, action_masks=mask, deterministic=self._deterministic)
            a = [int(x) for x in np.asarray(a).reshape(-1)]
            self._cmd = o._encode_cmd(a[0], a[1], a[2], 0)
            o._heading_choice(HDG_OFFSETS_DEG[a[0]])
            if a[3] == 1 and mask[-1]:
                if o._request_fire(t_sim):
                    fire, fire_at = 1, tgt
                else:
                    self._pending = (o, tgt)
            self._next_decision = t_sim + 1.0 / env.DECISION_HZ
        cmd = dict(self._cmd)
        cmd["fire"] = fire
        if fire_at is not None:
            cmd["target"] = fire_at
        return cmd


def check_red_policy(path: str, red_platform: str) -> list:
    """Load a red_policies checkpoint and return notes on it: an error is
    raised for a 2v1 checkpoint, a note given for a platform mismatch."""
    from bvr_selfplay import load_policy
    from bvr_library import scenario_of
    model, _ = load_policy(path)
    sc = scenario_of(model)
    if sc.get("format", "1v1") != "1v1":
        raise ValueError(f"{path}: a {sc.get('format')} checkpoint cannot fly red; "
                         f"use a 1v1 checkpoint")
    notes = []
    if sc.get("platform") != red_platform:
        notes.append(f"red policy {path} was trained as {sc.get('platform')}, here it flies "
                     f"{red_platform}")
    return notes


def make_policy_red(env, path: str, deterministic: bool = False, doctrine: str = None):
    """This episode's policy-flown red, from a checkpoint path."""
    import os
    from bvr_selfplay import load_policy
    model, privileged = load_policy(path)
    name = os.path.basename(path)
    return TeamPolicyRed(env, model, privileged, label="policy:" + (name[:-4] if name.endswith(".zip") else name),
                         deterministic=deterministic, doctrine=doctrine)
