"""
sim_world.py  —  Simulation world
==================================

N aircraft in two teams + list of AIM-120 missiles + per-frame event queue.

  world = SimWorld(seed=42)                       # 1v1: AC1 v AC2
  world.reset(ic, episode_id=7)
  for ...:
      tlm = world.step(cmd1, cmd2)  # one SIM_DT advance

  world = SimWorld(seed=42, platforms=[a, b, c], teams=[1, 1, 2])   # 2v1
  world.step_all([cmd1, cmd2, cmd3])
  world.telemetry(me=1, other=3)

Aircraft are numbered from 1 in the order given. A 1v1 world is the special
case platforms=[p1, p2], teams=[1, 2], and behaves exactly as the two-aircraft
world did before (same random draws, same packets).

`tlm` is a dict with exactly the field names bvr_env.py reads via _ingest(),
so the env needs no awareness of whether it is talking to a C++ process or
to this class.

A missile is guided by its shooter's command: cmd["msl_guidance"] (a radar
or datalink estimate) when the command carries that key, otherwise truth on
the missile's target, which is what scripted opponents get. A command may
name its target with cmd["target"] (an aircraft number); without it the
shooter fires at the first live enemy.

telemetry(me, other) builds the packet from aircraft `me`'s point of view
against enemy `other`. Missile and event owners are relabelled so the reader
needs no aircraft numbers: 1 = me, 3 = a teammate, 2 = the enemy `other`
and 4 = any other enemy; an enemy missile is owned by 2 when it is aimed at
me and by 4 when it is aimed at someone else. In
1v1 only 1 and 2 occur, exactly as before. view="team" instead labels by side
(my team 1, the enemy 2): a scripted opponent reads the fight that way.

Aircraft destroyed by a missile, or removed with kill() (a crash), stop
flying; missiles aimed at them go on to their fuze or flight-time limit
against the last position, as before the first kill ended every episode.
"""

import math
import numpy as np

from f16_sim   import F16Aircraft, _wrap_pi
from missile_sim import AIM120, MslPhase
from bvr_library import rcs_scale

G        = 9.80665
REF_LAT  = 39.0
REF_LON  = 35.0
R_EARTH  = 6_371_000.0
DEG2RAD  = math.pi / 180.0
RAD2DEG  = 180.0 / math.pi


# ── coordinate helpers ───────────────────────────────────────────────
def _enu_to_latlon(x: float, y: float, z: float):
    lat = REF_LAT + math.degrees(y / R_EARTH)
    lon = REF_LON + math.degrees(x / (R_EARTH * math.cos(REF_LAT * DEG2RAD)))
    return lat, lon, float(z)


def _latlon_to_enu(lat: float, lon: float, alt: float):
    x = (lon - REF_LON) * DEG2RAD * R_EARTH * math.cos(REF_LAT * DEG2RAD)
    y = (lat - REF_LAT) * DEG2RAD * R_EARTH
    return float(x), float(y), float(alt)


# ────────────────────────────────────────────────────────────────────
class SimWorld:

    SIM_DT    = 0.02      # s — 50 Hz
    WPN_COUNT = 4

    def __init__(self, seed: int = 42, platform1=None, platform2=None,
                 platforms=None, teams=None):
        self._rng    = np.random.default_rng(seed)
        # Library platforms (bvr_library.load_platform), one per aircraft.
        # None means the built-in F-16C / AIM-120 classes, as before the library.
        if platforms is None:
            platforms, teams = [platform1, platform2], [1, 2]
        if teams is None or len(teams) != len(platforms) or len(set(teams)) != 2:
            raise ValueError("SimWorld needs one team number per aircraft and exactly two teams")
        self.platforms = tuple(platforms)
        self.teams     = tuple(int(t) for t in teams)
        self.n         = len(self.platforms)
        self.acs       = [F16Aircraft(rng=self._rng, cfg=p.airframe if p else None)
                          for p in self.platforms]
        self.wpn_count = tuple(p.wpn_count if p else self.WPN_COUNT for p in self.platforms)
        self.missiles: list = []
        self._despawn_next: list = []
        self.events:   list = []
        self.t_sim     = 0.0
        self.episode_id = 0
        self.wpn       = [self.WPN_COUNT] * self.n
        self.alive     = [True] * self.n
        self._prev_fire = [0] * self.n
        self._rwr_t_warn: dict = {}

    # 1v1 names, kept for readers of the two-aircraft world
    ac1  = property(lambda self: self.acs[0])
    ac2  = property(lambda self: self.acs[1])
    wpn1 = property(lambda self: self.wpn[0], lambda self, v: self.wpn.__setitem__(0, v))
    wpn2 = property(lambda self: self.wpn[1], lambda self, v: self.wpn.__setitem__(1, v))

    def enemies(self, i: int) -> list:
        """Aircraft numbers of the other team, alive or not."""
        return [j for j in range(1, self.n + 1) if self.teams[j - 1] != self.teams[i - 1]]

    def friends(self, i: int) -> list:
        """Aircraft numbers of i's teammates, not counting i."""
        return [j for j in range(1, self.n + 1)
                if j != i and self.teams[j - 1] == self.teams[i - 1]]

    def pos(self, i: int) -> np.ndarray:
        a = self.acs[i - 1]
        return np.array([a.x, a.y, a.z])

    # ── episode setup ────────────────────────────────────────────────
    def reset(self, ic: dict, episode_id: int = 0) -> None:
        """
        ic keys, for each aircraft i (1..N): ac{i}_lat, ac{i}_lon, ac{i}_alt,
        ac{i}_psi, ac{i}_spd, and optionally wpn{i} (default: its platform's
        loadout; in 1v1 also the old names wpn and wpn_t).
        fuel_frac: optional, default 0.60.
        """
        self.t_sim       = 0.0
        self.episode_id  = int(episode_id)
        self.missiles    = []
        self._despawn_next = []
        self.events      = []
        self._prev_fire  = [0] * self.n
        self.alive       = [True] * self.n
        self._rwr_t_warn = {}
        legacy = {1: "wpn", 2: "wpn_t"} if self.n == 2 else {}
        self.wpn = [int(ic.get(f"wpn{i}", ic.get(legacy.get(i, ""), self.wpn_count[i - 1])))
                    for i in range(1, self.n + 1)]
        AIM120._counter = 0

        fuel = float(ic.get("fuel_frac", 0.60))
        for i, a in enumerate(self.acs, start=1):
            x, y, z = _latlon_to_enu(ic[f"ac{i}_lat"], ic[f"ac{i}_lon"], ic[f"ac{i}_alt"])
            a.reset_state(x, y, z, float(ic[f"ac{i}_psi"]), float(ic[f"ac{i}_spd"]), fuel)

    def kill(self, i: int, cause: str = "CRASH") -> None:
        """Remove aircraft i from the fight (a crash the env detected)."""
        if self.alive[i - 1]:
            self.alive[i - 1] = False
            self.events.append({"type": "AC_REMOVED", "ac": i, "cause": cause,
                                "t_sim": round(self.t_sim, 3)})

    # ── main step ───────────────────────────────────────────────────
    def step(self, cmd1: dict, cmd2: dict) -> dict:
        """
        1v1: advance one SIM_DT. Returns AC1's telemetry.
        cmd1 → AC1 (RL agent);   cmd2 → AC2 (opponent)
        """
        self.step_all([cmd1, cmd2])
        return self.telemetry(1)

    def step_all(self, cmds: list) -> None:
        """Advance one SIM_DT with one command per aircraft (ignored for dead ones)."""
        self.events = []

        # Remove missiles that hit or missed last frame
        self.missiles       = [m for m in self.missiles if m not in self._despawn_next]
        self._despawn_next  = []

        # ── guidance updates ─────────────────────────────────────────
        # From the shooter's own estimate when its command carries one;
        # scripted shooters send none and guide on the target's truth.
        for m in self.missiles:
            cmd = cmds[m.owner - 1]
            if "msl_guidance" in cmd:
                m.update_guidance(cmd["msl_guidance"])
            else:
                t = self.acs[m.target - 1]
                m.update_guidance({"valid": 1,
                                   "tgt_pos": [t.x, t.y, t.z],
                                   "tgt_vel": t.vel_enu.tolist(),
                                   "pos_sigma": 0.0,
                                   "t_est": self.t_sim})

        # ── fire commands (rising-edge) ──────────────────────────────
        for i, cmd in enumerate(cmds, start=1):
            f = int(cmd.get("fire", 0)) if self.alive[i - 1] else 0
            if f and not self._prev_fire[i - 1]:
                self._launch(i, cmd)
            self._prev_fire[i - 1] = f

        # ── step missiles ────────────────────────────────────────────
        ac_pos = {i: self.pos(i) for i in range(1, self.n + 1)}
        ac_vel = {i: a.vel_enu for i, a in enumerate(self.acs, start=1)}

        destroyed = []
        for m in list(self.missiles):
            tgt = m.target
            evs = m.step(self.SIM_DT, ac_pos[tgt], ac_vel[tgt])
            if not self.alive[tgt - 1]:
                # Nothing left to destroy: a wreck neither fuzes a missile
                # nor dies twice.
                evs = [e for e in evs if e.get("type") != "AC_DESTROYED"]
                for e in evs:
                    if e.get("type") == "MISSILE_HIT":
                        e["killed"] = 0
            self.events.extend(evs)
            destroyed += [e["ac"] for e in evs if e.get("type") == "AC_DESTROYED"]
            if m.phase in (MslPhase.HIT, MslPhase.MISS):
                self._despawn_next.append(m)

        # ── step aircraft ────────────────────────────────────────────
        for i, (a, cmd) in enumerate(zip(self.acs, cmds), start=1):
            if self.alive[i - 1]:
                a.step(self.SIM_DT, cmd)
        # Killed this frame: flown this frame (as the two-aircraft world did
        # on the frame that ended the episode), frozen from the next.
        for i in destroyed:
            self.alive[i - 1] = False
        self.t_sim += self.SIM_DT

    # ── missile launch ───────────────────────────────────────────────
    def _launch(self, owner: int, cmd: dict) -> None:
        if self.wpn[owner - 1] <= 0:
            return
        tgt_id = cmd.get("target")
        if tgt_id is None:
            live = [j for j in self.enemies(owner) if self.alive[j - 1]]
            if not live:
                return
            tgt_id = live[0]
        tgt_id = int(tgt_id)
        if not self.alive[tgt_id - 1] or self.teams[tgt_id - 1] == self.teams[owner - 1]:
            return

        ac_src = self.acs[owner - 1]
        ac_tgt = self.acs[tgt_id - 1]

        pos_src = np.array([ac_src.x, ac_src.y, ac_src.z])
        pos_tgt = np.array([ac_tgt.x, ac_tgt.y, ac_tgt.z])
        rng = float(np.linalg.norm(pos_tgt - pos_src))
        if rng > 150_000.0 or rng < 500.0:
            return

        # The shooter's missile type; its seeker, and so its hand-off range,
        # reach less far against a target with a smaller radar cross-section.
        p_src, p_tgt = self.platforms[owner - 1], self.platforms[tgt_id - 1]
        if p_src is None:
            mcfg, scale, lo, hi = None, 1.0, 8_000.0, 16_000.0
        else:
            mcfg = p_src.missile
            tgt_rcs = p_tgt.rcs if p_tgt is not None else mcfg.SEEKER_REF_RCS_M2
            scale = rcs_scale(tgt_rcs, mcfg.SEEKER_REF_RCS_M2)
            lo, hi = mcfg.HANDOFF_MIN * scale, mcfg.HANDOFF_MAX * scale
        handoff = float(self._rng.uniform(lo, hi))
        m = AIM120(owner, tgt_id, pos_src, ac_src.vel_enu.copy(), handoff,
                   cfg=mcfg, seeker_range_scale=scale)
        m.last_tgt_pos = pos_tgt.copy()
        m.last_tgt_vel = ac_tgt.vel_enu.copy()
        m.t_launch     = self.t_sim
        self._rwr_t_warn[m.id] = self.t_sim

        self.missiles.append(m)
        self.wpn[owner - 1] -= 1
        self.events.append({"type": "MISSILE_LAUNCH", "id": m.id,
                             "owner": owner, "t_sim": round(self.t_sim, 3)})

    # ── telemetry builder ────────────────────────────────────────────
    def _label(self, j: int, me: int, other: int, view: str, aimed_at: int = None) -> int:
        """
        Aircraft number j as `me` sees it (see the module docstring). For a
        missile's owner, aimed_at is its target: an enemy missile counts as
        the threat (2) only when it is aimed at me.
        """
        if view == "team":
            return 1 if self.teams[j - 1] == self.teams[me - 1] else 2
        if j == me:
            return 1
        if self.teams[j - 1] == self.teams[me - 1]:
            return 3
        if aimed_at is not None:
            return 2 if aimed_at == me else 4
        return 2 if j == other else 4

    def telemetry(self, me: int = 1, other: int = None, view: str = "agent") -> dict:
        """
        Telemetry from aircraft `me`'s point of view against enemy `other`:
        unsuffixed keys are `me`, `_t` keys `other`, and missile/event ids are
        relabelled (module docstring). telemetry(1) is the agent's packet in
        1v1; telemetry(2) lets a self-play opponent observe the fight exactly
        the way the agent does.
        """
        if other is None:
            live = [j for j in self.enemies(me) if self.alive[j - 1]]
            other = (live or self.enemies(me))[0]
        a1, a2 = self.acs[me - 1], self.acs[other - 1]
        lat1, lon1, alt1 = _enu_to_latlon(a1.x, a1.y, a1.z)
        lat2, lon2, alt2 = _enu_to_latlon(a2.x, a2.y, a2.z)

        # ── engagement geometry ──────────────────────────────────────
        dx, dy, dz = a2.x - a1.x, a2.y - a1.y, a2.z - a1.z
        rng   = max(math.sqrt(dx*dx + dy*dy + dz*dz), 1.0)
        los_u = np.array([dx, dy, dz]) / rng
        v1, v2 = a1.vel_enu, a2.vel_enu
        closure = float(-np.dot(v2 - v1, los_u))

        # heading unit vectors (horizontal, for aspect angles)
        fwd1 = np.array([math.sin(a1.chi), math.cos(a1.chi), 0.0])
        fwd2 = np.array([math.sin(a2.chi), math.cos(a2.chi), 0.0])
        aa_deg   = math.degrees(math.acos(float(np.clip(np.dot(fwd1,  los_u), -1, 1))))
        aa_deg_t = math.degrees(math.acos(float(np.clip(np.dot(fwd2, -los_u), -1, 1))))
        hca_deg  = math.degrees(math.acos(float(np.clip(np.dot(fwd1,  fwd2),  -1, 1))))
        v_rel    = v2 - v1
        los_rate = float(np.linalg.norm(np.cross(los_u, v_rel))) * RAD2DEG

        # ── RWR synthesis ─────────────────────────────────────────────
        pos1 = np.array([a1.x, a1.y, a1.z])
        my_team = self.teams[me - 1]
        inbound = [m for m in self.missiles
                   if self.teams[m.owner - 1] != my_team and m.target == me
                   and m.phase not in (MslPhase.HIT, MslPhase.MISS)]
        if inbound:
            cl = min(inbound, key=lambda m: float(np.linalg.norm(m.pos - pos1)))
            d  = cl.pos - pos1
            bear = math.atan2(float(d[0]), float(d[1]))
            rwr  = {"launch_warn":    1,
                    "launch_bearing": round(bear, 4),
                    "t_warn":         self._rwr_t_warn.get(cl.id, self.t_sim),
                    "spike":          int(float(np.linalg.norm(d)) < 25_000.0),
                    "seeker_active":  int(cl.seeker_active),
                    "n_threats":      len(inbound)}
        else:
            rwr = {"launch_warn": 0, "launch_bearing": 0.0, "t_warn": 0.0,
                   "spike": 0, "seeker_active": 0, "n_threats": 0}

        # ── missile list (live + ones despawning this frame) ─────────
        identity = self.n == 2 and me == 1 and view == "agent"
        msl_list = []
        for m in self.missiles + self._despawn_next:
            d = m.to_dict()
            la, lo, _ = _enu_to_latlon(*m.pos)
            d["lat"] = round(la, 6); d["lon"] = round(lo, 6)
            if not identity:
                d["owner"]  = self._label(m.owner, me, other, view, aimed_at=m.target)
                d["target"] = self._label(m.target, me, other, view)
            msl_list.append(d)
        events = list(self.events)
        if not identity:
            def relabel(ev):
                out = {}
                for k, v in ev.items():
                    if k in ("owner", "target", "ac") and isinstance(v, int) and 1 <= v <= self.n:
                        v = self._label(v, me, other, view,
                                        aimed_at=ev.get("target") if k == "owner" else None)
                    out[k] = v
                return out
            events = [relabel(ev) for ev in events]

        return {
            "type":       "telemetry",
            "episode_id": self.episode_id,
            "t_sim":      round(self.t_sim, 4),

            # ── AC1 (ownship) ─────────────────────────────────────────
            "lat":   lat1, "lon":   lon1, "alt":   alt1,
            "psi":   a1.chi,   "theta": a1.theta, "phi":   a1.phi,
            "alpha": a1.alpha, "beta":  a1.beta,
            "p":     a1.p,     "q":     a1.q,     "r_body": a1.r,
            "nz":    a1.nz,    "speed": a1.V,     "mach":  a1.mach,
            "gamma_fpa": a1.gamma, "fuel_frac": a1.fuel_frac,
            "maneuver_running": False,

            # ── AC2 truth (privileged critic + termination) ───────────
            "lat_t":  lat2, "lon_t":  lon2, "alt_t":  alt2,
            "psi_t":  a2.chi,   "theta_t": a2.theta, "phi_t":  a2.phi,
            "speed_t": a2.V,    "mach_t":  a2.mach,  "nz_t":   a2.nz,
            "gamma_fpa_t": a2.gamma,
            "range":   round(rng, 1),
            "closure": round(closure, 2),
            "aa_deg":  round(aa_deg, 2),
            "aa_deg_t": round(aa_deg_t, 2),
            "hca_deg": round(hca_deg, 2),
            "los_rate": round(los_rate, 4),
            "ata_deg": round(aa_deg, 2),

            # ── weapons ───────────────────────────────────────────────
            "wpn_remaining":   self.wpn[me - 1],
            "wpn_remaining_t": self.wpn[other - 1],
            "wpn_ready": 1, "wpn_ready_t": 1,

            # ── missiles ──────────────────────────────────────────────
            "missiles": msl_list,

            # ── RWR ───────────────────────────────────────────────────
            "rwr": rwr,

            # ── events (consumed once per frame) ──────────────────────
            "events": events,
        }
