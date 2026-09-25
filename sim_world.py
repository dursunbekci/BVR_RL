"""
sim_world.py  —  Simulation world
==================================

Two F-16 aircraft + list of AIM-120 missiles + per-frame event queue.

  world = SimWorld(seed=42)
  world.reset(ic, episode_id=7)
  for ...:
      tlm = world.step(cmd1, cmd2)  # one SIM_DT advance

`tlm` is a dict with exactly the field names bvr_env.py reads via _ingest(),
so the env needs no awareness of whether it is talking to a C++ process or
to this class.

cmd1 controls AC1 (RL agent), cmd2 controls AC2 (opponent).
AC1's missile guidance comes from cmd1["msl_guidance"] (the radar estimate).
AC2's comes from cmd2["msl_guidance"] when the opponent supplies one (a
self-play policy guiding from its own radar track); scripted opponents send
none and get truth guidance.

telemetry(me) builds the same packet from either aircraft's point of view.
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

    def __init__(self, seed: int = 42, platform1=None, platform2=None):
        self._rng    = np.random.default_rng(seed)
        # Library platforms (bvr_library.load_platform) for AC1 and AC2. None
        # means the built-in F-16C / AIM-120 classes, as before the library.
        self.platforms = (platform1, platform2)
        self.ac1     = F16Aircraft(rng=self._rng, cfg=platform1.airframe if platform1 else None)
        self.ac2     = F16Aircraft(rng=self._rng, cfg=platform2.airframe if platform2 else None)
        self.wpn_count = tuple(p.wpn_count if p else self.WPN_COUNT for p in self.platforms)
        self.missiles: list = []
        self._despawn_next: list = []
        self.events:   list = []
        self.t_sim     = 0.0
        self.episode_id = 0
        self.wpn1      = self.WPN_COUNT
        self.wpn2      = self.WPN_COUNT
        self._prev_fire1 = 0
        self._prev_fire2 = 0
        self._rwr_t_warn: dict = {}

    # ── episode setup ────────────────────────────────────────────────
    def reset(self, ic: dict, episode_id: int = 0) -> None:
        """
        ic keys: ac1_lat, ac1_lon, ac1_alt, ac1_psi, ac1_spd,
                 ac2_lat, ac2_lon, ac2_alt, ac2_psi, ac2_spd,
                 wpn, wpn_t            (optional, default WPN_COUNT)
                 fuel_frac             (optional, default 0.60)
        """
        self.t_sim       = 0.0
        self.episode_id  = int(episode_id)
        self.missiles    = []
        self._despawn_next = []
        self.events      = []
        self._prev_fire1 = 0
        self._prev_fire2 = 0
        self._rwr_t_warn = {}
        self.wpn1 = int(ic.get("wpn",   self.wpn_count[0]))
        self.wpn2 = int(ic.get("wpn_t", self.wpn_count[1]))
        AIM120._counter = 0

        fuel = float(ic.get("fuel_frac", 0.60))
        x1, y1, z1 = _latlon_to_enu(ic["ac1_lat"], ic["ac1_lon"], ic["ac1_alt"])
        x2, y2, z2 = _latlon_to_enu(ic["ac2_lat"], ic["ac2_lon"], ic["ac2_alt"])
        self.ac1.reset_state(x1, y1, z1, float(ic["ac1_psi"]), float(ic["ac1_spd"]), fuel)
        self.ac2.reset_state(x2, y2, z2, float(ic["ac2_psi"]), float(ic["ac2_spd"]), fuel)

    # ── main step ───────────────────────────────────────────────────
    def step(self, cmd1: dict, cmd2: dict) -> dict:
        """
        Advance one SIM_DT. Returns telemetry dict identical in schema to the
        C++ telemetry packets bvr_env.py's _ingest() was reading.
        cmd1 → AC1 (RL agent);   cmd2 → AC2 (scripted opponent)
        """
        self.events = []

        # Remove missiles that hit or missed last frame
        self.missiles       = [m for m in self.missiles if m not in self._despawn_next]
        self._despawn_next  = []

        # ── guidance updates ─────────────────────────────────────────
        g1 = cmd1.get("msl_guidance")        # estimate from AC1's radar
        if "msl_guidance" in cmd2:
            g2 = cmd2["msl_guidance"]        # self-play: AC2's own radar estimate
        else:
            g2 = {                           # scripted: tracks AC1 perfectly
                "valid": 1,
                "tgt_pos": [self.ac1.x, self.ac1.y, self.ac1.z],
                "tgt_vel": self.ac1.vel_enu.tolist(),
                "pos_sigma": 0.0,
                "t_est": self.t_sim,
            }
        for m in self.missiles:
            m.update_guidance(g1 if m.owner == 1 else g2)

        # ── fire commands (rising-edge) ──────────────────────────────
        f1 = int(cmd1.get("fire", 0))
        f2 = int(cmd2.get("fire", 0))
        if f1 and not self._prev_fire1: self._launch(1, cmd1)
        if f2 and not self._prev_fire2: self._launch(2, cmd2)
        self._prev_fire1 = f1
        self._prev_fire2 = f2

        # ── step missiles ────────────────────────────────────────────
        pos1 = np.array([self.ac1.x, self.ac1.y, self.ac1.z])
        pos2 = np.array([self.ac2.x, self.ac2.y, self.ac2.z])
        vel1 = self.ac1.vel_enu
        vel2 = self.ac2.vel_enu
        ac_pos = {1: pos1, 2: pos2}
        ac_vel = {1: vel1, 2: vel2}

        for m in list(self.missiles):
            tgt = m.target
            evs = m.step(self.SIM_DT, ac_pos[tgt], ac_vel[tgt])
            self.events.extend(evs)
            if m.phase in (MslPhase.HIT, MslPhase.MISS):
                self._despawn_next.append(m)

        # ── step aircraft ────────────────────────────────────────────
        self.ac1.step(self.SIM_DT, cmd1)
        self.ac2.step(self.SIM_DT, cmd2)
        self.t_sim += self.SIM_DT

        return self.telemetry(1)

    # ── missile launch ───────────────────────────────────────────────
    def _launch(self, owner: int, cmd: dict) -> None:
        wpn = self.wpn1 if owner == 1 else self.wpn2
        if wpn <= 0:
            return

        ac_src = self.ac1 if owner == 1 else self.ac2
        ac_tgt = self.ac2 if owner == 1 else self.ac1
        tgt_id = 2      if owner == 1 else 1

        pos_src = np.array([ac_src.x, ac_src.y, ac_src.z])
        pos_tgt = np.array([ac_tgt.x, ac_tgt.y, ac_tgt.z])
        rng = float(np.linalg.norm(pos_tgt - pos_src))
        if rng > 150_000.0 or rng < 500.0:
            return

        # The shooter's missile type; its seeker, and so its hand-off range,
        # reach less far against a target with a smaller radar cross-section.
        p_src, p_tgt = self.platforms[owner - 1], self.platforms[2 - owner]
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
        if owner == 1: self.wpn1 -= 1
        else:          self.wpn2 -= 1
        self.events.append({"type": "MISSILE_LAUNCH", "id": m.id,
                             "owner": owner, "t_sim": round(self.t_sim, 3)})

    # ── telemetry builder ────────────────────────────────────────────
    def telemetry(self, me: int = 1) -> dict:
        """
        Telemetry from aircraft `me`'s point of view: unsuffixed keys are `me`,
        `_t` keys the other aircraft, and missile/event ids are relabelled so
        that 1 always means `me`. telemetry(1) is the agent's packet;
        telemetry(2) lets a self-play opponent observe the fight exactly the
        way the agent does.
        """
        a1, a2 = (self.ac1, self.ac2) if me == 1 else (self.ac2, self.ac1)
        other = 3 - me
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
        inbound = [m for m in self.missiles
                   if m.owner == other and m.target == me
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
        msl_list = []
        for m in self.missiles + self._despawn_next:
            d = m.to_dict()
            la, lo, _ = _enu_to_latlon(*m.pos)
            d["lat"] = round(la, 6); d["lon"] = round(lo, 6)
            if me == 2:
                d["owner"], d["target"] = 3 - d["owner"], 3 - d["target"]
            msl_list.append(d)
        events = list(self.events)
        if me == 2:
            events = [{k: (3 - v if k in ("owner", "target", "ac") and v in (1, 2) else v)
                       for k, v in ev.items()} for ev in events]

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
            "wpn_remaining":   self.wpn1 if me == 1 else self.wpn2,
            "wpn_remaining_t": self.wpn2 if me == 1 else self.wpn1,
            "wpn_ready": 1, "wpn_ready_t": 1,

            # ── missiles ──────────────────────────────────────────────
            "missiles": msl_list,

            # ── RWR ───────────────────────────────────────────────────
            "rwr": rwr,

            # ── events (consumed once per frame) ──────────────────────
            "events": events,
        }
