"""
missile_sim.py  —  AIM-120 3-DOF point-mass model
==================================================

Implements the full contract from MISSILE_SPEC.md in pure Python.

Key design decisions:
  - Guide on the PARENT'S RADAR ESTIMATE (last_tgt_pos / last_tgt_vel),
    never on truth during midcourse. Truth is used only by the autonomous
    seeker in terminal phase and for endgame detection.
  - 3-second support timeout: if the datalink goes silent for > 3 s during
    midcourse, emit MISSILE_MISS {cause: SUPPORT_LOST} and despawn.
  - Handoff range randomised U(8, 16) km per missile at launch. The policy
    must read needs_support from telemetry — it cannot learn the exact number.
  - Deterministic kill inside fuze radius. Probabilistic pk_roll can be added
    later once the policy shoots reliably.
"""

import math
import numpy as np
from enum import Enum

G = 9.80665


class MslPhase(Enum):
    BOOST     = "BOOST"
    MIDCOURSE = "MIDCOURSE"
    TERMINAL  = "TERMINAL"
    HIT       = "HIT"
    MISS      = "MISS"


# ── AIM-120C-5 approximate parameters ───────────────────────────────
class MslCfg:
    MASS0            = 152.0      # kg at launch
    MASS_PROP        = 48.0       # kg propellant
    BOOST_TIME       = 6.0        # s motor burn duration
    THRUST           = 17_000.0   # N average thrust during boost
    S_REF            = 0.0324     # m² body cross-section (203 mm diam)
    N_PRO_NAV        = 4.0        # proportional-navigation gain
    MAX_G            = 30.0       # peak manoeuvre capability, g
    FUZE_RADIUS      = 10.0       # m kill-zone radius
    SEEKER_FOV       = 0.35       # rad half-angle (~20°)
    SEEKER_RANGE     = 18_000.0   # m seeker acquisition range at handoff
    SUPPORT_TIMEOUT  = 3.0        # s — the rule
    MAX_FLIGHT       = 120.0      # s hard lifetime
    MIN_MANOEUVRE_V  = 200.0      # m/s below which manoeuvrability degrades


def _isa_rho(alt_m: float) -> float:
    h = float(np.clip(alt_m, 0.0, 20_000.0))
    if h <= 11_000.0:
        T = 288.15 - 0.0065 * h
        p = 101_325.0 * (T / 288.15) ** 5.2561
    else:
        T = 216.65
        p = 22_632.1 * math.exp(-0.0001577 * (h - 11_000.0))
    return p / (287.05 * T)


def _isa_a(alt_m: float) -> float:
    h = float(np.clip(alt_m, 0.0, 11_000.0))
    T = 288.15 - 0.0065 * h
    return math.sqrt(1.4 * 287.05 * T)


def _drag_cd(mach: float) -> float:
    """Simplified body drag for a slender supersonic missile."""
    m = float(np.clip(mach, 0.0, 5.0))
    if m < 0.80: return 0.28
    if m < 1.00: return 0.28 + 0.70 * (m - 0.80) / 0.20
    if m < 1.50: return 0.98 - 0.22 * (m - 1.00) / 0.50
    return max(0.50, 0.76 - 0.15 * (m - 1.50))


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9: return 0.0
    return math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


# ── event factory helpers ────────────────────────────────────────────
def _ev(type_, **kw): return {"type": type_, **kw}


class AIM120:
    """
    One AIM-120 instance. Owns its full state. Call step() each sim frame.
    update_guidance() should be called before step() each frame.
    """

    _counter = 0

    def __init__(self, owner: int, target: int,
                 pos: np.ndarray, vel: np.ndarray,
                 handoff_range: float):
        AIM120._counter += 1
        self.id     = AIM120._counter
        self.owner  = int(owner)
        self.target = int(target)

        self.pos  = np.asarray(pos, dtype=np.float64).copy()
        self.vel  = np.asarray(vel, dtype=np.float64).copy()
        self.mass = float(MslCfg.MASS0)

        self.handoff_range   = float(handoff_range)
        self.phase           = MslPhase.BOOST
        self.t_flight        = 0.0
        self.t_since_update  = 0.0
        self.seeker_active   = False
        self.guidance_valid  = True   # seeded at launch

        # Last known target state (from datalink). These are ESTIMATES, not truth.
        self.last_tgt_pos = pos.copy()
        self.last_tgt_vel = np.zeros(3, dtype=np.float64)

        self.prev_range        = 1e12
        self._rng_to_tgt_cache = 0.0
        self.t_launch          = 0.0

    # ── datalink update (called from sim_world before step()) ────────
    def update_guidance(self, guidance: dict | None) -> None:
        """
        guidance must come from the parent's RADAR ESTIMATE, not from truth.
        This is the line that makes track quality affect missile Pk.
        """
        if guidance is None or not guidance.get("valid", 0):
            self.guidance_valid = False
            return
        self.guidance_valid = True
        self.last_tgt_pos   = np.array(guidance["tgt_pos"], dtype=np.float64)
        self.last_tgt_vel   = np.array(guidance["tgt_vel"], dtype=np.float64)
        self.t_since_update = 0.0

    # ── integration step ─────────────────────────────────────────────
    def step(self, dt: float,
             tgt_pos_true: np.ndarray,
             tgt_vel_true: np.ndarray) -> list:
        """
        Advance one time step.
        tgt_pos_true / tgt_vel_true are ONLY used for:
          - seeker-active terminal guidance
          - endgame detection (closest approach)
        NOT for midcourse. That would invalidate the radar model.
        """
        if self.phase in (MslPhase.HIT, MslPhase.MISS):
            return []

        events = []
        self.t_flight += dt
        alt   = float(self.pos[2])
        speed = float(np.linalg.norm(self.vel))
        rho   = _isa_rho(alt)
        a_snd = _isa_a(alt)
        mach  = speed / max(a_snd, 1.0)
        q_dyn = 0.5 * rho * max(speed, 1.0) ** 2
        rng_to_tgt = float(np.linalg.norm(tgt_pos_true - self.pos))

        # ── phase machine ────────────────────────────────────────────
        if self.phase == MslPhase.BOOST and self.t_flight >= MslCfg.BOOST_TIME:
            self.phase = MslPhase.MIDCOURSE

        # Seeker handoff
        if (self.phase == MslPhase.MIDCOURSE and not self.seeker_active
                and rng_to_tgt <= self.handoff_range):
            look = _angle_between(tgt_pos_true - self.pos, self.vel)
            if look < MslCfg.SEEKER_FOV and rng_to_tgt < MslCfg.SEEKER_RANGE:
                self.seeker_active = True
                self.phase = MslPhase.TERMINAL

        # ── 3-second support rule ─────────────────────────────────────
        if (self.phase in (MslPhase.BOOST, MslPhase.MIDCOURSE)
                and not self.seeker_active):
            if not self.guidance_valid:
                self.t_since_update += dt
                if self.t_since_update > MslCfg.SUPPORT_TIMEOUT:
                    self.phase = MslPhase.MISS
                    events.append(_ev("MISSILE_MISS", id=self.id, owner=self.owner,
                                      miss_dist=round(rng_to_tgt, 1),
                                      cause="SUPPORT_LOST"))
                    return events
            # guidance_valid: t_since_update reset in update_guidance()

        # ── guidance ─────────────────────────────────────────────────
        if self.seeker_active:
            aim_pos = tgt_pos_true
            aim_vel = tgt_vel_true
        else:
            # Midcourse: aim at extrapolated last-known target position.
            # NOTE: if support was valid this frame, t_since_update = 0, so
            # this uses last_tgt_pos directly (updated by update_guidance).
            aim_pos = self.last_tgt_pos + self.last_tgt_vel * self.t_since_update
            aim_vel = self.last_tgt_vel

        a_nav = self._pro_nav(aim_pos, aim_vel)

        # ── thrust ───────────────────────────────────────────────────
        if self.t_flight < MslCfg.BOOST_TIME:
            spd_dir  = self.vel / max(speed, 1.0)
            a_thrust = (MslCfg.THRUST / self.mass) * spd_dir
            burn_rate = MslCfg.MASS_PROP / MslCfg.BOOST_TIME
            self.mass = max(MslCfg.MASS0 - MslCfg.MASS_PROP, self.mass - burn_rate * dt)
        else:
            a_thrust = np.zeros(3)

        # ── drag ─────────────────────────────────────────────────────
        cd    = _drag_cd(mach)
        a_drag = -(q_dyn * cd * MslCfg.S_REF / self.mass) * (self.vel / max(speed, 1.0))

        # ── clamp PN acceleration ────────────────────────────────────
        # Available g falls with dynamic pressure (a real missile bleeds
        # manoeuvrability at altitude — this is what makes NEZ narrow on
        # late, high-altitude shots).
        g_scale = min(1.0, q_dyn / 35_000.0)
        if speed < MslCfg.MIN_MANOEUVRE_V:
            g_scale *= 0.15
        g_avail = MslCfg.MAX_G * G * g_scale
        an = float(np.linalg.norm(a_nav))
        if an > g_avail:
            a_nav = a_nav * (g_avail / an)

        # ── integrate ────────────────────────────────────────────────
        gravity = np.array([0.0, 0.0, -G])
        accel   = a_thrust + a_drag + a_nav + gravity
        self.vel = self.vel + accel * dt
        self.pos = self.pos + self.vel * dt

        # ── endgame: closest approach detection ──────────────────────
        rng_now = float(np.linalg.norm(tgt_pos_true - self.pos))
        self._rng_to_tgt_cache = rng_now

        # A guard of 500 m prevents false triggers while the missile is
        # turning onto its intercept course at launch.
        if rng_now > self.prev_range and self.prev_range < 500.0:
            md = self.prev_range
            if md < MslCfg.FUZE_RADIUS:
                self.phase = MslPhase.HIT
                events.append(_ev("MISSILE_HIT", id=self.id, owner=self.owner,
                                   target=self.target,
                                   miss_dist=round(md, 2), killed=1))
                events.append(_ev("AC_DESTROYED", ac=self.target, cause="MISSILE"))
            else:
                self.phase = MslPhase.MISS
                events.append(_ev("MISSILE_MISS", id=self.id, owner=self.owner,
                                   miss_dist=round(md, 1), cause="ENDGAME"))

        self.prev_range = rng_now

        # ── lifetime / terrain ───────────────────────────────────────
        if self.phase not in (MslPhase.HIT, MslPhase.MISS):
            if self.t_flight > MslCfg.MAX_FLIGHT or self.pos[2] < 0.0:
                self.phase = MslPhase.MISS
                events.append(_ev("MISSILE_MISS", id=self.id, owner=self.owner,
                                   miss_dist=round(rng_now, 1), cause="KINEMATIC"))

        return events

    # ── proportional navigation ──────────────────────────────────────
    def _pro_nav(self, aim_pos: np.ndarray, aim_vel: np.ndarray) -> np.ndarray:
        los = aim_pos - self.pos
        rng = float(np.linalg.norm(los))
        if rng < 1.0:
            return np.zeros(3)
        los_hat = los / rng
        v_rel   = aim_vel - self.vel
        closing = float(-np.dot(v_rel, los_hat))

        omega = np.cross(los, v_rel) / (rng ** 2)
        a_cmd = np.cross(omega * MslCfg.N_PRO_NAV, self.vel)
        if closing < 0.0:         # target opening: reduce effort
            a_cmd *= 0.25
        return a_cmd

    # ── properties ───────────────────────────────────────────────────
    @property
    def tgo_est(self) -> float:
        aim = (self.last_tgt_pos + self.last_tgt_vel * self.t_since_update
               if not self.seeker_active else self.last_tgt_pos)
        los = aim - self.pos
        rng = float(np.linalg.norm(los))
        los_hat = los / max(rng, 1.0)
        closing = float(-np.dot(self.vel, los_hat))
        return rng / closing if closing > 10.0 else 999.0

    @property
    def needs_support(self) -> bool:
        return (not self.seeker_active
                and self.phase in (MslPhase.BOOST, MslPhase.MIDCOURSE))

    def to_dict(self) -> dict:
        spd  = float(np.linalg.norm(self.vel))
        alt  = float(self.pos[2])
        a_s  = _isa_a(alt)
        return {
            "id":             self.id,
            "owner":          self.owner,
            "target":         self.target,
            "state":          self.phase.value,
            "lat":            0.0,   # filled by sim_world
            "lon":            0.0,
            "alt":            alt,
            "spd":            round(spd, 1),
            "mach":           round(spd / max(a_s, 1.0), 3),
            "t_flight":       round(self.t_flight, 3),
            "tgo_est":        round(self.tgo_est, 2),
            "seeker_active":  int(self.seeker_active),
            "needs_support":  int(self.needs_support),
            "t_since_update": round(self.t_since_update, 3),
            "handoff_range":  round(self.handoff_range, 0),
            "guidance_valid": int(self.guidance_valid),
            "rng_to_target":  round(self._rng_to_tgt_cache, 1),
            # ENU velocity — lets the display point the missile along its flight path
            "vel":            [round(float(self.vel[0]), 1),
                               round(float(self.vel[1]), 1),
                               round(float(self.vel[2]), 1)],
        }
