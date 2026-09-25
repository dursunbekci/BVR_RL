"""
bvr_opponents.py  —  Scripted BVR opponents (AC2)
=================================================

Same contract as opponents.py: act(state, t_sim) returns an UNSUFFIXED command
dict; bvr_env adds the `_t` suffix.

Curriculum, easiest → hardest:
    STRAIGHT → EVASIVE → NOTCHER → SHOOTER → ADAPTIVE_SHOOTER → SELF_PLAY

SELF_PLAY opponents are frozen policy snapshots, built by bvr_env through
bvr_selfplay.py rather than by create(), which only builds scripted ones.

SHOOTER is the important one. A rule-based BVR opponent that commits, shoots,
cranks and drags is a genuinely strong baseline — strong enough that beating
it consistently is a real result. Build it before self-play, and keep it in
the self-play pool permanently: without a committed aggressor in the pool,
self-play in BVR collapses into a mutual-disengagement equilibrium where both
policies turn cold at 80 km and nobody ever learns to fight.

These opponents see the same TRUTH state the env has. That is a deliberate
asymmetry in the agent's disfavour for the scripted stages — it means the
scripted shooter never loses track, so the agent must earn its wins against a
perfectly-informed enemy. Self-play removes the asymmetry.
"""

import math
import time
from enum import Enum, auto

import numpy as np

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


class BvrOpponentType(Enum):
    STRAIGHT = auto()          # non-manoeuvring, unarmed — learn search/track/shoot
    EVASIVE = auto()           # manoeuvres, unarmed — learn re-acquisition
    NOTCHER = auto()           # actively beams to break lock, unarmed
    SHOOTER = auto()           # armed, fixed doctrine
    ADAPTIVE_SHOOTER = auto()  # armed, randomised doctrine parameters
    SELF_PLAY = auto()         # policy snapshots; built by bvr_selfplay


CURRICULUM = [
    BvrOpponentType.STRAIGHT,
    BvrOpponentType.EVASIVE,
    BvrOpponentType.NOTCHER,
    BvrOpponentType.SHOOTER,
    BvrOpponentType.ADAPTIVE_SHOOTER,
    BvrOpponentType.SELF_PLAY,
]


def advance_curriculum(current: BvrOpponentType) -> BvrOpponentType:
    try:
        i = CURRICULUM.index(current)
    except ValueError:
        return current
    return CURRICULUM[min(i + 1, len(CURRICULUM) - 1)]


class BvrOpponent:

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
        self.reset({})

    def reset(self, ic: dict):
        self._base_psi = float(ic.get("ac2_psi", 0.0))
        self._base_alt = float(ic.get("ac2_alt", 9000.0))
        self._base_spd = float(ic.get("ac2_spd", 290.0))
        self._t0 = None

    def act(self, state: dict, t_sim: float) -> dict:
        raise NotImplementedError

    @staticmethod
    def create(opp_type: "BvrOpponentType", rng=None) -> "BvrOpponent":
        cls = {
            BvrOpponentType.STRAIGHT: StraightOpponent,
            BvrOpponentType.EVASIVE: EvasiveOpponent,
            BvrOpponentType.NOTCHER: NotchOpponent,
            BvrOpponentType.SHOOTER: ShooterOpponent,
            BvrOpponentType.ADAPTIVE_SHOOTER: AdaptiveShooterOpponent,
        }.get(opp_type)
        # No silent fallback: defaulting to STRAIGHT once turned SELF_PLAY into
        # the easiest opponent while every log still reported SELF_PLAY.
        if cls is None:
            raise NotImplementedError(
                f"create() builds scripted opponents only; {opp_type.name} is "
                "built by bvr_env via bvr_selfplay.make_selfplay_opponent()")
        return cls(rng=rng)

    # ── helpers ─────────────────────────────────────────────────────
    def _cmd(self, hdg=0.0, alt=9000.0, V=290.0, fire=0) -> dict:
        return {
            "mode": 0,
            "chiDot": 0.0, "gamma": 0.0,
            "V": float(V),
            "hdgCmd": float(hdg % (2 * math.pi)),
            "altTarget": float(np.clip(alt, 1000.0, 14000.0)),
            "altFPA": 25.0 * DEG2RAD,
            "hdgTurnRate": 12.0 * DEG2RAD,
            "hdgRelative": 0.0,
            "maneuver": "NONE", "task": "NONE",
            "radar_cmd": 1,
            "fire": int(fire),
            "t": time.time(),
        }

    @staticmethod
    def _bearing_to_ac1(state: dict) -> float:
        """
        Bearing from AC2 to AC1, rad. Derived from AC2's own heading and the
        aspect angle the C++ side already computes (aa_deg_t = AC2 nose off AC1).
        """
        psi_t = state.get("psi_t", 0.0)
        aa_t = state.get("aa_deg_t", 180.0) * DEG2RAD
        return (psi_t + aa_t) % (2 * math.pi)


# ─────────────────────────────────────────────────────────────────────
class StraightOpponent(BvrOpponent):
    """Constant heading/altitude/speed. Stage 1 target drone."""

    def act(self, state, t_sim):
        return self._cmd(self._base_psi, self._base_alt, self._base_spd)


class EvasiveOpponent(BvrOpponent):
    """
    Slow S-turns plus altitude changes. Unarmed. Forces the agent to keep the
    radar pointed and to handle a target whose aspect keeps changing.
    """

    def reset(self, ic):
        super().reset(ic)
        self._period = float(self.rng.uniform(25.0, 45.0))
        self._amp = float(self.rng.uniform(35.0, 70.0)) * DEG2RAD
        self._alt_amp = float(self.rng.uniform(800.0, 2200.0))
        self._phase = float(self.rng.uniform(0.0, 2 * math.pi))

    def act(self, state, t_sim):
        w = 2 * math.pi / self._period
        hdg = self._base_psi + self._amp * math.sin(w * t_sim + self._phase)
        alt = self._base_alt + self._alt_amp * math.sin(0.6 * w * t_sim)
        return self._cmd(hdg, alt, self._base_spd)


class NotchOpponent(BvrOpponent):
    """
    Beams the moment it is being tracked, i.e. turns to put the threat at 90°
    to null out closure and fall into the radar's doppler notch. Unarmed.
    Teaches the agent that a track is not permanent.
    """

    def reset(self, ic):
        super().reset(ic)
        self._notch_side = 1.0 if self.rng.random() < 0.5 else -1.0
        self._notch_until = -1.0

    def act(self, state, t_sim):
        spiked = bool(state.get("rwr_t", {}).get("spike", 0)) if isinstance(
            state.get("rwr_t"), dict) else False
        rng = state.get("range", 1e5)

        if spiked or rng < 90_000.0:
            if t_sim > self._notch_until:
                self._notch_until = t_sim + float(self.rng.uniform(12.0, 25.0))
                self._notch_side *= -1.0
            brg = self._bearing_to_ac1(state)
            hdg = brg + self._notch_side * math.pi / 2
            # Descend while beaming — ground clutter helps the defender.
            return self._cmd(hdg, self._base_alt - 2000.0, 320.0)

        return self._cmd(self._base_psi, self._base_alt, self._base_spd)


class ShooterOpponent(BvrOpponent):
    """
    Fixed BVR doctrine. This is the benchmark opponent.

        COMMIT   fly hot, close to launch range
        SHOOT    fire once inside SHOT_RMAX_FRAC of its true R-max on the
                 target (calibrated envelope; the env sets self.rmax_t)
        CRANK    turn to crank_angle — keeps the target inside the radar
                 gimbal while cutting closure, so the missile keeps its
                 datalink but our own exposure drops
        DRAG     once the missile goes autonomous (needs_support == 0),
                 turn fully cold and run
        DEFEND   if spiked by an inbound missile, beam then dive

    The CRANK→DRAG transition on `needs_support` is the behaviour the agent
    has to learn to exploit: the moment the bandit drags, its own track on
    us degrades, and that is the window to close.
    """

    # Fraction of true R-max at which it shoots. This was a fixed 55 km, set
    # against the old analytic envelope; the calibrated missile's head-on reach
    # at these conditions is 36-49 km, so every shot flew out of range and
    # SHOOTER never killed anything.
    SHOT_RMAX_FRAC = 0.80
    CRANK_ANGLE = 50.0 * DEG2RAD
    REATTACK_RANGE = 40_000.0
    MIN_SHOT_INTERVAL = 12.0

    def reset(self, ic):
        super().reset(ic)
        self._phase = "COMMIT"
        self._crank_side = 1.0 if self.rng.random() < 0.5 else -1.0
        self._last_shot = -999.0
        self._fire_edge = False

    def _own_missiles(self, state):
        out = []
        for m in state.get("missiles", []) or []:
            if m.get("owner") == 2 and m.get("state") not in ("HIT", "MISS", "DUD"):
                out.append(m)
        return out

    def _inbound(self, state):
        for m in state.get("missiles", []) or []:
            if m.get("owner") == 1 and m.get("state") not in ("HIT", "MISS", "DUD"):
                return m
        return None

    def act(self, state, t_sim):
        rng = float(state.get("range", 1e5))
        brg = self._bearing_to_ac1(state)
        wpn = int(state.get("wpn_remaining_t", 0))
        mine = self._own_missiles(state)
        inbound = self._inbound(state)

        # ── DEFEND overrides everything ─────────────────────────────
        if inbound is not None:
            tgo = float(inbound.get("tgo_est", 60.0))
            if tgo < 12.0:
                # Last-ditch: hard beam and dive, maximum energy
                hdg = brg + self._crank_side * (100.0 * DEG2RAD)
                return self._cmd(hdg, self._base_alt - 4000.0, 400.0)
            hdg = brg + self._crank_side * (90.0 * DEG2RAD)
            return self._cmd(hdg, self._base_alt - 1500.0, 360.0)

        # ── SHOOT ───────────────────────────────────────────────────
        fire = 0
        if (wpn > 0 and rng <= self.SHOT_RMAX_FRAC * self.rmax_t
                and (t_sim - self._last_shot) > self.MIN_SHOT_INTERVAL
                and self._phase in ("COMMIT", "REATTACK")):
            if not self._fire_edge:
                fire = 1
                self._fire_edge = True
                self._last_shot = t_sim
                self._phase = "CRANK"
        else:
            self._fire_edge = False

        # ── phase logic ─────────────────────────────────────────────
        if self._phase == "CRANK":
            # Grace period: the missile does not appear in telemetry until the
            # frame AFTER launch, so `any([])` on an empty list would collapse
            # CRANK straight to DRAG on the firing frame. Hold the crank until
            # a missile has actually been seen, or 4 s have passed.
            since_shot = t_sim - self._last_shot
            still_supporting = any(m.get("needs_support", 0) for m in mine)
            if since_shot > 4.0 and not still_supporting:
                self._phase = "DRAG"
            hdg = brg + self._crank_side * self.CRANK_ANGLE
            return self._cmd(hdg, self._base_alt + 500.0, 340.0, fire)

        if self._phase == "DRAG":
            if rng > self.REATTACK_RANGE + 20_000.0 and wpn > 0:
                self._phase = "REATTACK"
                self._crank_side *= -1.0
            hdg = brg + math.pi
            return self._cmd(hdg, self._base_alt - 1000.0, 400.0, fire)

        # COMMIT / REATTACK: hot, climbing for launch energy
        return self._cmd(brg, min(self._base_alt + 1500.0, 12_000.0), 330.0, fire)


class AdaptiveShooterOpponent(ShooterOpponent):
    """
    Same doctrine, randomised parameters per episode. Prevents the agent from
    overfitting to one commit range and one crank angle — which it absolutely
    will do against the fixed ShooterOpponent.
    """

    def reset(self, ic):
        super().reset(ic)
        self.SHOT_RMAX_FRAC = float(self.rng.uniform(0.60, 0.95))
        self.CRANK_ANGLE = float(self.rng.uniform(35.0, 70.0)) * DEG2RAD
        self.REATTACK_RANGE = float(self.rng.uniform(30_000.0, 55_000.0))
        self.MIN_SHOT_INTERVAL = float(self.rng.uniform(8.0, 20.0))
