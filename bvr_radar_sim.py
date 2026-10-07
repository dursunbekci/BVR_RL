"""
bvr_radar_sim.py  —  Fast radar surrogate
=========================================

Drop-in stand-in for the radar DLL during training. Runs Python-side inside
the env, costs ~20 us per call instead of 5 ms, and requires no C++ changes:
the C++ side just keeps sending the truth it already sends for WVR.

THE CONTRACT THAT MATTERS
    This emits `rae_state[6]` + `rae_cov[6][6]` in EXACTLY the format the DLL
    produces. bvr_track_adapter.py cannot tell them apart. Switching to the
    real DLL for evaluation or fine-tuning is a config flag, not a rewrite,
    and the policy's input distribution is the same shape either way.

WHAT IT MODELS (and why each one earns its microseconds)

  1. GIMBAL / FOV LIMIT — the important one.
     Crank past the limit and the track drops. This is the entire gradient
     behind "crank as far as you can while keeping the radar on him", which
     is the central BVR skill. Truth-training gives zero signal here and
     actively teaches the wrong answer (under truth, 90 deg of crank is
     strictly better than 50 — free geometry, no cost).

  2. MAX RANGE — commit distance has to mean something.

  3. RANGE-DEPENDENT NOISE.
     Radar SNR falls as 1/R^4, so angular accuracy degrades as R^2. Cheap to
     model and worth keeping even in a minimal surrogate: without it,
     pos_sigma / vel_sigma / trk_age are CONSTANT throughout training, their
     input weights are never meaningfully trained, and the day the real DLL
     is swapped in those channels start moving and inject noise into a
     network that never learned to condition on them.

  4. TEMPORALLY CORRELATED ERROR (Ornstein-Uhlenbeck, not white).
     A filter's output error is smooth, not jittery. Modelling it as white
     noise would make the estimate jump frame to frame in a way no real
     tracker does, and the policy would learn to distrust a signal that is
     actually stable. The OU correlation time also reproduces the property
     that makes cascading a second filter wrong.

WHAT IT DOES NOT MODEL (deliberately, for now)
     The doppler notch. Its absence means the agent will not learn to beam
     to break a lock, and will not learn that a beaming bandit breaks ITS
     lock. That is a real gap and it is worth closing before self-play — but
     it is ONE closure-rate comparison, already stubbed below behind
     `enable_notch`. Flip it on when you want the tactic; everything else
     stays put.

     Also absent: clutter, terrain masking, RCS aspect dependence, jamming,
     search-vs-track scan timing. None of these produce behaviour the policy
     can distinguish from tuned noise.
"""

import math
import numpy as np

DEG2RAD = math.pi / 180.0


class RadarSim:
    """
    Per-env instance. Holds the OU error state, so one instance per aircraft
    per episode. Call reset() at episode start.
    """

    # ── detection envelope ──────────────────────────────────────────
    MAX_RANGE = 90_000.0          # m, detection limit
    FOV_AZ = 60.0 * DEG2RAD       # half-angle, gimbal limit in azimuth
    FOV_EL = 60.0 * DEG2RAD       # half-angle in elevation

    # ── accuracy at the reference range ─────────────────────────────
    # These are FILTERED track sigmas (post-tracker), not raw measurement
    # noise — they are what a DLL reports, and roughly what the real one
    # should produce. Fit them to your DLL once with a short logging run.
    REF_RANGE = 50_000.0
    SIG_AZ_REF = 1.5e-3           # rad  (~75 m cross-range at 50 km)
    SIG_EL_REF = 1.5e-3           # rad
    SIG_RNG_REF = 35.0            # m
    SIG_RDOT_REF = 4.0            # m/s
    SIG_ADOT_REF = 1.2e-4         # rad/s

    # Error correlation time. Filter output wanders on this timescale.
    TAU = 2.0

    # ── low-doppler degradation ─────────────────────────────────────
    # NOT a hard notch. A target near zero radial velocity sits in mainlobe
    # clutter, so track quality degrades — but on an AESA the window is
    # narrow and short-lived (better sidelobe control, longer coherent
    # integration, fast beam-agile re-acquisition), so a clean dropout is the
    # wrong model. Sigma inflation is both more accurate and better for
    # learning: it gives a gradient ("beaming him makes my track worse, which
    # makes my shot worse") instead of a cliff.
    #
    # Note the beam manoeuvre's KINEMATIC value is modelled elsewhere and is
    # unaffected by this: beaming maximises the missile's required turn and
    # drains its energy, which the C++ missile model already produces. The
    # agent will learn to beam against inbound missiles regardless.
    LOWDOP_CLOSURE = 40.0         # m/s — below this, degradation begins
    LOWDOP_MAX_INFLATE = 4.0      # sigma multiplier at zero closure
    enable_lowdop = True

    # Hard dropout, off. Set True only to model an older pulse-doppler set.
    enable_notch = False
    NOTCH_CLOSURE = 25.0

    # Parameters a library radar item (bvr_library.radar_config) may set.
    CFG_KEYS = ("MAX_RANGE", "FOV_AZ", "FOV_EL", "REF_RANGE", "SIG_AZ_REF", "SIG_EL_REF",
                "SIG_RNG_REF", "SIG_RDOT_REF", "SIG_ADOT_REF", "TAU",
                "LOWDOP_CLOSURE", "LOWDOP_MAX_INFLATE")

    def __init__(self, rng=None, randomize=True, cfg=None, target_rcs=None):
        self.rng = rng or np.random.default_rng()
        self._randomize = randomize
        if cfg is not None:
            for k in self.CFG_KEYS:
                setattr(self, k, getattr(cfg, k))
            # Radar range equation: detection range goes as RCS^(1/4). Scaling
            # the accuracy reference range by the same factor keeps the error
            # at a given signal-to-noise ratio unchanged.
            if target_rcs is not None:
                from bvr_library import rcs_scale
                s = rcs_scale(target_rcs, cfg.REF_RCS_M2)
                self.MAX_RANGE = self.MAX_RANGE * s
                self.REF_RANGE = self.REF_RANGE * s
        self.reset()

    def reset(self):
        """
        Call at episode start. Domain-randomises the radar parameters, which
        matters more than it looks: it stops the policy from overfitting to
        one exact gimbal limit and one exact detection range, and it makes
        the eventual transfer to the real DLL a small distribution shift
        rather than a cliff.
        """
        self._err = np.zeros(6)
        if self._randomize:
            r = self.rng
            self.max_range = self.MAX_RANGE * float(r.uniform(0.85, 1.15))
            self.fov_az = self.FOV_AZ * float(r.uniform(0.85, 1.15))
            self.fov_el = self.FOV_EL * float(r.uniform(0.85, 1.15))
            self._sig_scale = float(r.uniform(0.7, 1.5))
        else:
            self.max_range = self.MAX_RANGE
            self.fov_az = self.FOV_AZ
            self.fov_el = self.FOV_EL
            self._sig_scale = 1.0

    # ── main entry ──────────────────────────────────────────────────
    def sense(self, dt, own_pos, own_att, own_vel, own_omega,
              tgt_pos, tgt_vel):
        """
        All arguments in ENU / rad / SI. own_omega = (p, q, r) body rates.

        Returns (rae_state[6], rae_cov[6,6], valid) — the DLL's interface.
        rae_state is [r, az, el, rdot, azdot, eldot], body-referenced,
        rates NOT compensated for ownship rotation (matching the convention
        bvr_track_adapter.py assumes; see RADAR_OMEGA_COMPENSATED).
        """
        from bvr_track_adapter import body_to_enu_matrix

        R = body_to_enu_matrix(*own_att)
        p_body = R.T @ (np.asarray(tgt_pos, float) - np.asarray(own_pos, float))
        v_rel = R.T @ (np.asarray(tgt_vel, float) - np.asarray(own_vel, float))
        # inertial -> rotating-frame derivative
        pdot_body = v_rel - np.cross(np.asarray(own_omega, float), p_body)

        x, y, z = p_body
        rng = float(np.linalg.norm(p_body))
        if rng < 1.0:
            return None, None, False

        az = math.atan2(y, x)
        el = math.atan2(-z, math.hypot(x, y))

        # ── detection gate ──────────────────────────────────────────
        if rng > self.max_range:
            return None, None, False
        if abs(az) > self.fov_az or abs(el) > self.fov_el:
            return None, None, False

        u = p_body / rng
        rdot = float(np.dot(u, pdot_body))
        rxy = max(math.hypot(x, y), 1e-6)
        azdot = (x * pdot_body[1] - y * pdot_body[0]) / (rxy ** 2)
        eldot = (-pdot_body[2] * rxy ** 2
                 - (-z) * (x * pdot_body[0] + y * pdot_body[1])) / (rng ** 2 * rxy)

        if self.enable_notch and abs(rdot) < self.NOTCH_CLOSURE:
            return None, None, False

        # ── range-dependent sigmas ──────────────────────────────────
        # SNR ~ 1/R^4  ->  angular sigma ~ R^2. Floored so it stays sane
        # at very short range, capped so it stays finite at the limit.
        k = float(np.clip((rng / self.REF_RANGE) ** 2, 0.15, 6.0)) * self._sig_scale

        # Low-doppler penalty: smooth ramp, no cliff. Full accuracy above
        # LOWDOP_CLOSURE, degrading to LOWDOP_MAX_INFLATE at zero closure.
        if self.enable_lowdop:
            f = min(abs(rdot) / self.LOWDOP_CLOSURE, 1.0)
            k *= 1.0 + (self.LOWDOP_MAX_INFLATE - 1.0) * (1.0 - f) ** 2

        sig = np.array([
            self.SIG_RNG_REF * math.sqrt(k),   # range is less SNR-sensitive
            self.SIG_AZ_REF * k,
            self.SIG_EL_REF * k,
            self.SIG_RDOT_REF * math.sqrt(k),
            self.SIG_ADOT_REF * k,
            self.SIG_ADOT_REF * k,
        ])

        # ── OU-correlated error ─────────────────────────────────────
        a = math.exp(-max(dt, 1e-4) / self.TAU)
        self._err = a * self._err + math.sqrt(max(1.0 - a * a, 1e-9)) * self.rng.standard_normal(6)

        truth = np.array([rng, az, el, rdot, azdot, eldot])
        rae = truth + self._err * sig
        rae[0] = max(rae[0], 100.0)

        cov = np.diag(sig ** 2)
        return rae, cov, True
