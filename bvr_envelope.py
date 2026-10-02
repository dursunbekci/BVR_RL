"""
bvr_envelope.py  —  AIM-120 launch envelope model
=================================================

Supplies R_max (kinematic launch range) and R_nez (no-escape zone) as a
function of launch conditions. These two numbers, normalised against current
range, are the single most informative features in the whole BVR observation:
almost every real decision reduces to

    r / R_max_mine      am I able to shoot?
    r / R_nez_mine      is the shot likely to connect against a reacting target?
    r / R_max_his       can HE shoot me?
    r / R_nez_his       am I in his heart of the envelope?

IMPORTANT — CALIBRATE THIS.
The analytic default below is a physically-shaped guess, not your missile.
It exists so the RL side can be developed before the C++ missile is finished.
Once `f16simmodel.exe` flies AIM-120s, run `sweep_envelope.py` to build a
lookup table (library/envelopes/<missile>-<fingerprint>.npz); BvrEnv loads
the one matching each side's missile.
An uncalibrated envelope will train a policy that shoots at the wrong range,
and that error is invisible in the reward curve — it just caps your Pk.

Envelope shape drivers, in order of importance:
  1. Target aspect   — head-on doubles reach vs a tail chase (closure adds
                       range, and the target runs INTO the missile)
  2. Launch altitude — thinner air, less drag; roughly linear in this band
  3. Launch Mach     — the missile inherits the launch aircraft's energy
  4. Target speed    — a fast retreating target eats the missile's margin
"""

import math
import numpy as np

DEG2RAD = math.pi / 180.0


class Aim120Envelope:

    # Reference point the analytic model is anchored to:
    # Mach 0.9, 10 000 m, head-on, target Mach 0.9.
    R_MAX_REF = 75_000.0
    R_NEZ_REF = 30_000.0

    ALT_REF = 10_000.0
    MACH_REF = 0.9

    def __init__(self, table_path: str = None):
        self._table = None
        if table_path:
            self.load_table(table_path)

    # ── calibrated path ─────────────────────────────────────────────
    @classmethod
    def from_table(cls, path: str) -> "Aim120Envelope":
        return cls(table_path=path)

    def load_table(self, path: str):
        """
        Expects an .npz produced by sweep_envelope.py with:
            mach_grid   (Nm,)      launch aircraft Mach
            alt_grid    (Na,)      launch altitude, m
            aspect_grid (Np,)      target aspect, deg (0 = head-on)
            r_max       (Nm,Na,Np) m
            r_nez       (Nm,Na,Np) m
        """
        d = np.load(path)
        self._table = {
            "mach": d["mach_grid"], "alt": d["alt_grid"], "aspect": d["aspect_grid"],
            "r_max": d["r_max"], "r_nez": d["r_nez"],
        }

    # ── main entry ──────────────────────────────────────────────────
    def compute(self, mach: float, alt: float, aspect_deg: float,
                target_mach: float = 0.9) -> tuple:
        """
        mach        : launch aircraft Mach
        alt         : launch altitude, m
        aspect_deg  : TARGET aspect — 0 = target nose-on to shooter (head-on),
                      180 = target tail-on (running away)
        target_mach : target Mach

        Returns (r_max, r_nez) in metres.
        """
        if self._table is not None:
            return self._interp_table(mach, alt, aspect_deg, target_mach)
        return self._analytic(mach, alt, aspect_deg, target_mach)

    def _analytic(self, mach, alt, aspect_deg, target_mach):
        mach = float(np.clip(mach, 0.4, 2.0))
        alt = float(np.clip(alt, 0.0, 20_000.0))
        aspect = float(np.clip(abs(aspect_deg), 0.0, 180.0))
        tmach = float(np.clip(target_mach, 0.3, 2.0))

        # 1. Aspect. Head-on (0°) is the reference; tail-on collapses to ~30%.
        ca = math.cos(aspect * DEG2RAD)
        f_aspect = 0.30 + 0.70 * (0.5 * (1.0 + ca)) ** 0.6

        # 2. Altitude. ~+4.5% reach per 1000 m above the reference.
        f_alt = 1.0 + 0.045 * (alt - self.ALT_REF) / 1000.0
        f_alt = float(np.clip(f_alt, 0.35, 2.0))

        # 3. Launch Mach. Missile inherits launch energy; strong effect.
        f_mach = (mach / self.MACH_REF) ** 0.85

        # 4. Target speed. A fast runner steals range, a fast closer adds it.
        #    Scaled by how much of its velocity lies along the LOS.
        f_tgt = 1.0 - 0.35 * (tmach / self.MACH_REF - 1.0) * (-ca)
        f_tgt = float(np.clip(f_tgt, 0.5, 1.6))

        r_max = self.R_MAX_REF * f_aspect * f_alt * f_mach * f_tgt

        # NEZ shrinks faster than R_max off the nose: at high aspect the target
        # only has to keep running, so the "no escape" band nearly vanishes.
        f_nez_aspect = 0.15 + 0.85 * (0.5 * (1.0 + ca)) ** 1.4
        r_nez = self.R_NEZ_REF * f_nez_aspect * f_alt * f_mach * f_tgt

        r_max = float(np.clip(r_max, 3_000.0, 160_000.0))
        r_nez = float(np.clip(r_nez, 1_500.0, r_max * 0.85))
        return r_max, r_nez

    def _interp_table(self, mach, alt, aspect_deg, target_mach):
        t = self._table
        aspect = abs(float(aspect_deg))

        def lin(grid, v):
            v = float(np.clip(v, grid[0], grid[-1]))
            j = int(np.searchsorted(grid, v)) - 1
            j = int(np.clip(j, 0, len(grid) - 2))
            w = (v - grid[j]) / max(grid[j + 1] - grid[j], 1e-9)
            return j, w

        im, wm = lin(t["mach"], mach)
        ia, wa = lin(t["alt"], alt)
        ip, wp = lin(t["aspect"], aspect)

        def tri(cube):
            c = 0.0
            for dm, fm in ((0, 1 - wm), (1, wm)):
                for da, fa in ((0, 1 - wa), (1, wa)):
                    for dp, fp in ((0, 1 - wp), (1, wp)):
                        c += fm * fa * fp * cube[im + dm, ia + da, ip + dp]
            return float(c)

        r_max = tri(t["r_max"])
        r_nez = tri(t["r_nez"])

        # Table is built at a nominal target speed; apply the same correction.
        ca = math.cos(aspect * DEG2RAD)
        f_tgt = float(np.clip(1.0 - 0.35 * (target_mach / self.MACH_REF - 1.0) * (-ca), 0.5, 1.6))
        return r_max * f_tgt, min(r_nez * f_tgt, r_max * f_tgt * 0.85)


def aspect_deg_from_vectors(shooter_pos, target_pos, target_vel) -> float:
    """
    Target aspect: angle between the target's velocity vector and the
    target→shooter line. 0° = target pointing at us. 180° = target running.
    """
    d = np.asarray(shooter_pos, dtype=np.float64) - np.asarray(target_pos, dtype=np.float64)
    nd = float(np.linalg.norm(d))
    v = np.asarray(target_vel, dtype=np.float64)
    nv = float(np.linalg.norm(v))
    if nd < 1.0 or nv < 1.0:
        return 90.0
    c = float(np.clip(np.dot(d, v) / (nd * nv), -1.0, 1.0))
    return math.degrees(math.acos(c))
