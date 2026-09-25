"""
f16_sim.py  —  Simplified F-16C/D flight model
===============================================

Point-mass 3-DOF translational + attitude for observation. All arithmetic
runs in pure Python/NumPy — no I/O, no UDP, no C++ required.

State variables (all SI, flat-earth ENU):
    x, y       ENU position, m
    z          altitude, m
    V          true airspeed, m/s
    gamma      flight path angle, rad  (+= climb)
    chi        heading, rad  (0=North, clockwise +)
    phi        bank angle, rad
    throttle   internal engine state [0,1]
    fuel_kg    current fuel mass, kg

Derived per step (not integrated, computed analytically for telemetry):
    alpha, beta, nz, mach, theta, p, q, r

Autopilot outer loop (BVR command interface):
    hdgCmd    (rad)   absolute heading command
    altTarget (m)     altitude command
    V         (m/s)   speed command
    altFPA    (rad)   FPA limit  (optional, default ±35°)
    climbFPA  (rad)   tighter limit on climbs only (optional; dives keep altFPA)

Calibration against published F-16C Block 50 data:
    9000 m, Mach 0.9, mil thrust:
        SEP   ≈ 52 m/s        (F-16 EM chart: ~50–80 m/s, MIL power)
        nz_max ≈ 5.7g         (aero limited at this alt/speed)
        chi_dot_max ≈ 10 °/s  (sustained, at 5g)
    SL, Mach 0.9:
        nz_avail ≈ 19g → capped to 9g structural
"""

import math
import numpy as np

G        = 9.80665
R_EARTH  = 6_371_000.0
DEG2RAD  = math.pi / 180.0
RAD2DEG  = 180.0 / math.pi


# ─────────────────────────────────────────────────────────────────────
# ISA atmosphere
# ─────────────────────────────────────────────────────────────────────
def isa(alt_m: float):
    """(rho kg/m³, T K, p Pa, a_sound m/s)"""
    h = float(np.clip(alt_m, 0.0, 20_000.0))
    if h <= 11_000.0:
        T = 288.15 - 0.0065 * h
        p = 101_325.0 * (T / 288.15) ** 5.2561
    else:
        T = 216.65
        p = 22_632.1 * math.exp(-0.0001577 * (h - 11_000.0))
    rho = p / (287.05 * T)
    a   = math.sqrt(1.4 * 287.05 * T)
    return rho, T, p, a


# ─────────────────────────────────────────────────────────────────────
# F-16C Block 50 parameters
# ─────────────────────────────────────────────────────────────────────
class F16Cfg:
    # mass
    MASS_EMPTY = 8_570.0     # kg airframe
    FUEL_FULL  = 3_175.0     # kg internal fuel
    MASS_WPNS  =   500.0     # kg weapons (≈4 × AIM-120)

    # aerodynamics (clean, subsonic)
    S_REF      = 27.87       # m² wing reference area
    CD0        = 0.013       # zero-lift parasite drag
    K_IND      = 0.133       # induced drag: 1/(π·e·AR), e=0.80, AR=3.0
    CL_ALPHA   = 3.5         # /rad linear lift-curve slope
    CL_MAX     = 1.45        # maximum usable lift coefficient
    CL_MIN     = -0.50

    # structural limits
    NZ_POS_MAX = 9.0
    NZ_NEG_MAX = -3.0

    # propulsion (F110-GE-129)
    T_MIL_SL   = 65_000.0   # N military power, sea level
    T_AB_SL    = 129_000.0  # N full afterburner, sea level
    ALT_LAPSE  = 0.70        # thrust sigma exponent

    # fuel flow
    FF_IDLE    = 0.06        # kg/s at idle
    FF_MIL     = 0.42        # kg/s at military power
    FF_AB      = 1.65        # kg/s at max afterburner

    # flight envelope
    V_STALL    = 70.0        # m/s — model not valid below this
    MACH_MAX   = 1.80
    ALT_CEIL   = 15_240.0   # m (50 000 ft)

    # ── autopilot gains ──────────────────────────────────────────────
    # Heading → bank
    K_HDG_PHI  = 2.0         # rad of bank per rad of heading error
    PHI_MAX    = 80.0 * DEG2RAD
    K_PHI_ROLL = 2.5         # roll rate per rad of bank error  (1/s)
    ROLL_MAX   = 4.0         # rad/s (~230 °/s physical limit)

    # Altitude → FPA
    # Was 0.12 with K_GAM_NZ 3.0: the FPA command sat at its limit until ~4 m
    # from the target and the slow FPA loop then overshot a 1 km step by
    # 254 m, taking 48 s to settle. Now 20 m / 23 s at 340 m/s, 3.1 g peak.
    K_ALT_GAM  = 0.0003      # rad of FPA per m of altitude error
    GAMMA_MAX  = 40.0 * DEG2RAD

    # FPA → load factor
    K_GAM_NZ   = 7.0         # nz correction per rad of FPA error

    # Fraction of available load factor a turn may spend on holding the
    # aircraft level; the rest stays in reserve for the altitude loop.
    BANK_NZ_MARGIN = 0.85

    # Speed → throttle
    # Was proportional only (0.012, no integral): steady errors up to 34 m/s,
    # e.g. 400 -> 366, 340 -> 310, 220 -> 247. Now PI: within 0.4 m/s.
    K_V_THROT  = 0.05        # throttle change per m/s of speed error
    K_V_INT    = 0.004       # throttle change per (m/s · s) of integrated error
    THROT_TC   = 2.5         # s engine lag time constant

    # Mach-dependent curves (defined below the class)
    @staticmethod
    def thrust_mach_factor(mach): return _thrust_mach_factor(mach)

    @staticmethod
    def cd_rise(mach): return _cd_rise(mach)


def _thrust_mach_factor(mach: float) -> float:
    """Dimensionless thrust scale vs Mach relative to M=0."""
    m = float(np.clip(mach, 0.0, 2.0))
    if m < 0.50: return 1.00 + 0.05 * m
    if m < 0.90: return 1.025 - 0.15 * (m - 0.50) / 0.40
    if m < 1.20: return 0.965 - 0.05 * (m - 0.90) / 0.30
    return 0.950 + 0.08 * (m - 1.20)


def _cd_rise(mach: float) -> float:
    """Additional drag in the transonic region, and the wave drag that remains above it."""
    if mach < 0.85: return 0.0
    if mach < 1.10: return 0.034 * math.sin(math.pi * (mach - 0.85) / 0.25)
    if mach < 1.40: return 0.034 * math.exp(-1.0 * (mach - 1.10))
    # Supersonic wave drag stays. This used to return 0 (peak 0.040, decay
    # 2.0): a 40% drop in total drag at Mach 1.4 that let the aircraft reach
    # its Mach 1.8 cap even at sea level. Now Mach 1.17 at 500 m, 1.37 at
    # 3 km, 1.66 at 9 km (published: about 1.2 low down, about 2 up high).
    return 0.034 * math.exp(-1.0 * (1.40 - 1.10))


# ─────────────────────────────────────────────────────────────────────
class F16Aircraft:
    """
    One F-16 instance. Call reset_state() at episode start, step() each frame.
    All state is public so sim_world can read it directly without method calls.
    """

    def __init__(self, rng: np.random.Generator = None, cfg=None):
        self._rng = rng or np.random.default_rng()
        # Airframe parameters: F16Cfg by default, or an airframe from the
        # parameter library (bvr_library.airframe_config), which has the same
        # attribute names plus the two Mach-dependent curves as callables.
        self.cfg = cfg if cfg is not None else F16Cfg
        # placeholders — overwritten by reset_state
        self.x = self.y = self.z = 0.0
        self.V = 280.0; self.gamma = 0.0; self.chi = 0.0; self.phi = 0.0
        self.throttle = 0.5; self.fuel_kg = self.cfg.FUEL_FULL * 0.5
        self._v_int = 0.0            # speed-hold integrator
        # derived
        self.alpha = 0.0; self.beta = 0.0; self.nz = 1.0; self.mach = 0.9
        self.theta = 0.0; self.p = 0.0; self.q = 0.0; self.r = 0.0
        self._rho = 0.466; self._a_sound = 295.0
        self.gamma_dot = 0.0; self.chi_dot = 0.0; self.phi_dot = 0.0

    # ── setup ────────────────────────────────────────────────────────
    def reset_state(self, x: float, y: float, z: float,
                    chi: float, V: float, fuel_frac: float = 0.60) -> None:
        self.x, self.y      = float(x), float(y)
        self.z              = float(np.clip(z, 100.0, self.cfg.ALT_CEIL))
        self.V              = float(np.clip(V, self.cfg.V_STALL + 20, 550.0))
        self.gamma          = 0.0
        self.chi            = float(chi) % (2.0 * math.pi)
        self.phi            = 0.0
        self.throttle       = 0.55
        self._v_int         = 0.0
        self.fuel_kg        = float(fuel_frac) * self.cfg.FUEL_FULL
        self._refresh_atmos()
        self.mach           = self.V / max(self._a_sound, 1.0)
        self.alpha          = 0.0; self.beta = 0.0; self.nz = 1.0
        self.theta          = 0.0; self.p = 0.0; self.q = 0.0; self.r = 0.0
        self.gamma_dot      = 0.0; self.chi_dot = 0.0; self.phi_dot = 0.0

    # ── main integration step ────────────────────────────────────────
    def step(self, dt: float, cmd: dict) -> None:
        """
        Advance one time step. cmd keys used:
            hdgCmd    (rad)  absolute heading
            altTarget (m)    altitude target
            V         (m/s)  speed target
            altFPA    (rad)  FPA limit (optional)
        """
        self._refresh_atmos()

        hdg_cmd = float(cmd.get("hdgCmd", self.chi))
        alt_cmd = float(cmd.get("altTarget", self.z))
        V_cmd   = float(cmd.get("V", self.V))
        fpa_lim = abs(float(cmd.get("altFPA", self.cfg.GAMMA_MAX)))
        fpa_lim = float(np.clip(fpa_lim, 0.05, self.cfg.GAMMA_MAX))
        climb_lim = min(fpa_lim, abs(float(cmd.get("climbFPA", fpa_lim))))

        mach = self.V / max(self._a_sound, 1.0)
        mach = float(np.clip(mach, 0.15, self.cfg.MACH_MAX))
        q_dyn = 0.5 * self._rho * self.V ** 2
        W = self.mass * G

        # ── 1. speed autopilot → throttle ────────────────────────────
        # Proportional plus integral. Proportional alone settled away from the
        # command (340 -> 310 m/s at 3 km, 400 -> 366, 220 -> 247). The
        # integral only runs while the throttle is not pinned at a limit, or
        # when the error would pull it off that limit, so it does not wind up
        # while the aircraft is energy-limited (a hard turn at full power).
        V_err = V_cmd - self.V
        k_i = getattr(self.cfg, "K_V_INT", 0.0)
        raw = 0.55 + self.cfg.K_V_THROT * V_err + k_i * self._v_int
        throt_cmd = float(np.clip(raw, 0.0, 1.0))
        if k_i > 0.0 and ((0.0 < raw < 1.0) or (raw >= 1.0 and V_err < 0.0)
                          or (raw <= 0.0 and V_err > 0.0)):
            self._v_int += V_err * dt
        lag = float(np.clip(dt / self.cfg.THROT_TC, 0.0, 1.0))
        self.throttle += lag * (throt_cmd - self.throttle)
        self.throttle = float(np.clip(self.throttle, 0.0, 1.0))

        # ── 2. thrust ────────────────────────────────────────────────
        sigma = self._rho / 1.225
        tf    = self.cfg.thrust_mach_factor(mach)
        if self.throttle <= 0.90:
            T = (self.throttle / 0.90) * self.cfg.T_MIL_SL * (sigma ** self.cfg.ALT_LAPSE) * tf
        else:
            fab   = (self.throttle - 0.90) / 0.10
            T_mil = self.cfg.T_MIL_SL * (sigma ** self.cfg.ALT_LAPSE) * tf
            T_ab  = self.cfg.T_AB_SL  * (sigma ** self.cfg.ALT_LAPSE) * tf
            T     = T_mil + fab * (T_ab - T_mil)
        T = max(0.0, T)

        # fuel flow
        if self.throttle <= 0.90:
            ff = self.cfg.FF_IDLE + (self.throttle / 0.90) * (self.cfg.FF_MIL - self.cfg.FF_IDLE)
        else:
            fab = (self.throttle - 0.90) / 0.10
            ff  = self.cfg.FF_MIL + fab * (self.cfg.FF_AB - self.cfg.FF_MIL)
        self.fuel_kg = max(0.0, self.fuel_kg - ff * dt)

        # ── 3. available load factor (aero limit and structural limit) ──
        if q_dyn > 100.0:
            nz_aero = q_dyn * self.cfg.S_REF * self.cfg.CL_MAX / max(W, 1.0)
        else:
            nz_aero = self.cfg.NZ_POS_MAX
        nz_avail = min(self.cfg.NZ_POS_MAX, nz_aero)

        # ── 4. heading autopilot → bank → roll rate ──────────────────
        # Holding level at bank φ takes nz = cos(γ)/cos φ. Banking past what
        # nz_avail can support means losing altitude no matter how hard the
        # aircraft pulls, so bank is capped at that point, less a margin.
        phi_lim  = min(self.cfg.PHI_MAX,
                       math.acos(min(1.0, math.cos(self.gamma)
                                     / (self.cfg.BANK_NZ_MARGIN * nz_avail))))
        hdg_err  = _wrap_pi(hdg_cmd - self.chi)
        phi_cmd  = float(np.clip(self.cfg.K_HDG_PHI * hdg_err, -phi_lim, phi_lim))
        phi_err  = phi_cmd - self.phi
        phi_dot  = float(np.clip(self.cfg.K_PHI_ROLL * phi_err,
                                  -self.cfg.ROLL_MAX, self.cfg.ROLL_MAX))
        self.phi = float(np.clip(self.phi + phi_dot * dt,
                                  -self.cfg.PHI_MAX, self.cfg.PHI_MAX))

        # ── 5. altitude autopilot → nz command ──────────────────────
        alt_err   = alt_cmd - self.z
        gamma_cmd = float(np.clip(self.cfg.K_ALT_GAM * alt_err, -fpa_lim, climb_lim))
        gamma_err = gamma_cmd - self.gamma
        # Only nz·cos φ acts vertically. Without the division a steep turn
        # spirals into the ground while altitude hold is commanded.
        nz_cmd    = ((math.cos(self.gamma) + self.cfg.K_GAM_NZ * gamma_err)
                     / max(math.cos(self.phi), 0.1))
        nz_cmd    = float(np.clip(nz_cmd, self.cfg.NZ_NEG_MAX, nz_avail))

        # ── 6. aerodynamics ──────────────────────────────────────────
        CL   = float(np.clip(nz_cmd * W / max(q_dyn * self.cfg.S_REF, 1.0),
                              self.cfg.CL_MIN, self.cfg.CL_MAX))
        CD   = self.cfg.CD0 + self.cfg.K_IND * CL ** 2 + self.cfg.cd_rise(mach)
        D    = q_dyn * self.cfg.S_REF * CD
        alpha = float(np.clip(CL / self.cfg.CL_ALPHA, -0.25, 0.45))

        # ── 7. equations of motion ───────────────────────────────────
        # Axial: speed change from net thrust minus drag minus gravity-along-path
        V_dot = (T * math.cos(alpha) - D) / self.mass - G * math.sin(self.gamma)

        # Normal: FPA rate from normal force
        gamma_dot = (nz_cmd * math.cos(self.phi) - math.cos(self.gamma)) * G \
                    / max(self.V, 1.0)

        # Lateral: heading rate from bank (bank-to-turn)
        chi_dot = G * nz_cmd * math.sin(self.phi) \
                  / max(self.V * math.cos(self.gamma), 1.0)

        # ── 8. integrate ─────────────────────────────────────────────
        self.V     = float(np.clip(self.V + V_dot * dt,
                                    self.cfg.V_STALL, self.cfg.MACH_MAX * self._a_sound))
        self.gamma = float(np.clip(self.gamma + gamma_dot * dt, -1.20, 1.20))
        self.chi   = (self.chi + chi_dot * dt) % (2.0 * math.pi)

        cg = math.cos(self.gamma); sg = math.sin(self.gamma)
        sc = math.sin(self.chi);   cc = math.cos(self.chi)
        self.x += self.V * cg * sc * dt
        self.y += self.V * cg * cc * dt
        self.z  = float(np.clip(self.z + self.V * sg * dt, 10.0, self.cfg.ALT_CEIL))

        # ── 9. derived observation quantities ────────────────────────
        self.mach  = mach
        self.alpha = alpha
        self.beta  = 0.0
        self.nz    = float(nz_cmd)
        self.theta = self.gamma + alpha   # pitch ≈ FPA + AoA

        # Body rates via Euler-rate transformation
        # p = φ̇  − ψ̇ · sin θ
        # q = θ̇ · cos φ  + ψ̇ · cos θ · sin φ
        # r = −θ̇ · sin φ + ψ̇ · cos θ · cos φ
        theta_dot = gamma_dot   # approximate (ignores alpha_dot)
        psi_dot   = chi_dot
        sp, cp    = math.sin(self.phi), math.cos(self.phi)
        st, ct    = math.sin(self.theta), math.cos(self.theta)
        self.p  = phi_dot   - psi_dot * st
        self.q  = theta_dot * cp + psi_dot * ct * sp
        self.r  = -theta_dot * sp + psi_dot * ct * cp

        # cache for sim_world
        self.gamma_dot = gamma_dot
        self.chi_dot   = chi_dot
        self.phi_dot   = phi_dot

    # ── helpers ──────────────────────────────────────────────────────
    def _refresh_atmos(self):
        rho, _, _, a = isa(self.z)
        self._rho     = rho
        self._a_sound = a

    @property
    def mass(self) -> float:
        return self.cfg.MASS_EMPTY + max(self.fuel_kg, 0.0) + self.cfg.MASS_WPNS

    @property
    def fuel_frac(self) -> float:
        return float(np.clip(self.fuel_kg / max(self.cfg.FUEL_FULL, 1.0), 0.0, 1.0))

    @property
    def vel_enu(self) -> np.ndarray:
        cg = math.cos(self.gamma); sg = math.sin(self.gamma)
        return np.array([self.V * cg * math.sin(self.chi),
                          self.V * cg * math.cos(self.chi),
                          self.V * sg])


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
