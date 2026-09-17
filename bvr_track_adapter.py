"""
bvr_track_adapter.py  —  DLL radar track (RAE) -> ENU state for the policy
==========================================================================

REPLACES bvr_tracker.py when the radar DLL already runs its own filter.

Do NOT run an EKF on top of a filtered track. The DLL's outputs are
correlated across time; a second filter treats the same information as fresh
evidence on every update and drives its covariance far below the truth. The
result is a confident, wrong `pos_sigma` — and since the policy's entire
"is my track any good?" signal comes from that number, it would be training
on a lie. Convert, don't re-filter.

WHAT THIS DOES
    1. Unscented conversion of [r, az, el, rdot, azdot, eldot] + 6x6 RAE
       covariance into ENU [x,y,z,vx,vy,vz] + 6x6 Cartesian covariance.
    2. Removes ownship angular rate from the LOS rates, which is REQUIRED —
       see the note in _rae_to_enu(). Skipping it produces velocity estimates
       that are almost entirely ownship-rotation artifact.
    3. Derives a 4-state track machine (NONE/ACQUIRING/TRACK/COAST) from
       validity plus covariance growth, so the observation layout in
       bvr_env.py is unchanged.

WHY UNSCENTED AND NOT A JACOBIAN
    At 80 km, 4 mrad of azimuth sigma is 320 m of cross-range. The
    uncertainty region is a long thin arc; linearising it gives a biased mean
    and an over-confident covariance (the classic converted-measurement
    problem). 13 sigma points costs microseconds and removes the bias.

STATE ORDER IS AN ASSUMPTION — VERIFY IT.
    RAE_ORDER below assumes [r, az, el, rdot, azdot, eldot]. If your DLL
    orders it [r, rdot, az, azdot, el, eldot] (also common), set
    RAE_ORDER = "interleaved". Run verify_rae_convention.py to check against
    truth before trusting anything downstream.
"""

import math
import numpy as np

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# "grouped"     -> [r, az, el, rdot, azdot, eldot]
# "interleaved" -> [r, rdot, az, azdot, el, eldot]
RAE_ORDER = "grouped"

_PERM_INTERLEAVED = [0, 2, 4, 1, 3, 5]


class TrackState:
    NONE = 0
    ACQUIRING = 1
    TRACK = 2
    COAST = 3
    NAMES = {0: "NONE", 1: "ACQUIRING", 2: "TRACK", 3: "COAST"}


def body_to_enu_matrix(psi: float, theta: float, phi: float) -> np.ndarray:
    """Body (x fwd, y right, z down) -> ENU (east, north, up)."""
    cpsi, spsi = math.cos(psi), math.sin(psi)
    cth, sth = math.cos(theta), math.sin(theta)
    cph, sph = math.cos(phi), math.sin(phi)
    R_ned = np.array([
        [cth * cpsi, sph * sth * cpsi - cph * spsi, cph * sth * cpsi + sph * spsi],
        [cth * spsi, sph * sth * spsi + cph * cpsi, cph * sth * spsi - sph * cpsi],
        [-sth,       sph * cth,                     cph * cth],
    ], dtype=np.float64)
    swap = np.array([[0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0]], dtype=np.float64)
    return swap @ R_ned


def _reorder(x6, P6):
    if RAE_ORDER == "grouped":
        return x6, P6
    p = _PERM_INTERLEAVED
    return x6[p], P6[np.ix_(p, p)]


def _rae_to_enu(x_rae, own_pos, own_att, own_vel, own_omega):
    """
    Deterministic map, applied to each sigma point.

    x_rae      : [r, az, el, rdot, azdot, eldot]  (body-referenced)
    own_omega  : (p, q, r) ownship body angular rates, rad/s

    THE ROTATION TERM
        The DLL's azdot/eldot are LOS rates in the ROTATING body frame. When
        the aircraft turns, the LOS sweeps even for a stationary target. The
        inertial derivative of the relative position vector is

            (d/dt p)_inertial = (d/dt p)_body + omega x p

        At 50 km with a 0.1 rad/s yaw rate, |omega x p| = 5000 m/s against a
        real target speed of ~300 m/s. Drop this term and the velocity
        estimate is ~94% rotation artifact, which silently destroys every
        aspect angle and threat-envelope feature in the observation.
    """
    r, az, el, rdot, azdot, eldot = x_rae
    r = max(float(r), 1.0)

    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)

    # LOS unit vector and its angular partials, body frame
    u = np.array([ce * ca, ce * sa, -se])
    du_daz = np.array([-ce * sa, ce * ca, 0.0])
    du_del = np.array([-se * ca, -se * sa, -ce])

    p_body = r * u
    pdot_body = rdot * u + r * (du_daz * azdot + du_del * eldot)

    # Transport theorem: rotating-frame derivative -> inertial derivative
    pdot_inertial = pdot_body + np.cross(np.asarray(own_omega, dtype=np.float64), p_body)

    R = body_to_enu_matrix(*own_att)
    pos_enu = np.asarray(own_pos, dtype=np.float64) + R @ p_body
    vel_enu = np.asarray(own_vel, dtype=np.float64) + R @ pdot_inertial
    return np.concatenate([pos_enu, vel_enu])


class RadarTrackAdapter:
    """
    Feed it the DLL output each time the radar produces a new track.
    Read `estimate()` for the same dict shape bvr_env.py already consumes.
    """

    # Unscented parameters. alpha small keeps sigma points near the mean,
    # which is right when the nonlinearity is mild in range but strong in
    # cross-range.
    ALPHA = 1e-3
    BETA = 2.0
    KAPPA = 0.0
    N = 6

    ACQUIRE_HITS = 5
    COAST_TIMEOUT = 15.0

    # Process-noise PSD for unmodelled target acceleration during coast,
    # m^2/s^3. 400 corresponds to ~2 g of sustained manoeuvre. This drives
    # covariance growth once the radar stops updating, and it is the whole
    # "my track is rotten, re-acquire" signal the policy conditions on.
    # Raise it if the agent trusts stale tracks too much; lower it if it
    # panics and re-acquires when it does not need to.
    COAST_ACCEL_PSD = 400.0

    # Track is demoted once cross-range uncertainty exceeds this. A track
    # with 2 km of position sigma is not a track you can shoot on.
    SIGMA_DEGRADE_M = 2000.0

    def __init__(self):
        lam = self.ALPHA ** 2 * (self.N + self.KAPPA) - self.N
        self._lam = lam
        self._gamma = math.sqrt(self.N + lam)
        wm = np.full(2 * self.N + 1, 1.0 / (2.0 * (self.N + lam)))
        wc = wm.copy()
        wm[0] = lam / (self.N + lam)
        wc[0] = lam / (self.N + lam) + (1.0 - self.ALPHA ** 2 + self.BETA)
        self._wm, self._wc = wm, wc
        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e8
        self.state = TrackState.NONE
        self.n_hits = 0
        self.t_now = 0.0
        self.t_last = -1e9
        self._valid = False
        self._last_rae = None

    def tick(self, dt: float):
        """Advance the clock. Call every sim frame, whether or not the DLL updated."""
        self.t_now += dt
        age = self.t_now - self.t_last
        if self.state == TrackState.TRACK and age > 0.5:
            self.state = TrackState.COAST
            self.n_hits = 0
        elif self.state in (TrackState.COAST, TrackState.ACQUIRING) and age > self.COAST_TIMEOUT:
            t = self.t_now
            self.reset()
            self.t_now = t

        # Dead-reckon through coast so the policy still gets a usable estimate,
        # and inflate covariance so it knows the estimate is decaying.
        #
        # The covariance MUST be propagated through the state transition,
        # P = F P F' + Q, not merely have Q added. The dt coupling in F is
        # what feeds velocity uncertainty into position uncertainty, and that
        # is the dominant growth term — a 9 m/s velocity sigma becomes 54 m of
        # extra position sigma after 6 s of coast. Adding Q alone leaves
        # pos_sigma essentially flat, which would have the filter reporting
        # full confidence in a track it has not seen for ten seconds.
        if self.state == TrackState.COAST and self._valid and dt > 0:
            F = np.eye(6)
            F[0, 3] = F[1, 4] = F[2, 5] = dt

            # Continuous-time white-acceleration discretisation. Using
            # q*dt^2 for the velocity block instead of q*dt is a classic
            # error: it makes total covariance growth proportional to the
            # STEP SIZE rather than to elapsed time, so halving dt halves
            # the reported uncertainty for the same real-world coast
            # duration. The form below depends only on how long the radar
            # has been silent, which is the physically meaningful quantity.
            q = self.COAST_ACCEL_PSD
            dt2, dt3 = dt * dt, dt ** 3
            Q = np.zeros((6, 6))
            for i in range(3):
                Q[i, i] = q * dt3 / 3.0
                Q[i, i + 3] = Q[i + 3, i] = q * dt2 / 2.0
                Q[i + 3, i + 3] = q * dt

            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q

    def update(self, rae_state, rae_cov, own_pos, own_att, own_vel, own_omega,
               valid: bool = True):
        """
        rae_state : (6,) from the DLL
        rae_cov   : (6,6) from the DLL
        own_omega : (p, q, r) body rates, rad/s — REQUIRED, see _rae_to_enu
        """
        if not valid:
            return

        x_rae = np.asarray(rae_state, dtype=np.float64).reshape(6)
        P_rae = np.asarray(rae_cov, dtype=np.float64).reshape(6, 6)
        x_rae, P_rae = _reorder(x_rae, P_rae)

        # Symmetrise and floor — DLL covariances are often slightly asymmetric
        # from accumulated float error, and Cholesky is unforgiving.
        P_rae = 0.5 * (P_rae + P_rae.T)
        P_rae += np.eye(6) * 1e-12
        try:
            S = np.linalg.cholesky((self.N + self._lam) * P_rae)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(P_rae)
            w = np.clip(w, 1e-12, None)
            S = V @ np.diag(np.sqrt(w * (self.N + self._lam)))

        # Sigma points
        pts = np.empty((2 * self.N + 1, 6))
        pts[0] = x_rae
        for i in range(self.N):
            pts[1 + i] = x_rae + S[:, i]
            pts[1 + self.N + i] = x_rae - S[:, i]

        Y = np.array([_rae_to_enu(p, own_pos, own_att, own_vel, own_omega)
                      for p in pts])

        mean = np.einsum("i,ij->j", self._wm, Y)
        d = Y - mean
        cov = np.einsum("i,ij,ik->jk", self._wc, d, d)
        cov = 0.5 * (cov + cov.T)

        self.x, self.P = mean, cov
        self._valid = True
        self._last_rae = x_rae.copy()
        self.t_last = self.t_now
        self.n_hits += 1

        pos_sigma = math.sqrt(max(float(np.trace(cov[:3, :3])), 0.0))
        if pos_sigma > self.SIGMA_DEGRADE_M:
            self.state = TrackState.ACQUIRING
        elif self.n_hits >= self.ACQUIRE_HITS:
            self.state = TrackState.TRACK
        else:
            self.state = TrackState.ACQUIRING

    def estimate(self) -> dict:
        pos_sigma = math.sqrt(max(float(np.trace(self.P[:3, :3])), 0.0))
        vel_sigma = math.sqrt(max(float(np.trace(self.P[3:, 3:])), 0.0))
        return {
            "valid": self._valid and self.state != TrackState.NONE,
            "state": self.state,
            "pos": self.x[:3].copy(),
            "vel": self.x[3:].copy(),
            "pos_sigma": pos_sigma,
            "vel_sigma": vel_sigma,
            "age": max(0.0, self.t_now - self.t_last),
            "speed": float(np.linalg.norm(self.x[3:])),
            "raw_rae": None if self._last_rae is None else self._last_rae.copy(),
        }
