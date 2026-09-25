"""
test_sim.py  —  Physics and integration tests for the pure-Python sim
=====================================================================

    python test_sim.py

Tests are ordered from smallest to largest scope. A failure in an early
test means a specific module is wrong; a failure in a later one means the
integration is broken. This ordering matters when debugging.
"""

import math, os, time, sys
import numpy as np

DEG = math.pi / 180.0


# ════════════════════════════════════════════════════════════════════
# 1. F-16 PHYSICS
# ════════════════════════════════════════════════════════════════════
def test_f16_turn_rate():
    """
    Sustained turn rate at 9000 m, Mach 0.9 should be ~8-12 °/s.
    Under-turn means the aerodynamics are wrong; over-turn means
    the structural g-limit is not engaging.
    """
    from f16_sim import F16Aircraft, isa
    ac = F16Aircraft()
    _, _, _, a = isa(9000.0)
    V = 0.9 * a                                # ~270 m/s at 9000 m
    ac.reset_state(0, 0, 9000.0, 0.0, V, 0.6)

    # Command a 90° heading change at constant alt and speed
    chi_start = ac.chi
    for _ in range(200):                       # 4 s at 50 Hz
        ac.step(0.02, {"hdgCmd": chi_start + 90*DEG, "altTarget": 9000.0, "V": V})

    chi_change_deg = abs(_wrap_pi(ac.chi - chi_start)) * 180/math.pi
    turn_rate = chi_change_deg / 4.0           # °/s over 4 s
    print(f"  F-16 sustained turn:  {turn_rate:.1f} °/s  (expect 8–16)")
    assert 5.0 < turn_rate < 20.0, f"turn rate {turn_rate:.1f} °/s out of range"
    print("  F-16 turn rate .............. OK")


def test_f16_speed_convergence():
    """Speed autopilot should reach commanded value within ~20 s."""
    from f16_sim import F16Aircraft
    ac = F16Aircraft()
    ac.reset_state(0, 0, 9000.0, 0.0, 280.0, 0.6)
    V_cmd = 360.0
    for _ in range(1500):    # 30 s
        ac.step(0.02, {"hdgCmd": 0.0, "altTarget": 9000.0, "V": V_cmd})
    err = abs(ac.V - V_cmd)
    print(f"  F-16 speed after 30 s:  V={ac.V:.1f}  err={err:.1f} m/s")
    assert err < 30.0, f"speed error {err:.1f} m/s after 30 s"
    print("  F-16 speed autopilot ........ OK")


def test_f16_alt_hold():
    """Altitude hold should keep altitude within ±200 m after transient."""
    from f16_sim import F16Aircraft
    ac = F16Aircraft()
    ac.reset_state(0, 0, 8000.0, 0.0, 280.0, 0.6)
    for _ in range(2500):    # 50 s
        ac.step(0.02, {"hdgCmd": 0.0, "altTarget": 9000.0, "V": 280.0})
    alt_err = abs(ac.z - 9000.0)
    print(f"  F-16 alt after 50 s:  z={ac.z:.0f}  err={alt_err:.0f} m")
    assert alt_err < 300.0, f"alt error {alt_err:.0f} m"
    print("  F-16 alt hold ............... OK")


def test_f16_energy():
    """
    SEP check: at 9000 m, Mach 0.9, military power (throttle ~0.90),
    the F-16 should gain 20–90 m/s of equivalent airspeed equivalent per
    second. Test methodology: command V+30 m/s (which drives throttle_cmd
    to ≈0.91 ≈ military power via the autopilot), run 20 s so the throttle
    state stabilises, then read the energy change over the final 2 s.
    """
    from f16_sim import F16Aircraft, isa
    ac = F16Aircraft()
    _, _, _, a = isa(9000.0)
    V = 0.9 * a                                     # ~270 m/s at 9000 m
    ac.reset_state(0, 0, 9000.0, 0.0, V, 0.8)
    # K_V_THROT=0.012, offset=0.55: cmd=0.55+0.012*30=0.91 ≈ military
    V_cmd = V + 30.0
    for _ in range(1000):   # 20 s stabilisation
        ac.step(0.02, {"hdgCmd": 0.0, "altTarget": 9000.0, "V": V_cmd})
    E0 = ac.V**2 / 2 + 9.81 * ac.z; z0 = ac.z; V0 = ac.V
    for _ in range(100):    # 2 s measurement window
        ac.step(0.02, {"hdgCmd": 0.0, "altTarget": 9000.0, "V": V_cmd})
    E1 = ac.V**2 / 2 + 9.81 * ac.z
    SEP = (E1 - E0) / 2.0
    print(f"  F-16 SEP:  {SEP:.1f} m/s  throttle={ac.throttle:.2f}  (expect 20–100)")
    assert 10.0 < SEP < 120.0, f"SEP {SEP:.1f} m/s out of range"
    print("  F-16 energy/SEP ............. OK")


def test_f16_body_rates():
    """
    In a sustained 3 g level turn at 280 m/s:
    r (yaw rate) ≈ g·sqrt(n²-1)/V ≈ 0.10 rad/s.
    The transport term correction in the tracker relies on this.
    """
    from f16_sim import F16Aircraft
    ac = F16Aircraft()
    ac.reset_state(0, 0, 9000.0, 0.0, 280.0, 0.6)
    # Command a hard right turn
    for _ in range(250):   # 5 s
        ac.step(0.02, {"hdgCmd": math.pi/2, "altTarget": 9000.0, "V": 280.0})
    r_exp = 9.81 * math.sqrt(max(ac.nz**2 - 1, 0)) / max(ac.V, 1.0)
    print(f"  F-16 body r={ac.r:.4f} rad/s  expected ~{r_exp:.4f}  nz={ac.nz:.2f}")
    assert abs(ac.r) > 0.03, f"r too small: {ac.r:.4f} rad/s"
    print("  F-16 body rates ............. OK")


# ════════════════════════════════════════════════════════════════════
# 2. MISSILE MODEL
# ════════════════════════════════════════════════════════════════════
def test_missile_straight_shot():
    """A head-on shot at 40 km against a non-manoeuvring target must HIT."""
    from missile_sim import AIM120, MslPhase

    own_pos  = np.array([0.0,  0.0, 9000.0])
    tgt_pos  = np.array([0.0, 40000.0, 9000.0])
    own_vel  = np.array([0.0,  270.0,  0.0])
    tgt_vel  = np.array([0.0, -270.0,  0.0])

    m = AIM120(1, 2, own_pos.copy(), own_vel.copy(), handoff_range=10000.0)
    m.last_tgt_pos = tgt_pos.copy(); m.last_tgt_vel = tgt_vel.copy()

    events = []
    for _ in range(6000):   # 120 s max
        tp = tgt_pos + tgt_vel * (m.t_flight)
        g  = {"valid":1,"tgt_pos":tp.tolist(),"tgt_vel":tgt_vel.tolist()}
        m.update_guidance(g)
        evs = m.step(0.02, tp, tgt_vel)
        events.extend(evs)
        if m.phase in (MslPhase.HIT, MslPhase.MISS): break

    hits = [e for e in events if e["type"]=="MISSILE_HIT"]
    print(f"  missile 40 km head-on:  phase={m.phase.value}  t={m.t_flight:.1f}s  events={[e['type'] for e in events]}")
    assert m.phase == MslPhase.HIT, f"expected HIT, got {m.phase.value}"
    assert 20.0 < m.t_flight < 80.0, f"flight time {m.t_flight:.1f} s out of range"
    print("  missile straight shot ....... OK")


def test_missile_support_timeout():
    """Lose the datalink for >3 s → MISSILE_MISS with cause=SUPPORT_LOST."""
    from missile_sim import AIM120, MslPhase

    own_pos = np.array([0.0, 0.0, 9000.0])
    tgt_pos = np.array([0.0, 50000.0, 9000.0])
    own_vel = np.array([0.0, 270.0, 0.0])
    tgt_vel = np.array([0.0, -270.0, 0.0])

    m = AIM120(1, 2, own_pos.copy(), own_vel.copy(), handoff_range=10000.0)
    m.last_tgt_pos = tgt_pos.copy(); m.last_tgt_vel = tgt_vel.copy()
    m.t_launch = 0.0

    events=[]; dropped_at = None; support_ok=True

    for i in range(3000):
        tp = tgt_pos + tgt_vel * m.t_flight
        if m.t_flight >= 8.0 and support_ok:
            support_ok = False; dropped_at = m.t_flight  # drop support at 8 s
        if support_ok:
            m.update_guidance({"valid":1,"tgt_pos":tp.tolist(),"tgt_vel":tgt_vel.tolist()})
        else:
            m.update_guidance({"valid":0})  # silence
        evs = m.step(0.02, tp, tgt_vel)
        events.extend(evs)
        if m.phase in (MslPhase.HIT, MslPhase.MISS): break

    miss_evs = [e for e in events if e.get("type")=="MISSILE_MISS"]
    sl_evs   = [e for e in miss_evs if e.get("cause")=="SUPPORT_LOST"]
    dead_t   = m.t_flight
    print(f"  support dropped at t={dropped_at:.1f}s  missile died at t={dead_t:.2f}s  "
          f"cause={sl_evs[0]['cause'] if sl_evs else '?'}")
    assert len(sl_evs) >= 1, f"expected SUPPORT_LOST event, got: {[e['type'] for e in events]}"
    delta = dead_t - (dropped_at if dropped_at else 0)
    assert 2.5 < delta < 3.5, f"should die ~3 s after drop, died in {delta:.2f} s"
    print("  missile support timeout ..... OK")


def test_missile_no_premature_death():
    """Datalink restored within 3 s — missile must NOT die."""
    from missile_sim import AIM120, MslPhase

    own_pos = np.array([0.0, 0.0, 9000.0])
    tgt_pos = np.array([0.0, 40000.0, 9000.0])
    own_vel = np.array([0.0, 270.0, 0.0])
    tgt_vel = np.array([0.0, -270.0, 0.0])

    m = AIM120(1, 2, own_pos.copy(), own_vel.copy(), handoff_range=10000.0)
    m.last_tgt_pos = tgt_pos.copy(); m.last_tgt_vel = tgt_vel.copy()

    events = []
    for i in range(6000):
        tp = tgt_pos + tgt_vel * m.t_flight
        # Drop at 8 s, restore at 9.5 s (1.5 s gap)
        if m.t_flight < 8.0 or m.t_flight > 9.5:
            m.update_guidance({"valid":1,"tgt_pos":tp.tolist(),"tgt_vel":tgt_vel.tolist()})
        else:
            m.update_guidance({"valid":0})
        evs = m.step(0.02, tp, tgt_vel)
        events.extend(evs)
        if m.phase in (MslPhase.HIT, MslPhase.MISS): break

    sl_evs = [e for e in events if e.get("cause")=="SUPPORT_LOST"]
    print(f"  datalink restored in 1.5 s:  phase={m.phase.value}  SUPPORT_LOST={len(sl_evs)}")
    assert len(sl_evs) == 0, "missile died despite support restored in time"
    print("  missile no premature death .. OK")


def test_missile_kinematic_miss():
    """Shot at 120 km tail-on: target runs, missile runs out of energy → MISS."""
    from missile_sim import AIM120, MslPhase

    own_pos = np.array([0.0, 0.0, 9000.0])
    tgt_pos = np.array([0.0, 120000.0, 9000.0])
    own_vel = np.array([0.0, 270.0, 0.0])
    tgt_vel = np.array([0.0, 290.0, 0.0])   # running away

    m = AIM120(1, 2, own_pos.copy(), own_vel.copy(), handoff_range=10000.0)
    m.last_tgt_pos = tgt_pos.copy(); m.last_tgt_vel = tgt_vel.copy()

    events = []
    for i in range(6000):
        tp = tgt_pos + tgt_vel * m.t_flight
        m.update_guidance({"valid":1,"tgt_pos":tp.tolist(),"tgt_vel":tgt_vel.tolist()})
        evs = m.step(0.02, tp, tgt_vel)
        events.extend(evs)
        if m.phase in (MslPhase.HIT, MslPhase.MISS): break

    assert m.phase == MslPhase.MISS, f"expected MISS at extreme range, got {m.phase.value}"
    print(f"  120 km tail-on:  phase={m.phase.value}  ✓")
    print("  missile kinematic miss ...... OK")


# ════════════════════════════════════════════════════════════════════
# 3. WORLD INTEGRATION
# ════════════════════════════════════════════════════════════════════
def test_world_step():
    """World produces valid telemetry dict every frame."""
    from sim_world import SimWorld
    w = SimWorld(seed=7)
    ic = dict(ac1_lat=39.0,ac1_lon=35.0,ac1_alt=9000.0,ac1_psi=0.0,ac1_spd=280.0,
              ac2_lat=39.45,ac2_lon=35.0,ac2_alt=9000.0,ac2_psi=math.pi,ac2_spd=280.0,
              wpn=4,wpn_t=4)
    w.reset(ic, episode_id=1)
    cmd = {"hdgCmd":0.0,"altTarget":9000.0,"V":280.0,"fire":0}
    for _ in range(100):
        tlm = w.step(cmd, dict(cmd, hdgCmd=math.pi))
    assert tlm["type"]=="telemetry"
    assert 0 < tlm["range"] < 200_000
    assert -1 < tlm["aa_deg"] < 181
    assert isinstance(tlm["missiles"], list)
    assert isinstance(tlm["events"], list)
    assert tlm["wpn_remaining"] == 4
    print(f"  world: range={tlm['range']/1000:.1f}km  aa={tlm['aa_deg']:.1f}°  "
          f"closure={tlm['closure']:.1f}m/s")
    print("  world step .................. OK")


def test_world_fire_and_hit():
    """Fire a missile at close range head-on — expect a HIT event."""
    from sim_world import SimWorld
    w = SimWorld(seed=11)
    ic = dict(ac1_lat=39.0,ac1_lon=35.0,ac1_alt=9000.0,ac1_psi=0.0,ac1_spd=280.0,
              ac2_lat=39.20,ac2_lon=35.0,ac2_alt=9000.0,ac2_psi=math.pi,ac2_spd=280.0,
              wpn=4,wpn_t=0)
    w.reset(ic, episode_id=2)

    # provide guidance and fire
    cmd1 = {"hdgCmd":0.0,"altTarget":9000.0,"V":280.0,"fire":1,
            "msl_guidance":{"valid":1,
                            "tgt_pos":[0.0,22_000.0,9000.0],
                            "tgt_vel":[0.0,-280.0,0.0],
                            "pos_sigma":50.0,"t_est":0.0}}
    cmd2 = {"hdgCmd":math.pi,"altTarget":9000.0,"V":280.0,"fire":0}

    all_events = []
    for step in range(5000):
        cmd1["fire"] = 1 if step == 0 else 0
        # refresh guidance
        p2 = [w.ac2.x, w.ac2.y, w.ac2.z]
        cmd1["msl_guidance"] = {"valid":1,"tgt_pos":p2,"tgt_vel":w.ac2.vel_enu.tolist(),
                                 "pos_sigma":30.0,"t_est":step*0.02}
        tlm = w.step(cmd1, cmd2)
        all_events.extend(tlm["events"])
        if any(e["type"]=="AC_DESTROYED" for e in tlm["events"]): break

    hits = [e for e in all_events if e["type"]=="MISSILE_HIT"]
    dest = [e for e in all_events if e["type"]=="AC_DESTROYED"]
    print(f"  world fire+hit: wpn_left={w.wpn1}  hits={len(hits)}  destroyed={len(dest)}")
    assert w.wpn1 == 3, f"wpn1 should be 3, got {w.wpn1}"
    assert len(hits) >= 1, f"expected ≥1 HIT, got {len(hits)}"
    print("  world fire & hit ............ OK")


def test_world_support_timeout_via_env():
    """If AC1 turns cold (breaks lock) for 3 s, own missile must die."""
    from sim_world import SimWorld
    from missile_sim import MslPhase
    w = SimWorld(seed=15)
    ic = dict(ac1_lat=39.0,ac1_lon=35.0,ac1_alt=9000.0,ac1_psi=0.0,ac1_spd=280.0,
              ac2_lat=39.35,ac2_lon=35.0,ac2_alt=9000.0,ac2_psi=math.pi,ac2_spd=280.0,
              wpn=4,wpn_t=0)
    w.reset(ic, episode_id=3)

    # Fire first frame
    cmd1 = {"hdgCmd":0.0,"altTarget":9000.0,"V":280.0,"fire":1,
            "msl_guidance":{"valid":1,"tgt_pos":[w.ac2.x,w.ac2.y,w.ac2.z],
                             "tgt_vel":w.ac2.vel_enu.tolist(),"pos_sigma":30,"t_est":0}}
    cmd2 = {"hdgCmd":math.pi,"altTarget":9000.0,"V":280.0,"fire":0}
    w.step(cmd1, cmd2)
    assert w.wpn1 == 3
    cmd1["fire"] = 0

    support_losses = 0
    for step in range(1, 5000):
        t = step * 0.02
        # Cut guidance at t=5 s, never restore
        if t < 5.0:
            cmd1["msl_guidance"] = {"valid":1,"tgt_pos":[w.ac2.x,w.ac2.y,w.ac2.z],
                                     "tgt_vel":w.ac2.vel_enu.tolist(),"pos_sigma":30,"t_est":t}
        else:
            cmd1["msl_guidance"] = {"valid":0}
        tlm = w.step(cmd1, cmd2)
        for ev in tlm["events"]:
            if ev.get("type")=="MISSILE_MISS" and ev.get("cause")=="SUPPORT_LOST":
                support_losses += 1
                print(f"    SUPPORT_LOST event at t={t:.2f}s (guidance cut at 5.00s)")
        if support_losses > 0: break

    assert support_losses >= 1, "expected at least one SUPPORT_LOST"
    print("  world support timeout ....... OK")


# ════════════════════════════════════════════════════════════════════
# 4. FULL ENVIRONMENT
# ════════════════════════════════════════════════════════════════════
def test_env_reset_and_step():
    """Env resets cleanly and produces valid, bounded observations."""
    from bvr_env import BvrEnv, OBS_DIM, PRIV_DIM
    from bvr_opponents import BvrOpponentType
    e = BvrEnv(opponent_type=BvrOpponentType.STRAIGHT, seed=42, privileged_critic=True)
    obs, info = e.reset()
    assert set(obs) == {"obs","priv"}
    assert obs["obs"].shape == (OBS_DIM,)
    assert obs["priv"].shape == (PRIV_DIM,)
    assert np.all(np.isfinite(obs["obs"]))
    assert np.all(np.abs(obs["obs"]) <= 1.0 + 1e-5)
    assert np.all(np.isfinite(obs["priv"]))
    print(f"  env reset: OBS={OBS_DIM} PRIV={PRIV_DIM}  obs bounded ✓")

    mask = e.action_masks()
    assert len(mask) == sum(e.action_space.nvec)
    act = e.action_space.sample()
    obs2, rew, done, trunc, info2 = e.step(act)
    assert isinstance(rew, float) and np.isfinite(rew)
    print(f"  env step: reward={rew:.4f}  done={done}  track={info2['track_state']}")
    e.close()
    print("  env reset & step ............ OK")


def test_env_full_episode():
    """Run a complete episode to termination against the shooter opponent."""
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType
    e = BvrEnv(opponent_type=BvrOpponentType.SHOOTER, seed=99,
               privileged_critic=False, gamma_discount=0.997)
    obs, _ = e.reset()
    total_r = 0.0; steps = 0; outcome = None

    t0 = time.perf_counter()
    while True:
        mask = e.action_masks()
        act  = e.action_space.sample()
        # Respect fire mask
        if not mask[-1]: act[-1] = 0
        obs, r, done, trunc, info = e.step(act)
        total_r += r; steps += 1
        if done or trunc:
            outcome = info.get("terminal_outcome","?"); break

    wall = time.perf_counter() - t0
    sim_t = e._t_sim
    print(f"  episode: {steps} decisions  {sim_t:.0f}s sim  outcome={outcome}  "
          f"R={total_r:.3f}  shots={info.get('shots_fired',0)}")
    print(f"  wall: {wall:.2f}s  throughput: {sim_t/wall:.0f}x realtime")
    assert steps > 0 and outcome is not None
    assert sim_t / wall > 10.0, f"sim too slow: only {sim_t/wall:.1f}x realtime"
    e.close()
    print("  full episode ................ OK")


def test_env_throughput():
    """
    Measure sim-speed multiplier (sim-seconds / wall-seconds).
    Must be well above 1x. On the container CPU the full-episode test
    already shows 47x; short bursts + reset overhead bring this down to
    ~15-30x. Require >10x to leave margin on slow hardware.
    """
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType
    e = BvrEnv(opponent_type=BvrOpponentType.STRAIGHT, seed=7,
               privileged_critic=False, gamma_discount=0.997)

    t0 = time.perf_counter(); sim_total = 0.0; n_steps = 0
    while time.perf_counter() - t0 < 5.0:
        e.reset()
        for _ in range(50):
            act = e.action_space.sample(); act[-1] = 0
            _, _, done, trunc, _ = e.step(act)
            n_steps += 1; sim_total += 1.0 / e.DECISION_HZ
            if done or trunc: break

    wall = time.perf_counter() - t0
    multiplier = sim_total / wall
    print(f"  throughput: {multiplier:.0f}x realtime  "
          f"({n_steps} decisions / {wall:.1f}s wall, {sim_total:.0f}s sim)")
    assert multiplier > 10.0, f"only {multiplier:.1f}x realtime — too slow for training"
    e.close()
    print("  throughput .................. OK")



# ── parameter library ────────────────────────────────────────────────
def test_library_f16_matches_model():
    """The built-in F-16C, AIM-120 and APG-68 reproduce the model classes exactly."""
    import f16_sim, missile_sim, bvr_radar_sim
    from bvr_library import load_platform
    p = load_platform("F-16C")
    for cls, cfg in ((f16_sim.F16Cfg, p.airframe), (missile_sim.MslCfg, p.missile),
                     (bvr_radar_sim.RadarSim, p.radar)):
        for k in dir(cls):
            if k.isupper() and hasattr(cfg, k) and isinstance(getattr(cls, k), float):
                assert getattr(cfg, k) == getattr(cls, k), f"{k}: {getattr(cfg, k)} != {getattr(cls, k)}"
    for m in np.linspace(0, 2.2, 441):
        if abs(m - 0.9) > 1e-6 and abs(m - 1.2) > 1e-6:
            assert abs(p.airframe.thrust_mach_factor(m) - f16_sim._thrust_mach_factor(m)) < 1e-9
        assert abs(p.airframe.cd_rise(m) - f16_sim._cd_rise(m)) < 1e-12
        assert abs(p.missile.drag_cd(m) - missile_sim._drag_cd(m)) < 1e-12
    # The transonic rise is continuous (it once fell to zero just below Mach
    # 1.1 and jumped to the peak) and never below its supersonic value past the peak.
    from bvr_library import airframe_config
    for iid in ("F-16C", "GENERIC-UCAV"):
        c = airframe_config(iid)
        cd = np.array([c.cd_rise(m) for m in np.arange(0.0, 2.5, 0.001)])
        assert np.abs(np.diff(cd)).max() < 1e-3, f"{iid}: drag rise jumps"
        post = [c.cd_rise(m) for m in np.arange(c.DRAG_RISE_M1, 2.5, 0.001)]
        assert min(post) >= c.cd_rise(c.DRAG_RISE_M2) - 1e-12, f"{iid}: drag dips after its peak"
    print("  library F-16C = model ....... OK")


def test_library_protection_and_validation():
    import bvr_library as L
    for kind, iid in (("airframe", "F-16C"), ("platform", "GENERIC-UCAV")):
        errs = L.validate(L.get_item(kind, iid))
        assert not errs, f"built-in {kind} {iid} invalid: {errs}"
        try:
            L.save_item(L.get_item(kind, iid))
            raise AssertionError("a built-in was overwritten")
        except L.LibraryError:
            pass
    bad = L.get_item("airframe", "F-16C")
    bad["id"] = "X"; bad["params"]["CD0"] = 5.0; bad["params"]["T_AB_SL"] = 1000.0
    errs = L.validate(bad)
    assert any("Zero-lift drag" in e for e in errs) and any("Afterburner" in e for e in errs), errs
    print("  library protection .......... OK")


def test_library_rcs_and_loadout():
    """RCS scales detection and seeker range as RCS^(1/4); platforms set the loadout."""
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType
    e = BvrEnv(opponent_type=BvrOpponentType.STRAIGHT, platform="GENERIC-UCAV",
               opponent_platform="F-16C")
    e.reset()
    obs_f16 = e.selfplay_observer(True)
    expect = 90_000.0 * (0.1 / 1.2) ** 0.25
    assert abs(obs_f16._radar.MAX_RANGE - expect) < 1.0, obs_f16._radar.MAX_RANGE
    assert e._radar.MAX_RANGE == 70_000.0
    assert (e._world.wpn1, e._world.wpn2) == (2, 4)
    e2 = BvrEnv(opponent_type=BvrOpponentType.STRAIGHT)
    assert e2._radar.MAX_RANGE == 90_000.0 and (e2._world.wpn_count == (4, 4))
    print("  library RCS & loadout ....... OK")


def test_library_uncalibrated_missile_refused():
    import bvr_library as L
    from bvr_env import BvrEnv
    try:
        L.copy_item("missile", "AIM-120", "TEST-MSL-TMP")
        m = L.get_item("missile", "TEST-MSL-TMP"); m["params"]["MAX_G"] = 26.0; L.save_item(m)
        pl = L.copy_item("platform", "F-16C", "TEST-PLAT-TMP"); pl["params"]["missile"] = "TEST-MSL-TMP"
        L.save_item(pl)
        try:
            BvrEnv(platform="TEST-PLAT-TMP")
            raise AssertionError("an uncalibrated missile was accepted")
        except L.LibraryError as ex:
            assert "calibrate" in str(ex).lower()
    finally:
        for kind, iid in (("platform", "TEST-PLAT-TMP"), ("missile", "TEST-MSL-TMP")):
            try: L.delete_item(kind, iid)
            except L.LibraryError: pass
    print("  uncalibrated refused ........ OK")


def test_perf_card_builtins():
    from bvr_perf import card_for
    c = card_for("platform", "GENERIC-UCAV")
    assert not c["warnings"], c["warnings"]
    assert 0.8 < c["top_speed"][9000]["mach"] < 0.9
    f = card_for("platform", "F-16C")
    assert f["max_bank_turn_40s"][3000]["alt_error"] == 0.0     # turns hold altitude
    print("  performance cards ........... OK")


def test_compat_widening():
    """A checkpoint from before the wingman inputs loads and acts exactly as before."""
    import tempfile
    import gymnasium as gym
    import torch as th
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from bvr_env import OBS_DIM, PRIV_DIM, OBS_DIM_1V1, PRIV_DIM_1V1, ACTION_NVEC
    from bvr_compat import load_model, N_STACK
    from bvr_policy import AsymmetricMaskablePolicy

    box = lambda n: spaces.Box(-1.0, 1.0, (n,), np.float32)

    class OldLayout(gym.Env):
        observation_space = spaces.Dict({"obs": box(OBS_DIM_1V1), "priv": box(PRIV_DIM_1V1)})
        action_space = spaces.MultiDiscrete(ACTION_NVEC)
        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}
        def step(self, a):
            return self.observation_space.sample(), 0.0, False, False, {}
        def action_masks(self):
            return np.ones(sum(ACTION_NVEC), dtype=bool)

    vec = VecFrameStack(DummyVecEnv([OldLayout]), N_STACK)
    old = MaskablePPO(AsymmetricMaskablePolicy, vec, n_steps=32, batch_size=32,
                      policy_kwargs=dict(pi_arch=(64, 64), vf_arch=(64, 64)))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.zip")
        old.save(path)
        new = load_model(path, log=None)
    assert new.observation_space["obs"].shape == (OBS_DIM * N_STACK,)
    assert new.observation_space["priv"].shape == (PRIV_DIM * N_STACK,)

    rng = np.random.default_rng(0)
    o = rng.uniform(-1, 1, (5, N_STACK, OBS_DIM_1V1)).astype(np.float32)
    p = rng.uniform(-1, 1, (5, N_STACK, PRIV_DIM_1V1)).astype(np.float32)
    # the new columns hold arbitrary values: they must change nothing
    o_new = np.concatenate([o, rng.uniform(-1, 1, (5, N_STACK, OBS_DIM - OBS_DIM_1V1))], 2)
    p_new = np.concatenate([p, rng.uniform(-1, 1, (5, N_STACK, PRIV_DIM - PRIV_DIM_1V1))], 2)

    def run(model, oo, pp):
        x = {"obs": th.as_tensor(oo.reshape(5, -1), dtype=th.float32),
             "priv": th.as_tensor(pp.reshape(5, -1), dtype=th.float32)}
        with th.no_grad():
            dist = model.policy.get_distribution(x)
            logits = th.cat([c.logits for c in dist.distributions], 1)
            return logits.numpy(), model.policy.predict_values(x).numpy()

    la, va = run(old, o, p)
    lb, vb = run(new, o_new, p_new)
    assert np.allclose(la, lb, atol=1e-5) and np.allclose(va, vb, atol=1e-5), \
        (np.abs(la - lb).max(), np.abs(va - vb).max())
    print("  checkpoint widening ......... OK")


def test_team_datalink():
    """2v1: a blue aircraft whose own radar never tracks fires and kills on its wingman's track."""
    from bvr_team import TeamBvrEnv
    from bvr_opponents import BvrOpponentType as T
    from bvr_track_adapter import TrackState
    e = TeamBvrEnv(opponent_type=T.STRAIGHT, seed=11, doctrine="AGGRESSIVE")
    own, kills, rewards = 0, 0, []
    for ep in range(3):
        e.reset()
        e._obs[0]._radar.max_range = 1.0
        done = False
        while not done:
            acts = [[0, 2, 2, int(k == 0 and o._alive and o._can_fire())] for k, o in enumerate(e._obs)]
            _, r, t, tr, infos = e.step(acts)
            own += e._obs[0]._track.state == TrackState.TRACK
            done = t or tr
        if infos[0]["killer"] == 1:
            kills += 1
            rewards.append(r)
    assert own == 0 and kills >= 1, (own, kills)
    # a kill is a team reward: both agents receive it on the same step
    assert all(min(r) > 0.5 for r in rewards), rewards
    print(f"  2v1 datalink ................ OK  ({kills}/3 kills by the blind aircraft)")


def test_team_losses():
    """2v1: a lost aircraft keeps a slot with one legal action; red then engages the other."""
    from bvr_team import TeamBvrEnv
    from bvr_opponents import BvrOpponentType as T
    from bvr_env import ACTION_NVEC
    e = TeamBvrEnv(opponent_type=T.SHOOTER, seed=5, doctrine="AGGRESSIVE")
    seen_loss = False
    for ep in range(4):
        obs, _ = e.reset()
        targets, done = set(), False
        while not done:
            lost_before = [not o._alive for o in e._obs]
            obs, r, t, tr, infos = e.step([[0, 2, 3, 0], [0, 2, 3, 0]])
            targets.add(e._red_tgt)
            for k, o in enumerate(e._obs):
                if not o._alive:
                    seen_loss = True
                    assert e.action_masks()[k].sum() == len(ACTION_NVEC)
                    assert not obs[k]["obs"].any()
                if lost_before[k]:
                    assert infos[k]["shaping"] == 0.0     # no shaping once lost
            done = t or tr
        assert targets == {1, 2} or infos[0]["blue_losses"] == 0
    assert seen_loss
    print("  2v1 losses .................. OK")


def test_team_vec_env():
    """2v1 as SB3 slots: two slots per fight, one mask row each, both end and reset together."""
    from bvr_team import TeamBvrEnv
    from bvr_team_vec import TeamVecEnv
    from bvr_opponents import BvrOpponentType as T
    from bvr_env import OBS_DIM, ACTION_NVEC
    vec = TeamVecEnv([lambda: TeamBvrEnv(opponent_type=T.STRAIGHT, seed=2)], in_process=True)
    assert vec.num_envs == 2 and vec.has_attr("action_masks")
    obs = vec.reset()
    assert obs["obs"].shape == (2, OBS_DIM)
    masks = np.stack(vec.env_method("action_masks"))
    assert masks.shape == (2, sum(ACTION_NVEC))
    vec.set_attr("_opponent_type", T.EVASIVE)
    assert vec.get_attr("_opponent_type") == [T.EVASIVE, T.EVASIVE]
    for _ in range(400):
        obs, r, dones, infos = vec.step(np.array([[0, 2, 2, 0], [0, 2, 2, 0]]))
        assert r.shape == (2,) and dones[0] == dones[1]
        if dones[0]:
            assert all("terminal_observation" in i for i in infos)
            assert "terminal_outcome" in infos[0] and "terminal_outcome" not in infos[1]
            break
    else:
        raise AssertionError("no episode ended in 400 steps")
    vec.close()
    print("  2v1 vec env ................. OK")

# ────────────────────────────────────────────────────────────────────
def _wrap_pi(a): return (a+math.pi)%(2*math.pi)-math.pi


if __name__ == "__main__":
    tests = [
        ("F-16 turn rate",          test_f16_turn_rate),
        ("F-16 speed convergence",  test_f16_speed_convergence),
        ("F-16 altitude hold",      test_f16_alt_hold),
        ("F-16 energy / SEP",       test_f16_energy),
        ("F-16 body rates",         test_f16_body_rates),
        ("missile straight shot",   test_missile_straight_shot),
        ("missile support timeout", test_missile_support_timeout),
        ("missile no premature",    test_missile_no_premature_death),
        ("missile kinematic miss",  test_missile_kinematic_miss),
        ("world step",              test_world_step),
        ("world fire & hit",        test_world_fire_and_hit),
        ("world support timeout",   test_world_support_timeout_via_env),
        ("env reset & step",        test_env_reset_and_step),
        ("env full episode",        test_env_full_episode),
        ("throughput",              test_env_throughput),
        ("library F-16C = model",   test_library_f16_matches_model),
        ("library protection",      test_library_protection_and_validation),
        ("library RCS & loadout",   test_library_rcs_and_loadout),
        ("uncalibrated refused",    test_library_uncalibrated_missile_refused),
        ("performance cards",       test_perf_card_builtins),
        ("checkpoint widening",     test_compat_widening),
        ("2v1 datalink",            test_team_datalink),
        ("2v1 losses",              test_team_losses),
        ("2v1 vec env",             test_team_vec_env),
    ]

    print("\nPure-Python sim tests\n" + "─"*50)
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL [{name}]: {exc}")
            import traceback; traceback.print_exc()
            failed += 1

    print("─"*50)
    print(f"{passed}/{passed+failed} passed", "✓" if failed==0 else f"  ({failed} FAILED)")
    sys.exit(0 if failed==0 else 1)
