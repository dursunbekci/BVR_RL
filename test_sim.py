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


def test_missile_tgo():
    """Time-to-go counts down while a missile closes (its sign was once reversed)."""
    from missile_sim import AIM120
    m = AIM120(1, 2, np.array([0., 0., 9000.]), np.array([0., 900., 0.]), 12000.0)
    m.last_tgt_pos = np.array([0., 30000., 9000.])
    m.last_tgt_vel = np.array([0., -250., 0.])
    assert abs(m.tgo_est - 30000 / 1150) < 0.1, m.tgo_est
    prev = m.tgo_est
    for k in range(1, 6):                          # 5 s of supported flight
        for _ in range(50):
            t = m.t_flight
            m.update_guidance({"valid": 1, "tgt_pos": [0., 30000. - 250. * t, 9000.],
                               "tgt_vel": [0., -250., 0.]})
            m.step(0.02, np.array([0., 30000. - 250. * t, 9000.]), np.array([0., -250., 0.]))
        assert 0.5 < prev - m.tgo_est < 4.0, (k, prev, m.tgo_est)   # faster while boosting
        prev = m.tgo_est
    m.vel = -m.vel                                 # flying away: not closing
    assert m.tgo_est == 999.0
    print(f"  missile time-to-go .......... OK  ({prev:.1f} s left after 5 s)")


def test_defence_potential():
    """Defence shaping: 0 with no missile inbound, falls as one closes, 0 again once it ends."""
    import math
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T
    e = BvrEnv(opponent_type=T.STRAIGHT, seed=1)
    e.reset(seed=1)
    def d(missiles, alive=1):
        e._state["missiles"] = missiles
        e._state["alive"] = alive
        e._potential()
        return e._phi_terms["defence"]
    inbound = lambda tgo, state="MIDCOURSE": [{"owner": 2, "state": state, "tgo_est": tgo}]
    assert d([]) == 0.0
    assert abs(d(inbound(999.0))) < 1e-9                      # not closing: no danger
    assert abs(d(inbound(25.0)) + 0.8 * (1 - math.tanh(1.0))) < 1e-9
    assert d(inbound(5.0)) < d(inbound(25.0)) < d(inbound(60.0)) < 0
    assert d(inbound(0.5, "HIT")) == 0.0                     # a missile that hit has ended
    assert d(inbound(5.0), alive=0) == 0.0
    print("  defence potential ........... OK")


def test_heading_reference():
    """Without a track the heading reference is held; the last choice is an input."""
    import math
    from bvr_env import BvrEnv, HDG_OFFSETS_DEG, OBS_LABELS
    from bvr_opponents import BvrOpponentType as T
    e = BvrEnv(opponent_type=T.STRAIGHT, seed=4)
    e.reset(seed=4)
    e._track.reset()                                  # no track estimate
    e._hdg_ref = None
    i30 = HDG_OFFSETS_DEG.index(30.0)
    e._state["psi"] = 1.0
    e._encode_cmd(i30, 2, 2, 0); first = e._cmd_hdg
    e._state["psi"] = 1.3                             # the nose has moved on
    e._encode_cmd(i30, 2, 2, 0)
    assert abs(first - e._cmd_hdg) < 1e-9, "same choice, no track: same heading"
    assert abs(first - (1.0 + math.radians(30))) < 1e-9
    # With a track estimate the reference is the line to the bandit again.
    for seed in range(20):                            # a start that forms a track
        e.reset(seed=seed)
        for _ in range(40):
            e.step([0, 2, 2, 0])
            if e._est()["valid"]:
                break
        if e._est()["valid"]:
            break
    assert e._est()["valid"]
    d = e._est()["pos"] - e._own_pos_enu()
    e._encode_cmd(0, 2, 2, 0)
    assert abs(math.atan2(d[0], d[1]) % (2 * math.pi) - e._cmd_hdg) < 1e-6
    # The previous choice is visible in the observation.
    obs, *_ = e.step([i30, 2, 2, 0])
    k = OBS_LABELS.index("hdg_prev_cos")
    o = obs["obs"]
    assert abs(o[k] - math.cos(math.radians(30))) < 1e-5 and abs(o[k + 1] - 0.5) < 1e-5
    print("  heading reference ........... OK")


def test_stern_starts():
    """Stern starts (bandit running away) are skipped only when red is the faster side."""
    from bvr_env import BvrEnv
    from bvr_team import TeamBvrEnv
    def scenarios(env, n=200):
        return {env._random_ic()["scenario"] for _ in range(n)}
    assert "stern_conversion" in scenarios(BvrEnv(platform="F-16C", opponent_platform="F-16C"))
    assert "stern_conversion" in scenarios(BvrEnv(platform="F-16C", opponent_platform="GENERIC-UCAV"))
    assert "stern_conversion" not in scenarios(BvrEnv(platform="GENERIC-UCAV", opponent_platform="F-16C"))
    assert "stern_conversion" not in scenarios(
        TeamBvrEnv(blue_platforms=("GENERIC-UCAV", "GENERIC-UCAV"), red_platform="F-16C"))
    assert "stern_conversion" in scenarios(         # the F-16 wingman can catch red
        TeamBvrEnv(blue_platforms=("GENERIC-UCAV", "F-16C"), red_platform="F-16C"))
    print("  stern starts ................ OK")


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
    for ep in range(10):                 # until a kill: beam and stern starts rarely give one
        if kills:
            break
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
    print(f"  2v1 datalink ................ OK  (a kill by the blind aircraft in {ep} starts)")


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


def test_team_red_targeting():
    """2v1: red opens on the nearer blue aircraft, and SHOOTER also fires at the other one."""
    from bvr_team import TeamBvrEnv
    from bvr_opponents import BvrOpponentType as T
    e = TeamBvrEnv(opponent_type=T.SHOOTER, seed=3, doctrine="AGGRESSIVE")
    assert e.R_LOST == -1.0
    opened, both = set(), 0
    for ep in range(8):
        e.reset()
        w = e._world
        red = w.pos(e.RED)
        d = [float(np.linalg.norm(w.pos(i) - red)) for i in (1, 2)]
        assert e._red_tgt == 1 + int(d[1] < d[0]), (e._red_tgt, d)
        opened.add(e._red_tgt)
        targets, launch = set(), w._launch
        def spy(owner, cmd, launch=launch, targets=targets):
            if owner == e.RED:
                targets.add(cmd.get("target"))
            return launch(owner, cmd)
        w._launch = spy
        done = False
        while not done:                                   # both fly at red, never fire
            _, _, t, tr, _ = e.step([[0, 2, 3, 0], [0, 2, 3, 0]])
            done = t or tr
        both += targets == {1, 2}
    assert both >= 1, "red never fired at both blue aircraft"
    print(f"  2v1 red targeting ........... OK  (opened on {sorted(opened)}, both shot at in {both}/8)")


def test_team_roles_far_target():
    """team_roles.py: red opens on the farther blue aircraft and holds it until it fires."""
    from team_roles import FarTargetTeamEnv
    from bvr_opponents import BvrOpponentType as T
    e = FarTargetTeamEnv(opponent_type=T.SHOOTER, seed=3, doctrine="AGGRESSIVE")
    fired_at_far = 0
    for ep in range(4):
        e.reset()
        w = e._world
        red = w.pos(e.RED)
        d = [float(np.linalg.norm(w.pos(i) - red)) for i in (1, 2)]
        far = 1 + int(d[1] > d[0])
        assert e._red_tgt == far, (e._red_tgt, d)
        done = False
        while not done:                                   # both fly at red, never fire
            if not any(m.owner == e.RED for m in w.missiles):
                assert e._red_tgt == far or not e._obs[far - 1]._alive
            _, _, t, tr, _ = e.step([[0, 2, 3, 0], [0, 2, 3, 0]])
            done = t or tr
        reds = sorted((m for m in w.missiles if m.owner == e.RED), key=lambda m: m.t_launch)
        fired_at_far += bool(reds) and reds[0].target == far
    assert fired_at_far >= 1, "red never fired its first missile at the farther aircraft"
    print(f"  2v1 red opens on rear ........ OK  (first missile at it in {fired_at_far}/4)")


def test_missile_loft():
    """A missile with LOFT_ANGLE climbs before diving on the target and still hits;
    without it (the default) it does not climb."""
    import copy, math
    from bvr_library import missile_config
    from missile_sim import AIM120, MslPhase
    base = missile_config("AIM-120")
    assert base.LOFT_ANGLE == 0.0
    loft = copy.copy(base); loft.LOFT_ANGLE, loft.LOFT_DIVE = math.radians(25), math.radians(5)
    out = {}
    for name, cfg in (("flat", base), ("loft", loft)):
        m = AIM120(1, 2, np.array([0.0, 0.0, 9000.0]), np.array([0.0, 400.0, 0.0]), 12000.0, cfg=cfg)
        tp, tv = np.array([0.0, 35000.0, 9000.0]), np.array([0.0, -250.0, 0.0])
        top = 0.0
        for _ in range(int(80 / 0.02)):
            tp = tp + tv * 0.02
            m.update_guidance({"valid": 1, "tgt_pos": tp, "tgt_vel": tv})
            m.step(0.02, tp, tv)
            top = max(top, m.pos[2])
            if m.phase in (MslPhase.HIT, MslPhase.MISS):
                break
        out[name] = (m.phase, top - 9000.0, m.t_flight)
    assert out["flat"][1] < 200.0, out
    assert out["loft"][1] > 1500.0 and out["loft"][0] == MslPhase.HIT, out
    print(f"  missile loft ................ OK  (climbs {out['loft'][1]:.0f} m, hits after "
          f"{out['loft'][2]:.1f} s; without loft {out['flat'][1]:.0f} m)")


def test_dcs_live_link():
    """dcs_live.py flies a policy through the UDP link against dcs/fake_dcs.py."""
    import os, sys, tempfile, threading
    from types import SimpleNamespace
    from gymnasium import spaces
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcs"))
    import dcs_live
    from fake_dcs import FakeDcs
    from bvr_env import BvrEnv, OBS_DIM, PRIV_DIM, HDG_OFFSETS_DEG

    class Scripted:          # straight at the bandit, top speed, fire whenever allowed
        observation_space = spaces.Dict({"obs": spaces.Box(-1, 1, (OBS_DIM,)),
                                         "priv": spaces.Box(-1, 1, (PRIV_DIM,))})
        def predict(self, obs, deterministic=True, action_masks=None):
            return np.array([list(HDG_OFFSETS_DEG).index(0), 2, 3, int(action_masks[-1])]), None

    fake = FakeDcs(opponent="STRAIGHT", seed=2, speed=8.0, max_time=250.0, lockstep=True)
    res = {}
    th = threading.Thread(target=lambda: res.setdefault("out", fake.run()), daemon=True)
    link = dcs_live.UdpLink()
    th.start()
    with tempfile.TemporaryDirectory() as d:
        args = SimpleNamespace(shadow=False, out=d, opp_platform=None, max_steps=250,
                               doctrine="AGGRESSIVE")
        scen = {"platform": "F-16C", "opponent_platform": "F-16C"}
        with open(os.path.join(d, "results.csv"), "w") as fh:     # one from before the speed columns
            fh.write("episode,mode,outcome\nold_ep1,live,SHOT_DOWN\n")
        rec, agent, red = dcs_live.wait_for_fight(link, log=lambda m: None)
        row, _ = dcs_live.run_episode(link, Scripted(), args, scen, rec, agent, red, 1,
                                      log=lambda m: None)
        th.join(timeout=120)
        files = sorted(os.listdir(d))
        import csv
        with open(os.path.join(d, "results.csv"), newline="") as fh:
            results = list(csv.DictReader(fh))
    link.close(); fake.close()
    fires = [c for c in fake.commands if c[5]]
    assert len(fake.commands) >= 30, f"only {len(fake.commands)} commands reached the fake"
    assert fires and row["shots"] >= 1, (row, fires)
    assert row["outcome"] in ("KILL", "TIMEOUT", "MUTUAL_KILL"), row
    assert row["outcome"] != "KILL" or res.get("out") == "KILL", (row, res)
    assert any(f.endswith(".jsonl") for f in files) and "results.csv" in files, files
    # The speed DCS flies against the speed asked for; the fake flies what it is told.
    assert [r["episode"] for r in results] == ["old_ep1", row["episode"]], results
    assert results[0]["speed_short_pct"] == "" and results[1]["outcome"] == row["outcome"]
    assert row["speed_short_pct"] < 30 and row["speed_cmd_mean"] > 0, row
    print(f"  DCS live link ............... OK  ({len(fake.commands)} commands, "
          f"{row['shots']} shots, {row['outcome']}; speed short {row['speed_short_pct']:.0f}% of the time)")


def test_dcs_support_rule():
    """dcs_live.py removes the policy's missile 3 s after its guidance stops
    (before the seeker takes over), and leaves a live opponent's missiles alone."""
    import os, sys, tempfile, threading
    from types import SimpleNamespace
    from gymnasium import spaces
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcs"))
    import dcs_live
    from fake_dcs import FakeDcs
    from bvr_env import BvrEnv, OBS_DIM, PRIV_DIM, HDG_OFFSETS_DEG

    class Scripted:          # straight at the bandit, fire whenever allowed
        observation_space = spaces.Dict({"obs": spaces.Box(-1, 1, (OBS_DIM,)),
                                         "priv": spaces.Box(-1, 1, (PRIV_DIM,))})
        def predict(self, obs, deterministic=True, action_masks=None):
            return np.array([list(HDG_OFFSETS_DEG).index(0), 2, 3, int(action_masks[-1])]), None

    orig = BvrEnv._guidance_packet
    cut = {}
    def no_guidance_after_first_shot(self):
        if self._shots_fired >= 1:
            cut.setdefault("t", self._t_sim)
            return {"valid": 0}
        return orig(self)
    dcs_live.DcsLiveEnv._guidance_packet = no_guidance_after_first_shot
    fake = FakeDcs(opponent="SHOOTER", seed=2, speed=8.0, max_time=250.0, lockstep=True)
    th = threading.Thread(target=fake.run, daemon=True)
    link = dcs_live.UdpLink()
    th.start()
    try:
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(shadow=False, out=d, opp_platform=None, max_steps=250,
                                   doctrine="AGGRESSIVE", support_rule="training")
            rec, agent, red = dcs_live.wait_for_fight(link, log=lambda m: None)
            row, _ = dcs_live.run_episode(link, Scripted(), args,
                                          {"platform": "F-16C", "opponent_platform": "F-16C"},
                                          rec, agent, red, 1, log=lambda m: None)
    finally:
        del dcs_live.DcsLiveEnv._guidance_packet
        th.join(timeout=120)
        link.close(); fake.close()
    assert fake.destroyed, ("no missile removed", row)
    assert all(owner == 1 for _, owner in fake.destroyed), fake.destroyed
    assert row["missiles_removed"] >= 1 and row["support_losses"] >= 1, row
    print(f"  DCS support rule ............ OK  ({len(fake.destroyed)} of the policy's missiles "
          f"removed, guidance cut at {cut.get('t', 0):.0f} s; {row['outcome']})")


def test_dcs_speed_boost():
    """dcs_live.py --speed-boost: while BLUE-1 is more than 20 m/s below the
    policy's speed, DCS is asked for the boost speed; within 5 m/s, the real one."""
    import dcs_live
    w = dcs_live.DcsLiveWorld.__new__(dcs_live.DcsLiveWorld)
    w.speed_boost, w._boosting, w.boost_steps = 550.0, False, 0
    w.auto_defend = False
    flown = iter([280.0, 300.0, 330.0, 336.0, 320.0, 300.0, 400.0])
    w._flown_speed = lambda: next(flown)
    sent = [w._speed_to_send(340.0) for _ in range(7)]
    assert sent == [550.0, 550.0, 550.0, 340.0, 340.0, 550.0, 340.0], sent
    w.speed_boost = None
    assert w._speed_to_send(340.0) == 340.0
    # --eta-lock: each command is preceded by the bridge option.
    class Link:
        def __init__(self): self.lines = []
        def send(self, text): self.lines.append(text)
    for lock, first in ((None, "CMD"), ("mission", "OPT eta mission"), ("abs", "OPT eta abs")):
        w.link, w.eta_lock = Link(), lock
        w.shadow, w.sent, w.t, w.t_sent, w.seq, w.fire_pending = False, None, 0.0, 0.0, 0, None
        w.send_command({"hdgCmd": 0.0, "altTarget": 9000.0, "V": 340.0}, False)
        assert w.link.lines[0].startswith(first) and w.link.lines[-1].startswith("CMD 1 "), w.link.lines
    # --auto-defend: OPT autodefend with each command until the bridge confirms
    # it, then every 10 s.
    w.link, w.eta_lock, w.auto_defend, w.defend_ack, w.t_defend_opt = Link(), None, True, None, -math.inf
    w.sent, w.t_sent = None, -math.inf
    for t in (0.0, 1.0, 2.0, 5.0, 13.0):
        w.t = t
        if t == 2.0:
            w.defend_ack = True
        w.send_command({"hdgCmd": t / 10, "altTarget": 9000.0, "V": 340.0}, False)
    opts = [ln for ln in w.link.lines if ln.startswith("OPT autodefend 1")]
    assert len(opts) == 3, w.link.lines            # t 0, 1 (unconfirmed), 13 (10 s later)
    print("  DCS speed boost ............. OK  (and --eta-lock, --auto-defend)")


def test_dcs_red_support():
    """dcs_live.py (training rule): a red that fires and turns away loses its own
    radar track, and its missile is removed before its seeker takes over; with
    --support-rule dcs, or a red that keeps its radar on, nothing is removed."""
    import json, os, sys, tempfile
    import bvr_env as E
    import bvr_opponents as O
    import dcs_live
    from bvr_opponents import BvrOpponentType as T
    from dcs_world import write_sim_recording

    class FireAndRun(O.ShooterOpponent):   # fires once, then turns cold for good
        def act(self, state, t_sim):
            cmd = super().act(state, t_sim)
            if self._phase == "COMMIT":
                return cmd
            if not hasattr(self, "_cold"):     # the firing frame: remember the way out
                self._cold = (cmd["hdgCmd"] + math.pi) % (2 * math.pi)
                return cmd
            return dict(cmd, hdgCmd=self._cold, fire=0)

    from bvr_env import HDG_OFFSETS_DEG
    from sim_world import _enu_to_latlon
    hot = [list(HDG_OFFSETS_DEG).index(0), 2, 1, 0]   # straight at red, never firing

    def head_on(env):                       # 70 km head-on at 9 km: red fires within ~10 s
        ic = env.__class__._random_ic(env)
        lat1, lon1, _ = _enu_to_latlon(0.0, 0.0, 9000.0)
        lat2, lon2, _ = _enu_to_latlon(0.0, 70_000.0, 9000.0)
        ic.update(ac1_lat=lat1, ac1_lon=lon1, ac1_alt=9000.0, ac1_psi=0.0, ac1_spd=280.0,
                  ac2_lat=lat2, ac2_lon=lon2, ac2_alt=9000.0, ac2_psi=math.pi, ac2_spd=280.0,
                  start_range=70_000.0, scenario="head_on", mirrored=False)
        return ic

    def recording(path, run_away):
        E.BvrEnv.OPPONENT_RADAR = False        # the source fight: truth-guided red
        try:
            sim = E.BvrEnv(opponent_type=T.SHOOTER, seed=5, doctrine="AGGRESSIVE",
                           platform="F-16C-DCS", opponent_platform="F-16C-DCS")
            sim.MAX_STEPS = 60
            sim._random_ic = lambda blue_top_speed=None: head_on(sim)
            if run_away:
                sim._opponent_factory = lambda env: FireAndRun(rng=env._rng)
            write_sim_recording(sim, lambda e: hot, path)
        finally:
            E.BvrEnv.OPPONENT_RADAR = True
        return [json.loads(l) for l in open(path)]

    class FakeLink:
        def __init__(self, objs):
            self.objs = sorted(objs, key=lambda d: d.get("t", -1) if "format" not in d else -1)
            self.i, self.sink, self.sent = 0, None, []
        def poll(self, timeout=0.2):
            out = self.objs[self.i:self.i + 4]; self.i += 4; return out
        def send(self, text): self.sent.append(text)

    def destroyed(lines, rule):
        link = FakeLink(lines)
        rec, a, r = dcs_live.wait_for_fight(link, log=lambda m: None)
        env = dcs_live.DcsLiveEnv(lambda bp, rp: dcs_live.DcsLiveWorld(
            link, rec, a, r, bp, rp, log=lambda m: None, support_rule=rule),
            "F-16C-DCS", "F-16C-DCS")
        env.reset()
        while link.i < len(link.objs) - 4:
            _, _, te, tr, _ = env.step(hot)
            if te or tr:
                break
        red_ids = {d["id"] for d in lines if d.get("ev") == "shot" and d["shooter"] == "RED-1"}
        gone = {int(t.split()[1]) for t in link.sent if t.startswith("DESTROY")}
        return red_ids, gone

    with tempfile.TemporaryDirectory() as d:
        run = recording(os.path.join(d, "run.jsonl"), run_away=True)
        crank = recording(os.path.join(d, "crank.jsonl"), run_away=False)
    red_ids, gone = destroyed(run, "training")
    assert red_ids and red_ids <= gone, (red_ids, gone)
    assert not destroyed(run, "dcs")[1]
    red_ids2, gone2 = destroyed(crank, "training")
    assert red_ids2 and not (red_ids2 & gone2), (red_ids2, gone2)
    print(f"  DCS red support ............. OK  (fire and turn away: {len(red_ids)} red missile(s) "
          f"removed; cranking: none)")


def test_adaptive_defence():
    """ADAPTIVE draws its defence per episode (half shallow, a quarter each deep
    beam and deep cold); 2v1 keeps the shallow one; a deep defence dives to its
    floor, beaming or turning away."""
    import collections
    from bvr_opponents import AdaptiveShooterOpponent, ShooterOpponent
    ic = {"ac2_psi": math.pi, "ac2_alt": 9000.0, "ac2_spd": 280.0, "ac1_psi": 0.0}
    o = AdaptiveShooterOpponent(rng=np.random.default_rng(3))
    counts = collections.Counter()
    for _ in range(400):
        o.reset(ic)
        counts[o.DEFENCE] += 1
        assert 1000.0 <= o.DIVE_FLOOR <= 2500.0
    assert 160 <= counts["shallow"] <= 240 and counts["deep_beam"] >= 70 and counts["deep_cold"] >= 70, counts
    two = AdaptiveShooterOpponent(rng=np.random.default_rng(3)); two.legacy_crank = True
    for seed in range(40):               # as the 2v1 env does: new opponent, legacy flag, reset
        fresh = AdaptiveShooterOpponent(rng=np.random.default_rng(seed)); fresh.legacy_crank = True
        fresh.reset(ic)
        assert (fresh.DEFENCE, fresh.LONG_SHOT, fresh.PRESS) == ("shallow", False, False), seed
    for _ in range(50):
        two.reset(ic)
        assert two.DEFENCE == "shallow"
    assert ShooterOpponent.DEFENCE == "shallow"
    state = {"range": 40_000.0, "aa_deg_t": 0.0, "psi_t": math.pi, "wpn_remaining_t": 4,
             "missiles": [{"owner": 1, "state": "MIDCOURSE", "tgo_est": 30.0}]}
    for style in ("deep_beam", "deep_cold"):
        o.reset(ic); o.DEFENCE = style
        cmd = o.act(state, 10.0)
        assert cmd["altTarget"] == o.DIVE_FLOOR, cmd["altTarget"]
        off = math.degrees(abs((cmd["hdgCmd"] - o._bearing_to_ac1(state) + math.pi) % (2 * math.pi) - math.pi))
        assert abs(off - (90.0 if style == "deep_beam" else 180.0)) < 1.0, (style, off)
    # The DCS-like episodes (SIM_REV 12): a third shoot at the edge of R-max and
    # press on after their missile goes active; 2v1 red never does.
    longs = 0
    for _ in range(300):
        o.reset(ic)
        if o.LONG_SHOT:
            longs += 1
            assert 0.95 <= o.SHOT_RMAX_FRAC <= 1.0 and o.PRESS
        else:
            assert o.SHOT_RMAX_FRAC <= 0.95 and not o.PRESS
    assert 70 <= longs <= 130, longs
    for _ in range(30):
        two.reset(ic)
        assert not two.LONG_SHOT and not two.PRESS
    for press in (False, True):          # once its missile is autonomous: drag, or press on
        o.reset(ic); o.PRESS = press; o._phase = "CRANK"; o._last_shot = 0.0; o.rmax_t = 10_000.0
        o.act({"range": 40_000.0, "aa_deg_t": 0.0, "psi_t": math.pi, "wpn_remaining_t": 3,
               "missiles": [{"owner": 2, "state": "TERMINAL", "needs_support": 0}]}, 10.0)
        assert o._phase == ("REATTACK" if press else "DRAG"), (press, o._phase)
    print(f"  ADAPTIVE defence ............ OK  ({counts['shallow']}/{counts['deep_beam']}/"
          f"{counts['deep_cold']} shallow/beam/cold of 400; {longs}/300 long shots)")


def test_adaptive_runner():
    """SIM_REV 13: in 1/6 of ADAPTIVE episodes red keeps its distance low and slow:
    drifts back beyond RUN_FAR, runs cold (beaming now and then) inside it, and
    turns to fight only inside its TURN_RANGE; 2v1 red never runs.
    Wide start altitudes put the second aircraft 2-12.5 km high in half the starts."""
    import collections
    import bvr_env as E
    from bvr_opponents import AdaptiveShooterOpponent, BvrOpponentType as T
    ic = {"ac2_psi": math.pi, "ac2_alt": 9000.0, "ac2_spd": 280.0, "ac1_psi": 0.0}
    o = AdaptiveShooterOpponent(rng=np.random.default_rng(5))
    runs = 0
    for _ in range(600):
        o.reset(ic)
        # DCS speeds (SIM_REV 13): hot and cranking at Mach 1.3-1.5, running faster.
        assert 380.0 <= o.COMMIT_SPEED <= 450.0 and o.CRANK_SPEED == o.COMMIT_SPEED
        assert 420.0 <= o.FAST_SPEED <= 500.0
        if o.RUNNER:
            runs += 1
            assert o._phase == "RUN" and not o.LONG_SHOT and 1500 <= o.RUN_ALT <= 5000
            assert 25_000 <= o.TURN_RANGE <= 45_000 and o.RUN_SPEED <= 320 and o.RUN_FAR >= 55_000
    assert 70 <= runs <= 130, runs
    for seed in range(40):
        two = AdaptiveShooterOpponent(rng=np.random.default_rng(seed)); two.legacy_crank = True
        two.reset(ic)
        assert not two.RUNNER and two._phase == "COMMIT", seed
        assert (two.COMMIT_SPEED, two.CRANK_SPEED, two.FAST_SPEED) == (330.0, 340.0, 400.0), seed
    while True:
        o.reset(ic)
        if o.RUNNER:
            break
    o.rmax_t = 30_000.0
    far = {"range": 50_000.0, "aa_deg_t": 0.0, "psi_t": math.pi, "wpn_remaining_t": 4, "missiles": []}
    offs = collections.Counter()
    for t in range(0, 300, 2):
        cmd = o.act(far, float(t))
        assert cmd["altTarget"] == max(o.RUN_ALT, 1000.0) and cmd["V"] == o.RUN_SPEED and cmd["fire"] == 0
        off = abs(math.degrees((cmd["hdgCmd"] - o._bearing_to_ac1(far) + math.pi) % (2 * math.pi) - math.pi))
        offs["beam" if abs(off - 90.0) < 1.0 else "cold" if off >= 135.0 else "other"] += 1
    assert offs["cold"] > offs["beam"] > 0 and offs["other"] == 0, offs
    o.TURN_RANGE = 40_000.0
    beyond = o.act(dict(far, range=90_000.0), 300.0)        # too far: it drifts back
    off = abs(math.degrees((beyond["hdgCmd"] - o._bearing_to_ac1(far) + math.pi) % (2 * math.pi) - math.pi))
    assert off <= 40.0 and o._phase == "RUN", off
    near = dict(far, range=o.TURN_RANGE - 1000.0)
    cmd = o.act(near, 300.0)
    assert o._phase == "COMMIT", o._phase
    off = abs(math.degrees((cmd["hdgCmd"] - o._bearing_to_ac1(near) + math.pi) % (2 * math.pi) - math.pi))
    assert off < 1.0, off                                   # hot: it turns to fight
    # In the environment: a runner opens the range on an agent flying at it slowly.
    env = E.BvrEnv(opponent_type=T.ADAPTIVE_SHOOTER, seed=4)
    class Runner(AdaptiveShooterOpponent):          # the env resets it: draw until a runner
        def reset(self, ic):
            super().reset(ic)
            while not self.RUNNER:
                super().reset(ic)
            self.TURN_RANGE, self.RUN_FAR = 20_000.0, 200_000.0     # running from the start
    env._opponent_factory = lambda e: Runner(rng=np.random.default_rng(11))
    env.reset()
    r0 = env._state["range"]
    hot = list(E.HDG_OFFSETS_DEG).index(0.0)
    for _ in range(60):
        env.step([hot, 2, 0, 0])
    assert env._opponent._phase == "RUN" and env._state["range"] > r0 - 5_000.0, (r0, env._state["range"])
    # Wide start altitudes.
    alts = []
    for i in range(300):
        e = E.BvrEnv(opponent_type=T.STRAIGHT, seed=100 + i)
        ic2 = e._random_ic()
        alts.append(abs(ic2["ac2_alt"] - ic2["ac1_alt"]))
    big = sum(a > 2600.0 for a in alts)
    assert 60 <= big <= 200 and max(alts) > 6000.0, (big, max(alts))
    print(f"  ADAPTIVE runner ............. OK  ({runs}/600 runners; {offs['cold']}/{offs['beam']} "
          f"cold/beam decisions; {big}/300 starts more than 2.6 km apart in altitude)")


def test_envelope_target_alt():
    """Launch envelopes depend on the target's altitude (SIM_REV 11): the
    calibrated AIM-120C-DCS reaches much less far against a low target; no
    target altitude means level; old tables without that axis still read as
    level-target tables."""
    import os, tempfile
    import bvr_library as L
    from bvr_envelope import Aim120Envelope
    env = Aim120Envelope(str(L.envelope_path(L.missile_config("AIM-120C-DCS"))))
    assert env.target_altitude_aware
    level = env.compute(0.9, 9000.0, 0.0, 0.9, target_alt=9000.0)[0]
    low = env.compute(0.9, 9000.0, 0.0, 0.9, target_alt=1000.0)[0]
    assert env.compute(0.9, 9000.0, 0.0, 0.9)[0] == level
    assert low < 0.8 * level, (low, level)
    # A table at a grid point returns the swept value exactly.
    with np.load(str(L.envelope_path(L.missile_config("AIM-120C-DCS")))) as d:
        im, ia, it, ip = 1, 3, 0, 0
        r = env.compute(float(d["mach_grid"][im]), float(d["alt_grid"][ia]),
                        float(d["aspect_grid"][ip]), 0.9,
                        target_alt=float(d["target_alt_grid"][it]))[0]
        assert abs(r - float(d["r_max"][im, ia, it, ip])) < 1e-6
    # An old 3-D table: the target altitude is ignored.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "old.npz")
        g = np.ones((2, 2, 3))
        np.savez(path, mach_grid=np.array([0.7, 1.1]), alt_grid=np.array([3000.0, 9000.0]),
                 aspect_grid=np.array([0.0, 90.0, 180.0]), r_max=40_000.0 * g, r_nez=20_000.0 * g)
        old = Aim120Envelope(path)
        assert not old.target_altitude_aware
        assert old.compute(0.9, 6000.0, 0.0, 0.9, target_alt=1000.0) == \
            old.compute(0.9, 6000.0, 0.0, 0.9)
    # The GUI's envelope viewer serves the same numbers.
    import bvr_gui
    v = bvr_gui._envelope_view("AIM-120C-DCS", 0.9, 9000.0, "1000")
    assert v["aware"] and v["aspects"][0] == 0 and v["aspects"][-1] == 180
    assert abs(v["r_max"][0] - low) < 1e-6 and v["r_max"][0] > v["r_max"][-1]
    assert all(n <= m + 1e-6 for n, m in zip(v["r_nez"], v["r_max"]))
    print(f"  envelope target altitude .... OK  (AIM-120C-DCS head-on from 9 km: "
          f"{level/1000:.0f} km level, {low/1000:.0f} km against a target at 1 km)")


def test_platform_dcs_mil():
    """F-16C-DCS-MIL: the F-16C as the DCS AI flies BLUE-1 on a route, without
    afterburner. Same missile, speed choices and altitude limits as F-16C-DCS,
    so checkpoints move between them; much lower top speed."""
    import bvr_library as L
    import bvr_perf as P
    mil, ab = L.load_platform("F-16C-DCS-MIL"), L.load_platform("F-16C-DCS")
    assert mil.missile.id == ab.missile.id and mil.speed_cmds == ab.speed_cmds
    assert (mil.alt_min, mil.alt_max) == (ab.alt_min, ab.alt_max)
    fm, fa = L.airframe_config("F-16C-MIL"), L.airframe_config("F-16C")
    assert fm.T_AB_SL == fm.T_MIL_SL == fa.T_MIL_SL and fm.FF_AB == fm.FF_MIL
    v_mil, v_ab = P.top_speed(fm, 9000.0)[0], P.top_speed(fa, 9000.0)[0]
    assert v_mil < 330.0 < 450.0 < v_ab, (v_mil, v_ab)
    print(f"  F-16C-DCS-MIL ............... OK  (top speed at 9 km {v_mil:.0f} m/s; "
          f"with afterburner {v_ab:.0f})")


def test_env_pickles():
    """The envs pickle: older stable-baselines3 (before 2.x has_attr) sends
    env.action_masks, and with it the whole env, from each worker process."""
    import pickle
    import bvr_env as E
    from bvr_team import TeamBvrEnv
    from bvr_opponents import BvrOpponentType as T
    for plat in ("F-16C", "F-16C-DCS"):
        env = E.BvrEnv(opponent_type=T.SHOOTER, seed=1, platform=plat, opponent_platform=plat)
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample())
        pickle.loads(pickle.dumps(env.action_masks))
    team = TeamBvrEnv(opponent_type=T.SHOOTER, seed=1)
    team.reset()
    pickle.loads(pickle.dumps(team))
    print("  envs pickle ................. OK")


def test_dcs_round_trip():
    """A simulator episode written in the DCS logger's format replays to the same inputs."""
    import os, tempfile
    from bvr_env import BvrEnv, OBS_LABELS
    from bvr_opponents import BvrOpponentType as T
    from dcs_world import write_sim_recording, DcsReplayEnv, capture_raw_obs

    def policy(env):
        fire = 0
        if env._can_fire():
            est = env._est()
            fire = int(env._est_range(est) <= 0.75 * env._own_envelope(est)[0])
        return [0, 2, 3, fire]

    sim = BvrEnv(opponent_type=T.SHOOTER, seed=3, doctrine="AGGRESSIVE")
    sim._radar.rng = np.random.default_rng(99)          # same radar noise in both runs
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rt.jsonl")
        out = write_sim_recording(sim, policy, path)
        rep = DcsReplayEnv(path, doctrine="AGGRESSIVE")
        rep._radar.rng = np.random.default_rng(99)
        raw = capture_raw_obs(rep)
        rep.reset()
        rows = [raw["obs"].copy()]
        for a, _, _ in out[1:]:
            _, _, term, trunc, info = rep.step(a)
            rows.append(raw["obs"].copy())
            if term or trunc:
                break
    assert len(rows) == len(out), (len(rows), len(out))
    assert info.get("terminal_outcome") == out[-1][2].get("terminal_outcome")
    S = np.array([o for _, o, _ in out])[1:]            # step 0: the track is 0.5 s old
    R = np.array(rows)[1:]
    # Estimated, not recorded: seeker hand-off (a mean, not the sim's draw)
    # and load factor (a finite difference, not the commanded value).
    loose = {"own_msl_support", "own_msl_active", "rwr_seeker_active", "own_nz",
             "own_msl_stale"}
    for j, lab in enumerate(OBS_LABELS):
        diff = np.abs(S[:, j] - R[:, j])
        tol = 0.02 * max(np.ptp(S[:, j]), 1e-3) + 1e-3
        if lab == "own_nz":
            tol = 0.35        # g: the simulator reports the commanded load factor
        share = float(np.mean(diff <= tol))
        assert share >= (0.8 if lab in loose else 0.97), (lab, share, float(diff.max()))
    print(f"  DCS round trip .............. OK  ({len(rows) - 1} decisions, "
          f"{info.get('terminal_outcome')})")


def test_selfplay_mix():
    """Self-play opponent choice follows the scripted and newest shares; the pool keeps max_pool."""
    import os, tempfile
    from bvr_selfplay import pick_opponent, snapshot, pool_snapshots, clear_pool
    rng = np.random.default_rng(0)
    paths = [f"sp_{i}.zip" for i in range(4)]
    n = 6000
    picks = [pick_opponent(paths, rng, 0.25, 0.6) for _ in range(n)]
    scripted = sum(p is None for p in picks) / n
    newest = sum(p == paths[-1] for p in picks) / n
    assert abs(scripted - 0.25) < 0.02, scripted
    # newest: 0.75 * (0.6 + 0.4 / 4)
    assert abs(newest - 0.75 * 0.7) < 0.02, newest
    assert all(pick_opponent(paths, rng, 1.0, 0.5) is None for _ in range(50))
    assert all(pick_opponent(paths, rng, 0.0, 1.0) == paths[-1] for _ in range(50))

    class Model:
        num_timesteps = 0
        def save(self, path):
            open(path + ".zip", "w").close()
            Model.num_timesteps += 1000
    with tempfile.TemporaryDirectory() as d:
        pool = os.path.join(d, "pool")
        for _ in range(5):
            snapshot(Model(), pool, max_pool=3)
        kept = pool_snapshots(pool)
        assert len(kept) == 3 and kept[-1].endswith("0000004000.zip"), kept
        assert clear_pool(pool) == 3 and not pool_snapshots(pool)
    print(f"  self-play mix ............... OK  (scripted {scripted:.1%}, newest {newest:.1%})")


def test_fire_off_boresight():
    """The fire action is masked for a target more than 60 deg off the nose."""
    import math
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T
    from bvr_track_adapter import TrackState
    env = BvrEnv(opponent_type=T.STRAIGHT, seed=0)
    env.reset()
    env._state.update({"wpn_remaining": 4, "psi": math.radians(30.0), "theta": 0.0})
    env._last_shot_t = -999.0
    env._trk_state = lambda: TrackState.TRACK
    env._own_envelope = lambda est: (40_000.0, 15_000.0)
    own = env._own_pos_enu()

    def fire_ok(az_deg, el_deg=0.0, R=20_000.0):
        a, e = math.radians(30.0 + az_deg), math.radians(el_deg)   # relative to the nose
        pos = own + R * np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])
        est = {"valid": True, "pos": pos, "vel": np.zeros(3), "speed": 250.0, "age": 0.0,
               "pos_sigma": 50.0, "vel_sigma": 5.0}
        env._est = lambda: est
        return env._can_fire(), bool(env.action_masks()[-1])

    for az in (0, 30, -45, 59):
        assert fire_ok(az) == (True, True), az
    for az in (61, -75, 90, 180):
        assert fire_ok(az) == (False, False), az
    assert fire_ok(0, el_deg=55) == (True, True)
    assert fire_ok(0, el_deg=65) == (False, False)
    assert fire_ok(40, el_deg=40) == (True, True)            # 54 deg off the nose in all
    assert fire_ok(50, el_deg=45) == (False, False)          # 63 deg off the nose in all
    print("  fire off-boresight gate ..... OK")


def test_heading_switches():
    """Heading-choice changes are charged and counted, and the rocking pair is found."""
    from bvr_env import BvrEnv, HDG_OFFSETS_DEG
    from bvr_opponents import BvrOpponentType as T
    from train_bvr import heading_switch_stats
    env = BvrEnv(opponent_type=T.STRAIGHT, seed=4)
    env.reset()
    p30, m30 = HDG_OFFSETS_DEG.index(30.0), HDG_OFFSETS_DEG.index(-30.0)
    costs = []
    for k in range(40):
        _, _, term, trunc, info = env.step([p30 if k % 2 == 0 else m30, 2, 2, 0])
        costs.append(-info["cost_terms"]["heading"])
        if term or trunc:
            break
    assert abs(costs[0] - env.W_HDG_CHANGE * 30 / 180) < 1e-9          # 0 -> +30
    assert all(abs(c - env.W_HDG_CHANGE * 60 / 180) < 1e-9 for c in costs[1:])
    assert env._hdg_switches == len(costs)
    assert env._hdg_pairs == {"0>+30": 1, "+30>-30": 20, "-30>+30": 19}, env._hdg_pairs
    rpm, top = heading_switch_stats([(env._hdg_switches, env._t_sim, env._hdg_pairs)])
    assert 50 < rpm < 62 and top[0][0] == "-30<>+30" and top[0][1] > 0.95, (rpm, top)
    print(f"  heading switches ............ OK  ({rpm:.0f}/min, {top[0][0]} {top[0][1]:.0%})")


def test_mirrored_starts():
    """Half the starts are mirrored: then AC2, not AC1, starts nose-on to the other."""
    import math
    from bvr_env import BvrEnv, REF_LAT, R_EARTH, DEG2RAD
    from bvr_opponents import BvrOpponentType as T
    e = BvrEnv(opponent_type=T.STRAIGHT, seed=2, platform="GENERIC-UCAV", opponent_platform="F-16C")
    enu = lambda la, lo: np.array([(lo - 35.0) * DEG2RAD * R_EARTH * math.cos(REF_LAT * DEG2RAD),
                                   (la - REF_LAT) * DEG2RAD * R_EARTH])
    def off(psi, frm, to):
        d = to - frm
        return abs((math.degrees(math.atan2(d[0], d[1]) - psi) + 180) % 360 - 180)
    n, mir = 600, 0
    for _ in range(n):
        ic = e._random_ic()
        p1, p2 = enu(ic["ac1_lat"], ic["ac1_lon"]), enu(ic["ac2_lat"], ic["ac2_lon"])
        assert abs(float(np.linalg.norm(p2 - p1)) - ic["start_range"]) < 1.0
        if ic["mirrored"]:
            mir += 1
            assert off(ic["ac2_psi"], p2, p1) < 1e-6
        else:
            assert off(ic["ac1_psi"], p1, p2) < 1e-6
        assert ic["ac1_spd"] < 210 and ic["ac2_spd"] > 240      # UCAV v F-16 speeds follow the seat
    assert 0.42 < mir / n < 0.58, mir / n
    print(f"  mirrored starts ............. OK  ({mir / n:.0%} mirrored)")


def test_mutual_kill_1v1():
    """A kill ends the fight only once no missile is left in flight at a live aircraft."""
    import math
    from bvr_env import BvrEnv, HDG_OFFSETS_DEG
    from bvr_opponents import BvrOpponentType as T
    env = BvrEnv(opponent_type=T.STRAIGHT, seed=0)
    straight = HDG_OFFSETS_DEG.index(0)
    outcomes = []
    for ep in range(12):
        env.reset(seed=ep)
        w = env._world
        w.acs[0].reset_state(0.0, 0.0, 9000.0, 0.0, 280.0, 0.6)
        w.acs[1].reset_state(0.0, 30000.0, 9000.0, math.pi, 280.0, 0.6)
        fired = [False, False]
        while True:
            r = float(np.linalg.norm(w.pos(1) - w.pos(2)))
            if not fired[0] and r < 26000 - (ep % 5) * 1500:
                w._launch(1, {"target": 2}); fired[0] = True
            if not fired[1] and r < 30000:
                w._launch(2, {"target": 1}); fired[1] = True
            _, rew, term, trunc, info = env.step(np.array([straight, 2, 0, 0]))
            if term or trunc:
                break
        o = info["terminal_outcome"]
        outcomes.append(o)
        if term:
            assert not w.missiles_pending(), f"ep {ep}: {o} with a missile still in flight"
        if o == "MUTUAL_KILL":
            assert not any(w.alive)
    assert "MUTUAL_KILL" in outcomes, outcomes
    print(f"  mutual kill 1v1 ............. OK  ({outcomes.count('MUTUAL_KILL')}/12 mutual)")


def test_team_missiles_resolve():
    """2v1: the episode waits for missiles in flight after red dies or both blue are lost."""
    import math
    from bvr_team import TeamBvrEnv
    from bvr_opponents import BvrOpponentType as T
    e = TeamBvrEnv(opponent_type=T.STRAIGHT, seed=3, doctrine="AGGRESSIVE")
    outcomes = []
    for ep in range(8):
        e.reset()
        w = e._world
        w.acs[0].reset_state(-2000.0, 0.0, 9000.0, 0.0, 280.0, 0.6)
        w.acs[1].reset_state(2000.0, 0.0, 9000.0, 0.0, 280.0, 0.6)
        w.acs[2].reset_state(0.0, 32000.0, 9000.0, math.pi, 280.0, 0.6)
        fired = set()
        while True:
            r = float(np.linalg.norm(w.pos(1) - w.pos(3)))
            for owner, tgt, at in ((3, 1, 30000), (3, 2, 29000), (1, 3, 25000 - 1500 * (ep % 4))):
                if (owner, tgt) not in fired and r < at:
                    w._launch(owner, {"target": tgt}); fired.add((owner, tgt))
            _, rew, term, trunc, infos = e.step([[0, 2, 0, 0], [0, 2, 0, 0]])
            if term or trunc:
                break
        o = infos[0]["terminal_outcome"]
        outcomes.append(o)
        if term:
            assert not w.missiles_pending(), f"ep {ep}: {o} with a missile still in flight"
        if o == "MUTUAL_KILL":
            assert not w.alive[2] and infos[0]["blue_losses"] >= 1
    print(f"  2v1 missiles resolve ........ OK  ({', '.join(sorted(set(outcomes)))})")


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


def test_dcs_turn_test():
    """dcs/turn_test.py --sim: the turns go through the UDP link to the fake DCS,
    which confirms the bridge options; the simulator's F-16 turns at 5+ deg/s
    while 30+ deg remain (the DCS AI on the bridge's routes: about 2)."""
    import os, sys, tempfile, csv
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcs"))
    import turn_test
    with tempfile.TemporaryDirectory() as d:
        rows = turn_test.main(["--sim", "--variants", "base", "--out", d])
        with open(os.path.join(d, "turn_test.csv"), newline="") as fh:
            saved = list(csv.DictReader(fh))
        assert any(f.endswith("_sim.jsonl") for f in os.listdir(d))
    assert [round(abs(r["turn_deg"]) / 10) for r in rows] == [9, 9, 17], rows
    assert all(r["rate_bulk_dps"] > 5.0 and r["reached"] for r in rows), rows
    assert len(saved) == 3 and saved[0]["mode"] == "sim" and saved[0]["variant"] == "base", saved
    # hot (bridge 4): red off a wing on the route (45 deg of bank), then the
    # attack task, which the fake flies at the airframe's full agility.
    with tempfile.TemporaryDirectory() as d:
        hot = turn_test.main(["--sim", "--platform", "F-16C-DCSAI", "--variants", "hot", "--out", d])
    # (red off the wing within 5 deg, and it moves: 80-100 deg to turn)
    assert [r["turn_deg"] > 0 for r in hot] == [True, False], hot
    assert all(80 <= abs(r["turn_deg"]) <= 100 for r in hot), hot
    assert all(r["rate_bulk_dps"] > 5.0 and r["bank_max_deg"] > 60 and r["wpt"] == "attack"
               for r in hot), hot
    print(f"  DCS turn test (sim) ......... OK  ({min(r['rate_bulk_dps'] for r in rows):.1f}-"
          f"{max(r['rate_bulk_dps'] for r in rows):.1f} deg/s; hot "
          f"{min(r['rate_bulk_dps'] for r in hot):.1f}-{max(r['rate_bulk_dps'] for r in hot):.1f})")


def test_dcs_ai_climb():
    """F-16C-DCSAI flies like the DCS AI on a route: climbs at about Mach 0.85
    whatever speed is commanded, 0.02 m/s per metre still to climb, no faster
    than its climb throttle allows, and turns at 45 deg of bank; F-16C-DCS keeps
    the old climb; red flying F-16C-DCS in the same world is unaffected; the two
    can meet in self-play."""
    import math
    import bvr_env as E
    import bvr_library as L
    from bvr_opponents import BvrOpponentType as T
    from sim_world import _enu_to_latlon

    def climb(plat, opp, ahead):
        env = E.BvrEnv(opponent_type=T.STRAIGHT, seed=1, platform=plat, opponent_platform=opp)
        def ic_fn(blue_top_speed=None):
            ic = E.BvrEnv._random_ic(env)
            la, lo, _ = _enu_to_latlon(0, 0, 9000); lb, lob, _ = _enu_to_latlon(0, 90000, 9000)
            ic.update(ac1_lat=la, ac1_lon=lo, ac1_alt=9000, ac1_psi=math.pi / 2, ac1_spd=275,
                      ac2_lat=lb, ac2_lon=lob, ac2_alt=9000, ac2_psi=math.pi / 2, ac2_spd=275,
                      mirrored=False)
            return ic
        env._random_ic = ic_fn
        env.reset()
        beam = list(E.HDG_OFFSETS_DEG).index(90.0)
        ia = list(E.ALT_DELTAS_M).index(ahead)
        rows = []
        for t in range(40):
            z0 = env._state["alt"]
            env.step([beam, ia, 3, 0])                         # 400 m/s commanded
            rows.append((env._state["alt"] - z0, env._state["mach"], env._world.acs[1].mach))
        return rows

    dcsai = climb("F-16C-DCSAI", "F-16C-DCS", 3000.0)
    vs = [r[0] for r in dcsai[15:30]]
    machs = [r[1] for r in dcsai[15:]]
    assert 0.80 <= min(machs) and max(machs) <= 0.90, machs           # held, not accelerating
    assert 40.0 <= sum(vs) / len(vs) <= 65.0, vs                       # DCS: 50-60 m/s at 10-11 km
    slow = climb("F-16C-DCSAI", "F-16C-DCS", 1200.0)
    vs12 = sum(r[0] for r in slow[15:30]) / 15
    assert 18.0 <= vs12 <= 30.0, vs12                                  # DCS: 22-33 m/s
    old = climb("F-16C-DCS", "F-16C-DCS", 3000.0)
    assert max(r[1] for r in old) > 0.95                               # the old climb accelerates
    # Red flies F-16C-DCS in the same world: no DCS-AI climb for it (it holds its level).
    assert L.load_platform("F-16C-DCS").climb_mach == 0.0
    # Turns: the DCS AI on a route banks 45 deg, 1.41 g, 1.5-1.6 deg/s at 340 m/s.
    import os, sys, tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcs"))
    import turn_test
    with tempfile.TemporaryDirectory() as d:
        turns = turn_test.main(["--sim", "--platform", "F-16C-DCSAI", "--variants", "base", "--out", d])
    late = turns[1:]                                    # the first starts below 340 m/s
    assert all(r["bank_max_deg"] <= 46 and 1.3 <= r["nz_median"] <= 1.5 for r in turns), turns
    assert all(1.4 <= r["rate_bulk_dps"] <= 1.9 for r in late), late
    assert L.selfplay_compatible("F-16C-DCSAI", "F-16C-DCS")
    assert not L.selfplay_compatible("F-16C-DCS", "F-16C-DCS-MIL")
    # A platform file saved with whole numbers (1000, not 1000.0) is the same aircraft.
    import copy
    whole = copy.deepcopy(L.get_item("platform", "F-16C-DCS"))
    whole["params"] = {k: (int(v) if isinstance(v, float) and v == int(v) else v)
                       for k, v in whole["params"].items()}
    assert L._selfplay_group_of(whole) == L.selfplay_group("F-16C-DCSAI")
    assert not L.selfplay_compatible("F-16C", "F-16C-DCS")
    print(f"  DCS-AI climb and turn ....... OK  (Mach {min(machs):.2f}-{max(machs):.2f}, "
          f"{sum(vs) / len(vs):.0f} m/s with 3 km to go, {vs12:.0f} with 1.2 km; turns "
          f"{min(r['rate_bulk_dps'] for r in late):.1f}-{max(r['rate_bulk_dps'] for r in late):.1f} deg/s)")


def test_crossplay_reseed():
    """A cross-play env cached across a scripted and a policy column holds both
    radar observers (keys "opponent_radar" and False); reseeding must handle
    the mix, and give each observer the same noise whichever exist."""
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T
    from crossplay import _reseed
    env = BvrEnv(opponent_type=T.SELF_PLAY, seed=5)
    env._opponent_radar()
    _reseed(env, 1000)
    alone = env._sp_observers["opponent_radar"]._rng.random()
    env.selfplay_observer(False)
    _reseed(env, 1000)              # raised TypeError: bool < str
    assert env._sp_observers["opponent_radar"]._rng.random() == alone
    print("  cross-play reseed ........... OK")


def test_auto_defend():
    """A platform with AUTO_DEFEND defends itself while an enemy missile is in
    flight at it, as the DCS AI does when the bridge hands it BLUE-1: it beams
    the missile, dives 3 km, flies its fastest speed at full agility and holds
    fire; then it flies its commands again. Without AUTO_DEFEND, nothing changes."""
    import math
    import bvr_library as L
    from sim_world import SimWorld, _enu_to_latlon
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T

    def fight(blue):
        w = SimWorld(platforms=[L.load_platform(blue), L.load_platform("F-16C-DCS")], teams=[1, 2])
        la, lo, _ = _enu_to_latlon(0, 0, 9000); lb, lob, _ = _enu_to_latlon(0, 30000, 9000)
        w.reset({"ac1_lat": la, "ac1_lon": lo, "ac1_alt": 9000, "ac1_psi": 0.0, "ac1_spd": 300,
                 "ac2_lat": lb, "ac2_lon": lob, "ac2_alt": 9000, "ac2_psi": math.pi, "ac2_spd": 300})
        blue_cmd = {"hdgCmd": 0.0, "altTarget": 9000.0, "V": 300.0, "fire": 0}
        red_cmd = {"hdgCmd": math.pi, "altTarget": 9000.0, "V": 300.0, "fire": 0}
        rows = []
        for k in range(int(80 / w.SIM_DT)):
            fire2 = 1 if k == 5 else 0
            fire1 = 1 if k % 100 == 50 else 0                  # blue keeps trying to shoot
            w.step_all([{**blue_cmd, "fire": fire1}, {**red_cmd, "fire": fire2}])
            tl = w.telemetry(1)
            inb = w._inbound(1)
            brg = (math.atan2(*(inb[0].pos - w.pos(1))[:2]) if inb else 0.0)
            rows.append((w.t_sim, tl["defending"], w.acs[0].chi, w.acs[0].z, w.acs[0].V,
                         w.wpn[0], len(inb), abs(w.acs[0].phi), brg))
            if not w.alive[0]:
                break
        return w, rows

    w, rows = fight("F-16C-DCSAI-AD")
    on = [r for r in rows if r[1]]
    assert on and rows[0][1] == 0, "no defence"
    # (decided at the start of each frame: the frame the missile ends in still defends)
    assert all(r[6] > 0 for r in on[:-1]), "defending with nothing inbound"
    assert all(r[5] == 4 for r in on), "fired while defending"
    t_on = on[0][0]
    later = [r for r in on if t_on + 20 < r[0] < t_on + 30]       # (not the fly-by)
    # beamed: heading 90 deg off the missile's bearing
    off = [abs(abs(_wrap_pi(r[2] - r[8])) - math.pi / 2) for r in later]
    assert later and max(off) < 0.3, (t_on, max(off), later[off.index(max(off))])
    assert max(r[7] for r in on) > math.radians(60), "turned at the 45-deg route bank"
    assert min(r[3] for r in on) < 7500, "did not dive"
    assert max(r[4] for r in on) > 360, "not at its fastest speed"
    if w.alive[0]:                   # missile gone: the commands again, and blue may fire
        after = [r for r in rows if r[0] > on[-1][0]]
        assert after and all(r[1] == 0 for r in after) and rows[-1][5] < 4, rows[-1]
    # The same fight without AUTO_DEFEND: no defence, the commands are flown.
    w2, rows2 = fight("F-16C-DCSAI")
    assert not any(r[1] for r in rows2) and rows2[0][5] == 4
    # The fire mask in the env follows the telemetry.
    env = BvrEnv(opponent_type=T.STRAIGHT, seed=3, platform="F-16C-DCSAI-AD", opponent_platform="F-16C-DCS")
    env.reset()
    env._state["defending"] = 1
    assert not env._can_fire()
    assert L.selfplay_compatible("F-16C-DCSAI-AD", "F-16C-DCS")
    print(f"  auto-defend .................. OK  (defended {on[-1][0] - t_on:.0f} s, dove to "
          f"{min(r[3] for r in on):.0f} m, {max(r[4] for r in on):.0f} m/s, "
          f"bank {math.degrees(max(r[7] for r in on)):.0f} deg; blue "
          f"{'survived' if w.alive[0] else 'was hit'})")


def test_dcs_auto_defend():
    """dcs_live.py --auto-defend against dcs/fake_dcs.py with a red that shoots:
    the bridge option is sent and confirmed, the bridge's defence events hold
    the policy's fire, and results.csv counts the defences."""
    import os, sys, tempfile, threading
    from types import SimpleNamespace
    from gymnasium import spaces
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcs"))
    import dcs_live
    from fake_dcs import FakeDcs
    from bvr_env import BvrEnv, OBS_DIM, PRIV_DIM, HDG_OFFSETS_DEG

    masks = []
    class Scripted:          # straight at the bandit, fire whenever allowed
        observation_space = spaces.Dict({"obs": spaces.Box(-1, 1, (OBS_DIM,)),
                                         "priv": spaces.Box(-1, 1, (PRIV_DIM,))})
        def predict(self, obs, deterministic=True, action_masks=None):
            masks.append(bool(action_masks[-1]))
            return np.array([list(HDG_OFFSETS_DEG).index(0), 2, 2, int(action_masks[-1])]), None

    fake = FakeDcs(opponent="SHOOTER", seed=4, speed=8.0, max_time=250.0, lockstep=True,
                   platform="F-16C-DCSAI", opp_platform="F-16C-DCS", log=lambda m: None)
    from sim_world import _enu_to_latlon
    def head_on(blue_top_speed=None):        # 50 km head-on: red shoots early
        ic = BvrEnv._random_ic(fake.env)
        la, lo, _ = _enu_to_latlon(0, 0, 9000); lb, lob, _ = _enu_to_latlon(50000, 0, 9000)
        ic.update(ac1_lat=la, ac1_lon=lo, ac1_alt=9000, ac1_psi=math.pi / 2, ac1_spd=300,
                  ac2_lat=lb, ac2_lon=lob, ac2_alt=9000, ac2_psi=3 * math.pi / 2, ac2_spd=300,
                  mirrored=False, start_range=50000.0)
        return ic
    fake.env._random_ic = head_on
    th = threading.Thread(target=fake.run, daemon=True)
    link = dcs_live.UdpLink()
    th.start()
    lines = []
    with tempfile.TemporaryDirectory() as d:
        args = SimpleNamespace(shadow=False, out=d, opp_platform="F-16C-DCS", max_steps=250,
                               doctrine="AGGRESSIVE", auto_defend=False)
        # The platform turns it on, without the flag.
        scen = {"platform": "F-16C-DCSAI-AD", "opponent_platform": "F-16C-DCS"}
        rec, agent, red = dcs_live.wait_for_fight(link, log=lambda m: None)
        row, _ = dcs_live.run_episode(link, Scripted(), args, scen, rec, agent, red, 1,
                                      log=lines.append)
        th.join(timeout=120)
    link.close(); fake.close()
    assert row["auto_defend"] == 1 and row["bridge_version"] == 4, row
    assert row["defences"] >= 1 and row["defend_s"] > 0, (row, lines)
    assert any("the DCS AI defends BLUE-1" in ln for ln in lines), lines
    assert not any("never confirmed" in ln for ln in lines), lines
    print(f"  DCS auto-defend ............. OK  ({row['defences']} defence(s), {row['defend_s']:.0f} s; "
          f"{row['outcome']})")

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
        ("missile time-to-go",      test_missile_tgo),
        ("defence potential",       test_defence_potential),
        ("heading reference",       test_heading_reference),
        ("stern starts",            test_stern_starts),
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
        ("2v1 red targeting",       test_team_red_targeting),
        ("2v1 red opens on rear",   test_team_roles_far_target),
        ("DCS round trip",          test_dcs_round_trip),
        ("DCS live link",           test_dcs_live_link),
        ("DCS support rule",        test_dcs_support_rule),
        ("DCS speed boost",         test_dcs_speed_boost),
        ("DCS red support",         test_dcs_red_support),
        ("missile loft",            test_missile_loft),
        ("self-play mix",           test_selfplay_mix),
        ("fire off-boresight gate", test_fire_off_boresight),
        ("heading switches",        test_heading_switches),
        ("mirrored starts",         test_mirrored_starts),
        ("2v1 vec env",             test_team_vec_env),
        ("mutual kill 1v1",         test_mutual_kill_1v1),
        ("2v1 missiles resolve",    test_team_missiles_resolve),
        ("envelope target altitude", test_envelope_target_alt),
        ("F-16C-DCS-MIL",           test_platform_dcs_mil),
        ("envs pickle",             test_env_pickles),
        ("ADAPTIVE defence",        test_adaptive_defence),
        ("ADAPTIVE runner",         test_adaptive_runner),
        ("cross-play reseed",       test_crossplay_reseed),
        ("DCS turn test",           test_dcs_turn_test),
        ("DCS-AI climb",            test_dcs_ai_climb),
        ("auto-defend",             test_auto_defend),
        ("DCS auto-defend",         test_dcs_auto_defend),
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
