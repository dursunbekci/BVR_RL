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
    """A checkpoint from before the wingman inputs and the SIM_REV 14 actions
    (10 headings, 4 speeds) loads and acts as before: the same values, the same
    logits for every old choice, and the new choices a quarter as likely as
    their neighbours (so never the most likely)."""
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
        action_space = spaces.MultiDiscrete([10, 5, 4, 2])
        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}
        def step(self, a):
            return self.observation_space.sample(), 0.0, False, False, {}
        def action_masks(self):
            return np.ones(21, dtype=bool)

    vec = VecFrameStack(DummyVecEnv([OldLayout]), N_STACK)
    old = MaskablePPO(AsymmetricMaskablePolicy, vec, n_steps=32, batch_size=32,
                      policy_kwargs=dict(pi_arch=(64, 64), vf_arch=(64, 64)))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.zip")
        old.save(path)
        new = load_model(path, log=None)
    assert new.observation_space["obs"].shape == (OBS_DIM * N_STACK,)
    assert new.observation_space["priv"].shape == (PRIV_DIM * N_STACK,)
    assert list(new.action_space.nvec) == list(ACTION_NVEC)
    with th.no_grad():         # a non-trivial head, so that the averages are tested
        old.policy.action_net.bias.copy_(th.linspace(-1, 1, 21))
        old.policy.action_net.weight.mul_(20.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.zip")
        old.save(path)
        new = load_model(path, log=None)

    rng = np.random.default_rng(0)
    o = rng.uniform(-1, 1, (5, N_STACK, OBS_DIM_1V1)).astype(np.float32)
    p = rng.uniform(-1, 1, (5, N_STACK, PRIV_DIM_1V1)).astype(np.float32)
    # the new columns hold arbitrary values: they must change nothing
    o_new = np.concatenate([o, rng.uniform(-1, 1, (5, N_STACK, OBS_DIM - OBS_DIM_1V1))], 2)
    p_new = np.concatenate([p, rng.uniform(-1, 1, (5, N_STACK, PRIV_DIM - PRIV_DIM_1V1))], 2)

    def run(model, oo, pp):
        x = {"obs": th.as_tensor(oo.reshape(5, -1), dtype=th.float32),
             "priv": th.as_tensor(pp.reshape(5, -1), dtype=th.float32)}
        with th.no_grad():     # the head's raw outputs (each choice's own logit)
            pol = model.policy
            latent = pol.mlp_extractor.forward_actor(pol.extract_features(x, pol.pi_features_extractor))
            return pol.action_net(latent).numpy(), pol.predict_values(x).numpy()

    from bvr_compat import HDG_LAYOUTS, NEW_CHOICE_LOGIT
    from bvr_env import HDG_OFFSETS_DEG
    la, va = run(old, o, p)
    lb, vb = run(new, o_new, p_new)
    assert np.allclose(va, vb, atol=1e-5), np.abs(va - vb).max()
    old_h = HDG_LAYOUTS[10]
    # new column of each old choice: headings by value, then alt 5, speed 4 (+1), fire 2
    col = [HDG_OFFSETS_DEG.index(v) for v in old_h] + [14 + i for i in range(5)] \
        + [19 + i for i in range(4)] + [24, 25]
    assert np.allclose(la, lb[:, col], atol=1e-5), np.abs(la - lb[:, col]).max()
    # the new choices: between their neighbours, a quarter as likely, never the top
    for v, nb in ((15.0, (0.0, 30.0)), (-40.0, (-50.0, -30.0))):
        k = HDG_OFFSETS_DEG.index(v)
        mid = (la[:, old_h.index(nb[0])] + la[:, old_h.index(nb[1])]) / 2
        assert np.allclose(lb[:, k], mid + NEW_CHOICE_LOGIT, atol=1e-4)
    assert np.allclose(lb[:, 23], la[:, 18] + NEW_CHOICE_LOGIT, atol=1e-5)    # fifth speed
    heads = np.argmax(la[:, :10], 1)
    assert [HDG_OFFSETS_DEG[i] for i in np.argmax(lb[:, :14], 1)] == [old_h[i] for i in heads]
    assert np.array_equal(np.argmax(lb[:, 19:24], 1), np.argmax(la[:, 15:19], 1))
    # Speeds by value, from the checkpoint's own platform: a speed added below
    # (GENERIC-UCAV's 150) takes the nearest old one's output, a quarter as likely.
    from bvr_compat import _speed_rows
    assert _speed_rows([220, 280, 340, 400], [220, 280, 340, 400, 460]) == \
        [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (3, NEW_CHOICE_LOGIT)]
    assert _speed_rows([170, 200, 225, 250], [150, 170, 200, 225, 250]) == \
        [(0, NEW_CHOICE_LOGIT), (0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    print("  checkpoint widening ......... OK  (inputs, and actions 10/4 -> 14/5)")


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


def test_dcs_live_stick():
    """dcs_live.py --stick: the policy's commands go to a StickPilot flying the player's aircraft through the
    Export link (dcs/stick/fake_export.py here), and FIRE runs the cockpit job (A-A, TWS, bug, master arm,
    weapon release held) whose launch comes back into the fire bookkeeping. The bridge (dcs/fake_dcs.py) gets
    no CMD line, so it never takes BLUE-1."""
    import os, sys, tempfile, threading, time as _time
    from types import SimpleNamespace
    from gymnasium import spaces
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "dcs"))
    sys.path.insert(0, os.path.join(here, "dcs", "stick"))
    import dcs_live
    import cockpit
    from fake_dcs import FakeDcs
    from fake_export import FakeLink
    from bvr_env import OBS_DIM, PRIV_DIM, HDG_OFFSETS_DEG

    class Scripted:          # 30 deg right, top speed, fire whenever allowed
        observation_space = spaces.Dict({"obs": spaces.Box(-1, 1, (OBS_DIM,)),
                                         "priv": spaces.Box(-1, 1, (PRIV_DIM,))})
        def predict(self, obs, deterministic=True, action_masks=None):
            return np.array([list(HDG_OFFSETS_DEG).index(0), 2, 3, int(action_masks[-1])]), None

    class SlowLink(FakeLink):
        """About ten times real time, so the pilot thread leaves the policy some CPU; the fake cockpit's target
        is kept 40 km ahead of the aircraft (the aircraft flies headings from the fake bridge's fight, where the
        cockpit's own target is not)."""
        def next(self, timeout=1.0):
            _time.sleep(0.002)
            c, ac = self.cockpit, self.ac
            if c.missile is None:
                c.tx, c.tz = ac.x + 40000.0 * math.cos(ac.hdg), ac.z + 40000.0 * math.sin(ac.hdg)
            return super().next(timeout)
    defs = cockpit.load_defs()
    stick = SlowLink(cockpit_defs=defs, scan_needed=999)
    stick.cockpit.icp_aa_works = False      # as in DCS on 15 October: A-A has to come from missile override
    gains = {"roll_sign": 1, "pitch_sign": 1, "thr_sign": 1, "thr_u_trim": 0.1,
             "roll_curve": [[0.5, 1.3], [1.0, 2.6]], "pitch_curve": [[0.5, 3.0], [1.0, 6.0]]}
    fake = FakeDcs(opponent="STRAIGHT", seed=2, speed=8.0, max_time=200.0, lockstep=True)
    th = threading.Thread(target=fake.run, daemon=True)
    link = dcs_live.UdpLink()
    th.start()
    logs = []
    try:
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(shadow=False, out=d, opp_platform=None, max_steps=200, doctrine="AGGRESSIVE",
                                   stick_link=stick, stick_gains=gains, point_before_fire=20.0)
            # The fake bridge's BLUE-1 does not fly the policy's headings (the fake cockpit's aircraft does), so
            # its nose is not on the target: the old 60 deg fire cone lets this test reach the cockpit's fire job.
            from bvr_env import BvrEnv as _B
            _gate = (_B.MAX_OFF_BORESIGHT_DEG, _B.OFFAXIS_IN_MASK)
            _B.MAX_OFF_BORESIGHT_DEG, _B.OFFAXIS_IN_MASK = 60.0, False
            rec, agent, red = dcs_live.wait_for_fight(link, log=lambda m: None)
            try:
                row, _ = dcs_live.run_episode(link, Scripted(), args, {"platform": "F-16C", "opponent_platform": "F-16C"},
                                              rec, agent, red, 1, log=logs.append)
            finally:
                _B.MAX_OFF_BORESIGHT_DEG, _B.OFFAXIS_IN_MASK = _gate
            import json as _json
            recs = [_json.loads(x) for x in open(os.path.join(d, "episodes.jsonl"))]
            # (the fake cockpit's missile never reaches the fake DCS world, so the record has no missiles here)
            assert len(recs) == 1 and recs[0]["red"]["style"] == "DCS AI" and recs[0]["phases"], recs
            # the console kept beside the fight's files, with the pilot's events in it
            con = [f for f in os.listdir(d) if f.endswith("_console.txt")]
            assert len(con) == 1 and "A-A master mode by the missile override" in open(os.path.join(d, con[0])).read(), con
    finally:
        th.join(timeout=120)
        link.close(); fake.close()
    assert not fake.commands, f"{len(fake.commands)} CMD lines reached the bridge"
    assert row["mode"] == "stick" and row["fire_requests"] >= 1, (row, logs[-10:])
    launched = [m for m in logs if "fire launched" in m]
    assert launched and stick.cockpit.missile is not None and stick.cockpit.mode == "TWS", (logs[-15:], row)
    assert not stick.controlling, "the axes were not released at the end"
    assert row["red_style"] == "DCS AI" and row["first_shot"] in ("ours", "red", "none") and row["hdg_err_med_deg"] != "", row
    assert row["hud_aa_pct"] != "" and row["hud_aa_pct"] > 50 and row["first_bug_km"] != "", row
    assert any("point: nose to the target before the release" in m for m in logs), logs[-20:]
    assert any("A-A master mode by the missile override" in m for m in logs), logs[:20]
    delays = [float(x) for x in row["launch_delays_s"].split(";") if x]
    assert delays and all(d > 0 for d in delays), row
    assert dcs_live.launch_delays([(1.0, "requested"), (5.5, "launched"), (9.0, "requested"), (19.0, "timeout"),
                                   (20.0, "requested"), (23.0, "launched")]) == [4.5, 3.0]
    print(f"  DCS live stick .............. OK  ({row['fire_requests']} fire requests, first launch "
          f"{launched[0].strip()}, {delays[0]:.1f} s after its request; the bridge got no command)")


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
    print("  DCS speed boost ............. OK  (and --eta-lock)")


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
    # The DCS-like episodes (SIM_REV 12): a third shoot where the DCS AI does
    # (0.72-0.90 of R-max, a fifth at the edge: SIM_REV 18) and press on after
    # their missile goes active; 2v1 red never does.
    longs = 0
    for _ in range(300):
        o.reset(ic)
        if o.LONG_SHOT:
            longs += 1
            assert (0.72 <= o.SHOT_RMAX_FRAC <= 0.90 or 0.95 <= o.SHOT_RMAX_FRAC <= 1.0) and o.PRESS
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
    # In the environment: a runner keeps its distance from an agent flying at it slowly.
    env = E.BvrEnv(opponent_type=T.ADAPTIVE_SHOOTER, seed=4)
    class Runner(AdaptiveShooterOpponent):          # the env resets it: draw until a runner
        def reset(self, ic):
            super().reset(ic)
            while not self.RUNNER:
                super().reset(ic)
            self.TURN_RANGE, self.RUN_FAR = 20_000.0, 200_000.0     # running from the start
    env._opponent_factory = lambda e: Runner(rng=np.random.default_rng(11))
    env.reset()
    hot = list(E.HDG_OFFSETS_DEG).index(0.0)
    # Red starts nose on and takes about 20 s to turn away. Since SIM_REV 21 "hot" is at red from the start (it
    # was north before the first track, so this agent flew away and the range opened from the first step).
    for _ in range(20):
        env.step([hot, 2, 0, 0])
    r0 = env._state["range"]
    for _ in range(100):
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


def test_dcs_unnamed_events():
    """A death or ammo event whose unit DCS could not name (the bridge's getName() failed, so Lua dropped the
    field) is skipped, live and replayed; it ended a live run with a KeyError on 21 October."""
    import json
    from dcs_world import DcsRecording, DcsLiveRecording, FORMAT

    def unit(name, coal, x, t):
        return {"name": name, "coal": coal, "type": "F-16C_50", "x": x + 250.0 * t, "y": 8000.0, "z": 0.0,
                "vx": 250.0, "vy": 0.0, "vz": 0.0, "fx": 1.0, "fy": 0.0, "fz": 0.0, "ux": 0.0, "uy": 1.0, "uz": 0.0,
                "fuel": 0.9, "radar_on": False}
    lines = [{"format": FORMAT, "agent": "BLUE-1", "red": "RED-1", "rate": 0.1, "t0": 0, "bridge": 7}]
    lines += [{"t": round(0.1 * k, 1), "units": [unit("BLUE-1", 2, 0.0, 0.1 * k), unit("RED-1", 1, 50_000.0, 0.1 * k)],
               "weapons": []} for k in range(1, 6)]
    lines += [{"ev": "dead", "t": 0.3, "cause": "crash"},              # no "unit": the name was nil in Lua
              {"ev": "ammo", "t": 0.3, "items": []},
              {"ev": "dead", "t": 0.4, "unit": "RED-1", "cause": "dead"}]
    rec = DcsRecording([json.dumps(x) for x in lines])
    live = DcsLiveRecording()
    live.extend(lines)
    for r in (rec, live):
        assert list(r.dead) == ["RED-1"] and abs(r.dead["RED-1"][0] - 0.4) < 1e-9, r.dead
    print("  DCS unnamed events .......... OK  (a death with no unit is skipped, live and replayed)")


def test_dcs_unlimited_weapons():
    """With DCS's Unlimited Weapons on, a fired missile comes back on the rail (live_20261021_142940: 4 -> 2 -> 4,
    eight shots from four). The policy's missile count is loaded minus fired, as in training, and the refills are
    counted."""
    import json
    from dcs_world import DcsRecording, DcsReplayWorld, FORMAT

    def unit(name, coal, x, t):
        return {"name": name, "coal": coal, "type": "F-16C_50", "x": x + 250.0 * t, "y": 8000.0, "z": 0.0,
                "vx": 250.0, "vy": 0.0, "vz": 0.0, "fx": 1.0, "fy": 0.0, "fz": 0.0, "ux": 0.0, "uy": 1.0, "uz": 0.0,
                "fuel": 0.9, "radar_on": False}
    aim = lambda n: [{"type": "weapons.missiles.AIM_120C", "count": n, "aam": True, "guidance": 3}]
    lines = [{"format": FORMAT, "agent": "BLUE-1", "red": "RED-1", "rate": 0.1, "t0": 0, "bridge": 7}]
    lines += [{"t": round(0.5 * k, 1), "units": [unit("BLUE-1", 2, 0.0, 0.5 * k), unit("RED-1", 1, 50_000.0, 0.5 * k)],
               "weapons": []} for k in range(0, 41)]
    lines += [{"ev": "ammo", "t": 0.0, "unit": "BLUE-1", "items": aim(4)}]
    for i, ts in enumerate((2.0, 4.0, 12.0), start=1):           # three shots; the rail refilled after the second
        lines.append({"ev": "shot", "t": ts, "id": i, "shooter": "BLUE-1", "target": "RED-1", "type": "AIM_120C"})
    lines += [{"ev": "ammo", "t": 2.1, "unit": "BLUE-1", "items": aim(3)},
              {"ev": "ammo", "t": 4.1, "unit": "BLUE-1", "items": aim(2)},
              {"ev": "ammo", "t": 10.0, "unit": "BLUE-1", "items": aim(4)},     # refilled
              {"ev": "ammo", "t": 12.1, "unit": "BLUE-1", "items": aim(3)}]
    rec = DcsRecording([json.dumps(x) for x in lines])
    w = DcsReplayWorld(rec, "BLUE-1", "RED-1")
    got = {t: w._wpn("BLUE-1", t) for t in (1.0, 3.0, 5.0, 11.0, 13.0)}
    assert got == {1.0: 4, 3.0: 3, 5.0: 2, 11.0: 2, 13.0: 1}, got
    assert rec.ammo_refills("BLUE-1") == 1 and rec.ammo_refills("BLUE-1", 9.0) == 0
    print("  DCS unlimited weapons ....... OK  (refilled rails: the count stays loaded minus fired, 4 3 2 2 1)")


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


def test_selfplay_heading_history():
    """A self-play snapshot records its heading choices, so its "previous
    heading" inputs read its last choice (they always read 0 before)."""
    from bvr_env import BvrEnv, HDG_OFFSETS_DEG, OBS_LABELS, HDG_OBS_LABELS
    from bvr_opponents import BvrOpponentType as T
    from bvr_selfplay import PolicyOpponent
    k = HDG_OFFSETS_DEG.index(40.0)
    class Model:
        def predict(self, obs, action_masks=None, deterministic=False):
            return np.array([k, 2, 3, 0]), None
    e = BvrEnv(opponent_type=T.SELF_PLAY, seed=3, privileged_critic=False)
    e._opponent_factory = lambda env: PolicyOpponent(env, Model(), False, "test")
    e.reset()
    for _ in range(3):
        e.step(np.array([0, 2, 3, 0]))
    o = e._opponent._obs
    assert o._prev_hdg_off == 40.0, o._prev_hdg_off
    obs = o._build_obs()
    i = OBS_LABELS.index(HDG_OBS_LABELS[1])                   # sin of the previous choice
    assert obs[i] != 0.0, obs[i]
    print("  self-play heading history ... OK")


def test_fire_off_boresight():
    """The fire action is masked for a target more than 40 deg off the nose (SIM_REV 19; it was 60); scripted
    red's own fire rule keeps 60."""
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

    assert env.MAX_OFF_BORESIGHT_DEG == 40.0
    for az in (0, 30, -39):
        assert fire_ok(az) == (True, True), az
    for az in (41, -45, 59, -75, 90, 180):
        assert fire_ok(az) == (False, False), az
    assert fire_ok(0, el_deg=35) == (True, True)
    assert fire_ok(0, el_deg=45) == (False, False)
    assert fire_ok(25, el_deg=30) == (True, True)            # 38 deg off the nose in all
    assert fire_ok(30, el_deg=30) == (False, False)          # 41 deg off the nose in all
    red = BvrEnv(opponent_type=T.ADAPTIVE_SHOOTER, seed=1)
    red.reset(seed=1)
    assert red._opp_radar.MAX_OFF_BORESIGHT_DEG == 60.0 and not red._opp_radar.OFFAXIS_IN_MASK
    print("  fire off-boresight gate ..... OK  (40 deg for the policy, 60 for scripted red)")


def test_launch_delay():
    """A platform's launch delay (LAUNCH_DELAY_MIN/MAX): the missile leaves 3-7 s
    after the fire decision on F-16C-DCSFIT, the fire mask is shut meanwhile, and
    a launch whose target has left the fire cone by then is dropped; a platform
    without one launches in the decision's own step, as before."""
    import math
    from bvr_env import BvrEnv, HDG_OFFSETS_DEG
    from bvr_opponents import BvrOpponentType as T
    from bvr_track_adapter import TrackState

    def rig(plat, az_deg=0.0):
        env = BvrEnv(opponent_type=T.STRAIGHT, seed=1, privileged_critic=False,
                     platform=plat, opponent_platform=plat)
        env.reset()
        env._trk_state = lambda: TrackState.TRACK
        env._own_envelope = lambda est: (60_000.0, 20_000.0)
        off = {"az": az_deg}

        def est():                     # a target 20 km out, az_deg off the nose
            s = env._state; a = s.get("psi", 0.0) + math.radians(off["az"])
            pos = env._own_pos_enu() + 20_000.0 * np.array([math.sin(a), math.cos(a), 0.0])
            return {"valid": True, "pos": pos, "vel": np.zeros(3), "speed": 250.0, "age": 0.0,
                    "pos_sigma": 50.0, "vel_sigma": 5.0}
        env._est = est
        return env, off

    k0 = HDG_OFFSETS_DEG.index(0.0)
    # without a delay: the shot goes in the decision's own step
    env, _ = rig("F-16C-DCS")
    assert env.action_masks()[-1]
    env.step(np.array([k0, 2, 3, 1]))
    assert env._shots_fired == 1 and env._fire_due is None, env._shots_fired
    # with one: the release only (2.5-4 s) for a track held 25 s and more (the pilot has bugged it), the bug as well
    # (4.5-10.5 s more) for a younger one; the mask shut until then
    waits = {}
    for held in (60.0, 2.0):
        waits[held] = []
        for seed in range(6):
            env, _ = rig("F-16C-DCSFIT")
            env._launch_rng = np.random.default_rng(seed)
            env._track.n_hits = int(held * env.TRACK_CONVERT_HZ)
            env.step(np.array([k0, 2, 3, 1]))
            n = 1
            while env._shots_fired == 0:
                assert env._fire_due is not None and not env.action_masks()[-1]
                env.step(np.array([k0, 2, 3, 1])); n += 1
                assert n < 18, n
            waits[held].append(n)
            assert env._fire_requests == 1 and env._fire_dropped == 0
    assert all(3 <= w <= 5 for w in waits[60.0]), waits
    assert all(7 <= w <= 16 for w in waits[2.0]) and len(set(waits[2.0])) > 1, waits
    # the target leaves the fire cone during the wait: the launch is dropped
    env, off = rig("F-16C-DCSFIT")
    env._track.n_hits = 600                      # a held track: the release only, so it is due within the 9 steps
    env.step(np.array([k0, 2, 3, 1]))
    off["az"] = 80.0
    for _ in range(9):
        env.step(np.array([k0, 2, 3, 0]))
    assert env._shots_fired == 0 and env._fire_dropped == 1 and env._fire_due is None
    print(f"  launch delay ................ OK  (F-16C-DCSFIT: launched {min(waits[60.0])}-{max(waits[60.0])} decisions "
          f"after the request on a held track, {min(waits[2.0])}-{max(waits[2.0])} on a young one; dropped when the "
          f"target left the cone; F-16C-DCS at once)")


def test_adaptive_dcs_frac():
    """ADAPTIVE's share of DCS-like episodes (train_bvr.py --adaptive-dcs-frac): unset, the same draws as before
    (1/3); 0.67, about two thirds, also as self-play's scripted opponent."""
    import numpy as np
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T
    def shares(frac, opp=T.ADAPTIVE_SHOOTER, n=300):
        env = BvrEnv(opponent_type=opp, seed=7, privileged_critic=False, adaptive_dcs_frac=frac)
        if opp == T.SELF_PLAY:
            env._sp_scripted_frac = 1.0           # every self-play episode against the scripted ADAPTIVE
        out = []
        for i in range(n):
            env._ready = False                    # a new episode each time (reset() right after reset() is a no-op)
            env.reset()
            out.append(bool(env._opponent.LONG_SHOT))
        return out
    a, b = shares(None), shares(1.0 / 3.0)
    assert a == b, "unset must draw exactly as before"
    hi = shares(0.67)
    sp = shares(0.67, T.SELF_PLAY, n=150)
    assert 0.25 < np.mean(a) < 0.42 and 0.58 < np.mean(hi) < 0.76 and 0.55 < np.mean(sp) < 0.8, \
        (np.mean(a), np.mean(hi), np.mean(sp))
    print(f"  ADAPTIVE DCS-like share ...... OK  (default {np.mean(a):.0%}, 0.67 gives {np.mean(hi):.0%}; "
          f"in self-play {np.mean(sp):.0%})")


def test_episode_end():
    """SIM_REV 16: running out of time ends the episode (terminated, not truncated: no bootstrap through it), and
    the end state's potential is 0, so the discounted shaping sums to -phi(start)."""
    import numpy as np
    from bvr_env import BvrEnv, HDG_OFFSETS_DEG
    from bvr_opponents import BvrOpponentType as T
    env = BvrEnv(opponent_type=T.STRAIGHT, seed=0, privileged_critic=False)
    env.MAX_STEPS = 60
    env.reset()
    phi0, g, tot, k, big = env._prev_phi, env._gamma, 0.0, 0, 0.0
    while True:
        obs, r, term, trunc, info = env.step(np.array([HDG_OFFSETS_DEG.index(0.0), 2, 2, 0]))
        tot += g ** k * info["shaping"]; k += 1
        big = max(big, abs(info["shaping"]))
        if term or trunc:
            break
    assert term and not trunc and info["terminal_outcome"] == "TIMEOUT", (term, trunc, info.get("terminal_outcome"))
    assert abs(tot + phi0) < 1e-6 and big > 0.05, (tot, phi0, big)
    print(f"  episode end ................. OK  (timeout terminated, not truncated; shaping steps up to "
          f"{big:.2f}, discounted sum {tot:+.4f} = -phi(start) {-phi0:+.4f})")


def test_missile_endgame():
    """SIM_REV 17: an intercept on a target flying straight ends within a metre or so (the endgame in sub-steps);
    at 50 Hz alone it left 2-10 m, and with the 10 m fuze good shots were scored misses."""
    import math
    import numpy as np
    from missile_sim import AIM120
    from bvr_library import missile_config
    cfg = missile_config("AIM-120C-DCS")
    worst = 0.0
    for rng in (20000.0, 30000.0):
        for hdg in (180.0, 90.0, 135.0):
            m = AIM120(2, 1, np.array([0.0, 0.0, 9000.0]), np.array([0.0, 300.0, 0.0]), 12000.0, cfg=cfg)
            h = math.radians(hdg)
            vel = 250.0 * np.array([math.sin(h), math.cos(h), 0.0])
            tp = np.array([0.0, rng, 9000.0])
            out = None
            for _ in range(7500):
                m.update_guidance({"valid": 1, "tgt_pos": tp, "tgt_vel": vel})
                ev = m.step(0.02, tp, vel)
                tp = tp + vel * 0.02
                out = next((e for e in ev if e["type"] in ("MISSILE_HIT", "MISSILE_MISS")), None)
                if out:
                    break
            assert out and out["type"] == "MISSILE_HIT", (rng, hdg, out)
            worst = max(worst, out["miss_dist"])
    assert worst < 2.0, worst
    print(f"  missile endgame ............. OK  (6 intercepts on a straight target hit, worst miss {worst:.2f} m)")


def test_red_as_dcs():
    """SIM_REV 18: no missile has the support rule (ours and red's alike; dcs_live leaves them to DCS), the radar's
    azimuth limit is drawn from 57-63 deg, the DCS-like red shoots at 0.72-0.90 of R-max (a fifth at 0.95-1.0);
    dcs_live.chaff_wanted reads the RWR inputs."""
    import numpy as np
    from missile_sim import AIM120
    from bvr_library import missile_config
    from bvr_env import BvrEnv, OBS_LABELS
    from bvr_opponents import BvrOpponentType as T, AdaptiveShooterOpponent
    from bvr_radar_sim import RadarSim
    cfg = missile_config("AIM-120C-DCS")
    ends = {}
    for rule in (True, False):
        m = AIM120(2, 1, np.array([0.0, 0.0, 9000.0]), np.array([0.0, 300.0, 0.0]), 12000.0, cfg=cfg)
        m.support_rule = rule
        tp, vel = np.array([0.0, 45000.0, 9000.0]), np.array([0.0, -250.0, 0.0])
        out = None
        for k in range(7500):
            # guidance for 2 s, then none: the shooter lost its track
            m.update_guidance({"valid": 1, "tgt_pos": tp, "tgt_vel": vel} if k < 100 else {"valid": 0})
            ev = m.step(0.02, tp, vel)
            tp = tp + vel * 0.02
            out = next((e for e in ev if e["type"] in ("MISSILE_HIT", "MISSILE_MISS")), None)
            if out:
                break
        ends[rule] = out
    assert ends[True]["cause"] == "SUPPORT_LOST", ends[True]
    assert ends[False]["type"] == "MISSILE_HIT", ends[False]      # flew on to the target and its seeker took it
    # the env: no missile has the rule, ours and red's alike (True: the old rule for both)
    env = BvrEnv(opponent_type=T.ADAPTIVE_SHOOTER, privileged_critic=False, seed=3)
    env.reset(seed=3)
    assert env._world.no_support_rule == {1, 2}
    env._world._launch(1, {}); env._world._launch(2, {})
    owners = {m.owner: m.support_rule for m in env._world.missiles}
    assert owners == {1: False, 2: False}, owners
    env.SUPPORT_RULE = True
    env._ready = False          # a second reset() without a step is otherwise a no-op
    env.reset(seed=4)
    env._world._launch(1, {}); env._world._launch(2, {})
    assert {m.owner: m.support_rule for m in env._world.missiles} == {1: True, 2: True}
    # DCS live leaves every missile to DCS by default, as training does
    import inspect, dcs_live as _dl
    assert inspect.signature(_dl.DcsLiveWorld.__init__).parameters["support_rule"].default == "dcs"
    # radar azimuth limit
    fov = [np.degrees(RadarSim(rng=np.random.default_rng(i)).fov_az) for i in range(200)]
    assert 56.9 < min(fov) and max(fov) < 63.1 and max(fov) - min(fov) > 4, (min(fov), max(fov))
    # where the DCS-like red shoots
    fr = []
    for i in range(400):
        o = AdaptiveShooterOpponent(rng=np.random.default_rng(i))
        o.long_shot_frac = 1.0
        o.reset({"ac2_psi": 0.0})
        assert o.LONG_SHOT
        fr.append(o.SHOT_RMAX_FRAC)
    fr = np.array(fr)
    edge = (fr >= 0.95).mean()
    assert np.all(((fr >= 0.72) & (fr <= 0.90)) | ((fr >= 0.95) & (fr <= 1.0))) and 0.13 < edge < 0.27, edge
    # chaff from the RWR inputs
    import dcs_live
    o = np.zeros(len(OBS_LABELS))
    ix = {k: OBS_LABELS.index(k) for k in ("rwr_warn", "rwr_seeker_active", "inbound_tti")}
    assert not dcs_live.chaff_wanted(o)
    o[ix["rwr_warn"]] = 1; o[ix["inbound_tti"]] = 35.0
    assert not dcs_live.chaff_wanted(o)
    o[ix["inbound_tti"]] = 15.0
    assert dcs_live.chaff_wanted(o)
    o[ix["inbound_tti"]] = 35.0; o[ix["rwr_seeker_active"]] = 1
    assert dcs_live.chaff_wanted(o)
    print(f"  missiles as in DCS .......... OK  (no rule: the missile flew on and hit, with it removed; both sides; "
          f"radar limit {min(fov):.1f}-{max(fov):.1f} deg; long shots at the edge {edge:.0%}; chaff on the RWR)")


def test_offaxis_envelope():
    """SIM_REV 18: the envelope shaping term counts our R-max as launched along our heading: all of it pointing
    at the target or at the lead point of a crossing one, less off the nose (more so low down)."""
    import math
    import numpy as np
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T
    env = BvrEnv(opponent_type=T.STRAIGHT, privileged_critic=False, seed=1)
    env.reset(seed=1)
    def f(crank, alt=9000.0, tv=(0.0, -270.0)):
        a = math.radians(crank)
        env._own_pos_enu = lambda: np.array([0.0, 0.0, alt])
        env._own_vel_enu = lambda: np.array([400.0 * math.sin(a), 400.0 * math.cos(a), 0.0])
        env._state = dict(env._state, alt=alt)
        return env._offaxis_factor({"valid": True, "pos": np.array([0.0, 50000.0, alt]),
                                    "vel": np.array([tv[0], tv[1], 0.0]), "speed": 270.0})
    assert abs(f(0) - 1.0) < 1e-9 and abs(f(-30) - f(30)) < 1e-9
    assert 0.93 < f(50) < 0.96 and 0.80 < f(50, alt=5000.0) < 0.86 and f(90) < f(60) < f(50)
    # a target crossing at 200 m/s: pointing at its lead point keeps all of the reach, at it does not
    lead = math.degrees(math.asin(200.0 / env.OFFAXIS_V_MSL))
    assert f(lead, tv=(200.0, -180.0)) > 0.999 and f(-lead, tv=(200.0, -180.0)) < f(0, tv=(200.0, -180.0))
    assert env._offaxis_factor({"valid": False}) == 1.0
    print(f"  off-axis envelope ........... OK  (50 deg off: {f(50):.2f} at 9 km, {f(50, alt=5000.0):.2f} at 5 km; "
          f"the lead point of a crossing target keeps it all)")


def test_episode_report():
    """bvr_report: each fight's record (info["report"]): red's style, every missile, each defence against a red
    missile, who fired first, heading choices by phase, track losses, and a timeout's end state."""
    import numpy as np
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T
    from bvr_report import EpisodeReport, summarise
    env = BvrEnv(opponent_type=T.ADAPTIVE_SHOOTER, privileged_critic=False, seed=7, adaptive_dcs_frac=1.0)
    reps = []
    for seed in range(6):
        env.reset(seed=seed)
        while True:
            m = env.action_masks()
            _, _, d, tr, info = env.step([0, 2, 3, int(m[-1])])     # fly at red, fire whenever allowed
            if d or tr:
                break
        r = info["report"]
        reps.append(r)
        assert r["outcome"] == info["terminal_outcome"] and r["red"]["style"] == "DCS-like", r["red"]
        n_dec = sum(sum(c.values()) for c in r["phases"].values())
        assert n_dec == info["step"], (n_dec, info["step"])
        ours = [x for x in r["missiles"] if x["owner"] == "ours"]
        assert len(ours) == info["shots_fired"]
        for x in r["missiles"]:
            assert x["end"] in ("HIT", "KINEMATIC", "ENDGAME", "SUPPORT_LOST", "DECOYED", "in flight"), x
            assert x["range_km"] > 0 and x["t_flight"] is not None and x["r_over_rmax"] is not None
        reds = [x for x in r["missiles"] if x["owner"] == "red"]
        assert len(r["defence"]) == len(reds)
        for dd in r["defence"]:
            assert dd["end"] == next(x["end"] for x in reds if x["id"] == dd["id"])
            if dd["t_break"] is not None:
                assert dd["tti_break"] is not None and dd["track_at_break"] in ("TRACK", "COAST", "NONE", "ACQUIRING")
        assert r["first"]["who"] in ("ours", "red", "none")
        if r["outcome"] == "TIMEOUT":
            assert r["timeout"]["cause"] and len(r["timeout"]["last_headings"]) <= 60
    assert any(r["defence"] for r in reps), "no red missile in 6 fights against the DCS-like red"
    assert all(set(r["phases"]["before_shot"]) <= {"0"} for r in reps)           # we flew at red throughout
    # the timeout labels
    tc = EpisodeReport.timeout_cause
    assert tc({"aa_deg_t": 170, "aa_deg": 10, "wpn_remaining": 2}) == "red running, we follow"
    assert tc({"aa_deg_t": 100, "aa_deg": 120, "wpn_remaining": 2}) == "both turned away"
    assert tc({"aa_deg_t": 10, "aa_deg": 20, "wpn_remaining": 2}) == "both pointing"
    assert tc({"wpn_remaining": 0, "wpn_remaining_t": 0}) == "both out of missiles"
    sm = summarise(reps)
    assert sm["n"] == 6 and sum(sm["first"].values()) == 6
    print(f"  fight records ............... OK  ({sum(len(r['missiles']) for r in reps)} missiles, "
          f"{sm['defence']['missiles']} red at us ({sm['defence']['hit']} hit, {sm['defence']['broke']} broken from) "
          f"in 6 fights; first shot {sm['first']})")


def test_geometry_inputs():
    """SIM_REV 19's inputs: which way the target crosses our line of sight (+ to our right), the signed
    line-of-sight rate, and range over the off-axis R-max (above the table's when cranked); a mirror image
    flips the first two."""
    import math
    import numpy as np
    from bvr_env import BvrEnv, OBS_LABELS
    from bvr_opponents import BvrOpponentType as T
    env = BvrEnv(opponent_type=T.STRAIGHT, privileged_critic=False, seed=2)
    env.reset(seed=2)
    ix = {k: OBS_LABELS.index(k) for k in ("trk_aspect_side", "los_rate_signed", "r_over_rmax_axis",
                                         "r_over_rmax_own", "trk_aspect_sin")}
    def obs(tvx, crank=0.0, mirror=1.0):
        a = math.radians(crank) * mirror
        env._own_pos_enu = lambda: np.array([0.0, 0.0, 9000.0])
        env._own_vel_enu = lambda: np.array([400.0 * math.sin(a), 400.0 * math.cos(a), 0.0])
        env._state = dict(env._state, alt=9000.0, psi=a, speed=400.0, mach=1.3)
        est = {"valid": True, "pos": np.array([0.0, 40000.0, 9000.0]), "vel": np.array([mirror * tvx, -250.0, 0.0]),
               "speed": float(math.hypot(tvx, 250.0)), "age": 0.0, "pos_sigma": 80.0, "vel_sigma": 5.0}
        env._est = lambda: est
        o = env._build_obs()
        lo, hi = env._obs_lo, env._obs_hi
        return {k: float(lo[i] + (o[i] + 1) * (hi[i] - lo[i]) / 2) for k, i in ix.items()}
    right, left = obs(200.0), obs(200.0, mirror=-1.0)
    assert right["trk_aspect_side"] > 0.5 and right["los_rate_signed"] > 0, right       # red flies to our right
    assert abs(left["trk_aspect_side"] + right["trk_aspect_side"]) < 1e-6
    assert abs(left["los_rate_signed"] + right["los_rate_signed"]) < 1e-6
    assert abs(left["trk_aspect_sin"] - right["trk_aspect_sin"]) < 1e-6                 # the old input cannot tell
    hot, cranked = obs(0.0), obs(0.0, crank=50.0)
    assert abs(hot["r_over_rmax_axis"] - hot["r_over_rmax_own"]) < 1e-3
    assert cranked["r_over_rmax_axis"] > 1.04 * cranked["r_over_rmax_own"]
    print(f"  geometry inputs ............. OK  (crossing {right['trk_aspect_side']:+.2f} / {left['trk_aspect_side']:+.2f}, "
          f"LOS rate {right['los_rate_signed']:+.2f} deg/s; cranked 50 deg r/Rmax {cranked['r_over_rmax_own']:.2f} -> "
          f"{cranked['r_over_rmax_axis']:.2f} off-axis)")


def test_threat_only_while_closing():
    """SIM_REV 20: a red missile warns (RWR, threat input) while it closes on us and 3 s after; one that has
    overshot drops off though it flies on."""
    import numpy as np
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T
    from missile_sim import THREAT_HOLD_S
    env = BvrEnv(opponent_type=T.STRAIGHT, privileged_critic=False, seed=4)
    env.reset(seed=4)
    w = env._world
    w._launch(2, {})
    m = w.missiles[-1]
    a1 = w.acs[0]
    # put the missile 3 km behind us, flying away from us: it has overshot
    fwd = np.array([np.sin(a1.chi), np.cos(a1.chi), 0.0])
    m.pos = np.array([a1.x, a1.y, a1.z]) - 3000.0 * fwd
    m.vel = -900.0 * fwd
    m.guidance_valid = False; m.support_rule = False
    t0 = w.t_sim
    warn = []
    for _ in range(int((THREAT_HOLD_S + 2.0) / w.SIM_DT)):
        w.step(env._encode_cmd(0, 2, 2, 0), {"mode": 0, "V": 250.0, "hdgCmd": 0.0, "altTarget": 9000.0, "fire": 0})
        tel = w.telemetry(1)
        warn.append((w.t_sim - t0, tel["rwr"]["launch_warn"], [x["threat"] for x in tel["missiles"] if x["owner"] == 2]))
    assert warn[0][1] == 1 and warn[-1][1] == 0, (warn[0], warn[-1])
    off = next(t for t, wn, _ in warn if not wn)
    assert THREAT_HOLD_S - 0.1 <= off <= THREAT_HOLD_S + 0.2, off
    assert m in w.missiles and warn[-1][2] == [0]                         # still flying, no longer a threat
    env._state = w.telemetry(1)
    assert env._missile_summary()[1] is None
    print(f"  threat only while closing ... OK  (an overshot missile warns {off:.1f} s, then drops off; it flies on)")


def test_start_reference():
    """SIM_REV 21: before the first track the heading choices are offsets from the bandit's bearing at the
    start (the picture call), not from north; without GCI_START, from the own start heading. Self-play's AC2
    gets the same."""
    import math
    import numpy as np
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T
    e = BvrEnv(opponent_type=T.STRAIGHT, seed=3)
    errs = []
    for seed in range(12):
        e._ready = False
        _, info = e.reset(seed=seed)
        ic = info["ic"]
        d = e._to_enu(ic["ac2_lat"], ic["ac2_lon"], 0) - e._to_enu(ic["ac1_lat"], ic["ac1_lon"], 0)
        brg = math.atan2(d[0], d[1])
        if not e._est()["valid"]:                      # the reference has not been replaced by a track yet
            errs.append(abs(math.remainder(e._hdg_ref - brg, 2 * math.pi)))
        assert abs(math.remainder(e._start_ref(ic, 2) - (brg + math.pi), 2 * math.pi)) < 1e-9
    assert errs and max(errs) < 1e-9, errs
    e.GCI_START = False
    e._ready = False
    _, info = e.reset(seed=1)
    assert e._est()["valid"] or abs(math.remainder(e._hdg_ref - info["ic"]["ac1_psi"], 2 * math.pi)) < 1e-9
    print(f"  start reference ............. OK  ({len(errs)} starts without a track: offsets from the bandit's bearing)")


def test_turn_hold():
    """SIM_REV 21: commands alternating either side of the tail (161/191 deg while flying 0, as dcsfit_v6 in
    DCS on 21 October) turn the aircraft round instead of reversing the bank every second (the stick pilot:
    dcs/stick/test_stick.py test_turn_hold)."""
    import math
    from f16_sim import F16Aircraft, turn_hold, isa
    r = math.radians
    assert turn_hold(r(-170), r(20)) > math.pi and turn_hold(r(170), r(-20)) < -math.pi   # held against the bank
    assert turn_hold(r(-170), r(2)) == r(-170)                                            # wings level: shorter way
    assert turn_hold(r(-140), r(30)) == r(-140)                                           # inside 150: shorter way
    ac = F16Aircraft()
    ac.reset_state(0, 0, 5000.0, 0.0, 0.85 * isa(5000.0)[3], 0.6)
    revs, sign = 0, 0
    for k in range(int(40 / 0.02)):
        cmd = r(161.0) if int(k * 0.02) % 2 == 0 else r(191.0)
        ac.step(0.02, {"hdgCmd": cmd, "altTarget": 5000.0, "V": ac.V})
        turned = abs(math.degrees(math.remainder(ac.chi - 0.0, 2 * math.pi)))
        if turned > 140:                            # round: the commands are ahead of the nose now
            break
        if abs(ac.phi) > r(15):
            sg = 1 if ac.phi > 0 else -1
            revs += sign and sg != sign
            sign = sg
    assert turned > 140 and revs <= 1, (turned, revs)
    print(f"  turn hold ................... OK  (alternating 161/191 deg off the tail: turned 140 deg in "
          f"{k * 0.02:.0f} s with {revs} bank reversal(s))")


def test_sim_rev_22():
    """SIM_REV 22: a beaming target decoys an active seeker, one flying at it hardly ever; a new track on the
    APG-68-DCS radar takes its 15 s, a short coast does not; a pressing red chases a fleeing target at dash speed."""
    import math
    import numpy as np
    from missile_sim import decoy_rate, DECOY_P_BEAM, DECOY_P_OTHER
    assert decoy_rate([0, 0, 0], [5000, 0, 0], [0, 300, 0]) == DECOY_P_BEAM          # beam
    assert decoy_rate([0, 0, 0], [5000, 0, 0], [-300, 0, 0]) == DECOY_P_OTHER        # flying at the missile
    assert decoy_rate([0, 0, 0], [15000, 0, 0], [0, 300, 0]) == 0.0                  # beyond 10 km
    from bvr_env import BvrEnv
    from bvr_opponents import BvrOpponentType as T, AdaptiveShooterOpponent
    e = BvrEnv(opponent_type=T.STRAIGHT, seed=2, platform="F-16C-DCSFIT-R", opponent_platform="F-16C-DCSFIT")
    assert e._track.acquire_first_hits == 150
    t_first = None
    for seed in range(6):                             # from the first detection (ACQUIRING) to TRACK
        e._ready = False
        e.reset(seed=seed)
        t_seen = None
        for k in range(150):
            e.step([0, 2, 3, 0])
            st = e._trk_state()
            if st == 1 and t_seen is None:
                t_seen = k
            if st == 2:
                t_first = None if t_seen is None else k - t_seen
                break
        if t_first:
            break
    assert t_first is not None and t_first >= 14, t_first
    o = AdaptiveShooterOpponent(rng=np.random.default_rng(1))
    o.reset({"ac2_psi": 0.0, "ac2_alt": 9000.0, "ac2_spd": 300.0, "ac1_psi": 0.0})
    o.PRESS, o._phase, o.FAST_SPEED, o.COMMIT_SPEED = True, "REATTACK", 480.0, 400.0
    o.rmax_t = 1.0
    st = {"lat": 0.2, "lon": 0.0, "lat_t": 0.0, "lon_t": 0.0, "range": 22000.0, "wpn_remaining_t": 2, "missiles": []}
    away, at = o.act(dict(st, psi=0.0), 10.0), o.act(dict(st, psi=math.pi), 11.0)
    assert away["V"] == 480.0 and at["V"] == 400.0, (away["V"], at["V"])
    print(f"  SIM_REV 22 .................. OK  (decoy {DECOY_P_BEAM}/s beaming, {DECOY_P_OTHER}/s otherwise; first "
          f"track after {t_first} s on APG-68-DCS; red chases a runner at dash speed)")


def test_symmetric_policy():
    """--symmetric-policy: the mirror image of the inputs gives the mirror image of the choice, exactly;
    the policy trains and saves with it; off, it is the plain policy."""
    import tempfile, os
    import numpy as np, torch as th
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO
    from bvr_env import OBS_DIM, PRIV_DIM, ACTION_NVEC
    from bvr_policy import AsymmetricMaskablePolicy, mirror_columns, heading_mirror
    from bvr_selfplay import N_STACK
    box = lambda n: spaces.Box(low=-1.0, high=1.0, shape=(n * N_STACK,), dtype=np.float32)
    class Dummy:
        observation_space = spaces.Dict({"obs": box(OBS_DIM), "priv": box(PRIV_DIM)})
        action_space = spaces.MultiDiscrete(ACTION_NVEC)
    import gymnasium as gym
    class E(gym.Env):
        observation_space = Dummy.observation_space; action_space = Dummy.action_space
        def reset(self, seed=None, options=None): return self.observation_space.sample(), {}
        def step(self, a): return self.observation_space.sample(), 0.0, False, False, {}
        def action_masks(self): return np.ones(sum(ACTION_NVEC), bool)
    m = MaskablePPO(AsymmetricMaskablePolicy, E(), n_steps=64, batch_size=32, n_epochs=1,
                    policy_kwargs=dict(pi_arch=(64, 64), vf_arch=(64, 64), symmetric=True), seed=0)
    rng = np.random.default_rng(0)
    o = rng.uniform(-1, 1, (8, OBS_DIM * N_STACK)).astype(np.float32)
    om = o.copy(); om[:, mirror_columns(OBS_DIM * N_STACK)] *= -1
    pv = rng.uniform(-1, 1, (8, PRIV_DIM * N_STACK)).astype(np.float32)
    def probs(model, x):
        t = {"obs": th.as_tensor(x), "priv": th.as_tensor(pv)}
        with th.no_grad():
            d = model.policy.get_distribution(t, action_masks=np.ones((8, sum(ACTION_NVEC)), bool))
        return [p.probs.numpy() for p in d.distributions]
    p, pm = probs(m, o), probs(m, om)
    mir = heading_mirror()
    assert np.allclose(p[0], pm[0][:, mir], atol=1e-6) and all(np.allclose(a, b, atol=1e-6) for a, b in zip(p[1:], pm[1:]))
    m.learn(128)                                                      # gradients through both views
    p, pm = probs(m, o), probs(m, om)
    assert np.allclose(p[0], pm[0][:, mir], atol=1e-6)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.zip"); m.save(path)
        m2 = MaskablePPO.load(path, env=E())
        assert m2.policy.symmetric and np.allclose(probs(m2, o)[0], p[0], atol=1e-6)
    m.policy.set_symmetric(False)                                     # off: the plain, asymmetric choice
    q, qm = probs(m, o), probs(m, om)
    assert not np.allclose(q[0], qm[0][:, mir], atol=1e-3)
    print("  symmetric policy ............ OK  (mirrored inputs give the mirrored choice exactly, after training "
          "too; kept on save and load)")


def test_mission_seed_starts():
    """dcs/make_mission.py --seed: starts drawn as training draws them, the same for a seed, different across seeds."""
    import importlib.util, os
    if importlib.util.find_spec("dcs") is None:
        print("  mission seed starts .......... skipped (pydcs not installed)")
        return
    spec = importlib.util.spec_from_file_location("make_mission", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                                "dcs", "make_mission.py"))
    mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
    draws = [mm.draw_start(s) for s in range(200)]
    assert mm.draw_start(7) == mm.draw_start(7)
    kinds = {d.split(":")[1].split(",")[0].split(" (")[0].strip() for _, d in draws}
    assert kinds == set(mm.SCENARIOS), kinds
    for v, d in draws:
        stern = "stern" in d
        assert (45 <= v["range_km"] <= 75) if stern else (70 <= v["range_km"] <= 110), d
        assert 2000 <= v["alt_blue"] <= 12500 and 2000 <= v["alt_red"] <= 12500 and 250 <= v["speed"] <= 320, d
        assert (v["blue_off"] == 0.0) != (v["red_off"] == 0.0) or abs(v["blue_off"] + v["red_off"]) < 9, d
    mirrored = sum("mirrored" in d for _, d in draws) / len(draws)
    assert 0.35 < mirrored < 0.65, mirrored
    print(f"  mission seed starts .......... OK  (all 5 scenarios, {mirrored:.0%} mirrored, ranges and altitudes as in training)")


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
    print(f"  DCS turn test (sim) ......... OK  ({min(r['rate_bulk_dps'] for r in rows):.1f}-"
          f"{max(r['rate_bulk_dps'] for r in rows):.1f} deg/s)")


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


def test_reward_balance():
    """The terminal rewards are ordered kill > mutual > timeout > shot down, and
    the heading cost cannot outweigh them: wing-rocking for a whole 400-step
    episode (a 60 deg flip every other step) costs less, discounted, than a
    kill is worth, and a timeout's discounted terminal reward plus that cost
    still comes to no worse than being shot down at the same time of day.
    (dcs_v8's audit: heading cost -0.7 to -2.8 an episode against terminal
    rewards of 1, so dying beat a timeout.)"""
    from bvr_env import BvrEnv
    from bvr_team import TeamBvrEnv
    g, T = 0.997, 400
    assert BvrEnv.R_KILL > BvrEnv.R_MUTUAL > BvrEnv.R_TIMEOUT > BvrEnv.R_KILLED
    assert TeamBvrEnv.R_TIMEOUT == BvrEnv.R_TIMEOUT
    flip = BvrEnv.W_HDG_CHANGE * 60 / 180                     # a +30 <-> -30 flip
    rocking = sum(flip * g ** t for t in range(0, T, 2))      # every other step
    assert rocking < BvrEnv.R_KILL, rocking
    # a timeout (terminal at the end, rocking all along) against a loss halfway
    timeout = BvrEnv.R_TIMEOUT * g ** T - rocking
    loss_mid = BvrEnv.R_KILLED * g ** (T // 2) - sum(flip * g ** t for t in range(0, T // 2, 2))
    assert timeout > loss_mid - 0.5, (timeout, loss_mid)
    print(f"  reward balance .............. OK  (rocking all episode costs {rocking:.2f} discounted; "
          f"timeout {timeout:+.2f}, loss at half time {loss_mid:+.2f})")

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
        ("DCS unnamed events",      test_dcs_unnamed_events),
        ("DCS unlimited weapons",   test_dcs_unlimited_weapons),
        ("start reference",         test_start_reference),
        ("turn hold",               test_turn_hold),
        ("SIM_REV 22",              test_sim_rev_22),
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
        ("DCS live stick",          test_dcs_live_stick),
        ("DCS support rule",        test_dcs_support_rule),
        ("DCS speed boost",         test_dcs_speed_boost),
        ("DCS red support",         test_dcs_red_support),
        ("missile loft",            test_missile_loft),
        ("self-play mix",           test_selfplay_mix),
        ("self-play heading history", test_selfplay_heading_history),
        ("launch delay", test_launch_delay),
        ("ADAPTIVE DCS-like share", test_adaptive_dcs_frac),
        ("episode end", test_episode_end),
        ("missile endgame", test_missile_endgame),
        ("missiles as in DCS", test_red_as_dcs),
        ("off-axis envelope", test_offaxis_envelope),
        ("fight records", test_episode_report),
        ("geometry inputs", test_geometry_inputs),
        ("threat only while closing", test_threat_only_while_closing),
        ("symmetric policy", test_symmetric_policy),
        ("mission seed starts", test_mission_seed_starts),
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
        ("reward balance",          test_reward_balance),
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
