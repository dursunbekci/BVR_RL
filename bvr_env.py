"""
bvr_env.py  —  F-16 1v1 BVR Gymnasium environment (pure-Python sim)
====================================================================

No UDP, no C++. The sim (f16_sim + missile_sim + sim_world) runs inside
the env process. step() is a direct Python function call.

Throughput: ~600 episodes/s/core vs ~3/s with realtime C++ lockstep.
Everything else (observation, reward, masking, potential) is unchanged.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim_world        import SimWorld
from bvr_track_adapter import RadarTrackAdapter, TrackState, body_to_enu_matrix
from bvr_envelope     import Aim120Envelope, aspect_deg_from_vectors
from bvr_radar_sim    import RadarSim
from bvr_opponents    import BvrOpponent, BvrOpponentType

G       = 9.80665
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
REF_LAT = 39.0
REF_LON = 35.0
R_EARTH = 6_371_000.0
A_SOUND = 300.0
RANGE_MAX  = 160_000.0
ALT_MIN_OP = 1_000.0
ALT_MAX_OP = 14_000.0

# ── action space ─────────────────────────────────────────────────────
HDG_OFFSETS_DEG = [0.0,30.0,-30.0,50.0,-50.0,90.0,-90.0,135.0,-135.0,180.0]
ALT_DELTAS_M    = [-3000.0,-1200.0,0.0,+1200.0,+3000.0]
SPEED_CMDS      = [220.0,280.0,340.0,400.0]
FIRE_OPTIONS    = [0,1]
ACTION_NVEC     = [len(HDG_OFFSETS_DEG),len(ALT_DELTAS_M),len(SPEED_CMDS),len(FIRE_OPTIONS)]

# ── observation layout ────────────────────────────────────────────────
OBS_LABELS = [
    "own_mach","own_alt","own_gamma_fpa","own_phi_cos","own_phi_sin","own_nz",
    "own_energy","own_alpha","wpn_remaining","fuel_frac",
    "trk_none","trk_acquiring","trk_track","trk_coast",
    "trk_range","trk_range_rate","trk_ata_cos","trk_ata_sin","trk_el",
    "trk_aspect_cos","trk_aspect_sin","trk_age","trk_pos_sigma","trk_vel_sigma",
    "tgt_speed_est","tgt_alt_rel_est","tgt_energy_delta","los_rate",
    "r_over_rmax_own","r_over_rnez_own","r_over_rmax_thr","r_over_rnez_thr",
    "own_msl_count","own_msl_tgo","own_msl_tflight",
    "own_msl_support","own_msl_active","own_msl_stale",
    "rwr_warn","rwr_age","rwr_bearing_cos","rwr_bearing_sin",
    "rwr_seeker_active","inbound_tti",
    "t_norm","since_last_shot",
]
OBS_DIM = len(OBS_LABELS)

OBS_PHYS_LOW = np.array([
    0.4,ALT_MIN_OP,-0.6,-1,-1,-4,0,-0.3,0,0,
    0,0,0,0,0,-900,-1,-1,-1,-1,-1,0,0,0,
    0,-8000,-12000,-10,0,0,0,0,
    0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,
],dtype=np.float32)
OBS_PHYS_HIGH = np.array([
    2,ALT_MAX_OP,0.6,1,1,9,30000,0.5,6,1,
    1,1,1,1,12,900,1,1,1,1,1,20,12,8,
    700,8000,12000,10,3,3,3,3,
    6,90,90,1,1,5,1,120,1,1,1,90,1,120,
],dtype=np.float32)
assert len(OBS_PHYS_LOW)==OBS_DIM==len(OBS_PHYS_HIGH)

PRIV_LABELS = [
    "true_range","true_closure",
    "true_aa_cos","true_aa_sin","true_aa_t_cos","true_aa_t_sin",
    "true_hca_cos","true_hca_sin",
    "true_alt_rel","true_speed_t","true_energy_delta",
    "true_tgt_wpn","true_tgt_has_lock",
    "true_nearest_threat_tgo","true_nearest_own_tgo","true_r_over_rmax_thr",
]
PRIV_DIM  = len(PRIV_LABELS)
PRIV_PHYS_LOW  = np.array([0,-900,-1,-1,-1,-1,-1,-1,-12000,0,-12000,0,0,0,0,0],dtype=np.float32)
PRIV_PHYS_HIGH = np.array([RANGE_MAX,900,1,1,1,1,1,1,12000,700,12000,6,1,120,120,3],dtype=np.float32)

# The five components of the shaping potential, in the order _potential()
# builds them. Consumers (trainer logging, GUI panel) key off this.
PHI_TERMS = ["envelope","track","energy","support","defence"]


class BvrEnv(gym.Env):
    metadata = {"render_modes":[]}

    DECISION_HZ  = 1.0
    MAX_STEPS    = 300
    MIN_ALT      = 300.0
    ESCAPE_RANGE = 150_000.0

    W_ENVELOPE = 0.60; W_TRACK = 0.25; W_ENERGY = 0.15
    W_SUPPORT  = 0.30; W_DEFENCE = 0.80

    R_KILL=+1.0; R_KILLED=-1.0; R_MUTUAL=-0.4; R_TIMEOUT=-0.5
    R_ESCAPE=-0.6; R_CRASH=-1.0; R_BANDIT_CRASH=+0.4; R_WASTED_MSL=-0.03

    RADAR_OMEGA_COMPENSATED = False
    TRACK_CONVERT_HZ        = 10.0

    def __init__(self, opponent_type=BvrOpponentType.STRAIGHT,
                 gamma_discount=0.997, seed=42, instance_id=0,
                 privileged_critic=True, envelope_table=None,
                 radar_model="sim", enable_viz=False):
        super().__init__()
        self._opponent_type = opponent_type
        self._gamma = float(gamma_discount)
        if self._gamma < 0.99:
            print(f"[bvr_env] WARNING: gamma={self._gamma} too low — use >=0.995")
        self._rng       = np.random.default_rng(seed + instance_id*1000)
        self._instance  = int(instance_id)
        self._privileged = bool(privileged_critic)

        self.action_space = spaces.MultiDiscrete(ACTION_NVEC)
        box = spaces.Box(low=-1.0,high=1.0,shape=(OBS_DIM,),dtype=np.float32)
        self.observation_space = (
            spaces.Dict({"obs":box,"priv":spaces.Box(low=-1.0,high=1.0,shape=(PRIV_DIM,),dtype=np.float32)})
            if self._privileged else box)

        self._obs_lo, self._obs_hi   = OBS_PHYS_LOW.copy(),  OBS_PHYS_HIGH.copy()
        self._priv_lo,self._priv_hi  = PRIV_PHYS_LOW.copy(), PRIV_PHYS_HIGH.copy()

        self._world   = SimWorld(seed=seed+instance_id*1000)
        self._track   = RadarTrackAdapter()
        self._env_mdl = Aim120Envelope(envelope_table)
        self._radar_model = radar_model
        self._radar   = RadarSim(rng=self._rng) if radar_model=="sim" else None

        self._state={}; self._step_num=0; self._t_sim=0.0
        self._ready=False; self._episode_id=0; self._opponent=None
        self._outcome=None; self._last_shot_t=-999.0
        self._shots_fired=0; self._misses=0; self._support_losses=0
        self._events_seen=set(); self._launch_log=[]; self._last_convert_t=-1e9
        self._cmd_hdg=0.0; self._cmd_alt=9000.0; self._cmd_spd=300.0
        self._prev_phi=0.0
        self._phi_terms={k:0.0 for k in PHI_TERMS}
        self._prev_phi_terms=dict(self._phi_terms)

        self._viz=None
        if enable_viz:
            try:
                from viz_publisher import VizPublisher
                self._viz=VizPublisher(ref_lat=REF_LAT,ref_lon=REF_LON)
            except Exception as e:
                print(f"[bvr_env] viz disabled: {e}")

    # ── reset ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._ready and self._step_num==0 and self._state:
            return self._build_obs(),{}

        self._step_num=0; self._t_sim=0.0; self._ready=False; self._outcome=None
        self._last_shot_t=-999.0; self._shots_fired=0; self._misses=0
        self._support_losses=0; self._events_seen.clear(); self._launch_log=[]
        self._state={}; self._last_convert_t=-1e9
        self._track.reset()
        if self._radar is not None: self._radar.reset()

        ic = self._random_ic()
        self._episode_id = int(self._rng.integers(1,2_000_000_000))
        self._opponent   = BvrOpponent.create(self._opponent_type,rng=self._rng)
        self._opponent.reset(ic)
        self._world.reset(ic, episode_id=self._episode_id)
        self._cmd_hdg=float(ic["ac1_psi"]); self._cmd_alt=float(ic["ac1_alt"])
        self._cmd_spd=float(ic["ac1_spd"])

        self._advance(0.5, self._encode_cmd(0,2,2,0), fire=False)
        self._ready=True
        obs=self._build_obs(); self._prev_phi=self._potential()
        self._prev_phi_terms=dict(self._phi_terms)
        return obs, {"ic":ic,"episode_id":self._episode_id}

    # ── step ──────────────────────────────────────────────────────────
    def step(self, action):
        if not self._ready: raise RuntimeError("step() before reset()")
        action = np.asarray(action,dtype=np.int64).reshape(-1)
        i_hdg,i_alt,i_spd,i_fire = int(action[0]),int(action[1]),int(action[2]),int(action[3])
        self._step_num += 1
        want_fire = bool(FIRE_OPTIONS[i_fire]) and self._can_fire()
        cmd = self._encode_cmd(i_hdg,i_alt,i_spd,i_fire)
        self._advance(1.0/self.DECISION_HZ, cmd, fire=want_fire)

        obs   = self._build_obs()
        phi   = self._potential()
        shaping = self._gamma*phi - self._prev_phi
        self._prev_phi = phi
        # Same gamma*new - old applied per term, so these sum to `shaping`
        # exactly and show which part of the potential moved the reward.
        terms = self._phi_terms
        shaping_terms = {k: self._gamma*terms[k] - self._prev_phi_terms[k]
                         for k in PHI_TERMS}
        self._prev_phi_terms = dict(terms)
        reward = float(shaping)

        terminated = self._outcome is not None
        truncated  = (not terminated) and (self._step_num >= self.MAX_STEPS)
        info = {"shaping":shaping,"phi":phi,"step":self._step_num,
                "phi_terms":dict(terms),"shaping_terms":shaping_terms,
                "t_sim":self._t_sim,"track_state":TrackState.NAMES[self._track.state],
                "fired":want_fire,"opponent":self._opponent_type.name}

        if terminated or truncated:
            if truncated: self._outcome="TIMEOUT"
            reward += self._terminal_reward(self._outcome)
            info.update({"terminal_outcome":self._outcome,
                         "shots_fired":self._shots_fired,
                         "misses":self._misses,
                         "support_losses":self._support_losses,
                         "launch_log":list(self._launch_log),
                         "wpn_remaining":self._state.get("wpn_remaining",0)})
            self._ready=False

        if self._viz:
            try: self._viz.publish(self._state,action,reward,info,
                                   self._step_num,self._outcome or "",
                                   self._opponent_type.name)
            except Exception: pass

        return obs, reward, terminated, truncated, info

    # ── action masking ────────────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        m_hdg=np.ones(len(HDG_OFFSETS_DEG),dtype=bool)
        m_alt=np.ones(len(ALT_DELTAS_M),dtype=bool)
        m_spd=np.ones(len(SPEED_CMDS),dtype=bool)
        m_fire=np.array([True,self._can_fire()],dtype=bool)
        return np.concatenate([m_hdg,m_alt,m_spd,m_fire])

    def _can_fire(self) -> bool:
        if self._state.get("wpn_remaining",0)<=0: return False
        if self._track.state!=TrackState.TRACK:   return False
        if (self._t_sim-self._last_shot_t)<3.0:   return False
        est=self._track.estimate()
        if not est["valid"]: return False
        r=self._est_range(est); r_max,_=self._own_envelope(est)
        # BUG FIX: this used to read `r<=1.15*r_max` — a shot up to 15% BEYOND
        # the missile's own computed kinematic reach. That is not a margin,
        # it is a guaranteed-miss shot: the missile runs the clock out
        # (MslCfg.MAX_FLIGHT) before it can close the gap. Verified against
        # missile_sim.AIM120 directly (sweep_envelope.py / see its docstring):
        # a fleet of episodes firing at the old 1.15x boundary scored 0/20
        # kills even against a non-manoeuvring, unarmed opponent, every miss
        # cause=KINEMATIC. Gate at r_max itself; let the reward shaping (the
        # r_over_rnez_own potential term) teach the policy WHERE inside that
        # window (documented target: r/r_max in 0.6-0.85) to actually pull
        # the trigger.
        return 1500.0<r<=r_max

    # ── sub-step loop ─────────────────────────────────────────────────
    def _advance(self, duration:float, cmd:dict, fire:bool) -> None:
        sim_dt      = self._world.SIM_DT
        n_frames    = max(1,int(round(duration/sim_dt)))
        fire_frames = 1 if fire else 0

        for k in range(n_frames):
            pkt = dict(cmd)
            pkt["fire"]         = 1 if k<fire_frames else 0
            pkt["msl_guidance"] = self._guidance_packet()

            try:    opp = self._opponent.act(self._state, self._t_sim)
            except Exception as e:
                print(f"[bvr_env] opponent: {e}"); opp={}

            tlm = self._world.step(pkt, opp)
            self._ingest(tlm)
            self._track.tick(sim_dt)
            if (self._track.t_now - self._last_convert_t) >= (1.0/self.TRACK_CONVERT_HZ):
                conv_dt = min(self._track.t_now - self._last_convert_t, 1.0)
                self._last_convert_t = self._track.t_now
                self._radar_update(dt=conv_dt)
            if self._check_terminal(): return

    def _ingest(self, pkt:dict) -> None:
        if pkt.get("episode_id") not in (None,self._episode_id): return
        self._state.update(pkt)
        self._t_sim = float(pkt.get("t_sim", self._t_sim+self._world.SIM_DT))
        for ev in pkt.get("events",[]) or []:
            key=(ev.get("type"),ev.get("id"),round(float(ev.get("t_sim",0.0)),3))
            if key in self._events_seen: continue
            self._events_seen.add(key)
            self._on_event(ev)

    def _on_event(self, ev:dict) -> None:
        t=ev.get("type")
        if t=="MISSILE_LAUNCH" and ev.get("owner")==1:
            self._shots_fired+=1; self._last_shot_t=self._t_sim
            est=self._track.estimate(); r=self._est_range(est)
            r_max,r_nez=self._own_envelope(est)
            self._launch_log.append({
                "missile_id":ev.get("id"),"t_sim":round(self._t_sim,2),
                "range_est":round(r,1),"range_true":round(self._state.get("range",-1),1),
                "r_over_rmax":round(r/max(r_max,1),3),"r_over_rnez":round(r/max(r_nez,1),3),
                "aspect_est":round(self._est_aspect(est),1),
                "own_mach":round(self._state.get("mach",0),3),
                "own_alt":round(self._state.get("alt",0),0),
                "track_age":round(est["age"],2),"pos_sigma":round(est["pos_sigma"],1)})
        elif t=="MISSILE_MISS" and ev.get("owner")==1:
            self._misses+=1
            if ev.get("cause")=="SUPPORT_LOST": self._support_losses+=1
        elif t in ("AC_DESTROYED","MISSILE_HIT"):
            if t=="MISSILE_HIT" and not ev.get("killed",1): return
            victim = ev.get("ac") if t=="AC_DESTROYED" else ev.get("target")
            if   victim==2: self._outcome = "MUTUAL_KILL" if self._outcome=="SHOT_DOWN" else "KILL"
            elif victim==1: self._outcome = "MUTUAL_KILL" if self._outcome=="KILL" else "SHOT_DOWN"
        elif t=="AC_CRASHED":
            if   ev.get("ac")==1: self._outcome="CRASH"
            elif ev.get("ac")==2 and self._outcome is None: self._outcome="BANDIT_CRASH"

    def _check_terminal(self) -> bool:
        if self._outcome is not None: return True
        s=self._state
        if not s: return False
        if s.get("alt",9000)<self.MIN_ALT:    self._outcome="CRASH";  return True
        if s.get("range",0)>self.ESCAPE_RANGE: self._outcome="ESCAPE"; return True
        return False

    def _terminal_reward(self, outcome:str) -> float:
        base={"KILL":self.R_KILL,"SHOT_DOWN":self.R_KILLED,"MUTUAL_KILL":self.R_MUTUAL,
              "CRASH":self.R_CRASH,"BANDIT_CRASH":self.R_BANDIT_CRASH,
              "ESCAPE":self.R_ESCAPE,"TIMEOUT":self.R_TIMEOUT}.get(outcome,0.0)
        if outcome=="KILL":
            base+=self.R_WASTED_MSL*float(self._state.get("wpn_remaining",0))
        return float(base)

    # ── radar / track adapter ─────────────────────────────────────────
    def _own_pos_enu(self) -> np.ndarray:
        s=self._state
        return self._to_enu(s.get("lat",REF_LAT),s.get("lon",REF_LON),s.get("alt",9000))

    def _to_enu(self,lat,lon,alt) -> np.ndarray:
        x=(lon-REF_LON)*DEG2RAD*R_EARTH*math.cos(REF_LAT*DEG2RAD)
        y=(lat-REF_LAT)*DEG2RAD*R_EARTH
        return np.array([x,y,alt],dtype=np.float64)

    def _radar_update(self, dt:float=0.1) -> None:
        s=self._state
        if not s: return
        att   =(s.get("psi",0),s.get("theta",0),s.get("phi",0))
        omega = np.zeros(3) if self.RADAR_OMEGA_COMPENSATED else \
                np.array([s.get("p",0),s.get("q",0),s.get("r_body",0)])
        op,ov = self._own_pos_enu(), self._own_vel_enu()

        if self._radar is not None:
            tp = self._to_enu(s.get("lat_t",REF_LAT),s.get("lon_t",REF_LON),s.get("alt_t",9000))
            tv = self._target_vel_enu()
            rae,cov,valid = self._radar.sense(dt,op,att,ov,omega,tp,tv)
            if not valid: return
            omega_used = omega
        else:
            r=s.get("radar")
            if not isinstance(r,dict) or not r.get("valid",0): return
            rae=r.get("rae_state"); cov=r.get("rae_cov")
            if rae is None or cov is None: return
            rae=np.asarray(rae,dtype=np.float64); cov=np.asarray(cov,dtype=np.float64)
            omega_used = np.zeros(3) if self.RADAR_OMEGA_COMPENSATED else omega

        self._track.update(rae_state=rae,rae_cov=cov,
                           own_pos=op,own_att=att,own_vel=ov,
                           own_omega=omega_used,valid=True)

    def _own_vel_enu(self) -> np.ndarray:
        s=self._state; v=s.get("speed",280); p=s.get("psi",0); g=s.get("gamma_fpa",0)
        cg=math.cos(g)
        return np.array([v*cg*math.sin(p),v*cg*math.cos(p),v*math.sin(g)],dtype=np.float64)

    def _target_vel_enu(self) -> np.ndarray:
        s=self._state; v=s.get("speed_t",290); p=s.get("psi_t",0); g=s.get("gamma_fpa_t",0)
        cg=math.cos(g)
        return np.array([v*cg*math.sin(p),v*cg*math.cos(p),v*math.sin(g)],dtype=np.float64)

    def _est_range(self,est) -> float:
        if not est["valid"]: return RANGE_MAX
        return float(np.linalg.norm(est["pos"]-self._own_pos_enu()))

    def _est_aspect(self,est) -> float:
        if not est["valid"]: return 90.0
        return aspect_deg_from_vectors(self._own_pos_enu(),est["pos"],est["vel"])

    def _own_envelope(self,est) -> tuple:
        s=self._state; mach=s.get("mach",s.get("speed",280)/A_SOUND)
        tgt_mach=est["speed"]/A_SOUND if est["valid"] else 0.9
        return self._env_mdl.compute(mach,s.get("alt",9000),self._est_aspect(est),tgt_mach)

    def _threat_envelope(self,est) -> tuple:
        if not est["valid"]: return 60_000.0,25_000.0
        s=self._state; tgt_mach=est["speed"]/A_SOUND
        own_vel=self._own_vel_enu()
        our_asp=aspect_deg_from_vectors(est["pos"],self._own_pos_enu(),own_vel)
        own_mach=s.get("mach",s.get("speed",280)/A_SOUND)
        return self._env_mdl.compute(tgt_mach,float(est["pos"][2]),our_asp,own_mach)

    def _guidance_packet(self) -> dict:
        est=self._track.estimate()
        if not est["valid"] or self._track.state==TrackState.NONE:
            return {"valid":0}
        return {"valid":1,
                "tgt_pos":[round(float(v),1) for v in est["pos"]],
                "tgt_vel":[round(float(v),2) for v in est["vel"]],
                "pos_sigma":round(est["pos_sigma"],1),
                "t_est":round(self._t_sim-est["age"],3)}

    # ── observation ────────────────────────────────────────────────────
    def _missile_summary(self):
        own=thr=None; n=0
        for m in self._state.get("missiles",[]) or []:
            if m.get("state") in ("HIT","MISS","DUD"): continue
            if m.get("owner")==1:
                n+=1
                if own is None or m.get("tgo_est",999)<own.get("tgo_est",999): own=m
            elif m.get("owner")==2:
                if thr is None or m.get("tgo_est",999)<thr.get("tgo_est",999): thr=m
        return own,thr,n

    def _build_obs(self):
        s=self._state; est=self._track.estimate(); op=self._own_pos_enu()
        speed=s.get("speed",280); alt=s.get("alt",9000)
        mach=s.get("mach",speed/A_SOUND); phi=s.get("phi",0)
        e_own=alt+speed**2/(2*G)
        oh=[0.0]*4; oh[int(self._track.state)]=1.0

        if est["valid"]:
            d=est["pos"]-op; rng_e=float(np.linalg.norm(d))
            Rb=body_to_enu_matrix(s.get("psi",0),s.get("theta",0),0)
            db=Rb.T@d
            ata=math.atan2(db[1],max(db[0],1e-6))
            el=math.atan2(-db[2],max(math.hypot(db[0],db[1]),1e-6))
            rv=est["vel"]-self._own_vel_enu()
            rrt=float(np.dot(rv,d/max(rng_e,1)))
            asp=self._est_aspect(est)*DEG2RAD
            ts=est["speed"]; tar=float(est["pos"][2])-alt
            et=float(est["pos"][2])+ts**2/(2*G)
            lr=float(np.linalg.norm(np.cross(d,rv))/max(rng_e**2,1))*RAD2DEG
        else:
            rng_e=RANGE_MAX; rrt=0; ata=el=0; asp=math.pi/2; ts=tar=0; et=e_own; lr=0

        rm,rn=self._own_envelope(est); rmt,rnt=self._threat_envelope(est)
        om,tm,nom=self._missile_summary()
        rwr=s.get("rwr",{}) or {}; rw=float(rwr.get("launch_warn",0))
        rb=float(rwr.get("launch_bearing",0)); rr=_wrap_pi(rb-s.get("psi",0))

        obs=np.array([
            mach,alt,s.get("gamma_fpa",0),math.cos(phi),math.sin(phi),
            s.get("nz",1),e_own,s.get("alpha",0),
            float(s.get("wpn_remaining",0)),float(s.get("fuel_frac",1)),
            *oh,
            math.log10(max(rng_e,100)),rrt,
            math.cos(ata),math.sin(ata),el,
            math.cos(asp),math.sin(asp),
            est["age"],math.log10(max(est["pos_sigma"],1)),math.log10(max(est["vel_sigma"],1)),
            ts,tar,e_own-et,lr,
            rng_e/max(rm,1),rng_e/max(rn,1),rng_e/max(rmt,1),rng_e/max(rnt,1),
            float(nom),
            float(om.get("tgo_est",0)) if om else 0,
            float(om.get("t_flight",0)) if om else 0,
            float(om.get("needs_support",0)) if om else 0,
            float(om.get("seeker_active",0)) if om else 0,
            min(float(om.get("t_since_update",0)),5) if om else 0,
            rw,
            max(0,self._t_sim-float(rwr.get("t_warn",self._t_sim))) if rw else 0,
            math.cos(rr),math.sin(rr),float(rwr.get("seeker_active",0)),
            float(tm.get("tgo_est",0)) if tm else 0,
            self._step_num/float(self.MAX_STEPS),
            min(self._t_sim-self._last_shot_t,120) if self._last_shot_t>-900 else 120,
        ],dtype=np.float32)
        obs=self._norm(obs,self._obs_lo,self._obs_hi)
        if not self._privileged: return obs
        return {"obs":obs,"priv":self._build_priv(est,om,tm)}

    def _build_priv(self,est,om,tm):
        s=self._state
        aa=s.get("aa_deg",180)*DEG2RAD; aat=s.get("aa_deg_t",180)*DEG2RAD
        hca=s.get("hca_deg",0)*DEG2RAD
        alt=s.get("alt",9000); spd=s.get("speed",280)
        altt=s.get("alt_t",9000); spdt=s.get("speed_t",280)
        eo=alt+spd**2/(2*G); et=altt+spdt**2/(2*G)
        rng=s.get("range",RANGE_MAX)
        ov=self._own_vel_enu()
        tp=self._to_enu(s.get("lat_t",REF_LAT),s.get("lon_t",REF_LON),altt)
        ua=aspect_deg_from_vectors(tp,self._own_pos_enu(),ov)
        rmt,_=self._env_mdl.compute(spdt/A_SOUND,altt,ua,spd/A_SOUND)
        p=np.array([
            rng,s.get("closure",0),
            math.cos(aa),math.sin(aa),math.cos(aat),math.sin(aat),
            math.cos(hca),math.sin(hca),
            alt-altt,spdt,eo-et,
            float(s.get("wpn_remaining_t",0)),
            1.0 if (s.get("rwr",{}) or {}).get("spike",0) else 0.0,
            float(tm.get("tgo_est",0)) if tm else 0,
            float(om.get("tgo_est",0)) if om else 0,
            rng/max(rmt,1),
        ],dtype=np.float32)
        return self._norm(p,self._priv_lo,self._priv_hi)

    @staticmethod
    def _norm(x,lo,hi):
        return (2*(np.clip(x,lo,hi)-lo)/np.maximum(hi-lo,1e-9)-1).astype(np.float32)

    # ── potential ──────────────────────────────────────────────────────
    def _potential(self) -> float:
        s=self._state; est=self._track.estimate(); rng=self._est_range(est)
        rm,rn=self._own_envelope(est); rmt,rnt=self._threat_envelope(est)
        pe=self.W_ENVELOPE*(_band(rng,rn,rm)-_band(rng,rnt,rmt))
        if   self._track.state==TrackState.TRACK:     t=1.0
        elif self._track.state==TrackState.COAST:     t=0.4*math.exp(-est["age"]/8)
        elif self._track.state==TrackState.ACQUIRING: t=0.5
        else: t=0.0
        pt=self.W_TRACK*t
        eo=s.get("alt",9000)+s.get("speed",280)**2/(2*G)
        et=(float(est["pos"][2])+est["speed"]**2/(2*G)) if est["valid"] else eo
        pe2=self.W_ENERGY*math.tanh((eo-et)/4000)
        om,tm,_=self._missile_summary()
        ps=0.0
        if om and om.get("needs_support",0) and self._track.state==TrackState.TRACK:
            ps=self.W_SUPPORT*math.exp(-est["pos_sigma"]/600)
        pd=0.0
        if tm: pd=self.W_DEFENCE*math.tanh(float(tm.get("tgo_est",60))/25)
        # Recorded from the very variables summed below, so the breakdown can
        # never drift from the potential it explains.
        self._phi_terms={"envelope":float(pe),"track":float(pt),
                         "energy":float(pe2),"support":float(ps),
                         "defence":float(pd)}
        return float(pe+pt+pe2+ps+pd)

    # ── action → command ───────────────────────────────────────────────
    def _encode_cmd(self,ih,ia,isp,ifire) -> dict:
        s=self._state; est=self._track.estimate()
        if est["valid"]:
            d=est["pos"]-self._own_pos_enu(); base=math.atan2(d[0],d[1])
        else: base=s.get("psi",0)
        self._cmd_hdg=_wrap_2pi(base+HDG_OFFSETS_DEG[ih]*DEG2RAD)
        self._cmd_alt=float(np.clip(s.get("alt",9000)+ALT_DELTAS_M[ia],ALT_MIN_OP,ALT_MAX_OP))
        self._cmd_spd=float(SPEED_CMDS[isp])
        return {"mode":0,"chiDot":0,"gamma":0,
                "V":self._cmd_spd,"hdgCmd":self._cmd_hdg,"altTarget":self._cmd_alt,
                "altFPA":25*DEG2RAD,"hdgTurnRate":12*DEG2RAD,
                "maneuver":"NONE","task":"NONE","radar_cmd":1,"fire":0}

    # ── IC generator ───────────────────────────────────────────────────
    _SCENARIOS=["head_on","offset_left","offset_right","beam","stern_conversion"]

    def _random_ic(self) -> dict:
        sc=str(self._rng.choice(self._SCENARIOS))
        rm=float(self._rng.uniform(70000,110000))
        br=float(self._rng.uniform(0,2*math.pi))
        a1a=float(self._rng.uniform(6000,11000))
        a2a=float(np.clip(a1a+self._rng.uniform(-2500,2500),4000,12500))
        a1s=float(self._rng.uniform(250,320)); a2s=float(self._rng.uniform(250,320))
        a1p=br
        if   sc=="head_on":      a2p=br+math.pi+float(self._rng.uniform(-0.15,0.15))
        elif sc=="offset_left":  a2p=br+math.pi+float(self._rng.uniform(0.25,0.70))
        elif sc=="offset_right": a2p=br+math.pi-float(self._rng.uniform(0.25,0.70))
        elif sc=="beam":         a2p=br+math.pi/2+float(self._rng.uniform(-0.30,0.30))
        else:                    a2p=br+float(self._rng.uniform(-0.30,0.30)); rm=float(self._rng.uniform(45000,75000))
        dl=rm*math.cos(br)/R_EARTH; dlo=rm*math.sin(br)/(R_EARTH*math.cos(REF_LAT*DEG2RAD))
        return dict(scenario=sc,
                    ac1_lat=REF_LAT,ac1_lon=REF_LON,ac1_alt=a1a,ac1_psi=_wrap_2pi(a1p),ac1_spd=a1s,
                    ac2_lat=REF_LAT+math.degrees(dl),ac2_lon=REF_LON+math.degrees(dlo),
                    ac2_alt=a2a,ac2_psi=_wrap_2pi(a2p),ac2_spd=a2s,
                    wpn=4,wpn_t=4,start_range=rm)

    def close(self): pass


def _wrap_pi(a):  return (a+math.pi)%(2*math.pi)-math.pi
def _wrap_2pi(a): return a%(2*math.pi)
def _band(r,rn,rm):
    if r<=rn: return 1.0
    if r>=rm: return 0.0
    return float((rm-r)/max(rm-rn,1.0))
