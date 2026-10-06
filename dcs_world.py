"""
dcs_world.py  —  DCS World recordings as a BVR_RL world
========================================================

Step 1 of running a trained policy in DCS World: replay an engagement that
dcs/bvr_logger.lua recorded through the environment's own observation code,
to see the inputs a policy would get in DCS. Nothing is sent back to DCS.

    rec = DcsRecording("bvr_rl_1234_567890.jsonl")
    env = DcsReplayEnv(rec, blue="Viper-1", red="Flanker-1")
    obs, _ = env.reset()
    while True:
        obs, r, term, trunc, info = env.step()   # action inferred from what DCS flew
        if term or trunc:
            break

DcsReplayWorld stands in for SimWorld. It answers the calls BvrEnv makes
(reset, step, missiles_pending, SIM_DT) with telemetry packets built from
the recording, and ignores the commands. BvrEnv's radar model, track
adapter, envelopes and observation code run unchanged on the recorded
truth, so the inputs differ from training only where DCS differs from the
simulator. 1v1 only for now: one blue and one red aircraft are replayed,
other aircraft in the recording are ignored.

What DCS does not report is estimated:
    angle of attack           pitch minus flight-path angle
    body rates, load factor   finite differences of recorded attitude, velocity
    missile seeker active     inside the platform missile's mean hand-off
                              range of its target (latched)
    missile time-to-go        range to target / closing speed, from truth
    weapons remaining         radar-guided air-to-air missiles in the last
                              ammo report; without one, loadout minus shots

Coordinates: DCS map metres (x north, z east, y up) are shifted so the
fight's first sample sits on the simulator's reference point. The env turns
lat/lon back into the same metres, so ranges and bearings are exact.

write_sim_recording() writes a simulator episode in the logger's format, so
the whole path can be checked without DCS: a replay of it must reproduce
the simulator's inputs (test_sim.test_dcs_round_trip).

Live (dcs_live.py): DcsLiveRecording grows as dcs/bvr_bridge.lua streams the
same lines over UDP, and dcs_live.DcsLiveWorld steps through it as DCS flies.
sim_frame / sim_events / sim_ammo turn a SimWorld into those lines
(dcs/fake_dcs.py).
"""

import json
import math

import numpy as np

from bvr_env import (BvrEnv, HDG_OFFSETS_DEG, ALT_DELTAS_M, DEG2RAD, RAD2DEG, G,
                     _wrap_pi, _wrap_deg)
from bvr_library import DEFAULT_PLATFORM, rcs_scale
from bvr_opponents import BvrOpponentType
from bvr_track_adapter import body_to_enu_matrix
from missile_sim import _isa_a
from sim_world import _enu_to_latlon

FORMAT = "bvr_rl.dcs.v1"
COAL_RED, COAL_BLUE = 1, 2          # DCS coalition.side numbers

# DCS Weapon.GuidanceType: the missiles counted as "weapons remaining".
RADAR_GUIDANCE = (3, 4)             # RADAR_ACTIVE, RADAR_SEMI_ACTIVE


def _rotation_vector(M) -> np.ndarray:
    """Axis * angle of rotation matrix M (the log map), exact for a constant rate."""
    c = float(np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(c)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    s = math.sin(ang)
    return 0.5 * v if s < 1e-6 else v * (ang / (2.0 * s))


def _dcs_to_enu(x, y, z):
    """DCS map axes (x north, y up, z east) -> ENU."""
    return np.array([z, x, y], dtype=np.float64)


# ────────────────────────────────────────────────────────────────────
class DcsRecording:
    """A bvr_logger.lua file: aircraft and missile tracks plus events, in ENU metres."""

    # A unit missing from the samples this long, without a death event, has
    # left the fight. 0 for a finished file; the live recording waits a little,
    # so one sample that lacks a unit doesn't end its fight.
    VANISH_S = 0.0

    def __init__(self, source):
        if isinstance(source, (list, tuple)):
            lines = list(source)
        else:
            with open(source, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        self._clear()
        for n, ln in enumerate(lines, start=1):
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError as e:
                # The last line of a file DCS was still writing can be cut short.
                if n == len(lines):
                    break
                raise ValueError(f"line {n}: {e}") from e
            self._add(d)
        if not self._frames:
            raise ValueError("the recording has no samples")
        self._build()

    def _clear(self):
        self.header, self._frames, self.events = {}, [], []
        self._x0 = self._z0 = None
        self._t_first = None

    def _add(self, d):
        """Take one parsed line: the header, an event or a sample."""
        if "format" in d:
            if d["format"] != FORMAT:
                raise ValueError(f"unknown recording format {d['format']!r}")
            # The live bridge repeats its header; the first one stands.
            self.header = self.header or d
        elif "ev" in d:
            self.events.append(d)
        elif "units" in d:
            self._frames.append(d)

    def _build(self):
        """Turn the lines taken so far into the arrays the queries read."""
        frames = self._frames
        frames.sort(key=lambda fr: fr["t"])
        self.events.sort(key=lambda e: e["t"])
        self.rate = float(self.header.get("rate", 0.1))
        if self._t_first is None:
            self._t_first = float(frames[0]["t"])
        self.t_start, self.t_end = self._t_first, float(frames[-1]["t"])

        # Origin: the middle of the first sample's aircraft, horizontally.
        if self._x0 is None:
            first = frames[0]["units"]
            self._x0 = float(np.mean([u["x"] for u in first])) if first else 0.0
            self._z0 = float(np.mean([u["z"] for u in first])) if first else 0.0

        self.units = {}
        for fr in frames:
            t = float(fr["t"])
            for u in fr["units"]:
                d = self.units.setdefault(u["name"], {
                    "coal": int(u.get("coal", 0)), "type": u.get("type", "?"),
                    "t": [], "pos": [], "vel": [], "fwd": [], "up": [], "fuel": []})
                d["t"].append(t)
                d["pos"].append(self._enu(u["x"], u["y"], u["z"]))
                d["vel"].append(_dcs_to_enu(u["vx"], u["vy"], u["vz"]))
                d["fwd"].append(_dcs_to_enu(u["fx"], u["fy"], u["fz"]))
                d["up"].append(_dcs_to_enu(u["ux"], u["uy"], u["uz"]))
                d["fuel"].append(float(u.get("fuel", 1.0)))
        for d in self.units.values():
            for k in ("t", "pos", "vel", "fwd", "up", "fuel"):
                d[k] = np.asarray(d[k], dtype=np.float64)

        self.weapons = {}
        for fr in frames:
            t = float(fr["t"])
            for w in fr.get("weapons", []):
                d = self._weapon(w["id"])
                d["type"] = w.get("type", d["type"])
                d["shooter"] = d["shooter"] or w.get("shooter")
                if w.get("target"):
                    d["target"] = w["target"]
                d["t"].append(t)
                d["pos"].append(self._enu(w["x"], w["y"], w["z"]))
                d["vel"].append(_dcs_to_enu(w["vx"], w["vy"], w["vz"]))

        self.dead, self.ammo, self.hits = {}, {}, []
        for e in self.events:
            kind, t = e["ev"], float(e["t"])
            if kind == "shot":
                d = self._weapon(e["id"])
                d["t_shot"] = t
                d["shooter"] = e.get("shooter") or d["shooter"]
                d["target"] = d["target"] or e.get("target")
                d["type"] = e.get("type", d["type"])
            elif kind == "weapon_gone":
                self._weapon(e["id"])["t_gone"] = t
            elif kind == "support_lost":
                # dcs_live.py had the bridge remove it: no support for too long.
                if e.get("status") == "destroyed":
                    self._weapon(e["id"])["support_lost"] = t
            elif kind == "hit":
                self.hits.append(e)
                if int(e.get("id", -1)) >= 0 and e.get("target"):
                    d = self._weapon(e["id"])
                    if d["t_hit"] is None:
                        d["t_hit"], d["hit_target"] = t, e["target"]
            elif kind == "kill":
                # S_EVENT_KILL: the moment DCS decides the unit is destroyed, with
                # the killer and weapon (its dead/crash event can come much later,
                # when the wreck reaches the ground).
                if e.get("unit"):
                    self.dead.setdefault(e["unit"], (t, "killed"))
                    self.hits.append({"t": t, "target": e["unit"], "shooter": e.get("killer"),
                                      "id": e.get("id", -1)})
                    if int(e.get("id", -1)) >= 0:
                        d = self._weapon(e["id"])
                        if d["t_hit"] is None:
                            d["t_hit"], d["hit_target"] = t, e["unit"]
            elif kind == "dead":
                self.dead.setdefault(e["unit"], (t, e.get("cause", "dead")))
            elif kind == "ammo":
                n = sum(int(i.get("count", 0)) for i in e.get("items", [])
                        if i.get("aam") and int(i.get("guidance", 0)) in RADAR_GUIDANCE)
                self.ammo.setdefault(e["unit"], []).append((t, n))

        for d in self.weapons.values():
            d["t"] = np.asarray(d["t"], dtype=np.float64)
            for k in ("pos", "vel"):
                d[k] = np.asarray(d[k], dtype=np.float64).reshape(len(d[k]), 3)
            if d["t_shot"] is None:
                d["t_shot"] = float(d["t"][0]) if len(d["t"]) else None
            end = d["t_gone"] if d["t_gone"] is not None else (
                float(d["t"][-1]) + self.rate if len(d["t"]) else d["t_shot"])
            if d["t_hit"] is not None:
                end = min(end, d["t_hit"])
            d["t_end"] = end

        # A missile hit is a kill, as in the simulator, from the moment of the
        # hit: DCS may report the death only when the wreck reaches the ground,
        # and meanwhile the falling wreck would read as a crash (or, if it was
        # the bandit, as an aircraft still in the fight).
        for d in self.weapons.values():
            if d["hit_target"] and d["t_hit"] is not None:
                cur = self.dead.get(d["hit_target"])
                if cur is None or d["t_hit"] < cur[0]:
                    self.dead[d["hit_target"]] = (d["t_hit"], "missile hit")

        # A unit that stops appearing without a death event left the fight then
        # (DCS removes a destroyed aircraft, and may report its death only when
        # the wreck reaches the ground).
        for name, d in self.units.items():
            if name not in self.dead and d["t"][-1] < self.t_end - max(self.VANISH_S, 1e-6):
                self.dead[name] = (float(d["t"][-1]) + self.rate, "vanished")

    def _enu(self, x, y, z):
        return np.array([z - self._z0, x - self._x0, y], dtype=np.float64)

    def _weapon(self, wid):
        return self.weapons.setdefault(int(wid), {
            "id": int(wid), "type": "?", "shooter": None, "target": None,
            "t": [], "pos": [], "vel": [], "t_shot": None, "t_gone": None,
            "t_hit": None, "hit_target": None, "t_end": None, "support_lost": None})

    # ── queries ──────────────────────────────────────────────────────
    def death_cause(self, name) -> dict:
        """How a dead unit died: {"t", "cause" (DCS's word), "weapon" (the last
        missile that hit it before its death, or None), "hit" (anything hit
        it before its death)}. A hit followed by a death, however long after,
        is a kill: DCS reports the death when the wreck lands or is removed."""
        t, cause = self.dead[name]
        hits = [d for d in self.weapons.values()
                if d["hit_target"] == name and d["t_hit"] is not None and d["t_hit"] <= t + 0.5]
        weapon = max(hits, key=lambda d: d["t_hit"]) if hits else None
        hit = weapon is not None or any(h.get("target") == name and float(h["t"]) <= t + 0.5
                                        for h in self.hits)
        return {"t": t, "cause": cause, "weapon": weapon, "hit": hit}

    def names(self, coal=None):
        return [n for n, d in self.units.items() if coal is None or d["coal"] == coal]

    def alive(self, name, t) -> bool:
        return name not in self.dead or t < self.dead[name][0]

    def state(self, name, t) -> dict:
        """Interpolated pos, vel, fwd, up (orthonormal) and fuel; held at the ends."""
        d = self.units[name]
        ts = d["t"]
        out = {}
        for k in ("pos", "vel", "fwd", "up"):
            a = d[k]
            out[k] = np.array([np.interp(t, ts, a[:, j]) for j in range(3)])
        out["fuel"] = float(np.interp(t, ts, d["fuel"]))
        f = out["fwd"] / max(np.linalg.norm(out["fwd"]), 1e-9)
        u = out["up"] - np.dot(out["up"], f) * f
        out["fwd"], out["up"] = f, u / max(np.linalg.norm(u), 1e-9)
        return out

    def weapon_state(self, wid, t) -> dict:
        d = self.weapons[wid]
        if not len(d["t"]):
            return None
        ts = d["t"]
        return {k: np.array([np.interp(t, ts, d[k][:, j]) for j in range(3)])
                for k in ("pos", "vel")}

    def radar_missiles(self, name, t):
        """Radar-guided AAMs in the unit's last ammo report at or before t; None if never reported."""
        rows = [n for (te, n) in self.ammo.get(name, []) if te <= t + 1e-9]
        return rows[-1] if rows else None


class DcsLiveRecording(DcsRecording):
    """A recording that grows while DCS runs (dcs_live.py): extend() it with
    parsed lines as they arrive. Only the last WINDOW_S seconds of samples
    are kept, which is all the queries need near the newest time; events
    are all kept."""

    WINDOW_S = 30.0
    VANISH_S = 1.0

    def __init__(self):
        self._clear()
        self.units, self.weapons, self.dead, self.ammo, self.hits = {}, {}, {}, {}, []
        self.rate, self.t_start, self.t_end = 0.1, None, None

    @property
    def ready(self) -> bool:
        return bool(self._frames)

    def extend(self, objs, reader_t=None) -> None:
        """Add parsed lines. reader_t: the earliest time still to be read; samples
        are kept from WINDOW_S before it (or before the newest, without it)."""
        n = len(self._frames)
        for d in objs:
            self._add(d)
        if not self._frames:
            return
        if len(self._frames) > n:
            newest = max(float(fr["t"]) for fr in self._frames[n:])
            ref = newest if reader_t is None else min(newest, float(reader_t))
            keep = ref - self.WINDOW_S
            if float(self._frames[0]["t"]) < keep:
                self._frames = [fr for fr in self._frames if float(fr["t"]) >= keep]
        elif self.t_end is None:
            return
        self._build()


# ────────────────────────────────────────────────────────────────────
class DcsReplayWorld:
    """SimWorld's 1v1 interface, answered from a DcsRecording. Commands are ignored."""

    SIM_DT = 0.02

    def __init__(self, rec: DcsRecording, blue: str, red: str,
                 blue_platform=None, red_platform=None, t_start=None, t_end=None):
        for n in (blue, red):
            if n not in rec.units:
                raise ValueError(f"no aircraft named {n!r} in the recording; "
                                 f"it has {sorted(rec.units)}")
        self.rec = rec
        self.names = {1: blue, 2: red}
        self.labels = {blue: 1, red: 2}
        self.plats = {blue: blue_platform, red: red_platform}
        self.t0 = rec.t_start if t_start is None else float(t_start)
        self.t_end = rec.t_end if t_end is None else min(float(t_end), rec.t_end)
        self.fd_dt = max(rec.rate, 0.05)      # finite-difference step for rates
        self.reset()

    # ── SimWorld interface ───────────────────────────────────────────
    def reset(self, ic=None, episode_id: int = 0) -> None:
        self.episode_id = int(episode_id)
        self.t = self.t0
        self._t_prev = self.t0 - 1e-9
        self._seeker = set()
        self._deaths_done = set()
        self.events = []

    @property
    def t_sim(self) -> float:
        return self.t - self.t0

    @property
    def finished(self) -> bool:
        return self.t >= self.t_end - 1e-9

    @property
    def alive(self) -> list:
        return [self.rec.alive(self.names[i], self.t) for i in (1, 2)]

    def step(self, cmd1: dict, cmd2: dict) -> dict:
        self.t = min(self.t + self.SIM_DT, self.t_end)
        self.events = self._events_between(self._t_prev, self.t)
        self._t_prev = self.t
        return self.telemetry()

    def missiles_pending(self) -> bool:
        return any(self.rec.alive(tgt, self.t) for _, tgt, _ in self._in_flight())

    def initial_ic(self) -> dict:
        ic = {}
        for i in (1, 2):
            k = self._kin(self.names[i], self.t0)
            lat, lon, alt = _enu_to_latlon(*k["pos"])
            ic.update({f"ac{i}_lat": lat, f"ac{i}_lon": lon, f"ac{i}_alt": alt,
                       f"ac{i}_psi": k["chi"], f"ac{i}_spd": k["V"],
                       f"wpn{i}": self._wpn(self.names[i], self.t0)})
        ic["scenario"] = "dcs"
        ic["start_range"] = float(np.linalg.norm(self._kin(self.names[2], self.t0)["pos"]
                                                 - self._kin(self.names[1], self.t0)["pos"]))
        return ic

    # ── kinematics from the recording ────────────────────────────────
    def _kin(self, name, t) -> dict:
        rec, h = self.rec, self.fd_dt
        s = rec.state(name, t)
        # Central differences, one-sided at the ends of the unit's track.
        ts = rec.units[name]["t"]
        ta, tb = max(t - h, ts[0]), min(t + h, ts[-1])
        if tb - ta < 1e-6:
            ta, tb = t - h, t + h
        s0, s1 = rec.state(name, ta), rec.state(name, tb)
        dt2 = tb - ta
        v = s["vel"]
        V = max(float(np.linalg.norm(v)), 1.0)
        fwd, up = s["fwd"], s["up"]
        right = np.cross(fwd, up)
        R = np.column_stack([fwd, right, -up])            # body (FRD) -> ENU
        R0 = np.column_stack([s0["fwd"], np.cross(s0["fwd"], s0["up"]), -s0["up"]])
        R1 = np.column_stack([s1["fwd"], np.cross(s1["fwd"], s1["up"]), -s1["up"]])
        w_body = _rotation_vector(R0.T @ R1) / dt2         # body rates, rad/s
        acc = (s1["vel"] - s0["vel"]) / dt2
        theta = math.asin(float(np.clip(fwd[2], -1, 1)))
        gamma = math.asin(float(np.clip(v[2] / V, -1, 1)))
        return {
            "pos": s["pos"], "vel": v, "V": V,
            "chi": math.atan2(v[0], v[1]) % (2 * math.pi),
            "gamma": gamma, "theta": theta,
            "phi": math.atan2(-right[2], up[2]),
            "alpha": theta - gamma,
            "p": float(w_body[0]), "q": float(w_body[1]), "r": float(w_body[2]),
            "nz": float(np.dot(acc + np.array([0.0, 0.0, G]), up) / G),
            "mach": V / _isa_a(float(s["pos"][2])),
            "fuel": float(np.clip(s["fuel"], 0.0, 1.0)),
        }

    def _wpn(self, name, t) -> int:
        n = self.rec.radar_missiles(name, t)
        if n is not None:
            return int(n)
        plat = self.plats.get(name)
        full = plat.wpn_count if plat is not None else 4
        shots = sum(1 for d in self.rec.weapons.values()
                    if d["shooter"] == name and d["t_shot"] is not None and d["t_shot"] <= t)
        return max(full - shots, 0)

    # ── missiles ─────────────────────────────────────────────────────
    def _target_of(self, d) -> str:
        """The weapon's target; unknown ones are taken to be aimed at the other side of the pair."""
        if d["target"] in self.labels:
            return d["target"]
        if d["target"] is None and d["shooter"] in self.labels:
            return self.names[3 - self.labels[d["shooter"]]]
        return d["target"]

    def _in_flight(self):
        """(weapon dict, target name, owner label) for replayed missiles in flight at t."""
        out = []
        for d in self.rec.weapons.values():
            if d["shooter"] not in self.labels or d["t_shot"] is None:
                continue
            if not (d["t_shot"] <= self.t < d["t_end"]):
                continue
            tgt = self._target_of(d)
            if tgt not in self.labels or tgt == d["shooter"]:
                continue
            out.append((d, tgt, self.labels[d["shooter"]]))
        return out

    def _handoff(self, shooter, target) -> float:
        p = self.plats.get(shooter)
        if p is None:
            return 12_000.0
        m = p.missile
        pt = self.plats.get(target)
        scale = rcs_scale(pt.rcs, m.SEEKER_REF_RCS_M2) if pt is not None else 1.0
        return 0.5 * (m.HANDOFF_MIN + m.HANDOFF_MAX) * scale

    def _missile_dict(self, d, tgt, owner):
        t = self.t
        ws = self.rec.weapon_state(d["id"], t)
        if ws is None:
            return None
        pos, vel = ws["pos"], ws["vel"]
        tk = self._kin(tgt, t)
        los = tk["pos"] - pos
        rng = float(np.linalg.norm(los))
        closing = float(np.dot(vel - tk["vel"], los / max(rng, 1.0)))
        tgo = rng / closing if closing > 10.0 else 999.0
        handoff = self._handoff(d["shooter"], tgt)
        if rng < handoff:
            self._seeker.add(d["id"])
        seeker = d["id"] in self._seeker
        t_flight = t - d["t_shot"]
        shooter_dead = self.rec.dead.get(d["shooter"])
        tsu = 0.0 if (shooter_dead is None or t < shooter_dead[0]) else t - shooter_dead[0]
        spd = float(np.linalg.norm(vel))
        lat, lon, alt = _enu_to_latlon(*pos)
        return {
            "id": d["id"], "owner": owner, "target": self.labels[tgt],
            "state": "TERMINAL" if seeker else ("BOOST" if t_flight < 6.0 else "MIDCOURSE"),
            "lat": round(lat, 6), "lon": round(lon, 6), "alt": alt,
            "spd": round(spd, 1), "mach": round(spd / _isa_a(alt), 3),
            "t_flight": round(t_flight, 3), "tgo_est": round(tgo, 2),
            "seeker_active": int(seeker), "needs_support": int(not seeker),
            "t_since_update": round(tsu, 3), "handoff_range": round(handoff, 0),
            "guidance_valid": int(tsu == 0.0), "rng_to_target": round(rng, 1),
            "vel": [round(float(x), 1) for x in vel],
        }

    # ── events ───────────────────────────────────────────────────────
    def _events_between(self, ta, tb) -> list:
        """SimWorld events for recording times in (ta, tb], relabelled 1 = blue, 2 = red."""
        rec, out = self.rec, []
        rel = lambda t: round(t - self.t0, 3)
        for d in rec.weapons.values():
            owner = self.labels.get(d["shooter"])
            if owner is None or d["t_shot"] is None:
                continue
            tgt = self._target_of(d)
            if ta < d["t_shot"] <= tb:
                out.append({"type": "MISSILE_LAUNCH", "id": d["id"], "owner": owner,
                            "t_sim": rel(d["t_shot"])})
            if d["t_end"] is not None and ta < d["t_end"] <= tb:
                if d["t_hit"] is not None and d["hit_target"] in self.labels:
                    pass                # a hit: the target's death, below, decides what it did
                elif tgt in self.labels:
                    out.append({"type": "MISSILE_MISS", "id": d["id"], "owner": owner,
                                "cause": "SUPPORT_LOST" if d["support_lost"] is not None else "DCS",
                                "t_sim": rel(d["t_end"])})
        # One event per death, as the simulator's reach the env (BvrEnv would
        # count a second event for the same death as a new kill), sent once,
        # when the death is known: DCS can report it well after the hit, and
        # live, a unit's death can become known after its time has passed.
        for name, lab in self.labels.items():
            dead = rec.dead.get(name)
            if dead is None or dead[0] > tb or name in self._deaths_done:
                continue
            self._deaths_done.add(name)
            how = rec.death_cause(name)
            if how["weapon"] is not None and how["weapon"]["shooter"] in self.labels:
                d = how["weapon"]
                out.append({"type": "MISSILE_HIT", "id": d["id"], "owner": self.labels[d["shooter"]],
                            "target": lab, "killed": 1, "t_sim": rel(dead[0])})
            else:
                out.append({"type": "AC_DESTROYED" if how["hit"] else "AC_CRASHED", "ac": lab,
                            "cause": dead[1], "t_sim": rel(dead[0])})
        return out

    # ── telemetry packet ─────────────────────────────────────────────
    def telemetry(self, me: int = 1) -> dict:
        """The packet aircraft `me` gets (1 = blue, the default; 2 = red, for red's
        radar in dcs_live.py): its own state, the other's as *_t, and missiles
        labelled from its side (1 = its own, 2 = aimed at it)."""
        t = self.t
        blue, red = self.names[me], self.names[3 - me]          # "blue" is `me` below
        a1, a2 = self._kin(blue, t), self._kin(red, t)
        lat1, lon1, alt1 = _enu_to_latlon(*a1["pos"])
        lat2, lon2, alt2 = _enu_to_latlon(*a2["pos"])

        d = a2["pos"] - a1["pos"]
        rng = max(float(np.linalg.norm(d)), 1.0)
        los_u = d / rng
        v1, v2 = a1["vel"], a2["vel"]
        closure = float(-np.dot(v2 - v1, los_u))
        fwd1 = np.array([math.sin(a1["chi"]), math.cos(a1["chi"]), 0.0])
        fwd2 = np.array([math.sin(a2["chi"]), math.cos(a2["chi"]), 0.0])
        aa_deg = math.degrees(math.acos(float(np.clip(np.dot(fwd1, los_u), -1, 1))))
        aa_deg_t = math.degrees(math.acos(float(np.clip(np.dot(fwd2, -los_u), -1, 1))))
        hca_deg = math.degrees(math.acos(float(np.clip(np.dot(fwd1, fwd2), -1, 1))))
        los_rate = float(np.linalg.norm(np.cross(los_u, v2 - v1))) * RAD2DEG

        msl, inbound = [], []
        for dw, tgt, owner in self._in_flight():
            m = self._missile_dict(dw, tgt, owner)
            if m is None:
                continue
            if me != 1:                                   # relabel from red's side
                m["owner"] = 1 if owner == me else 2
                m["target"] = 1 if tgt == blue else 2
            msl.append(m)
            if owner != me and tgt == blue:
                inbound.append((dw, m))
        if inbound:
            dw, m = min(inbound, key=lambda x: float(np.linalg.norm(
                self.rec.weapon_state(x[0]["id"], t)["pos"] - a1["pos"])))
            dp = self.rec.weapon_state(dw["id"], t)["pos"] - a1["pos"]
            rwr = {"launch_warn": 1, "launch_bearing": round(math.atan2(dp[0], dp[1]), 4),
                   "t_warn": round(dw["t_shot"] - self.t0, 3),
                   "spike": int(float(np.linalg.norm(dp)) < 25_000.0),
                   "seeker_active": m["seeker_active"], "n_threats": len(inbound)}
        else:
            rwr = {"launch_warn": 0, "launch_bearing": 0.0, "t_warn": 0.0,
                   "spike": 0, "seeker_active": 0, "n_threats": 0}

        return {
            "type": "telemetry", "episode_id": self.episode_id, "t_sim": round(self.t_sim, 4),
            "lat": lat1, "lon": lon1, "alt": alt1,
            "psi": a1["chi"], "theta": a1["theta"], "phi": a1["phi"],
            "alpha": a1["alpha"], "beta": 0.0,
            "p": a1["p"], "q": a1["q"], "r_body": a1["r"],
            "nz": a1["nz"], "speed": a1["V"], "mach": a1["mach"],
            "gamma_fpa": a1["gamma"], "fuel_frac": a1["fuel"], "maneuver_running": False,
            "lat_t": lat2, "lon_t": lon2, "alt_t": alt2,
            "psi_t": a2["chi"], "theta_t": a2["theta"], "phi_t": a2["phi"],
            "speed_t": a2["V"], "mach_t": a2["mach"], "nz_t": a2["nz"],
            "gamma_fpa_t": a2["gamma"],
            "range": round(rng, 1), "closure": round(closure, 2),
            "aa_deg": round(aa_deg, 2), "aa_deg_t": round(aa_deg_t, 2),
            "hca_deg": round(hca_deg, 2), "los_rate": round(los_rate, 4),
            "ata_deg": round(aa_deg, 2),
            "wpn_remaining": self._wpn(blue, t), "wpn_remaining_t": self._wpn(red, t),
            "wpn_ready": 1, "wpn_ready_t": 1,
            "alive": int(self.rec.alive(blue, t)), "alive_t": int(self.rec.alive(red, t)),
            "missiles": msl, "rwr": rwr, "events": list(self.events),
        }


# ────────────────────────────────────────────────────────────────────
class DcsReplayEnv(BvrEnv):
    """BvrEnv on a DCS recording: the observation code runs unchanged, the world is the replay.

    step() without an action infers one from what the recorded blue aircraft
    flew next (infer_action), so the inputs that echo the last action (the
    previous heading choice, the turn still to fly) look as they would have
    if the policy had flown the same path.
    """

    # No altitude floor: in training, below 300 m counts as hitting the ground.
    # In DCS an aircraft can fly lower (the AI dives to the sea to defend) and
    # DCS itself reports a real crash, which counts as one.
    MIN_ALT = -1e9

    INFER_HDG_S = 2.0      # heading choice: course this many seconds ahead
    INFER_ALT_S = 8.0      # altitude choice: height change over this many seconds
    INFER_SPD_S = 5.0      # speed choice: speed this many seconds ahead

    def __init__(self, recording, blue=None, red=None, platform=DEFAULT_PLATFORM,
                 opponent_platform=DEFAULT_PLATFORM, seed=0, t_start=None, t_end=None,
                 max_steps=None, **kw):
        rec = recording if isinstance(recording, DcsRecording) else DcsRecording(recording)
        blue = blue or next(iter(rec.names(COAL_BLUE)), None)
        red = red or next(iter(rec.names(COAL_RED)), None)
        if blue is None or red is None:
            raise ValueError(f"need a blue and a red aircraft; the recording has {sorted(rec.units)}")
        super().__init__(opponent_type=BvrOpponentType.STRAIGHT, seed=seed, platform=platform,
                         opponent_platform=opponent_platform, **kw)
        if max_steps is not None:
            self.MAX_STEPS = int(max_steps)
        self._rec = rec
        self._world = DcsReplayWorld(rec, blue, red, self._plat, self._opp_plat, t_start, t_end)

    def _random_ic(self, blue_top_speed=None) -> dict:
        return self._world.initial_ic()

    def infer_action(self) -> list:
        """The discrete action closest to what the recorded blue aircraft did next."""
        w, rec = self._world, self._rec
        name, t = w.names[1], w.t
        est = self._est()
        if est["valid"]:
            d = est["pos"] - self._own_pos_enu()
            ref = math.atan2(d[0], d[1])
        elif self._hdg_ref is not None:
            ref = self._hdg_ref
        else:
            ref = self._state.get("psi", 0.0)
        ahead = lambda dt: rec.state(name, min(t + dt, rec.t_end))
        v = ahead(self.INFER_HDG_S)["vel"]
        off = math.degrees(_wrap_pi(math.atan2(v[0], v[1]) - ref))
        ih = int(np.argmin([abs(_wrap_deg(off - o)) for o in HDG_OFFSETS_DEG]))
        dalt = float(ahead(self.INFER_ALT_S)["pos"][2] - rec.state(name, t)["pos"][2])
        ia = int(np.argmin([abs(a - dalt) for a in ALT_DELTAS_M]))
        # The aircraft approaches a commanded speed slowly: while it is still
        # speeding up (slowing down) the command was the next one above (below).
        spd = float(np.linalg.norm(ahead(self.INFER_SPD_S)["vel"]))
        trend = spd - float(np.linalg.norm(rec.state(name, t)["vel"]))
        cmds = list(self._plat.speed_cmds)
        above = [i for i, c in enumerate(cmds) if c >= spd]
        below = [i for i, c in enumerate(cmds) if c <= spd]
        if trend > 5.0 and above:
            isp = above[0]
        elif trend < -5.0 and below:
            isp = below[-1]
        else:
            isp = int(np.argmin([abs(c - spd) for c in cmds]))
        fire = int(any(dw["shooter"] == name and dw["t_shot"] is not None
                       and t < dw["t_shot"] <= t + 1.0 / self.DECISION_HZ
                       for dw in rec.weapons.values()))
        return [ih, ia, isp, fire]

    def step(self, action=None):
        if action is None:
            action = self.infer_action()
        obs, reward, terminated, truncated, info = super().step(action)
        info["action"] = [int(a) for a in np.asarray(action).reshape(-1)]
        if not (terminated or truncated) and self._world.finished:
            truncated = True
            info["terminal_outcome"] = self._outcome or "END_OF_RECORDING"
            self._ready = False
        return obs, reward, terminated, truncated, info


# ────────────────────────────────────────────────────────────────────
def capture_raw_obs(env) -> dict:
    """Make env keep its last observation vector before normalisation in box["obs"]."""
    box = {}
    norm = env._norm

    def keep(x, lo, hi):
        if lo is env._obs_lo:
            box["obs"] = np.array(x, dtype=np.float64)
        return norm(x, lo, hi)
    env._norm = keep
    return box


def sim_frame(w, names, t, x0=-250_000.0, z0=620_000.0, coal=(COAL_BLUE, COAL_RED)) -> dict:
    """A SimWorld's aircraft and missiles as one bvr_logger.lua sample at time t.
    names[i-1] is aircraft i's DCS name; (x0, z0) puts the fight on the DCS map."""
    units = []
    for i, a in enumerate(w.acs, start=1):
        if not w.alive[i - 1]:
            continue
        R = body_to_enu_matrix(a.chi, a.theta, a.phi)
        f, up, v = R[:, 0], -R[:, 2], a.vel_enu
        units.append({"name": names[i - 1], "coal": coal[i - 1], "type": "F-16C_50",
                      "x": a.y + x0, "y": a.z, "z": a.x + z0,
                      "vx": v[1], "vy": v[2], "vz": v[0],
                      "fx": f[1], "fy": f[2], "fz": f[0],
                      "ux": up[1], "uy": up[2], "uz": up[0],
                      "fuel": a.fuel_frac})
    wl = []
    for m in w.missiles:
        if m.phase.value in ("HIT", "MISS"):
            continue
        wl.append({"id": m.id, "type": "AIM_120C", "shooter": names[m.owner - 1],
                   "target": names[m.target - 1],
                   "x": m.pos[1] + x0, "y": m.pos[2], "z": m.pos[0] + z0,
                   "vx": m.vel[1], "vy": m.vel[2], "vz": m.vel[0]})
    return {"t": t, "units": units, "weapons": wl}


def sim_ammo(w, names, i, t) -> dict:
    return {"ev": "ammo", "t": t, "unit": names[i - 1], "items": [
        {"type": "AIM_120C", "count": int(w.wpn[i - 1]), "aam": True, "guidance": 3}]}


def sim_events(w, names, t) -> list:
    """The bvr_logger.lua event lines for a SimWorld step's events."""
    out = []
    for ev in w.events:
        k = ev.get("type")
        if k == "MISSILE_LAUNCH":
            m = next(m for m in w.missiles if m.id == ev["id"])
            out.append({"ev": "shot", "t": t, "id": m.id, "shooter": names[m.owner - 1],
                        "target": names[m.target - 1], "type": "AIM_120C"})
            out.append(sim_ammo(w, names, m.owner, t))
        elif k == "MISSILE_HIT":
            out.append({"ev": "hit", "t": t, "id": ev["id"], "shooter": names[ev["owner"] - 1],
                        "target": names[ev["target"] - 1], "type": "AIM_120C"})
            out.append({"ev": "weapon_gone", "t": t, "id": ev["id"]})
        elif k == "MISSILE_MISS":
            out.append({"ev": "weapon_gone", "t": t, "id": ev["id"]})
        elif k == "AC_DESTROYED":
            out.append({"ev": "dead", "t": t, "unit": names[ev["ac"] - 1], "cause": "dead"})
    return out


def write_sim_recording(env, policy, path, x0=-250_000.0, z0=620_000.0, t_offset=3600.0,
                        names=("BLUE-1", "RED-1")):
    """
    Fly one simulator episode and write it in bvr_logger.lua's format.

    env     a BvrEnv (1v1), not yet reset
    policy  callable(env) -> action
    Returns [(action, raw observation, info)] per decision, the reset's
    raw observation first with action None.
    """
    w = env._world
    rate_frames = 5                                   # 0.1 s at 50 Hz
    lines = [json.dumps({"format": FORMAT, "rate": 0.1, "t0": t_offset, "theatre": "sim"})]
    frame = {"n": 0}

    def T():
        return round(w.t_sim + t_offset, 4)

    def sample():
        lines.append(json.dumps(sim_frame(w, names, T(), x0, z0)))

    orig_reset, orig_step = w.reset, w.step

    def reset(ic, episode_id=0):
        orig_reset(ic, episode_id=episode_id)
        frame["n"] = 0
        for i in (1, 2):
            lines.append(json.dumps(sim_ammo(w, names, i, T())))
        sample()

    def step(c1, c2):
        tlm = orig_step(c1, c2)
        lines.extend(json.dumps(e) for e in sim_events(w, names, T()))
        frame["n"] += 1
        if frame["n"] % rate_frames == 0:
            sample()
        return tlm

    w.reset, w.step = reset, step
    raw = capture_raw_obs(env)
    try:
        env.reset()
        out = [(None, raw["obs"].copy(), {})]
        while True:
            a = policy(env)
            _, _, term, trunc, info = env.step(a)
            out.append((list(a), raw["obs"].copy(), info))
            if term or trunc:
                break
        sample()                   # the last frame, so the replay reaches the final events
    finally:
        w.reset, w.step = orig_reset, orig_step
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out
