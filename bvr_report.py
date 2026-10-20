"""
bvr_report.py  —  One record per 1v1 fight, the same in the simulator and in DCS
================================================================================

BvrEnv keeps an EpisodeReport and returns its record at the end of every episode
(info["report"]). WATCH (bvr_gui.py) and dcs_live.py write one JSON line per
fight from it, so a fight in the simulator and one in DCS are read the same way:

  red        who the bandit was: its scripted style (ADAPTIVE: DCS-like, runner,
             or its defence), a self-play snapshot, or the DCS AI
  missiles   every missile, ours and red's: at launch the range, the shooter's
             r/Rmax, how far off its nose the target was, both aircraft's speed
             and altitude; at the end hit, out of energy (KINEMATIC), an
             endgame miss and by how far, or still flying; its time of flight
             and the longest it flew without its shooter's guidance
  defence    every missile fired at us: the RWR warning and time to impact then,
             our heading choices, the first turn of 90 deg or more (time to
             impact, radar track then), chaff presses (DCS), how it ended
  first      who fired first, and by how many seconds
  phases     heading choices (|offset| deg) before our first shot, while our
             missile flies, under fire, and after
  track      radar track losses: when, how long, and the likely cause (outside
             the radar's azimuth limit, beyond its range, low closure)
  timeout    for a fight out of time: range, both aspects, closure, missiles
             left, fuel, red's style, our last 60 heading choices

Only the agent's view of the fight goes in, plus the truth the env already
holds (range, aspects): nothing here feeds the observation or the reward.
"""

import math

import numpy as np

PHASES = ("before_shot", "supporting", "under_fire", "after")


def red_style(opponent) -> dict:
    """The bandit's style for this episode: {"style": short key to group by, "detail": text}."""
    if opponent is None:
        return {"style": "DCS AI", "detail": "the DCS AI"}
    label = getattr(opponent, "label", None)
    if isinstance(label, str) and label.startswith("policy:"):
        return {"style": "self-play", "detail": label}
    name = type(opponent).__name__.replace("Opponent", "").upper()
    if hasattr(opponent, "LONG_SHOT") and hasattr(opponent, "DEFENCE"):
        frac = float(getattr(opponent, "SHOT_RMAX_FRAC", 0.0))
        if getattr(opponent, "LONG_SHOT", False):
            style = "DCS-like"
        elif getattr(opponent, "RUNNER", False):
            style = "runner"
        else:
            style = f"defence {getattr(opponent, 'DEFENCE', '?')}"
        return {"style": style, "detail": f"ADAPTIVE {style}, shoots at {frac:.2f} R-max"
                + (", presses" if getattr(opponent, "PRESS", False) else "")}
    if hasattr(opponent, "SHOT_RMAX_FRAC"):
        return {"style": name, "detail": f"{name}, shoots at {float(opponent.SHOT_RMAX_FRAC):.2f} R-max"}
    return {"style": name, "detail": name}


def _r(x, n=1):
    return None if x is None else round(float(x), n)


class EpisodeReport:
    """Collects one fight. BvrEnv calls reset(), on_event() for each event it
    takes in, on_decision() at each decision with the action chosen, and
    finish() when the episode ends."""

    TRACK_LOW_CLOSURE = 40.0     # m/s: a track lost with less closure than this is put down to it

    def __init__(self):
        self.reset(None)

    def reset(self, env, red=None):
        self.red = red or {"style": "?", "detail": "?"}
        self.missiles = {}           # id -> record
        self.defence = {}            # red missile id -> record
        self.phases = {p: {} for p in PHASES}
        self.headings = []           # (t, |offset|) every decision
        self.track = []              # {"t", "s", "cause"}
        self._trk_prev = None
        self._lost_at = None
        self.first = {"ours": None, "red": None}
        self.chaff_fn = getattr(env, "chaff_presses_fn", None) if env is not None else None

    # ── events (owners as the agent sees them: 1 ours, 2 red) ──────────
    def on_event(self, env, ev):
        t = ev.get("type")
        s = env._state or {}
        if t == "MISSILE_LAUNCH" and ev.get("owner") in (1, 2):
            owner = "ours" if ev.get("owner") == 1 else "red"
            ts = float(env._t_sim)
            if self.first[owner] is None:
                self.first[owner] = ts
            rec = {"owner": owner, "t": _r(ts), "range_km": _r(s.get("range", 0.0) / 1000.0, 2),
                   "off_nose_deg": _r(s.get("aa_deg") if owner == "ours" else s.get("aa_deg_t")),
                   "speed": _r(s.get("speed")), "alt": _r(s.get("alt"), 0),
                   "speed_t": _r(s.get("speed_t")), "alt_t": _r(s.get("alt_t"), 0),
                   "end": None, "miss_m": None, "t_flight": None, "max_unguided_s": 0.0}
            try:
                est = env._est()
                r = env._est_range(est)
                if owner == "ours":
                    rm, rn = env._own_envelope(est)
                    rec["r_over_rmax"] = _r(r / max(rm, 1.0), 3)
                    rec["r_over_rnez"] = _r(r / max(rn, 1.0), 3)
                else:
                    rmt, _ = env._threat_envelope(est)
                    rec["r_over_rmax_est"] = _r(r / max(rmt, 1.0), 3)
                    rec["r_over_rmax"] = _r(s.get("range", 0.0) / max(env._bandit_rmax(), 1.0), 3)
            except Exception:
                pass
            self.missiles[ev.get("id")] = rec
            if owner == "red":
                self.defence[ev.get("id")] = {
                    "t_launch": _r(ts), "range_km": rec["range_km"], "r_over_rmax": rec.get("r_over_rmax"),
                    "t_warn": None, "tti_warn": None, "t_break": None, "tti_break": None,
                    "track_at_break": None, "min_tti": None, "headings": {}, "end": None,
                    "chaff0": self._chaff()}
        elif t in ("MISSILE_HIT", "MISSILE_MISS"):
            rec = self.missiles.get(ev.get("id"))
            if rec is not None and rec["end"] is None:
                rec["end"] = "HIT" if t == "MISSILE_HIT" else ev.get("cause", "MISS")
                rec["miss_m"] = _r(ev.get("miss_dist"))
                rec["t_flight"] = _r(env._t_sim - rec["t"])
                d = self.defence.get(ev.get("id"))
                if d is not None:
                    d["end"] = rec["end"]
                    d["chaff"] = self._chaff_since(d)

    def _chaff(self):
        try:
            return int(self.chaff_fn()) if self.chaff_fn else None
        except Exception:
            return None

    def _chaff_since(self, d):
        now = self._chaff()
        return None if now is None or d.get("chaff0") is None else now - d["chaff0"]

    # ── decisions ──────────────────────────────────────────────────────
    def on_decision(self, env, hdg_offset_deg, track_state):
        """At each decision, before it is flown: the state the policy saw and the heading it chose."""
        s = env._state or {}
        t = float(env._t_sim)
        h = int(round(abs(float(hdg_offset_deg))))
        msl = [m for m in (s.get("missiles") or []) if m.get("state") not in ("HIT", "MISS", "DUD")]
        ours = [m for m in msl if m.get("owner") == 1]
        reds = [m for m in msl if m.get("owner") == 2]
        for m in msl:
            rec = self.missiles.get(m.get("id"))
            if rec is not None and not m.get("seeker_active", 0):
                rec["max_unguided_s"] = _r(max(rec["max_unguided_s"], float(m.get("t_since_update", 0.0))))
        if reds:
            phase = "under_fire"
        elif ours:
            phase = "supporting"
        elif not env._shots_fired:
            phase = "before_shot"
        else:
            phase = "after"
        ph = self.phases[phase]
        ph[h] = ph.get(h, 0) + 1
        self.headings.append((round(t, 1), h))
        warn = bool((s.get("rwr") or {}).get("launch_warn", 0))
        for m in reds:
            d = self.defence.get(m.get("id"))
            if d is None:
                continue
            tti = float(m.get("tgo_est", 0.0))
            if warn and d["t_warn"] is None:
                d["t_warn"], d["tti_warn"] = _r(t), _r(tti)
            d["min_tti"] = _r(tti if d["min_tti"] is None else min(d["min_tti"], tti))
            d["headings"][h] = d["headings"].get(h, 0) + 1
            if h >= 90 and d["t_break"] is None:
                d["t_break"], d["tti_break"], d["track_at_break"] = _r(t), _r(tti), track_state
        self._track_step(env, s, t, track_state)

    def _track_step(self, env, s, t, ts):
        """A loss is TRACK going to anything else (coast, none); it ends when TRACK comes back."""
        if self._trk_prev == "TRACK" and ts != "TRACK" and self._lost_at is None:
            self._lost_at = (t, self._loss_cause(env, s))
        elif ts == "TRACK" and self._lost_at is not None:
            t0, cause = self._lost_at
            self.track.append({"t": _r(t0), "s": _r(t - t0), "cause": cause, "regained": True})
            self._lost_at = None
        self._trk_prev = ts

    def _loss_cause(self, env, s):
        radar = getattr(env, "_radar", None)
        ata = abs(float(s.get("aa_deg", 0.0)))
        fov = math.degrees(getattr(radar, "fov_az", math.radians(60.0))) if radar is not None else 60.0
        if ata > 0.95 * fov:
            return f"azimuth ({ata:.0f} deg, limit {fov:.0f})"
        rmax = getattr(radar, "max_range", None)
        if rmax is not None and float(s.get("range", 0.0)) > 0.95 * rmax:
            return "range"
        if abs(float(s.get("closure", 0.0))) < self.TRACK_LOW_CLOSURE:
            return "low closure"
        return "other"

    # ── end ────────────────────────────────────────────────────────────
    def finish(self, env, outcome) -> dict:
        s = env._state or {}
        t = float(env._t_sim)
        if self._lost_at is not None:
            t0, cause = self._lost_at
            self.track.append({"t": _r(t0), "s": _r(t - t0), "cause": cause, "regained": False})
            self._lost_at = None
        for rec in self.missiles.values():
            if rec["end"] is None:
                rec["end"] = "in flight"
                rec["t_flight"] = _r(t - rec["t"])
        for d in self.defence.values():
            if d["end"] is None:
                d["end"] = "in flight"
                d["chaff"] = self._chaff_since(d)
            d.pop("chaff0", None)
        fo, fr = self.first["ours"], self.first["red"]
        if fo is None and fr is None:
            first = {"who": "none", "gap_s": None}
        elif fr is None or (fo is not None and fo <= fr):
            first = {"who": "ours", "gap_s": None if fr is None else _r(fr - fo)}
        else:
            first = {"who": "red", "gap_s": None if fo is None else _r(fo - fr)}
        rep = {"outcome": outcome, "t": _r(t), "red": self.red, "first": first,
               "missiles": [dict(id=k, **v) for k, v in self.missiles.items()],
               "defence": [dict(id=k, **v) for k, v in self.defence.items()],
               "phases": {p: {str(k): v for k, v in sorted(c.items())} for p, c in self.phases.items()},
               "track_losses": self.track}
        if outcome == "TIMEOUT":
            rep["timeout"] = {
                "range_km": _r(s.get("range", 0.0) / 1000.0, 2),
                "our_aspect_deg": _r(s.get("aa_deg")), "red_aspect_deg": _r(s.get("aa_deg_t")),
                "closure": _r(s.get("closure")), "wpn": s.get("wpn_remaining"), "wpn_red": s.get("wpn_remaining_t"),
                "fuel": _r(s.get("fuel_frac"), 3), "cause": self.timeout_cause(s),
                "last_headings": [h for _, h in self.headings[-60:]]}
        return rep

    @staticmethod
    def timeout_cause(s) -> str:
        """A rough label for a fight out of time, from the end state: red running away (its tail to us),
        both sides turned away, both still pointing, or out of missiles."""
        if not s.get("wpn_remaining", 1) and not s.get("wpn_remaining_t", 1):
            return "both out of missiles"
        ra = abs(float(s.get("aa_deg_t", 0.0)))      # off red's nose, to us
        oa = abs(float(s.get("aa_deg", 0.0)))        # off our nose, to red
        if ra > 120.0 and oa < 60.0:
            return "red running, we follow"
        if ra > 120.0:
            return "red running, we turned away"
        if oa > 90.0 and ra > 60.0:
            return "both turned away"
        if oa > 90.0:
            return "we turned away"
        if ra > 60.0:
            return "red cranking or beaming"
        return "both pointing"


def summarise(reports) -> dict:
    """Totals over many fights' reports (for dcs_live's console and WATCH's tables)."""
    reps = [r for r in reports if r]
    out = {"n": len(reps)}
    firsts = [r["first"]["who"] for r in reps]
    out["first"] = {k: firsts.count(k) for k in ("ours", "red", "none")}
    d = [x for r in reps for x in r.get("defence", [])]
    ended = [x for x in d if x["end"] != "in flight"]
    out["defence"] = {
        "missiles": len(d), "hit": sum(x["end"] == "HIT" for x in ended),
        "broke": sum(x["t_break"] is not None for x in d),
        "broke_with_track": sum(x["track_at_break"] == "TRACK" for x in d),
        "median_tti_break": None if not [x for x in d if x["tti_break"] is not None] else
        float(np.median([x["tti_break"] for x in d if x["tti_break"] is not None]))}
    to = [r["timeout"]["cause"] for r in reps if r.get("timeout")]
    out["timeouts"] = {c: to.count(c) for c in sorted(set(to))}
    return out
