"""
train_bvr.py  —  MaskablePPO training for 1v1 and 2v1 BVR
=========================================================

    pip install sb3-contrib

    python train_bvr.py                            # curriculum from STRAIGHT
    python train_bvr.py --opponent shooter         # fixed opponent
    python train_bvr.py --resume models_bvr/latest
    python train_bvr.py --n-envs 8                 # parallel environments
    python train_bvr.py --platform GENERIC-UCAV --opponent-platform F-16C
                                                   # library platforms (library/)
    python train_bvr.py --format 2v1 --platform GENERIC-UCAV --opponent-platform F-16C
                                                   # two agents (one shared policy) v one
    python train_bvr.py --format 2v1 --resume models_bvr/archive/best_ADAPTIVE_SHOOTER_0.700
                                                   # warm start 2v1 from a 1v1 checkpoint

2v1 (bvr_team): both blue aircraft are flown by the same policy, each on its
own observation, with their radar tracks shared over a datalink; red is the
scripted curriculum opponent (self-play is 1v1 only for now). --platform is
the lead's aircraft, --wingman-platform the wingman's (default: the same).
With --n-envs N there are N fights and 2N agent slots.

TensorBoard:
    bvr/win_rate            KILL fraction over last 50 TERMINAL episodes
    bvr/loss_rate           SHOT_DOWN fraction
    bvr/mutual_rate         MUTUAL_KILL fraction
    bvr/exchange_ratio      kills / losses
    bvr/realized_pk         kills / shots fired
    bvr/shots_per_episode
    bvr/mean_launch_range   km
    bvr/launch_r_over_rmax  where in the envelope it shoots (target ~0.6-0.85)
    bvr/track_frac          fraction of steps with a valid TRACK
    bvr/timeout_rate        watch this: a rising timeout rate means the
                            passivity collapse is starting
    bvr/bank_rev_per_min    wing-rocking: bank reversals per minute of flight
    bvr/<DOCTRINE>_shots    shots per episode under each missile doctrine
    bvr/<DOCTRINE>_pk       (with --doctrine mixed, they should separate)
    bvr/blue_losses_per_ep  2v1: blue aircraft lost per episode (0-2). MUTUAL_KILL
                            is red killed at the cost of a blue aircraft

Win rate is computed over TERMINAL episode outcomes only. Per-step outcome
counting dilutes it by ~the episode length and makes the curriculum advance
on a number that means nothing.
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import deque

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from bvr_env import BvrEnv, REWARD_TERMS, DOCTRINES, DOCTRINE_MIXED
from bvr_compat import load_model
from bvr_library import (DEFAULT_PLATFORM, LibraryError, load_platform, scenario_record,
                         scenario_of, scenario_drift)
from bvr_opponents import BvrOpponentType, CURRICULUM, advance_curriculum
from bvr_policy import AsymmetricMaskablePolicy
from bvr_selfplay import snapshot as selfplay_snapshot
from bvr_team import TeamBvrEnv
from bvr_team_vec import TeamVecEnv


# ═════════════════════════════════════════════════════════════════════
PPO_KWARGS = dict(
    # GAMMA. The single most important hyperparameter here. At 1 Hz with a
    # 40 s missile flight, the launch decision is 40 steps from its outcome.
    #   gamma=0.95  -> 0.13 of the terminal reward reaches the launch. Dead.
    #   gamma=0.99  -> 0.67. Marginal.
    #   gamma=0.997 -> 0.89. Works.
    gamma=0.997,
    gae_lambda=0.95,

    learning_rate=2.5e-4,

    # n_steps must comfortably span a full engagement (up to 300 decisions),
    # otherwise GAE truncates inside the episode and the launch→impact credit
    # path is cut exactly where it matters.
    n_steps=1024,
    batch_size=256,
    n_epochs=6,

    clip_range=0.2,
    clip_range_vf=0.2,      # bounded value updates — the WVR lesson
    target_kl=0.02,         # hard stop on oversized updates

    ent_coef=0.01,          # discrete actions need real exploration pressure
    vf_coef=0.5,
    max_grad_norm=0.5,

    verbose=1,
)

# Frame stacking supplies short-term memory. The EKF already carries the
# belief state, so a recurrent policy is usually unnecessary; try stacking
# first and only reach for RecurrentPPO if it plateaus.
N_STACK = 8

ADVANCE_WIN_RATE = 0.65
ADVANCE_MIN_EPISODES = 60

# While in SELF_PLAY, add the current policy to the opponent pool this often,
# so the agent keeps meeting versions of itself only a little behind it.
SELFPLAY_SNAPSHOT_STEPS = 250_000


# ═════════════════════════════════════════════════════════════════════
class BvrCallback(BaseCallback):

    def __init__(self, vec_env, opponent_type, save_dir="models_bvr",
                 auto_curriculum=True, selfplay_pool=None, allow_selfplay=True,
                 no_selfplay_reason="the two sides fly different platforms, so self-play "
                                    "does not apply", verbose=1):
        super().__init__(verbose)
        self.no_selfplay_reason = no_selfplay_reason
        # False when the two sides fly different platforms: a snapshot of the
        # agent was trained on the agent's platform and cannot fly the other.
        self.allow_selfplay = allow_selfplay
        # The VecEnv itself, not a list of raw envs: with n_envs > 1 the envs
        # live in other processes, where a Python reference cannot reach them.
        # set_attr() is the only channel that works for both vec env types.
        self.vec = vec_env
        self.opponent_type = opponent_type
        self.save_dir = save_dir
        self.auto_curriculum = auto_curriculum

        self.outcomes = deque(maxlen=50)
        self.shots = deque(maxlen=50)
        self.launch_rngs = deque(maxlen=200)
        self.launch_ratios = deque(maxlen=200)
        self.track_hits = deque(maxlen=5000)
        # |per-term shaping contribution|, to show which part of the potential
        # is actually driving the reward rather than only the total.
        self.term_abs = {k: deque(maxlen=5000) for k in REWARD_TERMS}
        self.ep_count = 0
        # Episodes finished against the CURRENT opponent. The advance gate
        # cannot use len(self.outcomes): that deque is capped at 50, so a
        # 60-episode minimum measured on it could never be met.
        self.stage_episodes = 0
        self.best_win = -1.0
        self.selfplay_pool = selfplay_pool
        self.last_snapshot_step = 0
        # Opponent behind each of the last 50 outcomes, for the SELF_PLAY split
        # between policy snapshots and the scripted shooter in the mix.
        self.opp_kinds = deque(maxlen=50)
        # (doctrine, outcome, shots) — longer than 50 so each of the three
        # doctrines in a mixed run still gets a usable sample.
        self.doctrine_eps = deque(maxlen=150)
        # (bank reversals, seconds flown) per episode.
        self.bank = deque(maxlen=50)
        # 2v1: blue aircraft lost per episode.
        self.blue_losses = deque(maxlen=50)
        # Identifies this run in bvr_metrics.json, so its history is not
        # mixed with a previous run's.
        self.run_id = f"{int(time.time())}-{os.getpid()}"
        os.makedirs(save_dir, exist_ok=True)

    def curriculum_state(self) -> dict:
        """What the auto-curriculum will do next, for the log and the GUI."""
        nxt = advance_curriculum(self.opponent_type)
        blocked = None
        if not self.auto_curriculum:
            blocked = "auto-curriculum is OFF"
        elif nxt == self.opponent_type:
            blocked = "this is the last stage"
        elif nxt == BvrOpponentType.SELF_PLAY and not self.allow_selfplay:
            blocked = self.no_selfplay_reason
        return {"auto": bool(self.auto_curriculum), "stage": self.opponent_type.name,
                "next": None if nxt == self.opponent_type else nxt.name,
                "blocked": blocked, "stage_episodes": self.stage_episodes,
                "min_episodes": ADVANCE_MIN_EPISODES, "win_rate": ADVANCE_WIN_RATE}

    def describe_curriculum(self) -> str:
        c = self.curriculum_state()
        if c["blocked"]:
            return f"curriculum: stays on {c['stage']} ({c['blocked']})"
        return (f"curriculum: ON, {c['stage']} -> {c['next']} once the win rate over the last 50 "
                f"episodes reaches {ADVANCE_WIN_RATE:.0%} after at least {ADVANCE_MIN_EPISODES} "
                f"episodes at the stage")

    def _on_step(self) -> bool:
        if (self.opponent_type == BvrOpponentType.SELF_PLAY
                and self.num_timesteps - self.last_snapshot_step >= SELFPLAY_SNAPSHOT_STEPS):
            self._snapshot()
        for info in self.locals.get("infos", []):
            if "track_state" in info:
                self.track_hits.append(1.0 if info["track_state"] == "TRACK" else 0.0)

            st = info.get("shaping_terms")
            if st:
                st = {**st, **info.get("cost_terms", {})}
                for k in REWARD_TERMS:
                    self.term_abs[k].append(abs(st.get(k, 0.0)))

            outcome = info.get("terminal_outcome")
            if outcome is None:
                continue

            self.ep_count += 1
            self.stage_episodes += 1
            self.outcomes.append(outcome)
            self.opp_kinds.append("policy" if str(info.get("opponent_detail", "")).startswith("policy:")
                                  else "scripted")
            self.shots.append(int(info.get("shots_fired", 0)))
            self.doctrine_eps.append((info.get("doctrine", "AGGRESSIVE"), outcome,
                                      int(info.get("shots_fired", 0))))
            self.bank.append((int(info.get("bank_reversals", 0)),
                              float(info.get("flight_time", 0.0))))
            if "blue_losses" in info:
                self.blue_losses.append(int(info["blue_losses"]))
            for lg in info.get("launch_log", []):
                self.launch_rngs.append(lg["range_true"] if lg["range_true"] > 0 else lg["range_est"])
                self.launch_ratios.append(lg["r_over_rmax"])

            if self.ep_count % 10 == 0:
                self._log()

        return True

    def _log(self):
        n = len(self.outcomes)
        if n < 5:
            return
        kills = self.outcomes.count("KILL")
        losses = self.outcomes.count("SHOT_DOWN")
        mutual = self.outcomes.count("MUTUAL_KILL")
        timeouts = self.outcomes.count("TIMEOUT")
        crashes = self.outcomes.count("CRASH")
        bandit_crashes = self.outcomes.count("BANDIT_CRASH")

        win_rate = kills / n
        total_shots = max(sum(self.shots), 1)

        rec = self.logger.record
        rec("bvr/win_rate", win_rate)
        rec("bvr/loss_rate", losses / n)
        rec("bvr/mutual_rate", mutual / n)
        rec("bvr/timeout_rate", timeouts / n)
        rec("bvr/escape_rate", self.outcomes.count("ESCAPE") / n)
        rec("bvr/crash_rate", crashes / n)
        rec("bvr/bandit_crash_rate", bandit_crashes / n)
        rec("bvr/exchange_ratio", kills / max(losses + mutual, 1))
        rec("bvr/realized_pk", (kills + mutual) / total_shots)
        rec("bvr/shots_per_episode", total_shots / n)
        if self.launch_rngs:
            rec("bvr/mean_launch_range_km", float(np.mean(self.launch_rngs)) / 1000.0)
            rec("bvr/launch_r_over_rmax", float(np.mean(self.launch_ratios)))
        if self.track_hits:
            rec("bvr/track_frac", float(np.mean(self.track_hits)))
        term_mean = {k: float(np.mean(v)) if v else 0.0
                     for k, v in self.term_abs.items()}
        term_total = sum(term_mean.values())
        for k, v in term_mean.items():
            rec(f"bvr/term_{k}", v)
            rec(f"bvr/termshare_{k}", v / term_total if term_total > 0 else 0.0)
        split = ""
        if self.opponent_type == BvrOpponentType.SELF_PLAY:
            for kind in ("policy", "scripted"):
                outs = [o for o, k in zip(self.outcomes, self.opp_kinds) if k == kind]
                if outs:
                    wr = outs.count("KILL") / len(outs)
                    rec(f"bvr/win_rate_vs_{kind}", wr)
                    split += f" | vs {kind} {wr:5.1%} ({len(outs)})"
        flown = sum(t for _, t in self.bank)
        bank_rpm = 60.0 * sum(r for r, _ in self.bank) / flown if flown > 0 else 0.0
        rec("bvr/bank_rev_per_min", bank_rpm)
        losses_ep = float(np.mean(self.blue_losses)) if self.blue_losses else None
        if losses_ep is not None:
            rec("bvr/blue_losses_per_ep", losses_ep)
        doctrine = {}
        for d in DOCTRINES:
            eps = [(o, sh) for dd, o, sh in self.doctrine_eps if dd == d]
            if not eps:
                continue
            k = sum(o in ("KILL", "MUTUAL_KILL") for o, _ in eps)
            fired = sum(sh for _, sh in eps)
            doctrine[d] = {"n": len(eps),
                           "win": round(sum(o == "KILL" for o, _ in eps) / len(eps), 3),
                           "shots": round(fired / len(eps), 2),
                           "pk": round(k / fired, 3) if fired else 0.0}
            rec(f"bvr/{d}_win", doctrine[d]["win"])
            rec(f"bvr/{d}_shots", doctrine[d]["shots"])
            rec(f"bvr/{d}_pk", doctrine[d]["pk"])
        dline = "".join(f" | {d[:3]} {v['shots']:.1f}sh Pk{v['pk']:.2f}"
                        for d, v in doctrine.items()) if len(doctrine) > 1 else ""
        rec("bvr/episodes", self.ep_count)
        rec("bvr/opponent", CURRICULUM.index(self.opponent_type))

        if self.verbose:
            print(f"[bvr] ep {self.ep_count:5d} | {self.opponent_type.name:16s} | "
                  f"win {win_rate:5.1%} loss {losses/n:5.1%} mut {mutual/n:5.1%} "
                  f"to {timeouts/n:5.1%} crash {crashes/n:5.1%} bcrash {bandit_crashes/n:5.1%} | "
                  f"Pk {(kills+mutual)/total_shots:4.2f} | "
                  f"shots/ep {total_shots/n:4.2f} | rev/min {bank_rpm:4.1f}"
                  f"{f' | lost/ep {losses_ep:4.2f}' if losses_ep is not None else ''}{split}{dline}")

        # Write metrics for the GUI. Appended to a rolling history list so
        # the browser can draw trend charts without accumulating unbounded data.
        try:
            mpath = os.path.join(self.save_dir, "..", "bvr_metrics.json")
            mpath = os.path.normpath(mpath)
            prev = {}
            if os.path.exists(mpath):
                try: prev = json.loads(open(mpath).read())
                except Exception: pass
            # One point per log (every 10 episodes) for the GUI's training
            # dashboard. History belongs to this run: an earlier run's points
            # in the same file are dropped rather than drawn as if continued.
            hist = prev.get("history", []) if prev.get("run") == self.run_id else []
            r3 = lambda v: round(float(v), 3)
            hist.append({
                "ep": self.ep_count, "steps": int(self.model.num_timesteps),
                "win": r3(win_rate), "loss": r3(losses / n), "mut": r3(mutual / n),
                "to": r3(timeouts / n),
                "exch": round((kills + mutual) / max(losses + mutual, 1), 2),
                "pk": r3((kills + mutual) / total_shots), "shots": r3(total_shots / n),
                "rrmax": r3(np.mean(self.launch_ratios)) if self.launch_ratios else None,
                "track": r3(np.mean(self.track_hits)) if self.track_hits else None,
                "rev": round(bank_rpm, 2),
                "lost": r3(losses_ep) if losses_ep is not None else None,
                "stage": self.opponent_type.name})
            # A long run keeps its whole shape: past 600 points, every other
            # older point is dropped, so the chart always spans the run.
            if len(hist) > 600:
                hist = hist[:-100:2] + hist[-100:]
            metrics = {
                "run":           self.run_id,
                "episode":       self.ep_count,
                "opponent":      self.opponent_type.name,
                "win_rate":      round(win_rate, 3),
                "loss_rate":     round(losses/n, 3),
                "mutual_rate":   round(mutual/n, 3),
                "timeout_rate":  round(timeouts/n, 3),
                "escape_rate":   round(self.outcomes.count("ESCAPE")/n, 3),
                "crash_rate":    round(crashes/n, 3),
                "bandit_crash_rate": round(bandit_crashes/n, 3),
                "exchange_ratio": round(kills/max(losses+mutual,1), 2),
                "realized_pk":   round((kills+mutual)/total_shots, 3),
                "shots_per_ep":  round(total_shots/n, 2),
                "mean_launch_km": round(float(np.mean(self.launch_rngs))/1000,2) if self.launch_rngs else 0,
                "launch_r_rmax": round(float(np.mean(self.launch_ratios)),3) if self.launch_ratios else 0,
                "track_frac":    round(float(np.mean(self.track_hits)),3) if self.track_hits else 0,
                "term_abs":      {k: round(v, 5) for k, v in term_mean.items()},
                "term_share":    {k: round(v/term_total, 4) if term_total > 0 else 0.0
                                  for k, v in term_mean.items()},
                "bank_rev_per_min": round(bank_rpm, 2),
                "blue_losses_per_ep": round(losses_ep, 3) if losses_ep is not None else None,
                "doctrine":      doctrine,
                "curriculum":    self.curriculum_state(),
                "run_notes":     getattr(self, "run_notes", []),
                "history":       hist,
                "steps_done":    int(self.model.num_timesteps),
            }
            with open(mpath, "w") as f:
                json.dump(metrics, f)
        except Exception:
            pass

        # Archive the best checkpoint OUTSIDE the rolling save directory.
        # Self-play regressions can and do destroy good policies, and a
        # checkpoint that lives in the same folder the trainer overwrites is
        # not a backup.
        if n >= 30 and win_rate > self.best_win:
            self.best_win = win_rate
            path = os.path.join(self.save_dir, "archive",
                                f"best_{self.opponent_type.name}_{win_rate:.3f}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)

        if self.stage_episodes >= ADVANCE_MIN_EPISODES and win_rate >= ADVANCE_WIN_RATE:
            if self.auto_curriculum:
                self._advance()
            elif self.ep_count - getattr(self, "_last_hold_note", -10**9) >= 200:
                # Said again every 200 episodes: a one-off line scrolls out of
                # the GUI's log long before anyone wonders why nothing moved.
                self._last_hold_note = self.ep_count
                print(f"[bvr] win rate {win_rate:.0%} is past the {ADVANCE_WIN_RATE:.0%} "
                      f"advance mark, but the auto-curriculum is OFF: staying on "
                      f"{self.opponent_type.name}")

    def _advance(self):
        nxt = advance_curriculum(self.opponent_type)
        if nxt == self.opponent_type:
            return
        if nxt == BvrOpponentType.SELF_PLAY and not self.allow_selfplay:
            # Repeated every 200 episodes, as for the OFF case above.
            if self.ep_count - getattr(self, "_last_hold_note", -10**9) >= 200:
                self._last_hold_note = self.ep_count
                print(f"[bvr] curriculum ends at ADAPTIVE_SHOOTER: {self.no_selfplay_reason}")
            return
        self.model.save(os.path.join(self.save_dir, "archive",
                                     f"pre_{nxt.name}"))
        print(f"[bvr] === CURRICULUM: {self.opponent_type.name} -> {nxt.name} ===")
        self.opponent_type = nxt
        if nxt == BvrOpponentType.SELF_PLAY:
            # Workers build SELF_PLAY opponents from the pool on their next
            # reset, so it must hold a snapshot before they switch.
            self._snapshot()
        # Reaches worker processes too. BvrEnv builds its opponent from
        # _opponent_type on reset(), so this takes effect next episode.
        self.vec.set_attr("_opponent_type", nxt)
        self.outcomes.clear()
        self.shots.clear()
        self.opp_kinds.clear()
        self.doctrine_eps.clear()
        self.blue_losses.clear()
        self.stage_episodes = 0
        self.best_win = -1.0


    def _snapshot(self):
        path = selfplay_snapshot(self.model, self.selfplay_pool)
        self.last_snapshot_step = self.num_timesteps
        print(f"[bvr] self-play snapshot -> {path}")


# ═════════════════════════════════════════════════════════════════════
def make_env(idx, opponent, seed, privileged, viz, envelope_table, gamma, selfplay_pool,
             doctrine, platform, opponent_platform):
    def _init():
        return BvrEnv(opponent_type=opponent, gamma_discount=gamma,
                      seed=seed + idx, instance_id=idx,
                      privileged_critic=privileged,
                      enable_viz=(viz and idx == 0),
                      envelope_table=envelope_table,
                      selfplay_pool=selfplay_pool, doctrine=doctrine,
                      platform=platform, opponent_platform=opponent_platform)
    return _init


def make_team_env(idx, opponent, seed, privileged, envelope_table, gamma, doctrine,
                  blue_platforms, red_platform):
    def _init():
        return TeamBvrEnv(opponent_type=opponent, gamma_discount=gamma, seed=seed + idx,
                          instance_id=idx, privileged_critic=privileged,
                          envelope_table=envelope_table, doctrine=doctrine,
                          blue_platforms=blue_platforms, red_platform=red_platform)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--opponent", type=str, default="straight")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--no-privileged", action="store_true")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--save-dir", type=str, default="models_bvr")
    ap.add_argument("--run-name", type=str, default="latest",
                     help="Base filename (no .zip) the final model is saved "
                          "as inside --save-dir, e.g. models_bvr/<run-name>.zip")
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lr",    type=float, default=2.5e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--doctrine", type=str, default=DOCTRINE_MIXED.lower(),
                    choices=[DOCTRINE_MIXED.lower()] + [d.lower() for d in DOCTRINES],
                    help="missile doctrine; 'mixed' draws one per episode, which the "
                         "policy sees, so one model learns all three")
    ap.add_argument("--platform", default=DEFAULT_PLATFORM,
                    help="library platform the agent flies (see library/)")
    ap.add_argument("--opponent-platform", default=DEFAULT_PLATFORM,
                    help="library platform the opponent flies")
    ap.add_argument("--format", default="1v1", choices=["1v1", "2v1"],
                    help="1v1, or 2v1: two agents on one shared policy against one opponent")
    ap.add_argument("--wingman-platform", default=None,
                    help="2v1: the wingman's platform (default: the same as --platform)")
    ap.add_argument("--envelope-table", type=str, default="library",
                    help="'library' (default): each missile's calibrated table from "
                         "library/envelopes; or a path to one table used for every missile")
    args = ap.parse_args()

    opponent = BvrOpponentType[args.opponent.upper()]
    if opponent not in CURRICULUM:
        ap.error(f"opponent {opponent.name} is not implemented; choose one of "
                 f"{', '.join(o.name.lower() for o in CURRICULUM)}")
    privileged = not args.no_privileged

    team = args.format == "2v1"
    wingman = (args.wingman_platform or args.platform) if team else None
    # Startup facts worth keeping in view: printed as usual and also written to
    # bvr_metrics.json, where the GUI's CURRENT RUN panel shows them for the
    # whole run instead of letting them scroll out of the log.
    run_notes = []
    def note(msg):
        print(f"[bvr] {msg}")
        run_notes.append(msg)
    if args.wingman_platform and not team:
        ap.error("--wingman-platform needs --format 2v1")

    # Resolve every platform now, so a missing item, an invalid value or an
    # uncalibrated missile stops the run here with a clear message.
    try:
        for pid in (args.platform, args.opponent_platform, wingman):
            if pid:
                load_platform(pid)
        from bvr_env import BvrEnv as _probe
        for pid in {args.platform, wingman} - {None}:
            _probe(opponent_type=BvrOpponentType.STRAIGHT, platform=pid,
                   opponent_platform=args.opponent_platform, envelope_table=args.envelope_table)
    except LibraryError as e:
        ap.error(str(e))
    heterogeneous = args.platform != args.opponent_platform
    if heterogeneous and opponent == BvrOpponentType.SELF_PLAY:
        ap.error(f"self-play needs both sides on the same platform (here {args.platform} v "
                 f"{args.opponent_platform}): a snapshot of the agent cannot fly the other side")
    if team and opponent == BvrOpponentType.SELF_PLAY:
        ap.error("2v1 has no self-play stage yet: choose a scripted opponent")
    envelope_table = args.envelope_table
    if team:
        note(f"2v1: agents {args.platform} + {wingman} v opponent {args.opponent_platform}")
    else:
        note(f"platforms: agent {args.platform} v opponent {args.opponent_platform}")

    # gamma MUST match the value PPO trains with (passed below via kwargs) —
    # potential-based shaping (bvr_env._potential / step()'s `gamma*phi -
    # prev_phi`) is only guaranteed policy-invariant (Ng, Harada & Russell,
    # 1999) when its discount matches the agent's. This used to read
    # PPO_KWARGS["gamma"] here, the module DEFAULT, so a --gamma override on
    # the CLI silently desynced the two and broke that guarantee.
    selfplay_pool = os.path.join(args.save_dir, "selfplay_pool")
    if team:
        fns = [make_team_env(i, opponent, args.seed, privileged, envelope_table, args.gamma,
                             args.doctrine.upper(), (args.platform, wingman), args.opponent_platform)
               for i in range(args.n_envs)]
        vec = TeamVecEnv(fns, in_process=(args.n_envs == 1))
    else:
        fns = [make_env(i, opponent, args.seed, privileged, args.viz, envelope_table, args.gamma,
                        selfplay_pool, args.doctrine.upper(), args.platform, args.opponent_platform)
               for i in range(args.n_envs)]
        vec = DummyVecEnv(fns) if args.n_envs == 1 else SubprocVecEnv(fns)

    # VecFrameStack over a Dict space stacks every sub-key, which is what we
    # want — the critic benefits from privileged history too.
    vec = VecFrameStack(vec, n_stack=N_STACK)

    kwargs = dict(PPO_KWARGS)
    kwargs["gamma"]         = args.gamma
    kwargs["learning_rate"] = args.lr
    kwargs["batch_size"]    = args.batch_size
    if privileged:
        policy = AsymmetricMaskablePolicy
        kwargs["policy_kwargs"] = dict(pi_arch=(256, 256), vf_arch=(256, 256))
    else:
        policy = "MlpPolicy"
        kwargs["policy_kwargs"] = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    if args.resume:
        # load_model also takes checkpoints saved before the latest observation
        # inputs were added (bvr_compat), widening them without changing a choice.
        model = load_model(args.resume, env=vec, **{
            k: v for k, v in kwargs.items() if k not in ("policy_kwargs",)})
        note(f"resumed from {args.resume}")
        prev = scenario_of(model)
        was = (prev["format"], prev["platform"], prev.get("wingman_platform"), prev["opponent_platform"])
        now = (args.format, args.platform, wingman, args.opponent_platform)
        if was != now:
            desc = lambda f, p, w, o: f"{f} {p}{' + ' + w if w else ''} v {o}"
            note(f"NOTE: this checkpoint was trained as {desc(*was)}; continuing as "
                 f"{desc(*now)} (a warm start, not a continuation)")
        for msg in scenario_drift(prev):
            note(f"NOTE: {msg}")
    else:
        model = MaskablePPO(policy, vec, tensorboard_log="tb_logs_bvr/", **kwargs)
    # Saved inside every checkpoint this run writes, self-play snapshots included.
    model.bvr_scenario = scenario_record(args.platform, args.opponent_platform,
                                         wingman_platform_id=wingman, fmt=args.format)

    if opponent == BvrOpponentType.SELF_PLAY:
        # learn() resets every env before any callback runs, and a SELF_PLAY
        # reset needs a snapshot in the pool — seed it with this policy now.
        print(f"[bvr] self-play snapshot -> {selfplay_snapshot(model, selfplay_pool)}")

    cb = BvrCallback(vec, opponent, save_dir=args.save_dir, selfplay_pool=selfplay_pool,
                     auto_curriculum=not args.no_curriculum,
                     allow_selfplay=not (heterogeneous or team),
                     **({"no_selfplay_reason": "2v1 has no self-play stage yet"} if team else {}))

    note(cb.describe_curriculum())
    cb.run_notes = run_notes
    # One machine-readable line for the GUI server, which shows these notes
    # at once rather than waiting for the first metrics write.
    print("[bvr] run-notes: " + json.dumps(run_notes), flush=True)
    failure = None
    try:
        model.learn(total_timesteps=args.steps, callback=cb,
                    tb_log_name=f"bvr_{int(time.time())}")
    except KeyboardInterrupt:
        print("\n[bvr] interrupted")
    except (EOFError, BrokenPipeError, ConnectionResetError):
        # A SubprocVecEnv worker raised and died; it printed its own traceback
        # (e.g. an OpponentError) above. Here the only symptom is a dead pipe.
        failure = "an environment worker process failed; its error is printed above"
    except Exception as e:
        traceback.print_exc()
        failure = f"{type(e).__name__}: {e}"
    finally:
        run_name = os.path.basename(args.run_name.strip()) or "latest"
        model.save(os.path.join(args.save_dir, run_name))
        print(f"[bvr] saved final model to {os.path.join(args.save_dir, run_name)}.zip")
        try:
            vec.close()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
            pass            # a worker is already dead; nothing left to close cleanly
    if failure:
        print(f"[bvr] TRAINING STOPPED BY AN ERROR: {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
