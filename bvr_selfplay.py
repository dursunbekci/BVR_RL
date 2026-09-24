"""
bvr_selfplay.py  —  Self-play opponent
======================================

The SELF_PLAY curriculum stage flies AC2 with a frozen snapshot of the
agent's own policy. Three things make that a fair fight rather than a
scripted opponent in disguise:

  * AC2 observes through its own radar. An observer runs the agent's exact
    observation, mask and command code on the world's telemetry(2) — the
    same packet the agent gets, seen from AC2 — with its own RadarSim and
    track adapter. No truth reaches the snapshot's actor.
  * AC2's missiles are guided by AC2's own track (the world only falls back
    to truth guidance for scripted opponents), so the 3-second support rule
    binds both sides.
  * Fire is gated by the same _can_fire() rule the agent's mask uses.

Opponents per episode: with probability SCRIPTED_FRAC an ADAPTIVE_SHOOTER,
which keeps a committed aggressor in the mix so self-play cannot settle into
both sides turning cold at long range. Otherwise a snapshot from the pool:
the newest with probability NEWEST_FRAC, else one drawn uniformly, so the
agent keeps beating its older selves while facing its current one.

The trainer owns the pool (snapshot()); environments only read it.
"""

import os
import glob
import time
from collections import OrderedDict

import numpy as np

N_STACK = 8                 # must equal train_bvr.N_STACK
SCRIPTED_FRAC = 1.0 / 3.0
NEWEST_FRAC = 0.5
MAX_POOL = 12               # snapshots kept on disk
_CACHE_MAX = 6              # loaded policies kept per process


class FrameStacker:
    """
    Reproduces stable_baselines3.common.vec_env.stacked_observations.
    StackedObservations for a single, non-vectorized env — verified
    byte-for-byte identical to it (same seed, same action sequence, both
    with and without a reset mid-sequence).

    Used wherever a policy is driven one env step at a time outside a
    VecEnv: the GUI's evaluation loop and the self-play opponent.
    """
    def __init__(self, n_stack: int):
        self.n_stack = n_stack
        self._buf: dict = {}

    def _ensure(self, key, dim, dtype):
        cur = self._buf.get(key)
        if cur is None or cur.shape[0] != dim * self.n_stack:
            self._buf[key] = np.zeros(dim * self.n_stack, dtype=dtype)

    def reset(self, obs):
        if isinstance(obs, dict):
            return {k: self._reset_one(k, v) for k, v in obs.items()}
        return self._reset_one("__box__", obs)

    def _reset_one(self, key, arr):
        arr = np.asarray(arr)
        self._ensure(key, arr.shape[0], arr.dtype)
        buf = self._buf[key]
        buf[:] = 0
        buf[-arr.shape[0]:] = arr
        return buf.copy()

    def update(self, obs):
        if isinstance(obs, dict):
            return {k: self._update_one(k, v) for k, v in obs.items()}
        return self._update_one("__box__", obs)

    def _update_one(self, key, arr):
        arr = np.asarray(arr)
        dim = arr.shape[0]
        buf = self._buf[key]
        buf[:] = np.roll(buf, -dim)
        buf[-dim:] = arr
        return buf.copy()


# ── pool ──────────────────────────────────────────────────────────────
def pool_snapshots(pool_dir: str) -> list:
    """Snapshot paths, oldest first (names embed the save time)."""
    return sorted(glob.glob(os.path.join(pool_dir, "sp_*.zip")))


def snapshot(model, pool_dir: str) -> str:
    """Save the current policy into the pool and prune the oldest."""
    os.makedirs(pool_dir, exist_ok=True)
    path = os.path.join(pool_dir, f"sp_{int(time.time())}_{int(model.num_timesteps):010d}")
    model.save(path)
    for old in pool_snapshots(pool_dir)[:-MAX_POOL]:
        try:
            os.remove(old)
        except OSError:
            pass
    return path + ".zip"


_CACHE: "OrderedDict[str, tuple]" = OrderedDict()


def load_policy(path: str):
    """(model, privileged) for a snapshot, cached per process."""
    hit = _CACHE.get(path)
    if hit is not None:
        _CACHE.move_to_end(path)
        return hit
    import torch
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO
    # Each vec-env worker would otherwise start a full-width torch thread pool
    # for a batch-of-one forward pass, oversubscribing the CPU many times over.
    torch.set_num_threads(1)
    model = MaskablePPO.load(path, device="cpu")
    entry = (model, isinstance(model.observation_space, spaces.Dict))
    _CACHE[path] = entry
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return entry


def make_selfplay_opponent(env, pool_dir: str, rng):
    """Pick this episode's SELF_PLAY opponent for `env` (a BvrEnv)."""
    from bvr_opponents import BvrOpponent, BvrOpponentType
    if rng.random() < SCRIPTED_FRAC:
        opp = BvrOpponent.create(BvrOpponentType.ADAPTIVE_SHOOTER, rng=rng)
        opp.label = "ADAPTIVE_SHOOTER"
        return opp
    paths = pool_snapshots(pool_dir) if pool_dir else []
    if not paths:
        raise RuntimeError(
            f"SELF_PLAY needs at least one policy snapshot in {pool_dir!r}; "
            "train_bvr.py writes one when the SELF_PLAY stage begins")
    if rng.random() < NEWEST_FRAC:
        path = paths[-1]
    else:
        path = paths[int(rng.integers(len(paths)))]
    model, privileged = load_policy(path)
    return PolicyOpponent(env, model, privileged,
                          label="policy:" + os.path.basename(path)[:-4])


# ── opponent ──────────────────────────────────────────────────────────
class PolicyOpponent:
    """AC2 flown by a frozen policy that sees the fight through its own radar."""

    def __init__(self, env, model, privileged: bool, label: str):
        self._env = env
        self._model = model
        self._obs = env.selfplay_observer(privileged)
        self._stack = FrameStacker(N_STACK)
        self.label = label
        self.rmax_t = None          # set by the env; unused here

    def reset(self, ic):
        o = self._obs
        o._track.reset()
        if o._radar is not None:
            o._radar.reset()
        o._state = {}
        o._last_shot_t = -999.0
        o._last_convert_t = -1e9
        self._next_decision = -1.0
        self._started = False
        self._cmd = None

    def act(self, state, t_sim):
        env, o = self._env, self._obs
        world = env._world

        # Same sensor cadence as the agent: the track clock ticks every frame,
        # the radar converts at TRACK_CONVERT_HZ.
        o._track.tick(world.SIM_DT)
        if (o._track.t_now - o._last_convert_t) >= 1.0 / env.TRACK_CONVERT_HZ:
            conv_dt = min(o._track.t_now - o._last_convert_t, 1.0)
            o._last_convert_t = o._track.t_now
            o._state = world.telemetry(2)
            o._radar_update(dt=conv_dt)

        fire = 0
        if t_sim >= self._next_decision:
            o._state = world.telemetry(2)
            o._t_sim = t_sim
            o._step_num = env._step_num
            raw = o._build_obs()
            obs = self._stack.update(raw) if self._started else self._stack.reset(raw)
            self._started = True
            mask = o.action_masks()
            a, _ = self._model.predict(obs, action_masks=mask, deterministic=False)
            a = [int(x) for x in np.asarray(a).reshape(-1)]
            self._cmd = o._encode_cmd(a[0], a[1], a[2], 0)
            if a[3] == 1 and mask[-1]:
                fire = 1
                o._last_shot_t = t_sim
            self._next_decision = t_sim + 1.0 / env.DECISION_HZ

        cmd = dict(self._cmd)
        cmd["fire"] = fire
        # The key must be present either way: without it the world falls back
        # to truth guidance, which is what scripted opponents get.
        if any(m.owner == 2 for m in world.missiles):
            cmd["msl_guidance"] = o._guidance_packet()
        else:
            cmd["msl_guidance"] = {"valid": 0}
        return cmd
