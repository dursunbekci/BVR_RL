"""
bvr_team_vec.py  —  Team environments as a Stable-Baselines3 VecEnv
===================================================================

Stable-Baselines3 trains one policy on a vector of single-agent slots. Here
each blue aircraft of each team environment (bvr_team.TeamBvrEnv) is one
slot, so the one policy flies every aircraft (parameter sharing) and every
aircraft's experience trains it. With n team environments of 2 agents the
trainer sees 2n slots, in the order team 0 agent 0, team 0 agent 1, team 1
agent 0, ...

A team's slots end their episodes together; the slot then carries the usual
info["terminal_observation"] and the team is reset. action_masks() returns
one row per slot, which is what MaskablePPO asks for.

    vec = TeamVecEnv([fn0, fn1, ...])                  # one worker process per team
    vec = TeamVecEnv([fn0], in_process=True)           # no worker processes

set_attr() and get_attr() act on the team environments, so the trainer's
curriculum (set_attr("_opponent_type", ...)) works unchanged.
"""

import multiprocessing as mp
import traceback

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper


def _team_step(env, actions):
    obs, rews, term, trunc, infos = env.step(actions)
    if term or trunc:
        for k, info in enumerate(infos):
            info["terminal_observation"] = obs[k]
            info["TimeLimit.truncated"] = bool(trunc and not term)
        obs, _ = env.reset()
    return obs, rews, bool(term or trunc), infos


def _worker(remote, parent_remote, fn_wrapper):
    parent_remote.close()
    env = fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        try:
            if cmd == "step":
                out = _team_step(env, data)
            elif cmd == "reset":
                out = env.reset()[0]
            elif cmd == "masks":
                out = env.action_masks()
            elif cmd == "spaces":
                out = (env.observation_space, env.action_space, env.n_agents)
            elif cmd == "get_attr":
                out = getattr(env, data)
            elif cmd == "has_attr":
                out = hasattr(env, data)
            elif cmd == "set_attr":
                setattr(env, data[0], data[1]); out = None
            elif cmd == "method":
                name, args, kwargs = data
                out = getattr(env, name)(*args, **kwargs)
            elif cmd == "close":
                env.close(); remote.close()
                break
            else:
                raise NotImplementedError(cmd)
            remote.send(("ok", out))
        except Exception as e:                     # report, don't die silently
            traceback.print_exc()
            remote.send(("error", e))


class TeamVecEnv(VecEnv):

    def __init__(self, env_fns, in_process=False, start_method=None):
        self._in_process = bool(in_process)
        self.n_teams = len(env_fns)
        if self._in_process:
            self._envs = [fn() for fn in env_fns]
            e = self._envs[0]
            obs_space, act_space, self.n_agents = e.observation_space, e.action_space, e.n_agents
        else:
            if start_method is None:
                start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
            ctx = mp.get_context(start_method)
            self._remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_teams)])
            self._procs = []
            for wr, r, fn in zip(work_remotes, self._remotes, env_fns):
                p = ctx.Process(target=_worker, args=(wr, r, CloudpickleWrapper(fn)), daemon=True)
                p.start()
                self._procs.append(p)
                wr.close()
            obs_space, act_space, self.n_agents = self._call(0, "spaces")
        self._closed = False
        self._actions = None
        super().__init__(self.n_teams * self.n_agents, obs_space, act_space)

    # ── plumbing ──────────────────────────────────────────────────────
    def _send(self, t, cmd, data=None):
        self._remotes[t].send((cmd, data))

    def _recv(self, t):
        status, out = self._remotes[t].recv()
        if status == "error":
            raise out
        return out

    def _call(self, t, cmd, data=None):
        self._send(t, cmd, data)
        return self._recv(t)

    def _all(self, cmd, data_per_team=None):
        """Run cmd on every team (in parallel with worker processes)."""
        if self._in_process:
            return [self._local(e, cmd, None if data_per_team is None else data_per_team[t])
                    for t, e in enumerate(self._envs)]
        for t in range(self.n_teams):
            self._send(t, cmd, None if data_per_team is None else data_per_team[t])
        return [self._recv(t) for t in range(self.n_teams)]

    @staticmethod
    def _local(env, cmd, data):
        if cmd == "step":     return _team_step(env, data)
        if cmd == "reset":    return env.reset()[0]
        if cmd == "masks":    return env.action_masks()
        if cmd == "get_attr": return getattr(env, data)
        if cmd == "set_attr": setattr(env, data[0], data[1]); return None
        if cmd == "method":
            name, args, kwargs = data
            return getattr(env, name)(*args, **kwargs)
        raise NotImplementedError(cmd)

    def _stack(self, per_team):
        """[[obs agent 0, obs agent 1], ...] -> the VecEnv batch."""
        flat = [o for team in per_team for o in team]
        if isinstance(self.observation_space, spaces.Dict):
            return {k: np.stack([o[k] for o in flat]) for k in self.observation_space.spaces}
        return np.stack(flat)

    def _teams_of(self, indices):
        idx = range(self.num_envs) if indices is None else (
            [indices] if isinstance(indices, int) else indices)
        return [i // self.n_agents for i in idx]

    # ── VecEnv ────────────────────────────────────────────────────────
    def reset(self):
        return self._stack(self._all("reset"))

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(self.n_teams, self.n_agents, -1)

    def step_wait(self):
        out = self._all("step", list(self._actions))
        obs = self._stack([o[0] for o in out])
        rews = np.concatenate([np.asarray(o[1], dtype=np.float32) for o in out])
        dones = np.repeat(np.array([o[2] for o in out]), self.n_agents)
        infos = [info for o in out for info in o[3]]
        return obs, rews, dones, infos

    def action_masks(self):
        return np.concatenate(self._all("masks"))

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            rows = self.action_masks()
            idx = range(self.num_envs) if indices is None else indices
            return [rows[i] for i in idx]
        teams = self._teams_of(indices)
        uniq = sorted(set(teams))
        res = {}
        for t in uniq:
            data = (method_name, method_args, method_kwargs)
            res[t] = (self._local(self._envs[t], "method", data) if self._in_process
                      else self._call(t, "method", data))
        return [res[t] for t in teams]

    def get_attr(self, attr_name, indices=None):
        teams = self._teams_of(indices)
        res = {t: (getattr(self._envs[t], attr_name) if self._in_process
                   else self._call(t, "get_attr", attr_name)) for t in sorted(set(teams))}
        return [res[t] for t in teams]

    def has_attr(self, attr_name):
        # Answered in the worker: fetching a bound method (MaskablePPO asks for
        # "action_masks") would mean pickling the whole environment.
        if self._in_process:
            return hasattr(self._envs[0], attr_name)
        return self._call(0, "has_attr", attr_name)

    def set_attr(self, attr_name, value, indices=None):
        for t in sorted(set(self._teams_of(indices))):
            if self._in_process:
                setattr(self._envs[t], attr_name, value)
            else:
                self._call(t, "set_attr", (attr_name, value))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._teams_of(indices))

    def seed(self, seed=None):
        return [None] * self.num_envs

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._in_process:
            for e in self._envs:
                e.close()
            return
        for r in self._remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
