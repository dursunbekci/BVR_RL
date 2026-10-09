"""
bvr_compat.py  —  Load checkpoints saved before the observation or the actions changed
=====================================================================================

Adding an input to the observation (e.g. doctrine_shot_cost) makes every
earlier checkpoint unloadable: its first layers expect a narrower input. This
module widens such a checkpoint so it loads and acts exactly as before.

The network input is N_STACK frames of the observation, frame after frame
(then, with the privileged critic, N_STACK frames of "priv"). New inputs are
always appended to the END of each frame (see OBS_LABELS and PRIV_LABELS), so
an old column maps to a known new column and the new columns get zero weights. A zero weight
means the new input has no effect until training gives it one: the widened
policy's action probabilities are identical to the old policy's.

Adam's moment estimates for those layers are widened the same way, so
resuming training continues the old optimizer state rather than resetting it.

The action space grew in SIM_REV 14 (headings ±15 and ±40, a fifth speed).
The action head of an older checkpoint is converted the same way: each old
choice keeps its own output, and a new choice gets the average of the
outputs of its neighbours (±15 between 0 and ±30, ±40 between ±30 and ±50,
a new speed from the nearest old one, by the speeds in the checkpoint's
scenario record), a quarter as likely as they are. The
converted policy picks what it picked before (its most likely choice is
unchanged) and tries the new choices now and then until training finds a use
for them.
"""

import math
import os
import tempfile

import numpy as np
import torch as th
from gymnasium import spaces

from bvr_env import OBS_DIM, PRIV_DIM, ACTION_NVEC, HDG_OFFSETS_DEG
from bvr_selfplay import N_STACK


def expected_space(privileged: bool):
    """The stacked observation space the current code feeds the policy."""
    box = lambda n: spaces.Box(low=-1.0, high=1.0, shape=(n * N_STACK,), dtype=np.float32)
    return spaces.Dict({"obs": box(OBS_DIM), "priv": box(PRIV_DIM)}) if privileged else box(OBS_DIM)


def _widths(space):
    """(obs width, priv width) of a stacked observation space."""
    if isinstance(space, spaces.Dict):
        return int(space["obs"].shape[0]), int(space["priv"].shape[0])
    return int(space.shape[0]), 0


def _column_maps(old_space, new_space):
    """{old input width: (new column of each old column, new input width)}."""
    o_old, p_old = _widths(old_space)
    o_new, p_new = _widths(new_space)
    if o_old % N_STACK or p_old % N_STACK or bool(p_old) != bool(p_new):
        raise ValueError(f"cannot migrate: checkpoint input obs={o_old} priv={p_old}, "
                         f"current code obs={o_new} priv={p_new}")

    def frame_map(w_old, w_new, what):
        d_old, d_new = w_old // N_STACK, w_new // N_STACK
        if d_old > d_new:
            raise ValueError(f"checkpoint has {d_old} {what} inputs per frame, the code "
                             f"only {d_new}: it was trained by newer code")
        c = np.arange(w_old)
        return (c // d_old) * d_new + c % d_old

    obs_map = frame_map(o_old, o_new, "observation")
    # The actor's first layer takes the obs frames only, the critic's (with a
    # privileged critic) obs then priv frames: two widths, two maps.
    maps = {o_old: (obs_map, o_new)}
    if p_old:
        priv_map = frame_map(p_old, p_new, "critic")
        maps[o_old + p_old] = (np.concatenate([obs_map, o_new + priv_map]), o_new + p_new)
    return maps


def _widen(t, maps):
    """A copy of 2-D tensor t with its input columns moved to their new place."""
    col, new_w = maps[t.shape[1]]
    out = th.zeros((t.shape[0], new_w), dtype=t.dtype)
    out[:, th.as_tensor(col)] = t
    return out


def _is_input_layer(name, t, maps):
    return (t.dim() == 2 and t.shape[1] in maps
            and name.startswith("mlp_extractor.") and name.endswith(".0.weight"))


def migrate(path: str, out_path: str) -> bool:
    """
    Write a widened copy of checkpoint `path` to `out_path`.
    Returns False (and writes nothing) if it already matches the current code.
    """
    from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
    data, params, pt_vars = load_from_zip_file(path, device="cpu")
    old_space = data["observation_space"]
    new_space = expected_space(isinstance(old_space, spaces.Dict))
    old_nvec = [int(n) for n in data["action_space"].nvec]
    obs_change = _widths(old_space) != _widths(new_space)
    act_change = old_nvec != list(ACTION_NVEC)
    if not (obs_change or act_change):
        return False
    pol = params["policy"]
    opt = params.get("policy.optimizer")

    if obs_change:
        maps = _column_maps(old_space, new_space)
        widened = {}
        for name, t in pol.items():
            if _is_input_layer(name, t, maps):
                widened[tuple(t.shape)] = True
                pol[name] = _widen(t, maps)
        if not widened:
            raise ValueError("found no input layer to widen")

        # Adam state is keyed by parameter index, not name; the input layers are
        # the only 2-D parameters whose width is an input width.
        if opt:
            for st in opt["state"].values():
                for k in ("exp_avg", "exp_avg_sq"):
                    t = st.get(k)
                    if t is not None and tuple(t.shape) in widened:
                        st[k] = _widen(t, maps)
        data["observation_space"] = new_space

    if act_change:
        rows = _action_rows(old_nvec, _speeds_of(data, old_nvec[2]))
        n_old = sum(old_nvec)
        w, b = pol["action_net.weight"], pol["action_net.bias"]
        if w.shape[0] != n_old or b.shape[0] != n_old:
            raise ValueError(f"action head has {w.shape[0]} outputs, the action space {n_old}")
        shapes = {tuple(w.shape), tuple(b.shape)}
        pol["action_net.weight"] = _convert_head(w, rows)
        pol["action_net.bias"] = _convert_head(b, rows)
        # Adam: the action head's are the only moments with its output count.
        if opt:
            for st in opt["state"].values():
                for k in ("exp_avg", "exp_avg_sq"):
                    t = st.get(k)
                    if t is not None and tuple(t.shape) in shapes:
                        st[k] = _convert_head(t, rows, offset=False)
        data["action_space"] = spaces.MultiDiscrete(np.array(ACTION_NVEC))
    for k in ("_last_obs", "_last_original_obs"):
        if k in data:
            data[k] = None
    save_to_zip_file(out_path, data=data, params=params, pytorch_variables=pt_vars)
    return True


# Heading choices of each action space so far, by their number.
HDG_LAYOUTS = {
    10: [0.0, 30.0, -30.0, 50.0, -50.0, 90.0, -90.0, 135.0, -135.0, 180.0],   # to SIM_REV 13
    len(HDG_OFFSETS_DEG): list(HDG_OFFSETS_DEG),
}
# A new choice starts this much less likely than its neighbours (a logit offset).
NEW_CHOICE_LOGIT = -math.log(4.0)


def _speed_rows(old_speeds, new_speeds):
    """For each current speed choice: (old speed choice, logit offset). By value:
    each new speed takes the nearest old one's output, unchanged if it is the
    same speed, a quarter as likely if not (the speed added in SIM_REV 14)."""
    out = []
    for v in new_speeds:
        i = int(np.argmin([abs(v - o) for o in old_speeds]))
        out.append((i, 0.0 if abs(v - old_speeds[i]) < 1.0 else NEW_CHOICE_LOGIT))
    return out


def _speeds_of(data, n_old):
    """(old speeds, current speeds) of the checkpoint's own platform, from its
    scenario record; None if unknown (then a new speed is taken to be the
    fastest, after the old ones)."""
    try:
        from bvr_library import load_platform
        rec = data.get("bvr_scenario") or {}
        old = [float(x) for x in rec["items"]["platform"]["platform"]["params"]["SPEED_CMDS"]]
        new = [float(x) for x in load_platform(rec["platform"]).speed_cmds]
        if len(old) == n_old and len(new) == ACTION_NVEC[2]:
            return old, new
    except Exception:
        pass
    return None


def _action_rows(old_nvec, speeds=None):
    """For each output of the current action head: (rows of the old head it is
    the average of, logit offset). speeds: (old, current) speed values, to map
    the speed choices by value."""
    old_nvec = [int(n) for n in old_nvec]
    if len(old_nvec) != len(ACTION_NVEC) or old_nvec[1] != ACTION_NVEC[1] \
            or old_nvec[3] != ACTION_NVEC[3] or old_nvec[2] > ACTION_NVEC[2] \
            or old_nvec[0] not in HDG_LAYOUTS:
        raise ValueError(f"cannot convert actions {old_nvec} to {ACTION_NVEC}")
    rows, base = [], 0
    old_h = HDG_LAYOUTS[old_nvec[0]]
    for v in HDG_OFFSETS_DEG:
        if v in old_h:
            rows.append(([old_h.index(v)], 0.0))
        else:
            lo = max((o for o in old_h if o < v), default=None)
            hi = min((o for o in old_h if o > v), default=None)
            nb = [old_h.index(o) for o in (lo, hi) if o is not None]
            rows.append((nb, NEW_CHOICE_LOGIT))
    base += old_nvec[0]
    rows += [([base + i], 0.0) for i in range(old_nvec[1])]           # altitude: unchanged
    base += old_nvec[1]
    n_sp = old_nvec[2]
    if speeds is not None:
        rows += [([base + i], o) for i, o in _speed_rows(*speeds)]
    else:                                                            # new speeds: the top one's
        rows += [([base + min(i, n_sp - 1)], 0.0 if i < n_sp else NEW_CHOICE_LOGIT)
                 for i in range(ACTION_NVEC[2])]
    base += n_sp
    rows += [([base + i], 0.0) for i in range(old_nvec[3])]          # fire: unchanged
    return rows


def _convert_head(t, rows, offset=True):
    """The new action head (weight or bias, or an Adam moment of one) from the old."""
    out = th.stack([t[r].mean(dim=0) for r, _ in rows])
    if offset and t.dim() == 1:
        out = out + th.as_tensor([o for _, o in rows], dtype=t.dtype)
    return out


def load_model(path: str, env=None, log=print, **kwargs):
    """
    MaskablePPO.load() that also accepts checkpoints from before the latest
    observation inputs were added. kwargs go to MaskablePPO.load().
    """
    from sb3_contrib import MaskablePPO
    fd, tmp = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        if migrate(path, tmp):
            if log:
                log(f"[bvr] {os.path.basename(path)}: from an older version of the code, "
                    f"converted: {OBS_DIM} inputs per frame, {PRIV_DIM} for the critic (new "
                    f"inputs start at zero weight); actions {ACTION_NVEC} (new choices start "
                    f"a quarter as likely as their neighbours)")
            return MaskablePPO.load(tmp, env=env, **kwargs)
        return MaskablePPO.load(path, env=env, **kwargs)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
