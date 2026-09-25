"""
bvr_compat.py  —  Load checkpoints saved before an observation input was added
==============================================================================

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
"""

import os
import tempfile

import numpy as np
import torch as th
from gymnasium import spaces

from bvr_env import OBS_DIM, PRIV_DIM
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
    if _widths(old_space) == _widths(new_space):
        return False
    maps = _column_maps(old_space, new_space)

    pol = params["policy"]
    widened = {}
    for name, t in pol.items():
        if _is_input_layer(name, t, maps):
            widened[tuple(t.shape)] = True
            pol[name] = _widen(t, maps)
    if not widened:
        raise ValueError("found no input layer to widen")

    # Adam state is keyed by parameter index, not name; the input layers are
    # the only 2-D parameters whose width is an input width.
    opt = params.get("policy.optimizer")
    if opt:
        for st in opt["state"].values():
            for k in ("exp_avg", "exp_avg_sq"):
                t = st.get(k)
                if t is not None and tuple(t.shape) in widened:
                    st[k] = _widen(t, maps)

    data["observation_space"] = new_space
    for k in ("_last_obs", "_last_original_obs"):
        if k in data:
            data[k] = None
    save_to_zip_file(out_path, data=data, params=params, pytorch_variables=pt_vars)
    return True


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
                log(f"[bvr] {os.path.basename(path)}: older observation layout, widened "
                    f"to {OBS_DIM} inputs per frame, {PRIV_DIM} for the critic (new inputs "
                    f"start at zero weight)")
            return MaskablePPO.load(tmp, env=env, **kwargs)
        return MaskablePPO.load(path, env=env, **kwargs)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
