"""
bvr_policy.py  —  Privileged (asymmetric) actor-critic for MaskablePPO
======================================================================

Actor  sees obs["obs"]   — radar/RWR-derived only. This is what deploys.
Critic sees obs["obs"] + obs["priv"] — plus ground truth.

Why bother
----------
Under partial observability the value function has to average over everything
the actor cannot see. In BVR that is a lot: the bandit's true heading, its
remaining missiles, whether it has a lock on us. The resulting value estimate
is high-variance, which makes every advantage estimate high-variance, which
makes PPO updates noisy in exactly the way you saw in WVR.

Giving the critic truth collapses that variance without changing what the
actor can do. The critic is discarded at deployment, so this is not cheating —
it is the standard asymmetric-information setup from Pinto et al. / OpenAI's
Dota work, and it is much cheaper to build now than to retrofit later.

Cost: one extra network branch. Benefit in my experience on this class of
problem: noticeably faster and steadier value convergence.

Requires: sb3-contrib (MaskablePPO).
"""

import torch as th
import torch.nn as nn
import numpy as np
from gymnasium import spaces

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SplitExtractor(BaseFeaturesExtractor):
    """
    Passthrough extractor. Keeps the two observation branches separate so the
    policy can route them to different heads. SB3's default MultiInput
    extractor concatenates everything and feeds it to both networks, which
    would leak truth into the actor — the exact thing we must avoid.
    """

    def __init__(self, observation_space: spaces.Dict):
        obs_dim = int(np.prod(observation_space["obs"].shape))
        priv_dim = int(np.prod(observation_space["priv"].shape))
        super().__init__(observation_space, features_dim=obs_dim + priv_dim)
        self.obs_dim = obs_dim
        self.priv_dim = priv_dim

    def forward(self, obs) -> th.Tensor:
        return th.cat([obs["obs"].float(), obs["priv"].float()], dim=1)


# Inputs that change sign in a mirror image of the fight (left and right swapped), and the heading choice
# each heading choice becomes (HDG_OFFSETS_DEG: +k <-> -k; 0 and 180 stay). Everything else is unchanged by
# a mirror: the simulator, its opponents and its starts are all mirror-symmetric.
MIRROR_LABELS = ("own_phi_sin", "trk_ata_sin", "rwr_bearing_sin", "wm_brg_sin", "wm_hdg_sin",
                 "hdg_prev_sin", "hdg_turn_sin", "trk_aspect_side", "los_rate_signed")


def mirror_columns(obs_width: int) -> np.ndarray:
    """The stacked observation's columns (N_STACK frames of OBS_LABELS) that change sign in a mirror."""
    from bvr_env import OBS_LABELS, OBS_DIM
    if obs_width % OBS_DIM:
        raise ValueError(f"observation width {obs_width} is not a whole number of {OBS_DIM}-input frames")
    idx = [OBS_LABELS.index(k) for k in MIRROR_LABELS if k in OBS_LABELS]
    return np.array([f * OBS_DIM + i for f in range(obs_width // OBS_DIM) for i in idx], dtype=np.int64)


def heading_mirror() -> np.ndarray:
    """For each heading choice, the index of its mirror image."""
    from bvr_env import HDG_OFFSETS_DEG
    off = list(HDG_OFFSETS_DEG)
    return np.array([off.index(-o if abs(o) not in (0.0, 180.0) else o) for o in off], dtype=np.int64)


class AsymmetricMaskablePolicy(MaskableActorCriticPolicy):
    """
    Splits the concatenated features: actor gets the first obs_dim columns,
    critic gets everything.

    symmetric=True (SIM_REV 20, train_bvr.py --symmetric-policy): the actor's choice is the average of its
    logits for the fight as seen and, mirrored back, for the fight's mirror image. A mirror-symmetric
    world gives no reason to prefer a side, and a policy left to itself settles on one: dcsfit_v4 and v5
    cranked to the same side in 99% of their choices whatever the geometry (a mirror image of the inputs
    flipped 2-3%). Averaged, a choice that is not tied to the geometry comes out even, so the side has to
    come from the inputs that tell left from right (bearing, crossing, line-of-sight rate). The critic is
    left as it is. Stored in the checkpoint's policy_kwargs; off for older checkpoints.
    """

    def __init__(self, *args,
                 pi_arch=(256, 256),
                 vf_arch=(256, 256),
                 symmetric=False,
                 **kwargs):
        self._pi_arch = tuple(pi_arch)
        self._vf_arch = tuple(vf_arch)
        self.symmetric = bool(symmetric)
        kwargs["features_extractor_class"] = SplitExtractor
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        ext = self.features_extractor
        self.mlp_extractor = _AsymmetricMlpExtractor(
            obs_dim=ext.obs_dim,
            full_dim=ext.obs_dim + ext.priv_dim,
            pi_arch=self._pi_arch,
            vf_arch=self._vf_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        self.mlp_extractor.set_symmetric(self.symmetric)
        self._hdg_mirror = th.as_tensor(heading_mirror(), device=self.device)

    def set_symmetric(self, on: bool) -> None:
        """Turn the mirror averaging on or off (a resumed checkpoint; the trainer also records it in
        model.policy_kwargs so the checkpoints it saves keep it)."""
        self.symmetric = bool(on)
        self.mlp_extractor.set_symmetric(self.symmetric)

    def _get_action_dist_from_latent(self, latent_pi):
        if not self.symmetric:
            return super()._get_action_dist_from_latent(latent_pi)
        half = latent_pi.shape[1] // 2
        logits = self.action_net(latent_pi[:, :half])
        logits_m = self.action_net(latent_pi[:, half:])
        n_h = len(self._hdg_mirror)
        hdg_m = logits_m[:, :n_h][:, self._hdg_mirror.to(logits_m.device)]
        logits_m = th.cat([hdg_m, logits_m[:, n_h:]], dim=1)
        return self.action_dist.proba_distribution(action_logits=0.5 * (logits + logits_m))


class _AsymmetricMlpExtractor(nn.Module):

    def __init__(self, obs_dim, full_dim, pi_arch, vf_arch, activation_fn, device):
        super().__init__()
        self.obs_dim = obs_dim

        def mlp(in_dim, arch):
            layers, d = [], in_dim
            for h in arch:
                layers += [nn.Linear(d, h), activation_fn()]
                d = h
            return nn.Sequential(*layers), d

        self.policy_net, self.latent_dim_pi = mlp(obs_dim, pi_arch)
        self.value_net, self.latent_dim_vf = mlp(full_dim, vf_arch)
        self.to(device)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def set_symmetric(self, on: bool) -> None:
        self.symmetric = bool(on)
        if self.symmetric and not hasattr(self, "_mirror_cols"):
            cols = mirror_columns(self.obs_dim)
            sign = th.ones(self.obs_dim)
            sign[th.as_tensor(cols)] = -1.0
            self.register_buffer("_mirror_sign", sign.to(next(self.parameters()).device), persistent=False)
            self._mirror_cols = cols

    def forward_actor(self, features):
        # Hard slice: the actor physically cannot see privileged columns.
        x = features[:, :self.obs_dim]
        if not getattr(self, "symmetric", False):
            return self.policy_net(x)
        # The latent of the fight as seen and of its mirror image, side by side (the policy averages their
        # logits: AsymmetricMaskablePolicy._get_action_dist_from_latent).
        return th.cat([self.policy_net(x), self.policy_net(x * self._mirror_sign)], dim=1)

    def forward_critic(self, features):
        return self.value_net(features)
