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


class AsymmetricMaskablePolicy(MaskableActorCriticPolicy):
    """
    Splits the concatenated features: actor gets the first obs_dim columns,
    critic gets everything.
    """

    def __init__(self, *args,
                 pi_arch=(256, 256),
                 vf_arch=(256, 256),
                 **kwargs):
        self._pi_arch = tuple(pi_arch)
        self._vf_arch = tuple(vf_arch)
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

    def forward_actor(self, features):
        # Hard slice: the actor physically cannot see privileged columns.
        return self.policy_net(features[:, :self.obs_dim])

    def forward_critic(self, features):
        return self.value_net(features)
