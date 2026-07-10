"""
Thumper's Critic: two standalone value estimators trained on imagined rollouts.

Deliberately not `Policy.value_head` -- that head shares the actor's trunk,
so its gradients would shape actor features. The critic here reads the same
world-model feature (deter ++ stoch ++ macro_context) but through its own
parameters, so actor and critic gradients stay disjoint (see
`training/actor_critic.py`). A target critic (hard-synced copy, see
`Thumper.sync_critic_target`) provides the bootstrap values for lambda-returns.

Two-stream split (tickets/0005): extrinsic (reward) and intrinsic
(disagreement) returns get separate heads, `ext_net`/`int_net`, not a shared
trunk with two output units. The intrinsic value function is deliberately
nonstationary (disagreement decays as the world model improves), and its
drift must not perturb the sparse, stationary extrinsic estimate through
shared hidden layers.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from model.world_model import _mlp_head


@dataclass
class CriticConfig:
    feature_dim: int = 416
    """Width of the world-model feature (deter ++ stoch ++ macro_context);
    derived in `ThumperConfig.__post_init__`, never trusted by hand."""
    hidden_dim: int = 256


class Critic(nn.Module):
    def __init__(self, cfg: CriticConfig | None = None):
        super().__init__()
        self.cfg = cfg or CriticConfig()
        self.ext_net = _mlp_head(self.cfg.feature_dim, self.cfg.hidden_dim, 1)
        self.int_net = _mlp_head(self.cfg.feature_dim, self.cfg.hidden_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        """(..., feature_dim) -> (..., 2) value estimates: channel 0 is the
        extrinsic (reward) value, channel 1 is the intrinsic (disagreement)
        value -- this channel order is a cross-file contract with
        `training/actor_critic.py`."""
        return torch.cat([self.ext_net(features), self.int_net(features)], dim=-1)
