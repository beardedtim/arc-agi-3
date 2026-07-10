"""
Thumper's Critic: a standalone value estimator trained on imagined rollouts.

Deliberately not `Policy.value_head` -- that head shares the actor's trunk,
so its gradients would shape actor features. The critic here reads the same
world-model feature (deter ++ stoch ++ macro_context) but through its own
parameters, so actor and critic gradients stay disjoint (see
`training/actor_critic.py`). A target critic (hard-synced copy, see
`Thumper.sync_critic_target`) provides the bootstrap values for lambda-returns.
"""
from dataclasses import dataclass

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
        self.net = _mlp_head(self.cfg.feature_dim, self.cfg.hidden_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        """(..., feature_dim) -> (...,) value estimate."""
        return self.net(features).squeeze(-1)
