"""
Actor-critic training on imagined rollouts (DreamerV2-style), the with-grad
second pass over `Thumper.dream`'s stored (grad-free) output.

Kept as pure functions (plus one small stateful normalizer) rather than
methods on Thumper/Trainer, mirroring how `WorldModel.compute_losses` owns
the world-model math -- this way the return computation and loss math unit
test in isolation from the env/replay-buffer machinery.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from model.critic import Critic
from model.policy import Policy


def lambda_returns(rewards: Tensor, discounts: Tensor, values: Tensor, lam: float) -> Tensor:
    """Dreamer-style TD(lambda), computed backward.

    rewards, discounts: (N, H). values: (N, H+1) -- target-critic values at
    dream states 0..H. Returns (N, H): returns[:, t] is the lambda-return
    *from* state t.
    """
    H = rewards.shape[1]
    returns = torch.zeros_like(rewards)
    next_return = values[:, -1]
    for t in reversed(range(H)):
        bootstrap = (1 - lam) * values[:, t + 1] + lam * next_return
        next_return = rewards[:, t] + discounts[:, t] * bootstrap
        returns[:, t] = next_return
    return returns


class ReturnNormalizer:
    """DreamerV3-style running scale on imagined returns, so advantages
    don't vanish as the world model improves and raw disagreement shrinks
    (`wm/disagreement_mean` sits around ~0.009 and keeps falling) -- without
    this the entropy term would dominate the actor loss. The scale is a
    single float that must survive checkpoint resume."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.scale = 1.0

    def update(self, returns: Tensor) -> None:
        spread = (
            torch.quantile(returns.detach(), 0.95) - torch.quantile(returns.detach(), 0.05)
        ).item()
        self.scale = self.decay * self.scale + (1 - self.decay) * spread

    def normalize(self, adv: Tensor) -> Tensor:
        return adv / max(1.0, self.scale)

    def state_dict(self) -> dict:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict) -> None:
        self.scale = state["scale"]


@dataclass
class ActorCriticConfig:
    gamma: float = 0.997
    return_lambda: float = 0.95
    entropy_scale: float = 1e-3
    intrinsic_scale: float = 1.0


def actor_critic_losses(
    dream: dict[str, Tensor],
    policy: Policy,
    critic: Critic,
    critic_target: Critic,
    normalizer: ReturnNormalizer,
    cfg: ActorCriticConfig,
) -> dict[str, Tensor]:
    """The with-grad second pass over a grad-free `Thumper.dream` rollout.

    Re-evaluates the dream's stored actions under `policy` (for log-probs
    and entropy) and `critic` (for the value baseline), so gradients flow
    only into `policy`/`critic` -- `dream`'s tensors carry no autograd graph
    to begin with (world model + old policy sample were under no_grad).
    """
    features = dream["features"]  # (N, H+1, feature_dim)
    N, H = dream["action_types"].shape

    rewards = dream["reward"] + cfg.intrinsic_scale * dream["intrinsic"]
    discounts = cfg.gamma * dream["continue_prob"]

    with torch.no_grad():
        target_values = critic_target(features)  # (N, H+1)
    returns = lambda_returns(rewards, discounts, target_values, cfg.return_lambda)  # (N, H)

    values = critic(features[:, :-1])  # (N, H), with grad
    critic_loss = F.mse_loss(values, returns.detach())

    features_flat = features[:, :-1].reshape(N * H, -1)
    action_types_flat = dream["action_types"].reshape(N * H)
    coords_flat = dream["coords"].reshape(N * H, 2)
    available_actions = dream["available_actions"]
    mask_flat = available_actions.unsqueeze(1).expand(N, H, -1).reshape(N * H, -1)
    log_prob, entropy = policy.log_prob_entropy(features_flat, action_types_flat, coords_flat, mask_flat)

    baseline = values.detach()
    advantage = normalizer.normalize(returns - baseline)
    actor_loss = -(log_prob * advantage.detach().flatten()).mean() - cfg.entropy_scale * entropy.mean()

    normalizer.update(returns)

    return {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy": entropy.mean().detach(),
        "imagined_return": returns.mean().detach(),
        "intrinsic_mean": dream["intrinsic"].mean().detach(),
        "extrinsic_mean": dream["reward"].mean().detach(),
        "value_mean": values.mean().detach(),
    }
