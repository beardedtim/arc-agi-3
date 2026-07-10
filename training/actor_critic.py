"""
Actor-critic training on imagined rollouts (DreamerV2-style), the with-grad
second pass over `Thumper.dream`'s stored (grad-free) output.

Kept as pure functions (plus one small stateful normalizer) rather than
methods on Thumper/Trainer, mirroring how `WorldModel.compute_losses` owns
the world-model math -- this way the return computation and loss math unit
test in isolation from the env/replay-buffer machinery.

Two-stream returns (tickets/0005): extrinsic (reward) and intrinsic
(disagreement) each get their own lambda-return, critic head, target-value
bootstrap, and `ReturnNormalizer` -- see `actor_critic_losses`. Splitting
them stops the (continuous, ~10x larger) intrinsic stream from drowning the
(sparse, O(1)) extrinsic stream inside one shared normalizer.

Absorbing-score extrinsic returns (tickets/0006): a predicted score is
absorbing for the extrinsic stream only -- once a dream banks one, the
discount chain for everything after it is multiplied toward zero, so a
single dream can claim at most ~one level completion's worth of extrinsic
return *from that rollout*. This caps hallucinated farming inside
imagination (the world model has seen very few real level-completion
transitions and the actor was learning to re-trigger the reward head on
states nothing real ever anchored) -- it is not a claim that the real
environment pays out only once per episode. Games routinely chain several
level completions in one episode (`arcengine`'s `next_level()` keeps
`state` at `NOT_FINISHED` and just advances `_current_level_index`/`_score`
until the last level); that multi-level structure is still learned, just
not from a single dream -- via the critic's bootstrapped value at the
dream's start state and via real-play buffer episodes, whose `continue`
target stays true across a level transition (only `result.done`, i.e.
WIN/GAME_OVER, terminates).
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
    """Weight on the *normalized* intrinsic advantage relative to the
    *normalized* extrinsic advantage (tickets/0005). Now a true weight
    between two same-scale (O(1)) streams, not a unit conversion between a
    sparse ~1-magnitude reward stream and a continuous ~10-magnitude
    disagreement stream -- see `ReturnNormalizer`'s `max(1, scale)` floor."""


def actor_critic_losses(
    dream: dict[str, Tensor],
    policy: Policy,
    critic: Critic,
    critic_target: Critic,
    normalizer_ext: ReturnNormalizer,
    normalizer_int: ReturnNormalizer,
    cfg: ActorCriticConfig,
) -> dict[str, Tensor]:
    """The with-grad second pass over a grad-free `Thumper.dream` rollout.

    Re-evaluates the dream's stored actions under `policy` (for log-probs
    and entropy) and `critic` (for the value baseline), so gradients flow
    only into `policy`/`critic` -- `dream`'s tensors carry no autograd graph
    to begin with (world model + old policy sample were under no_grad).

    Two-stream returns (tickets/0005): extrinsic (predicted reward) and
    intrinsic (ensemble disagreement) each get their own lambda-return,
    critic head (`Critic`'s channel 0 / channel 1), target-value bootstrap,
    and `ReturnNormalizer`. The actor's advantage is the weighted sum of the
    two *normalized* advantages -- this is what keeps a sparse +1 level
    completion visible against a continuous stream of exploration payment
    (see tickets/0005's Design principle 1).

    Absorbing scores (tickets/0006): the extrinsic stream's discount chain
    is multiplied by `(1 - clamp(reward, 0, 1))`, so a predicted score
    absorbs all extrinsic credit after it -- the intrinsic stream's
    discounts are untouched.
    """
    features = dream["features"]  # (N, H+1, feature_dim)
    N, H = dream["action_types"].shape

    discounts = cfg.gamma * dream["continue_prob"]

    # tickets/0006: a predicted score is absorbing for extrinsic credit --
    # one dream can bank at most ~one score. `dream["reward"]` is grad-free
    # by construction (Thumper.dream runs under no_grad), so this opens no
    # new gradient path into the reward head.
    absorb = 1.0 - dream["reward"].clamp(0.0, 1.0)
    discounts_ext = discounts * absorb

    with torch.no_grad():
        target_values = critic_target(features)  # (N, H+1, 2)
    returns_ext = lambda_returns(
        dream["reward"], discounts_ext, target_values[..., 0], cfg.return_lambda
    )  # (N, H)
    returns_int = lambda_returns(
        dream["intrinsic"], discounts, target_values[..., 1], cfg.return_lambda
    )  # (N, H)

    values = critic(features[:, :-1])  # (N, H, 2), with grad
    critic_loss = F.mse_loss(values[..., 0], returns_ext.detach()) + F.mse_loss(
        values[..., 1], returns_int.detach()
    )

    features_flat = features[:, :-1].reshape(N * H, -1)
    action_types_flat = dream["action_types"].reshape(N * H)
    coords_flat = dream["coords"].reshape(N * H, 2)
    available_actions = dream["available_actions"]
    mask_flat = available_actions.unsqueeze(1).expand(N, H, -1).reshape(N * H, -1)
    log_prob, entropy = policy.log_prob_entropy(features_flat, action_types_flat, coords_flat, mask_flat)

    baseline_ext = values[..., 0].detach()
    baseline_int = values[..., 1].detach()
    advantage_ext = normalizer_ext.normalize(returns_ext - baseline_ext)
    advantage_int = normalizer_int.normalize(returns_int - baseline_int)
    advantage = advantage_ext + cfg.intrinsic_scale * advantage_int
    actor_loss = -(log_prob * advantage.detach().flatten()).mean() - cfg.entropy_scale * entropy.mean()

    normalizer_ext.update(returns_ext)
    normalizer_int.update(returns_int)

    return {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy": entropy.mean().detach(),
        "imagined_return_ext": returns_ext.mean().detach(),
        "imagined_return_int": returns_int.mean().detach(),
        "intrinsic_mean": dream["intrinsic"].mean().detach(),
        "extrinsic_mean": dream["reward"].mean().detach(),
        "value_ext_mean": values[..., 0].mean().detach(),
        "value_int_mean": values[..., 1].mean().detach(),
        "dream_score_sum": dream["reward"].clamp(0.0, 1.0).sum(dim=1).mean().detach(),
    }
