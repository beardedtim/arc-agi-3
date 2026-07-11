"""
Decision-time planning :: policy-guided shooting (sampling-based MPC) over
the frozen world model (tickets/0011).

Not a model component -- no new nn.Module, no new parameters, no new losses,
nothing to checkpoint. `plan` is a pure function over a frozen `Thumper`:
it rolls N candidate futures with `Thumper.dream` (one greedy, N sampled),
scores each with `score_rollouts`, and returns the first action of the
best-scoring rollout. Nothing new is learned -- this composes `Thumper.dream`
and `training/actor_critic.py::lambda_returns`, re-aimed from "compute a
policy gradient" to "pick the best of N futures".

Scoring contract with `training/actor_critic.py`: a candidate's score is its
lambda-return from the start state, computed with the exact same gamma/lambda,
the same `gamma * continue_prob` discounts, the same tickets/0006 absorbing
multiplier on the extrinsic chain, and the same `critic_target` bootstrap that
`actor_critic_losses` uses for training. `score_rollouts` mirrors those lines
term for term rather than inventing a second notion of "return".
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from model.critic import Critic
from model.thumper import Thumper
from training.actor_critic import lambda_returns


@dataclass
class PlannerConfig:
    num_candidates: int = 64
    """Sampled-policy rollouts per planning call, in addition to the one
    greedy rollout (candidate 0). 0 -> pure greedy policy."""
    horizon: int = 15
    """Imagined steps per rollout. Default matches TrainerConfig.dream_horizon
    -- the depth training validated imagination at; deeper is untested."""
    gamma: float = 0.997
    return_lambda: float = 0.95
    intrinsic_scale: float = 0.0
    """Weight on the intrinsic (disagreement) lambda-return in a rollout's
    score. Unnormalized (no ReturnNormalizer exists at eval time) -- keep 0
    for exploitation planning; nonzero values are experiments."""


def score_rollouts(dream: dict[str, Tensor], critic_target: Critic, cfg: PlannerConfig) -> Tensor:
    """(N,) scores, one per dreamed rollout -- the same lambda-return math
    `actor_critic_losses` trains on (training/actor_critic.py), computed
    grad-free from an already grad-free dream."""
    discounts = cfg.gamma * dream["continue_prob"]
    absorb = 1.0 - dream["reward"].clamp(0.0, 1.0)
    target_values = critic_target(dream["features"])  # (N, H+1, 2), no_grad by caller convention
    returns_ext = lambda_returns(
        dream["reward"], discounts * absorb, target_values[..., 0], cfg.return_lambda
    )
    returns_int = lambda_returns(
        dream["intrinsic"], discounts, target_values[..., 1], cfg.return_lambda
    )
    return returns_ext[:, 0] + cfg.intrinsic_scale * returns_int[:, 0]


@torch.no_grad()
def plan(
    thumper: Thumper,
    deter: Tensor,
    stoch: Tensor,
    macro_context: Tensor,
    available_actions: Tensor,
    cfg: PlannerConfig,
) -> dict[str, Tensor]:
    """From one real posterior start state (the `(1, dim)` tensors
    `OnlineActor.act` holds right after its TaskEncoder fold), roll
    `cfg.num_candidates` sampled-policy dreams plus one greedy dream, score
    every candidate, and return the first action of the best-scoring one.

    The greedy rollout is always candidate 0 (Design principle 3): ties in
    the argmax resolve to the lowest index, so an uninformative scorer
    degrades the planner to exactly the greedy policy -- the planner's floor
    is the reactive agent, by construction. With `num_candidates == 0`, the
    returned action is exactly `thumper.act(..., greedy=True)`'s, since the
    executed first action is argmaxed from the real current features, before
    any imagined stochasticity.
    """
    greedy_dream = thumper.dream(
        deter, stoch, macro_context, available_actions, cfg.horizon, greedy=True
    )

    dreams = [greedy_dream]
    if cfg.num_candidates > 0:
        N = cfg.num_candidates
        deter_n = deter.expand(N, -1).contiguous()
        stoch_n = stoch.expand(N, -1).contiguous()
        macro_context_n = macro_context.expand(N, -1).contiguous()
        available_actions_n = available_actions.expand(N, -1).contiguous()
        sampled_dream = thumper.dream(
            deter_n, stoch_n, macro_context_n, available_actions_n, cfg.horizon, greedy=False
        )
        dreams.append(sampled_dream)

    reward = torch.cat([d["reward"] for d in dreams], dim=0)
    continue_prob = torch.cat([d["continue_prob"] for d in dreams], dim=0)
    intrinsic = torch.cat([d["intrinsic"] for d in dreams], dim=0)
    features = torch.cat([d["features"] for d in dreams], dim=0)
    action_types = torch.cat([d["action_types"] for d in dreams], dim=0)
    coords = torch.cat([d["coords"] for d in dreams], dim=0)

    scores = score_rollouts(
        {
            "reward": reward,
            "continue_prob": continue_prob,
            "intrinsic": intrinsic,
            "features": features,
        },
        thumper.critic_target,
        cfg,
    )
    best = scores.argmax()
    return {
        "action_type": action_types[best, 0:1],
        "coords": coords[best, 0:1],
        "score": scores[best],
        "greedy_chosen": best == 0,
    }
