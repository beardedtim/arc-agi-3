"""
Model :: Thumper

The baseline ARC-AGI-3 agent model and the benchmark for any others we
create: a Dreamer-style world model over 64x64 symbol grids (Vision ->
RSSM -> grid decoder) with a factored action policy (7 action types +
a 64x64 click pointer for ACTION6).

`Thumper` is the single trainable unit: one nn.Module owning every piece,
so there is exactly one thing to size up, checkpoint, move to a device,
and hand to an optimizer.
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from model.critic import Critic, CriticConfig
from model.policy import Policy, PolicyConfig
from model.world_model import WorldModel, WorldModelConfig


@dataclass
class ThumperConfig:
    """
    The configuration for all of Thumper, with sane defaults
    """
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)

    def __post_init__(self):
        # One wiring rule, same spirit as WorldModelConfig.__post_init__:
        # the policy/critic read the world model's latent
        # (deter ++ stoch ++ macro_context), so their feature width is
        # derived here rather than trusted to agree by hand. The policy must
        # also share the world model's picture of the action space.
        feature_dim = (
            self.world_model.rssm.deter_dim
            + self.world_model.rssm.stoch_dim
            + self.world_model.task_encoder.context_dim
        )
        self.policy.feature_dim = feature_dim
        self.critic.feature_dim = feature_dim
        self.policy.num_action_types = self.world_model.num_action_types
        self.policy.grid_size = self.world_model.grid_size


class Thumper(nn.Module):
    """Everything Thumper is, in one module.

    Components live in a plain attribute-per-component layout (not an opaque
    ModuleDict), so call sites read naturally (`thumper.world_model.encode`,
    `thumper.policy.act`) while `parameter_counts` / `save` / `load` discover
    them generically through nn.Module's own child registry -- adding a new
    component (a curiosity head, a search module, ...) is just another
    attribute assignment in __init__, and sizing/checkpointing pick it up
    with no further changes.
    """

    def __init__(self, config: ThumperConfig | None = None):
        super().__init__()
        self.config = config or ThumperConfig()
        self.world_model = WorldModel(self.config.world_model)
        self.policy = Policy(self.config.policy)
        self.critic = Critic(self.config.critic)
        # Target critic: a hard-synced copy providing the bootstrap values
        # for lambda-returns (training/actor_critic.py). Never optimized
        # directly -- see sync_critic_target.
        self.critic_target = Critic(self.config.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

    def features(
        self, deter: torch.Tensor, stoch: torch.Tensor, macro_context: torch.Tensor
    ) -> torch.Tensor:
        """The (deter ++ stoch ++ macro_context) latent both the heads and
        the policy read."""
        return self.world_model.features(deter, stoch, macro_context)

    @torch.no_grad()
    def act(
        self,
        deter: torch.Tensor,
        stoch: torch.Tensor,
        macro_context: torch.Tensor,
        available_actions: torch.Tensor | None = None,
        greedy: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Convenience: latent state -> sampled ARC-AGI-3 action.

        The caller owns the RSSM state loop (encode the frame stack, step the
        RSSM, pass (deter, stoch, macro_context) here); this just bridges
        world model features into the policy. See Policy.act for the return
        dict.
        """
        return self.policy.act(self.features(deter, stoch, macro_context), available_actions, greedy)

    def sync_critic_target(self) -> None:
        """Hard-copy `critic`'s weights into `critic_target`. Called by the
        trainer on a grad-step cadence (see TrainerConfig.critic_target_every)."""
        self.critic_target.load_state_dict(self.critic.state_dict())

    @torch.no_grad()
    def dream(
        self,
        deter: Tensor,
        stoch: Tensor,
        macro_context: Tensor,
        available_actions: Tensor,
        horizon: int,
        greedy: bool = False,
    ) -> dict[str, Tensor]:
        """Imagination rollout: from real (detached) posterior start states,
        act with the current policy and step the world model's prior for
        `horizon` steps, entirely in latent space and grad-free.

        deter: (N, deter_dim), stoch: (N, stoch_dim), macro_context:
            (N, context_dim) -- detached start states, one dream per row.
        available_actions: (N, num_action_types) bool, the *start* state's
            legal-action mask, held fixed for the whole rollout (see
            tickets/0003's masking rule -- the world model has only ever
            seen legal actions, so disagreement on illegal ones is
            untrained garbage).
        horizon: imagined steps H.
        greedy: argmax the policy's action choice instead of sampling it, for
            decision-time planning's always-present policy-faithful candidate
            (tickets/0011) -- the dynamics stay stochastic (`imagine_step`
            still samples `stoch`), only the action choice is argmaxed.
            Default False preserves every existing call site bit-for-bit.

        Returns:
          features: (N, H+1, feature_dim) -- start state's features first,
            then each imagined state's.
          action_types: (N, H) int64, coords: (N, H, 2) int64 -- the action
            taken *at* states 0..H-1.
          intrinsic: (N, H) -- disagreement of the transition arriving at
            state t+1, i.e. intrinsic[:, t] belongs to states[t] -> states[t+1].
          reward: (N, H), continue_prob: (N, H) -- reward/continue heads at
            the arriving states (features[:, 1:]).

        The macro-context freeze rule (tickets/0002, extended by 0003):
        `macro_context` is never stepped during a dream -- it's the belief
        `forward_sequence` produced at the start timestep, held frozen for
        every `rssm.imagine_step` in the rollout.
        """
        wm = self.world_model
        features_list = [wm.features(deter, stoch, macro_context)]
        action_types_list, coords_list, intrinsic_list = [], [], []

        for _ in range(horizon):
            features = features_list[-1]
            action = self.policy.act(features, available_actions, greedy=greedy)
            action_type, coords = action["action_type"], action["coords"]
            action_onehot = wm.encode_actions(action_type, coords)

            intrinsic = wm.disagreement(deter, stoch, action_onehot, macro_context)
            deter, stoch = wm.rssm.imagine_step(deter, stoch, action_onehot, macro_context)

            action_types_list.append(action_type)
            coords_list.append(coords)
            intrinsic_list.append(intrinsic)
            features_list.append(wm.features(deter, stoch, macro_context))

        features = torch.stack(features_list, dim=1)
        heads = wm.predict_heads(features[:, 1:])
        return {
            "features": features,
            "action_types": torch.stack(action_types_list, dim=1),
            "coords": torch.stack(coords_list, dim=1),
            "intrinsic": torch.stack(intrinsic_list, dim=1),
            "reward": heads["reward"],
            "continue_prob": torch.sigmoid(heads["continue_logit"]),
            "available_actions": available_actions,
        }

    def parameter_counts(self) -> dict[str, int]:
        """Trainable parameter count per top-level component, plus 'total'.

        Discovered via named_children, so new components are counted the
        moment they're assigned in __init__.
        """
        counts = {
            name: sum(p.numel() for p in child.parameters() if p.requires_grad)
            for name, child in self.named_children()
        }
        counts["total"] = sum(counts.values())
        return counts

    def summary(self) -> str:
        """Human-readable size breakdown, largest component first."""
        counts = self.parameter_counts()
        total = counts.pop("total")
        lines = ["Thumper"]
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:<12} {n:>12,}  ({n / total:.1%})")
        lines.append(f"  {'total':<12} {total:>12,}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """One checkpoint for the whole agent: weights + the config that
        shaped them, so `load` can rebuild the exact architecture without
        the caller reconstructing configs by hand."""
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "Thumper":
        """Rebuild a Thumper from a `save` checkpoint, weights included."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model
