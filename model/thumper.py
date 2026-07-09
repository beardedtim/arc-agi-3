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
from torch import nn

from model.policy import Policy, PolicyConfig
from model.world_model import WorldModel, WorldModelConfig


@dataclass
class ThumperConfig:
    """
    The configuration for all of Thumper, with sane defaults
    """
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def __post_init__(self):
        # One wiring rule, same spirit as WorldModelConfig.__post_init__:
        # the policy reads the world model's latent (deter ++ stoch), so its
        # feature width is derived here rather than trusted to agree by hand.
        # The two must also share one picture of the action space.
        self.policy.feature_dim = self.world_model.rssm.deter_dim + self.world_model.rssm.stoch_dim
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

    def features(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        """The (deter ++ stoch) latent both the heads and the policy read."""
        return self.world_model.features(deter, stoch)

    @torch.no_grad()
    def act(
        self,
        deter: torch.Tensor,
        stoch: torch.Tensor,
        available_actions: torch.Tensor | None = None,
        greedy: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Convenience: latent state -> sampled ARC-AGI-3 action.

        The caller owns the RSSM state loop (encode the frame stack, step the
        RSSM, pass (deter, stoch) here); this just bridges world model
        features into the policy. See Policy.act for the return dict.
        """
        return self.policy.act(self.features(deter, stoch), available_actions, greedy)

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
