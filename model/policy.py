"""
Thumper's Policy

Factored ARC-AGI-3 action head over the world model's latent features
(deter ++ stoch): a 7-way action-type head (RESET + ACTION1-5 + ACTION6)
and a 64x64 pointer head used only when ACTION6 is chosen. Both heads are
masked by the API's per-state `available_actions` before sampling, so the
policy never wastes probability mass on actions the game rejects.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.actions import ACTION6, GRID_SIZE, NUM_ACTION_TYPES


@dataclass
class PolicyConfig:
    feature_dim: int = 288
    """Size of the world-model feature (rssm.deter_dim + rssm.stoch_dim)"""
    hidden_dim: int = 256
    """Hidden width of the type/pointer/value MLPs"""
    num_action_types: int = NUM_ACTION_TYPES
    grid_size: int = GRID_SIZE


class Policy(nn.Module):
    """Actor-critic over world-model features with a factored action space.

    The pointer head decodes a coarse 8x8 spatial seed up to a full
    grid_size x grid_size logit map (rather than one flat Linear to 4096),
    so nearby cells share features -- clicks target objects, and objects
    are spatially coherent.
    """

    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        c = self.cfg

        self.trunk = nn.Sequential(
            nn.Linear(c.feature_dim, c.hidden_dim), nn.ELU(),
            nn.Linear(c.hidden_dim, c.hidden_dim), nn.ELU(),
        )
        self.type_head = nn.Linear(c.hidden_dim, c.num_action_types)
        self.value_head = nn.Linear(c.hidden_dim, 1)

        # pointer: hidden -> (32, 8, 8) seed, upsampled x2 per stage to
        # (1, grid_size, grid_size) logits.
        num_stages = (c.grid_size // 8).bit_length() - 1
        assert c.grid_size == 8 * 2**num_stages, "grid_size must be 8 * 2**n"
        self.pointer_fc = nn.Linear(c.hidden_dim, 32 * 8 * 8)
        layers: list[nn.Module] = []
        channels = 32
        for stage in range(num_stages):
            last = stage == num_stages - 1
            out_channels = 1 if last else channels
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv2d(channels, out_channels, 3, padding=1))
            if not last:
                layers.append(nn.ELU())
            channels = out_channels
        self.pointer_deconv = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        """features: (B, feature_dim). Returns:
        - type_logits: (B, num_action_types)
        - pointer_logits: (B, grid_size, grid_size), row y / column x
        - value: (B,)
        """
        h = self.trunk(features)
        seed = self.pointer_fc(h).view(-1, 32, 8, 8)
        pointer_logits = self.pointer_deconv(seed).squeeze(1)
        return {
            "type_logits": self.type_head(h),
            "pointer_logits": pointer_logits,
            "value": self.value_head(h).squeeze(-1),
        }

    def act(
        self,
        features: Tensor,
        available_actions: Tensor | None = None,
        greedy: bool = False,
    ) -> dict[str, Tensor]:
        """Sample an ARC-AGI-3 action.

        features: (B, feature_dim).
        available_actions: (B, num_action_types) bool mask from the API's
            per-frame `available_actions` list; unavailable types get -inf
            logits. None means everything is available.

        Returns action_type (B,), coords (B, 2) int64 (x, y) -- only
        meaningful where action_type == ACTION6 -- plus log_prob (B,)
        (joint over type and, for clicks, pointer) and value (B,).
        """
        out = self.forward(features)
        type_logits = out["type_logits"]
        if available_actions is not None:
            type_logits = type_logits.masked_fill(~available_actions, float("-inf"))

        type_dist = torch.distributions.Categorical(logits=type_logits)
        action_type = type_logits.argmax(-1) if greedy else type_dist.sample()

        B = features.shape[0]
        flat = out["pointer_logits"].reshape(B, -1)
        pointer_dist = torch.distributions.Categorical(logits=flat)
        flat_idx = flat.argmax(-1) if greedy else pointer_dist.sample()
        # flat index = y * grid_size + x (row-major over the (y, x) logit map)
        y, x = flat_idx // self.cfg.grid_size, flat_idx % self.cfg.grid_size
        coords = torch.stack([x, y], dim=-1)

        is_click = (action_type == ACTION6).float()
        log_prob = type_dist.log_prob(action_type) + is_click * pointer_dist.log_prob(flat_idx)
        return {
            "action_type": action_type,
            "coords": coords,
            "log_prob": log_prob,
            "value": out["value"],
        }

    def log_prob_entropy(
        self,
        features: Tensor,
        action_type: Tensor,
        coords: Tensor,
        available_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Re-evaluate stored actions for policy-gradient training.

        Returns (log_prob, entropy, value), each (B,). Entropy of the click
        pointer is only counted on ACTION6 steps, matching how `act` composes
        the joint log-prob.
        """
        out = self.forward(features)
        type_logits = out["type_logits"]
        if available_actions is not None:
            type_logits = type_logits.masked_fill(~available_actions, float("-inf"))
        type_dist = torch.distributions.Categorical(logits=type_logits)

        B = features.shape[0]
        flat = out["pointer_logits"].reshape(B, -1)
        pointer_dist = torch.distributions.Categorical(logits=flat)
        flat_idx = coords[..., 1] * self.cfg.grid_size + coords[..., 0]

        is_click = (action_type == ACTION6).float()
        log_prob = type_dist.log_prob(action_type) + is_click * pointer_dist.log_prob(flat_idx)
        entropy = type_dist.entropy() + is_click * pointer_dist.entropy()
        return log_prob, entropy, out["value"]
