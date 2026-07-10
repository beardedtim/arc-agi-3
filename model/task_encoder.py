"""
Thumper's TaskEncoder: the "slow" half of a slow-fast memory split.

The RSSM's GRU (`model/rssm.py`) is the fast memory: it carries
frame-to-frame dynamics and resets its usefulness quickly under a sparse,
instruction-free reward signal. The TaskEncoder is the slow memory: it
folds completed transitions (deter, stoch, action, reward) into a
macro-context vector m that represents the agent's evolving belief about
the current game's rules, and conditions the RSSM's stochastic heads and
the Plan2Explore ensemble on that belief (see model/world_model.py).
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TaskEncoderConfig:
    """
    The configuration for Thumper's TaskEncoder, with sane defaults
    """
    context_dim: int = 128
    """Hidden width of the macro-context vector (m); must match
    `RSSMConfig.macro_context_dim` -- wired in `WorldModelConfig.__post_init__`."""
    hidden_dim: int = 256
    """Hidden width of the internal recurrent/projection layers"""


class TaskEncoder(nn.Module):
    """(prev_context, transition) -> next_context.

    transition is the concatenation of a completed RSSM transition (deter,
    stoch, action) and its scalar reward -- the trunk representations must
    be detached by the caller (see `WorldModel.forward_sequence`) so this
    module reads the trunk without shaping its reconstruction/KL objective.
    """

    def __init__(self, cfg: TaskEncoderConfig, deter_dim: int, stoch_dim: int, action_dim: int):
        super().__init__()
        self.cfg = cfg
        self.context_dim = cfg.context_dim
        input_dim = deter_dim + stoch_dim + action_dim + 1
        self.gru_cell = nn.GRUCell(input_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.context_dim)
        self.context_to_hidden = nn.Linear(cfg.context_dim, cfg.hidden_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.context_dim, device=device)

    def forward(
        self, prev_context: Tensor, deter: Tensor, stoch: Tensor, action: Tensor, reward: Tensor
    ) -> Tensor:
        """prev_context: (..., context_dim). deter/stoch/action/reward: (...,
        dim) for a single completed transition (reward is (..., 1)).
        Returns the next macro-context (..., context_dim)."""
        x = torch.cat([deter, stoch, action, reward], dim=-1)
        hidden = self.context_to_hidden(prev_context)
        hidden = self.gru_cell(x, hidden)
        return self.out_proj(hidden)
