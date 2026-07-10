"""
Thumper's RSSM and grid decoder

The Dreamer-style recurrent state-space model at the heart of the world
model: a GRU carries the deterministic state, prior/posterior heads emit a
stochastic latent, and the decoder maps (deter, stoch) back to per-cell
symbol logits over the ARC-AGI-3 grid.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn

@dataclass
class RSSMConfig:
    """
    The configuration for Thumper's RSSM, with sane defaults
    """
    embed_dim: int = 256
    """Size of the observation embedding fed in by Vision; must match `VisionConfig.out_dim`"""
    action_dim: int = 135
    """Size of the encoded action vector. For ARC-AGI-3 this is the factored
    encoding built by `WorldModel.encode_actions`: one-hot action type (7)
    ++ one-hot click x (64) ++ one-hot click y (64) = 135; the coordinate
    one-hots are zeroed unless the type is ACTION6."""
    deter_dim: int = 256
    """Size of the deterministic recurrent state (the GRU's hidden state)"""
    stoch_dim: int = 32
    """Size of the stochastic latent state sampled from the prior/posterior"""
    macro_context_dim: int = 128
    """Size of the slow-memory macro-context vector (m) the TaskEncoder
    produces; must match `TaskEncoderConfig.context_dim` -- wired in
    `WorldModelConfig.__post_init__`. Conditions the prior/posterior heads
    only, not the GRU's deterministic recurrence."""


class RSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.deter_dim = cfg.deter_dim
        self.stoch_dim = cfg.stoch_dim

        self.gru = nn.GRUCell(cfg.stoch_dim + cfg.action_dim, cfg.deter_dim)

        # prior: predict stoch from deter + macro-context (used when
        # imagining, no obs)
        self.prior_head = nn.Sequential(
            nn.Linear(cfg.deter_dim + cfg.macro_context_dim, cfg.deter_dim), nn.ReLU(),
            nn.Linear(cfg.deter_dim, 2 * cfg.stoch_dim),
        )
        # posterior: predict stoch from deter + real observation embedding +
        # macro-context
        self.posterior_head = nn.Sequential(
            nn.Linear(cfg.deter_dim + cfg.embed_dim + cfg.macro_context_dim, cfg.deter_dim), nn.ReLU(),
            nn.Linear(cfg.deter_dim, 2 * cfg.stoch_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device):
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        return deter, stoch

    def _sample(self, stats: Tensor) -> Tensor:
        mean, std = self.dist_params(stats)
        return mean + std * torch.randn_like(mean)

    @staticmethod
    def dist_params(stats: Tensor) -> tuple[Tensor, Tensor]:
        mean, std_raw = stats.chunk(2, dim=-1)
        std = nn.functional.softplus(std_raw) + 0.1
        return mean, std

    def observe_step(
        self,
        prev_deter: Tensor,
        prev_stoch: Tensor,
        prev_action: Tensor,
        embed: Tensor,
        macro_context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One posterior step: fold in a real observation embedding."""
        deter = self.gru(torch.cat([prev_stoch, prev_action], dim=-1), prev_deter)
        stoch = self._sample(self.posterior_head(torch.cat([deter, embed, macro_context], dim=-1)))
        return deter, stoch

    def imagine_step(
        self,
        prev_deter: Tensor,
        prev_stoch: Tensor,
        prev_action: Tensor,
        macro_context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One prior step: predict forward with no observation (imagination)."""
        deter = self.gru(torch.cat([prev_stoch, prev_action], dim=-1), prev_deter)
        stoch = self._sample(self.prior_head(torch.cat([deter, macro_context], dim=-1)))
        return deter, stoch

    def step(
        self,
        prev_deter: Tensor,
        prev_stoch: Tensor,
        prev_action: Tensor,
        embed: Tensor,
        macro_context: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One training step: shares a single `deter` between the prior (no
        observation) and posterior (with observation) heads, so the KL term
        compares two distributions conditioned on the same recurrent state.

        Returns (deter, prior_stats, posterior_stats, posterior_stoch).
        """
        deter = self.gru(torch.cat([prev_stoch, prev_action], dim=-1), prev_deter)
        prior_stats = self.prior_head(torch.cat([deter, macro_context], dim=-1))
        posterior_stats = self.posterior_head(torch.cat([deter, embed, macro_context], dim=-1))
        posterior_stoch = self._sample(posterior_stats)
        return deter, prior_stats, posterior_stats, posterior_stoch


@dataclass
class ImageDecoderConfig:
    """
    The configuration for Thumper's grid decoder, with sane defaults
    """
    deter_dim: int = 256
    """Size of the deterministic recurrent state; must match `RSSMConfig.deter_dim`"""
    stoch_dim: int = 32
    """Size of the stochastic latent state; must match `RSSMConfig.stoch_dim`"""
    out_channels: int = 17
    """Number of output channels: per-cell *logits* over the symbol vocabulary
    (16 colors + pad); must match `VisionConfig.num_symbols`"""
    out_size: int = 64
    """Height/width of the reconstructed (square) grid; must be 4 * 2**n
    (the decoder doubles a 4x4 seed once per stage)"""


class ImageDecoder(nn.Module):
    """(deter, stoch) -> per-cell symbol logits at a fixed `out_size x out_size`.

    Uses Upsample+Conv blocks (rather than ConvTranspose2d) so the output
    resolution is exact and easy to reason about: 4 -> 8 -> ... -> out_size,
    each step a plain 2x nearest upsample followed by a 3x3 conv; channel
    width starts at 256 and halves per stage, floored at 32. The final layer
    has no activation, so its `out_channels` outputs are raw logits over the
    17-symbol vocabulary, trained with cross-entropy against the integer grid
    (see `WorldModel.compute_losses`) -- categorical cells make classification
    the right reconstruction objective, not MSE on color indices.
    """

    def __init__(self, cfg: ImageDecoderConfig):
        super().__init__()
        num_stages = (cfg.out_size // 4).bit_length() - 1
        assert cfg.out_size == 4 * 2**num_stages and num_stages >= 1, (
            f"out_size must be 4 * 2**n, got {cfg.out_size}"
        )
        self.cfg = cfg
        latent_dim = cfg.deter_dim + cfg.stoch_dim
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        layers: list[nn.Module] = []
        channels = 256
        for stage in range(num_stages):
            last = stage == num_stages - 1
            out_channels = cfg.out_channels if last else max(channels // 2, 32)
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv2d(channels, out_channels, 3, padding=1))
            if not last:
                layers.append(nn.ReLU())
            channels = out_channels
        self.deconv = nn.Sequential(*layers)

    def forward(self, deter: Tensor, stoch: Tensor) -> Tensor:
        x = self.fc(torch.cat([deter, stoch], dim=-1))
        x = x.view(-1, 256, 4, 4)
        return self.deconv(x)