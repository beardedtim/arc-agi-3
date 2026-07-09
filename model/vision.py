"""
Thumper's Vision

Responsible for taking in the raw ARC-AGI-3 grid of the environment and
transforming it into N features
"""

import torch
import torch.nn as nn

from dataclasses import dataclass

from model.actions import GRID_SIZE, NUM_SYMBOLS


@dataclass
class VisionConfig:
    """
    The configuration for Thumper's Vision, with sane defaults
    """
    num_symbols: int = NUM_SYMBOLS
    """Cell vocabulary size (16 ARC colors + 1 padding symbol). Cells are
    categorical, not RGB, so each symbol gets a learned embedding rather
    than being fed as raw intensity."""
    cell_embed_dim: int = 16
    """Embedding dim per cell per frame"""
    frame_stack: int = 4
    """K most recent frames stacked per observation (oldest first, settled
    frame last). Game dynamics -- movement, causality, mid-animation state --
    are only visible across time, and the ARC-AGI-3 API can return several
    animation frames for a single action. The conv stack's input channel
    count is frame_stack * cell_embed_dim."""
    out_dim: int = 256
    """How many feature dimensions should this output for the rest of Thumper to use?"""
    padding: int = 1
    """How much on either side do we add when making 2d blocks"""
    stride: int = 2
    """How far the conv kernel slides each step; stride=2 halves the spatial resolution per layer"""
    kernel_size: int = 3
    """The width/height of the square conv kernel each layer slides over the input"""
    input_size: int = GRID_SIZE
    """Height/width of the input grid -- must match WorldModelConfig.grid_size, the only
    resolution the model ever feeds Vision (see WorldModel.preprocess). Fixes the head's
    input width so it no longer needs a lazy first-forward-pass inference."""


class Vision(nn.Module):
    """
    Encodes a stack of K ARC-AGI-3 grids (64x64 cells, integer symbols 0-16)
    into a flat feature vector for the rest of Thumper (e.g. the RSSM) to
    consume.

    Each cell's symbol is looked up in a learned embedding table and the K
    frames are concatenated channel-wise, giving a
    (frame_stack * cell_embed_dim, 64, 64) map; then a stack of strided convolutions
    repeatedly halves spatial resolution while doubling channel width, in the
    style of the DreamerV3 encoder (LayerNorm + SiLU per layer). The resulting
    feature map is flattened and projected down to `cfg.out_dim`, so the
    output size is fixed regardless of input resolution.
    """

    _DEPTH = 4
    """Number of conv layers in the stack"""
    _BASE_CHANNELS = 32
    """Channel width of the first conv layer; doubles each subsequent layer"""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg

        self.cell_embed = nn.Embedding(cfg.num_symbols, cfg.cell_embed_dim)

        layers = []
        in_channels = cfg.frame_stack * cfg.cell_embed_dim
        out_channels = self._BASE_CHANNELS
        for _ in range(self._DEPTH):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride,
                    padding=cfg.padding,
                )
            )
            # GroupNorm(1, C) normalizes each sample over (C, H, W), i.e. LayerNorm
            # applied to a conv feature map (nn.LayerNorm expects a trailing dim
            # of fixed size, which conv maps don't have).
            layers.append(nn.GroupNorm(1, out_channels))
            layers.append(nn.SiLU())
            in_channels = out_channels
            out_channels *= 2

        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        size = cfg.input_size
        for _ in range(self._DEPTH):
            size = (size + 2 * cfg.padding - cfg.kernel_size) // cfg.stride + 1
        flat_dim = out_channels // 2 * size * size  # out_channels already doubled past the last layer
        self.head = nn.Linear(flat_dim, cfg.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, frame_stack, height, width) int64 cell symbols in
            [0, num_symbols), oldest frame first.
        returns: (batch, cfg.out_dim) feature tensor.
        """
        B, K, H, W = x.shape
        # (B, K, H, W, E) -> (B, K*E, H, W): each frame contributes its own
        # embedding channels, concatenated in time order.
        embedded = self.cell_embed(x).permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        features = self.flatten(self.conv(embedded))
        return self.head(features)
