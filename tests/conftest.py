"""
Shared fixtures: a small Thumper so the suite runs in seconds on CPU.

grid_size=16 satisfies both size constraints (decoder needs 4 * 2**n,
policy pointer needs 8 * 2**n) while cutting the conv/deconv work ~16x
versus the real 64x64. Everything else is shrunk proportionally; the
wiring in the __post_init__ hooks is what's actually under test, so the
absolute sizes don't matter.
"""
import pytest
import torch

from model.rssm import RSSMConfig
from model.thumper import Thumper, ThumperConfig
from model.vision import VisionConfig
from model.world_model import EnsembleConfig, WorldModelConfig

GRID = 16
STACK = 2


def small_config() -> ThumperConfig:
    return ThumperConfig(
        world_model=WorldModelConfig(
            grid_size=GRID,
            frame_stack=STACK,
            head_hidden_dim=32,
            vision=VisionConfig(cell_embed_dim=4, out_dim=32),
            rssm=RSSMConfig(deter_dim=32, stoch_dim=8),
            ensemble=EnsembleConfig(ensemble_size=3, ensemble_hidden_dim=16),
        )
    )


@pytest.fixture(scope="session")
def thumper() -> Thumper:
    torch.manual_seed(0)
    return Thumper(small_config())


@pytest.fixture()
def obs_batch() -> torch.Tensor:
    """(B, T, K, H, W) integer frame stacks, deliberately smaller than
    grid_size so preprocess's padding path is exercised."""
    torch.manual_seed(1)
    return torch.randint(0, 16, (2, 3, STACK, 10, 12))
