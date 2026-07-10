"""Vision encoder and RSSM primitives."""
import torch

from model.rssm import RSSM, RSSMConfig
from tests.conftest import GRID, STACK


def test_vision_output_shape(thumper):
    enc = thumper.world_model.encoder
    x = torch.randint(0, enc.cfg.num_symbols, (5, STACK, GRID, GRID))
    out = enc(x)
    assert out.shape == (5, enc.cfg.out_dim)
    assert torch.isfinite(out).all()


def test_rssm_step_shapes():
    cfg = RSSMConfig(embed_dim=16, action_dim=10, deter_dim=24, stoch_dim=6, macro_context_dim=8)
    rssm = RSSM(cfg)
    B = 3
    deter, stoch = rssm.initial_state(B, torch.device("cpu"))
    assert deter.shape == (B, 24) and (deter == 0).all()
    action = torch.randn(B, 10)
    embed = torch.randn(B, 16)
    macro_context = torch.randn(B, 8)

    deter, prior_stats, post_stats, post_stoch = rssm.step(deter, stoch, action, embed, macro_context)
    assert deter.shape == (B, 24)
    assert prior_stats.shape == post_stats.shape == (B, 2 * 6)
    assert post_stoch.shape == (B, 6)

    d2, s2 = rssm.observe_step(deter, post_stoch, action, embed, macro_context)
    d3, s3 = rssm.imagine_step(deter, post_stoch, action, macro_context)
    assert d2.shape == d3.shape == (B, 24)
    assert s2.shape == s3.shape == (B, 6)


def test_dist_params_std_is_positive():
    stats = torch.randn(4, 12) * 10  # extreme raw values
    mean, std = RSSM.dist_params(stats)
    assert mean.shape == std.shape == (4, 6)
    assert (std >= 0.1).all()  # softplus + 0.1 floor
