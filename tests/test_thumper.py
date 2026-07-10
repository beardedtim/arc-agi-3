"""Thumper top level: config wiring, act bridge, parameter accounting,
and checkpoint round-trip."""
import torch

from model.thumper import Thumper, ThumperConfig
from tests.conftest import small_config


def test_config_wiring():
    """The __post_init__ hooks must keep cross-component dims agreeing."""
    cfg = small_config()
    wm, pol = cfg.world_model, cfg.policy
    assert pol.feature_dim == wm.rssm.deter_dim + wm.rssm.stoch_dim + wm.task_encoder.context_dim
    assert pol.num_action_types == wm.num_action_types
    assert pol.grid_size == wm.grid_size
    assert wm.rssm.embed_dim == wm.vision.out_dim
    assert wm.rssm.action_dim == wm.action_dim
    assert wm.rssm.macro_context_dim == wm.task_encoder.context_dim
    assert wm.vision.input_size == wm.grid_size
    assert wm.vision.frame_stack == wm.frame_stack


def test_default_config_builds():
    Thumper(ThumperConfig())  # full-size defaults must construct cleanly


def test_act_bridges_latent_to_action(thumper):
    B = 4
    c = thumper.config.world_model.rssm
    context_dim = thumper.config.world_model.task_encoder.context_dim
    deter, stoch = torch.randn(B, c.deter_dim), torch.randn(B, c.stoch_dim)
    macro_context = torch.randn(B, context_dim)
    out = thumper.act(deter, stoch, macro_context)
    assert set(out) == {"action_type", "coords", "log_prob"}
    assert out["action_type"].shape == (B,)


def test_parameter_counts(thumper):
    counts = thumper.parameter_counts()
    assert set(counts) == {"world_model", "policy", "critic", "critic_target", "total"}
    # critic_target is never optimized (requires_grad_(False)), so it
    # contributes 0 to both its own count and the total -- that's correct.
    assert counts["critic_target"] == 0
    assert counts["total"] == counts["world_model"] + counts["policy"] + counts["critic"]
    assert counts["total"] == sum(p.numel() for p in thumper.parameters() if p.requires_grad)
    assert "Thumper" in thumper.summary()


def test_save_load_roundtrip(thumper, tmp_path):
    path = tmp_path / "thumper.pt"
    thumper.save(path)
    loaded = Thumper.load(path)
    # same architecture and identical weights
    assert loaded.config.world_model.grid_size == thumper.config.world_model.grid_size
    for (n1, p1), (n2, p2) in zip(
        thumper.state_dict().items(), loaded.state_dict().items()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)
    # loaded model produces identical latent features
    c = thumper.config.world_model.rssm
    context_dim = thumper.config.world_model.task_encoder.context_dim
    deter, stoch = torch.randn(2, c.deter_dim), torch.randn(2, c.stoch_dim)
    macro_context = torch.randn(2, context_dim)
    a = thumper.act(deter, stoch, macro_context, greedy=True)
    b = loaded.act(deter, stoch, macro_context, greedy=True)
    assert (a["action_type"] == b["action_type"]).all()
    assert (a["coords"] == b["coords"]).all()
