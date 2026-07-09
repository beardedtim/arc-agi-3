"""Policy: forward shapes, action-availability masking, sampling, and
log-prob re-evaluation consistency."""
import torch

from model.actions import ACTION6, NUM_ACTION_TYPES
from tests.conftest import GRID


def _features(thumper, B):
    torch.manual_seed(2)
    return torch.randn(B, thumper.config.policy.feature_dim)


def test_forward_shapes(thumper):
    B = 3
    out = thumper.policy(_features(thumper, B))
    assert out["type_logits"].shape == (B, NUM_ACTION_TYPES)
    assert out["pointer_logits"].shape == (B, GRID, GRID)
    assert out["value"].shape == (B,)


def test_act_output_ranges(thumper):
    B = 8
    out = thumper.policy.act(_features(thumper, B))
    assert out["action_type"].shape == (B,)
    assert out["action_type"].min() >= 0 and out["action_type"].max() < NUM_ACTION_TYPES
    assert out["coords"].shape == (B, 2)
    assert out["coords"].min() >= 0 and out["coords"].max() < GRID
    assert torch.isfinite(out["log_prob"]).all()


def test_act_respects_available_actions_mask(thumper):
    B = 16
    mask = torch.zeros(B, NUM_ACTION_TYPES, dtype=torch.bool)
    mask[:, ACTION6] = True  # only clicks allowed
    out = thumper.policy.act(_features(thumper, B), available_actions=mask)
    assert (out["action_type"] == ACTION6).all()


def test_act_greedy_is_deterministic(thumper):
    f = _features(thumper, 4)
    a = thumper.policy.act(f, greedy=True)
    b = thumper.policy.act(f, greedy=True)
    assert (a["action_type"] == b["action_type"]).all()
    assert (a["coords"] == b["coords"]).all()


def test_log_prob_entropy_matches_act(thumper):
    """Re-evaluating a sampled action must reproduce act's joint log-prob,
    including the pointer term only on ACTION6 steps."""
    f = _features(thumper, 32)
    sampled = thumper.policy.act(f)
    log_prob, entropy, value = thumper.policy.log_prob_entropy(
        f, sampled["action_type"], sampled["coords"]
    )
    assert torch.allclose(log_prob, sampled["log_prob"], atol=1e-5)
    assert torch.allclose(value, sampled["value"], atol=1e-5)
    assert (entropy > 0).all()
