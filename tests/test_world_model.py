"""World model: preprocessing, action encoding, sequence forward, losses,
imagination, and the disagreement ensemble."""
import pytest
import torch

from model.actions import ACTION1, ACTION6, NUM_ACTION_TYPES, PAD_SYMBOL
from model.world_model import kl_divergence_diag_gaussian
from tests.conftest import GRID, STACK


class TestPreprocess:
    def test_pads_small_grids_with_pad_symbol(self, thumper):
        wm = thumper.world_model
        g = torch.zeros(2, STACK, 10, 12, dtype=torch.long)
        out = wm.preprocess(g)
        assert out.shape == (2, STACK, GRID, GRID)
        # original region untouched, padded region is PAD_SYMBOL not color 0
        assert (out[..., :10, :12] == 0).all()
        assert (out[..., 10:, :] == PAD_SYMBOL).all()
        assert (out[..., :, 12:] == PAD_SYMBOL).all()

    def test_full_size_grid_unchanged(self, thumper):
        g = torch.randint(0, 16, (1, STACK, GRID, GRID))
        out = thumper.world_model.preprocess(g)
        assert (out == g).all()


class TestEncodeActions:
    def test_shape_and_type_onehot(self, thumper):
        wm = thumper.world_model
        types = torch.tensor([ACTION1, ACTION6])
        coords = torch.tensor([[5, 7], [3, 9]])
        enc = wm.encode_actions(types, coords)
        assert enc.shape == (2, NUM_ACTION_TYPES + 2 * GRID)
        assert enc[0, ACTION1] == 1.0 and enc[1, ACTION6] == 1.0
        assert enc[:, :NUM_ACTION_TYPES].sum(-1).allclose(torch.ones(2))

    def test_coords_zeroed_unless_action6(self, thumper):
        wm = thumper.world_model
        types = torch.tensor([ACTION1, ACTION6])
        coords = torch.tensor([[5, 7], [3, 9]])
        enc = wm.encode_actions(types, coords)
        # non-click step: coordinate one-hots all zero despite coords given
        assert (enc[0, NUM_ACTION_TYPES:] == 0).all()
        # click step: exactly one hot in x block and one in y block
        assert enc[1, NUM_ACTION_TYPES + 3] == 1.0
        assert enc[1, NUM_ACTION_TYPES + GRID + 9] == 1.0
        assert enc[1, NUM_ACTION_TYPES:].sum() == 2.0

    def test_none_coords(self, thumper):
        enc = thumper.world_model.encode_actions(torch.tensor([ACTION1]))
        assert (enc[:, NUM_ACTION_TYPES:] == 0).all()


class TestForwardSequence:
    def test_output_shapes(self, thumper, obs_batch):
        wm = thumper.world_model
        B, T = obs_batch.shape[:2]
        types = torch.randint(0, NUM_ACTION_TYPES, (B, T))
        out = wm.forward_sequence(obs_batch, types)
        stoch = wm.config.rssm.stoch_dim
        assert out["post_mean"].shape == (B, T, stoch)
        assert out["prior_std"].shape == (B, T, stoch)
        assert out["recon"].shape == (B, T, wm.config.num_symbols, GRID, GRID)
        assert out["deter"].shape == (B, T, wm.config.rssm.deter_dim)
        assert out["reward"].shape == (B, T)
        assert out["continue_logit"].shape == (B, T)
        assert out["internal_state"].shape == (B, T, 1)

    def test_is_first_zeroes_placeholder_action(self, thumper, obs_batch):
        wm = thumper.world_model
        B, T = obs_batch.shape[:2]
        types = torch.full((B, T), ACTION1)
        is_first = torch.zeros(B, T, dtype=torch.bool)
        is_first[:, 0] = True
        out = wm.forward_sequence(obs_batch, types, is_first=is_first)
        assert (out["action_onehot"][:, 0] == 0).all()
        assert (out["action_onehot"][:, 1:].sum(-1) == 1).all()


class TestComputeLosses:
    def _batch(self, B, T):
        return {
            "rewards": torch.zeros(B, T),
            "terminateds": torch.zeros(B, T, dtype=torch.bool),
            "is_first": torch.zeros(B, T, dtype=torch.bool),
            "internal_states": torch.zeros(B, T, 1),
        }

    def test_pure_recon_kl(self, thumper, obs_batch):
        wm = thumper.world_model
        B, T = obs_batch.shape[:2]
        out = wm.forward_sequence(obs_batch, torch.zeros(B, T, dtype=torch.long))
        losses = wm.compute_losses(out, obs_batch)
        assert {"recon_loss", "kl_loss", "kl_raw", "ensemble_loss", "total_loss"} <= losses.keys()
        assert "reward_loss" not in losses  # no batch targets given
        assert torch.isfinite(losses["total_loss"])

    def test_with_batch_targets_and_backward(self, thumper, obs_batch):
        wm = thumper.world_model
        B, T = obs_batch.shape[:2]
        out = wm.forward_sequence(obs_batch, torch.zeros(B, T, dtype=torch.long))
        losses = wm.compute_losses(out, obs_batch, batch=self._batch(B, T))
        assert {"reward_loss", "continue_loss", "internal_state_loss"} <= losses.keys()
        wm.zero_grad()
        losses["total_loss"].backward()
        # every trainable component receives gradient from one training step
        for name in ("encoder", "rssm", "decoder", "reward_head", "ensemble"):
            module = getattr(wm, name)
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
            ), f"{name} got no gradient"
        wm.zero_grad()

    def test_train_reward_head_flag(self, thumper, obs_batch):
        wm = thumper.world_model
        B, T = obs_batch.shape[:2]
        out = wm.forward_sequence(obs_batch, torch.zeros(B, T, dtype=torch.long))
        losses = wm.compute_losses(
            out, obs_batch, batch=self._batch(B, T), train_reward_head=False
        )
        assert "reward_loss" not in losses

    def test_kl_weight_override(self, thumper, obs_batch):
        wm = thumper.world_model
        B, T = obs_batch.shape[:2]
        out = wm.forward_sequence(obs_batch, torch.zeros(B, T, dtype=torch.long))
        base = wm.compute_losses(out, obs_batch)
        zero = wm.compute_losses(out, obs_batch, kl_weight=0.0)
        expected_drop = wm.config.kl_weight * base["kl_loss"]
        assert torch.allclose(base["total_loss"] - zero["total_loss"], expected_drop, atol=1e-5)


class TestImagination:
    def test_imagine_from_first_frame_shapes(self, thumper):
        wm = thumper.world_model
        B, T = 2, 4
        first = torch.randint(0, 16, (B, STACK, GRID, GRID))
        types = torch.randint(0, NUM_ACTION_TYPES, (B, T))
        frames = wm.imagine_from_first_frame(first, types)
        assert frames.shape == (B, T + 1, wm.config.num_symbols, GRID, GRID)
        # argmax recovers valid integer grids
        grids = frames.argmax(dim=2)
        assert grids.min() >= 0 and grids.max() < wm.config.num_symbols


class TestEnsemble:
    def test_forward_and_disagreement_shapes(self, thumper):
        wm = thumper.world_model
        c = wm.config
        B = 3
        deter = torch.randn(B, c.rssm.deter_dim)
        stoch = torch.randn(B, c.rssm.stoch_dim)
        action = torch.randn(B, c.action_dim)
        preds = wm.ensemble(deter, stoch, action)
        assert preds.shape == (c.ensemble.ensemble_size, B, c.rssm.stoch_dim)
        dis = wm.disagreement(deter, stoch, action)
        assert dis.shape == (B,)
        # independently initialized heads must actually disagree
        assert (dis > 0).all()


def test_kl_divergence_zero_for_identical_gaussians():
    mean, std = torch.randn(4, 8), torch.rand(4, 8) + 0.5
    kl = kl_divergence_diag_gaussian(mean, std, mean, std)
    assert torch.allclose(kl, torch.zeros(4), atol=1e-6)
    # and positive when they differ
    assert (kl_divergence_diag_gaussian(mean, std, mean + 1.0, std) > 0).all()
