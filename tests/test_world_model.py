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
        rewards = torch.randn(B, T)
        out = wm.forward_sequence(obs_batch, types, rewards=rewards)
        stoch = wm.config.rssm.stoch_dim
        context_dim = wm.config.task_encoder.context_dim
        assert out["post_mean"].shape == (B, T, stoch)
        assert out["prior_std"].shape == (B, T, stoch)
        assert out["recon"].shape == (B, T, wm.config.num_symbols, GRID, GRID)
        assert out["deter"].shape == (B, T, wm.config.rssm.deter_dim)
        assert out["macro_context"].shape == (B, T, context_dim)
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

    def test_rewards_default_to_zero(self, thumper, obs_batch):
        """No rewards passed -> TaskEncoder folds in all-zero rewards, same
        as an explicit all-zero tensor."""
        wm = thumper.world_model
        B, T = obs_batch.shape[:2]
        types = torch.full((B, T), ACTION1)
        torch.manual_seed(3)
        out_default = wm.forward_sequence(obs_batch, types)
        torch.manual_seed(3)
        out_explicit = wm.forward_sequence(obs_batch, types, rewards=torch.zeros(B, T))
        assert torch.allclose(out_default["macro_context"], out_explicit["macro_context"])


class TestBurnIn:
    """tickets/0004: forward_sequence grows a burn-in prefix that builds a
    macro-context held frozen for the whole loss window."""

    def _batch(self, thumper, B, T_total):
        torch.manual_seed(2)
        observations = torch.randint(0, 16, (B, T_total, STACK, GRID, GRID))
        types = torch.randint(0, NUM_ACTION_TYPES, (B, T_total))
        rewards = torch.randn(B, T_total)
        is_first = torch.zeros(B, T_total, dtype=torch.bool)
        is_first[:, 0] = True
        return observations, types, rewards, is_first

    def test_macro_context_frozen_and_nonzero_within_window(self, thumper):
        wm = thumper.world_model
        B, burn_in, T = 2, 4, 3
        observations, types, rewards, is_first = self._batch(thumper, B, burn_in + T)
        out = wm.forward_sequence(observations, types, is_first=is_first, rewards=rewards, burn_in=burn_in)
        macro_context = out["macro_context"]
        assert macro_context.shape[1] == T
        for t in range(1, T):
            assert torch.equal(macro_context[:, t], macro_context[:, 0])
        assert (macro_context[:, 0].abs().sum(dim=-1) > 0).all()

    def test_burn_in_zero_matches_direct_reference(self, thumper):
        """burn_in=0 must reproduce forward_sequence's own frozen-window
        computation exactly -- the compatibility anchor is that the new
        burn_in-parameterized path is well-defined and deterministic at 0,
        not a literal match to the pre-0004 per-step-stepping code (which
        this ticket intentionally removes -- see tickets/0004's root cause)."""
        wm = thumper.world_model
        B, T = 2, 5
        observations, types, rewards, is_first = self._batch(thumper, B, T)

        torch.manual_seed(7)
        out1 = wm.forward_sequence(observations, types, is_first=is_first, rewards=rewards, burn_in=0)
        torch.manual_seed(7)
        out2 = wm.forward_sequence(observations, types, is_first=is_first, rewards=rewards, burn_in=0)

        for key in ("post_mean", "prior_mean", "recon", "deter", "stoch", "macro_context"):
            assert out1[key].shape == out2[key].shape
            assert torch.equal(out1[key], out2[key])
        # No burn-in phase ran, so macro_context never left its zero initial_state.
        assert (out1["macro_context"] == 0).all()

    def test_window_excludes_burn_in_but_context_depends_on_it(self, thumper):
        wm = thumper.world_model
        B, burn_in, T = 2, 3, 4
        observations, types, rewards, is_first = self._batch(thumper, B, burn_in + T)

        out = wm.forward_sequence(observations, types, is_first=is_first, rewards=rewards, burn_in=burn_in)
        assert out["macro_context"].shape[1] == T
        assert out["recon"].shape[1] == T

        corrupted_observations = observations.clone()
        corrupted_observations[:, :burn_in] = torch.randint(0, 16, corrupted_observations[:, :burn_in].shape)
        out_corrupted = wm.forward_sequence(
            corrupted_observations, types, is_first=is_first, rewards=rewards, burn_in=burn_in
        )
        assert not torch.equal(out["macro_context"], out_corrupted["macro_context"])
        assert out_corrupted["macro_context"].shape[1] == T  # loss window length unaffected

        # compute_losses must run cleanly on the sliced targets, no shape errors
        losses = wm.compute_losses(out, observations[:, burn_in:])
        assert torch.isfinite(losses["total_loss"])

    def test_task_encoder_trains_through_burn_in_boundary(self, thumper):
        """Backward from a loss-window head loss must leave nonzero grad on
        task_encoder params: the only path to them is
        loss-window heads -> frozen macro_context -> burn-in TaskEncoder
        steps, which must survive the boundary detach (deter/stoch, not
        macro_context, are detached there)."""
        wm = thumper.world_model
        B, burn_in, T = 2, 3, 4
        observations, types, rewards, is_first = self._batch(thumper, B, burn_in + T)

        wm.zero_grad()
        out = wm.forward_sequence(observations, types, is_first=is_first, rewards=rewards, burn_in=burn_in)
        reward_loss = out["reward"].pow(2).mean()
        reward_loss.backward()

        task_encoder_grad = sum(
            p.grad.abs().sum() for p in wm.task_encoder.parameters() if p.grad is not None
        )
        assert task_encoder_grad > 0
        wm.zero_grad()

    def test_boundary_detach_blocks_grad_into_burn_in_trunk_computation(self, thumper):
        """deter/stoch are detached at the burn_in/loss-window boundary, so
        no gradient from a loss-window loss can reach the burn-in phase's
        RSSM computation -- verified structurally: the boundary state must
        carry no grad_fn back into the burn-in loop."""
        wm = thumper.world_model
        B, burn_in, T = 2, 3, 4
        observations, types, rewards, is_first = self._batch(thumper, B, burn_in + T)

        captured = {}
        original_step = wm.rssm.step
        call_count = {"n": 0}

        def spy_step(*args, **kwargs):
            call_count["n"] += 1
            result = original_step(*args, **kwargs)
            if call_count["n"] == burn_in + 1:  # first loss-window step
                captured["deter_in"] = args[0]
            return result

        wm.rssm.step = spy_step
        try:
            wm.forward_sequence(observations, types, is_first=is_first, rewards=rewards, burn_in=burn_in)
        finally:
            wm.rssm.step = original_step

        # The first loss-window step's input deter is the detached boundary
        # state: it must not require grad / have no grad_fn tracing into the
        # burn-in loop's GRU computation.
        assert not captured["deter_in"].requires_grad


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
    def test_imagine_with_burn_in_shapes(self, thumper):
        wm = thumper.world_model
        B, burn_in, horizon = 2, 3, 4
        observations = torch.randint(0, 16, (B, burn_in + 1, STACK, GRID, GRID))
        types = torch.randint(0, NUM_ACTION_TYPES, (B, burn_in + 1 + horizon))
        frames = wm.imagine_with_burn_in(
            observations, types, coords=None, is_first=None, rewards=None,
            burn_in=burn_in, horizon=horizon,
        )
        assert frames.shape == (B, horizon + 1, wm.config.num_symbols, GRID, GRID)
        # argmax recovers valid integer grids
        grids = frames.argmax(dim=2)
        assert grids.min() >= 0 and grids.max() < wm.config.num_symbols

    def test_imagine_with_burn_in_runs_without_rewards(self, thumper):
        """rewards=None must not crash -- same all-zero fallback as
        forward_sequence."""
        wm = thumper.world_model
        B, burn_in, horizon = 2, 2, 3
        observations = torch.randint(0, 16, (B, burn_in + 1, STACK, GRID, GRID))
        types = torch.randint(0, NUM_ACTION_TYPES, (B, burn_in + 1 + horizon))
        frames = wm.imagine_with_burn_in(
            observations, types, coords=None, is_first=None, rewards=None,
            burn_in=burn_in, horizon=horizon,
        )
        assert torch.isfinite(frames).all()

    def test_imagine_with_burn_in_freezes_macro_context(self, thumper):
        """The macro-context built from the burn-in phase must be held
        frozen for every imagine_step of the rollout (tickets/0004): the
        TaskEncoder is never called again once imagination starts."""
        wm = thumper.world_model
        B, burn_in, horizon = 2, 3, 4
        observations = torch.randint(0, 16, (B, burn_in + 1, STACK, GRID, GRID))
        types = torch.randint(0, NUM_ACTION_TYPES, (B, burn_in + 1 + horizon))

        seen_contexts = []
        original_imagine_step = wm.rssm.imagine_step

        def spy(prev_deter, prev_stoch, prev_action, macro_context):
            seen_contexts.append(macro_context)
            return original_imagine_step(prev_deter, prev_stoch, prev_action, macro_context)

        wm.rssm.imagine_step = spy
        try:
            frames = wm.imagine_with_burn_in(
                observations, types, coords=None, is_first=None, rewards=None,
                burn_in=burn_in, horizon=horizon,
            )
        finally:
            wm.rssm.imagine_step = original_imagine_step

        assert len(seen_contexts) == horizon
        for ctx in seen_contexts[1:]:
            assert torch.equal(ctx, seen_contexts[0])
        assert frames.shape == (B, horizon + 1, wm.config.num_symbols, GRID, GRID)

    def test_imagine_with_burn_in_never_calls_task_encoder_during_rollout(self, thumper):
        """Monkeypatch task_encoder.forward to raise once the burn-in phase
        (burn_in + 1 real steps) has consumed its calls -- mirrors the
        0003 dream test's freeze-rule check."""
        wm = thumper.world_model
        B, burn_in, horizon = 2, 2, 3
        observations = torch.randint(0, 16, (B, burn_in + 1, STACK, GRID, GRID))
        types = torch.randint(0, NUM_ACTION_TYPES, (B, burn_in + 1 + horizon))

        original_forward = wm.task_encoder.forward
        calls = {"n": 0}

        def counting_forward(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > burn_in + 1:
                raise AssertionError("TaskEncoder must not be called after the burn-in phase")
            return original_forward(*args, **kwargs)

        wm.task_encoder.forward = counting_forward
        try:
            wm.imagine_with_burn_in(
                observations, types, coords=None, is_first=None, rewards=None,
                burn_in=burn_in, horizon=horizon,
            )
        finally:
            wm.task_encoder.forward = original_forward
        assert calls["n"] == burn_in + 1


class TestEnsemble:
    def test_forward_and_disagreement_shapes(self, thumper):
        wm = thumper.world_model
        c = wm.config
        B = 3
        deter = torch.randn(B, c.rssm.deter_dim)
        stoch = torch.randn(B, c.rssm.stoch_dim)
        action = torch.randn(B, c.action_dim)
        macro_context = torch.randn(B, c.task_encoder.context_dim)
        preds = wm.ensemble(deter, stoch, action, macro_context)
        assert preds.shape == (c.ensemble.ensemble_size, B, c.rssm.stoch_dim)
        dis = wm.disagreement(deter, stoch, action, macro_context)
        assert dis.shape == (B,)
        # independently initialized heads must actually disagree
        assert (dis > 0).all()


class TestGradientIsolation:
    def test_ensemble_loss_does_not_train_task_encoder(self, thumper, obs_batch):
        """ensemble_loss reads macro_context detached -- backward from it
        alone must leave task_encoder with no grad."""
        wm = thumper.world_model
        B, T = obs_batch.shape[:2]
        types = torch.randint(0, NUM_ACTION_TYPES, (B, T))
        rewards = torch.randn(B, T)
        wm.zero_grad()
        out = wm.forward_sequence(obs_batch, types, rewards=rewards)
        losses = wm.compute_losses(out, obs_batch)
        losses["ensemble_loss"].backward()
        for p in wm.task_encoder.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0
        wm.zero_grad()

    def test_task_encoder_path_does_not_perturb_trunk_grad(self, thumper, obs_batch):
        """The TaskEncoder consumes detached (deter, stoch): a loss that
        only flows through the TaskEncoder's output (macro_context) must
        leave the RSSM trunk (GRU/prior/posterior) with zero grad from that
        backward pass, since detach() cuts the path back into the trunk.
        Needs burn_in > 0 (tickets/0004): with no burn-in phase there is no
        TaskEncoder gradient path to test in the first place (macro_context
        stays at its zero initial_state throughout the frozen loss window)."""
        wm = thumper.world_model
        B, T_total = obs_batch.shape[:2]
        burn_in = 1
        types = torch.randint(0, NUM_ACTION_TYPES, (B, T_total))
        rewards = torch.randn(B, T_total)
        wm.zero_grad()
        out = wm.forward_sequence(obs_batch, types, rewards=rewards, burn_in=burn_in)
        # A loss purely on macro_context (the TaskEncoder's output).
        macro_context_loss = out["macro_context"].pow(2).mean()
        macro_context_loss.backward()
        for name in ("gru", "prior_head", "posterior_head"):
            module = getattr(wm.rssm, name)
            for p in module.parameters():
                assert p.grad is None or p.grad.abs().sum() == 0
        wm.zero_grad()


def test_kl_divergence_zero_for_identical_gaussians():
    mean, std = torch.randn(4, 8), torch.rand(4, 8) + 0.5
    kl = kl_divergence_diag_gaussian(mean, std, mean, std)
    assert torch.allclose(kl, torch.zeros(4), atol=1e-6)
    # and positive when they differ
    assert (kl_divergence_diag_gaussian(mean, std, mean + 1.0, std) > 0).all()
