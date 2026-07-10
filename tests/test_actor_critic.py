"""training/actor_critic.py: lambda-return math, Thumper.dream shapes/freeze
rule/masking, and actor-critic gradient isolation."""
import torch

from model.actions import NUM_ACTION_TYPES
from tests.conftest import small_config
from training.actor_critic import ActorCriticConfig, ReturnNormalizer, actor_critic_losses, lambda_returns


class TestLambdaReturns:
    def test_hand_computed_case(self):
        # N=1, H=3, nonuniform discounts -- computed by hand, backward:
        # returns[2] = r2 + d2 * ((1-lam)*v3 + lam*v3)          = r2 + d2*v3
        # returns[1] = r1 + d1 * ((1-lam)*v2 + lam*returns[2])
        # returns[0] = r0 + d0 * ((1-lam)*v1 + lam*returns[1])
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        discounts = torch.tensor([[0.9, 0.8, 0.7]])
        values = torch.tensor([[0.5, 1.0, 1.5, 2.0]])  # v0..v3
        lam = 0.9

        r2 = 3.0 + 0.7 * 2.0
        r1 = 2.0 + 0.8 * ((1 - lam) * 1.5 + lam * r2)
        r0 = 1.0 + 0.9 * ((1 - lam) * 1.0 + lam * r1)
        expected = torch.tensor([[r0, r1, r2]])

        got = lambda_returns(rewards, discounts, values, lam)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_shape(self):
        N, H = 4, 5
        rewards = torch.randn(N, H)
        discounts = torch.rand(N, H)
        values = torch.randn(N, H + 1)
        got = lambda_returns(rewards, discounts, values, 0.95)
        assert got.shape == (N, H)


class TestReturnNormalizer:
    def test_normalize_uses_running_scale(self):
        norm = ReturnNormalizer(decay=0.0)  # decay=0 -> scale tracks last update exactly
        returns = torch.tensor([[0.0, 10.0]])  # p95-p05 spread is large
        norm.update(returns)
        assert norm.scale > 1.0
        adv = torch.tensor([[2.0]])
        assert torch.allclose(norm.normalize(adv), adv / norm.scale)

    def test_scale_floored_at_one(self):
        norm = ReturnNormalizer(decay=0.0)
        norm.update(torch.zeros(1, 4))  # zero spread
        assert norm.normalize(torch.tensor([2.0])).item() == 2.0  # divided by max(1.0, scale)

    def test_state_dict_roundtrip(self):
        norm = ReturnNormalizer()
        norm.scale = 3.5
        restored = ReturnNormalizer()
        restored.load_state_dict(norm.state_dict())
        assert restored.scale == 3.5

    def test_spread_under_one_passes_through_unscaled(self):
        norm = ReturnNormalizer(decay=0.0)
        norm.update(torch.tensor([[0.0, 0.5]]))  # spread < 1
        adv = torch.tensor([2.0])
        assert norm.normalize(adv).item() == 2.0

    def test_spread_around_ten_scales_by_tenth(self):
        norm = ReturnNormalizer(decay=0.0)
        # 100 points uniform on [0, 10]: p95-p05 spread is ~9.0-9.5, close
        # enough to the "~10" ballpark this test is checking (order of
        # magnitude, not an exact value).
        norm.update(torch.linspace(0.0, 10.0, 100).unsqueeze(0))
        assert 8.0 < norm.scale < 10.0
        adv = torch.tensor([norm.scale])
        assert torch.allclose(norm.normalize(adv), torch.tensor([1.0]), atol=1e-4)


def _start_states(thumper, N):
    c = thumper.config.world_model.rssm
    context_dim = thumper.config.world_model.task_encoder.context_dim
    torch.manual_seed(7)
    deter = torch.randn(N, c.deter_dim)
    stoch = torch.randn(N, c.stoch_dim)
    macro_context = torch.randn(N, context_dim)
    available_actions = torch.ones(N, NUM_ACTION_TYPES, dtype=torch.bool)
    return deter, stoch, macro_context, available_actions


class TestDream:
    def test_shapes(self, thumper):
        N, H = 5, 4
        deter, stoch, macro_context, mask = _start_states(thumper, N)
        dream = thumper.dream(deter, stoch, macro_context, mask, horizon=H)

        feature_dim = thumper.config.policy.feature_dim
        assert dream["features"].shape == (N, H + 1, feature_dim)
        assert dream["action_types"].shape == (N, H)
        assert dream["coords"].shape == (N, H, 2)
        assert dream["intrinsic"].shape == (N, H)
        assert dream["reward"].shape == (N, H)
        assert dream["continue_prob"].shape == (N, H)

    def test_no_grad_by_construction(self, thumper):
        deter, stoch, macro_context, mask = _start_states(thumper, 3)
        dream = thumper.dream(deter, stoch, macro_context, mask, horizon=3)
        for key in ("features", "action_types", "coords", "intrinsic", "reward", "continue_prob"):
            assert not dream[key].requires_grad, key

    def test_macro_context_frozen_task_encoder_never_called(self, thumper):
        deter, stoch, macro_context, mask = _start_states(thumper, 2)

        def boom(*args, **kwargs):
            raise AssertionError("TaskEncoder must not be stepped during a dream")

        original_forward = thumper.world_model.task_encoder.forward
        thumper.world_model.task_encoder.forward = boom
        try:
            thumper.dream(deter, stoch, macro_context, mask, horizon=4)
        finally:
            thumper.world_model.task_encoder.forward = original_forward

    def test_masking_restricts_action_types(self, thumper):
        from model.actions import ACTION3

        N = 6
        deter, stoch, macro_context, _ = _start_states(thumper, N)
        mask = torch.zeros(N, NUM_ACTION_TYPES, dtype=torch.bool)
        mask[:, ACTION3] = True
        dream = thumper.dream(deter, stoch, macro_context, mask, horizon=5)
        assert (dream["action_types"] == ACTION3).all()


class TestActorCriticLosses:
    def _dream(self, thumper, N=4, H=3):
        deter, stoch, macro_context, mask = _start_states(thumper, N)
        return thumper.dream(deter, stoch, macro_context, mask, horizon=H)

    def test_losses_finite(self, thumper):
        dream = self._dream(thumper)
        normalizer_ext = ReturnNormalizer()
        normalizer_int = ReturnNormalizer()
        cfg = ActorCriticConfig()
        losses = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target, normalizer_ext, normalizer_int, cfg
        )
        for k, v in losses.items():
            assert torch.isfinite(v).all(), k

    def test_gradient_isolation(self, thumper):
        dream = self._dream(thumper)
        normalizer_ext = ReturnNormalizer()
        normalizer_int = ReturnNormalizer()
        cfg = ActorCriticConfig()
        losses = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target, normalizer_ext, normalizer_int, cfg
        )

        thumper.zero_grad()
        losses["actor_loss"].backward(retain_graph=True)
        for p in thumper.world_model.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0
        for p in thumper.critic.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

        thumper.zero_grad()
        losses = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target, normalizer_ext, normalizer_int, cfg
        )
        losses["critic_loss"].backward()
        for p in thumper.world_model.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0
        for p in thumper.policy.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0
        for p in thumper.critic_target.parameters():
            assert p.grad is None
        thumper.zero_grad()

    def test_actor_and_critic_params_move(self):
        from model.thumper import Thumper

        torch.manual_seed(0)
        thumper = Thumper(small_config())  # local instance -- this test mutates weights
        dream = self._dream(thumper)
        normalizer_ext = ReturnNormalizer()
        normalizer_int = ReturnNormalizer()
        cfg = ActorCriticConfig()

        before_policy = {n: p.clone() for n, p in thumper.policy.named_parameters()}
        before_critic = {n: p.clone() for n, p in thumper.critic.named_parameters()}
        before_target = {n: p.clone() for n, p in thumper.critic_target.named_parameters()}

        actor_opt = torch.optim.Adam(thumper.policy.parameters(), lr=1e-2)
        critic_opt = torch.optim.Adam(thumper.critic.parameters(), lr=1e-2)
        losses = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target, normalizer_ext, normalizer_int, cfg
        )
        actor_opt.zero_grad()
        losses["actor_loss"].backward()
        actor_opt.step()

        losses = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target, normalizer_ext, normalizer_int, cfg
        )
        critic_opt.zero_grad()
        losses["critic_loss"].backward()
        critic_opt.step()

        assert any(not torch.equal(before_policy[n], p) for n, p in thumper.policy.named_parameters())
        assert any(not torch.equal(before_critic[n], p) for n, p in thumper.critic.named_parameters())
        assert all(
            torch.equal(before_target[n], p) for n, p in thumper.critic_target.named_parameters()
        )

        thumper.sync_critic_target()
        for (n1, p1), (n2, p2) in zip(
            thumper.critic.state_dict().items(), thumper.critic_target.state_dict().items()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)

    def test_sync_critic_target_copies_both_heads(self, thumper):
        """tickets/0005: sync_critic_target's state_dict copy must cover
        both ext_net and int_net, not just one head."""
        with torch.no_grad():
            for p in thumper.critic.ext_net.parameters():
                p.add_(1.0)
            for p in thumper.critic.int_net.parameters():
                p.add_(1.0)
        thumper.sync_critic_target()
        for (n1, p1), (n2, p2) in zip(
            thumper.critic.state_dict().items(), thumper.critic_target.state_dict().items()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)
        assert any("ext_net" in n for n, _ in thumper.critic_target.named_parameters())
        assert any("int_net" in n for n, _ in thumper.critic_target.named_parameters())

    def test_extrinsic_only_return_matches_hand_computed_lambda_return(self, thumper):
        """Step 6, test 1: with intrinsic reward == 0 and a single +1
        extrinsic reward at the last dream step, imagined_return_ext matches
        a hand-computed lambda-return (using this critic's own bootstrap
        values, since the target critic is untrained/random), and with
        intrinsic_scale=0 the actor advantage is driven by the extrinsic
        stream alone."""
        dream = self._dream(thumper, N=2, H=3)
        dream = dict(dream)
        dream["intrinsic"] = torch.zeros_like(dream["intrinsic"])
        reward = torch.zeros_like(dream["reward"])
        reward[:, -1] = 1.0
        dream["reward"] = reward

        with torch.no_grad():
            target_values = thumper.critic_target(dream["features"])
        discounts = ActorCriticConfig().gamma * dream["continue_prob"]
        # tickets/0006: the extrinsic stream's discounts absorb at the
        # predicted score (here just the reward tensor itself, which is
        # exactly what dream["reward"] is set to above).
        absorb = 1.0 - dream["reward"].clamp(0.0, 1.0)
        discounts_ext = discounts * absorb
        expected_ext = lambda_returns(dream["reward"], discounts_ext, target_values[..., 0], ActorCriticConfig().return_lambda)
        expected_int = lambda_returns(dream["intrinsic"], discounts, target_values[..., 1], ActorCriticConfig().return_lambda)

        normalizer_ext = ReturnNormalizer()
        normalizer_int = ReturnNormalizer()
        cfg = ActorCriticConfig(intrinsic_scale=0.0)
        losses = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target, normalizer_ext, normalizer_int, cfg
        )
        assert torch.allclose(losses["imagined_return_ext"], expected_ext.mean(), atol=1e-5)
        assert torch.allclose(losses["imagined_return_int"], expected_int.mean(), atol=1e-5)
        # intrinsic == 0 stream still produces a (bootstrap-driven) return,
        # but intrinsic_scale=0 means it contributes nothing to the actor loss.
        losses_zeroed_int_stream = actor_critic_losses(
            {**dream, "intrinsic": torch.full_like(dream["intrinsic"], 999.0)},
            thumper.policy, thumper.critic, thumper.critic_target,
            ReturnNormalizer(), ReturnNormalizer(), cfg,
        )
        assert torch.allclose(losses["actor_loss"], losses_zeroed_int_stream["actor_loss"], atol=1e-6)

    def test_drowning_regression_extrinsic_advantage_invariant_to_intrinsic_scale(self, thumper):
        """Step 6, test 2 -- this ticket's reason to exist: scaling the
        intrinsic stream 100x must leave the extrinsic advantage component
        bit-identical, and the combined advantage's extrinsic contribution
        must not be diminished."""
        dream = self._dream(thumper, N=4, H=3)
        normalizer_ext_a = ReturnNormalizer()
        normalizer_int_a = ReturnNormalizer()
        cfg = ActorCriticConfig(intrinsic_scale=0.0)

        # extrinsic-only advantage (intrinsic_scale=0 isolates it) under the
        # dream as-is
        with torch.no_grad():
            target_values = thumper.critic_target(dream["features"])
        from training.actor_critic import lambda_returns

        discounts = cfg.gamma * dream["continue_prob"]
        returns_ext = lambda_returns(dream["reward"], discounts, target_values[..., 0], cfg.return_lambda)
        values = thumper.critic(dream["features"][:, :-1])
        adv_ext_a = normalizer_ext_a.normalize(returns_ext - values[..., 0].detach())

        dream_scaled = dict(dream)
        dream_scaled["intrinsic"] = dream["intrinsic"] * 100.0
        normalizer_ext_b = ReturnNormalizer()
        normalizer_int_b = ReturnNormalizer()
        returns_ext_b = lambda_returns(
            dream_scaled["reward"], discounts, target_values[..., 0], cfg.return_lambda
        )
        adv_ext_b = normalizer_ext_b.normalize(returns_ext_b - values[..., 0].detach())

        assert torch.equal(adv_ext_a, adv_ext_b)

        # And with intrinsic_scale=0, the combined advantage from
        # actor_critic_losses is exactly the extrinsic advantage in both
        # cases -- the intrinsic stream's magnitude has zero effect.
        losses_a = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target,
            ReturnNormalizer(), ReturnNormalizer(), cfg,
        )
        losses_b = actor_critic_losses(
            dream_scaled, thumper.policy, thumper.critic, thumper.critic_target,
            ReturnNormalizer(), ReturnNormalizer(), cfg,
        )
        assert torch.allclose(losses_a["actor_loss"], losses_b["actor_loss"], atol=1e-6)
        assert losses_a["imagined_return_ext"].item() == losses_b["imagined_return_ext"].item()


class TestAbsorbingScoreExtrinsicReturns:
    """tickets/0006: a predicted score is absorbing for the extrinsic
    lambda-return only -- one dream can bank at most ~one score."""

    def test_absorption_math_hand_computed(self):
        # N=1, H=6, continue_prob == 1 (discounts == gamma), rewards +1 at
        # steps 2 and 5. Absorption at step 2 must zero the discount chain
        # for every step after it, so step 5's reward and the tail bootstrap
        # beyond step 2 contribute exactly zero to returns[0..2].
        gamma = 0.9
        H = 6
        rewards = torch.zeros(1, H)
        rewards[0, 2] = 1.0
        rewards[0, 5] = 1.0
        continue_prob = torch.ones(1, H)
        discounts = gamma * continue_prob
        absorb = 1.0 - rewards.clamp(0.0, 1.0)
        discounts_ext = discounts * absorb
        values = torch.full((1, H + 1), 3.0)  # arbitrary nonzero bootstrap
        lam = 0.95

        got = lambda_returns(rewards, discounts_ext, values, lam)

        # Steps 3, 4, 5 discount from step 2 is zero -- so returns[2] is
        # just the reward at step 2, full stop.
        assert torch.allclose(got[0, 2], torch.tensor(1.0), atol=1e-6)
        # returns[3] and returns[4] can't see the step-5 reward either,
        # since discounts_ext[3] and discounts_ext[4] are the pre-absorption
        # gamma (no score has happened yet *at* step 3/4) -- but discounts_ext[2]
        # is what's zeroed, cutting returns[0..1]'s view of anything beyond
        # step 2's own reward.
        # returns[1] should equal r1 + d1 * ((1-lam)*v2 + lam*returns[2])
        # with r1 = 0, d1 = gamma (absorb[1] = 1, no score at step 1).
        expected_1 = 0.0 + gamma * ((1 - lam) * values[0, 2] + lam * got[0, 2])
        assert torch.allclose(got[0, 1], expected_1, atol=1e-6)
        expected_0 = 0.0 + gamma * ((1 - lam) * values[0, 1] + lam * got[0, 1])
        assert torch.allclose(got[0, 0], expected_0, atol=1e-6)

    def test_inflation_loop_break(self):
        # The ticket's reason to exist: an absurdly inflated bootstrap value
        # must not leak past an absorbing score. +1 reward at step 0,
        # continue_prob == 1, target values pinned to 100 everywhere.
        H = 4
        rewards = torch.zeros(1, H)
        rewards[0, 0] = 1.0
        continue_prob = torch.ones(1, H)
        discounts = 0.997 * continue_prob
        absorb = 1.0 - rewards.clamp(0.0, 1.0)
        discounts_ext = discounts * absorb
        values = torch.full((1, H + 1), 100.0)

        got = lambda_returns(rewards, discounts_ext, values, 0.95)
        assert torch.allclose(got[0, 0], torch.tensor(1.0), atol=1e-6)

    def test_no_score_neutrality(self, thumper):
        # With dream["reward"] all zero, R_ext must be bit-identical to
        # lambda_returns computed with the unmodified (pre-fix) discounts.
        deter, stoch, macro_context, mask = _start_states(thumper, 3)
        dream = thumper.dream(deter, stoch, macro_context, mask, horizon=4)
        dream = dict(dream)
        dream["reward"] = torch.zeros_like(dream["reward"])

        cfg = ActorCriticConfig()
        discounts = cfg.gamma * dream["continue_prob"]
        with torch.no_grad():
            target_values = thumper.critic_target(dream["features"])
        expected = lambda_returns(dream["reward"], discounts, target_values[..., 0], cfg.return_lambda)

        losses = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target,
            ReturnNormalizer(), ReturnNormalizer(), cfg,
        )
        assert torch.allclose(losses["imagined_return_ext"], expected.mean(), atol=1e-6)

    def test_negative_rewards_do_not_absorb(self):
        # A level loss (-1) is clamped to 0 by clamp(r, 0, 1), so it must
        # leave the extrinsic discount chain untouched at that step.
        rewards = torch.tensor([[0.0, -1.0, 0.0]])
        absorb = 1.0 - rewards.clamp(0.0, 1.0)
        assert torch.equal(absorb, torch.ones_like(rewards))

    def test_intrinsic_stream_isolation(self, thumper):
        """R_int and the intrinsic advantage must be bit-identical whatever
        the extrinsic rewards are -- absorption must not touch the
        intrinsic discount chain (extends 0005's drowning regression test)."""
        deter, stoch, macro_context, mask = _start_states(thumper, 4)
        dream = thumper.dream(deter, stoch, macro_context, mask, horizon=3)
        dream_a = dict(dream)
        dream_a["reward"] = torch.zeros_like(dream["reward"])
        dream_b = dict(dream)
        dream_b["reward"] = torch.ones_like(dream["reward"])  # score every step

        cfg = ActorCriticConfig()
        with torch.no_grad():
            target_values = thumper.critic_target(dream["features"])
        discounts = cfg.gamma * dream["continue_prob"]
        returns_int_a = lambda_returns(
            dream_a["intrinsic"], discounts, target_values[..., 1], cfg.return_lambda
        )
        returns_int_b = lambda_returns(
            dream_b["intrinsic"], discounts, target_values[..., 1], cfg.return_lambda
        )
        assert torch.equal(returns_int_a, returns_int_b)

        losses_a = actor_critic_losses(
            dream_a, thumper.policy, thumper.critic, thumper.critic_target,
            ReturnNormalizer(), ReturnNormalizer(), cfg,
        )
        losses_b = actor_critic_losses(
            dream_b, thumper.policy, thumper.critic, thumper.critic_target,
            ReturnNormalizer(), ReturnNormalizer(), cfg,
        )
        assert torch.allclose(losses_a["imagined_return_int"], losses_b["imagined_return_int"], atol=1e-6)

    def test_no_new_grad_path_into_reward_head(self, thumper):
        """actor_loss/critic_loss.backward() must still leave the world
        model's reward head grad-free -- the absorb factor is computed from
        a grad-free dream tensor, so no new gradient path opens."""
        deter, stoch, macro_context, mask = _start_states(thumper, 3)
        dream = thumper.dream(deter, stoch, macro_context, mask, horizon=3)
        cfg = ActorCriticConfig()
        normalizer_ext = ReturnNormalizer()
        normalizer_int = ReturnNormalizer()

        losses = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target, normalizer_ext, normalizer_int, cfg
        )
        thumper.zero_grad()
        losses["actor_loss"].backward(retain_graph=True)
        for p in thumper.world_model.reward_head.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

        thumper.zero_grad()
        losses = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target, normalizer_ext, normalizer_int, cfg
        )
        losses["critic_loss"].backward()
        for p in thumper.world_model.reward_head.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0
        thumper.zero_grad()

    def test_dream_score_sum_metric(self, thumper):
        deter, stoch, macro_context, mask = _start_states(thumper, 3)
        dream = thumper.dream(deter, stoch, macro_context, mask, horizon=3)
        dream = dict(dream)
        reward = torch.zeros_like(dream["reward"])
        reward[:, 0] = 1.0
        reward[:, 1] = 2.0  # clamped to 1
        dream["reward"] = reward

        cfg = ActorCriticConfig()
        losses = actor_critic_losses(
            dream, thumper.policy, thumper.critic, thumper.critic_target,
            ReturnNormalizer(), ReturnNormalizer(), cfg,
        )
        assert torch.allclose(losses["dream_score_sum"], torch.tensor(2.0), atol=1e-6)
