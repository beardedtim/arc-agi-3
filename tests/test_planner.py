"""training/planner.py: policy-guided shooting MPC (tickets/0011) -- scoring
math mirrors actor_critic_losses, the greedy-floor degenerate case, mask
respect, and that planning never perturbs OnlineActor's real latent state."""
import torch

from model.actions import ACTION3, ACTION6, NUM_ACTION_TYPES, RESET
from tests.conftest import small_config
from training.actor_critic import lambda_returns
from training.online_actor import OnlineActor
from training.planner import PlannerConfig, plan, score_rollouts


def _start_states(thumper, N=1):
    c = thumper.config.world_model.rssm
    context_dim = thumper.config.world_model.task_encoder.context_dim
    torch.manual_seed(7)
    deter = torch.randn(N, c.deter_dim)
    stoch = torch.randn(N, c.stoch_dim)
    macro_context = torch.randn(N, context_dim)
    available_actions = torch.ones(N, NUM_ACTION_TYPES, dtype=torch.bool)
    return deter, stoch, macro_context, available_actions


class _ConstantCriticTarget:
    """Stub critic_target: constant (N, H+1, 2) values regardless of input,
    so score_rollouts's lambda-returns are hand-computable."""

    def __init__(self, ext_value: float, int_value: float):
        self.ext_value = ext_value
        self.int_value = int_value

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        N, T = features.shape[:2]
        ext = torch.full((N, T, 1), self.ext_value)
        intr = torch.full((N, T, 1), self.int_value)
        return torch.cat([ext, intr], dim=-1)


def _hand_dream(N, H, rewards, continue_prob, intrinsic):
    feature_dim = 4  # arbitrary, unused by score_rollouts beyond shape
    return {
        "reward": rewards,
        "continue_prob": continue_prob,
        "intrinsic": intrinsic,
        "features": torch.zeros(N, H + 1, feature_dim),
    }


class TestScoreRollouts:
    def test_mirrors_training_lambda_returns_and_ranks_by_discounting(self):
        """3 hand-built candidates: reward=1 at step 0 should outscore
        reward=1 at the last step (discounting), which outscores all-zeros."""
        H = 4
        cfg = PlannerConfig(gamma=0.99, return_lambda=0.9, intrinsic_scale=0.0)
        critic_target = _ConstantCriticTarget(ext_value=0.0, int_value=0.0)

        rewards = torch.zeros(3, H)
        rewards[0, 0] = 1.0  # candidate 0: reward at step 0
        rewards[1, -1] = 1.0  # candidate 1: reward at last step
        # candidate 2: all zeros
        continue_prob = torch.ones(3, H)
        intrinsic = torch.zeros(3, H)

        dream = _hand_dream(3, H, rewards, continue_prob, intrinsic)
        scores = score_rollouts(dream, critic_target, cfg)

        assert scores[0] > scores[1] > scores[2]

        # Cross-check against lambda_returns directly, term for term.
        discounts = cfg.gamma * continue_prob
        absorb = 1.0 - rewards.clamp(0.0, 1.0)
        target_values = critic_target(dream["features"])
        expected_ext = lambda_returns(rewards, discounts * absorb, target_values[..., 0], cfg.return_lambda)
        expected_int = lambda_returns(intrinsic, discounts, target_values[..., 1], cfg.return_lambda)
        expected = expected_ext[:, 0] + cfg.intrinsic_scale * expected_int[:, 0]
        assert torch.allclose(scores, expected, atol=1e-6)

    def test_absorbing_extrinsic_credit_with_intrinsic_still_accruing(self):
        """A candidate scoring at step 0 AND step 1 should score
        (approximately) the same as one scoring only at step 0 -- the second
        score is absorbed -- while a nonzero intrinsic_scale still lets the
        intrinsic stream accrue past a score."""
        H = 4
        cfg = PlannerConfig(gamma=0.99, return_lambda=0.9, intrinsic_scale=1.0)
        critic_target = _ConstantCriticTarget(ext_value=0.0, int_value=0.0)

        rewards_single = torch.zeros(1, H)
        rewards_single[0, 0] = 1.0
        rewards_double = torch.zeros(1, H)
        rewards_double[0, 0] = 1.0
        rewards_double[0, 1] = 1.0
        continue_prob = torch.ones(1, H)
        intrinsic_zero = torch.zeros(1, H)

        dream_single = _hand_dream(1, H, rewards_single, continue_prob, intrinsic_zero)
        dream_double = _hand_dream(1, H, rewards_double, continue_prob, intrinsic_zero)
        score_single = score_rollouts(dream_single, critic_target, cfg)
        score_double = score_rollouts(dream_double, critic_target, cfg)
        assert torch.allclose(score_single, score_double, atol=1e-6)

        # Intrinsic stream still accrues past an extrinsic score.
        intrinsic_late = torch.zeros(1, H)
        intrinsic_late[0, -1] = 1.0
        dream_with_intrinsic = _hand_dream(1, H, rewards_single, continue_prob, intrinsic_late)
        score_with_intrinsic = score_rollouts(dream_with_intrinsic, critic_target, cfg)
        assert score_with_intrinsic.item() > score_single.item()


class TestPlanGreedyFloor:
    def test_num_candidates_zero_matches_greedy_act_exactly(self, thumper):
        """Design principle 3: with num_candidates=0, plan(...) is exactly
        thumper.act(..., greedy=True) on the same inputs -- the executed
        first action is argmaxed from real features, before any imagined
        stochasticity, so this is exact equality."""
        deter, stoch, macro_context, mask = _start_states(thumper)
        cfg = PlannerConfig(num_candidates=0, horizon=3)

        torch.manual_seed(123)
        out_plan = plan(thumper, deter, stoch, macro_context, mask, cfg)
        torch.manual_seed(123)
        out_greedy = thumper.act(deter, stoch, macro_context, mask, greedy=True)

        assert torch.equal(out_plan["action_type"], out_greedy["action_type"])
        assert torch.equal(out_plan["coords"], out_greedy["coords"])
        assert bool(out_plan["greedy_chosen"])


class TestPlanMasking:
    def test_mask_restricts_action_types_and_coords_well_formed(self, thumper):
        for seed in range(5):
            torch.manual_seed(seed)
            deter, stoch, macro_context, _ = _start_states(thumper)
            mask = torch.zeros(1, NUM_ACTION_TYPES, dtype=torch.bool)
            mask[:, RESET] = True
            mask[:, ACTION3] = True
            cfg = PlannerConfig(num_candidates=4, horizon=3)

            out = plan(thumper, deter, stoch, macro_context, mask, cfg)
            action_type = int(out["action_type"].item())
            assert action_type in (RESET, ACTION3)

    def test_action6_masked_out_coords_still_well_formed(self, thumper):
        torch.manual_seed(3)
        deter, stoch, macro_context, _ = _start_states(thumper)
        mask = torch.zeros(1, NUM_ACTION_TYPES, dtype=torch.bool)
        mask[:, RESET] = True
        cfg = PlannerConfig(num_candidates=4, horizon=3)

        out = plan(thumper, deter, stoch, macro_context, mask, cfg)
        assert int(out["action_type"].item()) != ACTION6
        coords = out["coords"][0].tolist()
        assert all(isinstance(c, int) for c in coords)


class TestPlannerLeavesActorStateAlone:
    def test_reactive_and_planning_actors_end_with_identical_latent_state(self, thumper):
        """Design principle 4: driving two OnlineActors (one reactive, one
        planning) through identical begin_episode/act/observe sequences (same
        frames/rewards, feeding the reactive actor's own actions to both
        observe calls) must leave them with bit-identical _deter/_stoch/
        _macro_context -- imagined states never touch real posterior state."""
        GRID = thumper.config.world_model.grid_size
        frame0 = torch.randint(0, 16, (GRID, GRID))
        frames = [torch.randint(0, 16, (GRID, GRID)) for _ in range(4)]
        available = [RESET, 1, 2]

        reactive = OnlineActor(thumper, "cpu")
        planning = OnlineActor(thumper, "cpu", planner=PlannerConfig(num_candidates=2, horizon=2))

        torch.manual_seed(0)
        reactive.begin_episode(frame0)
        torch.manual_seed(0)
        planning.begin_episode(frame0)

        for t in range(4):
            torch.manual_seed(100 + t)
            action_type, coords, _mask = reactive.act(available, greedy=True)
            torch.manual_seed(100 + t)
            planning.act(available, greedy=True)

            reactive.observe(action_type, coords, 0.0, frames[t])
            planning.observe(action_type, coords, 0.0, frames[t])

        assert torch.equal(reactive._deter, planning._deter)
        assert torch.equal(reactive._stoch, planning._stoch)
        assert torch.equal(reactive._macro_context, planning._macro_context)


class TestDreamGreedyPassthrough:
    def test_greedy_dream_deterministic_action_types_at_step_0(self, thumper):
        deter, stoch, macro_context, mask = _start_states(thumper, N=2)

        torch.manual_seed(55)
        dream_a = thumper.dream(deter, stoch, macro_context, mask, horizon=3, greedy=True)
        torch.manual_seed(55)
        dream_b = thumper.dream(deter, stoch, macro_context, mask, horizon=3, greedy=True)

        assert torch.equal(dream_a["action_types"][:, 0], dream_b["action_types"][:, 0])

    def test_greedy_defaults_to_false(self, thumper):
        deter, stoch, macro_context, mask = _start_states(thumper, N=2)
        torch.manual_seed(1)
        dream_default = thumper.dream(deter, stoch, macro_context, mask, horizon=3)
        torch.manual_seed(1)
        dream_explicit = thumper.dream(deter, stoch, macro_context, mask, horizon=3, greedy=False)
        assert torch.equal(dream_default["action_types"], dream_explicit["action_types"])
        assert torch.equal(dream_default["coords"], dream_explicit["coords"])
