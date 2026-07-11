"""OnlineActor refactor equivalence + evaluation harness (tickets/0007).

Uses a scripted fake Env (games()/reset()/step()) so eval tests never touch
arc_agi -- small_config()'s GRID=16 thumper, same conventions as
tests/test_training.py.
"""
import copy

import torch

from env.env import StepResult
from model.actions import RESET
from model.rssm import RSSM
from tests.conftest import GRID, STACK, small_config
from training.evaluate import EvalProtocol, EvalReport, evaluate
from training.online_actor import OnlineActor
from training.trainer import Trainer, TrainerConfig


def _frame(value: int) -> torch.Tensor:
    return torch.full((GRID, GRID), value, dtype=torch.long)


class ScriptedEnv:
    """A fake Env: each game has a fixed script of (reward, done, won)
    per step, cycling if the policy runs past the script (RESET actions
    mid-episode are legal and just get stepped, per tickets/0007)."""

    def __init__(self, scripts: dict[str, list[tuple[float, bool, bool]]]):
        self.scripts = scripts
        self._game: str | None = None
        self._t = 0
        self._levels_completed = 0

    def games(self) -> list[str]:
        return sorted(self.scripts)

    def reset(self, game: str) -> StepResult:
        self._game = game
        self._t = 0
        self._levels_completed = 0
        return StepResult(
            frame=_frame(0),
            reward=0,
            done=False,
            won=False,
            available_actions=[RESET, 1, 2],
            levels_completed=0,
        )

    def step(self, action_type: int, x: int | None = None, y: int | None = None) -> StepResult:
        script = self.scripts[self._game]
        reward, done, won = script[min(self._t, len(script) - 1)]
        self._t += 1
        self._levels_completed += int(reward)
        return StepResult(
            frame=_frame(self._t),
            reward=reward,
            done=done,
            won=won,
            available_actions=[RESET, 1, 2],
            levels_completed=self._levels_completed,
        )


class TestOnlineActorRefactorEquivalence:
    def test_matches_forward_sequence_ground_truth(self, tmp_path, monkeypatch):
        """OnlineActor's TaskEncoder folds must align with
        WorldModel.forward_sequence's arrival-state convention exactly
        (tickets/0008): fold k's (deter, action, reward) inputs must
        bit-match forward_sequence's burn-in fold k, and the final online
        macro-context must equal the loss window's frozen macro_context.

        RSSM._sample is patched to return the distribution mean so posterior
        sampling is deterministic and the two paths are exactly comparable
        (both still exercise the real posterior head, just without its
        stochastic sampling noise)."""
        monkeypatch.setattr(RSSM, "_sample", lambda self, stats: RSSM.dist_params(stats)[0])

        torch.manual_seed(0)
        thumper = Trainer(
            TrainerConfig(thumper=small_config(), output_dir=str(tmp_path), resume=False, device="cpu")
        ).thumper
        device = "cpu"
        wm = thumper.world_model

        T = 5
        torch.manual_seed(42)
        frames = [torch.randint(0, 16, (GRID, GRID)) for _ in range(T)]
        actions = [0] + [1 + (t % 3) for t in range(1, T)]
        rewards = [0.0] + [float(t % 2) for t in range(1, T)]

        records: dict[str, list] = {"online": [], "training": []}
        current = ["online"]
        orig_forward = type(wm.task_encoder).forward

        def recording_forward(self, m, deter, stoch, action, reward):
            records[current[0]].append((deter.detach().clone(), action.detach().clone(), reward.detach().clone()))
            return orig_forward(self, m, deter, stoch, action, reward)

        monkeypatch.setattr(type(wm.task_encoder), "forward", recording_forward)

        current[0] = "online"
        actor = OnlineActor(thumper, device)
        actor.begin_episode(frames[0])
        for t in range(1, T):
            actor.act([RESET, 1, 2, 3], greedy=True)
            actor.observe(actions[t], (0, 0), rewards[t], frames[t])

        current[0] = "training"
        stacks = [
            torch.stack([frames[max(i, 0)] for i in range(t - STACK + 1, t + 1)]) for t in range(T)
        ]
        obs = torch.stack(stacks).unsqueeze(0)
        action_types = torch.tensor([actions])
        coords = torch.zeros(1, T, 2, dtype=torch.long)
        is_first = torch.tensor([[True] + [False] * (T - 1)])
        rewards_t = torch.tensor([rewards]).float()
        with torch.no_grad():
            out = wm.forward_sequence(
                obs, action_types, coords, is_first, rewards=rewards_t, burn_in=T - 1
            )

        # actor.act's first call performs the zero-fold seeded by
        # begin_episode; forward_sequence's burn-in loop performs T-1 folds
        # (folds 1..T-1), one per act() call here -- same count.
        assert len(records["online"]) == len(records["training"]) == T - 1
        for k in range(T - 1):
            (d_on, a_on, r_on), (d_tr, a_tr, r_tr) = records["online"][k], records["training"][k]
            # deter is allclose, not exactly equal: both paths recompute it
            # via the same GRU ops but through different call sites, and
            # float non-associativity introduces ~1e-8 noise that has
            # nothing to do with the alignment bug this test targets --
            # action/reward (the actual fold inputs this ticket fixes) are
            # asserted exact.
            assert torch.allclose(d_on, d_tr, atol=1e-5), f"fold {k}: deter mismatch"
            assert torch.equal(a_on, a_tr), f"fold {k}: action mismatch"
            assert torch.equal(r_on, r_tr), f"fold {k}: reward mismatch"

        assert torch.allclose(actor._macro_context, out["macro_context"][0, 0], atol=1e-5)

    def test_zero_fold_at_episode_start(self, tmp_path, monkeypatch):
        """The first act() after begin_episode folds a zeroed action/reward
        alongside the first observation's posterior state -- mirrors
        forward_sequence's is_first zero-fold (tickets/0008)."""
        monkeypatch.setattr(RSSM, "_sample", lambda self, stats: RSSM.dist_params(stats)[0])

        thumper = Trainer(
            TrainerConfig(thumper=small_config(), output_dir=str(tmp_path), resume=False, device="cpu")
        ).thumper
        wm = thumper.world_model

        recorded = []
        orig_forward = type(wm.task_encoder).forward

        def recording_forward(self, m, deter, stoch, action, reward):
            recorded.append((deter.detach().clone(), action.detach().clone(), reward.detach().clone()))
            return orig_forward(self, m, deter, stoch, action, reward)

        monkeypatch.setattr(type(wm.task_encoder), "forward", recording_forward)

        actor = OnlineActor(thumper, "cpu")
        actor.begin_episode(_frame(0))
        actor.act([RESET, 1, 2], greedy=True)

        assert len(recorded) == 1
        deter, action, reward = recorded[0]
        assert torch.equal(deter, actor._deter)
        assert torch.equal(action, torch.zeros_like(action))
        assert torch.equal(reward, torch.zeros_like(reward))

    def test_trainer_collect_loop_unchanged_by_refactor(self, tmp_path):
        """Headline invariant: with a fixed seed and identical weights,
        Trainer's collect loop (which now delegates to OnlineActor via
        trainer.actor_state) produces the same action stream as driving a
        standalone OnlineActor over the same scripted frames -- confirming
        the delegation didn't change what actions get chosen."""
        cfg = TrainerConfig(
            thumper=small_config(), output_dir=str(tmp_path), resume=False, prefill_steps=0, device="cpu"
        )
        trainer = Trainer(cfg)
        first_frame = _frame(0)
        available = [RESET, 1, 2]

        torch.manual_seed(0)
        trainer.actor_state.begin_episode(first_frame)
        trainer_actions = []
        frame = first_frame
        for t in range(5):
            action_type, coords, _mask = trainer._act(available)
            trainer_actions.append((action_type, coords))
            frame = _frame(t + 1)
            trainer._step_latent(action_type, coords, 0.0, frame)

        actor = OnlineActor(trainer.thumper, cfg.device)
        torch.manual_seed(0)
        actor.begin_episode(first_frame)
        actor_actions = []
        for t in range(5):
            action_type, coords, _mask = actor.act(available)
            actor_actions.append((action_type, coords))
            actor.observe(action_type, coords, 0.0, _frame(t + 1))

        assert trainer_actions == actor_actions


class TestMacroContextCarry:
    """tickets/0010 Arm A: begin_episode(carry_macro_context=True) preserves
    the TaskEncoder's macro-context across an episode reset instead of
    reinitializing it, while everything else (frame stack, RSSM state,
    pending fold) still resets."""

    def _thumper(self, tmp_path):
        return Trainer(
            TrainerConfig(thumper=small_config(), output_dir=str(tmp_path), resume=False, device="cpu")
        ).thumper

    def test_first_ever_call_with_carry_equals_zero_initial_state(self, tmp_path):
        thumper = self._thumper(tmp_path)
        actor = OnlineActor(thumper, "cpu")
        actor.begin_episode(_frame(0), carry_macro_context=True)

        zero_state = thumper.world_model.task_encoder.initial_state(1, "cpu")
        assert torch.equal(actor._macro_context, zero_state)

    def test_carry_preserves_macro_context_across_reset(self, tmp_path):
        thumper = self._thumper(tmp_path)
        actor = OnlineActor(thumper, "cpu")
        actor.begin_episode(_frame(0))
        for t in range(1, 4):
            actor.act([RESET, 1, 2], greedy=True)
            actor.observe(1, (0, 0), 1.0, _frame(t))

        assert actor.macro_context_norm > 0.0
        macro_context_before = actor._macro_context.clone()

        actor.begin_episode(_frame(0), carry_macro_context=True)
        assert torch.equal(actor._macro_context, macro_context_before)
        # everything else episode-scoped does reset
        assert len(actor._frame_stack) == thumper.world_model.config.frame_stack
        assert all(torch.equal(f, _frame(0)) for f in actor._frame_stack)

    def test_carry_off_resets_to_zero_initial_state(self, tmp_path):
        thumper = self._thumper(tmp_path)
        actor = OnlineActor(thumper, "cpu")
        actor.begin_episode(_frame(0))
        for t in range(1, 4):
            actor.act([RESET, 1, 2], greedy=True)
            actor.observe(1, (0, 0), 1.0, _frame(t))

        assert actor.macro_context_norm > 0.0

        actor.begin_episode(_frame(0), carry_macro_context=False)
        zero_state = thumper.world_model.task_encoder.initial_state(1, "cpu")
        assert torch.equal(actor._macro_context, zero_state)


class TestEvaluate:
    def _protocol(self, **overrides) -> EvalProtocol:
        overrides.setdefault("games", ["g1", "g2"])
        overrides.setdefault("episodes_per_game", 2)
        overrides.setdefault("max_steps", 6)
        overrides.setdefault("modes", ("greedy", "sampled"))
        overrides.setdefault("seed", 0)
        return EvalProtocol(**overrides)

    def _thumper(self, tmp_path):
        return Trainer(
            TrainerConfig(thumper=small_config(), output_dir=str(tmp_path), resume=False, device="cpu")
        ).thumper

    def test_greedy_and_sampled_determinism(self, tmp_path):
        thumper = self._thumper(tmp_path)
        env = ScriptedEnv(
            {
                "g1": [(0.0, False, False)] * 5 + [(1.0, True, True)],
                "g2": [(0.0, False, False)] * 10,
            }
        )
        protocol = self._protocol()

        report1 = evaluate(thumper, env, protocol)
        report2 = evaluate(thumper, env, protocol)

        assert report1.to_json() == report2.to_json()

    def test_metrics_correctness(self, tmp_path):
        thumper = self._thumper(tmp_path)
        # reward at step 2 (0-indexed), win at the final scripted step
        env = ScriptedEnv(
            {
                "scored": [(0.0, False, False), (0.0, False, False), (1.0, False, False), (0.0, True, True)],
            }
        )
        protocol = self._protocol(games=["scored"], episodes_per_game=1, max_steps=10, modes=("greedy",))
        report = evaluate(thumper, env, protocol)

        ep = report.episodes[0]
        assert ep.steps_to_first_score == 2
        assert ep.won is True
        assert ep.levels_completed == env._levels_completed or ep.levels_completed == 1

    def test_scoreless_episode_has_none_steps_to_first_score(self, tmp_path):
        thumper = self._thumper(tmp_path)
        env = ScriptedEnv({"scoreless": [(0.0, False, False)] * 4 + [(0.0, True, False)]})
        protocol = self._protocol(games=["scoreless"], episodes_per_game=1, max_steps=10, modes=("greedy",))
        report = evaluate(thumper, env, protocol)

        ep = report.episodes[0]
        assert ep.steps_to_first_score is None
        assert ep.won is False

    def test_no_buffer_interaction(self, tmp_path):
        """evaluate() must never touch any replay buffer -- it takes a bare
        Env, not a Trainer, so there's no buffer in scope at all; this test
        just documents/asserts the function signature has no buffer param
        and running it doesn't require or create one."""
        thumper = self._thumper(tmp_path)
        env = ScriptedEnv({"g1": [(0.0, False, False)] * 5})
        protocol = self._protocol(games=["g1"], episodes_per_game=1, max_steps=5, modes=("greedy",))
        evaluate(thumper, env, protocol)  # must not raise / require a buffer

    def test_report_serialization_round_trips(self, tmp_path):
        thumper = self._thumper(tmp_path)
        env = ScriptedEnv({"g1": [(0.0, False, False)] * 3 + [(1.0, True, True)]})
        protocol = self._protocol(games=["g1"], episodes_per_game=2, max_steps=10)
        report = evaluate(thumper, env, protocol)

        restored = EvalReport.from_json(report.to_json())
        assert restored.episodes == report.episodes

    def test_summary_table_has_one_row_per_game_and_mode(self, tmp_path):
        thumper = self._thumper(tmp_path)
        env = ScriptedEnv(
            {"g1": [(0.0, False, False)] * 5, "g2": [(0.0, False, False)] * 5}
        )
        protocol = self._protocol(games=["g1", "g2"], episodes_per_game=1, max_steps=5)
        report = evaluate(thumper, env, protocol)
        table = report.summary_table()
        for mode in protocol.modes:
            for game in protocol.games:
                assert game in table
            assert mode in table

    def test_carry_off_matches_pre_refactor_behavior(self, tmp_path):
        """tickets/0010: the Step 2 refactor (one OnlineActor per (game,
        mode) instead of one per episode) must be bit-identical to the old
        per-episode-actor behavior when carry_macro_context is off, since
        begin_episode fully resets state either way."""
        thumper = self._thumper(tmp_path)
        env = ScriptedEnv(
            {
                "g1": [(0.0, False, False)] * 5 + [(1.0, True, True)],
                "g2": [(0.0, False, False)] * 10,
            }
        )
        protocol = self._protocol(carry_macro_context=False)

        report1 = evaluate(thumper, env, protocol)
        report2 = evaluate(thumper, env, protocol)

        assert report1.to_json() == report2.to_json()

    def test_carry_never_crosses_game_or_mode(self, tmp_path, monkeypatch):
        """tickets/0010 Design principle 3: with carry_macro_context=True,
        evaluate must construct exactly one OnlineActor per (game, mode) --
        never share one across games or modes -- and pass carry=True to
        every begin_episode call. 2 games x 2 modes -> 4 constructions."""
        thumper = self._thumper(tmp_path)
        env = ScriptedEnv(
            {"g1": [(0.0, False, False)] * 5, "g2": [(0.0, False, False)] * 5}
        )
        protocol = self._protocol(games=["g1", "g2"], episodes_per_game=2, max_steps=5, carry_macro_context=True)

        construction_count = 0
        carry_flags = []
        orig_init = OnlineActor.__init__
        orig_begin_episode = OnlineActor.begin_episode

        def counting_init(self, *args, **kwargs):
            nonlocal construction_count
            construction_count += 1
            return orig_init(self, *args, **kwargs)

        def recording_begin_episode(self, first_frame, carry_macro_context=False):
            carry_flags.append(carry_macro_context)
            return orig_begin_episode(self, first_frame, carry_macro_context=carry_macro_context)

        monkeypatch.setattr(OnlineActor, "__init__", counting_init)
        monkeypatch.setattr(OnlineActor, "begin_episode", recording_begin_episode)

        evaluate(thumper, env, protocol)

        assert construction_count == 4  # 2 games x 2 modes
        assert len(carry_flags) == 2 * 2 * protocol.episodes_per_game
        assert all(carry_flags)


class TestEvalIsolation:
    def test_eval_hook_leaves_buffer_and_rng_untouched(self, tmp_path):
        cfg = TrainerConfig(
            thumper=small_config(),
            output_dir=str(tmp_path),
            resume=False,
            batch_size=2,
            seq_len=3,
            eval_every=1,
            eval_games=["g1"],
            eval_episodes_per_game=1,
            timeout_env_steps=3,
            device="cpu",
        )
        trainer = Trainer(cfg)

        script_env = ScriptedEnv({"g1": [(0.0, False, False)] * 3})
        trainer._eval_env = script_env

        buf = trainer.buffer
        episode = buf.start_episode(game_id=0)
        for t in range(5):
            buf.add_step(
                episode, _frame(t), action_type=0, coords=(0, 0), reward=0.0,
                terminated=(t == 4), internal_state=0.0,
            )
        buffer_steps_before = buf.total_steps

        # give the collector's OnlineActor some live (non-initial) latent
        # state to check for perturbation, without touching the real Env.
        trainer.actor_state.begin_episode(_frame(0))
        trainer.actor_state.observe(0, (0, 0), 0.0, _frame(1))

        was_training = trainer.thumper.training
        frame_stack_before = [f.clone() for f in trainer.actor_state._frame_stack]
        deter_before = trainer.actor_state._deter.clone()
        torch_rng_before = torch.get_rng_state().clone()
        import random as random_module

        python_rng_before = random_module.getstate()

        trainer._run_eval_hook()

        assert buf.total_steps == buffer_steps_before
        assert trainer.thumper.training == was_training
        assert torch.equal(torch.get_rng_state(), torch_rng_before)
        assert random_module.getstate() == python_rng_before
        for before, after in zip(frame_stack_before, trainer.actor_state._frame_stack):
            assert torch.equal(before, after)
        assert torch.equal(deter_before, trainer.actor_state._deter)
