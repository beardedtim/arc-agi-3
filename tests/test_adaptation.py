"""Test-time adaptation harness (tickets/0010): init_from_full's full-weight
warm start, the adapt.py novelty guard, and env_steps_to_first_score. Fast,
CPU-only, shrunken conftest Thumper -- no real env needed."""
import torch

from adapt import Args, build_trainer_config, check_novelty, env_steps_to_first_score
from tests.conftest import GRID, STACK, small_config
from training.replay_buffer import ReplayBuffer
from training.trainer import Trainer, TrainerConfig


def _fill_episode(buffer: ReplayBuffer, length: int, rewards: list[float] | None = None, game_id: int = 0):
    episode = buffer.start_episode(game_id=game_id)
    rewards = rewards or [0.0] * length
    for t in range(length):
        frame = torch.full((GRID, GRID), t, dtype=torch.long)
        buffer.add_step(
            episode,
            frame,
            action_type=t % 7,
            coords=(0, 0),
            reward=rewards[t],
            terminated=(t == length - 1),
            internal_state=0.0,
        )
    return episode


class TestInitFromFull:
    def _trainer_with_data(self, **overrides) -> Trainer:
        overrides.setdefault("output_dir", "/tmp/thumper-test-adapt-runs")
        overrides.setdefault("resume", False)
        cfg = TrainerConfig(thumper=small_config(), batch_size=2, seq_len=3, **overrides)
        trainer = Trainer(cfg)
        _fill_episode(trainer.buffer, length=10)
        return trainer

    def test_init_from_full_loads_entire_thumper(self, tmp_path):
        source = self._trainer_with_data(output_dir=str(tmp_path / "source"))
        source.train_step()
        source.save_checkpoint()

        warm = Trainer(
            TrainerConfig(
                thumper=small_config(),
                output_dir=str(tmp_path / "fresh"),
                init_from=str(source.checkpoint_path()),
                init_from_full=True,
            )
        )

        for (n1, p1), (n2, p2) in zip(
            source.thumper.policy.state_dict().items(), warm.thumper.policy.state_dict().items()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)
        for (n1, p1), (n2, p2) in zip(
            source.thumper.critic.state_dict().items(), warm.thumper.critic.state_dict().items()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)
        assert warm.grad_steps == 0
        assert warm.env_steps == 0
        assert warm.buffer.total_steps == 0

    def test_init_from_without_full_does_not_load_policy(self, tmp_path):
        source = self._trainer_with_data(output_dir=str(tmp_path / "source"))
        source.train_step()
        source.save_checkpoint()

        warm = Trainer(
            TrainerConfig(
                thumper=small_config(),
                output_dir=str(tmp_path / "fresh"),
                init_from=str(source.checkpoint_path()),
                init_from_full=False,
            )
        )

        for (n1, p1), (n2, p2) in zip(
            source.thumper.world_model.state_dict().items(), warm.thumper.world_model.state_dict().items()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)

        policy_matches = all(
            torch.equal(p1, p2)
            for p1, p2 in zip(
                source.thumper.policy.state_dict().values(), warm.thumper.policy.state_dict().values()
            )
        )
        assert not policy_matches

    def test_init_from_full_without_init_from_raises(self, tmp_path):
        try:
            Trainer(
                TrainerConfig(
                    thumper=small_config(), output_dir=str(tmp_path), init_from_full=True, resume=False
                )
            )
            assert False, "expected ValueError"
        except ValueError as e:
            assert "init_from_full" in str(e)


class TestNoveltyGuard:
    def test_raises_when_game_in_train_games(self):
        try:
            check_novelty("cd82", ["cd82", "r11l"], allow_trained_game=False)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "cd82" in str(e)

    def test_raises_when_train_games_empty(self):
        try:
            check_novelty("cd82", [], allow_trained_game=False)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "cd82" in str(e)

    def test_passes_for_held_out_game(self):
        check_novelty("cd82", ["r11l", "sk48"], allow_trained_game=False)  # must not raise

    def test_allow_trained_game_downgrades_to_warning(self):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_novelty("cd82", ["cd82"], allow_trained_game=True)
            assert len(caught) == 1
            assert "cd82" in str(caught[0].message)


class TestBuildTrainerConfigAnnealingArm:
    def test_annealing_arm_carried_through(self):
        args = Args(
            checkpoint="ckpt.pt",
            games=["cd82"],
            intrinsic_scale=1.0,
            intrinsic_scale_final=0.0,
        )
        cfg = build_trainer_config("ckpt.pt", "cd82", args)
        assert cfg.intrinsic_scale == 1.0
        assert cfg.intrinsic_scale_final == 0.0

    def test_defaults_are_constant_scale(self):
        args = Args(checkpoint="ckpt.pt", games=["cd82"])
        cfg = build_trainer_config("ckpt.pt", "cd82", args)
        assert cfg.intrinsic_scale == 1.0
        assert cfg.intrinsic_scale_final is None


class TestEnvStepsToFirstScore:
    def test_finds_first_reward_across_episodes(self):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        _fill_episode(buffer, length=5, rewards=[0.0] * 5, game_id=0)
        _fill_episode(buffer, length=4, rewards=[0.0] * 4, game_id=0)
        _fill_episode(buffer, length=6, rewards=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], game_id=0)

        assert env_steps_to_first_score(buffer) == 5 + 4 + 2

    def test_none_when_never_scored(self):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        _fill_episode(buffer, length=5, rewards=[0.0] * 5, game_id=0)
        _fill_episode(buffer, length=4, rewards=[0.0] * 4, game_id=0)

        assert env_steps_to_first_score(buffer) is None
