"""Replay buffer and trainer: buffer roundtrip (including save/load), one
training iteration on synthetic data, checkpoint/resume, and the qualitative
sample renderers. Uses small_config() so the suite stays fast and CPU-only;
no real env needed."""
import torch

from tests.conftest import GRID, STACK, small_config
from training.qualitative import save_imagination_check, save_recon_check
from training.replay_buffer import ReplayBuffer
from training.trainer import Trainer, TrainerConfig


def _fill_episode(buffer: ReplayBuffer, length: int, fill_value_offset: int = 0, game_id: int = 0):
    episode = buffer.start_episode(game_id=game_id)
    for t in range(length):
        frame = torch.full((GRID, GRID), t + fill_value_offset, dtype=torch.long)
        buffer.add_step(
            episode,
            frame,
            action_type=t % 7,
            coords=(1, 2),
            reward=0.0,
            terminated=(t == length - 1),
            internal_state=0.0,
        )
    return episode


class TestReplayBuffer:
    def test_sample_shapes_and_dtypes(self):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        _fill_episode(buffer, length=10)

        batch = buffer.sample(batch_size=4, seq_len=5)
        assert batch["observations"].shape == (4, 5, STACK, GRID, GRID)
        assert batch["observations"].dtype == torch.int64
        assert batch["action_types"].shape == (4, 5)
        assert batch["coords"].shape == (4, 5, 2)
        assert batch["is_first"].shape == (4, 5)
        assert batch["is_first"].dtype == torch.bool
        assert batch["rewards"].shape == (4, 5)
        assert batch["terminateds"].shape == (4, 5)
        assert batch["internal_states"].shape == (4, 5, 1)
        assert batch["game_ids"].shape == (4, 5)
        assert batch["game_ids"].dtype == torch.int64

    def test_frame_stack_does_not_leak_across_episode_start(self):
        """At an episode's first step, the stack must repeat frame 0 --
        never reach back into a previous episode's frames."""
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        # first episode: distinct fill value so leakage is detectable
        _fill_episode(buffer, length=5, fill_value_offset=100)
        # second episode: sample windows starting at t=0 must be all-zeros
        _fill_episode(buffer, length=5, fill_value_offset=0)

        second_episode = buffer.episodes[1]
        stack = buffer._stack(second_episode, t=0)
        assert stack.shape == (STACK, GRID, GRID)
        # every frame in the stack is frame 0 of the *second* episode (value 0),
        # never the first episode's frames (values >= 100)
        assert (stack < 100).all()
        assert (stack == 0).all()

    def test_fifo_eviction_caps_total_steps(self):
        buffer = ReplayBuffer(capacity=15, frame_stack=STACK)
        _fill_episode(buffer, length=10)
        _fill_episode(buffer, length=10)
        assert buffer.total_steps <= 15 + 10  # soft cap: last episode always survives
        assert len(buffer.episodes) < 2 or buffer.total_steps <= 20

    def test_short_episode_padded_to_seq_len(self):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        _fill_episode(buffer, length=2)
        batch = buffer.sample(batch_size=2, seq_len=8)
        assert batch["observations"].shape == (2, 8, STACK, GRID, GRID)

    def test_save_load_roundtrip(self, tmp_path):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        _fill_episode(buffer, length=6, fill_value_offset=3, game_id=2)
        _fill_episode(buffer, length=4, game_id=5)
        path = tmp_path / "buffer.pt"
        buffer.save(path)

        loaded = ReplayBuffer.load(path, capacity=1000, frame_stack=STACK)
        assert loaded.total_steps == buffer.total_steps
        assert len(loaded.episodes) == len(buffer.episodes)
        for original, restored in zip(buffer.episodes, loaded.episodes):
            assert restored.game_id == original.game_id
            assert restored.action_types == original.action_types
            assert restored.coords == original.coords
            assert restored.rewards == original.rewards
            assert restored.terminateds == original.terminateds
            assert restored.internal_states == original.internal_states
            for f1, f2 in zip(original.frames, restored.frames):
                assert f2.dtype == torch.int64
                assert torch.equal(f1, f2)


class TestTrainer:
    def _trainer_with_data(self, **overrides) -> Trainer:
        overrides.setdefault("output_dir", "/tmp/thumper-test-runs")
        overrides.setdefault("resume", False)  # never pick up stale checkpoints
        cfg = TrainerConfig(thumper=small_config(), batch_size=2, seq_len=3, **overrides)
        trainer = Trainer(cfg)
        _fill_episode(trainer.buffer, length=10)
        return trainer

    def test_train_step_losses_finite(self):
        trainer = self._trainer_with_data()
        metrics = trainer.train_step()
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        assert "total_loss" in metrics
        assert "grad_norm" in metrics

    def test_train_step_updates_world_model_not_policy(self):
        trainer = self._trainer_with_data()
        before = {n: p.clone() for n, p in trainer.thumper.named_parameters()}
        trainer.train_step()

        wm_changed = any(
            not torch.equal(before[n], p)
            for n, p in trainer.thumper.named_parameters()
            if n.startswith("world_model")
        )
        policy_changed = any(
            not torch.equal(before[n], p)
            for n, p in trainer.thumper.named_parameters()
            if n.startswith("policy")
        )
        assert wm_changed
        assert not policy_changed

    def test_kl_warmup_ramps_weight(self):
        trainer = self._trainer_with_data(kl_warmup_steps=10)
        base = trainer.config.thumper.world_model.kl_weight
        assert trainer.kl_weight() == 0.0
        trainer.grad_steps = 5
        assert abs(trainer.kl_weight() - base / 2) < 1e-9
        trainer.grad_steps = 50
        assert trainer.kl_weight() == base

    def test_checkpoint_resume_restores_state(self, tmp_path):
        trainer = self._trainer_with_data(output_dir=str(tmp_path))
        trainer.env_steps = 42
        for _ in range(3):
            trainer.train_step()
        trainer.save_checkpoint()

        resumed = Trainer(
            TrainerConfig(thumper=small_config(), batch_size=2, seq_len=3, output_dir=str(tmp_path))
        )
        assert resumed.grad_steps == trainer.grad_steps == 3
        assert resumed.env_steps == 42
        assert resumed.buffer.total_steps == trainer.buffer.total_steps
        assert len(resumed.optimizer.state_dict()["state"]) == len(
            trainer.optimizer.state_dict()["state"]
        )
        for (n1, p1), (n2, p2) in zip(
            trainer.thumper.state_dict().items(), resumed.thumper.state_dict().items()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)

    def test_init_from_warm_starts_world_model_only(self, tmp_path):
        source = self._trainer_with_data(output_dir=str(tmp_path / "source"))
        source.train_step()
        source.save_checkpoint()

        warm = Trainer(
            TrainerConfig(
                thumper=small_config(),
                output_dir=str(tmp_path / "fresh"),
                init_from=str(source.checkpoint_path()),
            )
        )
        # world-model weights copied; counters/buffer fresh
        for (n1, p1), (n2, p2) in zip(
            source.thumper.world_model.state_dict().items(),
            warm.thumper.world_model.state_dict().items(),
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)
        assert warm.grad_steps == 0
        assert warm.env_steps == 0
        assert warm.buffer.total_steps == 0


class TestQualitativeChecks:
    def _batch(self):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        _fill_episode(buffer, length=8)
        torch.manual_seed(0)
        return buffer.sample(batch_size=2, seq_len=6)

    def test_recon_check_writes_png(self, tmp_path):
        from model.world_model import WorldModel

        wm = WorldModel(small_config().world_model)
        save_recon_check(wm, self._batch(), step=7, out_dir=tmp_path, num_frames=4)
        out = tmp_path / "recon_step_0000007.png"
        assert out.exists() and out.stat().st_size > 0

    def test_imagination_check_writes_png(self, tmp_path):
        from model.world_model import WorldModel

        wm = WorldModel(small_config().world_model)
        save_imagination_check(wm, self._batch(), step=7, out_dir=tmp_path, horizon=3)
        out = tmp_path / "imagine_step_0000007.png"
        assert out.exists() and out.stat().st_size > 0
