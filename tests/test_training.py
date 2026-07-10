"""Replay buffer and trainer: buffer roundtrip (including save/load), one
training iteration on synthetic data, checkpoint/resume, and the qualitative
sample renderers. Uses small_config() so the suite stays fast and CPU-only;
no real env needed."""
import torch

from env.env import Env, StepResult
from model.actions import NUM_ACTION_TYPES, RESET
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
        assert batch["available_actions"].shape == (4, 5, NUM_ACTION_TYPES)
        assert batch["available_actions"].dtype == torch.bool

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
            for a1, a2 in zip(original.available_actions, restored.available_actions):
                assert a2.dtype == torch.bool
                assert torch.equal(a1, a2)
            for f1, f2 in zip(original.frames, restored.frames):
                assert f2.dtype == torch.int64
                assert torch.equal(f1, f2)

    def test_available_actions_roundtrip_with_real_masks(self, tmp_path):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        episode = buffer.start_episode(game_id=0)
        mask = torch.zeros(NUM_ACTION_TYPES, dtype=torch.bool)
        mask[0] = True
        mask[3] = True
        buffer.add_step(
            episode,
            torch.zeros(GRID, GRID, dtype=torch.long),
            action_type=3,
            coords=(0, 0),
            reward=0.0,
            terminated=True,
            internal_state=0.0,
            available_actions=mask,
        )
        path = tmp_path / "buffer.pt"
        buffer.save(path)
        loaded = ReplayBuffer.load(path, capacity=1000, frame_stack=STACK)
        assert torch.equal(loaded.episodes[0].available_actions[0], mask)

    def test_load_without_available_actions_key_falls_back_to_all_true(self, tmp_path):
        """Backward compat: a pre-0003 buffer.pt has no 'available_actions'
        key -- loading it must not crash, and every step's mask defaults to
        all-True (what 'no mask' already means to Policy.act)."""
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        episode = buffer.start_episode(game_id=0)
        for t in range(3):
            episode.frames.append(torch.zeros(GRID, GRID, dtype=torch.long))
            episode.action_types.append(0)
            episode.coords.append((0, 0))
            episode.rewards.append(0.0)
            episode.terminateds.append(t == 2)
            episode.internal_states.append(0.0)
        buffer.total_steps = 3

        # Hand-build a payload without the available_actions key, mirroring
        # a pre-0003 buffer.save().
        payload = {
            "episodes": [
                {
                    "game_id": episode.game_id,
                    "frames": torch.stack(episode.frames, dim=0).to(torch.uint8),
                    "action_types": torch.tensor(episode.action_types, dtype=torch.int16),
                    "coords": torch.tensor(episode.coords, dtype=torch.int16),
                    "rewards": torch.tensor(episode.rewards, dtype=torch.float32),
                    "terminateds": torch.tensor(episode.terminateds, dtype=torch.bool),
                    "internal_states": torch.tensor(episode.internal_states, dtype=torch.float32),
                }
            ]
        }
        path = tmp_path / "old_buffer.pt"
        torch.save(payload, path)

        loaded = ReplayBuffer.load(path, capacity=1000, frame_stack=STACK)
        assert len(loaded.episodes[0].available_actions) == 3
        for mask in loaded.episodes[0].available_actions:
            assert mask.dtype == torch.bool
            assert mask.all()


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

    def test_train_step_with_burn_in_end_to_end(self):
        """tickets/0004 smoke test: one synthetic train_step + implicit
        policy_train_step at a small explicit burn_in, exercising the
        sliced-window plumbing through compute_losses and the dream starts."""
        trainer = self._trainer_with_data(burn_in=2)
        metrics = trainer.train_step()
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        assert "total_loss" in metrics
        assert "actor_loss" in metrics
        assert "critic_loss" in metrics

    def test_train_step_updates_world_model_and_policy(self):
        """train_step now does two updates per grad step (tickets/0003): the
        world-model pass, and an actor-critic update trained in imagination
        -- both policy and critic parameters should move."""
        trainer = self._trainer_with_data()
        before = {n: p.clone() for n, p in trainer.thumper.named_parameters()}
        trainer.train_step()

        def changed(prefix: str) -> bool:
            return any(
                not torch.equal(before[n], p)
                for n, p in trainer.thumper.named_parameters()
                if n.startswith(prefix)
            )

        assert changed("world_model")
        assert changed("policy")
        assert changed("critic.")
        assert not changed("critic_target")

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
        assert len(resumed.actor_optimizer.state_dict()["state"]) == len(
            trainer.actor_optimizer.state_dict()["state"]
        )
        assert len(resumed.critic_optimizer.state_dict()["state"]) == len(
            trainer.critic_optimizer.state_dict()["state"]
        )
        assert resumed.return_normalizer_ext.scale == trainer.return_normalizer_ext.scale
        assert resumed.return_normalizer_int.scale == trainer.return_normalizer_int.scale
        for (n1, p1), (n2, p2) in zip(
            trainer.thumper.state_dict().items(), resumed.thumper.state_dict().items()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)

    def test_collect_step_with_policy(self, tmp_path, monkeypatch):
        """After prefill, the online collector must act with the policy
        (Thumper.act via Trainer._act), not _random_action -- exercised
        directly since a real env isn't available in the test suite."""
        trainer = self._trainer_with_data(output_dir=str(tmp_path), prefill_steps=0)
        result, episode = trainer._begin_episode()
        action_type, coords, mask = trainer._act(result.available_actions)
        assert 0 <= action_type < trainer.config.thumper.world_model.num_action_types
        assert len(coords) == 2
        assert mask.dtype == torch.bool
        trainer._step_latent(action_type, coords, 0.0, result.frame)
        # latent state advanced: frame stack now holds the (repeated) new frame
        assert len(trainer._frame_stack) == trainer.config.thumper.world_model.frame_stack

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


class TestTrainGamesFilter:
    """tickets/0009: train_games restricts the online collector's round-robin
    to a subset of downloaded games, and a resume with a different list must
    refuse to run (buffer contamination guard)."""

    def _fake_result(self, game: str) -> StepResult:
        return StepResult(
            frame=torch.zeros(GRID, GRID, dtype=torch.long),
            reward=0,
            done=False,
            won=False,
            available_actions=[RESET],
            levels_completed=0,
        )

    def _wire_fake_env(self, trainer: Trainer, games: list[str]) -> None:
        """Point the trainer's real Env instance at a scripted 3-game
        catalog -- games()/reset()/step() only, following
        TestTruncation's pattern (no real arc_agi env touched)."""
        trainer.env.games = lambda: games
        trainer.env.reset = lambda game: self._fake_result(game)
        trainer.env.step = lambda action_type, x=None, y=None: self._fake_result(trainer._current_game)

    def _trainer(self, tmp_path, games=("g1", "g2", "g3"), **overrides) -> Trainer:
        overrides.setdefault("output_dir", str(tmp_path))
        overrides.setdefault("resume", False)
        cfg = TrainerConfig(thumper=small_config(), batch_size=2, seq_len=3, **overrides)
        trainer = Trainer(cfg)
        self._wire_fake_env(trainer, list(games))
        return trainer

    def test_filter_respected_and_sorted(self, tmp_path):
        trainer = self._trainer(tmp_path, train_games=["g2", "g1"])
        # sorted order preserved from Env.games(), not train_games' order
        assert trainer._games() == ["g1", "g2"]
        assert trainer.game_ids() == {"g1": 0, "g2": 1}

    def test_filter_restricts_round_robin_and_buffer_game_ids(self, tmp_path):
        trainer = self._trainer(tmp_path, train_games=["g2", "g1"], prefill_steps=10_000)
        seen_games = set()
        for _ in range(6):
            result, episode = trainer._begin_episode()
            seen_games.add(trainer._current_game)
            trainer._step_latent(RESET, (0, 0), 0.0, result.frame)

        assert seen_games <= {"g1", "g2"}
        assert "g3" not in seen_games
        id_to_game = {i: g for g, i in trainer.game_ids().items()}
        for episode in trainer.buffer.episodes:
            assert id_to_game[episode.game_id] in {"g1", "g2"}

    def test_unknown_game_raises_loudly(self, tmp_path):
        trainer = self._trainer(tmp_path, train_games=["g1", "nope"])
        try:
            trainer._games()
            assert False, "expected ValueError"
        except ValueError as e:
            assert "nope" in str(e)
            assert "g1" in str(e) and "g2" in str(e) and "g3" in str(e)

    def test_default_empty_list_is_all_games(self, tmp_path):
        trainer = self._trainer(tmp_path, train_games=[])
        assert trainer._games() == ["g1", "g2", "g3"]

    def test_resume_guard_fires_on_mismatch(self, tmp_path):
        trainer = self._trainer(tmp_path, train_games=["g1"])
        trainer.env_steps = 5
        trainer.save_checkpoint()

        try:
            Trainer(TrainerConfig(thumper=small_config(), output_dir=str(tmp_path), train_games=["g1", "g2"]))
            assert False, "expected ValueError"
        except ValueError as e:
            assert "g1" in str(e)

    def test_resume_guard_allows_matching_list(self, tmp_path):
        trainer = self._trainer(tmp_path, train_games=["g1"])
        trainer.env_steps = 5
        trainer.save_checkpoint()

        resumed = Trainer(TrainerConfig(thumper=small_config(), output_dir=str(tmp_path), train_games=["g1"]))
        assert resumed.env_steps == 5

    def test_resume_guard_old_checkpoint_defaults_to_empty_list(self, tmp_path):
        """A checkpoint saved before this ticket's trainer_config.train_games
        field existed resumes cleanly only against train_games=[] (the
        implicit historical behavior of training on every game)."""
        trainer = self._trainer(tmp_path, train_games=[])
        trainer.env_steps = 5
        trainer.save_checkpoint()

        # simulate an old-style payload missing the field entirely
        payload = torch.load(trainer.checkpoint_path(), weights_only=False)
        del payload["trainer_config"].train_games
        torch.save(payload, trainer.checkpoint_path())

        resumed = Trainer(TrainerConfig(thumper=small_config(), output_dir=str(tmp_path), train_games=[]))
        assert resumed.env_steps == 5

        try:
            Trainer(TrainerConfig(thumper=small_config(), output_dir=str(tmp_path), train_games=["g1"]))
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_eval_independent_of_train_games(self, tmp_path):
        """train_games restricts collection only -- eval_games names any
        downloaded game regardless, and the collector's buffer never touches
        a held-out game (tickets/0009 Design principle 1 / Step 3.5)."""
        trainer = self._trainer(
            tmp_path, train_games=["g1"], eval_games=["g2"], eval_episodes_per_game=1, prefill_steps=10_000
        )
        eval_env = Env()
        eval_env.games = lambda: ["g1", "g2", "g3"]
        eval_env.reset = lambda game: self._fake_result(game)
        eval_env.step = lambda action_type, x=None, y=None: self._fake_result("g2")
        trainer._eval_env = eval_env

        for _ in range(3):
            result, episode = trainer._begin_episode()
            trainer._step_latent(RESET, (0, 0), 0.0, result.frame)

        trainer._run_eval_hook()

        id_to_game = {i: g for g, i in trainer.game_ids().items()}
        for episode in trainer.buffer.episodes:
            assert id_to_game[episode.game_id] == "g1"


class TestTruncation:
    """tickets/0005: the continue head must only ever see real deaths --
    a step-cap timeout is truncation, not termination."""

    def _fake_result(self, done: bool = False) -> StepResult:
        return StepResult(
            frame=torch.zeros(GRID, GRID, dtype=torch.long),
            reward=0,
            done=done,
            won=False,
            available_actions=[RESET],
            levels_completed=0,
        )

    def _trainer(self, tmp_path, **overrides):
        overrides.setdefault("output_dir", str(tmp_path))
        overrides.setdefault("resume", False)
        overrides.setdefault("prefill_steps", 10_000)  # stay on random-action path
        cfg = TrainerConfig(thumper=small_config(), batch_size=2, seq_len=3, **overrides)
        trainer = Trainer(cfg)
        trainer.env.games = lambda: ["fake_game"]
        trainer.env.reset = lambda game: self._fake_result(done=False)
        return trainer

    def test_step_cap_stores_terminated_false_and_still_rotates_episode(self, tmp_path):
        trainer = self._trainer(tmp_path, timeout_env_steps=1, total_env_steps=1)
        trainer.env.step = lambda action_type, x=None, y=None: self._fake_result(done=False)
        trainer.train()

        # exactly one env step happened, hit the 1-step cap, rotated to a
        # new episode -- the buffer's first episode's second (capped) entry
        # must be terminated=False.
        first_episode = trainer.buffer.episodes[0]
        assert first_episode.terminateds[-1] is False
        assert len(trainer.buffer.episodes) == 2  # rotated into a fresh episode

    def test_real_done_stores_terminated_true(self, tmp_path):
        trainer = self._trainer(tmp_path, timeout_env_steps=1_000_000, total_env_steps=1)
        trainer.env.step = lambda action_type, x=None, y=None: self._fake_result(done=True)
        trainer.train()

        first_episode = trainer.buffer.episodes[0]
        assert first_episode.terminateds[-1] is True
        assert len(trainer.buffer.episodes) == 2  # done still rotates the episode


class TestStratifiedSampling:
    def _episode_with_one_reward(self, buffer: ReplayBuffer, length: int, reward_t: int, game_id: int = 0):
        episode = buffer.start_episode(game_id=game_id)
        for t in range(length):
            frame = torch.full((GRID, GRID), t, dtype=torch.long)
            buffer.add_step(
                episode,
                frame,
                action_type=t % 7,
                coords=(1, 2),
                reward=1.0 if t == reward_t else 0.0,
                terminated=(t == length - 1),
                internal_state=0.0,
            )
        return episode

    def test_reward_frac_one_puts_event_in_every_loss_window(self):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        self._episode_with_one_reward(buffer, length=30, reward_t=20)

        loss_offset = 4
        seq_len = 10
        for _ in range(5):
            batch = buffer.sample(batch_size=8, seq_len=seq_len, reward_frac=1.0, loss_offset=loss_offset)
            loss_window_rewards = batch["rewards"][:, loss_offset:]
            assert (loss_window_rewards != 0).any(dim=1).all()

    def test_reward_frac_with_zero_events_falls_back_to_uniform(self):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        _fill_episode(buffer, length=10)  # all-zero rewards
        batch = buffer.sample(batch_size=4, seq_len=5, reward_frac=0.25, loss_offset=1)
        assert batch["observations"].shape == (4, 5, STACK, GRID, GRID)

    def test_event_position_within_loss_window_varies_across_draws(self):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        self._episode_with_one_reward(buffer, length=40, reward_t=25)

        loss_offset = 4
        seq_len = 12
        positions = set()
        for _ in range(20):
            batch = buffer.sample(batch_size=4, seq_len=seq_len, reward_frac=1.0, loss_offset=loss_offset)
            nonzero = (batch["rewards"][:, loss_offset:] != 0).nonzero()
            for row in nonzero:
                positions.add(int(row[1]))
        assert len(positions) > 1


class TestQualitativeChecks:
    def _batch(self):
        buffer = ReplayBuffer(capacity=1000, frame_stack=STACK)
        _fill_episode(buffer, length=8)
        torch.manual_seed(0)
        return buffer.sample(batch_size=2, seq_len=6)

    def test_recon_check_writes_png(self, tmp_path):
        from model.world_model import WorldModel

        wm = WorldModel(small_config().world_model)
        save_recon_check(wm, self._batch(), burn_in=2, step=7, out_dir=tmp_path, num_frames=4)
        out = tmp_path / "recon_step_0000007.png"
        assert out.exists() and out.stat().st_size > 0

    def test_imagination_check_writes_png(self, tmp_path):
        from model.world_model import WorldModel

        wm = WorldModel(small_config().world_model)
        save_imagination_check(wm, self._batch(), burn_in=2, step=7, out_dir=tmp_path, horizon=3)
        out = tmp_path / "imagine_step_0000007.png"
        assert out.exists() and out.stat().st_size > 0
