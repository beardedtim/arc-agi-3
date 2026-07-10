"""
Trainer :: online world-model training loop for Thumper.

Interleaves collection and training the way theo's online loop does: each
env step (uniform-random policy over available_actions, round-robin across
every downloaded game) feeds the replay buffer, and gradient steps on
WorldModel.compute_losses catch up at a fixed env-steps-per-grad-step ratio
(`train_every`) after a `prefill_steps` warmup. The policy module is
untouched (see tickets/0001): only thumper.world_model.parameters() are
optimized.

Telemetry goes three places:
  - TensorBoard scalars under `<output_dir>/tb` (losses, per-game episode
    returns/lengths/win rates, ensemble disagreement, perf counters);
  - qualitative recon/imagination sample PNGs under `<output_dir>/samples`
    (see training/qualitative.py);
  - a console line every `log_every` grad steps with an ETA.

Checkpointing writes `<output_dir>/latest.pt` (Thumper.load-compatible:
weights + config, plus optimizer state and step counters) and `buffer.pt`
(the replay buffer) on an env-step cadence, both via write-temp-then-rename
so a crash mid-save can't leave a truncated file that blocks the next
resume. A fresh Trainer resumes from them automatically when
`config.resume` (the default) finds them in `output_dir`.
"""
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from env.env import Env, StepResult
from model.actions import ACTION6, NUM_ACTION_TYPES, RESET
from model.thumper import Thumper, ThumperConfig
from training.qualitative import save_imagination_check, save_recon_check
from training.replay_buffer import Episode, ReplayBuffer

# The API's score field (FrameDataRaw.levels_completed) is documented 0-254;
# normalizing keeps the internal_state_head's MSE target on a sane scale.
MAX_SCORE = 254.0


@dataclass
class TrainerConfig:
    thumper: ThumperConfig = field(default_factory=ThumperConfig)

    # loop budget & scheduling
    total_env_steps: int = 100_000
    """Env steps to run before returning (episode resets don't count)."""
    train_every: int = 2
    """Gradient step every this many env steps (once past prefill)."""
    prefill_steps: int = 1_000
    """Env steps collected before the first gradient step."""
    timeout_env_steps: int = 600
    """How many steps before we rotate games if the agent does not end the game first"""

    # buffer
    buffer_capacity: int = 200_000
    """Max total steps kept in the replay buffer (FIFO eviction of whole episodes)."""

    # sequence sampling
    batch_size: int = 16
    seq_len: int = 16

    # optimization
    lr: float = 3e-4
    grad_clip: float = 100.0
    kl_warmup_steps: int = 0
    """Gradient steps of linear KL-weight warmup (0 -> config value); 0 disables."""

    # logging / qualitative checks (in gradient steps)
    log_every: int = 50
    """Gradient steps between console metric lines / perf scalars."""
    qualitative_every: int = 500
    """Gradient steps between recon-vs-real sample images."""
    imagine_every: int = 1_000
    """Gradient steps between imagination-vs-real sample images."""
    imagine_horizon: int = 8
    """Imagined steps per imagination check; needs seq_len >= horizon + 1."""

    # checkpointing (in env steps; saves model+optimizer+counters+buffer)
    checkpoint_every: int = 5_000
    output_dir: str = "runs/world_model"
    resume: bool = True
    """Pick up from output_dir/latest.pt (and buffer.pt) if present."""
    init_from: str = ""
    """Checkpoint path to warm-start *world-model weights only* from, used
    only when `resume` finds nothing: fresh optimizer/counters/buffer, so
    it seeds a new run from prior work rather than resuming it."""

    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def format_duration(seconds: float) -> str:
    """Human-readable duration for ETA logging, e.g. '2h15m', '45m30s',
    '38s'. Negative input (a rate spike making the estimate momentarily go
    negative) clamps to 0s rather than printing something nonsensical."""
    total_seconds = int(max(0.0, seconds))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days > 0:
        return f"{days}d{hours}h"
    if hours > 0:
        return f"{hours}h{minutes}m"
    if minutes > 0:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


class Trainer:
    def __init__(self, config: TrainerConfig | None = None):
        self.config = config or TrainerConfig()
        c = self.config
        random.seed(c.seed)
        torch.manual_seed(c.seed)

        self.out_dir = Path(c.output_dir)
        self.env = Env()
        self._game_cycle: list[str] = []
        self._game_idx = 0
        self._current_game: str | None = None

        # Resume decides the architecture: a checkpoint's saved ThumperConfig
        # wins over c.thumper, so the rebuilt module matches its weights.
        payload = None
        if c.resume and self.checkpoint_path().exists():
            payload = torch.load(self.checkpoint_path(), map_location=c.device, weights_only=False)
            c.thumper = payload["config"]

        self.thumper = Thumper(c.thumper).to(c.device)
        self.optimizer = torch.optim.Adam(self.thumper.world_model.parameters(), lr=c.lr)
        self.buffer = ReplayBuffer(
            capacity=c.buffer_capacity,
            frame_stack=c.thumper.world_model.frame_stack,
            internal_state_dim=c.thumper.world_model.internal_state_dim,
        )
        self.env_steps = 0
        self.grad_steps = 0
        self.writer: SummaryWriter | None = None
        self._last_batch: dict[str, torch.Tensor] | None = None

        if payload is not None:
            self.thumper.load_state_dict(payload["state_dict"])
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self.env_steps = payload["env_steps"]
            self.grad_steps = payload["grad_steps"]
            if self.buffer_path().exists():
                self.buffer = ReplayBuffer.load(
                    self.buffer_path(),
                    capacity=c.buffer_capacity,
                    frame_stack=c.thumper.world_model.frame_stack,
                    internal_state_dim=c.thumper.world_model.internal_state_dim,
                )
            print(
                f"Resumed from {self.checkpoint_path()}: env_steps={self.env_steps}, "
                f"grad_steps={self.grad_steps}, buffer={self.buffer.total_steps} steps."
            )
        elif c.init_from:
            # Warm start: only the world model's weights -- no optimizer
            # state, no counters, no buffer. A fresh run seeded from prior
            # work, not a resume (a crashed run resumes its own progress
            # above, never re-warm-starts).
            init_payload = torch.load(c.init_from, map_location=c.device, weights_only=False)
            prefix = "world_model."
            world_model_state = {
                k.removeprefix(prefix): v
                for k, v in init_payload["state_dict"].items()
                if k.startswith(prefix)
            }
            self.thumper.world_model.load_state_dict(world_model_state)
            print(f"Initialized world model weights from {c.init_from} (fresh optimizer/counters/buffer).")

    # --- game round-robin -------------------------------------------------

    def _games(self) -> list[str]:
        if not self._game_cycle:
            self._game_cycle = self.env.games()
            if not self._game_cycle:
                raise RuntimeError("no games found in environment_files/")
        return self._game_cycle

    def game_ids(self) -> dict[str, int]:
        """Stable small-int id per game (enumeration order), stored on each
        buffer episode to key per-game telemetry."""
        return {game: i for i, game in enumerate(self._games())}

    def _next_game(self) -> str:
        games = self._games()
        game = games[self._game_idx % len(games)]
        self._game_idx += 1
        return game

    def _random_action(self, available_actions: list[int]) -> tuple[int, tuple[int, int]]:
        legal = [a for a in available_actions if a < NUM_ACTION_TYPES] or [RESET]
        action_type = random.choice(legal)
        grid_size = self.config.thumper.world_model.grid_size
        coords = (random.randrange(grid_size), random.randrange(grid_size)) if action_type == ACTION6 else (0, 0)
        return action_type, coords

    def _begin_episode(self) -> tuple[StepResult, Episode]:
        """Reset into the next game and store its first frame. The first
        step's RESET is a placeholder never consumed by forward_sequence /
        compute_losses (is_first marks it), and storing it doesn't count as
        an env step."""
        game = self._next_game()
        self._current_game = game
        result = self.env.reset(game)
        episode = self.buffer.start_episode(game_id=self.game_ids()[game])
        self.buffer.add_step(
            episode, result.frame, RESET, (0, 0), 0.0, False, result.levels_completed / MAX_SCORE
        )
        return result, episode

    # --- training ----------------------------------------------------------

    def kl_weight(self) -> float:
        """Configured KL weight, linearly warmed up over kl_warmup_steps."""
        base = self.config.thumper.world_model.kl_weight
        if self.config.kl_warmup_steps <= 0:
            return base
        return base * min(1.0, self.grad_steps / self.config.kl_warmup_steps)

    def train_step(self) -> dict[str, float]:
        c = self.config
        batch = self.buffer.sample(c.batch_size, c.seq_len)
        batch = {k: v.to(c.device) for k, v in batch.items()}
        self._last_batch = batch  # reused by the qualitative checks

        wm = self.thumper.world_model
        outputs = wm.forward_sequence(
            batch["observations"],
            batch["action_types"],
            batch["coords"],
            batch["is_first"],
            rewards=batch["rewards"],
        )
        losses = wm.compute_losses(
            outputs, batch["observations"], batch=batch, kl_weight=self.kl_weight()
        )

        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        grad_norm = nn.utils.clip_grad_norm_(wm.parameters(), c.grad_clip)
        self.optimizer.step()
        self.grad_steps += 1

        metrics = {k: v.item() for k, v in losses.items()}
        metrics["grad_norm"] = grad_norm.item()
        self._log_train_scalars(metrics, outputs, batch)
        return metrics

    def _log_train_scalars(
        self, metrics: dict[str, float], outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> None:
        if self.writer is None:
            return
        step = self.grad_steps
        w = self.writer
        w.add_scalar("loss/total", metrics["total_loss"], step)
        w.add_scalar("loss/recon", metrics["recon_loss"], step)
        w.add_scalar("loss/kl", metrics["kl_loss"], step)
        w.add_scalar("loss/kl_raw", metrics["kl_raw"], step)
        w.add_scalar("loss/continue", metrics["continue_loss"], step)
        w.add_scalar("loss/internal_state", metrics["internal_state_loss"], step)
        if "reward_loss" in metrics:
            w.add_scalar("loss/reward", metrics["reward_loss"], step)
        if "ensemble_loss" in metrics:
            w.add_scalar("wm/ensemble_loss", metrics["ensemble_loss"], step)
        w.add_scalar("train/kl_weight", self.kl_weight(), step)
        w.add_scalar("train/grad_norm", metrics["grad_norm"], step)
        w.add_scalar("online/env_steps", self.env_steps, step)
        w.add_scalar("online/buffer_size", self.buffer.total_steps, step)

        # Disagreement telemetry: computed no_grad on this batch's states --
        # the eventual exploration currency (Plan2Explore), so watch whether
        # it stays meaningfully above zero and differs across games.
        if outputs["deter"].shape[1] > 1:
            with torch.no_grad():
                disagreement = self.thumper.world_model.disagreement(
                    outputs["deter"][:, :-1],
                    outputs["stoch"][:, :-1],
                    outputs["action_onehot"][:, 1:],
                    outputs["macro_context"][:, 1:],
                )
                flat = disagreement.reshape(-1)
                w.add_scalar("wm/disagreement_mean", flat.mean().item(), step)
                w.add_scalar("wm/disagreement_p90", torch.quantile(flat, 0.9).item(), step)
                id_to_game = {i: g for g, i in self.game_ids().items()}
                flat_games = batch["game_ids"][:, 1:].reshape(-1)
                for gid in torch.unique(flat_games).tolist():
                    name = id_to_game.get(gid, f"unknown_{gid}")
                    w.add_scalar(
                        f"wm/disagreement_by_game/{name}", flat[flat_games == gid].mean().item(), step
                    )

    # --- the loop -----------------------------------------------------------

    def train(self) -> None:
        """Run collection + training until total_env_steps, checkpointing on
        an env-step cadence and saving qualitative samples on grad-step
        cadences."""
        c = self.config
        samples_dir = self.out_dir / "samples"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.out_dir / "tb"))
        print(f"TensorBoard logs: {self.out_dir / 'tb'}  (run: uv run tensorboard --logdir {self.out_dir / 'tb'})")

        try:
            result, episode = self._begin_episode()
            episode_return = 0.0
            episode_len = 0
            last_checkpoint_env_steps = self.env_steps
            env_steps_at_last_grad = self.env_steps
            window_start = time.monotonic()
            window_grad_steps = 0
            window_start_env_steps = self.env_steps
            metrics: dict[str, float] = {}
            env_steps = 0
            while self.env_steps < c.total_env_steps:
                env_steps += 1
                # --- environment: one random action -> one buffer step
                action_type, coords = self._random_action(result.available_actions)
                x, y = (coords if action_type == ACTION6 else (None, None))
                result = self.env.step(action_type, x=x, y=y)
                terminate_due_to_max_steps = env_steps >= c.timeout_env_steps
                terminated = result.done or terminate_due_to_max_steps

                self.buffer.add_step(
                    episode,
                    result.frame,
                    action_type,
                    coords,
                    float(result.reward),
                    terminated,
                    result.levels_completed / MAX_SCORE,
                )

                self.env_steps += 1
                episode_return += float(result.reward)
                episode_len += 1
                if terminated:
                    game = self._current_game
                    self.writer.add_scalar("online/episode_return", episode_return, self.env_steps)
                    self.writer.add_scalar("online/episode_len", episode_len, self.env_steps)
                    self.writer.add_scalar(f"online/episode_return/{game}", episode_return, self.env_steps)
                    self.writer.add_scalar(f"online/episode_len/{game}", episode_len, self.env_steps)
                    self.writer.add_scalar(f"online/win_rate/{game}", float(result.won), self.env_steps)
                    episode_return = 0.0
                    episode_len = 0
                    env_steps = 0

                    result, episode = self._begin_episode()

                # --- training: catch up to the env-steps-per-grad-step ratio
                can_train = self.env_steps >= c.prefill_steps and self.buffer.total_steps > c.seq_len
                if not can_train:
                    # keep the ratio anchor tracking during prefill so the
                    # first trainable step doesn't trigger a catch-up burst
                    env_steps_at_last_grad = self.env_steps
                while can_train and self.env_steps - env_steps_at_last_grad >= c.train_every:
                    env_steps_at_last_grad += c.train_every
                    metrics = self.train_step()
                    window_grad_steps += 1
                    step = self.grad_steps

                    if step % c.log_every == 0 or step == 1:
                        now = time.monotonic()
                        elapsed = max(now - window_start, 1e-9)
                        grad_steps_per_sec = window_grad_steps / elapsed
                        # env-steps/sec over the same window, since
                        # total_env_steps (the loop's stopping condition) is
                        # what the ETA divides against -- the two rates
                        # aren't a fixed ratio (prefill, catch-up bursts).
                        env_steps_per_sec = (self.env_steps - window_start_env_steps) / elapsed
                        window_start = now
                        window_grad_steps = 0
                        window_start_env_steps = self.env_steps
                        self.writer.add_scalar("perf/grad_steps_per_sec", grad_steps_per_sec, step)
                        self.writer.add_scalar("perf/env_steps_per_sec", env_steps_per_sec, step)
                        eta = format_duration(
                            max(c.total_env_steps - self.env_steps, 0) / max(env_steps_per_sec, 1e-9)
                        )
                        message = (
                            f"env {self.env_steps:07d}/{c.total_env_steps} | grad {step:07d} "
                            f"| ETA {eta} "
                            f"| total {metrics['total_loss']:.4f} "
                            f"| recon {metrics['recon_loss']:.4f} "
                            f"| kl_raw {metrics['kl_raw']:.4f} "
                        )
                        if "reward_loss" in metrics:
                            message += f"| reward {metrics['reward_loss']:.4f} "
                        message += (
                            f"| cont {metrics['continue_loss']:.4f} "
                            f"| buffer {self.buffer.total_steps} | {grad_steps_per_sec:.2f} grad steps/s"
                        )
                        print(message)

                    if step % c.qualitative_every == 0 or step == 1:
                        save_recon_check(
                            self.thumper.world_model, self._last_batch, step, samples_dir, writer=self.writer
                        )
                    if step % c.imagine_every == 0 or step == 1:
                        save_imagination_check(
                            self.thumper.world_model, self._last_batch, step, samples_dir,
                            c.imagine_horizon, writer=self.writer,
                        )

                # --- checkpoint on env-step cadence
                if self.env_steps - last_checkpoint_env_steps >= c.checkpoint_every:
                    last_checkpoint_env_steps = self.env_steps
                    self.save_checkpoint()
                    print(
                        f"Checkpointed at env_steps={self.env_steps} "
                        f"(grad_steps={self.grad_steps}, buffer={self.buffer.total_steps})"
                    )

            self.save_checkpoint()
            print(
                f"Done: env_steps={self.env_steps}, grad_steps={self.grad_steps}. "
                f"Final checkpoint at {self.checkpoint_path()}."
            )
        finally:
            self.writer.close()
            self.writer = None

    # --- checkpointing -------------------------------------------------------

    def checkpoint_path(self) -> Path:
        return self.out_dir / "latest.pt"

    def buffer_path(self) -> Path:
        return self.out_dir / "buffer.pt"

    def save_checkpoint(self) -> None:
        """Write latest.pt (Thumper.load-compatible: 'config' + 'state_dict',
        plus optimizer state and counters) and buffer.pt, each to a temp file
        then renamed, so a crash mid-save can't leave a truncated checkpoint
        that blocks the next resume."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.thumper.config,
            "state_dict": self.thumper.state_dict(),
            "trainer_config": self.config,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "env_steps": self.env_steps,
            "grad_steps": self.grad_steps,
        }
        tmp = self.checkpoint_path().with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(self.checkpoint_path())

        buffer_tmp = self.buffer_path().with_suffix(".pt.tmp")
        self.buffer.save(buffer_tmp)
        buffer_tmp.replace(self.buffer_path())
