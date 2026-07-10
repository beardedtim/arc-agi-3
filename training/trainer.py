"""
Trainer :: online world-model + actor-critic training loop for Thumper.

Interleaves collection and training: each env step feeds the replay buffer
(uniform-random over available_actions during `prefill_steps`, then the
policy itself, round-robin across every downloaded game), and gradient
steps catch up at a fixed env-steps-per-grad-step ratio (`train_every`).
The buffer only ever stores a real death (`result.done`) as `terminated` --
a step-cap timeout still rotates the episode locally but is truncation, not
termination, so it never reaches the continue head (tickets/0005). Batch
sampling is reward-event-stratified (`reward_window_frac`,
`training/replay_buffer.py`): a target fraction of each batch is drawn from
windows whose loss window contains a nonzero-reward step, since those are
rare (~0.015% of steps in Run 3) and dilute under plain uniform sampling as
the buffer grows.
Each grad step does two things (see tickets/0003):
  1. one WorldModel.compute_losses update (unchanged from tickets/0001);
  2. one actor-critic update trained entirely in imagination
     (`Thumper.dream` + `training/actor_critic.py`): two-stream returns
     (tickets/0005) -- extrinsic (predicted reward) and intrinsic
     (Plan2Explore disagreement) each get their own lambda-return, critic
     head, and `ReturnNormalizer`, so a sparse level completion can't be
     drowned by continuous exploration payment. REINFORCE with a learned
     critic baseline (no backprop through dynamics, since the action space
     is categorical).
The trained policy replaces the random collector after prefill, so
exploration becomes disagreement-seeking instead of uniform.

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
from training.actor_critic import ActorCriticConfig, ReturnNormalizer, actor_critic_losses
from training.evaluate import EvalProtocol, evaluate
from training.online_actor import OnlineActor
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
    train_games: list[str] = field(default_factory=list)
    """Restrict online collection's round-robin to these games; [] -> every
    downloaded game (Env.games()), today's default behavior. The held-out-
    games generalization protocol (tickets/0009) uses this to keep a fixed
    subset of downloaded games out of training entirely, so eval on the rest
    measures zero-shot transfer. `eval_games`/`eval.py` are independent and
    may evaluate any downloaded game regardless of this filter. Invariant:
    a generalization run must train from scratch (no `init_from`) -- every
    checkpoint produced before this filter existed trained on all games, so
    warm-starting from one bakes held-out dynamics into the weights in a way
    this filter cannot detect or undo."""

    # buffer
    buffer_capacity: int = 200_000
    """Max total steps kept in the replay buffer (FIFO eviction of whole episodes)."""

    # sequence sampling
    batch_size: int = 16
    seq_len: int = 16
    burn_in: int = 16
    """Real steps consumed to build each window's frozen macro-context and
    warm the RSSM state (tickets/0004); 0 disables burn-in (macro_context
    stays at its zero initial_state for the whole window -- there's no
    prefix to build it from). The buffer sample length becomes
    burn_in + seq_len."""

    # optimization
    lr: float = 3e-4
    grad_clip: float = 100.0
    kl_warmup_steps: int = 0
    """Gradient steps of linear KL-weight warmup (0 -> config value); 0 disables."""

    # imagination policy training (tickets/0003)
    dream_horizon: int = 15
    """Imagined steps per policy update (Dreamer default); drop toward 8-10
    if dreams destabilize training -- Run 1 validated coherence to ~8 steps."""
    gamma: float = 0.997
    """Discount; episodes run to 600 steps, so 0.99 is too myopic."""
    return_lambda: float = 0.95
    """TD(lambda) mixing."""
    entropy_scale: float = 1e-3
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    intrinsic_scale: float = 1.0
    """Weight on the normalized intrinsic advantage vs the normalized
    extrinsic advantage (tickets/0005) -- see ActorCriticConfig."""
    critic_target_every: int = 100
    """Grad steps between hard target-critic syncs."""
    return_norm_decay: float = 0.99
    reward_window_frac: float = 0.25
    """Target fraction of each sampled batch drawn from windows whose loss
    window (the trailing seq_len steps, after burn-in) contains a
    nonzero-reward step (tickets/0005); 0 disables. Reward events are rare
    (~0.015% of steps in Run 3), so uniform window sampling dilutes them as
    the buffer grows -- this keeps the reward head (and the rest of
    compute_losses) training on scoring transitions at a controlled rate
    independent of buffer size."""

    # logging / qualitative checks (in gradient steps)
    log_every: int = 50
    """Gradient steps between console metric lines / perf scalars."""
    qualitative_every: int = 500
    """Gradient steps between recon-vs-real sample images."""
    imagine_every: int = 1_000
    """Gradient steps between imagination-vs-real sample images."""
    imagine_horizon: int = 8
    """Imagined steps per imagination check; needs seq_len >= horizon + 1."""

    # periodic in-training eval (tickets/0007); off by default -- a full
    # sweep is 25 games x 5 episodes x 2 modes x <=600 steps ~= 150k env
    # interactions, far too heavy for a frequent in-loop cadence. Enable
    # deliberately with a small eval_games subset and/or
    # eval_episodes_per_game=1.
    eval_every: int = 0
    """Env steps between in-training eval sweeps; 0 disables (the default)."""
    eval_games: list[str] = field(default_factory=list)
    """[] -> every downloaded game (Env.games())."""
    eval_episodes_per_game: int = 1

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
            # Guard against silent buffer contamination (tickets/0009): a
            # resumed run's buffer.pt was collected under the checkpoint's
            # train_games, and Episode.game_id indexes that filtered
            # enumeration -- changing the list across a resume would both
            # re-key every stored episode's game_id and (for a held-out
            # protocol) let held-out episodes already in the buffer train
            # the world model. Old checkpoints predating this field resumed
            # under the implicit train_games=[] (all games).
            checkpoint_train_games = getattr(payload.get("trainer_config"), "train_games", [])
            if checkpoint_train_games != c.train_games:
                raise ValueError(
                    f"Resume train_games mismatch: checkpoint {self.checkpoint_path()} was collected "
                    f"with train_games={checkpoint_train_games!r}, but this run requested "
                    f"train_games={c.train_games!r}. Changing the training game list across a resume "
                    f"silently re-keys buffered episodes' game_id and can leak held-out data into "
                    f"training. Start a fresh --config.output-dir instead."
                )

        self.thumper = Thumper(c.thumper).to(c.device)
        self.actor_state = OnlineActor(self.thumper, c.device)
        self.optimizer = torch.optim.Adam(self.thumper.world_model.parameters(), lr=c.lr)
        self.actor_optimizer = torch.optim.Adam(self.thumper.policy.parameters(), lr=c.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.thumper.critic.parameters(), lr=c.critic_lr)
        self.return_normalizer_ext = ReturnNormalizer(decay=c.return_norm_decay)
        self.return_normalizer_int = ReturnNormalizer(decay=c.return_norm_decay)
        self.buffer = ReplayBuffer(
            capacity=c.buffer_capacity,
            frame_stack=c.thumper.world_model.frame_stack,
            internal_state_dim=c.thumper.world_model.internal_state_dim,
            num_action_types=c.thumper.world_model.num_action_types,
        )
        self.env_steps = 0
        self.grad_steps = 0
        self.writer: SummaryWriter | None = None
        self._last_batch: dict[str, torch.Tensor] | None = None
        self._eval_env: Env | None = None  # built lazily on first eval_every hook (own Env instance)

        if payload is not None:
            self.thumper.load_state_dict(payload["state_dict"])
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
            self.return_normalizer_ext.load_state_dict(payload["return_norm_scales"]["ext"])
            self.return_normalizer_int.load_state_dict(payload["return_norm_scales"]["int"])
            self.env_steps = payload["env_steps"]
            self.grad_steps = payload["grad_steps"]
            if self.buffer_path().exists():
                self.buffer = ReplayBuffer.load(
                    self.buffer_path(),
                    capacity=c.buffer_capacity,
                    frame_stack=c.thumper.world_model.frame_stack,
                    internal_state_dim=c.thumper.world_model.internal_state_dim,
                    num_action_types=c.thumper.world_model.num_action_types,
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
            available = self.env.games()
            if not available:
                raise RuntimeError("no games found in environment_files/")
            train_games = self.config.train_games
            if train_games:
                unknown = sorted(set(train_games) - set(available))
                if unknown:
                    raise ValueError(
                        f"train_games contains games not found in environment_files/: {unknown}. "
                        f"Available games: {available}"
                    )
                # preserve Env.games()'s sorted order, not train_games' order
                available = [g for g in available if g in set(train_games)]
            self._game_cycle = available
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

    def _mask_from_available(self, available_actions: list[int]) -> torch.Tensor:
        """API's `available_actions` (int list) -> (num_action_types,) bool mask."""
        num_types = self.config.thumper.world_model.num_action_types
        mask = torch.zeros(num_types, dtype=torch.bool)
        for a in available_actions:
            if a < num_types:
                mask[a] = True
        return mask

    def _begin_episode(self) -> tuple[StepResult, Episode]:
        """Reset into the next game, store its first frame, and reset the
        per-episode latent state the online policy acts from (frame-stack
        deque, RSSM state, macro-context, previous action) -- mirrors
        forward_sequence's is_first convention: no real action produced the
        first observation, so prev_action starts at zero. The first step's
        RESET is a placeholder never consumed by forward_sequence /
        compute_losses (is_first marks it), and storing it doesn't count as
        an env step."""
        game = self._next_game()
        self._current_game = game
        result = self.env.reset(game)
        episode = self.buffer.start_episode(game_id=self.game_ids()[game])
        mask = self._mask_from_available(result.available_actions)
        self.buffer.add_step(
            episode, result.frame, RESET, (0, 0), 0.0, False, result.levels_completed / MAX_SCORE, mask
        )
        self.actor_state.begin_episode(result.frame)
        return result, episode

    def _act(self, available_actions: list[int]) -> tuple[int, tuple[int, int], torch.Tensor]:
        """One online step of the policy -- delegates to the shared
        OnlineActor (training/online_actor.py), the same acting loop the
        evaluation harness uses."""
        return self.actor_state.act(available_actions)

    def _step_latent(self, action_type: int, coords: tuple[int, int], reward: float, frame: torch.Tensor) -> None:
        """Fold a completed real transition into the online latent state --
        delegates to the shared OnlineActor."""
        self.actor_state.observe(action_type, coords, reward, frame)

    @property
    def _frame_stack(self):
        return self.actor_state._frame_stack

    # --- training ----------------------------------------------------------

    def kl_weight(self) -> float:
        """Configured KL weight, linearly warmed up over kl_warmup_steps."""
        base = self.config.thumper.world_model.kl_weight
        if self.config.kl_warmup_steps <= 0:
            return base
        return base * min(1.0, self.grad_steps / self.config.kl_warmup_steps)

    def train_step(self) -> dict[str, float]:
        c = self.config
        batch = self.buffer.sample(
            c.batch_size, c.burn_in + c.seq_len, reward_frac=c.reward_window_frac, loss_offset=c.burn_in
        )
        batch = {k: v.to(c.device) for k, v in batch.items()}
        window_batch = {k: v[:, c.burn_in :] for k, v in batch.items()}
        self._last_batch = batch  # full (burn_in-prefixed) batch, reused by the qualitative checks
        reward_windows_in_batch = int((window_batch["rewards"] != 0).any(dim=1).sum().item())

        wm = self.thumper.world_model
        outputs = wm.forward_sequence(
            batch["observations"],
            batch["action_types"],
            batch["coords"],
            batch["is_first"],
            rewards=batch["rewards"],
            burn_in=c.burn_in,
        )
        losses = wm.compute_losses(
            outputs, window_batch["observations"], batch=window_batch, kl_weight=self.kl_weight()
        )

        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        grad_norm = nn.utils.clip_grad_norm_(wm.parameters(), c.grad_clip)
        self.optimizer.step()
        self.grad_steps += 1

        metrics = {k: v.item() for k, v in losses.items()}
        metrics["grad_norm"] = grad_norm.item()
        metrics["reward_windows_in_batch"] = float(reward_windows_in_batch)
        self._log_train_scalars(metrics, outputs, window_batch)

        policy_metrics = self.policy_train_step(outputs, window_batch)
        metrics.update(policy_metrics)
        self._log_policy_scalars(policy_metrics)
        return metrics

    def policy_train_step(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """One actor-critic update trained entirely in imagination (see
        tickets/0003): dream from every posterior state this grad step's
        world-model pass produced, then update actor/critic on the
        with-grad second pass over that (grad-free) rollout. One policy
        update per world-model update -- train_every scheduling untouched."""
        c = self.config
        thumper = self.thumper
        B, T = outputs["deter"].shape[:2]
        N = B * T

        deter = outputs["deter"].reshape(N, -1).detach()
        stoch = outputs["stoch"].reshape(N, -1).detach()
        macro_context = outputs["macro_context"].reshape(N, -1).detach()
        available_actions = batch["available_actions"].reshape(N, -1)

        dream = thumper.dream(deter, stoch, macro_context, available_actions, horizon=c.dream_horizon)
        ac_cfg = ActorCriticConfig(
            gamma=c.gamma,
            return_lambda=c.return_lambda,
            entropy_scale=c.entropy_scale,
            intrinsic_scale=c.intrinsic_scale,
        )
        losses = actor_critic_losses(
            dream,
            thumper.policy,
            thumper.critic,
            thumper.critic_target,
            self.return_normalizer_ext,
            self.return_normalizer_int,
            ac_cfg,
        )

        self.actor_optimizer.zero_grad()
        losses["actor_loss"].backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(thumper.policy.parameters(), c.grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        losses["critic_loss"].backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(thumper.critic.parameters(), c.grad_clip)
        self.critic_optimizer.step()

        if self.grad_steps % c.critic_target_every == 0:
            thumper.sync_critic_target()

        metrics = {k: v.item() for k, v in losses.items()}
        metrics["grad_norm_actor"] = actor_grad_norm.item()
        metrics["grad_norm_critic"] = critic_grad_norm.item()
        return metrics

    def _log_policy_scalars(self, metrics: dict[str, float]) -> None:
        if self.writer is None:
            return
        step = self.grad_steps
        w = self.writer
        w.add_scalar("policy/actor_loss", metrics["actor_loss"], step)
        w.add_scalar("policy/critic_loss", metrics["critic_loss"], step)
        w.add_scalar("policy/entropy", metrics["entropy"], step)
        w.add_scalar("policy/imagined_return_ext", metrics["imagined_return_ext"], step)
        w.add_scalar("policy/imagined_return_int", metrics["imagined_return_int"], step)
        w.add_scalar("policy/intrinsic_reward_mean", metrics["intrinsic_mean"], step)
        w.add_scalar("policy/extrinsic_reward_mean", metrics["extrinsic_mean"], step)
        w.add_scalar("policy/value_ext_mean", metrics["value_ext_mean"], step)
        w.add_scalar("policy/value_int_mean", metrics["value_int_mean"], step)
        w.add_scalar("policy/dream_score_sum", metrics["dream_score_sum"], step)
        w.add_scalar("policy/return_norm_scale_ext", self.return_normalizer_ext.scale, step)
        w.add_scalar("policy/return_norm_scale_int", self.return_normalizer_int.scale, step)
        w.add_scalar("policy/grad_norm_actor", metrics["grad_norm_actor"], step)
        w.add_scalar("policy/grad_norm_critic", metrics["grad_norm_critic"], step)

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
        w.add_scalar("train/reward_windows_in_batch", metrics["reward_windows_in_batch"], step)
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
            last_eval_env_steps = self.env_steps
            env_steps_at_last_grad = self.env_steps
            window_start = time.monotonic()
            window_grad_steps = 0
            window_start_env_steps = self.env_steps
            metrics: dict[str, float] = {}
            env_steps = 0
            action_type_counts = torch.zeros(c.thumper.world_model.num_action_types)
            while self.env_steps < c.total_env_steps:
                env_steps += 1
                # --- environment: act (random during prefill, policy after)
                # -> one buffer step
                use_policy = self.env_steps >= c.prefill_steps
                if use_policy:
                    action_type, coords, mask = self._act(result.available_actions)
                else:
                    action_type, coords = self._random_action(result.available_actions)
                    mask = self._mask_from_available(result.available_actions)
                action_type_counts[action_type] += 1
                x, y = (coords if action_type == ACTION6 else (None, None))
                result = self.env.step(action_type, x=x, y=y)
                terminate_due_to_max_steps = env_steps >= c.timeout_env_steps
                terminated = result.done or terminate_due_to_max_steps

                # The continue head must only ever see real deaths -- a
                # step-cap timeout is truncation, not termination, so the
                # buffer stores result.done (not `terminated`, which also
                # rotates the episode below). See tickets/0005.
                self.buffer.add_step(
                    episode,
                    result.frame,
                    action_type,
                    coords,
                    float(result.reward),
                    result.done,
                    result.levels_completed / MAX_SCORE,
                    mask,
                )
                self._step_latent(action_type, coords, float(result.reward), result.frame)

                self.env_steps += 1
                episode_return += float(result.reward)
                episode_len += 1
                if self.env_steps % c.log_every == 0:
                    total = action_type_counts.sum().clamp(min=1.0)
                    for a in range(c.thumper.world_model.num_action_types):
                        self.writer.add_scalar(
                            f"online/action_type_frac/{a}",
                            (action_type_counts[a] / total).item(),
                            self.env_steps,
                        )
                    self.writer.add_scalar(
                        "online/macro_context_norm", self.actor_state.macro_context_norm, self.env_steps
                    )
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
                can_train = self.env_steps >= c.prefill_steps and self.buffer.total_steps > c.burn_in + c.seq_len
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
                            self.thumper.world_model, self._last_batch, c.burn_in, step, samples_dir,
                            writer=self.writer,
                        )
                    if step % c.imagine_every == 0 or step == 1:
                        save_imagination_check(
                            self.thumper.world_model, self._last_batch, c.burn_in, step, samples_dir,
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

                # --- optional periodic eval sweep on its own env-step cadence
                if c.eval_every > 0 and self.env_steps - last_eval_env_steps >= c.eval_every:
                    last_eval_env_steps = self.env_steps
                    self._run_eval_hook()

            self.save_checkpoint()
            print(
                f"Done: env_steps={self.env_steps}, grad_steps={self.grad_steps}. "
                f"Final checkpoint at {self.checkpoint_path()}."
            )
        finally:
            self.writer.close()
            self.writer = None

    # --- periodic in-training eval -------------------------------------------

    def _run_eval_hook(self) -> None:
        """Run an eval sweep against the live weights at this env-step count
        and log eval/* scalars. Must not perturb training: uses its own Env
        (the collector's is mid-episode) and its own OnlineActor, and
        save/restores torch/random RNG state around the call so enabling
        eval doesn't fork an otherwise-identical run's trajectory (the
        collector's action sampling and the buffer's random.choice sampling
        both depend on process-global RNG state)."""
        c = self.config
        if self._eval_env is None:
            self._eval_env = Env()

        torch_rng_state = torch.get_rng_state()
        python_rng_state = random.getstate()
        try:
            protocol = EvalProtocol(
                games=c.eval_games or None,
                episodes_per_game=c.eval_episodes_per_game,
                max_steps=c.timeout_env_steps,
            )
            report = evaluate(self.thumper, self._eval_env, protocol)
        finally:
            torch.set_rng_state(torch_rng_state)
            random.setstate(python_rng_state)

        if self.writer is None:
            return
        for mode in protocol.modes:
            stats_by_game = report.per_game(mode)
            for game, stats in stats_by_game.items():
                self.writer.add_scalar(f"eval/{mode}/score/{game}", stats.mean_levels_completed, self.env_steps)
                self.writer.add_scalar(f"eval/{mode}/win/{game}", stats.win_rate, self.env_steps)
                self.writer.add_scalar(f"eval/{mode}/len/{game}", stats.mean_length, self.env_steps)
            self.writer.add_scalar(f"eval/{mode}/games_scored", report.games_scored(mode), self.env_steps)
            self.writer.add_scalar(f"eval/{mode}/games_won", report.games_won(mode), self.env_steps)

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
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "return_norm_scales": {
                "ext": self.return_normalizer_ext.state_dict(),
                "int": self.return_normalizer_int.state_dict(),
            },
            "env_steps": self.env_steps,
            "grad_steps": self.grad_steps,
        }
        tmp = self.checkpoint_path().with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(self.checkpoint_path())

        buffer_tmp = self.buffer_path().with_suffix(".pt.tmp")
        self.buffer.save(buffer_tmp)
        buffer_tmp.replace(self.buffer_path())
