# 0001 — Basic world-model training loop for Thumper

> Implemented July 9, 2026

## Goal

Stand up the first end-to-end training loop: collect experience from the offline
ARC-AGI-3 games with a random exploration policy, store it in a replay buffer, and
train **only the world model** (`Thumper.world_model`) on sampled sequences until its
losses visibly decrease. Policy/actor-critic training is explicitly out of scope
(follow-up ticket); the policy module just sits untrained inside the checkpoint.

Success looks like: `uv run python train.py` runs on CPU or GPU, prints falling loss
curves, writes periodic checkpoints, and can resume from one. A smoke test exercises
one full collect→train→checkpoint iteration with the tiny test config in seconds.

## Why

Thumper already has every modeling piece (`WorldModel.forward_sequence`,
`compute_losses`, the Plan2Explore ensemble) but nothing feeds it data. A trustworthy
world model is the foundation for imagination-based policy training later, and for
generalizing across games it must be trained on _all_ available games from day one.

## Environment facts (verified against the installed `arc_agi` package)

Don't re-derive these; they were probed live:

- `Arcade(operation_mode=OperationMode.OFFLINE)`; `arcade.get_environments()` returns
  `EnvironmentInfo` objects whose `game_id` looks like `"ls20-9607627b"`, **but
  `arcade.make()` wants the short name** (`"ls20"`) — strip the `-<hash>` suffix (or
  use `local_dir`'s first path component). Passing the full id logs an error and
  returns `None`.
- `env = arcade.make("ls20")` → `LocalEnvironmentWrapper` with `.step(action)` and
  `.reset()`. `step` takes an `arcengine.GameAction` (RESET=0, ACTION1..7; ACTION6 is
  the click and takes `data={"x": .., "y": ..}` — verify the exact kwarg when
  implementing).
- `step` returns `FrameDataRaw` with:
  - `.frame`: nested list, shape `(1, 64, 64)`, integer symbols 0–15 → squeeze the
    leading dim, `torch.tensor(..., dtype=torch.long)`.
  - `.state`: `GameState` enum (`NOT_FINISHED`, plus win/game-over states — inspect
    the enum). Use it for episode termination and the continue-head target.
  - `.levels_completed`: int — the score signal. Reward per step =
    `levels_completed` delta (sparse; almost always 0 under random play, that's fine).
  - `.available_actions`: list of ints (e.g. `[1, 2, 3, 4]`) — mask random action
    sampling with it, and store it if convenient for the later policy ticket.
- Currently only `ls20` is downloaded into `environment_files/`, but write the loop
  against `get_environments()` so new games are picked up automatically.
- Note: the engine exposes ACTION7 but the model's action space
  (`model/actions.py`, `NUM_ACTION_TYPES = 7`) covers RESET + ACTION1–6 only. For
  this ticket, simply never sample ACTION7. Leave a comment; reconciling the action
  space is not this ticket.

## Model API you build on (read these first)

- `model/world_model.py`: `preprocess` (pads/crops raw grids), `encode_actions`
  (type one-hot ++ click x/y one-hots, coords zeroed unless ACTION6),
  `forward_sequence(obs, actions)` → posterior/prior states + head outputs,
  `compute_losses(...)` → dict of losses (reconstruction, KL, reward, continue,
  internal-state, ensemble). Check its exact signature for what targets it expects —
  the replay buffer's sample format should be shaped by `compute_losses`' needs, not
  the other way around.
- `model/vision.py`: frames are stacked `K = frame_stack` (default 4) — the buffer
  must store enough history to build `(B, T, K, H, W)` frame-stack windows, or store
  raw frames and assemble stacks at sample time (preferred: cheaper memory).
- `model/thumper.py`: `Thumper.save/load` already checkpoint weights + config; the
  training loop only needs to add optimizer state + step counters alongside (either
  extend the checkpoint dict or save a sibling file — keep `Thumper.load`
  compatibility).

## Deliverables

1. **`env/env.py` — grow the wrapper.** It's currently a 4-line stub. Give it:
   game enumeration (short names), `reset(game) -> obs`, a
   `step(action_type, x=None, y=None)` that speaks tensors/ints outward and
   `GameAction`/data-dicts inward, and reward/done/available-actions extraction per
   the facts above. Keep it thin — no training logic in here.

2. **`training/replay_buffer.py`** (new package `training/`). Episodic buffer storing
   per-step `(frame, action_type, click_coords, reward, done, first)` per game
   episode. `sample(batch_size, seq_len)` returns the tensors `compute_losses` needs,
   with frame stacks assembled on the fly (repeat the first frame to fill the stack at
   episode starts — mirror however `imagine_from_first_frame`/tests treat boundaries).
   Cap total steps (config, default ~200k) with FIFO eviction of whole episodes.

3. **`training/trainer.py` + `train.py`** (repo-root entry point).
   - Config dataclass (`TrainerConfig`) following the repo's `__post_init__`-derivation
     pattern: steps per collect phase, train ratio, batch size, seq len, lr, device
     (`"cuda" if torch.cuda.is_available() else "cpu"`), checkpoint/log intervals,
     checkpoint dir (gitignored, e.g. `runs/`).
   - Loop: round-robin across all games → collect N env steps with uniform-random
     actions over `available_actions` (uniform-random click coords for ACTION6) →
     M gradient steps on replay samples → repeat. Single Adam over
     `thumper.world_model.parameters()` (not the policy), grad-norm clipping.
   - Metrics: per-loss console lines every log interval (a simple
     `step | loss_total | recon | kl | reward | continue | ensemble` print is enough);
     if you add tensorboard, keep it optional and import-guarded.
   - Checkpointing: every K steps write `Thumper.save` + optimizer/step-count state;
     `train.py --resume <path>` restores all of it.

4. **`tests/test_training.py`.** Use `small_config()` from `tests/conftest.py`.
   At minimum:
   - buffer roundtrip: push a couple of synthetic episodes, sample, assert shapes/dtypes
     and that frame stacks at episode starts don't leak frames from the previous episode;
   - one trainer iteration on synthetic data (no real env needed): losses are finite,
     a world-model parameter actually changed, policy parameters did **not** change;
   - checkpoint→resume restores step count and optimizer state.
     A real-env smoke test (few steps of `ls20`) is welcome but mark it or keep it tiny —
     the suite must stay fast and CPU-only.

5. **Housekeeping**: add the checkpoint/run dir to `.gitignore`; add a short
   "Training" section to `CLAUDE.md` commands (`uv run python train.py`).

## Non-goals

- No policy/actor-critic updates, no imagination rollouts, no intrinsic-reward use
  (the ensemble loss is still trained — it comes free from `compute_losses` — but its
  disagreement signal is not consumed).
- No online/API mode, no hyperparameter search, no multi-GPU.
- No reconciliation of ACTION7 or reward shaping beyond the levels-completed delta.

## Acceptance criteria

- `uv run pytest` passes, including the new tests.
- `uv run python train.py` (fresh clone with `ls20` present) runs a few
  collect/train cycles without error on CPU, prints decreasing reconstruction loss,
  and leaves a loadable checkpoint (`Thumper.load` works on it).
- `--resume` continues from the saved step count.
- Works unchanged when more games appear in `environment_files/`.
