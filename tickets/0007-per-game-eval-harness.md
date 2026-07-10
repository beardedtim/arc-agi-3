# 0007 — Per-Game Evaluation Harness: Make "Can Thumper Play This Game?" Measurable

## Overview & Architectural Justification

Every scalar we currently watch is a *training-time side effect*. The
`online/*` family logs whatever episodes the collector happens to play —
under a stochastic, exploration-biased policy, round-robin across games, at
whatever cadence the loop reaches each game. That was fine while the
questions were "is the world model learning" (Runs 1–3) and "can the score
signal reach the actor at all" (Runs 4–6). The next questions are
behavioral, and the current telemetry structurally cannot answer them:

1. **"Which games can Thumper play at all?"** has no answer today. There is
   no table of per-game score/win-rate under the policy's best behavior.
   `online/episode_return/<game>` mixes exploration sampling into every
   number and gives each game a data point only when the rotation lands on
   it — a 25-game round-robin at 600-step caps means each game reports
   roughly once per 15k env steps.
2. **Run adjudication is anecdote-driven.** Run 6's headline behavioral
   criterion is "scoring recurs on lp85/sp80/cd82 more than incidentally" —
   currently judged by squinting at sparse `online/*` events. Every future
   ticket (credit assignment, exploration shaping, horizon changes) needs a
   before/after comparison on a *fixed, repeatable* protocol, or we can't
   tell a real improvement from collection noise.
3. **Generalization is the actual competition** and we have zero
   measurement machinery for it. A held-out-games eval (train on N games,
   eval on the rest) is the closest offline proxy to ARC-AGI-3's unseen
   games. This ticket builds the harness; the held-out *protocol* (which
   games, what training changes) is a follow-up, but the harness must not
   bake in "eval games == training games".

This ticket adds a **deterministic-protocol evaluation harness**: roll the
current policy (greedy and sampled) for a fixed number of episodes on every
downloaded game, report per-game score / win rate / episode length /
steps-to-first-score, and expose it three ways — a standalone
`uv run python eval.py` CLI against any checkpoint, an optional periodic
in-training eval on its own `Env`, and TensorBoard `eval/*` scalars so run
adjudication reads off a chart instead of anecdotes.

**Read these first** (the ticket references their exact APIs):

- `training/trainer.py` — `_begin_episode` / `_act` / `_step_latent`: the
  online acting loop (frame-stack deque, RSSM `observe_step`,
  TaskEncoder update, action masking) this ticket factors out for reuse.
  Note the per-episode latent state currently lives as `Trainer._deter`
  etc. — private attributes an eval harness must not reach into.
- `env/env.py` — `Env.games()`, `reset(game)`, `step(...)`, `StepResult`
  (`won`, `levels_completed`, `done`). One game at a time per `Env`
  instance; a mid-training eval therefore needs its **own** `Env` so the
  collector's in-flight episode is untouched.
- `model/thumper.py::act` / `model/policy.py::act` — `greedy: bool` already
  exists (argmax over type and pointer heads); no model change needed.
- `model/actions.py` — `RESET`, `ACTION6`, `NUM_ACTION_TYPES`.
- tickets/0005 §"Verification" item 4 — "episode returns on cd82/r11l/sp80
  becoming _repeatable_ rather than one-offs" is exactly the question this
  harness makes answerable.
- TRAINING_LOG.md Run 6's "What to Look for" — the behavioral criterion
  this harness will adjudicate.

---

## Design & Core Principles

1. **Eval is a fixed protocol, not a peek at training.** An evaluation is
   defined by: checkpoint, game list, episodes per game, step cap, action
   mode (greedy / sampled), and seed. Same inputs → comparable outputs
   across runs and tickets. Defaults: every downloaded game, 5 episodes per
   game, the trainer's 600-step cap, both action modes, seed 0.
2. **One acting loop, shared by collector and evaluator.** The
   frame-stack / RSSM / macro-context bookkeeping in `Trainer._begin_episode`
   / `_act` / `_step_latent` is subtle (is_first conventions, the
   TaskEncoder-steps-online rule from tickets/0003) and must not be
   duplicated — a divergence between the two copies would silently make
   eval measure a different agent than the one being trained. Factor it
   into an `OnlineActor` class in a new `training/online_actor.py`; the
   Trainer becomes its first consumer, the evaluator its second.
3. **Eval never perturbs training state.** The in-training hook constructs
   its own `Env` (design fact: `Env` holds one live game), runs under
   `torch.no_grad()` + `thumper.eval()` (restoring `.train()` after), uses
   its own `OnlineActor` instance, and writes nothing to the replay buffer.
   Eval episodes are measurement, not experience — feeding greedy episodes
   back into the buffer would change the training distribution, which is a
   deliberate non-goal (see Non-goals).
4. **Report the metrics decisions actually need.** Per game × action mode:
   mean/max `levels_completed` at episode end, win rate, mean episode
   length, and mean steps-to-first-score (∞/absent when no score) — the
   last one is the credit-assignment diagnostic (is the first score deeper
   than `dream_horizon` can see?). Plus the aggregate: number of games with
   any score, number with any win. Greedy answers "what has the policy
   committed to"; sampled answers "what can it reach with its own entropy".
5. **Three surfaces, one implementation.** A pure
   `evaluate(thumper, env, protocol) -> EvalReport` function; `eval.py`
   (CLI via `tyro`, prints a table + writes `eval_report.json` next to the
   checkpoint); trainer hook `eval_every` (env steps, `0` = disabled,
   **default 0** — a full sweep is 25 games × 5 episodes × 2 modes × ≤600
   steps ≈ 150k env interactions, far too heavy for a frequent in-loop
   cadence; enable deliberately with a small `eval_games` subset and/or
   `eval_episodes_per_game=1`, e.g. the three known scoring games while
   watching Run 6+).

---

## Implementation Tasks

### Step 1: `training/online_actor.py` — factor out the acting loop

- New `OnlineActor` class owning the per-episode latent state currently
  spread across `Trainer._begin_episode` / `_act` / `_step_latent`:
  - `begin_episode(first_frame: Tensor) -> None` — reset frame-stack deque
    (K copies of the first frame), RSSM `initial_state`, TaskEncoder
    `initial_state`, zero `prev_action_onehot`.
  - `act(available_actions: list[int], greedy: bool = False)
    -> tuple[int, tuple[int, int], Tensor]` — encode stack, `observe_step`,
    mask, `thumper.act(..., greedy=greedy)`; returns
    `(action_type, coords, mask)` exactly as `Trainer._act` does today.
  - `observe(action_type, coords, reward, frame) -> None` — the
    `_step_latent` body (TaskEncoder steps on real transitions; the
    imagination freeze rule applies only to dreamed ones — keep that
    comment).
  - `macro_context_norm` property (the trainer logs it).
- Constructor takes `(thumper, device)`; reads `frame_stack`, dims etc.
  from `thumper.world_model.config` — no config duplication.
- **Behavior-preserving refactor of `trainer.py`:** replace the three
  private methods' bodies with an `OnlineActor` instance
  (`self.actor_state` or similar). The buffer writes, episode rotation, and
  telemetry stay in the Trainer. Existing trainer tests must pass
  unchanged.

### Step 2: `training/evaluate.py` — the protocol and report

- `@dataclass EvalProtocol`: `games: list[str] | None = None` (None → all
  downloaded), `episodes_per_game: int = 5`, `max_steps: int = 600`,
  `modes: tuple[str, ...] = ("greedy", "sampled")`, `seed: int = 0`.
- `@dataclass EvalEpisode`: game, mode, final `levels_completed`, `won`,
  length, `steps_to_first_score: int | None`.
- `@dataclass EvalReport`: list of episodes plus derived
  `per_game(mode) -> dict[str, GameStats]` and aggregate properties
  (`games_scored`, `games_won`); `to_json()` / `summary_table() -> str`
  (plain-text table, one row per game × mode).
- `evaluate(thumper, env, protocol) -> EvalReport`:
  - seed torch/random from `protocol.seed` (sampled mode must be
    reproducible), `thumper.eval()` + `torch.no_grad()` around the whole
    sweep, restore prior training mode after (`torch.is_grad_enabled` is
    untouched by design — use the context manager).
  - per game × mode: fresh `OnlineActor`, `env.reset(game)`,
    `actor.begin_episode(...)`, step until `result.done` or `max_steps`;
    record the `EvalEpisode`. RESET actions chosen by the policy mid-episode
    are legal and just get stepped (they are part of the action space —
    an agent that spams RESET scoring zero is a *finding*, not a bug).
  - No buffer interaction anywhere in this module.

### Step 3: `eval.py` — standalone CLI

- `tyro`-generated CLI mirroring `train.py`'s pattern: `--checkpoint`
  (default `runs/world_model/latest.pt`), plus the `EvalProtocol` fields.
- Loads via `Thumper.load(checkpoint)`, builds an `Env`, runs `evaluate`,
  prints `summary_table()`, writes `eval_report.json` **next to the
  checkpoint** (so a run dir accumulates its own eval history — suffix the
  filename with env-step count when the checkpoint payload carries
  `env_steps`, e.g. `eval_47000.json`, falling back to `eval_report.json`).

### Step 4: trainer hook — optional periodic eval

- `TrainerConfig` gains `eval_every: int = 0` (env steps; 0 disables, and
  stays the default — see Design principle 5's cost math),
  `eval_games: list[str] = []` ([] → all), `eval_episodes_per_game: int = 1`.
- On the same cadence pattern as checkpointing (an
  env-steps-since-last-eval check in the main loop): build the protocol,
  run `evaluate` against a **second `Env`** created lazily on first use,
  then log per game × mode: `eval/<mode>/score/<game>`,
  `eval/<mode>/win/<game>`, `eval/<mode>/len/<game>`, and the aggregates
  `eval/<mode>/games_scored`, `eval/<mode>/games_won` — all against
  `self.env_steps` as the x-axis (these are properties of the agent at an
  env-step count, not of a grad step).
- The collector's in-flight episode, its `OnlineActor`, RNG for action
  sampling, and the buffer must be untouched — eval seeds a **local**
  `torch.Generator`-free scope by saving/restoring
  `torch.get_rng_state()`/`random.getstate()` around the call, so enabling
  eval does not fork an otherwise-identical training run's trajectory.

### Step 5: Docs

- CLAUDE.md: Commands section gains
  `uv run python eval.py --checkpoint runs/<dir>/latest.pt` — the per-game
  eval sweep; Training section notes `eval_every`/`eval_games` and the
  `eval/*` scalar family; Architecture section's trainer bullet mentions
  `training/online_actor.py` as the shared acting loop.
- Module docstrings for the two new modules, following the house style
  (what it is, why it's separate, the invariants it protects).

### Step 6: Tests (`tests/`)

`small_config()` conventions; fast, CPU-only. Eval tests should stub the
env (a tiny fake with scripted `StepResult`s), not touch `arc_agi`:

1. **Refactor equivalence (this ticket's headline invariant):** with fixed
   seeds and a scripted fake env, the trainer's collect loop produces the
   same buffer contents (actions, rewards, masks) before and after the
   `OnlineActor` refactor — pin via a recorded expectation, or by driving
   `OnlineActor` and a copy of the old inline logic side by side and
   asserting identical action streams.
2. **Greedy determinism:** two `evaluate` calls with the same protocol on
   the same fake env produce identical `EvalReport`s (greedy *and* sampled
   modes — sampled via the protocol seed).
3. **Metrics correctness:** a scripted episode with a reward at step k and
   a win at step n yields `steps_to_first_score == k`, `won == True`,
   `levels_completed` from the final `StepResult`; a scoreless episode
   yields `steps_to_first_score is None`.
4. **Isolation:** running the trainer's eval hook leaves
   `buffer.total_steps` unchanged, leaves the collector's `OnlineActor`
   latent state tensors bit-identical, restores `thumper.training` to its
   prior value, and restores torch/random RNG state (assert
   `torch.get_rng_state()` equality around the hook).
5. **Report serialization:** `to_json()` round-trips; `summary_table()`
   contains one row per game × mode.

### Step 7: Training Log

No new training run — this ticket is measurement-only. Instead, run the
CLI against the current Run 6 checkpoint
(`uv run python eval.py --checkpoint runs/two_stream_returns/latest.pt`)
and append the resulting table to Run 6's Findings as its behavioral
adjudication evidence. That table is this ticket's proof-of-value: Run 6's
"scoring recurs more than incidentally" criterion gets judged by it.

---

## Non-goals

- **No eval-episode replay into the buffer** — eval is measurement.
  Feeding greedy trajectories back would change the training distribution
  (that's a possible future *exploitation* ticket, decided on its own
  merits).
- **No held-out-games protocol yet** — the harness takes an arbitrary game
  list precisely so a follow-up ticket can define train/held-out splits
  without touching this code. Don't hardcode "all games" anywhere below the
  CLI default.
- **No pausing/checkpoint-swapping tricks** — in-training eval uses the
  live weights at that env step; evaluating older checkpoints is what the
  standalone CLI is for.
- **No video/frame dumps** — `training/qualitative.py` already covers
  qualitative inspection; add eval-episode rendering only if a future
  debugging need names it.
- **No parallel envs / speedups** — a full sweep is offline and occasional;
  optimize only if the CLI sweep's wall time actually becomes a problem.

---

## Verification & Acceptance Criteria

1. **`uv run pytest` fully green**, including all of Step 6 and the
   pre-existing trainer tests untouched by the refactor (test 1's
   equivalence is the gate for Step 1).
2. **CLI works against a real checkpoint:**
   `uv run python eval.py --checkpoint runs/two_stream_returns/latest.pt`
   completes a full 25-game sweep, prints the table, writes the JSON
   report. Recorded numbers go into Run 6's Findings (Step 7).
3. **Determinism spot-check:** running the CLI twice with the same
   arguments produces byte-identical JSON reports.
4. **In-training hook smoke test:** a short run with
   `--config.eval-every 2000 --config.eval-games '["lp85","sp80","cd82"]'
   --config.eval-episodes-per-game 1` shows `eval/*` scalars in
   TensorBoard, with training throughput (`perf/env_steps_per_sec`) not
   degraded outside eval windows and the run's `loss/*` curves matching an
   eval-disabled run of the same seed for the pre-first-eval segment (RNG
   isolation working).
5. **The deliverable question is answerable:** from the CLI output alone,
   a reader can state which games Thumper scores on, wins, and how deep the
   first score sits relative to `dream_horizon` — no TensorBoard
   archaeology required.
