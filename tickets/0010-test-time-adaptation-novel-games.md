# 0010 — Test-Time Adaptation: Cross-Episode Learning on a Novel Game

## Overview & Architectural Justification

ARC-AGI-3's premise is *learning efficiency on unseen games*: the agent gets
a budget of interactions with a game it has never played and is judged on how
quickly it starts scoring. Today Thumper is frozen at eval time. `evaluate`
(`training/evaluate.py`) builds a **fresh `OnlineActor` per episode**, so
even the macro-context `m` — the one mechanism explicitly designed to infer a
game's rules from experience (tickets/0002) — is thrown away between episodes
of the *same* game. No gradient step ever touches eval experience. Nothing
whatsoever improves across the episode budget. The held-out protocol
(tickets/0009) therefore measures pure zero-shot transfer, which is a floor,
not the benchmark's actual target. This is Missing Features #1 in ARCH.md and
was flagged in the July 2026 tech-lead review as the top un-ticketed
follow-up; tickets/0009's Non-goals explicitly deferred it here ("write it as
0010+ once a zero-shot table exists").

This ticket adds the two cheapest real adaptation mechanisms and the fixed
protocol that measures them against the 0009 zero-shot baseline:

- **Arm A — carried macro-context (no gradients):** reuse one `OnlineActor`
  across all episodes of a game during eval, carrying `m` forward through
  episode resets so the task belief built in episode 1 is available in
  episode 2. Costs nothing but plumbing.
- **Arm B — test-time training:** for each novel game, copy the trained
  checkpoint's **full** weights (world model + policy + critics) into a
  fresh single-game training run with a fixed env-step budget, then measure
  the policy before vs. after. This is literally the existing `Trainer` loop
  pointed at one game — the ticket's job is the safe plumbing (full-weights
  warm start, contamination guards) and the harness that makes the runs
  repeatable and reportable.

Everything reuses existing machinery: `OnlineActor` is already the single
shared acting loop (tickets/0007/0008), `Trainer` already supports
single-game collection via `train_games` (tickets/0009), and `evaluate` is
already the fixed measurement protocol. No new model components, no new
losses.

**Read these first (in this order):**

- `ARCH.md` §3.4 (TaskEncoder / macro-context, the freeze rule, the
  train/act mismatch) and §4.7 (evaluation) — the concepts this ticket
  manipulates.
- `training/online_actor.py` — `begin_episode` (the state you will
  selectively preserve), `act`/`observe` (the fold ordering from
  tickets/0008; **do not touch it**).
- `training/evaluate.py` — `EvalProtocol`, `evaluate`, `_run_episode` (the
  per-episode actor construction you will lift out).
- `training/trainer.py` — `TrainerConfig` (`train_games`, `init_from`,
  `resume`), `Trainer.__init__`'s resume/init_from branches (lines ~202–279;
  the `init_from` branch is what Step 3 extends), `save_checkpoint`'s
  payload keys (`state_dict`, `config`, `trainer_config`, `env_steps`).
- `eval.py` — the CLI pattern (tyro dataclass args, report written next to
  the checkpoint) that `adapt.py` mirrors.
- `tests/test_evaluate.py` — the `_FakeEnv` pattern Step 5's tests follow;
  `tests/conftest.py` — the shrunken Thumper config every test uses.
- tickets/0009 — the held-out split this protocol consumes, and the
  contamination reasoning Step 4's guard extends.

---

## Design & Core Principles

1. **Adaptation is per-game and starts fresh each time.** The competition
   setting is "here is one novel game, here is your budget." So Arm B gives
   *each* held-out game its own adaptation run: its own copy of the source
   checkpoint's weights, its own fresh replay buffer containing only that
   game's experience, its own output dir. Nothing adapted on game A is
   reused for game B. (This also makes catastrophic forgetting a non-issue
   here: we only ever evaluate the adapted weights on the game they adapted
   to.)

2. **Arm A is a measured ablation, not an assumed win.** The TaskEncoder has
   only ever been trained to build `m` within a 16-step burn-in of a single
   episode; carrying `m` across an episode reset is off-distribution (a
   stronger version of the already-logged long-horizon mismatch, ARCH.md
   Missing Features #4). It might help (the belief about the game's rules
   survives the reset) or hurt (the belief encodes stale within-episode
   state). That is exactly why it is a protocol *arm* with its own numbers,
   default **off** everywhere.

3. **Carried `m` scope: within one (game, mode) block only.** During an
   eval sweep with carry enabled, `m` persists across the
   `episodes_per_game` episodes of one game in one mode, and resets at every
   game or mode boundary. Carrying across games would contaminate the very
   belief being measured. During Arm B's *adaptation phase*, `m` handling is
   untouched — the collector already resets it per episode; changing
   collection-time behavior is out of scope (see Non-goals).

4. **Full-weights warm start is a mode of `init_from`, not a new pathway.**
   `TrainerConfig` gains `init_from_full: bool = False`. With `init_from`
   set and `init_from_full=True`, the init branch loads the **entire
   Thumper state dict** (world model, policy, both critics, critic target)
   and adopts the checkpoint's saved `ThumperConfig` (exactly as the resume
   branch does) so the constructed module always matches the weights.
   Optimizers, return normalizers, counters, and buffer stay fresh — the
   existing `init_from` semantics. Fresh return normalizers are deliberate:
   their scales re-estimate within a few hundred grad steps, and the source
   run's scales describe a 20-game mixture, not this one game.

5. **Novelty fails loudly.** Adapting on a game the source checkpoint
   *trained on* is not test-time adaptation — it silently measures continued
   training. `adapt.py` reads the source checkpoint's saved
   `trainer_config.train_games` and **refuses to run** if the target game is
   in it (or if the saved list is `[]`/missing, meaning the checkpoint
   trained on all games). An explicit `--allow-trained-game` flag overrides
   for debugging, printing a loud warning. This mirrors 0009's
   resume-guard philosophy: contamination is an error, not a footnote.

6. **The budget is pre-registered and includes everything.** Default
   adaptation budget: **20,000 env steps per game** (~1 h at Run 3+'s ~5.5
   steps/s), including `prefill_steps = 500` of random warmup (a novel game
   needs *some* data before the first grad step, and in competition random
   actions spend budget too). These are `adapt.py` defaults, overridable,
   but the pre-registered run (Step 7) uses them as-is.

7. **Measurement stays `evaluate`, unchanged in meaning.** Before/after
   comparisons and the in-adaptation learning curve all go through the
   existing harness (same episode counts, step cap, modes, seed). The only
   `evaluate` change is the carry option — everything else about tickets/
   0007's protocol is untouched, so numbers remain comparable to the 0009
   zero-shot table.

---

## Implementation Tasks

### Step 1: `training/online_actor.py` — optional macro-context carry

- Change the signature to
  `begin_episode(self, first_frame, carry_macro_context: bool = False)`.
- Behavior: everything resets exactly as today (frame-stack deque, RSSM
  `initial_state`, zeroed prev-action and pending fold) **except** that when
  `carry_macro_context=True` *and* `self._macro_context is not None` (a
  previous episode ran), `self._macro_context` is left as-is instead of
  being reset to `task_encoder.initial_state`. First-ever call with
  `carry_macro_context=True` therefore behaves identically to `False`.
- Do **not** carry the pending fold, the frame stack, or the RSSM state —
  frames and fast memory are episode-scoped by definition; only the slow
  task belief carries.
- Update the class docstring's "state is reconstructed from a first frame,
  not carried across episodes" sentence to describe the new option and its
  scope (Design principle 3), citing tickets/0010.
- Do not touch `act`/`observe` — the tickets/0008 fold ordering is load-
  bearing and out of scope.

### Step 2: `training/evaluate.py` — carry-aware protocol

- `EvalProtocol` gains `carry_macro_context: bool = False` with a docstring:
  reuse one actor per (game, mode), carrying `m` across that game's
  episodes (tickets/0010 Arm A); default off preserves 0007/0009 semantics.
- Refactor `_run_episode(thumper, env, game, mode, greedy, max_steps,
  device)` → `_run_episode(actor, env, game, mode, greedy, max_steps)`:
  the actor is constructed by the caller and passed in. Inside, replace the
  `OnlineActor(...)` construction with
  `actor.begin_episode(result.frame, carry_macro_context=...)` — but see
  next bullet for where the flag lives.
- In `evaluate`'s loop: construct **one `OnlineActor` per (game, mode)
  pair** (i.e., inside the `for game in games:` loop, before the episode
  loop), and pass `protocol.carry_macro_context` through to each episode's
  `begin_episode`. Since the actor is freshly constructed per (game, mode),
  the carry can never leak across games or modes regardless of the flag —
  the scoping of Design principle 3 is structural, not conditional. With
  the flag off this is behaviorally identical to today (a fresh
  `begin_episode` fully resets state; per the `OnlineActor` docstring, a
  reused instance and a fresh instance are equivalent).
- `eval.py` needs **no changes**: tyro auto-exposes the new field as
  `--protocol.carry-macro-context`. Verify this with `--help`.

### Step 3: `training/trainer.py` — `init_from_full`

- `TrainerConfig` gains `init_from_full: bool = False`, docstring: with
  `init_from`, load the *entire* Thumper state (world model + policy +
  critics + critic target) and adopt the checkpoint's ThumperConfig,
  instead of world-model weights only; optimizers/counters/buffer stay
  fresh either way. Used by test-time adaptation (tickets/0010) so the
  adapted run starts from the full trained agent, not just its dynamics
  model. Setting it without `init_from` is a config error (raise
  `ValueError` in `Trainer.__init__`).
- In `Trainer.__init__`: the init payload must now be loaded **before**
  `Thumper(c.thumper)` is constructed when `init_from_full` is set, because
  the checkpoint's `payload["config"]` must replace `c.thumper` first
  (mirror the resume branch's `c.thumper = payload["config"]`). Concretely:
  in the `payload is None` case with `c.init_from` set, load
  `init_payload = torch.load(c.init_from, ...)` up where the resume check
  happens; if `c.init_from_full`, do `c.thumper = init_payload["config"]`.
  After module construction, the existing `elif c.init_from:` branch
  becomes: full mode → `self.thumper.load_state_dict(init_payload["state_dict"])`;
  world-model-only mode → the existing prefix-filtered load, unchanged.
  Keep the printed message distinct for each mode.
- The resume-vs-init precedence is unchanged: a checkpoint already in
  `output_dir` resumes and `init_from` is ignored, exactly as today.

### Step 4: `adapt.py` — the test-time adaptation harness (new file, repo root)

CLI in the mold of `eval.py` (tyro dataclass `Args`), one process per
invocation adapting one or more games sequentially:

```python
@dataclass
class Args:
    checkpoint: str                      # source (e.g. runs/held_out_v1/latest.pt)
    games: list[str]                     # games to adapt on, one run each
    output_dir: str = "runs/adapt"       # per-game runs land in <output_dir>/<game>/
    budget: int = 20_000                 # env steps per game, prefill included
    prefill_steps: int = 500
    eval_every: int = 5_000              # in-adaptation curve cadence
    episodes_per_game: int = 5           # eval protocol episodes (before/after/curve)
    allow_trained_game: bool = False     # Design principle 5 override
    seed: int = 0
```

Per game, in order:

1. **Novelty guard** (Design principle 5): load the source checkpoint's
   payload, read `getattr(payload.get("trainer_config"), "train_games", [])`;
   if the list is empty or the game is in it, raise `ValueError` naming the
   game and the checkpoint's train list (unless `allow_trained_game`, which
   prints a warning instead).
2. **Pre-adaptation eval** (the zero-shot baseline, measured in-process so
   the report is self-contained): build a fresh `Thumper` from the
   checkpoint (`Thumper.load`), run `evaluate` with
   `EvalProtocol(games=[game], episodes_per_game=args.episodes_per_game,
   seed=args.seed)` — carry off, matching 0009's semantics.
3. **Adapt**: construct
   `TrainerConfig(train_games=[game], init_from=args.checkpoint,
   init_from_full=True, output_dir=f"{args.output_dir}/{game}",
   total_env_steps=args.budget, prefill_steps=args.prefill_steps,
   eval_every=args.eval_every, eval_games=[game],
   eval_episodes_per_game=args.episodes_per_game, seed=args.seed)`
   and call `Trainer(config).train()`. Everything else stays at
   `TrainerConfig` defaults. Note resume works for free: rerunning `adapt.py`
   against the same output dir picks the per-game run up from its own
   `latest.pt` (same `train_games=[game]`, so the 0009 guard passes), and
   `init_from` is correctly ignored on resume.
4. **Post-adaptation eval**: `evaluate` on the adapted trainer's `thumper`,
   same protocol as step 2, run twice — once with `carry_macro_context=False`
   and once `=True` — so the report contains the Arm A ablation on both the
   frozen and the adapted weights (step 2 + these two = three of the four
   cells; the fourth, frozen+carry, add to step 2 the same way: run
   `evaluate` twice there too).
5. **Learning-curve scalars**: `env_steps_to_first_score` — walk the
   trainer's buffer episodes in insertion order, summing episode lengths
   until the first step with nonzero reward; `None` if the run never
   scored. (The buffer holds the entire adaptation run: 20k steps ≪ the
   200k capacity, and it contains only this game.) Implement as a small
   pure helper in `adapt.py` (takes the buffer, returns `int | None`) so
   Step 5 can unit-test it against a hand-built buffer.
6. **Report**: write `<output_dir>/<game>/adapt_report.json` containing:
   the source checkpoint path and its `env_steps`, the game, budget/prefill/
   seed, `env_steps_to_first_score`, and all four eval cells (frozen/adapted
   × carry off/on), each stored via `EvalReport.to_json()`'s episode-list
   payload. Print a compact per-game summary table to stdout: the four
   cells' `mean_levels_completed`/`win_rate` (greedy and sampled) plus
   `env_steps_to_first_score`.

Module docstring: what the harness measures, the four-cell reading, a usage
example, and the pointer to tickets/0010.

### Step 5: Tests (`tests/`)

Fast, CPU-only, shrunken conftest Thumper, `_FakeEnv` pattern from
`test_evaluate.py`. New file `tests/test_adaptation.py` (plus additions to
`test_evaluate.py` where noted):

1. **Carry preserves `m`:** drive an `OnlineActor` through a few
   `act`/`observe` steps (macro-context norm becomes nonzero), call
   `begin_episode(frame, carry_macro_context=True)` → `_macro_context`
   unchanged (same tensor values), while a reference actor with
   `carry_macro_context=False` returns to the zero initial state. Also:
   first-ever `begin_episode` with `carry_macro_context=True` equals the
   zero initial state.
2. **Carry never crosses (game, mode):** `evaluate` with
   `carry_macro_context=True` on a fake env with 2 games × 2 episodes — use
   the Step 2 refactor to assert one actor per (game, mode) (e.g.
   monkeypatch `OnlineActor.__init__`/`begin_episode` to count
   constructions and record carry flags): 2 games × 2 modes = 4
   constructions, `begin_episode` called with carry=True each time.
   (test_evaluate.py.)
3. **Carry off is bit-identical:** `evaluate` with the flag off produces an
   identical `EvalReport` to the pre-refactor behavior — in practice,
   assert two back-to-back sweeps with the same seed and flag off/on-a-
   fresh-model give identical episode lists, and existing evaluate tests
   pass untouched.
4. **`init_from_full` loads everything:** save a trainer-style checkpoint
   payload (`config`, `state_dict`, `trainer_config`, minimal counters)
   from a Thumper with randomly initialized weights; build a `Trainer` with
   `init_from` + `init_from_full=True` (fake/monkeypatched env so no game
   files are needed — follow test_training.py's pattern) and assert a
   policy parameter and a critic parameter match the source exactly;
   with `init_from_full=False` assert the world model matches but the
   policy does **not**. `init_from_full=True` without `init_from` raises.
5. **Novelty guard:** the Step 4.1 guard raises when the game is in the
   checkpoint's `train_games` and when the saved list is empty; passes for
   a genuinely held-out game; `allow_trained_game=True` downgrades to a
   warning. (Factor the guard into a pure function in `adapt.py` so the
   test needs no Trainer.)
6. **`env_steps_to_first_score`:** hand-build a `ReplayBuffer` with three
   episodes (rewards all-zero, all-zero, nonzero at step k) and assert the
   helper returns the correct cumulative index; all-zero buffer → `None`.

### Step 6: Docs

- **ARCH.md**: rewrite Missing Features #1 to reflect what now exists
  (carried `m` at eval, per-game test-time training via `adapt.py`) and
  what still doesn't (episodic memory, decision-time planning — cross-link
  #2); add a short §4.7 paragraph on the adaptation protocol and the
  four-cell reading.
- **CLAUDE.md**: add an `adapt.py` line to Commands (mirroring `eval.py`'s),
  and one clause each in Training for `--config.init-from-full` and
  `--protocol.carry-macro-context`.
- Config/docstrings per Steps 1–4.

### Step 7: Run guidance (for the future TRAINING_LOG entry, not executed here)

The adaptation run rides on a completed 0009 generalization run — its
checkpoint is the source, its zero-shot table is the baseline. Pre-register:

```sh
uv run python adapt.py \
  --checkpoint runs/held_out_v1/latest.pt \
  --games cd82 r11l ft09 sk48 wa30 \
  --output-dir runs/adapt_v1
```

(~5 h: 5 games × 20k steps at ~5.5 steps/s.) Read each game as four cells —
frozen vs adapted × carry off/on — against two pre-registered questions:
**(1)** does 20k steps of test-time training beat zero-shot on the games
where scoring is known reachable (`cd82`, `r11l` — the headline)? **(2)**
does carried `m` help or hurt, and does the answer differ between frozen and
adapted weights? Also report `env_steps_to_first_score` per game — the
closest analog of the competition's actual metric — against the random-play
baseline from Run 1's data. Expectations to pre-register honestly: prefill
is only 500 steps and grad steps start almost immediately on a buffer of
one game, so world-model loss should drop fast; whether the *policy*
converts that into score within 20k steps is genuinely open, and a null
result on (1) is a real finding (it points at decision-time planning,
Missing Features #2, as the necessary next mechanism).

---

## Non-goals

- **No collection-time carry.** The trainer's online collector keeps
  resetting `m` per episode; changing what the TaskEncoder experiences
  during *training* is the Missing-Features-#4 mismatch ticket, not this
  one. Here `m`-carry exists only inside the eval harness.
- **No TaskEncoder retraining for the cross-episode regime.** Arm A
  measures the encoder as-is, off-distribution and honestly labeled as such
  (Design principle 2).
- **No episodic memory / novelty term** (Missing Features #5) and **no
  decision-time planning** (Missing Features #2) — each is its own ticket;
  this one establishes the adaptation numbers they would be measured
  against.
- **No anti-forgetting machinery** (buffer mixing with training games,
  regularization toward source weights). Per-game fresh copies make
  forgetting unobservable here by construction (Design principle 1).
- **No hyperparameter search over the adaptation recipe** (budget, LRs,
  prefill). One pre-registered recipe first; tuning against held-out
  outcomes is test-set selection, exactly what pre-registration prevents.
- **No live-API/competition operation mode** (Missing Features #3). The
  harness runs OFFLINE like everything else; the budget structure merely
  *imitates* competition constraints.

---

## Verification & Acceptance Criteria

1. **`uv run pytest` fully green**, including all of Step 5; every
   pre-existing test untouched (flag-off paths are bit-identical behavior).
2. **Arm A works end to end:** `uv run python eval.py --checkpoint <any
   existing checkpoint> --protocol.games <one game> --protocol.carry-macro-context`
   runs, and the same command without the flag reproduces that checkpoint's
   existing eval numbers exactly (same seed → same episodes).
3. **A smoke adaptation run completes:** `adapt.py` against any existing
   checkpoint with `--allow-trained-game`, one game, `--budget 1500
   --prefill-steps 200 --eval-every 500 --episodes-per-game 1` produces
   `runs/adapt_smoke/<game>/adapt_report.json` with all four eval cells and
   an `env_steps_to_first_score` field, plus the per-game `latest.pt`/
   `buffer.pt`/`tb/` a normal run would have. Rerunning the same command
   resumes rather than restarting.
4. **The novelty guard fires in anger:** the same smoke command *without*
   `--allow-trained-game` refuses with the message naming the game and the
   checkpoint's train list.
5. **`init_from_full` verified on real weights:** a `Trainer` constructed
   with `init_from_full=True` from a real checkpoint produces identical
   greedy eval episodes (before any adaptation steps) to `eval.py` on that
   checkpoint directly — the whole agent came across, not just the world
   model.
6. **The deliverable question is answerable on paper:** from this ticket +
   Step 7, the future adaptation run's TRAINING_LOG entry can be written
   with no further protocol decisions — source checkpoint, games, budget,
   the four-cell reading, and both pre-registered questions are fixed here.
