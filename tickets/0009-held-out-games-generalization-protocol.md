# 0009 — Held-Out-Games Protocol: Measure Generalization, the Actual Competition

## Overview & Architectural Justification

Every run so far trains on all 25 downloaded games and evaluates on the same
25. That measures "can Thumper learn the games it practices on" — useful for
debugging the learning machinery (Runs 1–6), but it is not the competition.
ARC-AGI-3 hands the agent games it has never seen; the number that predicts
competition performance is score/win-rate on games that contributed **zero**
training data. We have no such number, and nothing currently prevents us
from optimizing the wrong thing: every architecture choice justified by
"generalize across games" (the TaskEncoder macro-context above all —
tickets/0002's entire reason to exist) is untested on its actual claim. A
macro-context that merely *memorizes* 25 game identities and one that
*infers rules from transitions* look identical on training games; only a
held-out game separates them.

tickets/0007 built the measurement half deliberately game-list-agnostic so
this ticket could exist without touching it (its stated non-goal: "the
held-out *protocol* is a follow-up"). What's missing is the **training**
half: the trainer cannot currently be told to collect from a subset of
downloaded games — `Trainer._games()` takes whatever `Env.games()` reports.
This ticket adds that control, defines the fixed split, and hardens the two
contamination paths that would silently invalidate the measurement (buffer
resume and checkpoint warm-start).

This ticket is protocol + plumbing only. The generalization *run* (Run 7 or
later, depending on Run 6's adjudication) gets its own TRAINING_LOG entry
per the conventions; a command sketch and pre-registration guidance are
provided in Step 6 for that entry to build on.

**Read these first:**

- `training/trainer.py` — `_games()`, `game_ids()`, `_next_game()`: the
  round-robin this ticket filters; the resume path in `__init__` (the
  checkpoint's `trainer_config` is already saved in the payload — the
  resume-validation hook this ticket needs is one comparison away).
- `training/evaluate.py` — `EvalProtocol.games`; already takes an arbitrary
  list. **No changes needed there**; that's the point of 0007's design.
- `training/replay_buffer.py` — `Episode.game_id` (an index into the
  trainer's game enumeration, "stable within a run lineage") — why changing
  the game list across a resume silently corrupts per-game telemetry, and
  why Step 2 validates it.
- TRAINING_LOG.md Run 6's Findings / tickets/0008 Step 4 — the eval tables
  whose "bottleneck is generalization" reading triggers this protocol.

---

## Design & Core Principles

1. **The split is data, not code.** The trainer takes an arbitrary
   `train_games` list (default `[]` → all downloaded, today's behavior
   unchanged); the *particular* split lives in the run entry's
   pre-registered command, per 0007's "don't hardcode the game list"
   rule. The split defined in principle 2 is this ticket's pre-registered
   recommendation, not a constant in the codebase.
2. **The recommended split — 20 train / 5 held out, fixed:**

   ```
   held_out = ["cd82", "r11l", "ft09", "sk48", "wa30"]
   ```

   Rationale, pre-registered so it can't drift: `cd82` and `r11l` are two
   of the four games with *demonstrated* score reachability (Run 3 —
   scoreable but never trained-to-mastery, so held-out scoring on them is
   achievable, making the held-out eval informative rather than
   vacuously zero). The other two demonstrated scorers stay in training on
   purpose: `lp85` keeps the extrinsic stream a live training signal, and
   `sp80` is the only game that terminates naturally under random play —
   the continue head's main source of real terminals must not leave the
   training set. `ft09`/`sk48`/`wa30` are arbitrary-but-fixed picks with no
   known scoring history — they measure the harder zero-shot case, and
   holding them out costs training nothing we know how to measure. 20/5
   keeps 80% of the (already thin) per-game step budget in training.
3. **Contamination fails loudly, not silently.** Two paths can quietly turn
   a "held-out" eval into a lie:
   - **Buffer resume:** a resumed run whose `buffer.pt` contains held-out
     episodes trains the world model on held-out data. Since `game_id` is
     an index into the *filtered* enumeration, a changed `train_games`
     across resume also silently re-keys every stored episode. Remedy: on
     resume, compare the checkpoint's saved `trainer_config.train_games`
     with the current config and **refuse to run** on mismatch (error, not
     warning — the operator must consciously start a fresh dir).
   - **Checkpoint warm-start:** `--config.init-from` any checkpoint whose
     world model trained on all 25 games (every existing run) bakes
     held-out dynamics into the weights. This cannot be detected from the
     weights, so it is a *protocol invariant, enforced procedurally*: the
     generalization run trains **from scratch** (no `init-from`), stated in
     Step 6's run guidance and in the config field's docstring.
4. **Zero-shot only, here.** This protocol measures what the frozen agent
   does on a never-seen game via the eval harness (macro-context adaptation
   is the only test-time mechanism in play). Measuring *test-time learning*
   — resuming online training on a held-out game and counting actions to
   first score, the closest analog of the real competition loop — is the
   natural follow-up ticket once a zero-shot baseline number exists to
   compare against (see Non-goals).

---

## Implementation Tasks

### Step 1: `training/trainer.py` — the `train_games` filter

- `TrainerConfig` gains `train_games: list[str] = field(default_factory=list)`
  — docstring: `[]` → every downloaded game (current behavior); a nonempty
  list restricts **collection** to those games (eval via `eval_games` /
  `eval.py` is independent and may name any downloaded game). Note the
  from-scratch invariant from Design principle 3 in the docstring.
- `_games()`: filter `self.env.games()` to `train_games` when nonempty,
  preserving the sorted order `Env.games()` establishes. Unknown names
  (typos, not-downloaded games) raise `ValueError` listing what *is*
  available — a typo silently shrinking the training set must be
  impossible.
- `game_ids()` continues to enumerate the filtered list — ids stay
  "stable within a run lineage" exactly because Step 2 forbids the lineage
  from changing the list.

### Step 2: resume validation — the buffer-contamination guard

- In the resume branch of `Trainer.__init__` (where `payload` is loaded):
  compare `payload["trainer_config"].train_games` (treat a missing
  attribute on old checkpoints as `[]`) against `c.train_games`; on
  mismatch raise with a message stating both lists and the remedy (fresh
  `--config.output-dir`). This simultaneously guards the game-id re-keying
  and the held-out-episodes-in-buffer paths — a buffer collected under the
  same `train_games` cannot contain held-out episodes.

### Step 3: Tests (`tests/`)

Fast, CPU-only, house conventions (fake/stub env where an `Env` would be
touched — follow test_evaluate.py's `_FakeEnv` pattern):

1. **Filter respected:** a trainer with `train_games=["g2", "g1"]` against
   a fake env reporting `["g1", "g2", "g3"]` round-robins only g1/g2
   (sorted), and every buffer episode's `game_id` indexes the filtered
   list.
2. **Typo fails loudly:** `train_games=["g1", "nope"]` raises `ValueError`
   naming the available games.
3. **Default unchanged:** `train_games=[]` reproduces today's behavior
   (all games; existing trainer tests pass untouched).
4. **Resume guard:** checkpoint saved with `train_games=["g1"]`, resumed
   with `["g1", "g2"]` → raises; resumed with `["g1"]` → loads. An old-style
   payload whose `trainer_config` lacks the field resumes only against
   `train_games=[]`.
5. **Eval independence:** with `train_games=["g1"]` and
   `eval_games=["g2"]`, the eval hook evaluates g2 while the collector
   never touches it (buffer contains no g2 episodes).

### Step 4: Docs

- CLAUDE.md Training section: one clause on `--config.train-games` (the
  held-out split lever, tickets/0009) and the resume guard.
- TrainerConfig docstrings per Step 1.

### Step 5: `eval.py` / `training/evaluate.py`

- **No code changes** (verify this stays true — it is 0007's acceptance
  that this ticket rides on). The final sweep just runs the default
  all-games protocol; the reader partitions rows by the split. If that
  proves annoying in practice, a `--held-out` label column is a follow-up
  nicety, not this ticket.

### Step 6: Run guidance (for the future run entry, not executed here)

The generalization run's TRAINING_LOG entry should pre-register at least:

```sh
# From scratch (no --config.init-from: every existing checkpoint trained on
# all 25 games -- Design principle 3), fresh output dir, 20-game training
# round-robin, periodic zero-shot eval on the 5 held-out games.
uv run python train.py \
  --config.output-dir runs/held_out_v1 \
  --config.train-games ar25 bp35 cn04 dc22 g50t ka59 lf52 lp85 ls20 m0r0 re86 s5i5 sb26 sc25 sp80 su15 tn36 tr87 tu93 vc33 \
  --config.eval-every 10000 \
  --config.eval-games cd82 r11l ft09 sk48 wa30 \
  --config.eval-episodes-per-game 1
```

plus a final full 25-game `eval.py` sweep, read as three numbers: train-set
score (did restricting to 20 games cost learning?), held-out score on
cd82/r11l (zero-shot transfer where scoring is known reachable — the
headline), and held-out score on the three unknowns. Pre-register the
honest baseline comparison: held-out performance vs what *random play*
achieves on those games (Run 1 data exists), since "beats random zero-shot"
is the first defensible claim of generalization. Budget note: from-scratch
at Run 3+ throughput (~5.5 env steps/s) makes 100k steps ~5h; the entry
should decide its budget knowing 20-way round-robin gives ~5k steps/game.

---

## Non-goals

- **No test-time-learning protocol** (resume online training on a held-out
  game, count actions to first score). It is the truest competition analog,
  but it needs this ticket's zero-shot number as its baseline, and its own
  design decisions (what unfreezes, what budget, does the buffer reset).
  Write it as 0010+ once a zero-shot table exists.
- **No split-search or game-difficulty analysis.** The split is fixed by
  the rationale above; optimizing which games to hold out against measured
  outcomes would be selecting the test set — exactly the drift
  pre-registration exists to prevent.
- **No multi-seed / cross-validation folds.** One fixed split first; fold
  rotation is cheap to add later precisely because the split is a CLI list,
  not code.
- **No eval-harness changes** (Step 5).

---

## Verification & Acceptance Criteria

1. **`uv run pytest` fully green**, including all of Step 3; pre-existing
   trainer/eval tests untouched by the filter (`train_games=[]` is
   bit-identical behavior).
2. **A smoke run collects only train games:** a short real run (a few
   hundred env steps, tiny budget) with a 2–3 game `train_games` list
   produces a buffer whose episodes' game ids map only onto those games,
   and `online/*` scalars mention no others.
3. **The resume guard fires in anger:** resuming that smoke run with a
   different `train_games` list errors with the pre-registered message;
   resuming with the same list continues cleanly.
4. **`eval.py` unchanged** still sweeps all 25 downloaded games against any
   checkpoint (spot-check against the smoke checkpoint).
5. **The deliverable question is answerable on paper:** from this ticket +
   Step 6's sketch, the future run entry can be written without any further
   protocol decisions — split, invariants, commands, and the three-number
   reading are all fixed here.
