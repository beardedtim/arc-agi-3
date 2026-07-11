# 0012 — Test-Time Intrinsic Annealing: Exploit What Adaptation Finds

## Overview & Architectural Justification

Run 8 (TRAINING_LOG.md, tickets/0010) validated test-time adaptation on
cd82 — and surfaced its clearest failure mode on r11l. The adaptation
objective is the unmodified training objective: the actor's advantage is
`advantage_ext + intrinsic_scale * advantage_int` with
`intrinsic_scale = 1.0`, constant for the entire 20k-env-step budget
(`training/actor_critic.py:166`, fed from `TrainerConfig.intrinsic_scale`
at `training/trainer.py:465`). On a single novel game with sparse extrinsic
reward, the intrinsic (ensemble-disagreement) stream therefore dominates
the actor's objective from the first grad step to the last — the agent is
paid continuously to explore and approximately never to exploit.

Run 8's evidence, game by game (all numbers in that entry's Findings; TB
under `runs/adapt_v1/<game>/tb`, reports in
`runs/adapt_v1/<game>/adapt_report.json`):

- **r11l — the headline failure.** Adaptation found a real scoring event
  at env step **1,103** (faster than cd82's 6,763) and never scored again
  in the remaining ~19k steps. Worse than merely not improving: the
  adapted policy **lost the frozen policy's transient scoring events**
  (frozen sampled play had reward events at steps 359/501; adapted play
  has zero events in every cell). Mechanism visible in TB:
  `policy/entropy` collapsed ~7.4 → **~0.58** at the tail (vs cd82's
  healthy ~1.5), `policy/value_ext_mean` inflated to ~0.76 anchored by
  that single event, and adapted greedy play switched from the frozen
  policy's 62-step deaths to riding the 600-step cap — 20k steps of
  intrinsic-dominated training taught survival and disagreement-farming,
  not scoring.
- **ft09 — the same signature, different surface.** The adapted policy
  terminates every episode at ~32 steps (frozen greedy varied 36–251) —
  consistent with learning a fast reset to farm the novelty around
  episode starts.
- **cd82 — the success, and why it doesn't contradict this ticket.**
  Adaptation won (greedy 0.00 → 0.80, sampled 0.20 → 1.00) because
  scoring became *recurrent* — once the extrinsic stream carried repeated
  real events, it competed with the intrinsic stream on its own. The
  lever this ticket adds must not break that: cd82 is the pre-registered
  do-no-harm control.

The fix is the one Run 8's pre-registration named in advance ("would
directly motivate down-weighting intrinsic reward at test time"): a
**linear schedule on `intrinsic_scale` over the adaptation budget** —
explore early (r11l's step-1,103 discovery shows full-intrinsic early
exploration works), exploit late (where constant-1.0 currently farms
disagreement instead of consolidating the discovered score). Default off
everywhere; exposed through `adapt.py`; measured against Run 8's existing
reports as the baseline arm.

**Read these first (in this order):**

- `TRAINING_LOG.md` Run 8 Findings — the evidence and the pre-registered
  motivation. Run 7's Findings for what the source checkpoint is.
- `training/actor_critic.py` — `ActorCriticConfig` (lines 87–97,
  `intrinsic_scale`'s docstring: a weight between two *normalized* O(1)
  streams) and `actor_critic_losses` lines 150–167. Note carefully: the
  critic loss (lines 150–153) trains **both** value heads unweighted;
  `cfg.intrinsic_scale` touches only the actor's advantage mix (line 166).
  This ticket preserves that split exactly.
- `training/trainer.py` — `TrainerConfig` around lines 119–124
  (`entropy_scale`/`intrinsic_scale`: trainer-level knobs, deliberately
  *not* part of the checkpoint's saved `ThumperConfig`, so they take
  effect on resume — Run 5's Findings verified this) and
  `policy_train_step` lines 442–493, specifically the `ActorCriticConfig`
  construction at lines 461–466: the single hook point this ticket needs.
- `adapt.py` — `Args`, `adapt_game`'s `TrainerConfig` construction
  (lines 137–148), and the report dict (lines 159–173).
- `tests/test_training.py::TestTrainer` (constructing a small real
  `Trainer` from `small_config()`) and `tests/test_adaptation.py` — the
  two test patterns Step 4 follows. `tests/conftest.py` for the shrunken
  Thumper.
- `tickets/0005` (why the two streams and the normalizers exist) and
  `tickets/0010` (the adaptation harness this modifies).

---

## Design & Core Principles

1. **Anneal the actor's mix only; never touch the critic.** The schedule
   changes what the *actor* is paid for, not what the critics learn: both
   value heads keep training unweighted (`actor_critic_losses` lines
   150–153 are untouched), so the intrinsic value estimates remain
   available (e.g. to `training/planner.py`'s `intrinsic_scale` scoring
   knob) even when the actor has fully handed off to extrinsic. The only
   behavioral change is the blend at line 166, via the `cfg` value passed
   in — `actor_critic_losses` itself needs **zero changes**.

2. **A pure function of counters and config — no new state anywhere.**
   The effective scale is computed fresh each `policy_train_step` from
   `self.env_steps` and two config fields. Nothing new is checkpointed;
   resume is correct by construction (env_steps is already restored).
   Documented consequence, accepted: resuming with a different
   `total_env_steps` moves the schedule (it is a fraction of the
   *configured* budget, not of some frozen original) — same class of
   behavior as `entropy_scale` being changeable on resume, which Run 5
   used deliberately.

3. **Default off, everywhere, bit-for-bit.** The new field is
   `intrinsic_scale_final: float | None = None`; `None` means "constant
   `intrinsic_scale`", reproducing today's behavior exactly — every
   existing command, test, and resume is unaffected. This is a
   *test-time* lever: `train.py`'s main-loop default stays constant-1.0,
   and promoting annealing into main training is its own future decision
   requiring its own run (see Non-goals).

4. **Linear over the whole budget, not event-triggered.** r11l scored at
   step 1,103 under full intrinsic — early exploration is not the
   problem; late-stage intrinsic dominance is. A linear ramp from
   `intrinsic_scale` to `intrinsic_scale_final` across `total_env_steps`
   captures explore-early/exploit-late with **one** new config field and
   no state. Event-triggered variants ("decay on first score") are
   plausibly better and explicitly deferred (Non-goals) — land the
   simple, measurable version first and cite its numbers when proposing
   anything cleverer.

5. **Run 8 is the baseline arm — do not rerun it.** Same checkpoint, same
   protocol, same seed, same budget; only the annealed arm is new, in a
   fresh output dir. For the comparison to stay self-describing, the two
   schedule values must be recorded in `adapt_report.json` (Step 3) — a
   report that doesn't say which arm it is, is not evidence.

6. **Expected secondary mechanism, pre-registered so the run can check
   it:** as the intrinsic advantage shrinks toward `final`, the fixed
   `entropy_scale=1e-3` bonus becomes relatively stronger, so r11l's
   entropy collapse (~0.58) should partially self-correct late in an
   annealed run *without touching `entropy_scale`*. If entropy instead
   pins at max because advantages vanish entirely (a game with no
   extrinsic events and annealed-away intrinsic), that is the
   pre-registered signature for preferring a nonzero `final` (e.g. 0.1)
   — a finding for the run entry, not a knob to pre-tune.

---

## Implementation Tasks

### Step 1: `training/trainer.py` — config field + schedule function

- Add to `TrainerConfig`, directly under `intrinsic_scale` (line ~122):

  ```python
  intrinsic_scale_final: float | None = None
  """If set, the actor's intrinsic_scale anneals linearly from
  `intrinsic_scale` at env step 0 to this value at `total_env_steps`
  (clamped there for any overtime steps). None = constant, today's
  behavior. Trainer-level like entropy_scale (not part of the saved
  ThumperConfig), so it applies on resume; the schedule is a pure
  function of env_steps and config -- nothing new is checkpointed.
  Test-time adaptation lever (tickets/0012); main training keeps the
  constant default."""
  ```

- Add a **module-level pure function** (unit-testable with no Trainer):

  ```python
  def intrinsic_scale_at(
      initial: float, final: float | None, env_steps: int, total_env_steps: int
  ) -> float:
      """tickets/0012: linear anneal of the actor's intrinsic-advantage
      weight over the run's env-step budget; final=None disables."""
      if final is None:
          return initial
      frac = min(1.0, env_steps / max(1, total_env_steps))
      return initial + frac * (final - initial)
  ```

### Step 2: `policy_train_step` wiring + telemetry

- In `policy_train_step` (lines 461–466), replace
  `intrinsic_scale=c.intrinsic_scale` with the scheduled value:

  ```python
  intrinsic_scale=intrinsic_scale_at(
      c.intrinsic_scale, c.intrinsic_scale_final, self.env_steps, c.total_env_steps
  ),
  ```

- Add the effective value to the returned metrics dict (after the
  `losses` items are copied in, line ~490):
  `metrics["intrinsic_scale_effective"] = <the computed value>` —
  compute it once into a local before building `ac_cfg` so the same
  float is used and logged.
- In `_log_policy_scalars`, log it as `policy/intrinsic_scale` following
  the existing scalar-write pattern in that method. This is the scalar
  the pre-registered run reads to confirm the ramp is live.

### Step 3: `adapt.py` — expose the arm and record it

- Add to `Args`:

  ```python
  intrinsic_scale: float = 1.0
  """Actor intrinsic-advantage weight at adaptation start (TrainerConfig
  passthrough)."""
  intrinsic_scale_final: float | None = None
  """tickets/0012: if set, anneal intrinsic_scale linearly to this value
  over the adaptation budget. None = constant (the Run 8 baseline arm)."""
  ```

- Pass both through in `adapt_game`'s `TrainerConfig(...)` construction
  (lines 137–148): `intrinsic_scale=args.intrinsic_scale,
  intrinsic_scale_final=args.intrinsic_scale_final,`.
- Record both in the report dict (lines 159–173), top level, next to
  `budget`/`seed`: `"intrinsic_scale": args.intrinsic_scale,
  "intrinsic_scale_final": args.intrinsic_scale_final,` (Design
  principle 5 — reports self-describe their arm; the Run 8 baseline
  reports simply lack the keys, which itself identifies them).
- To keep the plumbing unit-testable without running a 20k-step
  adaptation, factor the `TrainerConfig(...)` construction out of
  `adapt_game` into a small helper
  `build_trainer_config(checkpoint: str, game: str, args: Args) ->
  TrainerConfig` and call it from `adapt_game` — pure refactor, no
  behavior change, tested in Step 4.

### Step 4: Tests

Fast, CPU-only. Schedule tests are pure-function tests; the wiring test
follows `tests/test_training.py::TestTrainer`'s small-real-Trainer
pattern; the adapt test follows `tests/test_adaptation.py`'s style.

1. `test_intrinsic_scale_constant_when_final_none` — `intrinsic_scale_at`
   returns `initial` at env_steps 0, half, total, and 10× total when
   `final is None`.
2. `test_intrinsic_scale_linear_endpoints_midpoint_clamp` — with
   `initial=1.0, final=0.0, total=20_000`: returns 1.0 at step 0, 0.5 at
   10_000, 0.0 at 20_000, and stays 0.0 at 30_000 (clamped); also one
   *rising* case (`initial=0.0, final=1.0`) to pin the sign, and
   `total_env_steps=0` does not divide by zero (the `max(1, ...)` guard).
3. `test_policy_train_step_uses_scheduled_scale` — build a small `Trainer`
   (the `TestTrainer` fixture pattern) with `intrinsic_scale=1.0,
   intrinsic_scale_final=0.0`; monkeypatch
   `training.trainer.actor_critic_losses` with a wrapper that records the
   `cfg` it receives and then calls the real function (so the step still
   completes). Assert: with `trainer.env_steps = 0` the captured
   `cfg.intrinsic_scale == 1.0`; with `trainer.env_steps =
   cfg.total_env_steps` it is `0.0`; and the returned metrics dict
   contains `intrinsic_scale_effective` matching. Also assert that with
   `intrinsic_scale_final=None` the captured value equals the config
   constant at any env_steps (the default-off guarantee).
4. `test_adapt_trainer_config_carries_annealing_arm` — `Args` with
   `intrinsic_scale=1.0, intrinsic_scale_final=0.0` →
   `build_trainer_config(...)` yields a `TrainerConfig` with both fields
   set; defaults yield `1.0` / `None`.

### Step 5: Docs

- **CLAUDE.md**: extend the `adapt.py` command bullet with the two new
  flags and one clause on what annealing is for (tickets/0012); one
  clause in the Training section where `intrinsic_scale` semantics are
  described (the tickets/0005 mention) noting the optional test-time
  schedule.
- **ARCH.md**: one sentence where the two-stream advantage mix is
  described (§4.6 territory): the mix weight can anneal over a run's
  budget, test-time lever, tickets/0012 pointer.
- Config/field docstrings per Steps 1–3 (already specified inline above).

### Step 6: Run guidance (for the future TRAINING_LOG entry, not executed here)

One new arm against Run 8's existing baseline (Design principle 5) — a
fresh output dir, same source checkpoint, same seed/budget/protocol:

```sh
# The annealed arm: full intrinsic at step 0, zero by the 20k budget.
# runs/adapt_v1 (Run 8, constant 1.0) is the baseline -- do not rerun it.
uv run python adapt.py \
  --checkpoint runs/held_out_v1/latest.pt \
  --games r11l cd82 ft09 \
  --output-dir runs/adapt_v2_anneal \
  --intrinsic-scale-final 0.0
```

Three games, three pre-registered questions (in priority order):

1. **r11l — the headline.** Does the annealed arm convert the (expected
   ~step-1k) early discovery into *retained* scoring — any adapted cell
   > 0.00, and adapted play keeping (not erasing) sampled-mode scoring
   events? Mechanistic checks in TB: `policy/intrinsic_scale` ramping
   1.0 → 0.0; `policy/entropy` tail meaningfully above Run 8's ~0.58
   (Design principle 6's self-correction); `value_ext_mean` not inflating
   past Run 8's ~0.76 on the same single-event diet.
2. **cd82 — do-no-harm control.** Adapted cells at or near Run 8's
   greedy 0.80 / sampled 1.00. A meaningful regression here means late
   exploration was still doing useful work on a game with recurrent
   extrinsic signal — the pre-registered signature for a nonzero
   `--intrinsic-scale-final` (e.g. 0.1) as the follow-up arm, not for
   abandoning the mechanism.
3. **ft09 — the degeneracy probe.** Does the ~32-step fast-reset
   signature disappear by the final eval? (Episode lengths at the 20k
   eval point are the readout; scoring is not expected — no known path.)

Budget: 3 games × 20k env steps + eval cells ≈ Run 8's per-game
wall-clock × 3 (Run 8 ran 5 games overnight, ~5–6h; this is ~3–4h).
sk48/wa30 are omitted deliberately: neither has a scoring path nor a
distinctive pathology, so they'd spend ~2h re-measuring "all zero".

---

## Non-goals

- **No event-triggered or adaptive schedules** (decay-on-first-score,
  disagreement-threshold gates, per-game auto-tuning). Linear-over-budget
  first; cite this ticket's run when proposing anything richer.
- **No change to main training.** `train.py` keeps constant
  `intrinsic_scale=1.0` (default `intrinsic_scale_final=None`); whether
  the *training* loop should anneal is a separate question needing a
  from-scratch comparison run, not a rider on this ticket.
- **No `entropy_scale` retuning in the same change.** Run 8's entropy
  observations are watch-items; changing two exploration knobs at once
  makes the annealing arm unreadable. Design principle 6 pre-registers
  the interaction instead.
- **No `ReturnNormalizer` changes** — the schedule composes with the
  existing normalize-then-mix pipeline untouched.
- **No new checkpoint state** (Design principle 2): the budget-change
  resume jump is documented and accepted, not engineered around.
- **No critic-loss weighting** — both value heads keep training at full
  weight regardless of the actor's mix (Design principle 1).

---

## Verification & Acceptance Criteria

1. **`uv run pytest` fully green**, including all of Step 4, with every
   pre-existing test untouched — the `None` default must be bit-identical
   to today's objective (covered explicitly by test 3's default-off
   assertion).
2. **CLI surfaces render:** `uv run python train.py --help` shows
   `--config.intrinsic-scale-final` and `uv run python adapt.py --help`
   shows `--intrinsic-scale` / `--intrinsic-scale-final` (tyro renders
   the `float | None` field; passing a bare float sets it, omitting it
   leaves `None`).
3. **The ramp is observable end-to-end:** a short smoke adaptation
   (`adapt.py` against `runs/held_out_v1/latest.pt`, one held-out game,
   `--budget 2000 --eval-every 1000 --intrinsic-scale-final 0.0`) writes
   a TB `policy/intrinsic_scale` curve decaying toward 0 across the run's
   grad steps, and its `adapt_report.json` carries
   `"intrinsic_scale": 1.0, "intrinsic_scale_final": 0.0`.
4. **Default-off is inert on the real path:** the same smoke command
   *without* `--intrinsic-scale-final` logs `policy/intrinsic_scale`
   flat at 1.0 and produces a report with `"intrinsic_scale_final": null`.
5. **The pre-registered arm runs as written:** Step 6's command is
   directly launchable, and its reports are comparable cell-for-cell
   against `runs/adapt_v1/{r11l,cd82,ft09}/adapt_report.json` (same
   checkpoint, seed, budget, protocol — only the schedule differs).
