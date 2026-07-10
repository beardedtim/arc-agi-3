# 0006 — Absorbing-Score Extrinsic Returns: Stop Reward Farming in Imagination

## Overview & Architectural Justification

Run 5 (TRAINING_LOG.md, `runs/two_stream_returns`, assessed mid-run at ~47k
env steps / ~23k grad steps) validated tickets/0005's wiring exactly as
designed — and exposed the failure mode sitting one layer beneath it.

The causal chain, from the run's own scalars:

1. Intrinsic-driven play produced the run's first two scoring episodes
   (lp85 at env step ~44.2k, sp80 at ~45.5k — 2 scoring episodes out of
   104 completed).
2. Within ~200 grad steps of the first one landing in the buffer,
   `train/reward_windows_in_batch` went 0 → 4 (grad step 21,438) and stayed
   there — the stratified sampler works, hitting its 25%-of-batch target
   from exactly 2 distinct events.
3. Then the pre-registered "Ugly" fired, in slow motion. Over the next
   ~2,000 grad steps `policy/value_ext_mean` climbed **0.03 → 1.6 (peak
   2.55) and was still rising monotonically** when assessed;
   `policy/return_norm_scale_ext` 0.11 → 5.35; ext critic loss spiked to
   2.86 (700× its baseline) and settled ~10× elevated;
   `policy/extrinsic_reward_mean` in dreams reached ~0.03/step — **~1,000×
   the real per-step reward rate** (2 events in 47k steps). With
   `gamma=0.997`, a hallucinated 0.03/step supports a steady-state value
   ≈ 10; the climb had no reason to stop.

Checkpoint probes (run against `latest.pt` + `buffer.pt` at ~23k grad
steps) pin the mechanism:

- **Dreams from reward-window starts farm the score.** Under the current
  policy, 15-step dreams starting inside real scoring windows collect a
  mean of **1.63 predicted extrinsic reward per dream (p90 3.2, max 4.3)**
  — the real games pay +1 for a level completion **once**. 11% of all dream
  steps from those starts claim reward > 0.5. Under uniform-random dream
  actions from the same starts the mean is 0.007 — the policy has
  *specifically learned* action sequences that re-trigger the reward head.
- **The reward head is fine on real data** (`loss/reward` ≈ 1.5e-5 after
  the spike). It over-predicts only on *imagined* states the policy steers
  into — states nothing real ever anchors, because the world model has
  seen exactly 2 scoring transitions and cannot know that a level
  completion consumes itself (the level changes; the same transition can't
  pay twice).
- **Ensemble disagreement does not flag the farmed states**
  (corr(dream reward, disagreement) ≈ −0.01 from reward-window starts;
  disagreement at reward>0.5 steps ≈ disagreement elsewhere). The
  oversampled event windows are exactly where the ensemble is most
  converged, and all members generalize the same wrong way. So a
  MOPO-style uncertainty penalty on dream reward — the textbook fix —
  **would not bite here**; this was tested before being rejected.
- **The critic has decoupled from reality everywhere**: v_ext ≈ 6.2 at
  reward-window start states (true achievable: ~1), and ≈ 0.56 averaged
  over uniform starts (true: ~0.001). Bootstrapped λ-return targets built
  from farmed rewards + inflated target values feed the inflation back
  through `critic_target` every sync.

Everything else about the run is healthy — recon ~0.009 and samples sharp,
imagination holds the horizon, disagreement alive (~0.004), entropy 1.9–2.4
after the ext stream woke (above the 0.3 trigger), no NaN/inf. The world
model is not the problem. The *objective the actor optimizes inside it* is:
an unbounded sum of a reward the environment pays at most once per level.

**The fix: make a predicted score absorbing for the extrinsic return.**
Fold the reward head's (clamped) prediction into the extrinsic stream's
discount chain, so that once a dream banks a score, everything after it —
further farmed rewards *and* the inflated bootstrap tail — is multiplied
toward zero. A dream can then earn at most ~one score of extrinsic return,
which is exactly what one real level completion is worth. This removes the
farming incentive from the actor and bounds the critic's regression
targets to ~[−1, 1], deflating the inflation loop, in one small change to
`actor_critic_losses`. No parameter shapes change; Run 5's checkpoint
stays resumable.

**Read these first:**

- TRAINING_LOG.md Run 5's Findings — the evidence above, including the two
  probe scripts' numbers.
- `training/actor_critic.py` — `lambda_returns` (unchanged; the fix is in
  what gets passed as its `discounts`) and `actor_critic_losses` (the
  function this ticket edits).
- `model/thumper.py::dream` — confirms `reward[:, t]` / `continue_prob[:, t]`
  are the heads at the *arriving* state of transition t, aligned with
  `discounts[:, t]`; the absorbing factor must use the same alignment.
- tickets/0005 — the two-stream design this ticket amends; every isolation
  rule there still holds. The intrinsic stream is untouched.

---

## Design & Core Principles

1. **A score ends the extrinsic credit stream, not the dream.** The dream
   rollout, the intrinsic stream, and the continue-based discounting all
   stay exactly as they are. Only the *extrinsic* λ-return sees a modified
   discount:

   ```
   p_t            = clamp(dream["reward"][:, t], 0, 1)        # predicted "a score happened here"
   discounts_ext  = discounts * (1 - p_t)                     # discounts = gamma * continue_prob, as today
   R_ext          = lambda_returns(dream["reward"], discounts_ext, v_ext_target, lam)
   ```

   In `lambda_returns`' backward recursion, `R_t = r_t + disc_t * (...)`,
   so the score at step t is itself fully credited and the tail beyond it
   is scaled by `(1 - p_t)` — a confident +1 prediction kills the tail
   outright; a spurious 0.1 dampens it multiplicatively. This is the
   standard absorbing-state trick, applied per-stream.
2. **Why this matches the environment.** ARC-AGI-3 reward is the
   `levels_completed` delta: each +1 is a one-shot transition into a new
   level whose dynamics the world model (2 positive examples, ever) cannot
   yet represent. Post-score imagined states are unfalsifiable fantasy;
   giving them zero extrinsic weight is honest, not conservative. Multi-
   level returns are not lost — level-2 states enter the buffer through
   *real* play and become their own dream starts; they are simply not
   claimable from inside a single dream.
3. **Negative rewards do not absorb.** `clamp(r, 0, 1)`: a level *loss*
   (−1) is a real, modelable in-game event, not a discontinuity into an
   unmodeled regime; it keeps discounting through `continue_prob` as today.
4. **The clamp input is grad-free by construction** (`dream` runs under
   `no_grad`), so no new gradient path opens from the actor/critic losses
   into the reward head. State it in a comment; assert it in tests.

---

## Implementation Tasks

### Step 1: `training/actor_critic.py` — the fix

- In `actor_critic_losses`, after `discounts` is built: compute
  `absorb = 1.0 - dream["reward"].clamp(0.0, 1.0)` and use
  `discounts * absorb` for the **extrinsic** `lambda_returns` call only.
  The intrinsic call keeps `discounts` unchanged.
- Comment the two lines with the invariant ("a predicted score is
  absorbing for extrinsic credit: one dream can bank at most ~one score —
  see tickets/0006") — this is the kind of constraint the code can't show.
- Telemetry: add `dream_score_sum` (mean over dreams of
  `dream["reward"].clamp(0, 1).sum(dim=1)`) to the returned metrics — the
  direct farming gauge. Run 5's checkpoint measures ~1.6 from
  reward-window starts; post-fix training should pull the batch mean well
  under 1 and *hold* it there. Trainer logs it as
  `policy/dream_score_sum`.
- No signature, config, or parameter-shape changes. No toggle: the
  pre-fix behavior is not a defensible alternative worth A/B-ing (it is
  unbounded by construction), and Run 5's curves are the "before" arm.

### Step 2: `training/trainer.py` — telemetry passthrough

- Log the new `policy/dream_score_sum` scalar in `_log_policy_scalars`'s
  metric set (or wherever the policy metrics dict is consumed — follow the
  existing names).

### Step 3: Docs

- CLAUDE.md's `training/actor_critic.py` mention gains "extrinsic returns
  are absorbing at predicted scores (tickets/0006)".
- `actor_critic.py` header docstring: one sentence on the absorbing rule
  next to the existing two-stream description.

### Step 4: Tests (`tests/`)

Fast, CPU-only, extending the existing actor-critic unit tests:

1. **Absorption math:** hand-built dream with rewards +1 at steps 2 and 5,
   `continue_prob ≡ 1`: `R_ext[0]` matches the hand-computed λ-return in
   which step 5's reward and all post-step-2 value bootstraps contribute
   exactly zero (the step-2 score itself fully credited).
2. **Inflation-loop break (the ticket's reason to exist):** target values
   set to an absurd constant (e.g. 100) with a +1 reward at step 0:
   `R_ext[0] = 1` exactly — the inflated bootstrap tail cannot leak past
   an absorbing score.
3. **No-score neutrality:** with `dream["reward"] ≡ 0`, `R_ext` is
   bit-identical to the pre-fix computation (i.e. to `lambda_returns` with
   unmodified discounts).
4. **Negative rewards don't absorb:** reward −1 mid-dream leaves the
   extrinsic discount chain untouched at that step.
5. **Intrinsic stream isolation:** `R_int` and the intrinsic advantage
   component are bit-identical whatever the extrinsic rewards are (extend
   0005's drowning regression test pattern).
6. **No new grad path:** `actor_loss.backward()` and
   `critic_loss.backward()` still leave the world model's reward head
   grad-free.

### Step 5: Training Log

Add the Run 6 entry per TRAINING_LOG.md conventions before launching:
**resume `runs/two_stream_returns` in place** (no parameter shapes change,
so `latest.pt` loads; the buffer's 2 scoring events and 47k steps of
on-policy data are too valuable to discard — `--init-from` would drop
them). Pre-register the recovery signature: `policy/value_ext_mean`
falling from ~1.6 back under ~1 within a few hundred grad steps (bounded
targets deflating the poisoned critic), `policy/return_norm_scale_ext`
decaying from ~5 as the return spread shrinks, `policy/dream_score_sum`
settling < 1, and — the actual behavioral question, carried over from
Run 5 — scoring recurring on lp85/sp80/cd82 more than incidentally.

---

## Non-goals

- **No uncertainty/pessimism penalty on dream reward** (MOPO-style) — the
  Run 5 probe showed ensemble disagreement is uncorrelated with the farmed
  reward states; the penalty would cost a knob and buy nothing here.
  Revisit only with evidence of hallucinated reward that absorption
  doesn't cap.
- **No stratified-sampling changes.** `reward_window_frac=0.25` from 2
  distinct events means each event trains ~2×/batch, every batch — that
  concentration accelerated the blowup but did not cause it (the farmed
  return is unbounded at any sampling rate). If post-fix runs show the
  reward head's basin around events still over-widening, cap rows per
  *distinct* event in a follow-up ticket.
- **No grounding the ext critic on real replay returns** — attacks the
  value inflation but leaves the actor's farming incentive intact
  (R_ext − v_ext stays large at farmable states); absorption fixes both
  ends. Reconsider as a complement if post-fix v_ext is still biased.
- **No gamma/horizon retuning, no reward-head changes, no world-model
  objective changes.**

---

## Verification & Acceptance Criteria

1. **`uv run pytest` fully green**, including all of Step 4.
2. **Test 2 (inflation-loop break) passes** — the headline invariant: no
   value can leak past an absorbing score.
3. **Checkpoint probe replayed against the post-fix run's checkpoint**
   (script preserved in TRAINING_LOG.md Run 5's Findings): per-dream total
   extrinsic λ-return from reward-window starts bounded ≈ ≤ 1, vs 1.63
   mean / 4.3 max measured on Run 5's.
4. **The resumed run is out of scope** — it gets its own TRAINING_LOG.md
   entry (Step 5) with the pre-registered recovery signature above.
