# 0011 — Decision-Time Planning: Use the World Model at Act Time

## Overview & Architectural Justification

Thumper carries a full dynamics model everywhere it goes and never consults
it when choosing an action. The world model is only ever used for
_training-time_ imagination (`Thumper.dream` feeding
`training/actor_critic.py`); at act time the agent is a single reactive
forward pass — `OnlineActor.act` steps the RSSM posterior and samples the
policy once. This is ARCH.md Missing Features #2, and tickets/0010 named it
as the designated next mechanism: test-time training only ever improves the
reactive policy, it never _searches_, so on a novel game where the policy's
habits don't transfer, search over the (possibly adapted) world model is the
most plausible source of competent behavior. A null result from the 0010
pre-registered adaptation run points here directly.

The cheapest real planner is **policy-guided shooting** (sampling-based
MPC): from the actor's current latent state, roll `N` candidate futures by
letting the policy act against the RSSM **prior** for `H` steps, score each
rollout with exactly the return math the actor-critic already trains on
(TD(λ) over predicted rewards, absorbing extrinsic discounts, target-critic
bootstrap), and execute the **first action of the best rollout**. Then
replan from scratch at the next step (the MPC loop). Nothing new is learned:
this is `Thumper.dream` + `training/actor_critic.py::lambda_returns`,
re-aimed from "compute a policy gradient" to "pick the best of N futures".

The design guarantees the planner can only refine the policy, never
freelance away from it: candidates are sampled _from the policy_ (so search
stays inside the distribution the world model was trained under), and one
candidate is always the policy's own greedy rollout, placed first so score
ties resolve toward it. Planning is exposed as a third eval mode (`"plan"`,
alongside `"greedy"`/`"sampled"`), so its value is measured on the exact
fixed protocol every other mechanism is measured on.

**Read these first (in this order):**

- `ARCH.md` §2 (the Dreamer primer — imagination is the machinery being
  reused), §3.5 (`imagine_step`, the ensemble/disagreement), §4.6 (the
  return math the planner's scoring mirrors), Missing Features #2.
- `model/thumper.py::dream` — the grad-free rollout this ticket reuses
  wholesale: signature, the returned dict's exact keys/shapes, the
  macro-context freeze rule, the held-fixed `available_actions` mask.
- `training/actor_critic.py` — `lambda_returns` (the scorer's core),
  `actor_critic_losses` lines 132–148 (discounts from `continue_prob`, the
  tickets/0006 absorbing multiplier, the `critic_target(features)` call and
  its channel contract). The planner's scoring must be these same lines,
  factored to be callable from both places is **not** required — but the
  math must match term for term.
- `training/online_actor.py` — `act` (where planning plugs in; the
  step→fold→act ordering from tickets/0008 is load-bearing and must not
  move), `_mask_from_available` (the mask the planner must consume).
- `training/evaluate.py` — `EvalProtocol.modes`, `evaluate`'s per-(game,
  mode) actor construction (tickets/0010), `_run_episode`.
- `model/critic.py::forward` — the `(..., 2)` channel contract (0 =
  extrinsic, 1 = intrinsic).
- `tests/test_actor_critic.py` (hand-built dream dicts) and
  `tests/test_evaluate.py` (`_FakeEnv`) — the two test patterns Step 5
  follows; `tests/conftest.py` — the shrunken Thumper config.

---

## Design & Core Principles

1. **Planning is a mode of acting, not a new model component.** No new
   `nn.Module`, no new parameters, no new losses, nothing to checkpoint.
   The planner is a pure function over a frozen `Thumper` — it composes
   `dream` and `lambda_returns` and returns an action. It lives in
   `training/planner.py` (it imports from `training/actor_critic.py` and
   is consumed by `training/online_actor.py`, both already in `training/`).

2. **Score rollouts with the training-time return math, verbatim.** A
   candidate's score is its λ-return _from the start state_:
   `returns_ext[:, 0] + intrinsic_scale · returns_int[:, 0]`, computed with
   the same γ/λ, the same `gamma · continue_prob` discounts, the same
   tickets/0006 absorbing multiplier on the extrinsic chain, and the same
   `critic_target` bootstrap that `actor_critic_losses` uses. Inventing a
   second, subtly different notion of "return" is how planning and training
   end up optimizing different objectives. Default `intrinsic_scale = 0.0`:
   planning is an _exploitation_ mechanism (eval-time), and the raw
   intrinsic stream is ~10× the extrinsic one with no `ReturnNormalizer`
   available at eval time (`Thumper.load` carries no normalizer state —
   the scales live in the Trainer). A nonzero value is exposed for
   experiments, honestly labeled unnormalized.

3. **The policy proposes; the greedy rollout is always candidate 0.** All
   `N` candidates are sampled from the policy (candidate diversity comes
   from both action sampling and the prior's stochastic latent), plus one
   rollout where the policy acts greedily. Ties in the argmax resolve to
   the lowest index, so with the greedy candidate at index 0 an
   uninformative scorer (e.g. a flat reward head) degrades the planner to
   exactly the greedy policy — the planner's floor is the reactive agent,
   by construction. With `num_candidates = 0` the planner _is_ the greedy
   policy for the executed first action (the greedy rollout's first action
   is argmaxed from the real current features, before any imagined
   stochasticity) — Step 5's exactness test relies on this.

4. **The actor's latent-state bookkeeping is untouched.** Planning replaces
   only the "sample the policy once" line inside `OnlineActor.act` — the
   frame-stack, `observe_step`, the tickets/0008 TaskEncoder fold, and
   `observe` are byte-for-byte the same code path. The dream happens _after_
   the fold, from the same `(deter, stoch, m)` the reactive policy would
   have read, and its imagined states are discarded — nothing imagined ever
   contaminates the actor's real posterior state. A planning actor and a
   reactive actor fed the same transitions have identical latent state.

5. **Planning is eval-only for now.** The collector keeps acting reactively:
   planning during collection would change the training data distribution,
   entangle this ticket with exploration questions (a plan-greedy collector
   stops paying the intrinsic stream), and multiply collection wall-time by
   ~N·H world-model steps per env step. First measure whether search helps
   at all on the fixed protocol; feeding planned experience back into
   training is its own ticket with its own risks (see Non-goals). Structural
   consequence: `Trainer` passes no planner and needs no changes.

6. **`"plan"` is a third protocol mode, measured like the other two.**
   `EvalProtocol.modes` already parameterizes the sweep; planning slots in
   as a mode string rather than a parallel harness, so every existing
   surface (`eval.py` CLI, the in-training hook, `adapt.py`'s cells) can
   request it with a flag change and zero new plumbing, and its numbers
   land in the same `EvalReport`/`summary_table` rows as greedy/sampled.
   Defaults (`("greedy", "sampled")`) are unchanged — no existing run,
   report, or test changes behavior.

7. **Fixed mask, fixed `m`, fresh replan every step.** Inside one planning
   call, the start state's `available_actions` mask is held for the whole
   rollout and `m` is frozen — the same two rules `dream` already enforces
   for training (the world model has never seen an illegal action;
   a game's rules don't change mid-dream). Across steps, the plan is
   discarded and recomputed from the new posterior — no plan caching, no
   action-sequence commitment (that's an optimization to consider only if
   wall-time demands it; see Non-goals).

---

## Implementation Tasks

### Step 1: `model/thumper.py` — `dream` grows a `greedy` flag

- `dream(..., horizon: int, greedy: bool = False)`; the flag is passed
  through to the existing `self.policy.act(features, available_actions)`
  call (which already accepts `greedy`). Default `False` preserves every
  existing call site (the trainer's `policy_train_step`) bit-for-bit.
- Docstring: one added sentence — greedy rollouts are used by decision-time
  planning (tickets/0011) for the always-present policy-faithful candidate;
  note the dynamics stay stochastic (`imagine_step` still samples `stoch`),
  only the action choice is argmaxed.

### Step 2: `training/planner.py` — the planner (new file)

Module docstring in the house style: what it is (policy-guided shooting
MPC over the frozen world model), why it's not a model component (Design
principle 1), the scoring contract with `actor_critic.py` (Design
principle 2), and the tickets/0011 pointer.

```python
@dataclass
class PlannerConfig:
    num_candidates: int = 64
    """Sampled-policy rollouts per planning call, in addition to the one
    greedy rollout (candidate 0). 0 -> pure greedy policy."""
    horizon: int = 15
    """Imagined steps per rollout. Default matches TrainerConfig.dream_horizon
    -- the depth training validated imagination at; deeper is untested."""
    gamma: float = 0.997
    return_lambda: float = 0.95
    intrinsic_scale: float = 0.0
    """Weight on the intrinsic (disagreement) lambda-return in a rollout's
    score. Unnormalized (no ReturnNormalizer exists at eval time) -- keep 0
    for exploitation planning; nonzero values are experiments."""
```

Two functions, both pure and grad-free:

- `score_rollouts(dream: dict[str, Tensor], critic_target: Critic,
cfg: PlannerConfig) -> Tensor` — `(N,)` scores. Mirrors
  `actor_critic_losses` exactly: `discounts = cfg.gamma *
dream["continue_prob"]`; `absorb = 1 - dream["reward"].clamp(0, 1)`;
  `target_values = critic_target(dream["features"])` (no_grad);
  `returns_ext = lambda_returns(dream["reward"], discounts * absorb,
target_values[..., 0], cfg.return_lambda)`; `returns_int =
lambda_returns(dream["intrinsic"], discounts, target_values[..., 1],
cfg.return_lambda)`; return
  `returns_ext[:, 0] + cfg.intrinsic_scale * returns_int[:, 0]`.
  Import `lambda_returns` from `training.actor_critic` — do not copy it.
- `plan(thumper: Thumper, deter: Tensor, stoch: Tensor,
macro_context: Tensor, available_actions: Tensor,
cfg: PlannerConfig) -> dict[str, Tensor]` — inputs are the `(1, dim)`
  state tensors `OnlineActor.act` holds right after its fold, plus the
  `(1, num_action_types)` bool mask. Under `torch.no_grad()`:
  1. Greedy rollout: `thumper.dream(deter, stoch, macro_context,
available_actions, cfg.horizon, greedy=True)` on the 1-row inputs.
  2. Sampled rollouts (skip when `num_candidates == 0`): expand each state
     tensor and the mask to `num_candidates` rows
     (`.expand(N, -1).contiguous()` — `dream` writes nothing into its
     inputs, but `contiguous` keeps the rollout rows independent and
     cheap-to-index), one `thumper.dream(..., greedy=False)` call.
  3. Concatenate the two dreams' `reward`/`continue_prob`/`intrinsic`/
     `features`/`action_types`/`coords` along dim 0, **greedy first**
     (Design principle 3's tie-break), `score_rollouts`, `best =
scores.argmax()`, and return
     `{"action_type": action_types[best, 0:1], "coords": coords[best, 0:1],
"score": scores[best], "greedy_chosen": best == 0}` — first-step
     action of the winning rollout, shaped like `Policy.act`'s 1-row
     output so the caller's unpacking is identical.

### Step 3: `training/online_actor.py` — planning as an actor option

- `OnlineActor.__init__(self, thumper, device, planner: PlannerConfig |
None = None)` — stored as `self.planner`; `None` (the default, and what
  the Trainer keeps passing) is the reactive actor, unchanged.
- In `act`: everything through the TaskEncoder fold and
  `mask = self._mask_from_available(...)` is untouched (Design
  principle 4). Then:
  - `self.planner is None` → the existing `self.thumper.act(...)` call,
    unchanged.
  - else → `out = plan(self.thumper, self._deter, self._stoch,
self._macro_context, mask.unsqueeze(0).to(device), self.planner)`;
    unpack `action_type`/`coords` exactly as the reactive branch does. The
    `greedy` argument is ignored on this branch (the planner has its own
    action-selection semantics) — note that in the docstring rather than
    asserting, since `evaluate` passes `greedy=False` for non-greedy modes.
- Class/method docstrings: one sentence each on the planner option, its
  eval-only intent, and that imagined states never touch the actor's real
  latent state (Design principle 4), citing tickets/0011.

### Step 4: `training/evaluate.py` + surfaces — the `"plan"` mode

- `EvalProtocol` gains `planner: PlannerConfig =
field(default_factory=PlannerConfig)` with a docstring: consumed only
  when `"plan"` is in `modes`; `--protocol.modes greedy sampled plan`
  requests the mode (tyro parses the tuple as space-separated values, like
  `eval-games`).
- In `evaluate`'s mode loop: `greedy = mode == "greedy"` stays;
  construct the per-(game, mode) actor as
  `OnlineActor(thumper, str(device), planner=protocol.planner if mode ==
"plan" else None)`. Nothing else changes — `_run_episode` already passes
  `greedy` through and the plan branch ignores it. Mode strings other than
  the three known ones should now raise a `ValueError` naming the valid
  set (previously anything non-"greedy" silently meant sampled; with three
  modes a typo like `"palm"` must not silently measure sampled).
- `eval.py`: no changes — tyro auto-exposes `--protocol.planner.num-candidates`
  etc. and `--protocol.modes`. Verify with `--help`.
- `adapt.py` / the trainer's `eval_every` hook: no changes — both build
  `EvalProtocol`s and inherit the mode for free when a user requests it.
  (Do **not** add plan cells to `adapt_report.json`'s fixed four-cell
  schema — see Non-goals.)

### Step 5: Tests (`tests/test_planner.py`, plus one in `test_evaluate.py`)

Fast, CPU-only, shrunken conftest Thumper. The scoring tests build dream
dicts by hand (the `test_actor_critic.py` pattern) — no env, no rollouts:

1. **Scoring math mirrors training:** a hand-built 3-candidate dream dict
   (known rewards/continues/intrinsics, a stub `critic_target` returning
   constant values) → `score_rollouts` returns the λ-returns computable by
   hand; the candidate with reward 1.0 at step 0 outscores one with
   reward 1.0 at the last step (discounting), which outscores all-zeros.
2. **Absorbing extrinsic credit:** a candidate with reward 1.0 at step 0
   _and_ 1.0 at step 1 scores (approximately) the same as one with a
   single reward at step 0 — the second score is absorbed — while the
   intrinsic stream (nonzero `intrinsic_scale`) still accrues past a score.
3. **Greedy floor is exact:** with `num_candidates=0`, `plan(...)` returns
   the same `action_type`/`coords` as
   `thumper.act(deter, stoch, m, mask, greedy=True)` on the same inputs
   (torch seed fixed; the executed first action is pre-imagination, so this
   is exact equality, not approximate — Design principle 3).
4. **Mask is respected:** over several seeds and random small Thumpers,
   `plan` with a mask allowing only e.g. {RESET, ACTION3} never returns
   another type; with ACTION6 masked out, returned coords are still
   well-formed ints (they're just unused).
5. **Planner leaves actor state alone (Design principle 4):** two
   `OnlineActor`s on the same thumper — one `planner=None`, one with a
   small `PlannerConfig` — driven through identical
   `begin_episode`/`act`/`observe` sequences (same frames/rewards; feed the
   _reactive_ actor's actions to both `observe` calls) end with bit-identical
   `_deter`/`_stoch`/`_macro_context`.
6. **`"plan"` mode end-to-end + determinism** (`test_evaluate.py`):
   `evaluate` with `modes=("plan",)`, a tiny `PlannerConfig`
   (`num_candidates=2, horizon=2`), and the `_FakeEnv` produces an
   `EvalReport` whose episodes all carry `mode == "plan"`; two runs with
   the same seed produce identical episode lists; an unknown mode string
   raises.
7. **`dream(greedy=True)` passthrough:** with a fixed torch seed, two
   greedy dreams from the same start state produce identical
   `action_types` at step 0; `greedy=False` remains the default (existing
   trainer tests untouched).

### Step 6: Docs

- **ARCH.md**: rewrite Missing Features #2 as partially addressed (shooting
  MPC at eval time; still missing: planning during collection, iterative/
  tree search — cross-link the Non-goals), add a short §4.7 paragraph on
  the `"plan"` mode and the scoring contract, and a `training/planner.py`
  mention in the §4 module tour.
- **CLAUDE.md**: one clause in the `eval.py` command line (the `plan` mode
  and `--protocol.planner.*` flags) and one sentence in the Training
  section's `evaluate` description.
- **README.md**: one section in the `## Current Progress` to be updated to
  reflect current progress and status.
- Module/config docstrings per Steps 1–4.

### Step 7: Run guidance (for the future TRAINING_LOG entry, not executed here)

No training run — measurement only, like tickets/0007. Against the best
available checkpoint (the 0009 generalization run's, and the 0010 adapted
per-game checkpoints once they exist):

```sh
uv run python eval.py --checkpoint runs/held_out_v1/latest.pt \
  --protocol.modes greedy plan
```

Pre-registered questions: **(1)** does `plan` beat `greedy` on
`mean_levels_completed`/`win_rate` anywhere — on the training games (where
the reward head is best-anchored, so planning has real signal to search
over) vs the held-out games (where Missing Features #2 predicted it matters
most)? **(2)** on the 0010 adapted checkpoints, does planning stack with
test-time training (frozen+plan vs adapted+plan vs the existing cells)?
Also record `greedy_chosen`'s rate if debugging demands it (how often
search overrides the policy — near 1.0 means the scorer is uninformative,
near 0.0 with no score gain means the reward head is hallucinating
plannable futures; both are findings). Honest expectation: with a weak
reward head, `plan` ≈ `greedy` (the designed floor); genuine gains require
the reward/continue heads to rank futures better than the policy's own
habits, which is exactly what this measures. Wall-time note: `plan` costs
~`(num_candidates+1) × horizon` world-model steps per env step — scope
sweeps with `--protocol.games`/`--protocol.episodes-per-game` accordingly.

---

## Non-goals

- **No planning during training collection.** The collector stays reactive
  (Design principle 5); feeding planned trajectories into the buffer is a
  future ticket that must reckon with distribution shift and the intrinsic
  stream.
- **No iterative refinement (CEM) or tree search (MCTS).** Single-shot
  policy-guided shooting first; escalate to CEM/MCTS only if shooting shows
  signal worth amplifying (and cite this ticket's numbers when proposing it).
- **No plan caching / action-sequence commitment / batched-env speedups.**
  Replan every step, eat the cost, scope the sweep instead (Step 7).
- **No learned planner components** — no value/policy distillation from
  plans, no planner-specific heads, nothing new in checkpoints.
- **No `adapt_report.json` schema change.** The 0010 four-cell report stays
  fixed; plan-mode numbers on adapted checkpoints come from running
  `eval.py` against the per-game `latest.pt` files (Step 7), not from
  widening the adaptation harness.
- **No normalizer plumbing for the intrinsic score term.** Persisting
  `ReturnNormalizer` scales into `Thumper.save` just to support
  `intrinsic_scale > 0` planning is not worth the checkpoint-format churn;
  the field ships unnormalized and documented as such.

---

## Verification & Acceptance Criteria

1. **`uv run pytest` fully green**, including all of Step 5, with every
   pre-existing test untouched (default-off everywhere: `planner=None`,
   `modes` default unchanged, `dream(greedy=False)` default).
2. **The floor holds on a real checkpoint:** `eval.py` with
   `--protocol.modes greedy plan --protocol.planner.num-candidates 0` on
   any existing checkpoint produces identical `plan` and `greedy` rows
   (same seed, same episodes) — the planner's degenerate case _is_ the
   greedy policy.
3. **A real planning sweep completes:** `eval.py --protocol.modes plan
--protocol.games <two known-scoring games, e.g. lp85 cd82>
--protocol.episodes-per-game 2` runs against a real checkpoint, prints
   `plan` rows in the summary table, and writes them into the JSON report;
   running it twice yields byte-identical reports.
4. **Reactive paths are bit-identical:** a `greedy`+`sampled` eval of an
   existing checkpoint before and after this ticket produces identical
   reports (same seed), and a short training run's `loss/*`/`policy/*`
   curves match a pre-ticket run of the same seed (the `dream` signature
   change is inert at default).
5. **The deliverable question is answerable from the CLI output alone:**
   one summary table showing greedy vs plan per game answers "does
   decision-time search over the world model beat the reactive policy,
   and where" — no TensorBoard archaeology, and Step 7's pre-registered
   questions need no further protocol decisions.
