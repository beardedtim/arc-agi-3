# 0005 — Two-Stream Returns: Make Extrinsic Reward Visible to the Actor

## Overview & Architectural Justification

Run 3 (TRAINING_LOG.md, `runs/burn_in_fix`) validated the world model
end-to-end: imagination holds over the full horizon, and the reward head —
despite seeing only **15 nonzero-reward steps in 100,199** (0.015%; games
cd82/r11l/sp80) — predicts those transitions almost exactly (±1.0→±1.1 on
scoring transitions vs ~±0.007 on ordinary windows; note rewards go
**negative** when a level is lost, and the head learned that too). The
modeling pathway to score is done. What remains is that the _actor's
objective_ structurally cannot see it:

1. **Extrinsic is drowned by construction.** `actor_critic_losses` sums
   `reward + intrinsic_scale * intrinsic` into one return stream and
   normalizes advantages by one `ReturnNormalizer`. Run 3's
   `policy/return_norm_scale` settled around ~10, set entirely by the
   intrinsic stream (disagreement pays every imagined step;
   `policy/imagined_return` of 1.8–4.2 is essentially all intrinsic). A +1
   level completion is therefore worth ~0.1 normalized units, once — against
   a continuous stream of exploration payment. The critic is a disagreement
   forecaster; the policy is a pure explorer _no matter how much it trains_.
   This is exactly the follow-up tickets/0003's non-goals pre-registered:
   "splitting is a follow-up if the intrinsic term drowns extrinsic once
   scores appear". Scores appeared; it drowns.
2. **Timeouts are recorded as deaths.** The trainer stores
   `terminated = result.done or timeout` in the buffer, and the continue
   head trains on `terminateds`. On the 24/25 games whose episodes only
   ever end at the 600-step cap, the head is taught that ordinary states
   randomly terminate — which clips imagined value horizons for no reason
   grounded in the game.
3. **Reward events are needles in the buffer.** Under uniform window
   sampling a reward event lands in a batch ~7% of the time today, and that
   dilutes as the buffer grows toward its 200k capacity and as old scoring
   episodes evict. The head that currently nails ±1 only stays sharp if
   scoring windows keep getting sampled.

This ticket fixes all three: **separate extrinsic and intrinsic λ-returns,
each with its own critic head and its own return normalizer** (the
Plan2Explore→task-transfer recipe, single shared policy), **truncation ≠
termination** in collection, and **reward-event-stratified window
sampling**. No world-model objective changes; no new exploration machinery.

**Read these first** (the ticket references their exact APIs):

- TRAINING_LOG.md Run 3's Findings — the evidence above, plus the
  disagreement-consumption watch item this ticket's telemetry serves.
- `training/actor_critic.py` — `lambda_returns` (reused per stream,
  unchanged), `ReturnNormalizer` (instantiated twice), and
  `actor_critic_losses` (the function this ticket restructures). Note the
  `max(1.0, scale)` floor in `normalize`: it is the mechanism that makes
  splitting work — sparse extrinsic returns keep a spread < 1 and pass
  through _unscaled_, while the intrinsic stream's ~10 spread is tamed.
- `model/critic.py` — gains a second value head.
- `training/trainer.py` — `policy_train_step` (builds `ActorCriticConfig`
  from flat config fields), the collect loop (`terminate_due_to_max_steps`
  is where timeout is conflated with death), `train_step`'s
  `buffer.sample(...)` call, and the checkpoint payload
  (`return_norm_scale`).
- `training/replay_buffer.py` — `sample`'s window/idx conventions
  (repeat-padding, prev-action-at-t) and `Episode.rewards`.
- `model/world_model.py::compute_losses` — confirms the continue head
  regresses `batch["terminateds"]`; untouched by this ticket, which changes
  only what gets _stored_ there.
- tickets/0003 — the actor-critic design this ticket amends (its
  heads-at-arriving-state and gradient-isolation rules all still hold).

---

## Design & Core Principles

1. **One policy, two value streams.** Extrinsic (reward head) and intrinsic
   (ensemble disagreement) each get their own λ-return, critic head, target
   values, and `ReturnNormalizer`. The actor's advantage is the weighted sum
   of the two _normalized_ advantages:

   ```
   advantage = norm_ext(R_ext − v_ext) + intrinsic_scale · norm_int(R_int − v_int)
   ```

   With the `max(1, scale)` floor, extrinsic advantages are effectively raw
   (±1-ish, by env design) until extrinsic returns genuinely grow, while
   intrinsic advantages are scaled to O(1) instead of dominating 10:1.
   `intrinsic_scale` (default 1.0, unchanged) now weighs two same-scale
   streams, so it finally means what its name says — and annealing it later
   becomes a meaningful lever rather than a unit conversion.

2. **Streams must not contaminate each other.** Separate critic MLPs (not a
   shared trunk with two outputs): the intrinsic value function is
   deliberately nonstationary (disagreement decays as the world model
   improves — Run 3: `wm/disagreement_mean` 0.009→0.0023), and its drift
   must not perturb the sparse, stationary extrinsic estimate through
   shared hidden layers. The regression test for this ticket (Step 6,
   test 2) is: scaling the intrinsic stream by 100× leaves the extrinsic
   advantage component bit-identical.
3. **Only real deaths teach the continue head.** The env's `done` is the
   only terminal signal stored in the buffer; the trainer's 600-step cap
   rotates episodes but stores `terminated=False` at that step. Dreams
   already never see timeouts; now training targets match.
4. **Scoring windows stay in the training distribution.** A configurable
   fraction of each sampled batch is drawn from windows positioned so a
   nonzero-reward step lands inside the loss window (the trailing `seq_len`
   steps, after burn-in). This trains the whole `compute_losses` stack on
   those windows — reward head most importantly — at a controlled rate
   independent of buffer size. All gradient-isolation rules from 0002/0003/
   0004 are untouched.

---

## Implementation Tasks

### Step 1: `model/critic.py` — two value heads

- `Critic.__init__`: replace `self.net` with `self.ext_net` and
  `self.int_net`, both `_mlp_head(feature_dim, hidden_dim, 1)`.
- `forward(features) -> Tensor` now maps `(..., feature_dim) -> (..., 2)`,
  channel 0 = extrinsic value, channel 1 = intrinsic value
  (`torch.stack`/`cat` of the two squeezed heads; document the channel
  order in the docstring — it is a cross-file contract with Step 2).
- Update the module docstring (the "standalone value estimator" rationale
  stands; add the stream-isolation rationale from Design principle 2).
- `Thumper` needs no change: `critic`/`critic_target` remain single
  attributes, `sync_critic_target()`'s `state_dict` copy covers both heads,
  and generic component discovery is untouched.

### Step 2: `training/actor_critic.py` — two-stream losses

- `lambda_returns` is unchanged — call it once per stream.
- `ActorCriticConfig`: fields unchanged (`intrinsic_scale` keeps its name
  and default; its docstring/comment updates per Design principle 1).
- **`actor_critic_losses` signature:** replace the single `normalizer`
  parameter with `normalizer_ext: ReturnNormalizer, normalizer_int:
ReturnNormalizer` (keep them as two explicit parameters, not a tuple —
  call sites stay greppable).
- Body:
  1. `discounts = cfg.gamma * dream["continue_prob"]` — shared by both
     streams (one imagined trajectory, one termination process).
  2. Target values `(N, H+1, 2)` from `critic_target(features)` under
     `no_grad`; `R_ext = lambda_returns(dream["reward"], discounts,
target_values[..., 0], cfg.return_lambda)` and `R_int =
lambda_returns(dream["intrinsic"], discounts, target_values[..., 1],
cfg.return_lambda)`.
  3. **Critic loss:** `values = critic(features[:, :-1])` `(N, H, 2)`;
     `critic_loss = mse(values[..., 0], R_ext.detach()) +
mse(values[..., 1], R_int.detach())`. Both heads regress raw
     (unnormalized) returns, as today.
  4. **Advantage** per Design principle 1, both baselines detached, then
     the same REINFORCE + entropy actor loss as today over the combined
     advantage.
  5. `normalizer_ext.update(R_ext)`; `normalizer_int.update(R_int)`.
- Telemetry dict: replace `imagined_return`/`value_mean` with
  `imagined_return_ext`, `imagined_return_int`, `value_ext_mean`,
  `value_int_mean`; keep `actor_loss`, `critic_loss`, `entropy`,
  `intrinsic_mean`, `extrinsic_mean`.

### Step 3: `training/trainer.py` — normalizers, truncation, wiring

- **Two normalizers:** `self.return_normalizer_ext` /
  `self.return_normalizer_int`, both `ReturnNormalizer(decay=
c.return_norm_decay)`. Checkpoint payload key `return_norm_scale`
  becomes `return_norm_scales = {"ext": ..., "int": ...}`; resume restores
  both. (Pre-0005 checkpoints are not resumable anyway — Step 1 changed
  critic shapes; see the compatibility note.)
- **`policy_train_step`:** pass both normalizers through to
  `actor_critic_losses`; log the new metric names plus
  `policy/return_norm_scale_ext` and `policy/return_norm_scale_int`. The
  pair replicating Run 3's imbalance _visibly_ (ext staying ≲1 while int
  climbs toward ~10) is itself a deliverable — it turns the drowning from
  an inference into a chart.
- **Truncation fix (collect loop):** keep the local
  `terminate_due_to_max_steps` / episode-rotation behavior exactly as is
  (episode ends, per-game scalars log, next game rotates in), but store
  `terminated=result.done` in `buffer.add_step` — the timeout no longer
  reaches the buffer. One-line change; add a comment stating the invariant
  ("the continue head must only ever see real deaths — timeouts are
  truncation, see tickets/0005").
- **Stratified sampling wiring:** `TrainerConfig` gains
  `reward_window_frac: float = 0.25` ("target fraction of each batch drawn
  from windows whose loss window contains a nonzero-reward step; 0
  disables"). `train_step` passes it (and `loss_offset=c.burn_in`, see
  Step 4) into `buffer.sample`.

### Step 4: `training/replay_buffer.py` — reward-event-stratified sampling

- `sample(batch_size, seq_len)` grows two keyword args:
  `reward_frac: float = 0.0` and `loss_offset: int = 0` (the number of
  leading burn-in steps in each window; events must land at index
  ≥ `loss_offset`).
- Implementation: collect reward-event coordinates
  `[(episode_idx, t), ...]` across episodes (a scan over
  `episode.rewards` per call is O(total_steps) and fine at current scale —
  keep it simple; cache per-Episode nonzero indices only if profiling ever
  says otherwise). If the list is empty, fall back to fully uniform
  sampling — `reward_frac` is a target, not a promise.
- For `k = round(batch_size * reward_frac)` rows: pick an event uniformly
  at random, then choose the window's start so the event's index within the
  window is uniform over `[loss_offset, seq_len - 1]`, reusing `sample`'s
  existing idx-clamping/repeat-padding conventions for events too close to
  an episode's start (when clamping would push the event into the burn-in
  prefix, that row degrades gracefully into a near-start window — don't
  special-case it). The remaining `batch_size - k` rows sample exactly as
  today. No shuffling needed (SGD doesn't care about row order).
- Return an extra batch key `"num_reward_windows": int` (or have the
  trainer count nonzero `batch["rewards"][:, loss_offset:]` rows — pick
  one; the trainer logs it as `train/reward_windows_in_batch`).
- `save`/`load` are untouched: the event index is derived data, rebuilt
  from `rewards` — **no schema change**, old `buffer.pt` files stay loadable
  in both directions.

### Step 5: Docs

- CLAUDE.md: `training/actor_critic.py` mention gains "two-stream
  extrinsic/intrinsic returns with separate critics and normalizers
  (tickets/0005)"; the trainer bullet notes stratified reward-window
  sampling and timeout-as-truncation.
- Module docstrings touched by Steps 1–4 update in place (in particular
  `trainer.py`'s collect-loop comment and `actor_critic.py`'s header).

### Step 6: Tests (`tests/`)

`small_config()` conventions; fast, CPU-only. Update whatever Steps 1–4
broke (`uv run pytest`, fix all of it), plus:

1. **Two-stream λ-return math:** with intrinsic ≡ 0 and a single +1
   extrinsic reward at the last dream step, `imagined_return_ext` matches a
   hand-computed λ-return and the actor advantage is driven by the
   extrinsic stream alone.
2. **The drowning regression test (this ticket's reason to exist):**
   construct two dreams identical except the intrinsic stream is scaled
   ×100; the extrinsic advantage component (`norm_ext(R_ext − v_ext)`) is
   bit-identical in both, and the combined advantage's extrinsic component
   is not diminished (assert via `intrinsic_scale=0` equality).
3. **Normalizer floor:** a stream with spread < 1 passes through
   `normalize` unscaled; a stream with spread ~10 is scaled by ~1/10.
   Both normalizer scales survive a checkpoint save/resume roundtrip
   (extend the trainer resume test to the new payload key).
4. **Gradient isolation (re-run 0003's invariants against the new
   shapes):** actor backward leaves `world_model` and both critic heads
   grad-free; critic backward leaves `world_model`/`policy` grad-free;
   `sync_critic_target()` copies both heads.
5. **Truncation:** a synthetic collect step that hits the step cap stores
   `terminated=False` (and still rotates the episode); a step whose
   `StepResult.done` is True stores `terminated=True`.
6. **Stratified sampling:** a synthetic buffer with exactly one
   reward event and `reward_frac=1.0, loss_offset=b` yields batches where
   every row's `rewards[loss_offset:]` contains the event (episode long
   enough); `reward_frac=0.25` with _zero_ events falls back to uniform
   without error; event positions within the loss window vary across draws
   (not pinned to one index).

### Step 7: Training Log

Update the TRAINING_LOG.md file, following its convetions, with the next command
that we expect to run for this training run

---

## Checkpoint & buffer compatibility note

`Critic`'s parameter names/shapes change (Step 1), so **pre-0005
`latest.pt` files are not strictly resumable** — same situation as 0003's
note, same remedy: `--config.init-from` copies world-model weights only and
works against any post-0002 checkpoint. The next run should
`--config.init-from runs/burn_in_fix/latest.pt` in a **fresh output dir**:
Run 3's world model is validated, and a fresh policy/critic under the fixed
objective is the point of the run. `buffer.pt` schema is unchanged both
ways; note that Run 3's buffer carries ~166 timeout steps mislabeled
`terminated=True` (one per capped episode), which is another mild reason to
start the buffer fresh rather than resume it.

---

## Non-goals

- **No separate explorer/exploiter policies** (full Plan2Explore trains
  two); one policy on the summed _normalized_ advantages. Split only if a
  future run shows the compromise policy fails at both jobs.
- **No `intrinsic_scale` annealing schedule** — the split makes the knob
  meaningful; scheduling it is a follow-up informed by the next run's
  `policy/return_norm_scale_*` curves.
- **No reward shaping** — reward stays the raw `levels_completed` delta (no
  win bonus, no negative-reward clipping). The env's signal is already
  well-scaled; shaping is a last resort, not a default.
- **No general prioritized experience replay** (TD-error priorities,
  importance weights) — Step 4 is deliberate stratification of a known-rare
  event class, not PER.
- **No world-model objective changes** — `compute_losses`,
  `reward_loss_balance`, KL settings all untouched.
- **No online-collection policy changes** beyond the `terminated` flag.

---

## Verification & Acceptance Criteria

1. **`uv run pytest` fully green**, including all of Step 6.
2. **The drowning regression test passes** (Step 6, test 2) — this is the
   ticket's headline invariant: extrinsic advantages are invariant to the
   intrinsic stream's magnitude.
3. **Smoke run** (a few hundred grad steps past prefill, tiny budget,
   fresh output dir, `--config.init-from runs/burn_in_fix/latest.pt`):
   both `policy/return_norm_scale_*` scalars finite and diverging from each
   other (int climbing, ext near its 1.0 floor), `train/reward_windows_in_batch`
   nonzero once any scoring window enters the buffer, entropy healthy,
   `loss/recon` falling as in prior runs.
4. **The full training run is out of scope** — it gets its own
   TRAINING_LOG.md entry with pre-registered expectations. Headline things
   that run should look for (recorded here so the entry can cite them):
   `policy/value_ext_mean` and `policy/imagined_return_ext` moving off
   zero; episode returns on cd82/r11l/sp80 becoming _repeatable_ rather
   than Run 3's one-offs; any nonzero `online/win_rate/*`; `loss/recon`
   not degraded by stratified sampling (watch the recon samples for
   over-representation of the three scoring games); and the Run 3 watch
   item — whether disagreement-as-reward eats its own signal — now
   readable per-stream via `policy/imagined_return_int`.
