# 0003 — Dreamer-Style Actor-Critic Trained in Imagination with Disagreement Intrinsic Reward

## Overview & Architectural Justification

Thumper's world model is trained and validated (TRAINING_LOG Run 1: near
pixel-perfect recon across all 25 games, imagination coherent over most of an
8-step horizon, healthy ensemble disagreement), but the policy is still the
untrained module that ships in the checkpoint, and collection is uniform
random. Run 1 showed the consequence: episode length is pinned at the 600-step
cap on 24 of 25 games, the reward/continue heads see almost no signal, and the
buffer is uniform-random-by-construction. The world model is now
**data-limited, not architecture-limited** — the only way to get better data
is a policy that seeks it out.

This ticket closes the loop: train the existing actor (`model/policy.py`) and
a new critic **entirely in imagination** (DreamerV2-style latent rollouts from
replayed posterior states), with per-step reward = predicted extrinsic reward
(reward head) + **Plan2Explore intrinsic reward** (variance across the
`TransitionEnsemble`'s heads — the "exploration currency" that has been logged
as `wm/disagreement_*` since Run 1 but never consumed). The trained policy
replaces the random collector in the online loop, so exploration becomes
disagreement-seeking instead of uniform.

Actor gradients are **REINFORCE with a learned baseline**, not
backprop-through-dynamics: the action space is categorical (7-way type +
64×64 pointer), so there is no reparameterized path through the action sample.

**Read these first** (the ticket references their exact APIs):

- `model/policy.py` — `Policy.act` / `Policy.log_prob_entropy` (built for
  exactly this re-evaluation pattern), the factored type+pointer action space
  and `available_actions` masking.
- `model/world_model.py` — `forward_sequence` outputs (`deter`, `stoch`,
  `macro_context`, `action_onehot`), `features`, `predict_heads`
  (continue head outputs a *logit*), `encode_actions`, `disagreement`, and
  `compute_losses`' head convention: **heads predict at the arriving state**
  (the reward at features of step *t* is the reward for arriving at *t*).
- `model/rssm.py` — `imagine_step(prev_deter, prev_stoch, prev_action,
  macro_context)`.
- `training/trainer.py` — the online loop this ticket rewires; note
  `_random_action`, `_begin_episode`, the prefill/`train_every` scheduling,
  and the checkpoint payload.
- `training/replay_buffer.py` — step convention (prev-action-at-t), sample
  dict shapes.
- `tickets/0002-meta-rl-macro-context.md` — the imagination "freeze rule" and
  gradient-isolation principles, both of which this ticket extends.

---

## Architectural Rules & Core Principles

1. **The world model is a frozen simulator during policy training.** No actor
   or critic gradient may reach `world_model.parameters()`. Imagination
   rollouts run under `torch.no_grad()`; the actor/critic losses are computed
   in a second pass over *detached, stored* features. (This also keeps memory
   flat: no graph is retained through H RSSM steps.)
2. **Imagination freeze rule (extended from tickets/0002):** during a dream,
   the `TaskEncoder` is never stepped. Each dream start state carries the
   `macro_context` that `forward_sequence` produced *at that timestep*, and
   that exact vector is passed, frozen, into every `rssm.imagine_step` of its
   rollout. (0002 froze the zero-initialized context in
   `imagine_from_first_frame`; here the frozen value is per-start-state and
   generally nonzero.)
3. **Heads-at-arriving-state indexing.** For a dream of horizon H from state
   s₀ with actions a₀..a_{H−1} producing states s₁..s_H:
   - extrinsic reward for transition t→t+1 = `reward_head(features(s_{t+1}))`;
   - continue probability used to discount past t+1 =
     `sigmoid(continue_head(features(s_{t+1})))`;
   - intrinsic reward for t→t+1 = `disagreement(deter_t, stoch_t,
     onehot(a_t), macro_context)` — the ensemble's convention is
     "(deter, stoch) at t plus the action that *produces* t+1", matching
     `compute_losses`' alignment.
4. **Action masking in imagination is required, not optional.** The world
   model has only ever seen legal actions (the random collector sampled from
   `available_actions`), so ensemble disagreement on illegal actions is
   untrained garbage — without masking, the intrinsic reward would teach the
   policy to chase impossible actions in dreams. The replay buffer must
   therefore store each step's `available_actions` mask (Step 5), and dreams
   mask the actor with the *start state's* mask, held fixed for the rollout
   (legal action sets are effectively static within an ARC-AGI-3 game state
   regime; per-step imagined masks don't exist because the mask comes from
   the real env).
5. **Separate critic, separate optimizers.** The critic is a new standalone
   module (Step 1), not `Policy.value_head` (which shares the actor trunk —
   critic gradients would shape actor features). Actor, critic, and world
   model each get their own Adam. A **target critic** (hard-synced copy)
   provides the bootstrap values for λ-returns.

---

## Implementation Tasks

### Step 1: Create `model/critic.py` (new module)

- **`CriticConfig` (dataclass):**
  - `feature_dim: int = 416` — width of the world-model feature
    (deter ++ stoch ++ macro_context); derived in
    `ThumperConfig.__post_init__` (Step 3), never trusted by hand.
  - `hidden_dim: int = 256`
- **`Critic(nn.Module)`:** `forward(features: Tensor) -> Tensor` mapping
  `(..., feature_dim) -> (...,)` (squeeze the last dim). Reuse the
  `_mlp_head(feature_dim, hidden_dim, 1)` builder from `model/world_model.py`
  (import it) — same 2-hidden-layer ELU shape as the world model's heads.

### Step 2: Remove `Policy.value_head` (`model/policy.py`)

The critic now lives in Step 1's module; a value head sharing the actor trunk
must not remain (it would silently become dead weight, or worse, get wired in
by mistake).

- Delete `self.value_head`; drop `"value"` from `forward`'s return dict and
  from `act`'s return dict; change `log_prob_entropy` to return
  `(log_prob, entropy)` (drop the value element). Update all three docstrings.
- Update `tests/test_policy.py` and any other call site
  (`grep -rn "value_head\|log_prob_entropy\|\[.value.\]" --include='*.py'`)
  accordingly. `Thumper.act`'s return type is `Policy.act`'s dict, so its
  docstring reference updates too.

### Step 3: Update `model/thumper.py`

- **`ThumperConfig`:** add
  `critic: CriticConfig = field(default_factory=CriticConfig)`. In
  `__post_init__`, derive `self.critic.feature_dim` from the same sum used
  for `policy.feature_dim` (repo's config-invariant pattern).
- **`Thumper.__init__`:** add `self.critic = Critic(self.config.critic)` and
  `self.critic_target = Critic(self.config.critic)`, then
  `self.critic_target.load_state_dict(self.critic.state_dict())` and
  `self.critic_target.requires_grad_(False)`. Both are plain attribute
  assignments, so `parameter_counts`/`save`/`load` pick them up generically
  (the target contributes 0 to trainable counts — that's correct).
- **Add `Thumper.sync_critic_target()`:** hard-copies
  `critic.state_dict()` into `critic_target`. Called by the trainer on a
  grad-step cadence (Step 6).
- **Add `Thumper.dream(...)`** — the imagination rollout. Lives on `Thumper`
  (it needs both `world_model` and `policy`; `WorldModel` must not import
  `Policy`):

  ```python
  @torch.no_grad()
  def dream(
      self,
      deter: Tensor,            # (N, deter_dim)   detached start states
      stoch: Tensor,            # (N, stoch_dim)
      macro_context: Tensor,    # (N, context_dim) frozen for the whole dream
      available_actions: Tensor,  # (N, num_action_types) bool, frozen too
      horizon: int,
  ) -> dict[str, Tensor]:
  ```

  Loop `for t in range(horizon)`: build `features` from the current state,
  sample an action via `self.policy.act(features, available_actions)`, encode
  it with `world_model.encode_actions`, compute
  `intrinsic = world_model.disagreement(deter, stoch, action_onehot,
  macro_context)` **before** stepping, then
  `deter, stoch = world_model.rssm.imagine_step(deter, stoch, action_onehot,
  macro_context)`. Return stacked tensors:

  - `features`: `(N, H+1, feature_dim)` — start state's features first, then
    each imagined state's;
  - `action_types`: `(N, H)` int64 and `coords`: `(N, H, 2)` int64 — the
    action taken *at* states 0..H−1;
  - `intrinsic`: `(N, H)` — aligned so `intrinsic[:, t]` belongs to the
    transition arriving at state t+1;
  - `reward`: `(N, H)` — `predict_heads(features[:, 1:])["reward"]` (arriving
    states only);
  - `continue_prob`: `(N, H)` —
    `sigmoid(predict_heads(features[:, 1:])["continue_logit"])`.

  (`predict_heads` can be called once on the stacked `features[:, 1:]` after
  the loop rather than per step.) Everything returned is grad-free by
  construction of the `@torch.no_grad()` decorator — the trainable second
  pass happens in Step 4/6.

### Step 4: Create `training/actor_critic.py` (pure functions + normalizer)

Keep these free functions so they unit-test in isolation (mirroring how
`compute_losses` owns the world-model math).

- **`lambda_returns(rewards, discounts, values, lam) -> Tensor`:**
  `rewards`/`discounts`: `(N, H)`, `values`: `(N, H+1)` (from the **target**
  critic, states 0..H). Standard Dreamer TD(λ), computed backward:

  ```python
  next_return = values[:, -1]
  for t in reversed(range(H)):
      bootstrap = (1 - lam) * values[:, t + 1] + lam * next_return
      next_return = rewards[:, t] + discounts[:, t] * bootstrap
      returns[:, t] = next_return
  ```

  Returns `(N, H)`: `returns[:, t]` is the λ-return *from* state t.

- **`ReturnNormalizer`** — a small stateful class holding one float `scale`,
  updated per policy step with
  `scale = decay * scale + (1 - decay) * (quantile(returns, 0.95) -
  quantile(returns, 0.05))` (`decay=0.99`, DreamerV3-style). Exposes
  `normalize(adv) -> adv / max(1.0, scale)`. This is **required, not
  optional**: `wm/disagreement_mean` sits around ~0.009 and shrinks as the
  world model improves, so unnormalized advantages would vanish and the
  entropy term would dominate. The scalar must survive resume (Step 6's
  checkpoint payload).

- **`actor_critic_losses(dream, policy, critic, critic_target,
  normalizer, cfg) -> dict[str, Tensor]`** — the with-grad second pass:

  1. `rewards = dream["reward"] + cfg.intrinsic_scale * dream["intrinsic"]`;
     `discounts = cfg.gamma * dream["continue_prob"]`.
  2. `target_values = critic_target(dream["features"])` (no grad by
     construction) → `lambda_returns(...)` → `returns` `(N, H)`.
  3. **Critic loss:** `F.mse_loss(critic(dream["features"][:, :-1]),
     returns.detach())` — the critic regresses raw (unnormalized) λ-returns
     at states 0..H−1.
  4. **Actor loss:** re-evaluate the stored actions with
     `policy.log_prob_entropy(features_flat, action_types_flat, coords_flat,
     available_actions_flat)` (flatten (N, H) → (N·H,); broadcast the
     per-start mask across H — same mask the dream sampled under, so the
     re-evaluated distribution matches exactly). Then
     `advantage = normalizer.normalize(returns - critic(features[:, :-1]).detach())`
     (baseline detached — the critic trains only through its own MSE) and

     ```python
     actor_loss = -(log_prob * advantage.detach().flatten()).mean() \
                  - cfg.entropy_scale * entropy.mean()
     ```

  Return `{"actor_loss", "critic_loss", "entropy", "imagined_return",
  "intrinsic_mean", "extrinsic_mean", "value_mean"}` (the last five detached
  scalars, for telemetry).

### Step 5: Update `training/replay_buffer.py` — store `available_actions`

- `Episode` gains `available_actions: list[torch.Tensor]` — a
  `(num_action_types,)` bool mask per step (the mask that was legal *at* that
  observed state). Thread it through `append`, `add_step`, `sample`
  (→ `"available_actions": (B, T, num_action_types)` bool in the batch dict),
  `save` (pack as a `(T, num_action_types)` bool tensor per episode), and
  `load`.
- **Backward compat in `load`:** an old `buffer.pt` has no
  `"available_actions"` key — fall back to all-True masks so
  `--config.init-from` workflows against pre-0003 runs don't crash. (All-True
  is what "no mask" already means to `Policy.act`.)
- The buffer's constructor needs `num_action_types` (default
  `NUM_ACTION_TYPES` from `model/actions.py`) for the fallback and for
  synthetic tests.

### Step 6: Update `training/trainer.py` — act with the policy, train it

**Config additions** (`TrainerConfig`, all flat fields so `tyro` exposes them
as flags):

```python
# imagination policy training
dream_horizon: int = 15        # imagined steps per policy update (Dreamer default);
                               # drop toward 8-10 if dreams destabilize training —
                               # Run 1 validated coherence to ~8 steps
gamma: float = 0.997           # discount; episodes run to 600 steps, so 0.99 is too myopic
return_lambda: float = 0.95    # TD(lambda) mixing
entropy_scale: float = 1e-3
actor_lr: float = 8e-5
critic_lr: float = 8e-5
intrinsic_scale: float = 1.0   # weight on disagreement vs predicted extrinsic reward
critic_target_every: int = 100 # grad steps between hard target-critic syncs
return_norm_decay: float = 0.99
```

**Optimizers & checkpointing:**

- Add `self.actor_optimizer = Adam(thumper.policy.parameters(), lr=c.actor_lr)`
  and `self.critic_optimizer = Adam(thumper.critic.parameters(),
  lr=c.critic_lr)` alongside the existing world-model Adam. (The target
  critic gets no optimizer.)
- Checkpoint payload gains `actor_optimizer_state_dict`,
  `critic_optimizer_state_dict`, and `return_norm_scale`; resume restores all
  three. See the compatibility note below.

**Policy update — add `policy_train_step(outputs, batch)`**, called from
`train_step` immediately after the world-model `optimizer.step()` (one policy
update per world-model update; `train_every` scheduling is untouched):

- Start states: flatten every posterior state of the world-model pass —
  `outputs["deter"]`, `outputs["stoch"]`, `outputs["macro_context"]`
  reshaped `(B·T, dim)` and **`.detach()`ed**, plus
  `batch["available_actions"]` reshaped `(B·T, num_action_types)`. (Starting
  a few dreams from terminal/padded states is harmless — the continue-head
  discount zeroes what follows; don't bother filtering.)
- `dream = self.thumper.dream(..., horizon=c.dream_horizon)` →
  `actor_critic_losses(...)` → two independent backward/step passes (the
  actor and critic graphs are disjoint — log-probs touch only `policy`,
  values only `critic`): zero-grad, backward, `clip_grad_norm_` at
  `c.grad_clip`, step, for each.
- Every `critic_target_every` grad steps, `self.thumper.sync_critic_target()`.
- Update the `ReturnNormalizer` with this step's raw returns.

**Online collection — replace `_random_action` with the policy:**

- The trainer must now hold per-episode latent state: a frame-stack deque of
  the last K frames (first frame replicated at episode start — mirror
  `ReplayBuffer._stack`), plus `(deter, stoch, macro_context)` initialized
  from `rssm.initial_state(1)` / `task_encoder.initial_state(1)` in
  `_begin_episode`, and the previous action's onehot (zeros at episode
  start — no real action produced the first frame, same convention as
  `forward_sequence`'s `is_first` masking).
- Per env step, under `torch.no_grad()`, mirroring `forward_sequence`'s
  ordering exactly:
  1. `embed = world_model.encode(stack)`;
     `deter, stoch = rssm.observe_step(deter, stoch, prev_action_onehot,
     embed, macro_context)`.
  2. Build the bool mask from `result.available_actions`;
     `out = thumper.act(deter, stoch, macro_context, mask)`; extract
     `action_type`, `coords` (coords only consulted for ACTION6, as today).
  3. Step the env, store the step in the buffer (now including the mask the
     action was chosen under — i.e. the mask *of the state it was chosen at*;
     note the buffer's prev-action-at-t convention means the mask stored with
     frame t is the mask that was legal at frame t, observed alongside it).
  4. Fold the completed transition into the slow memory:
     `macro_context = task_encoder(macro_context, deter, stoch,
     action_onehot, reward)` — the TaskEncoder **does** step online, on real
     transitions; the freeze rule applies only to imagined ones.
- Keep uniform-random actions (existing `_random_action`) until
  `env_steps >= prefill_steps`, then switch to the policy. Store the true
  env mask for random steps too.
- **Accepted limitation (record, don't fix here):** online, `macro_context`
  accumulates over a whole episode (up to 600 steps), while training only
  ever builds it over `seq_len=16`-step windows — a train/act distribution
  mismatch. This is tickets/0002's known "within-window context" limitation
  surfacing at act time; the fix (longer-horizon context training / burn-in)
  is that ticket's declared follow-up, not scope here. Log
  `online/macro_context_norm` so drift is visible.

**Telemetry** (grad-step axis unless noted):

- `policy/actor_loss`, `policy/critic_loss`, `policy/entropy`,
  `policy/imagined_return`, `policy/intrinsic_reward_mean`,
  `policy/extrinsic_reward_mean`, `policy/value_mean`,
  `policy/return_norm_scale`, `policy/grad_norm_actor`,
  `policy/grad_norm_critic`.
- `online/action_type_frac/<type>` (env-step axis, windowed): the fraction of
  each action type the acting policy chooses. **Watch item:** RESET is always
  legal and restarts the game mid-episode — if its fraction climbs, the
  policy has found a novelty exploit (restart = cheap disagreement) and
  `intrinsic_scale`/entropy need attention.
- The existing `online/episode_*` and `wm/disagreement_*` scalars now become
  the actual readout of whether exploration works: expect episode returns to
  move off zero on *some* games, and per-game disagreement to *fall faster*
  than under random play (the policy consumes novelty).

**Docs:** update the module docstring (it currently says "the policy is
untouched (see tickets/0001)"), and CLAUDE.md's Training + Architecture
sections (policy now trains; new `critic`/`critic_target` components; the
collector acts with the policy after prefill).

### Step 7: Tests (`tests/`)

Use `small_config()` / conventions from `tests/conftest.py`; the suite must
stay fast and CPU-only. New coverage (plus updating whatever Step 2/3/5 broke
— run `uv run pytest` and fix all of it):

1. **`lambda_returns` correctness:** hand-compute a tiny case (N=1, H=2 or 3,
   nonuniform discounts) and assert exact values.
2. **Dream shapes & freeze rule:** `dream` from a handful of start states
   returns the documented shapes; assert `macro_context` passed in is
   bit-identical throughout (e.g. by checking the rollout never calls the
   TaskEncoder — simplest: dream with a nonzero context and monkeypatch
   `task_encoder.forward` to raise).
3. **Masking:** with a mask that forbids all but one action type, every
   `action_types` entry in the dream is that type.
4. **Gradient isolation:** after `actor_loss.backward()`, every
   `world_model` and `critic` parameter has `p.grad is None`; after
   `critic_loss.backward()`, every `world_model` and `policy` parameter has
   `p.grad is None`; `critic_target` never accumulates grads.
5. **Parameters move (and only the right ones):** one `policy_train_step` on
   synthetic data changes some `policy` and `critic` parameter, changes no
   `world_model` parameter, and leaves `critic_target` unchanged until
   `sync_critic_target()` makes it equal to `critic`.
6. **Buffer roundtrip with masks:** `available_actions` survives
   save/load with correct shape/dtype; loading a payload *without* the key
   yields all-True masks (backward compat).
7. **Trainer smoke:** extend `tests/test_training.py`'s synthetic iteration
   to cover a collect step where the policy (not `_random_action`) picks the
   action, plus checkpoint→resume restoring both new optimizer states and
   the return-normalizer scale.

---

## Checkpoint & buffer compatibility note

New parameters (`critic.*`, `critic_target.*`), a *removed* parameter
(`policy.value_head.*`), and two new optimizer states mean **pre-0003
`latest.pt` files are not resumable** (strict `load_state_dict`).
`--config.init-from` still works against them — it copies world-model weights
only, which is exactly the right warm start here: the first 0003 run should
`--config.init-from` the completed 0002 run's checkpoint so the policy trains
against a converged world model from step one. Old `buffer.pt` files load via
Step 5's all-True fallback. First run: fresh `--config.output-dir`, and add a
TRAINING_LOG.md entry (command + pre-registered expectations) per that file's
conventions before launching.

---

## Non-goals

- No separate exploration vs. exploitation policies (Plan2Explore trains
  two; we train one policy on the summed reward — splitting is a follow-up
  if the intrinsic term drowns extrinsic once scores appear).
- No fix for the within-window macro-context limitation (tickets/0002
  follow-up); this ticket only *logs* the online drift.
- No backprop-through-dynamics actor gradients, no search/MCTS, no
  prioritized replay, no ACTION7 reconciliation, no hyperparameter sweeps.
- No changes to the world-model objective itself (`compute_losses` is
  untouched).

---

## Verification & Acceptance Criteria

1. **`uv run pytest` fully green**, including all of Step 7.
2. **Isolation invariants hold** (test 4 above is the gate): actor/critic
   training cannot perturb the world model, and the dream pass allocates no
   autograd graph (assert no returned tensor from `dream` has
   `requires_grad`).
3. **Smoke run** (a few hundred grad steps past prefill, tiny budget, fresh
   output dir, ideally `--config.init-from` the 0002 checkpoint):
   `policy/actor_loss` and `policy/critic_loss` finite and moving,
   `policy/entropy` positive and not collapsed to ~0, `policy/imagined_return`
   nonzero (intrinsic reward guarantees this even with zero extrinsic),
   `online/action_type_frac/*` non-degenerate (no single action >95% while
   entropy_scale is at default), world-model `loss/recon` still falling as
   in prior runs (the policy update must not have disturbed it).
4. **The full training run is out of scope for this ticket** — it gets its
   own TRAINING_LOG.md entry with pre-registered expectations. The headline
   things that run should look for (recorded here so the log entry can cite
   them): buffer composition shifting (episode lengths moving off the
   600-cap on some games), per-game disagreement falling faster than Run 1's,
   and any nonzero `online/win_rate/*`.
