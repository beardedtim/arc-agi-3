# 0002 - Implement History-Based Meta-RL Macro-Context (Slow-Fast Memory)

## Overview & Architectural Justification

Thumper is currently implementing a standard Dreamer-style Recurrent State-Space Model (RSSM) where a single GRU carries the deterministic state ($h_t$). Because ARC-AGI-3 requires solving unknown games via trial-and-error with sparse score feedback and no instructions or static demonstration pairs, a single GRU bottleneck experiences severe amnesia regarding task-level mechanics over long horizons.

To solve this, we will implement a **Hierarchical Slow-Fast Memory** (Meta-RL Macro-Context). We will introduce a history-based `TaskEncoder` module that ingests historical transitions `(deter, stoch, action, reward)` to iteratively build a slow-updating macro-context vector ($m$). This vector represents the agent's evolving "belief" about the current game's rules. We will condition the RSSM's prior/posterior heads and the Plan2Explore `TransitionEnsemble` on this macro-context, enabling disagreement-driven exploration to actively seek out game mechanics.

**Known limitation (accepted for this ticket):** during training, $m$ is re-initialized to zeros at the start of every sampled window, so the slow memory only ever accumulates over `seq_len` (default 16) steps. That makes $m$ a within-window context (RL²-style over short windows), not a persistent cross-episode belief. This is fine as a first cut — the plumbing is identical either way — and extending the effective horizon (longer `seq_len`, carried-over context, burn-in) is deliberately a follow-up ticket, not scope creep here.

---

## Architectural Rules & Core Principles

1. **Slow-Fast Division of Labor:**
   - **Fast Memory ($h_t, z_t$):** Handled by the existing GRU/RSSM trunk. Responsible for frame-to-frame spatial dynamics and immediate cursor positioning.
   - **Slow Memory ($m$):** Handled by the new `TaskEncoder`. Responsible for cross-step rule retention and task identification.
2. **The Imagination "Freeze" Rule:**
   - During training (`forward_sequence`), the `TaskEncoder` steps forward at every timestep $t$ using the completed transition to update $m$ for step $t+1$.
   - During latent planning/imagination (`imagine_from_first_frame`), **do not step the `TaskEncoder`**. Initialize the static belief $m$ at $t=0$ and pass that exact same frozen vector into `self.rssm.imagine_step(...)` for every step of the rollout. Game rules do not change during a short latent dream, and keeping $m$ frozen prevents self-predicted imagination errors from corrupting task belief.
3. **Gradient Isolation:**
   - When feeding `deter` and `stoch` into the `TaskEncoder` or when passing `macro_context` into the `TransitionEnsemble`, you must `.detach()` the tensors. The ensemble and task encoder must read the trunk's representation without distorting the underlying reconstruction and KL objectives.
4. **Ordering rule (no target leakage):** at step $t$, the RSSM and the reward/continue/internal-state heads consume the $m_t$ built from transitions up to $t-1$ only; the transition that lands on step $t$ (including `rewards[:, t]`) updates $m$ *after* step $t$'s outputs are produced. The task list below encodes this ordering — preserve it.

---

## Implementation Tasks

### Step 1: Create `model/task_encoder.py` (New Module)

Create a new file containing the configuration and module for the history-based macro-context encoder.

- **Implement `TaskEncoderConfig` (Dataclass):**
  - `context_dim: int = 128` — The hidden width of the macro-context vector ($m$).
  - `hidden_dim: int = 256` — Hidden width of the internal RNN/MLP layers.
- **Implement `TaskEncoder(nn.Module)`:**
  - **Input Width Calculation:** The module must accept a concatenated transition vector of size `deter_dim + stoch_dim + action_dim + 1` (where $+1$ accounts for the scalar reward). `action_dim` is `WorldModelConfig.action_dim` (135 at defaults: one-hot type ++ one-hot click x ++ one-hot click y).
  - **Architecture:** Use an `nn.GRUCell` (or a causal recurrent block) that maps `(prev_context, transition_input) -> next_context`. Add an MLP output projection if necessary to shape the output to `context_dim`.
  - **State Initialization:** Implement `def initial_state(self, batch_size: int, device: torch.device) -> Tensor:` that returns a zeroed tensor of shape `(batch_size, context_dim)`.

---

### Step 2: Update `model/rssm.py`

Modify the RSSM to condition its stochastic latent distributions on the macro-context.

- **Update `RSSMConfig`:**
  - Add `macro_context_dim: int = 128` (must match `TaskEncoderConfig.context_dim` — wire this in `WorldModelConfig.__post_init__`, see Step 3, rather than trusting the two defaults to agree by hand; that's this repo's config-invariant pattern).
- **Update `RSSM.__init__`:**
  - Increase the input dimension of the first `nn.Linear` in `self.prior_head` from `cfg.deter_dim` to `cfg.deter_dim + cfg.macro_context_dim`.
  - Increase the input dimension of the first `nn.Linear` in `self.posterior_head` from `cfg.deter_dim + cfg.embed_dim` to `cfg.deter_dim + cfg.embed_dim + cfg.macro_context_dim`.
- **Update State Methods (`observe_step`, `imagine_step`, `step`):**
  - Add `macro_context: Tensor` as a required positional argument to all three methods.
  - In `observe_step`: Concatenate `[deter, embed, macro_context]` along `dim=-1` before passing to `self.posterior_head`.
  - In `imagine_step`: Concatenate `[deter, macro_context]` along `dim=-1` before passing to `self.prior_head`.
  - In `step`: Apply the corresponding concatenations to both `self.prior_head` and `self.posterior_head`.
  - The GRU itself is unchanged — $m$ conditions the stochastic heads only, not the deterministic recurrence.

---

### Step 3: Update `model/world_model.py`

Wire the `TaskEncoder` into the sequence forward pass, features, loss calculations, and imagination rollouts.

- **Update `WorldModelConfig`:**
  - Import and add `task_encoder: TaskEncoderConfig = field(default_factory=TaskEncoderConfig)`.
  - In `__post_init__`, add the wiring invariant: `self.rssm.macro_context_dim = self.task_encoder.context_dim`.
- **Update `WorldModel.__init__`:**
  - Instantiate the encoder: `self.task_encoder = TaskEncoder(c.task_encoder, deter_dim=c.rssm.deter_dim, stoch_dim=c.rssm.stoch_dim, action_dim=c.action_dim)` (or pass sizes via config).
  - Update the downstream head feature sizing: Change `feature_dim = c.rssm.deter_dim + c.rssm.stoch_dim` to `feature_dim = c.rssm.deter_dim + c.rssm.stoch_dim + c.task_encoder.context_dim`. _(Note: Because `reward_head`, `continue_head`, and `internal_state_head` are built from `feature_dim`, they automatically adapt to the wider input. The `ImageDecoder` is deliberately **not** widened — it keeps reading `(deter, stoch)` only.)_
- **Update `TransitionEnsemble`:**
  - In `__init__`: Increase `in_dim` from `deter_dim + stoch_dim + action_dim` to `deter_dim + stoch_dim + action_dim + macro_context_dim` (pass the extra dim in from `WorldModel.__init__`).
  - In `forward` and `disagreement`: Add `macro_context: Tensor` as an input argument. Concatenate it alongside the other inputs: `x = torch.cat([deter, stoch, action, macro_context], dim=-1)`.
- **Update `WorldModel.disagreement`** (the thin wrapper around `self.ensemble.disagreement`): add `macro_context: Tensor` to its signature and pass it through. (The trainer calls this wrapper for telemetry — see Step 5.)
- **Update `WorldModel.features`:**
  - Change signature to `def features(self, deter: Tensor, stoch: Tensor, macro_context: Tensor) -> Tensor:`.
  - Return `torch.cat([deter, stoch, macro_context], dim=-1)`.
- **Update `WorldModel.forward_sequence`:**
  - Add optional `rewards: Tensor | None = None` to the method signature.
  - Add fallback handling at the start of the method:
    ```python
    if rewards is None:
        rewards = torch.zeros(B, T, 1, dtype=torch.float, device=observations.device)
    elif rewards.ndim == 2:
        rewards = rewards.unsqueeze(-1)
    ```
  - Mask placeholder rewards the same way placeholder actions already are: where `is_first` is True the buffer's reward is a placeholder, not a real transition — zero it before feeding the `TaskEncoder` (mirror the existing `action_onehot * (~is_first)` masking).
  - Initialize `macro_context` before the sequence loop: `macro_context = self.task_encoder.initial_state(B, device)`.
  - Inside the `for t in range(T):` loop:
    1. Pass `macro_context` into `self.rssm.step(deter, stoch, action_onehot[:, t], embed, macro_context)`.
    2. Store the current `macro_context` in a list (`macro_contexts.append(macro_context)`).
    3. Update `macro_context` for the next timestep $t+1$ by stepping `self.task_encoder` with detached trunk representations:
       ```python
       macro_context = self.task_encoder(
           macro_context,
           deter.detach(),
           stoch.detach(),
           action_onehot[:, t],
           rewards[:, t]
       )
       ```
    (This ordering — store $m_t$, *then* fold in transition $t$ — is what keeps `rewards[:, t]` out of the features the reward head uses to predict step $t$. Don't reorder it.)
  - Stack the stored context sequence: `macro_context_seq = torch.stack(macro_contexts, dim=1)`.
  - Add `"macro_context": macro_context_seq` to the returned `outputs` dictionary.
  - Update the call to `self.predict_heads(...)` to pass `self.features(deter_seq, stoch_seq, macro_context_seq)`.
- **Update `WorldModel.compute_losses`:**
  - In the disagreement ensemble block (`if T > 1:`): the ensemble head for the transition landing at $t+1$ must see the same context the RSSM prior sees at $t+1$, which is `macro_context[t+1]` (built from transitions up to $t$, available at decision time). So the slice is `macro_context_in = outputs["macro_context"][:, 1:].detach()` — **`[:, 1:]`, aligned with `action_onehot[:, 1:]` and the `post_mean[:, 1:]` target, not `[:, :-1]`**.
  - Pass `macro_context_in` into the ensemble call: `preds = self.ensemble(deter_in, stoch_in, action_in, macro_context_in)`.
- **Update `WorldModel.imagine_from_first_frame`:**
  - Initialize static belief at $t=0$: `macro_context = self.task_encoder.initial_state(B, device)`.
  - Pass `macro_context` into `self.rssm.observe_step(deter, stoch, prev_action, embed, macro_context)`.
  - Inside the imagination loop (`for t in range(T):`), pass the **same frozen `macro_context`** into `self.rssm.imagine_step(deter, stoch, action_onehot[:, t], macro_context)`. Do not step the `TaskEncoder` during imagination.

---

### Step 4: Update `model/thumper.py` (Policy wiring — do not skip)

`ThumperConfig.__post_init__` currently derives `policy.feature_dim = deter_dim + stoch_dim`, and `Thumper.features`/`Thumper.act` forward into `WorldModel.features`. Widening `features` without touching these produces a runtime shape mismatch in the policy's first `nn.Linear`.

- In `ThumperConfig.__post_init__`, change the invariant to `self.policy.feature_dim = deter_dim + stoch_dim + self.world_model.task_encoder.context_dim`.
- Change `Thumper.features` and `Thumper.act` signatures to accept and pass through `macro_context`. The policy is untrained in the current phase (tickets/0001), so this is pure plumbing — but it must compile and the invariant test must reflect it.

### Step 5: Update call sites in `training/` (do not skip)

- **`training/trainer.py` — `train_step`:** pass rewards into the sequence forward: `wm.forward_sequence(batch["observations"], batch["action_types"], batch["coords"], batch["is_first"], rewards=batch["rewards"])`. Without this the `TaskEncoder` trains on all-zero rewards and learns nothing reward-related.
- **`training/trainer.py` — `_log_train_scalars` disagreement telemetry:** the `wm.disagreement(...)` call must now also pass `outputs["macro_context"][:, 1:]` (same slice rationale as `compute_losses`).
- **`training/qualitative.py` — `save_recon_check`:** pass the batch's rewards slice into `forward_sequence` too, so the qualitative recon uses the same context the training pass does (the zero-reward fallback is for callers that genuinely have no rewards, not a license to skip them here). `save_imagination_check` needs no change beyond what Step 3 did to `imagine_from_first_frame`.

### Step 6: Update existing tests

These currently pass and will break; update them alongside the change (plus add the new checks in the acceptance criteria):

- `tests/test_thumper.py` — the `feature_dim == deter_dim + stoch_dim` invariant assertion.
- `tests/test_world_model.py::test_forward_and_disagreement_shapes` — `wm.disagreement(deter, stoch, action)` gains a `macro_context` argument.
- Anything in `tests/test_policy.py` that builds features from `policy.feature_dim` should still pass unmodified (it reads the config), but run the full suite (`uv run pytest`) to confirm.

---

## Checkpoint compatibility note

This change adds new parameters (`task_encoder.*`) and widens existing ones (RSSM heads, reward/continue/internal-state heads, ensemble). **Run 1's `runs/world_model/latest.pt` is not loadable** — neither auto-resume nor `--config.init-from` will work against it (strict `load_state_dict`). The first post-0002 training run must use a fresh `--config.output-dir` and train from scratch; add a TRAINING_LOG entry per the log's conventions, and expect to compare its `loss/recon` trajectory against Run 1's as the regression check (see acceptance criterion 4).

---

## Verification & Acceptance Criteria

1. **Shape Sanity Check:** Run a dummy batch through `forward_sequence` with random rewards and verify that `outputs["macro_context"]` has shape `(B, T, 128)` and that all loss terms compute without shape mismatches. Use the shrunken test config from `tests/conftest.py` (grid_size 16) rather than full 64×64 — that's the suite's convention.
2. **Gradient Isolation Check:** Verify by inspecting `.grad` attributes that a backward pass from `ensemble_loss` alone leaves `task_encoder` parameters with no grad, and that a backward pass through the `TaskEncoder` path leaves RSSM trunk (GRU/prior/posterior) parameters' grads unchanged by the detached inputs. (`torch.autograd.gradcheck` is overkill here — direct `.grad` inspection is fine.)
3. **Imagination Consistency:** Verify that calling `imagine_from_first_frame` executes successfully without requiring a reward tensor and maintains a static `macro_context` across all $T$ imagined steps.
4. **No regression vs Run 1:** `uv run pytest` fully green, and a short smoke training run (a few hundred grad steps in a fresh output dir) shows `loss/recon` falling from the start, as Run 1's did — the macro-context should be at worst neutral for reconstruction this early.
