# 0004 — Train/Dream-Consistent Macro-Context (fix the Run 2 imagination regression)

## Overview & Root Cause

Run 2 (TRAINING_LOG.md, `runs/meta_rl`, stopped at 19,500 grad steps) showed
that the tickets/0002 macro-context regressed imagination catastrophically
while leaving every trained scalar looking healthy or better: recon tracked
Run 1, smoothed `kl_raw` ran *lower* than Run 1's, yet at matched grad steps
on identical batches (both runs are seed-0 deterministic) dreams disintegrate
within 2–6 imagined frames — often morphing into *other games'* textures —
where Run 1's dreams stay on-game.

An A/B experiment against the Run 2 checkpoint isolated the cause. From the
same burned-in posterior start state and the same real action sequence,
dreams conditioned on a burned-in macro-context degrade about as badly as
dreams conditioned on zeros. So the problem is not *which* frozen context
imagination uses — the prior itself is weaker, whatever context it gets.

**Mechanism:** in `WorldModel.forward_sequence`, the prior at step $t$ is
conditioned on $m_t$, rebuilt *every step* from the real transitions up to
$t-1$ — including the (detached) posterior stoch samples, which carry
observation information. $m$ is therefore a second, teacher-forced memory
stream that always holds fresh ground-truth trajectory information at
training time. The prior learns to lean on it — that is exactly why
`kl_raw` beat Run 1's — and at dream time the frozen $m$ goes stale after
one step, taking the prior's apparent competence with it. Worse, because
tickets/0002 re-initializes $m$ to zero at every 16-step window, $m$
currently carries no cross-episode task belief at all: it is pure *fast*
memory duplicating the GRU with privileged access, i.e. the opposite of the
slow-memory design intent.

**The fix principle:** the RSSM prior (and posterior — they must share
conditioning for the KL to be meaningful) may only ever be conditioned on a
macro-context that is **constant over the whole training window** and built
from real transitions *preceding* that window. Then "frozen $m$" at dream
time is exactly what training saw, the teacher-forcing crutch is
structurally impossible, and $m$ is forced into its intended role: a slow,
window-scale belief, not a per-step side channel.

This blocks everything downstream: tickets/0003 is implemented and
smoke-tested, but its actor-critic trains entirely inside `Thumper.dream` —
launching its full run against a prior that hallucinates cross-game mush
would optimize the policy against noise.

**Read these first:**

- TRAINING_LOG.md Run 2's Findings — the evidence this ticket answers.
- `model/world_model.py::forward_sequence` — the per-step
  `self.task_encoder(...)` update inside the `for t in range(T)` loop is
  what this ticket removes from the loss window; also
  `imagine_from_first_frame` (to be replaced) and `compute_losses` (window
  alignment of the ensemble slice).
- `model/task_encoder.py` — unchanged parameters, changed call pattern.
- `model/thumper.py::dream` — the 0003 consumer; note it already receives a
  per-start-state `macro_context` and holds it frozen, which becomes
  consistent-by-construction after this change.
- `training/trainer.py` — `train_step` / `policy_train_step` (dream starts
  are flattened `outputs["deter"|"stoch"|"macro_context"]`), the online
  collection loop (steps the TaskEncoder per real step), and
  `TrainerConfig` (`seq_len`, sampling).
- `training/qualitative.py::save_imagination_check` — currently dreams from
  a cold start with $m=0$; gets the burn-in treatment too.
- `training/replay_buffer.py::sample` — window sampling this ticket extends.

---

## Design: burn-in prefix + within-window freeze

Training windows grow a **burn-in prefix**: sample `burn_in + seq_len`
contiguous steps instead of `seq_len`. The first `burn_in` steps are
consumed to build the window's macro-context (and warm the RSSM state); the
remaining `seq_len` steps are the loss window, conditioned on that $m$
**held frozen throughout**. The TaskEncoder is never stepped inside the
loss window.

- Within-window freeze ⇒ prior conditioning at training time is identical
  to dream time (a constant, real-data $m$). The Run 2 pathology cannot
  recur by construction.
- Burn-in from real preceding steps ⇒ $m$ is nonzero and informative (task
  identity), which is the part 0002 never delivered (its $m$ was zero at
  every window start). This also directly upgrades 0002's "within-window
  context" limitation to a "window-scale belief from the last
  `burn_in` real steps".
- Side benefit: the loss window starts from a warmed RSSM state instead of
  the artificial zero `initial_state`, removing the cold-start transient
  from every training sequence.

**Fallback, pre-registered (implement only if acceptance criterion 4
fails):** if burn-in + freeze does not restore Run 1-level imagination,
strip `macro_context` from the RSSM prior/posterior conditioning entirely
(revert `model/rssm.py` head input widths; keep $m$ in
`WorldModel.features`, the decoder-adjacent heads, the ensemble, and the
policy/critic). That is the empirically safe configuration — Run 1's
dynamics exactly, with $m$ still informing reward/continue/value/action.
Decide by the criterion, not by taste; do not implement both speculatively.

---

## Implementation Tasks

### Step 1: `model/world_model.py` — burn-in in `forward_sequence`

- Add `burn_in: int = 0` to the signature. Callers pass full-length tensors
  (`T_total = burn_in + T`); the method consumes the first `burn_in` steps
  internally and **returns outputs only for the trailing `T` steps**, so
  every downstream consumer (`compute_losses`, dream starts, qualitative)
  keeps its existing `(B, T, ...)` shape expectations.
- Burn-in phase (steps `0..burn_in-1`): step the RSSM posterior and the
  TaskEncoder exactly as the loop does today (same is_first masking of
  placeholder actions/rewards), but store nothing and compute no decoder
  output — `self.decoder(...)` per burn-in step is pure waste; skip it.
- At the window boundary: `.detach()` `deter` and `stoch` before entering
  the loss window, so BPTT length through the trunk stays `seq_len`
  (unchanged compute/credit-assignment versus today, isolating this
  ticket's effect to the context change). Do **not** detach
  `macro_context`: the gradient path
  loss-window heads → frozen $m$ → burn-in TaskEncoder steps is the *only*
  way the TaskEncoder now trains. (Its inputs remain detached trunk
  tensors, as 0002 required — the trunk still can't be shaped through it.)
- Loss-window phase (the remaining `T` steps): the existing loop **minus**
  the `self.task_encoder(...)` update — every step's `rssm.step(...)`,
  heads, and stored `macro_contexts.append(...)` use the same frozen $m$.
  The returned `outputs["macro_context"]` is thus constant along the time
  axis per row; keep returning it (features, dream starts, and telemetry
  all index it).
- `compute_losses` needs no change: it already consumes `outputs` +
  `target_observations` of matching length — but its callers must now pass
  the **sliced** targets/batch (`[:, burn_in:]`), see Step 3. The ensemble
  slice comment ("macro_context at t+1 = what the prior saw") remains
  correct — it's just constant now.
- With `burn_in=0` the behavior must be bit-identical to today (that's the
  tests' compatibility anchor).

### Step 2: `model/world_model.py` — replace `imagine_from_first_frame`

Delete it (its zero-context cold start is exactly the misleading
measurement Run 2 exposed) and add:

```python
@torch.no_grad()
def imagine_with_burn_in(
    self,
    observations: Tensor,   # (B, burn_in + 1, K, H, W) real frame stacks
    action_types: Tensor,   # (B, burn_in + 1 + horizon) buffer convention
    coords: Tensor | None,
    is_first: Tensor | None,
    rewards: Tensor | None,
    burn_in: int,
    horizon: int,
) -> Tensor:
```

Posterior-step (and TaskEncoder-step) through the first `burn_in + 1` real
frames, then freeze $m$ and roll `rssm.imagine_step` for `horizon` steps
using the real action slice (same prev-action-at-t indexing notes as the
old docstring). Return decoded logits `(B, horizon + 1, ...)`: the last
burned-in frame's posterior recon first, then the imagined frames — same
render contract the qualitative check builds rows from.

### Step 3: `training/trainer.py` — thread `burn_in` through

- **`TrainerConfig`:** add `burn_in: int = 16` ("real steps consumed to
  build each window's frozen macro-context and warm the RSSM state; 0
  reproduces the pre-0004 behavior"). `seq_len` keeps its meaning (loss
  window length); the buffer sample length becomes `burn_in + seq_len`.
- **`train_step`:** sample `c.burn_in + c.seq_len`; call
  `forward_sequence(..., burn_in=c.burn_in)`; pass the *sliced* batch to
  `compute_losses` (`target_observations=batch["observations"][:, c.burn_in:]`
  and a batch dict with every per-step field sliced `[:, c.burn_in:]` —
  slice once into a `window_batch` rather than at each use). The dream
  starts in `policy_train_step` flatten the returned (already
  window-only) outputs unchanged — including `available_actions`, which
  must come from the sliced window too.
- **Episode-length note:** windows now need `burn_in + seq_len = 32`
  contiguous steps; the buffer's existing repeat-padding handles shorter
  episodes, but a padded burn-in is mostly-repeated frames (a weak but
  harmless context). No change required — just don't "fix" the padding.
- **Online collection:** unchanged this ticket — the collector keeps
  stepping the TaskEncoder every real step, so its $m$ horizon (whole
  episode) still exceeds training's (`burn_in`). This is a *milder*
  mismatch than before (nonzero-vs-zero has become long-vs-short) and
  `online/macro_context_norm` already tracks it; carrying training context
  across window boundaries / matching horizons stays the 0002 follow-up,
  now much smaller.

### Step 4: `training/qualitative.py` — burn-in the imagination check

`save_imagination_check` calls `imagine_with_burn_in` with the batch's
leading `burn_in` steps (cap `burn_in + horizon + 1` at the batch's
`seq_len`+`burn_in` total; the trainer passes `c.burn_in` in). Render real
frames `burn_in .. burn_in + horizon` on top, the imagined row beneath —
so the PNG now answers "given a warmed-up belief, does the prior hold?",
which is the question 0003's dreams actually pose. `save_recon_check` just
needs the sliced window (pass the full batch and `burn_in`, or have the
trainer hand it the pre-sliced `window_batch` — pick one and be consistent
with `train_step`).

### Step 5: Tests

Update signature-breakage fallout (`uv run pytest`, fix everything), plus
new coverage — use `small_config()` conventions:

1. **Freeze invariant:** `forward_sequence(..., burn_in=4)` returns
   `macro_context` constant along the time axis (exactly equal, every
   step), and *nonzero* when the burn-in contains real transitions.
2. **Backward compatibility:** `burn_in=0` output matches the pre-change
   path (same shapes; with fixed seeds, same values as a directly-computed
   reference within the test).
3. **Window exclusion:** with `burn_in=k`, corrupting the burn-in frames
   changes `outputs["macro_context"]` but the loss window length stays
   `T - k` and `compute_losses` runs on sliced targets without shape
   errors.
4. **TaskEncoder still trains:** backward from a loss-window head loss
   (e.g. reward MSE on the returned outputs) leaves nonzero grads on
   `task_encoder` parameters (the burn-in gradient path survives the
   boundary detach), while trunk grads do not flow through the boundary
   (deter/stoch detached: no grad reaches burn-in-step GRU computation
   from the recon loss).
5. **`imagine_with_burn_in`:** shape check, runs without rewards
   (`None` fallback), and the frozen-$m$ property (monkeypatch
   `task_encoder.forward` to raise after the burn-in phase, mirroring the
   0003 dream test).
6. **Trainer smoke** (extend `tests/test_training.py`): one synthetic
   train_step + policy_train_step at `burn_in=2` end-to-end.

### Step 6: Docs

- CLAUDE.md architecture bullet for `model/task_encoder.py`: replace the
  "re-initialized to zero at the start of every training window" /
  known-limitation sentence with the burn-in + within-window-freeze
  behavior, referencing this ticket.
- `model/task_encoder.py` and `forward_sequence` docstrings: same update
  (the "stepped once per training timestep" claim becomes wrong).
- tickets/0002: no edit needed (tickets are historical records), but this
  ticket supersedes its "Imagination Freeze Rule" framing: freezing is now
  a training-time property, not an imagination-only exception.

---

## Checkpoint compatibility note

**No parameters are added, removed, or resized** — this is a data-flow
change only, so post-0003 checkpoints still `load_state_dict` cleanly.
That does *not* make Run 2's weights usable: they *are* the pathology
(a prior trained against per-step fresh context). The validation run must
train from scratch in a fresh `--config.output-dir`. `runs/meta_rl` is
deliberately kept as the "before" side of the comparison.

## Non-goals

- No persistent cross-episode / cross-window context carry-over (the
  remaining, now much smaller, 0002 follow-up).
- No online-collection changes beyond what Step 3 notes.
- No KL/`free_nats`/`kl_weight` retuning — watch `kl_raw` in the
  validation run (it should *rise* toward Run 1's level as the crutch
  disappears; that is expected recovery, not a regression) and ticket any
  tuning separately.
- Do not implement the fallback (prior de-conditioning) unless acceptance
  criterion 4 fails.

## Verification & Acceptance Criteria

1. `uv run pytest` fully green, including all of Step 5.
2. `forward_sequence(..., burn_in=0)` reproduces pre-0004 behavior
   (criterion for "this ticket changed one thing").
3. **Repeat Run 2's A/B experiment** on the new architecture's validation
   checkpoint (script preserved at the bottom of this ticket): dreams from
   a burned-in start state with the burned-in $m$ must now hold structure
   on dynamic games for most of an 8-step horizon, and must beat the
   zero-context variant (with the crutch gone, an informative $m$ should
   finally *help*).
4. **The decisive check — a Run 2 rerun at the same budget it was stopped
   at:** train from scratch (fresh output dir, defaults + `burn_in=16`) to
   ~20k grad steps and compare `imagine_step_*.png` at 9k/11k/14k/17k
   against both `runs/world_model` (Run 1, the bar to meet) and
   `runs/meta_rl` (the failure to beat). Seed-0 determinism makes these
   frame-for-frame comparable. Pass = matched-step dreams on par with
   Run 1's (on-game, degrading gracefully), with recon within ~20% of
   Run 1's matched-step values and `train/grad_norm` back near Run 1's
   band. Add the TRAINING_LOG entry (pre-registered, per its conventions)
   before launching; if it passes, the same entry's decision line should
   green-light the tickets/0003 full run.

---

## Appendix: the Run 2 A/B script (for criterion 3)

Adapted from the investigation (original lived in the session scratchpad).
Burn in over the first 8 real steps of sampled sequences, then dream the
remaining 8 with (a) the burned-in $m$, (b) $m=0$, from the same start
state, rendering real/burned/zero rows per sequence. Under the new
architecture use `forward_sequence(..., burn_in=8)` for the burn-in phase
and take any timestep of the returned constant `macro_context`.

```python
import torch
from model.world_model import WorldModel
from training.qualitative import grid_to_image, _row_of_frames, _stack_rows
from training.replay_buffer import ReplayBuffer

BURN, SEQ = 8, 16
payload = torch.load("runs/<run>/latest.pt", map_location="cpu", weights_only=False)
wm = WorldModel(payload["config"].world_model)
wm.load_state_dict({k.removeprefix("world_model."): v
                    for k, v in payload["state_dict"].items() if k.startswith("world_model.")})
wm.eval()
buffer = ReplayBuffer.load("runs/<run>/buffer.pt", frame_stack=wm.config.frame_stack,
                           internal_state_dim=wm.config.internal_state_dim)
batch = buffer.sample(64, SEQ)
settled = batch["observations"][:, :, -1]
dynamism = (settled[:, 1:] != settled[:, :-1]).float().mean(dim=(1, 2, 3))
for rank, i in enumerate(dynamism.argsort(descending=True)[:3].tolist()):
    o, at = batch["observations"][i:i+1], batch["action_types"][i:i+1]
    cd, fi, rw = batch["coords"][i:i+1], batch["is_first"][i:i+1], batch["rewards"][i:i+1]
    with torch.no_grad():
        out = wm.forward_sequence(o[:, :BURN], at[:, :BURN], cd[:, :BURN], fi[:, :BURN], rewards=rw[:, :BURN])
        deter0, stoch0, m = out["deter"][:, -1], out["stoch"][:, -1], out["macro_context"][:, -1]
        actions = wm.encode_actions(at[:, BURN:], cd[:, BURN:])
        def dream(m):
            d, s, frames = deter0, stoch0, []
            for t in range(SEQ - BURN):
                d, s = wm.rssm.imagine_step(d, s, actions[:, t], m)
                frames.append(wm.decoder(d, s).argmax(dim=1)[0])
            return frames
        real = wm.preprocess(o[0, BURN:])[:, -1]
        rows = [_row_of_frames([grid_to_image(f) for f in fs])
                for fs in (list(real), dream(m), dream(torch.zeros_like(m)))]
    _stack_rows(rows).save(f"context_ab_{rank}.png")  # rows: real | burned-in m | m=0
```
