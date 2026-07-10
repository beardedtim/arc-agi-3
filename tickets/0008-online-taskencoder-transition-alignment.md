# 0008 — Align OnlineActor's TaskEncoder Folds with forward_sequence: Fix the Act-Time Transition Off-By-One

## Overview & Architectural Justification

`OnlineActor` (training/online_actor.py) claims to mirror
`WorldModel.forward_sequence`'s per-step ordering exactly — that claim is the
whole reason it exists (tickets/0007: one acting loop, so eval can never
measure a different agent than the one being trained). It doesn't. The
TaskEncoder fold is misaligned by one step relative to the convention the
TaskEncoder is actually *trained* under, and since the macro-context it
produces conditions the RSSM posterior, the misalignment also drags the
`(deter, stoch)` latents the online policy acts from off the trained
distribution.

**The two conventions, side by side.** The buffer's prev-action-at-t
convention means `action[t]`/`reward[t]` are the action/reward that
*arrived at* observation t. In `forward_sequence`'s burn-in loop, fold t is:

```
rssm.step(..., action[t], embed[t], m)  -> deter_t, stoch_t   # posterior at obs t
m = task_encoder(m, deter_t, stoch_t, action[t], reward[t])   # (arrival state, arriving action, arriving reward)
```

In `OnlineActor`, `act()` advances the RSSM onto frame t and *then* the
policy acts; the fold happens later, in `observe(a_t, r_{t+1}, frame_{t+1})`,
**before** the RSSM has advanced onto frame t+1:

```
m = task_encoder(m, deter_t, stoch_t, onehot(a_t), r_{t+1})   # (PRE-state, OUTGOING action, NEXT reward)
```

So training folds `(s_t, a_t, r_t)` tuples and acting folds
`(s_t, a_{t+1}, r_{t+1})` tuples — same states, action/reward shifted one
step. The TaskEncoder's only gradient path is the training one
(loss-window heads → frozen m → burn-in folds, tickets/0004), so it learns
the training convention; online it is fed inputs it has never been trained
to interpret. Additionally, training's fold 0 processes the episode's first
observation with is_first-zeroed action/reward
(`task_encoder(m0, deter_0, 0, 0)`), a step the online loop never performs
at all — its first fold already carries the first real action.

**Empirical proof** (script in the Appendix — RSSM sampling patched to
deterministic means so both paths are exactly comparable on a scripted
5-step episode, TaskEncoder inputs recorded via monkeypatch):

- Every fold's `(action, reward)` pair mismatches between the two paths
  (shifted by exactly one step); fold 0's mismatch is the missing zero-fold.
- The corrupted m feeds back through the posterior head: `deter` matches at
  folds 0–1, then drifts (max diff 7e-4 by fold 2 on an *untrained* toy
  model after just 2 misaligned folds).
- Final macro-context after 4 folds: max element diff 0.05 on a vector of
  norm ~0.5 — a ~10% relative deviation after 4 steps. Online episodes run
  to 600 steps.

**Why this matters.** This is the same failure class tickets/0004 fixed at
training time: a train/act distribution mismatch in the macro-context
pathway, which Run 2 proved can be quietly catastrophic for the one thing
the scalars can't see. Here the direction is reversed — training is
consistent (dreams, burn-in, and the actor-critic all use the
forward_sequence convention), and it's *acting* that's off-distribution:
every post-prefill step ever collected, and every eval episode ever
measured, ran through the misaligned loop. And because
`test_matches_inline_pre_refactor_logic` (tests/test_evaluate.py) uses a
hand-rolled copy of the *same buggy logic* as its reference, the suite
can't see it: both sides of the equivalence share the bug. The fix below
replaces that reference with the real ground truth, `forward_sequence`
itself.

**Read these first:**

- `training/online_actor.py` — the whole module; the fix restructures
  `act`/`observe`/`begin_episode`.
- `model/world_model.py::forward_sequence`, the burn-in loop — the
  convention being matched (step, then fold, both consuming index t).
- tickets/0004 — the training-side precedent for why macro-context
  train/act mismatches are treated as first-class bugs here.
- tickets/0003, "Online collection" step 4 — the spec that introduced the
  bug (it prescribes folding `(deter, stoch)` *before* the RSSM advances,
  i.e. the spec itself was wrong, not just the implementation; this ticket
  supersedes that step).
- tests/test_evaluate.py::TestOnlineActorRefactorEquivalence — the test
  this ticket replaces.

---

## Design & Core Principles

1. **One fold convention, stated once:** the TaskEncoder consumes completed
   transitions as `(arrival state deter/stoch, arriving action, arriving
   reward)` — the forward_sequence convention, because that is the only
   convention it is ever trained under. Acting conforms to training, not
   the other way around (retraining the TaskEncoder onto the acting
   convention would leave every existing checkpoint's TaskEncoder
   misaligned instead).
2. **Move the fold to where the arrival state exists.** Online, the arrival
   state for transition t→t+1 is only computed inside the *next* `act()`
   call (its `observe_step`). So `observe()` stops calling the TaskEncoder
   and instead stores the pending `(action_onehot, reward)`; `act()` folds
   the pending pair right after its `observe_step`, then lets the policy
   act on the post-fold m. Per-step ordering becomes exactly
   forward_sequence's: step → fold → (policy reads features).
3. **Episode start mirrors is_first.** `begin_episode` seeds the pending
   pair with a zeroed action onehot and 0.0 reward, so the first `act()`
   performs the same zero-fold on the first observation that training's
   burn-in performs at an is_first step. (Consequence: the *last*
   transition of an episode is never folded — there is no next `act()` —
   which is harmless: m dies with the episode at the next
   `begin_episode`.)
4. **No parameter, config, or checkpoint-format changes.** Same modules,
   same shapes; Run 6's `latest.pt`/`buffer.pt` stay loadable and
   resumable. The buffer stores raw transitions (convention-free), so
   pre-fix data remains fully valid training data.

---

## Implementation Tasks

### Step 1: `training/online_actor.py` — the fix

- `begin_episode`: additionally set
  `self._pending_action_onehot = torch.zeros(1, wm.config.action_dim, device)`
  and `self._pending_reward = torch.zeros(1, 1, device)`.
- `act()`: after `observe_step` produces the new `(deter, stoch)`, fold
  `self._macro_context = wm.task_encoder(self._macro_context, self._deter,
  self._stoch, self._pending_action_onehot, self._pending_reward)` —
  before building features for the policy. Comment the invariant: this is
  forward_sequence's step-then-fold ordering, arrival-state convention
  (tickets/0008).
- `observe()`: drop the TaskEncoder call; set the pending pair
  (`_pending_action_onehot = action_onehot`,
  `_pending_reward = tensor([[reward]])`) alongside the existing
  `_prev_action_onehot` and frame-stack append. (`_prev_action_onehot` and
  `_pending_action_onehot` hold the same value between calls but serve
  different consumers — RSSM input vs TaskEncoder input; keep them
  separate named fields so the two conventions stay legible.)
- Update the module/method docstrings: the current text says the
  TaskEncoder "steps online on real transitions" without stating the
  alignment; state the arrival-state convention explicitly.

### Step 2: Tests (`tests/`)

1. **Replace `test_matches_inline_pre_refactor_logic`** (its reference is a
   copy of the buggy pre-refactor logic and now disagrees with the fix)
   with the ground-truth equivalence test this ticket's Appendix script
   sketches: drive `OnlineActor` (greedy=True to avoid policy RNG) over a
   scripted episode with `RSSM._sample` monkeypatched to return the mean,
   record every TaskEncoder call's inputs via monkeypatch, run
   `forward_sequence` over the same episode (frames stacked the
   `ReplayBuffer._stack` way, `burn_in = T - 1`), and assert (a) per-fold
   `(deter, action, reward)` inputs match exactly between the two paths and
   (b) the final online macro-context equals the loss window's frozen
   `outputs["macro_context"][0, 0]`. This is the invariant 0007 wanted all
   along, anchored to the real training path instead of a hand copy.
2. **Zero-fold at episode start:** first `act()` after `begin_episode`
   folds a zeroed action/reward with the first observation's posterior
   state (assert via the recording monkeypatch).
3. **Bug-detection check (one-off, manual):** confirm the new test 1 fails
   against the pre-fix `OnlineActor` (e.g. stash the Step-1 change and run
   it) — evidence the new reference actually bites where the old one
   couldn't.
4. `test_trainer_collect_loop_unchanged_by_refactor` should pass unchanged
   (it compares trainer delegation against a standalone `OnlineActor`,
   both post-fix). All other existing tests must pass unchanged.

### Step 3: Docs

- CLAUDE.md's `training/online_actor.py` mention gains a clause: TaskEncoder
  folds use forward_sequence's arrival-state convention (tickets/0008).

### Step 4: Measure the impact — eval before/after on the same checkpoint

Run 6 (`runs/two_stream_returns`, in flight when this ticket was written)
must have **completed** first; use its final checkpoint. Same weights, same
protocol, same seed — the only variable is the acting loop:

```sh
# BEFORE applying Step 1 (pre-fix acting loop):
uv run python eval.py --checkpoint runs/two_stream_returns/latest.pt
# mv the written eval JSON aside (e.g. eval_<steps>_prefix.json), apply the
# fix, then AFTER:
uv run python eval.py --checkpoint runs/two_stream_returns/latest.pt
```

Append both summary tables to TRAINING_LOG.md Run 6's Findings, labeled
pre-fix/post-fix — this doubles as tickets/0007 Step 7's outstanding
adjudication of Run 6's behavioral criterion, now with the alignment fix's
effect isolated on top. Pre-register the reading: the post-fix table should
be no worse, and any improvement is a direct measure of how much the
misaligned m was costing the policy at act time. (A full 25-game × 5-episode
× 2-mode sweep is ~150k env interactions of pure acting — expect on the
order of an hour or two of wall-clock; run it after Run 6's process exits so
they don't contend for the device.)

### Step 5: Training Log

No new training run inside this ticket. The next run's entry (Run 7,
whatever its headline question) should note that it is the first run whose
*collection* uses the aligned acting loop, so its online scalars are not
strictly comparable to Runs 3–6's — the same caveat class Run 3 recorded
for the 0003 policy switch-on.

---

## Non-goals

- **The episode-length context-accumulation mismatch stays open.** Online,
  m still accumulates over up-to-600-step episodes while training builds it
  over 16-step burn-ins — the documented tickets/0002/0003 accepted
  limitation (`online/macro_context_norm` watches it). This ticket makes
  the per-step *convention* identical; the *horizon* mismatch is its own
  (already-logged) follow-up.
- **No TaskEncoder architecture or training-objective changes.**
- **No retraining decision.** Whether the aligned actor warrants resuming
  Run 6's dir vs a fresh run belongs to the next TRAINING_LOG entry's
  pre-registration, judged with Step 4's before/after tables in hand.

---

## Verification & Acceptance Criteria

1. **`uv run pytest` fully green**, including Step 2's replacement
   equivalence test — per-fold TaskEncoder inputs bit-match
   `forward_sequence`'s over a scripted episode, and the final online m
   equals the training path's frozen macro-context.
2. **Step 2 test 3 performed once:** the new equivalence test demonstrably
   fails on the pre-fix `OnlineActor`.
3. **No shape changes:** Run 6's `latest.pt` loads and resumes cleanly
   post-fix (covered by existing resume tests plus the Step 4 CLI runs).
4. **Step 4's before/after eval tables recorded** in Run 6's Findings, with
   the pre-registered reading stated before the post-fix table is produced.

---

## Appendix — diagnostic script (the proof recorded above)

```sh
PYTHONPATH=.:tests uv run python - <<'EOF'
import torch
from conftest import small_config, GRID, STACK
from model.thumper import Thumper
from model.rssm import RSSM
from training.online_actor import OnlineActor

torch.manual_seed(0)
thumper = Thumper(small_config())
wm = thumper.world_model
RSSM._sample = lambda self, stats: RSSM.dist_params(stats)[0]  # deterministic

T = 5
torch.manual_seed(42)
frames = [torch.randint(0, 16, (GRID, GRID)) for _ in range(T)]
actions = [0] + [1 + (t % 3) for t in range(1, T)]
rewards = [0.0] + [float(t % 2) for t in range(1, T)]

records, current = {"online": [], "training": []}, ["?"]
orig = type(wm.task_encoder).forward
def rec(self, m, deter, stoch, action, reward):
    records[current[0]].append((deter.detach().clone(), action.detach().clone(), reward.detach().clone()))
    return orig(self, m, deter, stoch, action, reward)
type(wm.task_encoder).forward = rec

current[0] = "online"
actor = OnlineActor(thumper, "cpu")
actor.begin_episode(frames[0])
for t in range(1, T):
    actor.act([0, 1, 2, 3], greedy=True)
    actor.observe(actions[t], (0, 0), rewards[t], frames[t])

current[0] = "training"
stacks = [torch.stack([frames[max(i, 0)] for i in range(t - STACK + 1, t + 1)]) for t in range(T)]
obs = torch.stack(stacks).unsqueeze(0)
at = torch.tensor([actions]); cd = torch.zeros(1, T, 2, dtype=torch.long)
fi = torch.tensor([[True] + [False] * (T - 1)]); rw = torch.tensor([rewards]).float()
with torch.no_grad():
    out = wm.forward_sequence(obs, at, cd, fi, rewards=rw, burn_in=T - 1)

for k in range(min(len(records["online"]), len(records["training"]))):
    (d_on, a_on, r_on), (d_tr, a_tr, r_tr) = records["online"][k], records["training"][k]
    print(f"fold {k}: action match={torch.equal(a_on, a_tr)} "
          f"reward match={torch.equal(r_on.squeeze(), r_tr.squeeze())} "
          f"deter diff={(d_on - d_tr).abs().max().item():.4f}")
m_on, m_tr = actor._macro_context, out["macro_context"][0, 0]
print(f"final m: max diff={(m_on - m_tr).abs().max().item():.4f} "
      f"(norms {m_on.norm():.4f} vs {m_tr.norm():.4f})")
EOF
```

Pre-fix output (July 10, 2026): every fold's action/reward mismatches
(one-step shift), deter drifts from fold 2 (7e-4 on an untrained toy
model), final m max-diff 0.05 on norm ~0.5. Post-fix, every line should
report exact matches and a ~0.0 final diff.
