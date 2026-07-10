# ARCH.md — How Thumper Works

A living reference for the Thumper agent: what each module is, what it does, why it
exists, and how the training loop assembles them into an ARC-AGI-3 player. Written for
a reader fluent in deep learning but new to Dreamer-style model-based RL — each
borrowed concept gets a short primer before the codebase-specific detail. Update this
document as the architecture evolves (tickets/ holds the design history; this holds
the current state).

---

## 1. The problem Thumper is built for

[ARC-AGI-3](https://arcprize.org/arc-agi/3) is a benchmark of small video games played
through a minimal API. Every game presents the same interface:

- **Observation**: a grid of up to 64×64 cells, each an integer color 0–15. One action
  may return several animation frames.
- **Actions**: `RESET`, five abstract buttons `ACTION1–5`, and `ACTION6(x, y)` — a
  click at a grid coordinate. Each state advertises which actions are legal
  (`available_actions`).
- **Reward**: none, explicitly. The only progress signal is `levels_completed`
  (score 0–254), which ticks up on rare events like finishing a level. There are no
  instructions, no goal description, no reward shaping.

That combination dictates everything about the design:

1. **No instructions** → the agent must *discover* each game's mechanics by
   experimenting. Exploration cannot be random (random play essentially never scores);
   it must be directed at what the agent doesn't yet understand.
2. **Sparse, near-absent reward** → almost all learning signal has to come from
   somewhere other than reward: predicting the world itself.
3. **Many different games, one agent** → the agent needs a way to infer *which rules
   are currently in play* and condition its predictions and behavior on that belief,
   rather than averaging all games into mush.
4. **Interaction is expensive** (a real API, ~5–6 steps/s in our offline setup) →
   squeeze maximal learning out of every real step, which favors replaying experience
   and training the policy in *imagination* rather than on live rollouts.

**Thumper** answers these with three ideas stacked together: a **Dreamer-style world
model** (learn to predict the environment, train the policy inside the learned model),
**Plan2Explore intrinsic motivation** (explore where an ensemble of dynamics models
disagrees), and a **slow/fast memory split** (a per-episode "task belief" vector that
tells the world model which game's rules to apply).

---

## 2. The three borrowed ideas, briefly

**World models / Dreamer** (Hafner et al.). Instead of learning a policy directly from
pixels-to-actions, learn a *model of the environment*: compress each observation into
a latent state, and learn latent dynamics — "given my state and an action, what state
comes next, and what reward/termination arrives?" Once that model is decent, the
policy can be trained on **imagined rollouts**: start from a real state, let the
policy pick actions, and step the *learned dynamics* instead of the real game. This
converts a handful of real steps into unlimited cheap training experience, which is
exactly what an expensive, sparse-reward environment demands. The latent dynamics
model is an **RSSM** (Recurrent State-Space Model): a deterministic recurrent path (a
GRU, which carries reliable memory) plus a stochastic latent (which forces the model
to represent genuine uncertainty about what happens next, rather than blurrily
averaging outcomes).

**Plan2Explore** (Sekar et al.). With no reward to chase, what should the policy *do*?
Train an ensemble of K small dynamics predictors on the same data from independent
initializations. Where the world is well-understood they converge and agree; where the
agent has little experience they disagree. The **variance across the ensemble's
predictions** is an intrinsic reward: "go do the thing you can't predict yet." This
turns exploration from uniform dithering into targeted experimentation — clicking the
unclicked object, entering the unentered room — which is how a game's mechanics get
discovered without instructions.

**Task inference / slow-fast memory** (in the spirit of meta-RL methods like RL²/VariBAD).
One network playing 20+ games has a problem the single-game Dreamer never had: the
same observation can mean different things under different rules. So Thumper carries
two memories at different timescales. The RSSM's GRU is the *fast* memory — frame-to-
frame dynamics, resets every episode. The **TaskEncoder** is the *slow* memory: it
folds the episode's history of transitions into a **macro-context vector `m`** — the
agent's evolving belief about *which game and which rules* it is in — and everything
that predicts the future (the RSSM's stochastic heads, the ensemble, the reward/
continue heads, the policy, the critic) is conditioned on `m`.

---

## 3. The model (`model/`)

`Thumper` (`model/thumper.py`) is one `nn.Module` owning every component, so there is
exactly one thing to checkpoint, size, move to a device, and optimize. Data flow:

```
grid frames ──► Vision ──► RSSM (conditioned on TaskEncoder's macro-context m)
                              │
                              ▼
              features = deter ++ stoch ++ m          (256 + 32 + 128 = 416)
                              │
        ┌───────────┬─────────┼──────────────┬────────────┐
        ▼           ▼         ▼              ▼            ▼
   ImageDecoder  reward/continue/       Policy         Critic       Ensemble
   (recon)       internal-state heads   (actor)     (ext + int)   (disagreement)
```

### 3.1 `model/actions.py` — the interface contract

Shared constants that pin the model to the ARC-AGI-3 API: 64×64 max grid,
`NUM_SYMBOLS = 17` (16 colors + a dedicated `PAD_SYMBOL` for cells outside a
smaller-than-64×64 playfield — a separate vocabulary entry so the model can
distinguish "outside the board" from "black cell"), and 7 action types
(`RESET`, `ACTION1–5`, `ACTION6` with a click coordinate).

### 3.2 `model/vision.py` — Vision (the encoder)

**What**: a stack of the K=4 most recent grids → one flat 256-dim feature vector.

**How**: each cell's symbol is looked up in a learned 16-dim embedding (cells are
*categorical*, not intensities — treating color 7 as "brighter than" color 3 would be
meaningless), the K frames are concatenated channel-wise (4×16 = 64 channels at
64×64), then a DreamerV3-style stack of 4 strided convolutions halves resolution and
doubles width each layer (32→64→128→256 channels, 64→4 spatial), and a final linear
projects the flattened map to `out_dim = 256`.

**Why frame stacking**: game dynamics — movement, causality, mid-animation state —
are only visible *across* time, and the API can return several animation frames for a
single action. The stack (oldest first, settled frame last) exposes that motion to the
encoder in one shot.

### 3.3 `model/rssm.py` — RSSM (latent dynamics) + ImageDecoder

**RSSM state** = `(deter, stoch)`:

- `deter` (256): a GRU hidden state. The GRU consumes `(prev_stoch ++ prev_action)`
  each step — the deterministic backbone that carries memory reliably through time.
- `stoch` (32): a diagonal-Gaussian latent sampled fresh each step. Two heads emit its
  distribution parameters:
  - the **prior** predicts `stoch` from `(deter, m)` alone — no observation. This is
    the head that *imagination* runs on: it is the model's guess about the future.
  - the **posterior** predicts `stoch` from `(deter, obs embedding, m)` — it gets to
    peek at the real frame. This is the "filtered" state used whenever real
    observations exist.

Training pulls the prior toward the posterior (see §4.5's KL term), which is what
makes imagined futures behave like filtered reality. Both heads — but deliberately
*not* the GRU recurrence — are conditioned on the macro-context `m`: the belief about
which game's rules apply shapes *what happens next*, while the mechanical carrying of
state is game-agnostic.

**ImageDecoder**: maps `(deter, stoch)` → per-cell **logits over the 17 symbols** at
64×64, via a linear to a 4×4×256 seed then repeated (2× nearest-upsample + 3×3 conv)
stages. Reconstruction is per-cell 17-way classification (cross-entropy), not MSE —
the right objective for categorical cells. The decoder deliberately does **not** see
`m`: reconstruction of the current frame shouldn't lean on the task belief, keeping
`m`'s gradient pressure focused on *dynamics* rather than appearance.

### 3.4 `model/task_encoder.py` — TaskEncoder (the slow memory)

**What**: a GRUCell that folds one completed transition
`(deter, stoch, action, reward)` at a time into the macro-context `m` (128-dim).

**Why**: `m` is the agent's answer to "what game am I in and how does it work?" It is
the only channel through which cross-game identity information reaches the dynamics
model, so it is what allows one set of weights to model 20+ different rule systems
without averaging them together.

**Two rules that make it work** (learned the hard way — see tickets/0002 and 0004):

1. **The trunk is detached.** The TaskEncoder reads `deter`/`stoch` but its loss
   gradients never flow back into them — the world model's reconstruction/KL objective
   must not be warped by the task-inference objective riding on top.
2. **The freeze rule.** `m` is *never* stepped inside an imagined rollout, and — since
   tickets/0004 — never stepped inside a training loss window either. Each training
   window builds `m` from a `burn_in` prefix of real steps, then holds it **frozen**
   for the whole loss window, exactly matching what a dream sees (a frozen `m` from
   its start state). The earlier design stepped `m` every training timestep, which let
   the prior lean on fresh per-step context that goes stale after one imagined step —
   the root cause of a full imagination collapse (TRAINING_LOG Run 2). The rationale:
   a game's rules don't change mid-dream, and stepping the encoder on self-predicted
   transitions would corrupt the belief with hallucinated evidence.

Online (real play), `m` **does** step every real transition — the belief should sharpen
as the episode reveals more of the game. Note the residual train/act mismatch: online
`m` accumulates over a whole episode, training only ever builds it from a 16-step
burn-in (logged as `online/macro_context_norm`; open follow-up from tickets/0002).

### 3.5 `model/world_model.py` — WorldModel (the assembly) 

Owns Vision + RSSM + TaskEncoder + ImageDecoder plus:

- **Three MLP heads** on the 416-dim feature `(deter ++ stoch ++ m)`:
  - `reward_head` — predicts the score *delta* arriving at a state. This is what
    imagination pays the policy with, so a sparse real signal must survive into it
    (see the weighted loss in §4.5).
  - `continue_head` — predicts (as a logit) "the episode did not end here". It gates
    imagined rollouts: discount chains multiply by its probability, so imagined
    trajectories fade past predicted deaths.
  - `internal_state_head` — auxiliary regression on the normalized cumulative score
    (`levels_completed / 254`). Pure representation shaping: it forces "how far along
    am I" to be decodable from the latent, which reward deltas alone don't guarantee.
- **Action encoding** (`encode_actions`): one-hot type (7) ++ one-hot click x (64) ++
  one-hot click y (64) = **135 dims**, coordinate one-hots zeroed unless the type is
  ACTION6. One-hots rather than two normalized scalars so the dynamics model gets a
  spatially crisp signal about *where* a click landed.
- **The Plan2Explore ensemble** (`TransitionEnsemble`): K=5 independently initialized
  MLPs, each predicting the next posterior `stoch` *mean* from
  `(deter, stoch, action, m)`. Variance across heads = intrinsic reward. Inputs and
  targets are detached — the ensemble reads the trunk without shaping it. Conditioning
  on `m` makes "what's unknown" *task-relative*: mastered mechanics in one game stay
  novel in another. Implemented with stacked parameters + `torch.vmap`, so K heads
  cost K small-MLP FLOPs, not K kernel launches.
- **`forward_sequence`** — the training-time unroll (posterior teacher-forcing) with
  the burn-in structure described in §4.5.
- **`compute_losses`** — the full world-model objective (§4.5).
- **`imagine_with_burn_in`** — a no-grad qualitative check: reconstruct one real frame
  then roll the prior forward, decoding each imagined state to a PNG (the
  `samples/imagine_*.png` images that catch imagination collapse by eye).

### 3.6 `model/policy.py` — Policy (the actor)

**What**: world-model features → a factored action distribution:
- a 7-way **type head** (RESET + ACTION1–5 + ACTION6), and
- a 64×64 **pointer head** for the click coordinate, used only when ACTION6 is chosen.

The pointer is decoded from an 8×8 spatial seed upsampled to 64×64 (rather than one
flat linear to 4096 logits) so nearby cells share features — clicks target *objects*,
and objects are spatially coherent.

Both heads are masked by the API's `available_actions` (illegal types get −inf logits)
before sampling, so the policy never spends probability mass on actions the game
rejects. The joint log-prob is `log p(type) + [type == ACTION6] · log p(x, y)`;
entropy composes the same way. `act()` samples (or argmaxes, for greedy eval);
`log_prob_entropy()` re-evaluates stored actions for the policy-gradient update.

**Why REINFORCE, not backprop-through-dynamics**: the action space is categorical
(you can't differentiate through a sampled click), so the actor trains from
REINFORCE with a learned critic baseline rather than Dreamer's pathwise gradients.

### 3.7 `model/critic.py` — Critic (two value estimators) + target

Two standalone MLPs over the same 416-dim feature: `ext_net` estimates expected
discounted *extrinsic* (score) return, `int_net` the *intrinsic* (disagreement)
return. Three deliberate non-sharings:

- **Not a head on the policy trunk** — critic gradients must not shape actor features.
- **No shared trunk between the two streams** — the intrinsic value is deliberately
  nonstationary (disagreement decays as the world model improves) and its drift must
  not perturb the sparse, stationary extrinsic estimate through shared hidden layers.
- **A frozen `critic_target` copy** (hard-synced every 100 grad steps) provides the
  bootstrap values for return computation, the standard trick that stops the critic
  chasing its own moving predictions.

### 3.8 Config wiring

Every component has a dataclass config; cross-component invariants are **derived in
`__post_init__` hooks, never set by hand**: `RSSM.embed_dim = Vision.out_dim`,
`RSSM.macro_context_dim = TaskEncoder.context_dim`, policy/critic
`feature_dim = deter + stoch + context`, and one shared action space. Preserve this
pattern when adding config fields — three configs "agreeing by hand" is how silent
shape bugs happen.

---

## 4. The training loop (`training/`, `train.py`)

One process interleaves **collection** (playing real games into a replay buffer) and
**training** (world-model + actor-critic updates from buffer samples) at a fixed ratio:
one grad step per `train_every = 2` env steps, after a `prefill_steps = 1000` random
warmup. Defaults: batch 16 windows × (16 burn-in + 16 loss) steps, world-model LR
3e-4, actor/critic LR 8e-5, all Adam, grad-clip 100.

### 4.1 `env/env.py` — Env

Thin wrapper over `arc_agi.Arcade` in OFFLINE mode (games live in
`environment_files/`). Speaks tensors/ints outward: frames as `(H, W)` int64 grids,
**reward = the `levels_completed` delta** since the previous step (so it's 0 almost
always and +1 on a level completion), `done` = WIN or GAME_OVER, plus the raw
cumulative score and the state's legal-action list (RESET is always appended — the
engine only ever names ACTION1–7, and ACTION7 is outside the model's action space and
simply never used).

### 4.2 `training/replay_buffer.py` — ReplayBuffer

Stores raw per-step frames per episode (not pre-stacked; stacks are assembled at
sample time) with FIFO eviction of whole episodes past 200k total steps; persists
to/from `buffer.pt` (frames packed to uint8, ~8× smaller).

**The step convention everything else relies on**: step *t*'s `action_type`/`coords`/
`reward` are the ones that **produced** `frame[t]` (prev-action-at-t / arrival-state
convention). An episode's first step therefore has placeholder action/reward and
`is_first[0] = True`; every consumer masks those placeholders rather than trusting
them.

**Terminated vs truncated**: the buffer only ever stores a *real* death
(`result.done`) as `terminated`. The trainer's 600-step cap rotates the episode
locally but is truncation, not termination — it never reaches the continue head, which
must not learn "the world ends at step 600".

**Reward-stratified sampling** (tickets/0005): scoring steps are vanishingly rare
(~0.015% of steps), so plain uniform window sampling dilutes them as the buffer grows.
`sample(reward_frac=0.25, loss_offset=burn_in)` aims a quarter of each batch at
windows whose *loss window* contains a nonzero-reward step (event positions derived
from `Episode.rewards` at sample time, uniformly placed inside the window). A target,
not a promise — it degrades to uniform until the buffer contains any scoring event.

### 4.3 `training/online_actor.py` — OnlineActor (the one real-time acting loop)

Maintains the per-episode latent state for live play: the K-frame deque, the RSSM
posterior `(deter, stoch)`, the macro-context `m`, and the previous action. It exists
as a **single shared class** because both the collector and the eval harness need this
loop, and two copies would silently diverge into measuring a different agent than the
one being trained.

The subtle part (tickets/0008): the TaskEncoder folds under the **arrival-state
convention** — a fold consumes the transition that *arrived at* the state it's called
with. Online, that arrival state for transition t→t+1 only exists once the next
`act()` has stepped the RSSM onto t+1. So `observe(action, reward, frame)` merely
*stashes* the pending (action, reward), and `act()` performs the fold right after its
`observe_step`, before the policy reads features. Per-step ordering is then exactly
training's: **step → fold → act**.

### 4.4 The collection side (`training/trainer.py`, `Trainer.train`)

Round-robin over games (all downloaded games by default; `--config.train-games`
restricts the cycle for the held-out generalization protocol, tickets/0009). Per env
step:

1. Choose an action: uniform-random over `available_actions` during prefill, the
   policy (via OnlineActor, sampled, masked) afterwards. So exploration transitions
   from uniform to disagreement-seeking the moment the policy takes over.
2. `env.step`, write the step into the current buffer episode (frame, action, reward
   delta, `result.done` as terminated, normalized score, the legal-action mask the
   action was chosen under), and advance the OnlineActor's latent state.
3. On termination or the 600-step cap: log per-game episode return/length/win-rate,
   rotate to the next game in the cycle.
4. Catch up on grad steps at the `train_every` ratio (a `while`, so hiccups produce a
   catch-up burst rather than a permanently drifted ratio).
5. On env-step cadences: atomic checkpoint (`latest.pt` + `buffer.pt`, temp-then-
   rename), and optionally the in-training eval hook (§4.7).

Resume is automatic and total: model, all three optimizers, both return-normalizer
scales, step counters, and the buffer come back from `output_dir`. A resumed run's
`train_games` must match the checkpoint's (changing it would silently re-key buffered
episodes' game ids and could leak held-out data — it raises instead).
`--config.init-from` is the other workflow: seed a *fresh* run's world-model weights
only.

### 4.5 Grad step, part 1 — the world-model update (`train_step`)

Sample a batch of `(16 burn-in + 16 loss)`-step windows, then `forward_sequence`:

1. **Burn-in phase** (16 steps): step the RSSM posterior and the TaskEncoder over real
   steps to warm the recurrent state and build a real, nonzero `m`. Nothing is decoded
   or stored — these steps are context, not targets.
2. **Boundary**: detach `(deter, stoch)` — BPTT length through the trunk stays exactly
   `seq_len`. `m` is *not* detached: the gradient path (loss-window heads → frozen `m`
   → burn-in TaskEncoder steps) is the only way the TaskEncoder trains.
3. **Loss window** (16 steps): posterior teacher-forcing with `m` **frozen**. Each
   step yields prior stats, posterior stats, the decoded reconstruction, and the head
   predictions.

Then `compute_losses`, one Adam step over `world_model.parameters()`:

| Term | What | Why it's shaped that way |
|---|---|---|
| `recon` | Cross-entropy of decoded logits vs each stack's **settled (last) frame** | Cells are categorical; earlier stack frames are encoder context, not targets |
| `kl` | KL(posterior ‖ prior), **balanced** (α = 0.8 on training prior→sg(post), 0.2 on post→sg(prior)), weight 0.2, **free-bits floor** of 1.0 total nat | Balancing trains the prior hard toward the posterior (good imagination) while lightly regularizing the posterior; the floor stops KL being crushed to zero (posterior collapse). Watch `kl_raw`, not `kl_loss` — below the floor the clamp zeroes the gradient |
| `reward` | MSE, per-step weight `1 + 10·\|r\|`, is_first masked | Reward is ~all zeros; unweighted MSE regresses the head to 0 everywhere and level completions never reach imagination |
| `continue` | BCE vs `not terminated`, is_first masked | Only real deaths are terminations (§4.2) |
| `internal_state` | MSE vs normalized score | Auxiliary representation shaping (§3.5) |
| `ensemble` | Each head's MSE predicting `post_mean[t+1]` from detached `(deter[t], stoch[t], action[t+1], m[t+1])` | Rides the same batches; detached both ways so it can't perturb the trunk |

### 4.6 Grad step, part 2 — the actor-critic update in imagination

(`policy_train_step` + `training/actor_critic.py`; design in tickets/0003/0005/0006.)

1. **Dream** (`Thumper.dream`, no-grad): from *every* posterior state the world-model
   pass just produced (16×16 = 256 start states, detached), roll the policy against
   the RSSM **prior** for `horizon = 15` steps. `m` stays frozen; the start state's
   `available_actions` mask is held for the whole rollout (the world model has only
   ever seen legal actions — disagreement on illegal ones is untrained garbage). Each
   imagined transition records: features, action, ensemble **disagreement** (the
   intrinsic reward), predicted reward, and continue probability.
2. **Two return streams** (tickets/0005). Extrinsic (predicted score) and intrinsic
   (disagreement) each get their own TD(λ) return (γ = 0.997, λ = 0.95, bootstrapped
   from the *target* critic), their own critic head, and their own `ReturnNormalizer`
   (a running 5%–95% spread with a `max(1, scale)` floor, DreamerV3-style). Splitting
   them is what keeps a sparse +1 level completion visible against a continuous stream
   of exploration payment — under one shared normalizer the ~10× larger intrinsic
   stream drowned the extrinsic one.
3. **Absorbing scores** (tickets/0006): the extrinsic discount chain is multiplied by
   `(1 − clamp(reward, 0, 1))`, so a predicted score absorbs all extrinsic credit
   after it — one dream can bank at most ~one level completion. This kills
   hallucinated reward-farming inside imagination (the actor had learned to re-trigger
   the barely-trained reward head on states nothing real ever anchored). Multi-level
   episodes are still learned — through the critic's bootstrap at the dream's start
   state and through real buffer episodes — just not inside a single dream. Intrinsic
   discounts are untouched.
4. **Losses** (the with-grad second pass, gradients flow only into policy/critic):
   - actor: REINFORCE, `−log π(a) · advantage − 1e-3 · entropy`, where
     `advantage = norm_ext(R_ext − V_ext) + intrinsic_scale · norm_int(R_int − V_int)`;
   - critic: MSE of both heads against their (detached) λ-returns.
   Separate Adam optimizers for policy and critic; target critic hard-synced every
   100 grad steps.

### 4.7 Evaluation (`training/evaluate.py`, `eval.py`, tickets/0007)

The online collector's scalars answer "is the world model learning", not "which games
can the policy play" — they come from a stochastic, exploration-biased policy sampled
whenever the round-robin lands on a game. `evaluate(thumper, env, protocol)` runs a
**fixed, repeatable protocol** instead: same game list, episode count (default 5),
600-step cap, seed, in both greedy and sampled modes, driving the same `OnlineActor`
the trainer uses. Measurement only — nothing touches the replay buffer. Three callers:
the `eval.py` CLI (writes `eval_report.json` next to the checkpoint), the optional
in-training hook (`--config.eval-every`, on its own Env with RNG state save/restored
so enabling eval doesn't fork the run's trajectory), and future automation.

The **held-out generalization protocol** (tickets/0009) composes these:
`--config.train-games` keeps a subset out of collection entirely, `--config.eval-games`
still measures the held-out games zero-shot. Such a run must train from scratch —
every pre-0009 checkpoint saw all 25 games, so warm-starting bakes held-out dynamics
into the weights undetectably.

### 4.8 Telemetry

TensorBoard under `<output_dir>/tb`: `loss/*` (world model), `policy/*` (actor-critic,
including `dream_score_sum` — the average predicted score banked per dream, the
hallucinated-farming alarm — and both return-normalizer scales), `wm/*` (ensemble
disagreement, overall and per game), `online/*` (per-game episode returns/lengths/
win rates, action-type fractions, `macro_context_norm`), `train/*`
(`reward_windows_in_batch` — how many sampled windows actually contain a scoring
event), `eval/*`, `perf/*`. Qualitative recon and imagination PNGs land in
`<output_dir>/samples` (`training/qualitative.py`, official ARC palette). Every run
gets a `TRAINING_LOG.md` entry: command, pre-registered expectations, findings.

---

## 5. Key numbers at a glance

| Quantity | Value |
|---|---|
| Grid / vocab | 64×64, 17 symbols (16 colors + pad) |
| Frame stack K | 4 |
| Vision out / RSSM embed | 256 |
| RSSM deter / stoch | 256 / 32 (diag Gaussian, std = softplus + 0.1) |
| Macro-context `m` | 128 |
| Feature dim (heads/policy/critic) | 416 = 256 + 32 + 128 |
| Action encoding | 135 = 7 type ++ 64 x ++ 64 y one-hots |
| Ensemble | K = 5 heads, hidden 256 |
| Window | 16 burn-in + 16 loss steps, batch 16 |
| KL | weight 0.2, balance 0.8, free bits 1.0 total nat |
| Reward loss weighting | 1 + 10·\|r\| per step |
| Dream | horizon 15, from 256 start states per grad step |
| Returns | γ = 0.997, λ = 0.95, entropy 1e-3, intrinsic_scale 1.0 |
| LRs | world model 3e-4, actor 8e-5, critic 8e-5 (Adam ×3) |
| Loop | train_every 2, prefill 1000, episode cap 600, buffer 200k steps |

---

## Missing Features

Gaps between what exists and what an actual ARC-AGI-3 submission needs. Roughly
ordered by how load-bearing they are.

1. **Test-time adaptation / cross-episode learning on a novel game.** The benchmark's
   premise is *learning efficiency on unseen games*, but at eval time Thumper is
   frozen: `evaluate` builds a fresh `OnlineActor` per episode, so even the
   macro-context — the one mechanism designed to infer a game's rules — is thrown away
   between episodes of the *same* game. Nothing improves across the episode budget: no
   carried `m`, no gradient steps on eval experience, no episodic memory of what was
   tried. Held-out eval today measures pure zero-shot transfer, which is a floor, not
   the benchmark's actual target. (Flagged in the July 2026 tech-lead review as the
   top un-ticketed follow-up.)
2. **No decision-time planning.** The world model is only ever used for training-time
   imagination; at act time the policy is a single reactive forward pass. The entire
   point of having a dynamics model at test time — rolling candidate action sequences
   forward and picking the best (MPC/MCTS-style search) — is unused. On a novel game
   where the policy's habits don't transfer, search over the (adapted) world model is
   the most plausible source of competent behavior.
3. **Online/competition operation mode.** Everything runs against `OperationMode.OFFLINE`
   and local game files. A real submission speaks the live API: scorecard lifecycle,
   the per-game action budget, rate limits, and the arc-agi agents protocol. No
   adapter exists yet, and the eval protocol's 600-step cap is a training convenience,
   not the benchmark's budget structure.
4. **The train/act macro-context mismatch.** Training only ever builds `m` from a
   16-step burn-in; online play accumulates it over episodes up to 600 steps. The
   TaskEncoder is never trained on the long-horizon regime it is actually used in
   (tickets/0002's remaining follow-up; `online/macro_context_norm`'s early spike in
   Run 7 — max ~67 vs O(1) later — is this mismatch showing up live).
5. **Exploration has no episodic novelty term.** Ensemble disagreement is *global*
   novelty — it decays permanently as the model learns, and it cannot tell "I haven't
   tried this yet *this episode*" from "I've never tried this". Sparse multi-step
   puzzles likely also need episodic novelty (e.g. within-episode state counts) and/or
   goal-conditioned exploration to string discovered mechanics into plans.
6. **ACTION7 is outside the action space.** The engine defines it; `env.py` silently
   filters it from `available_actions`. If any evaluation game requires ACTION7, the
   agent cannot play it at all. Worth verifying against the live game set before
   submission.
7. **No mechanism for level structure.** `internal_state_head` regresses cumulative
   score, but nothing represents "level index" or resets fast-memory expectations at a
   level transition, even though levels within a game can differ almost as much as
   games do.

*Not listed as missing*: bigger models, more games, longer runs — scaling knobs, not
architectural gaps.
