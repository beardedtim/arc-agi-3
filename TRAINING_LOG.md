# Training Log

Running log of every training run: the exact command, what we expect to see
(written **before** launch, so success criteria can't drift after the fact),
and what actually happened.

Conventions:

- One entry per run, newest at the bottom, dated. Each entry has the launch
  command (with comments explaining every non-default flag), a **What to Look
  for** section pre-registering good/bad/ugly outcomes, and a **Findings**
  section filled in after (or during) the run.
- Reference tickets (`tickets/NNNN-*.md`) for the design/fix behind each run
  rather than restating them.
- Every run gets its own `--config.output-dir` under `runs/` (gitignored),
  holding `latest.pt` + `buffer.pt` checkpoints, `tb/` TensorBoard logs, and
  `samples/` qualitative PNGs. **Resume gotcha:** `resume` defaults to `True`
  and silently picks up `<output_dir>/latest.pt` if present — a comparison
  run must use a fresh output dir, or it will continue the old run instead.
  `--config.init-from <ckpt>` warm-starts a new run's world-model weights
  only (fresh optimizer/counters/buffer).
- Record deliberate early stops and their reason explicitly; a stopped run
  with an answered question beats a finished run nobody drew conclusions from.

---

# Run 1 — world-model-only baseline (July 9, 2026)

First end-to-end run of the tickets/0001 training loop, restructured (July 9)
into an online-style loop before ever launching: uniform-random policy
collects experience round-robin across every downloaded game — all 25
available games were downloaded July 9, per tickets/0001's "trained on all
games from day one" — and only `Thumper.world_model` trains; the policy sits
untouched in the checkpoint. Defaults throughout (`TrainerConfig`): 100k env
steps, one grad step per 2 env steps after a 1k-step prefill (≈49.5k grad
steps), batch 16, seq_len 16, lr 3e-4. 100k steps round-robined across 25
games is only ~4k steps per game — thin coverage; judge recon quality with
that in mind and expect to extend the budget.

```sh
# Baseline world-model training, all defaults. Everything lands under
# runs/world_model: TensorBoard in tb/, recon/imagination sample PNGs in
# samples/, latest.pt + buffer.pt checkpoints every 5k env steps. Rerunning
# the same command resumes automatically from latest.pt.
uv run python train.py
```

## What to Look for

Checks below, roughly in the order to run them. TensorBoard is
`uv run tensorboard --logdir runs/world_model/tb`; the console line
(`env | grad | ETA | total | recon | kl_raw | reward | cont | buffer |
grad steps/s`) mirrors the headline scalars.

**Good:**

- `loss/recon` falling clearly and monotonically-ish — the primary signal
  that Vision + RSSM + decoder are learning the games' frames at all. This
  is the headline metric; everything else is secondary on a first run.
- `loss/kl_raw` settling _above_ the `free_nats` floor (1.0 total /
  `stoch_dim` per-dim) but not pinned dead flat at it. `loss/kl` is the
  clamped term that's optimized; once it sits at the floor it's a constant,
  so always judge posterior health from `kl_raw`. Flat-at-floor with recon
  still improving is acceptable; flat-at-floor with recon stalled means
  posterior collapse — `kl_weight`/`free_nats` in `WorldModelConfig` are the
  knobs.
- **Qualitative frames** (`runs/world_model/samples/recon_step_*.png`, every
  500 grad steps): bottom (reconstruction) row converging on the top (real)
  row — playfield structure, HUD bars, then individual sprites. The step-1
  sample is pure noise by construction; what matters is the trend. Samples
  are drawn from random buffer sequences, so across checkpoints they should
  show _many different games_, not one game's look — 25 games round-robin
  is the whole point of this run.
- **Imagination check** (`samples/imagine_step_*.png`, every 1k grad steps):
  imagined rows staying recognizable for more of the 8-step horizon as
  training progresses. Early on expect them to wash out after a frame or
  two — that's the prior still untrained, not a bug. This is the actual test
  of learned dynamics; recon alone can be single-frame compression.
- `loss/reward` and `loss/continue` near zero and quiet. Under random play
  on a sparse game almost every target is 0/not-done, so low values mean
  "little signal yet," **not** mastery — don't over-read them.
- `wm/ensemble_loss` falling, and `wm/disagreement_mean`/`p90` staying
  meaningfully above zero — the disagreement signal is the future
  exploration currency, so it going identically flat would matter later even
  though nothing consumes it yet. `wm/disagreement_by_game/*` differing
  across games is the interesting readout: it should be higher on games the
  buffer has seen less of or whose dynamics are stranger.
- `online/episode_len/<game>` / `online/episode_return/<game>` — a first
  look at how long random-play episodes run per game and whether random
  play ever scores anywhere. `online/win_rate/*` at ~0 throughout is
  expected; any game where random play _does_ win is worth noting for
  later policy work.
- Checkpoints appearing every 5k env steps, and a kill + rerun of the same
  command resuming from `latest.pt` with continuous counters (worth one
  manual smoke check early in the run before walking away).

**Bad (tune, don't necessarily stop):**

- `loss/recon` plateauing early at a visibly non-trivial value with samples
  still blurry — with 25 games sharing ~4k steps each, the model may just
  need more steps (`--config.total-env-steps`); extend before adding knobs.
- `kl_raw` climbing steeply while recon falls — prior/posterior diverging;
  `kl_weight` (0.2 default) is the documented tradeoff dial, and
  `--config.kl-warmup-steps` (off by default) is the prepared lever.
- Episodes ending almost immediately every reset (`online/episode_len`
  pinned low on some game) — random play barely explores it and its buffer
  share is mostly first-frames; worth a ticket on the exploration side, not
  a trainer bug. Also watch for the opposite: a game whose episodes _never_
  end under random play will dominate the length-weighted episode sampling.

**Ugly (stop and investigate):**

- NaN/inf in any loss, or `train/grad_norm` spiking and staying high —
  numeric blowup; `grad_clip` is already 100, so suspect the data path
  (frame preprocessing, action encoding) before the optimizer.
- `loss/recon` not moving at all in the first few hundred grad steps —
  plumbing bug (wrong targets, frames not reaching the loss), not tuning.
- Resume producing counters or losses discontinuous with the checkpoint —
  checkpointing bug; nothing downstream can be trusted until fixed.

Open questions this run should inform (not gate on):

- Throughput (`perf/env_steps_per_sec`) and wall-clock for 100k env steps on
  this machine — sets expectations for scaling later.
- Whether ~4k random steps per game is enough for crisp recon and a
  multi-step-coherent imagination across 25 visually different games with
  one shared latent, or whether the budget (or model capacity) needs to
  grow — this run is the baseline that decision reads off.

## Findings

**Success.** Ran to completion (100k env steps, 49,500/49,500 grad steps,
~1h wall-clock at ~27 env steps/s / ~13.4 grad steps/s). Every pre-registered
"Good" criterion was met; no "Bad" or "Ugly" triggers fired.

- `loss/recon` fell hard and cleanly, ~0.134 → ~0.0075 with no plateau. The
  final recon samples are near pixel-perfect — playfield, HUD, sprites, even
  the click-cursor highlight — and checkpoints span visually distinct games,
  confirming one shared latent handles all 25.
- **Imagination works**: late `imagine_step_*.png` rows track ground truth
  across most of the 8-step horizon; artifacts appear only at discrete
  transition frames (e.g. a piece landing), then recover. The prior learned
  real multi-step dynamics, not just single-frame compression.
- No numeric issues: `train/grad_norm` settled 2.1 → 0.49 (one early spike
  to 84, well under the clip); losses moved immediately from step 1.
- Ensemble healthy: `wm/ensemble_loss` 0.38 → 0.12 while
  `disagreement_mean` stayed meaningfully above zero (~0.009, p90 ~0.02) —
  the exploration currency exists and is non-degenerate.
- `loss/reward` / `loss/continue` near zero and quiet, as expected under
  random play; no game had a nonzero win rate.

Two observations worth carrying forward (neither a blocker):

1. **`kl_raw` (~0.07) sits well _below_ the 1.0 free-nats floor**, so the KL
   term is fully clamped and inert. _[Correction, July 10 — see Run 6
   Findings: the floor is per-dim, 1/32 ≈ 0.031, so 0.07 is above it and
   the KL term was active.]_ This is the benign case — recon kept
   improving, so posterior ≈ prior because the prior itself got good — but it
   means `kl_weight` currently does nothing. Revisit when tickets/0002's
   macro-context conditioning changes the prior/posterior heads; there is
   headroom for the context to absorb task identity without fighting the KL.
2. **Episode length is pinned at the 600-step cap on 24 of 25 games** (only
   `sp80` terminates naturally, ~250 steps). Random play essentially never
   finishes an episode, so the continue head sees almost no "done" signal and
   buffer composition is uniform-by-construction. This is exactly the
   motivation for the exploration work in tickets/0002, not a reason to
   change it.

Decision: baseline validated; proceed with tickets/0002 as written.

---

# Run 2 — full tickets/0002 world model, macro-context (July 9, 2026)

First full run of the tickets/0002 architecture (TaskEncoder macro-context
conditioning the RSSM prior/posterior, heads, and ensemble), at Run 1's
defaults in a fresh output dir. Purpose: the real regression check against
Run 1 — recon trajectory, final samples, and imagination quality. Because
both runs use `seed=0` against a deterministic offline env and collector,
they gather and sample _identical_ data, so samples and scalars are
comparable frame-for-frame at matched grad steps.

```sh
uv run python train.py \
  --config.output-dir runs/meta_rl
```

## What to Look for

(Reconstructed — this entry was launched with only the command recorded,
violating the pre-registration convention; criteria below are from
tickets/0002's acceptance criterion 4 and Run 1's "Good" list, written down
mid-run when the investigation began.)

- `loss/recon` matching Run 1's trajectory at matched grad steps — the
  macro-context should be at worst neutral for reconstruction.
- Imagination samples (`imagine_step_*.png`) at least as coherent as Run 1's
  at matched steps.
- `kl_raw` healthy, `train/grad_norm` bounded, ensemble non-degenerate.

## Findings

**Stopped deliberately at ~40k/100k env steps (19,500 grad steps):
imagination regressed badly versus Run 1, and the question this run existed
to answer — does 0002's macro-context regress the world model? — was
answered. Yes, in the one place the scalars can't see.**

- **Reconstruction: fine.** `loss/recon` tracked Run 1 at matched steps
  (0.160/0.043/0.021 at 5k/10k/15k vs Run 1's 0.093/0.033/0.018 — ~15%
  behind, converging). Recon samples at matched steps are near-identical to
  Run 1's, including the same late-to-learn small sprites.
- **Every trained loss looked healthy — deceptively.** Smoothed `kl_raw` ran
  _equal or lower_ than Run 1's (~0.09–0.14 vs ~0.10–0.18); ensemble loss
  noisier but falling. Only `train/grad_norm` differed in shape, running ~2×
  Run 1's throughout (3–5 vs 1.5–3).
- **Imagination collapsed.** At matched grad steps on identical batches,
  Run 1's dreams stay on-game with minor artifacts; this run's dreams
  (9k/11k/14k/17k samples) disintegrate within 2–6 imagined frames, often
  morphing into _other games'_ textures.
- **Root-cause A/B (script preserved in tickets/0004):** from the same
  burned-in posterior start state and real action sequence, dreams with the
  burned-in macro-context vs a zeroed one degrade about equally (burned-in
  sometimes worse) — so the regression is not the qualitative check's
  freeze-at-zero query; the prior itself is weaker no matter what context
  it's given.
- **Mechanism:** during training, the prior at step t is conditioned on
  `m_t`, freshly built every step from the _real_ transitions up to t−1
  (including detached posterior samples, which carry observation
  information). `m` is therefore a second, teacher-forced memory stream
  that always carries fresh ground-truth trajectory info at training time —
  the prior learns to lean on it (which is _why_ `kl_raw` beat Run 1's) —
  and at dream time the frozen context goes stale after one step, taking
  the prior's apparent competence with it. And because tickets/0002
  re-initializes `m` to zero each 16-step window, `m` currently carries no
  cross-episode task belief at all: it is pure fast memory duplicating the
  GRU with privileged access. The 0002 "known limitation" is not a benign
  deferral — in this form the macro-context is actively harmful to the one
  consumer that matters (imagination, where the tickets/0003 actor-critic
  trains).

Decision: stopped (per the conventions above — the question is answered).
The fix is tickets/0004 (train/dream-consistent macro-context); no further
full runs until it lands, since tickets/0003's policy would be trained
inside exactly these broken dreams. Run artifacts kept at `runs/meta_rl`
for the 0004 before/after comparison.

---

# Run 3 — tickets/0004 burn-in fix, the decisive rerun (July 9, 2026)

First full run of the tickets/0004 architecture: `forward_sequence` grows a
`burn_in`-step prefix (default 16) that warms the RSSM state and builds the
window's macro-context, then holds it frozen for the entire `seq_len`-step
loss window — the TaskEncoder is never stepped inside the window itself.
This is meant to remove Run 2's pathology at the root: training's
prior/posterior conditioning is now the same _kind_ of frozen, real-data
context that `Thumper.dream` and `imagine_with_burn_in` use, so the prior
can no longer lean on a per-step teacher-forced context that goes stale the
instant a dream starts. Same defaults as Runs 1/2 otherwise (100k env
steps, `train_every=2`, `prefill_steps=1_000`, batch 16, seq*len 16, lr
3e-4), fresh output dir, `seed=0` against the deterministic offline env —
so this run gathers and samples data \_identical* to Runs 1/2, keeping all
three frame-for-frame comparable at matched grad steps. `burn_in=16` is
passed explicitly below even though it's the default, for self-documentation.

```sh
# Fresh output dir (mandatory: runs/meta_rl still holds Run 2's pre-fix
# checkpoint/buffer for comparison, and resume defaults to True). All other
# flags at Run 1/2 defaults so scalars/samples stay matched-step comparable.
uv run python train.py \
  --config.output-dir runs/burn_in_fix \
  --config.burn-in 16
```

Before comparing full-run results, first replay tickets/0004's A/B script
(preserved in that ticket's appendix) against the **existing** Run 2
checkpoint — this is a quick sanity check, not part of this run's budget,
and needs no training:

```sh
uv run python - <<'EOF'
import torch
from model.world_model import WorldModel
from training.qualitative import grid_to_image, _row_of_frames, _stack_rows
from training.replay_buffer import ReplayBuffer

BURN, SEQ = 8, 16
payload = torch.load("runs/meta_rl/latest.pt", map_location="cpu", weights_only=False)
wm = WorldModel(payload["config"].world_model)
wm.load_state_dict({k.removeprefix("world_model."): v
                    for k, v in payload["state_dict"].items() if k.startswith("world_model.")})
wm.eval()
buffer = ReplayBuffer.load("runs/meta_rl/buffer.pt", frame_stack=wm.config.frame_stack,
                           internal_state_dim=wm.config.internal_state_dim)
batch = buffer.sample(64, SEQ)
settled = batch["observations"][:, :, -1]
dynamism = (settled[:, 1:] != settled[:, :-1]).float().mean(dim=(1, 2, 3))
for rank, i in enumerate(dynamism.argsort(descending=True)[:3].tolist()):
    o, at = batch["observations"][i:i+1], batch["action_types"][i:i+1]
    cd, fi, rw = batch["coords"][i:i+1], batch["is_first"][i:i+1], batch["rewards"][i:i+1]
    with torch.no_grad():
        out = wm.forward_sequence(o[:, :BURN], at[:, :BURN], cd[:, :BURN], fi[:, :BURN], rewards=rw[:, :BURN], burn_in=0)
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
    _stack_rows(rows).save(f"context_ab_{rank}.png")
EOF
```

## What to Look for

Two separate checks, run in this order: the A/B script above (a diagnostic
replay against Run 2's _unfixed_ checkpoint — it does not exercise this run's
code path, only confirms the pre-0004 mechanism), then the full rerun's
scalars/samples against both Run 1 (the bar to meet) and Run 2 (the failure
to beat), per tickets/0004's acceptance criteria 3–4.

**A/B script — Good:**

- The three `context_ab_*.png` rows (real / burned-in-`m` dream / zeroed-`m`
  dream) confirm Run 2's finding again: burned-in and zeroed context degrade
  about equally on Run 2's checkpoint. This isn't a pass/fail gate on its
  own — it's a repeat of the diagnosis, confirming nothing has silently
  changed about _why_ Run 2 failed before judging whether the fix (a
  different training run entirely) addresses it.

**Full rerun — Good (criterion 4):**

- `imagine_step_*.png` at matched grad steps (9k/11k/14k/17k) hold structure
  on dynamic games for most of the 8-step horizon, on par with Run 1's — not
  disintegrating into cross-game mush within 2–6 frames the way Run 2's did.
  This is the headline check; everything else here is secondary.
- `loss/recon` within ~20% of Run 1's matched-step values (Run 1:
  0.093/0.033/0.018 at 5k/10k/15k) — burn-in shouldn't make reconstruction
  meaningfully worse, only spend a bit more compute per window on
  non-trained burn-in steps.
- `train/grad_norm` back near Run 1's band (~1.5–3), not Run 2's ~2× elevated
  shape (3–5) — a mechanical signature that the prior is no longer leaning
  on a teacher-forced per-step context.
- `loss/kl_raw` **rising** back toward Run 1's level (~0.10–0.18) as the
  per-step-context crutch disappears. This is expected recovery per
  tickets/0004's non-goals, not a regression — do not retune `kl_weight` or
  `free_nats` in response; ticket any tuning separately.
- `wm/disagreement_mean`/`p90` non-degenerate, same order of magnitude as
  Run 1's (~0.009 / ~0.02) — the ensemble should be reading a comparably
  expressive latent, not one degraded by a broken prior.
- Checkpoints and `buffer.pt` writing on the usual 5k-env-step cadence with
  no shape/load errors — the burn-in change should be purely a data-flow
  change (no new/removed parameters), so this run's `latest.pt` should also
  `load_state_dict` cleanly against a pre-0004 architecture check if ever
  needed for comparison.

**Bad (tune, don't necessarily stop):**

- Imagination clearly better than Run 2 but still visibly short of Run 1
  (e.g. holding structure for only 4-5 of 8 steps instead of most of it) —
  before concluding the fix is a partial win, extend the budget past 20k
  grad steps first; Run 1/2 at this same matched-step range were both still
  converging.
- `loss/recon` more than ~20% behind Run 1 at matched steps but still
  falling cleanly — likely just the burn-in phase's extra real steps
  competing for buffer sampling attention early on; revisit after more
  steps before touching `burn_in`/`seq_len`.

**Ugly (stop and investigate):**

- Imagination at matched steps still disintegrating the way Run 2's did —
  the fix did not address the root cause as diagnosed; re-open tickets/0004
  rather than re-tuning hyperparameters. Per that ticket's fallback clause,
  the next step would be stripping `macro_context` from the RSSM
  prior/posterior conditioning entirely (keeping it only in the decoder-
  adjacent heads/ensemble/policy/critic) — but only after confirming this
  criterion actually failed, not preemptively.
- NaN/inf in any loss, or a shape/load error out of `compute_losses` /
  `forward_sequence` — the burn-in windowing (slicing `batch[:, burn_in:]`,
  the boundary detach) is new plumbing and the most likely place for an
  off-by-one.
- `task_encoder` parameters showing zero gradient throughout (check
  `grad_norm` isn't misleadingly nonzero only from the trunk) — would mean
  the burn-in gradient path (loss-window heads → frozen `m` → burn-in
  TaskEncoder steps) isn't actually connected, silently freezing
  `task_encoder` at its initialization.

Decision this run should produce: if criterion 4 passes, this entry's
Findings should say so explicitly and green-light launching tickets/0003's
full actor-critic run (so far blocked on this fix landing) against this
run's checkpoint via `--config.init-from`.

## Findings

(Filled in July 10, morning after the overnight run.)

**Success. Criterion 4 passes: imagination is fixed.** Ran to completion
(100k env steps, 49,500/49,500 grad steps, ~4h50m wall-clock at ~5.7 env
steps/s / ~2.9 grad steps/s — ~5× slower than Run 1, from the burn-in
doubling each window's forward pass plus the actor-critic update). The
"Ugly" imagination criterion did not fire; the fix addressed the root cause
as diagnosed.

**One comparability caveat the pre-registration got wrong:** this run
exercised the _full_ tickets/0003 loop — the actor-critic trained from grad
step 1 and the policy (not the random collector) gathered data after the 1k
prefill. Runs 1/2 predate 0003 and were world-model-only under random play,
so despite `seed=0` the collected data diverges after prefill and nothing is
frame-for-frame comparable. Matched-step comparisons below are qualitative.
(This also means the pre-registered "green-light a separate 0003 run" decision
is moot — the 0003 actor-critic already ran here, end to end, on top of the
fix.)

- **A/B script (run July 10, before judging the rerun):** on Run 2's
  unfixed checkpoint, dreams with burned-in vs zeroed macro-context degrade
  about equally — the pre-0004 diagnosis reproduces exactly. Outputs were
  written to a scratch dir, not kept; the script is in the entry above.
- **Imagination (the headline):** `imagine_step_*.png` at 9k/17k/31k/49k all
  hold game identity and playfield structure across the full 8-step horizon
  — moving sprites tracked, only minor sprite-level blurring late in the
  horizon on the most dynamic games. Nothing resembling Run 2's 2–6-frame
  disintegration into cross-game mush. On par with Run 1's quality.
- **`train/grad_norm` recovered** to Run 1's band and below (medians
  2.7/2.0/1.1 at 5k/10k/15k vs Run 2's 4.0/4.4/3.0; 0.20 by the end) — the
  mechanical signature that the prior no longer leans on teacher-forced
  per-step context.
- **`loss/recon`: behind Run 1 beyond the ~20% band, but falling cleanly.**
  Medians 0.056/0.037/0.029 at 5k/10k/15k vs Run 1's 0.042/0.022/0.018
  (~35–65% behind), final 0.0100 vs 0.0060. Final recon samples are still
  sharp. Per the pre-registered "Bad" clause this is the extend-the-budget
  case, not a stop — and it's confounded: the policy's on-distribution data
  is less uniform than random play's, and each window now trains on only
  half its steps. One single-batch transient at grad step 10,067 (recon 3.1,
  grad_norm 59) recovered immediately; no NaN/inf anywhere.
- **`kl_raw` did _not_ rise back toward Run 1's level** — the opposite:
  ~0.05 flat vs Run 1's ~0.13→0.07 (Run 3 ends at 0.034). _[Correction,
  July 10 — see Run 6 Findings: the floor is per-dim ≈ 0.031, so these
  values hover at/above it and the KL term was active, not inert.]_ The
  expected-recovery prediction was wrong, but the dreams prove the low KL is
  now honest prior competence rather than the Run 2 crutch. No retuning, per
  the ticket's non-goals.
- **Ensemble non-degenerate but lower:** `disagreement_mean` 0.0023 at the
  end vs Run 1's 0.0074 (same order of magnitude, as pre-registered).
  Plausibly the new consumption loop at work: the policy is now _paid_ in
  disagreement (`intrinsic_scale=1.0`), collects where the ensemble is
  uncertain, and trains the uncertainty away. Watch it doesn't collapse to
  zero on longer runs.
- **First signs of a learning policy:** `online/episode_return/cd82` hit 1.0
  on 3 of its last episodes (env steps ~62k/74k/88k) and `r11l` once (~94k);
  `online/win_rate/*` stayed 0 everywhere. Only 8 episodes per game total,
  so this is suggestive, not conclusive. Actor-critic internals healthy
  throughout: entropy 2.6–3.9, `policy/value_mean` tracking
  `policy/imagined_return`, imagined returns dominated by intrinsic reward
  (extrinsic ≈ 0), actor grad norms small and stable.
- `online/macro_context_norm` rose to ~4.5 by ~30k env steps then decayed
  to ~1.5–2.1 — the online collector's episode-long `m` accumulation (the
  0002 follow-up mismatch) is live but not exploding.

Decision: tickets/0004 validated and closed by this run; Run 2's pathology
is gone at the root. Since the 0003 actor-critic already trained here, the
pre-registered `--config.init-from` follow-up is unnecessary. Next run
should target the two open threads this run surfaced: recon convergence
under the halved effective window + policy-collected data (extend
`total_env_steps` before touching knobs), and whether disagreement-as-reward
eats its own signal over longer horizons.

---

# Run 4 — tickets/0005 two-stream returns, smoke run (July 10, 2026)

Run 3 validated the world model end-to-end and showed exactly the failure
tickets/0003 pre-registered as a follow-up trigger: the reward head predicts
scoring transitions almost exactly, but the actor's objective structurally
can't see it — one shared `ReturnNormalizer` sees intrinsic disagreement's
continuous ~10-magnitude spread and normalizes a +1 level completion down to
~0.1 units, once. tickets/0005 splits extrinsic (reward) and intrinsic
(disagreement) into separate λ-returns, critic heads, and normalizers (one
policy, advantage = `norm_ext(R_ext − v_ext) + intrinsic_scale ·
norm_int(R_int − v_int)`), fixes the trainer storing step-cap timeouts as
`terminated=True` (only `result.done` counts now), and adds reward-event-
stratified batch sampling (`reward_window_frac=0.25` default) so the ~0.015%
of steps with nonzero reward don't dilute out of training as the buffer
fills.

`Critic`'s parameter shapes changed (`ext_net`/`int_net` replace the single
`net`), so Run 3's `latest.pt` is not resumable as-is —
`--config.init-from` copies world-model weights only, same remedy as
tickets/0003's note. Fresh output dir, fresh policy/critic/buffer under the
fixed objective (Run 3's buffer also carries ~166 timeout steps mislabeled
`terminated=True`, another reason not to resume it). All other flags at
Run 1/2/3 defaults; this is a small smoke run, not the full 100k-step
comparison run (out of scope per tickets/0005's Verification section) — a
few hundred grad steps past prefill is enough to see whether the split
scalars behave as designed before committing a full budget.

```sh
uv run python train.py \
  --config.output-dir runs/two_stream_returns \
  --config.init-from runs/burn_in_fix/latest.pt \
  --config.total-env-steps 3000
```

## What to Look for

This is a wiring smoke test (tickets/0005 acceptance criterion 3), not a
verdict on whether the split changes policy behavior — that needs the full
run this entry will be superseded by.

**Good:**

- `policy/return_norm_scale_ext` and `policy/return_norm_scale_int` both
  finite and visibly diverging from each other: `_int` climbing toward
  Run 3's ~10 (disagreement dominates its own stream, same as before the
  split), `_ext` staying near its 1.0 floor (extrinsic returns are still
  sparse at this budget — nothing wrong with that yet, it's the pre-split
  drowning made visible as a chart rather than fixed by itself).
- `train/reward_windows_in_batch` nonzero at least once any scoring window
  (cd82/r11l/sp80 transitions) is in the buffer — confirms the stratified
  sampler is finding and using them, not just falling back to uniform.
- `policy/entropy` in a healthy range (not collapsing toward 0, not pinned
  at max), comparable to Run 3's 2.6–3.9.
- `loss/recon` falling as in Run 3, not degraded by stratified sampling
  over-representing the three scoring games in early batches.
- No NaN/inf anywhere; checkpoint write/resume round-trips both
  `return_norm_scales.ext` and `.int`.

**Bad (tune, don't stop):**

- `return_norm_scale_ext` and `_int` tracking each other closely instead of
  diverging — at this short a budget could just mean too few scoring
  windows sampled yet; extend before concluding the split isn't taking
  effect.
- `reward_windows_in_batch` staying at 0 the whole run — check the buffer
  actually contains a scoring episode before suspecting the sampler itself
  (Run 3 games only score ~0.015% of steps; a 3k-step smoke run's buffer may
  legitimately have none carried over from init, since `--init-from` does
  not copy the buffer).

**Ugly (stop, something's broken):**

- Either critic head's loss or value mean is NaN/inf, or grad norms spike
  and don't recover — likely a channel-order mismatch between
  `Critic.forward`'s `(ext, int)` stacking and `actor_critic_losses`'
  indexing (`[..., 0]` / `[..., 1]`).
- `sync_critic_target` visibly not copying one of the two heads (compare
  `critic`/`critic_target` state dicts) — would show up as one stream's
  target values frozen at init forever.

## Findings

(Watched live and filled in July 10, immediately after the run.)

**Pass — promote to the full run.** Ran to completion (3,000 env steps,
1,000/1,000 grad steps, ~7 min wall-clock at ~5.5 env steps/s / ~2.8 grad
steps/s — Run 3's throughput, so the split costs nothing measurable). No
"Ugly" criterion fired; every "Good" criterion passed outright or resolved
to its pre-registered benign case. One watch-item (entropy, below) to
pre-register for the full run, not a blocker.

- **Scales split and diverging:** `return_norm_scale_ext` settled ~0.183
  (flat) while `_int` climbed 0.33 → 0.775 over the back half — finite,
  independent, and visibly diverging. Neither approached Run 3's ~10; the
  pre-registration missed that `--init-from` warm-starts a _converged_
  world model, so disagreement starts at Run 3's final ~0.005, not a fresh
  model's large early values. Note the logged value is the raw spread EMA;
  the `max(1, scale)` floor is applied at normalize time, so with both
  scales under 1.0 advantages currently pass through unscaled — by design.
- **No channel swap** (the pre-registered "Ugly"): `value_ext_mean` tracks
  `imagined_return_ext` (~0.12) and `value_int_mean` tracks
  `imagined_return_int` (~0.87) — two clearly distinct, correctly-paired
  streams. Both critic losses finite and falling; actor/critic grad norms
  small and stable; no NaN/inf anywhere.
- **`reward_windows_in_batch` stayed 0 — the benign case, confirmed:** the
  buffer's five completed episodes (ar25/bp35/cd82/cn04/dc22) all hit the
  600-step cap with zero return, so there were no scoring windows to
  stratify toward (cd82 scored in Run 3 but not in these 600 steps). The
  sampler path is covered by unit tests
  (`test_reward_frac_one_puts_event_in_every_loss_window`, fallback-to-
  uniform, position-varies; suite: 76/76 passing). The full run must show
  this go nonzero once any game scores — carry that forward as a criterion.
- **Checkpoint/resume round-trip verified:** `latest.pt` holds
  `return_norm_scales.ext/.int` matching the final TB values exactly
  (0.18276 / 0.77457); `critic` and `critic_target` both contain
  `ext_net`/`int_net` (target sync covered by
  `test_sync_critic_target_copies_both_heads`). Rerunning the same command
  resumed with continuous counters (`env_steps=3000, grad_steps=1000`) and
  exited cleanly at the met budget. On resume `--init-from` is ignored
  (trainer's `elif`), so the promote command below can keep the flag.
- **World model unharmed:** `loss/recon` started at warm-start quality
  (~0.009) and drifted down (~0.005 median late); grad step 1000's recon
  is near pixel-perfect and its imagination holds structure and game
  identity across the full horizon. `kl_raw` ~0.05, `train/grad_norm`
  bounded (<1.2). Stratified sampling never engaged (no events), so its
  early-batch skew remains untested — full-run item, not a smoke item.
- **Watch-item — entropy runs lower than Run 3's band:** median fell 3.7 →
  ~0.9–1.0 nats over 1,000 grad steps (min 0.41), vs Run 3's 2.6–3.9. The
  mechanism is the fix itself working: pre-split, the shared normalizer's
  ~10 scale shrank _all_ advantages ~10×, making `entropy_scale=1e-3`
  relatively strong; post-split both scales sit at the 1.0 floor, so
  advantages pass through full-size and the entropy bonus is ~10× weaker
  relative to them. Behaviorally nothing is degenerate — online action
  fractions stay spread across all 7 types (7.5–21%), and
  `intrinsic_reward_mean` _tripled_ (0.0055 → 0.016): the policy is
  actively steering into ensemble disagreement, which is precisely what
  the un-drowned intrinsic stream was built to buy. Entropy also flattened
  ~0.8–1.1 over the last 300 steps rather than continuing toward 0.

Decision: **promoted.** Extend this same run in place (resume picks up
`latest.pt`/`buffer.pt` at 3k steps; a fresh dir would only discard 3k
steps of on-objective data):

```sh
uv run python train.py \
  --config.output-dir runs/two_stream_returns \
  --config.init-from runs/burn_in_fix/latest.pt \
  --config.total-env-steps 100000
```

Pre-registered triggers for the full run (in addition to Run 3's open
threads — recon convergence and disagreement self-consumption):

- **Entropy:** if `policy/entropy` trends toward ~0 (median < ~0.3
  sustained) or any single `online/action_type_frac/*` exceeds ~0.6, raise
  `entropy_scale` (3e-3 first) — do not touch the normalizers; their floor
  behaving this way is the design.
- **Stratification:** once any `online/episode_return/*` > 0,
  `train/reward_windows_in_batch` must go nonzero within a few hundred
  grad steps; if it doesn't, the sampler's event index isn't seeing real
  episodes the way the unit tests' synthetic ones do — stop and debug.
- **Extrinsic stream waking up:** on scoring events, `_ext` scale and
  `policy/imagined_return_ext` should move; the whole point of the split
  is that a +1 level completion now lands as an O(1) advantage.

(These triggers are folded into Run 5's pre-registration below.)

---

# Run 5 — tickets/0005 two-stream returns, full run (July 10, 2026)

Promotion of Run 4's validated wiring to the full budget, per that entry's
decision: **extend the same run in place** — resume picks up
`runs/two_stream_returns/latest.pt` + `buffer.pt` at 3k env steps / 1k grad
steps, keeping the smoke run's on-objective data and warm world model. The
question this run exists to answer is the one tickets/0005 was written for:
**once a +1 scoring event lands as an O(1) extrinsic advantage, does the
actor learn to score on purpose?** Run 3's reward head predicted scoring
transitions almost exactly while its actor structurally couldn't see them
(4 incidental scores across ~200 episodes, win rate 0 everywhere); this run
points the split streams, the stratified sampler, and a warm-started reward
head at exactly that failure. It also carries Run 3's two open threads —
recon convergence under policy-collected data with the halved effective
window, and whether disagreement-as-reward eats its own signal over long
horizons — plus Run 4's entropy watch-item.

Budget arithmetic: ~97k env steps remain at Run 4's ~5.5 env steps/s ≈
**~5h wall-clock**, adding ~48.5k grad steps (49.5k lifetime). There is no
second prefill on resume: the policy collects from step 3,001 onward.
Comparisons against Run 3 are qualitative-only (warm-started world model,
fresh policy/critic, different objective — nothing is frame-for-frame
comparable despite `seed=0`).

```sh
# Same output dir as Run 4 (deliberately, for once): resume continues the
# smoke run instead of discarding its 3k steps. --init-from is inert on
# resume (trainer's elif) but kept as the correct cold-start fallback if
# latest.pt were ever missing. total-env-steps is the 100k default, passed
# explicitly because raising it is the entire difference from Run 4.
uv run python train.py \
  --config.output-dir runs/two_stream_returns \
  --config.init-from runs/burn_in_fix/latest.pt \
  --config.total-env-steps 100000
```

## What to Look for

`uv run tensorboard --logdir runs/two_stream_returns/tb`. Note the smoke
run's first 1k grad steps are part of the same curves; "early" below means
grad steps ~1k–5k, and matched-step comparisons to Run 3 offset by nothing
(both measure grad steps from their own step 1, close enough at this
granularity).

**Good:**

- **The headline — extrinsic stream wakes up, in causal order.** (1) Some
  game scores under intrinsic-driven play (`online/episode_return/*` > 0 —
  cd82/r11l are the Run 3 precedents, sp80 the natural terminator); (2)
  within a few hundred grad steps of that episode completing,
  `train/reward_windows_in_batch` goes nonzero and stays intermittently
  nonzero (stratification targets `reward_window_frac=0.25` of each batch,
  subject to available events); (3) `policy/return_norm_scale_ext` lifts
  off its ~0.18 resting value and `policy/imagined_return_ext` /
  `policy/value_ext_mean` move together off their bootstrap-dominated
  ~0.12; (4) scoring recurs _more often than Run 3's ~4-in-200-episodes
  incidental rate_ — repeated returns on the same game, ideally any
  `online/win_rate/*` > 0. Steps 1–3 are wiring doing its job; step 4 is
  the actual behavioral claim of tickets/0005. Partial credit is
  informative: 1–3 without 4 is a credit-assignment finding, not a wiring
  failure (see Decision).
- **Entropy stabilizes rather than collapsing** (Run 4's watch-item):
  `policy/entropy` holding a floor around ~0.5–1.5 nats with all
  `online/action_type_frac/*` staying under ~0.6. Lower than Run 3's
  2.6–3.9 is _expected_ (advantages are no longer shrunk ~10× by the old
  shared scale, so `entropy_scale=1e-3` is relatively weaker by design).
- **Recon keeps converging:** `loss/recon` ending at or below Run 3's final
  0.0100 (warm start + full budget should beat it; Run 1's 0.0060 is the
  stretch bar), samples staying sharp across many games, imagination
  holding the horizon as in Runs 3/4.
- **Disagreement decays without dying** (Run 3's self-consumption thread):
  `wm/disagreement_mean` may fall well below Run 3's final 0.0023 as the
  policy hunts it down — that's the loop working — but should stay
  measurably above zero with `p90` structure across games.
  `return_norm_scale_int` tracking the intrinsic return spread downward is
  the normalizer doing its job; note the `max(1, scale)` floor means
  sub-1.0 intrinsic advantages pass through unscaled and _naturally
  shrink_ as disagreement depletes, gracefully handing dominance to the
  extrinsic stream. That handoff visibly starting (int stream's share of
  the advantage falling while ext's rises) would be the best possible
  version of this run.
- **Mechanics:** checkpoints every 5k env steps; kill + rerun resumes with
  continuous counters and both normalizer scales; `train/grad_norm`
  bounded; `online/macro_context_norm` in Run 3's ~1.5–5 band; no NaN/inf.

**Bad (tune, don't necessarily stop):**

- **Entropy trigger fires:** `policy/entropy` median below ~0.3 sustained
  for a few hundred grad steps, or any single `online/action_type_frac/*`
  above ~0.6. Remedy: kill, resume with `--config.entropy-scale 3e-3`
  (`entropy_scale` is a _trainer_-level flag, not part of the checkpoint's
  saved `ThumperConfig`, so it takes effect on resume — verified in
  trainer.py). Log the change and the grad step it happened at here. Do
  not touch the normalizers; their floor behavior is the design.
- **No scoring events all run:** the split can't be judged either way —
  an _exploration_ shortfall, not a tickets/0005 defect. Don't retune this
  run's knobs in response; ticket the exploration side (Run 1's episode-cap
  observation and tickets/0002's original motivation) and consider whether
  intrinsic-only play should find reward more often than random did.
- **Recon behind Run 3 at matched budget but falling cleanly** — same
  extend-before-tuning clause as Runs 3/4.
- `return_norm_scale_int` climbing past ~1 and beyond: normalization is
  then actually engaging on the intrinsic stream — fine in itself; only
  worth attention if it runs away upward (unbounded disagreement growth
  usually means the world model is being destabilized by its own data).

**Ugly (stop and investigate):**

- **Scoring episodes exist in the buffer but `reward_windows_in_batch`
  stays 0** for ~1k grad steps after one completes — the stratified
  sampler isn't finding real events the way it finds the unit tests'
  synthetic ones (`Episode.rewards`-derived index vs. real episode layout;
  suspect the `loss_offset`/burn-in windowing math first). Everything
  downstream of the split depends on this path; nothing else is worth
  reading until it works.
- **Extrinsic critic destabilizes at first contact with real events:**
  `policy/value_ext_mean` or the ext half of `policy/critic_loss` spiking
  or going NaN right after the first nonzero reward windows are sampled —
  a +1 target after tens of thousands of zeros is exactly where an
  unclipped regression head jumps. (The two-stream design isolates any
  such damage from the int stream — verify the int stream indeed stays
  clean, which localizes the bug.)
- **Joint intrinsic/entropy death:** `wm/disagreement_mean` at ~0 _and_
  entropy pinned near 0 _and_ no extrinsic signal yet — the policy fully
  exploited a depleted intrinsic signal and has no exploration pressure
  left. This is the self-consumption failure mode Run 3 flagged; it needs
  a design response (disagreement annealing, entropy floor, or count-based
  fallback), not a longer run.
- The perennials: NaN/inf anywhere, sustained grad-norm blowup, resume
  discontinuities.

Decision this run should produce — one of three exits, pre-committed:

1. **Behavior change confirmed** (Good #1 including step 4): tickets/0005
   closes validated; next work is scale (budget, capacity, more games in
   parallel) and win-rate-oriented evaluation.
2. **Wiring confirmed, behavior unchanged** (steps 1–3 fire, scoring stays
   incidental): the bottleneck has moved past normalization to credit
   assignment — imagination horizon, `gamma`, or reward-to-action distance;
   write that ticket with this run's scalars as evidence.
3. **No scoring events at all:** exploration ticket (see Bad); rerunning
   0005 harder is explicitly not the move.

## Findings

(Filled in mid-run, July 10, at ~47k/100k env steps / ~23k grad steps —
per the conventions, a stopped run with an answered question beats a
finished one. **Recommendation: stop here.** The run answered its question
as pre-committed exit 2 — wiring confirmed, behavior not yet changed — and
surfaced the "Ugly" ext-critic destabilization in a slow-motion form; the
remaining ~2.5h would deepen a diagnosed pathology, not add information.
Fix is tickets/0006; resume this same output dir after it lands.)

**Wiring: the entire Good-#1 causal chain through step 3 fired, in order.**

- (1) Intrinsic-driven play scored: lp85 at env ~44.2k, sp80 at ~45.5k —
  2 scoring episodes of 104 completed (Run 3's incidental rate; step 4 not
  yet judgeable, ~10 episodes after the first event).
- (2) `train/reward_windows_in_batch` went 0 → 4 at grad step 21,438,
  within ~200 grad steps of the first scoring episode completing, and held
  at 4/16 — the stratified sampler hits its 25% target from 2 distinct
  events. The pre-registered "sampler can't find real events" Ugly did not
  fire.
- (3) The ext stream moved: `return_norm_scale_ext` 0.11 → 5.35,
  `imagined_return_ext`/`value_ext_mean` off ~0.03 together. Int stream
  stayed clean throughout (isolation working as designed).

**But step 3 kept going — the pre-registered Ugly, in slow motion.** Over
~2k grad steps after first contact, `policy/value_ext_mean` climbed 0.03 →
1.6 (peak 2.55), still rising monotonically at assessment; ext critic loss
spiked to 2.86 (700× baseline, grad_norm_critic to 11), settling ~10×
elevated; dream `extrinsic_reward_mean` reached ~0.03/step — **~1,000× the
real reward rate** (2 events in 47k steps). With `gamma=0.997` that
supports v ≈ 10; nothing bounded the climb.

**Checkpoint probes pinned the mechanism as world-model exploitation —
the actor farms the reward head in imagination:**

- 15-step dreams from reward-window starts collect mean **1.63** predicted
  extrinsic reward per dream (p90 3.2, max 4.3) vs the +1 a real level
  completion pays _once_; 11% of dream steps claim r > 0.5. Uniform-random
  dream actions from the same starts get 0.007 — the policy specifically
  learned re-triggering sequences. `loss/reward` on real data is fine
  (~1.5e-5); the head over-predicts only on imagined, policy-steered
  states no real data anchors (the model has seen 2 scoring transitions
  ever and can't know a completion consumes itself).
- Critic decoupled from reality everywhere: v_ext ≈ 6.2 at reward-window
  starts (true ≈ 1), ≈ 0.56 at uniform starts (true ≈ 0.001) — the
  bootstrap/target-sync loop feeding itself.
- **Disagreement does not flag the farmed states** (corr ≈ −0.01; equal
  disagreement at claimed-reward steps vs elsewhere) — an uncertainty
  penalty on dream reward would not bite; tested before being rejected in
  tickets/0006's non-goals.

**Everything else is healthy** — this is an objective bug, not a model
one: recon flat-good ~0.008–0.010 (≈ Run 3's final; not improving further,
same extend-before-tuning clause), samples sharp, imagination holds the
horizon, `kl_raw` ~0.04, `disagreement_mean` ~0.004 with per-game
structure (no self-consumption collapse), grad_norm ~0.2, no NaN/inf.
Entropy: one early dip (min 0.36 at grad ~1.3k) self-recovered to ~3,
then eased to 1.9–2.4 as ext advantages arrived — the <0.3-sustained
trigger never fired, no action-type frac above 0.28. New since the ext
wake-up: many games' episodes now end early (deaths at 45–150 steps on
~10 games vs Run 1's everything-at-cap), so the continue head finally has
real terminals.

The probe (rerun it against the post-0006 checkpoint per that ticket's
acceptance criterion 3 — per-dream ext λ-credit should be bounded ≈ ≤ 1):

```sh
PYTHONPATH=. uv run python - <<'EOF'
import torch
from model.thumper import Thumper
from training.replay_buffer import ReplayBuffer

torch.manual_seed(0)
th = Thumper.load("runs/two_stream_returns/latest.pt"); th.eval()
wm = th.world_model
buf = ReplayBuffer.load("runs/two_stream_returns/buffer.pt",
                        frame_stack=wm.config.frame_stack,
                        internal_state_dim=wm.config.internal_state_dim)
BURN, SEQ, H = 16, 16, 15
for name, frac in [("uniform starts", 0.0), ("reward-window starts", 1.0)]:
    b = buf.sample(16, BURN + SEQ, reward_frac=frac, loss_offset=BURN)
    with torch.no_grad():
        out = wm.forward_sequence(b["observations"], b["action_types"], b["coords"],
                                  b["is_first"], rewards=b["rewards"], burn_in=BURN)
        N = 16 * SEQ
        d = th.dream(out["deter"].reshape(N, -1), out["stoch"].reshape(N, -1),
                     out["macro_context"].reshape(N, -1),
                     b["available_actions"][:, BURN:].reshape(N, -1), horizon=H)
        vals = th.critic(d["features"])
    r, dis = d["reward"], d["intrinsic"]
    print(f"== {name} ==")
    print(f"  dream ext reward: mean/step={r.mean():.4f} per-dream sum mean={r.sum(1).mean():.3f} "
          f"p90={r.sum(1).quantile(0.9):.3f} max={r.sum(1).max():.3f}")
    print(f"  v_ext: t0 mean={vals[:, 0, 0].mean():.3f} max={vals[..., 0].max():.3f}")
    print(f"  corr(reward, disagreement)={torch.corrcoef(torch.stack([r.flatten(), dis.flatten()]))[0, 1]:.3f}")
EOF
```

Decision: exit 2, sharpened — the bottleneck moved past normalization not
to credit assignment but to **unbounded extrinsic returns in imagination**
(reward farming). tickets/0006 makes predicted scores absorbing for the
extrinsic λ-return (one dream banks at most ~one score, bounding critic
targets and deleting the farming incentive). No parameter shapes change;
Run 6 resumes this output dir, keeping the buffer's scoring events.

---

# Run 6 — tickets/0006 absorbing-score extrinsic returns (July 10, 2026)

Direct continuation of Run 5's diagnosis: the wiring (tickets/0005) and the
world model are both healthy, but the actor's extrinsic objective inside
imagination was an unbounded sum of a reward the environment pays at most
once per level, and it learned to farm the reward head for it. tickets/0006
makes a predicted score absorbing for the extrinsic λ-return only — the
discount chain past a claimed score is multiplied by `(1 - clamp(reward, 0,
1))`, so a single dream can bank at most ~one score's worth of extrinsic
credit. No parameter shapes change, so this **resumes `runs/two_stream_returns`
in place** rather than a fresh output dir: the buffer's 2 real scoring
episodes (lp85, sp80) and ~47k steps of on-policy data are the only signal
this fix has to work with, and `--init-from` would discard them along with
the warm world model.

```sh
# Same output dir as Run 5 (deliberate resume) -- no new flags, since the
# fix is entirely inside actor_critic_losses and carries no config surface.
uv run python train.py \
  --config.output-dir runs/two_stream_returns \
  --config.init-from runs/burn_in_fix/latest.pt \
  --config.total-env-steps 100000
```

## What to Look for

`uv run tensorboard --logdir runs/two_stream_returns/tb`. Grad steps here
continue Run 5's numbering (~23k at the point Run 5 was assessed and this
fix applied); "early" below means the few hundred grad steps immediately
after resume.

**Good:**

- `policy/value_ext_mean` falls from ~1.6 (Run 5's peak) back under ~1
  within a few hundred grad steps — the poisoned critic deflating now that
  its regression targets are bounded.
- `policy/return_norm_scale_ext` decays from ~5.35 as the return spread
  shrinks back toward O(1).
- `policy/dream_score_sum` (new metric, this ticket) settles and holds
  under 1 — the direct farming gauge; Run 5's checkpoint measured ~1.6 mean
  / 4.3 max from reward-window starts.
- ext critic loss comes back down off its ~10×-elevated plateau toward
  baseline (~4e-3 territory, per Run 5's pre-spike numbers).
- The actual behavioral question, carried over from Run 5: scoring recurs
  on lp85/sp80/cd82 more than incidentally now that the farming incentive
  is gone and the extrinsic advantage stays a clean, bounded signal.
- Everything Run 5 already had healthy stays healthy: recon ~0.009, sharp
  imagination samples, `wm/disagreement_mean` alive (~0.004), entropy
  holding 1.9–2.4, no NaN/inf.

**Bad:**

- `dream_score_sum` falls under 1 but scoring frequency in real play does
  _not_ recover from Run 5's incidental rate (2/104 episodes) — would mean
  absorption fixed the imagination pathology but the actor still isn't
  learning a _useful_ scoring policy from the bounded signal (a credit-
  assignment or exploration problem, not this ticket's problem to solve).
- `value_ext_mean` deflates but overshoots negative or oscillates instead
  of settling — would suggest the critic's regression targets are still
  noisy at this data volume (2 real events) regardless of the bound.

**Ugly:**

- `dream_score_sum` does not fall under 1, or the ext critic/return-scale
  metrics don't move from Run 5's inflated plateau at all — would mean the
  absorbing factor isn't reaching the extrinsic lambda-return as wired
  (implementation bug, re-check `actor_critic_losses` against tickets/0006
  before re-running).
- New instability appears in the _intrinsic_ stream (`value_int_mean`,
  `return_norm_scale_int` moving off their Run 5 baselines) — would mean
  the fix leaked into the intrinsic discount chain despite the isolation
  tests (tickets/0006's Step 4 test 5) passing in unit tests but not
  covering some interaction the full run exposes.

**Checkpoint probe, updated for this ticket.** Run 5's probe (reused above)
measured raw dream reward sums, which absorption never touches by design —
the fix operates on the λ-return's discount chain, downstream of the reward
head's raw predictions. A checkpoint can still show per-dream reward sums

> 1 post-fix; that alone proves nothing about tickets/0006. The quantity
> the acceptance criterion actually names is the extrinsic λ-return, `R_ext`,
> computed with the absorbing discount:

```sh
PYTHONPATH=. uv run python - <<'EOF'
import torch
from model.thumper import Thumper
from training.actor_critic import ActorCriticConfig, lambda_returns
from training.replay_buffer import ReplayBuffer

torch.manual_seed(0)
th = Thumper.load("runs/two_stream_returns/latest.pt"); th.eval()
wm = th.world_model
buf = ReplayBuffer.load("runs/two_stream_returns/buffer.pt",
                        frame_stack=wm.config.frame_stack,
                        internal_state_dim=wm.config.internal_state_dim)
cfg = ActorCriticConfig()
BURN, SEQ, H = 16, 16, 15
for name, frac in [("uniform starts", 0.0), ("reward-window starts", 1.0)]:
    b = buf.sample(16, BURN + SEQ, reward_frac=frac, loss_offset=BURN)
    with torch.no_grad():
        out = wm.forward_sequence(b["observations"], b["action_types"], b["coords"],
                                  b["is_first"], rewards=b["rewards"], burn_in=BURN)
        N = 16 * SEQ
        d = th.dream(out["deter"].reshape(N, -1), out["stoch"].reshape(N, -1),
                     out["macro_context"].reshape(N, -1),
                     b["available_actions"][:, BURN:].reshape(N, -1), horizon=H)
        target_values = th.critic_target(d["features"])

    discounts = cfg.gamma * d["continue_prob"]
    r = d["reward"]
    print(f"== {name} ==")
    print(f"  dream ext reward (raw, NOT what this ticket bounds): "
          f"per-dream sum mean={r.sum(1).mean():.3f} max={r.sum(1).max():.3f}")

    # Pre-fix: what R_ext would have been with the unmodified discount chain.
    returns_ext_unbounded = lambda_returns(r, discounts, target_values[..., 0], cfg.return_lambda)
    # Post-fix (tickets/0006): absorbing discount -- this is the quantity
    # the acceptance criterion bounds to ~<= 1.
    absorb = 1.0 - r.clamp(0.0, 1.0)
    returns_ext_bounded = lambda_returns(r, discounts * absorb, target_values[..., 0], cfg.return_lambda)

    print(f"  R_ext, pre-fix discount:  t0 mean={returns_ext_unbounded[:, 0].mean():.3f} "
          f"max={returns_ext_unbounded[:, 0].max():.3f}")
    print(f"  R_ext, absorbing (0006): t0 mean={returns_ext_bounded[:, 0].mean():.3f} "
          f"max={returns_ext_bounded[:, 0].max():.3f}")
EOF
```

Reading this probe's output: `R_ext, absorbing (0006)` is the number to
check against criterion 3 (bounded ≈ ≤ 1 from reward-window starts). If the
checkpoint being probed was trained _before_ the fix landed in
`actor_critic_losses` (i.e. its critic/policy were shaped by the unbounded
objective), `R_ext, absorbing` will still be computed correctly here since
this script applies the absorbing discount itself regardless of what the
checkpoint's training loop used — but `value_ext_mean`/`return_norm_scale_ext`
baked into that checkpoint's critic will still reflect the old, inflated
training. Only a checkpoint whose _training_ used the fix (i.e. produced by
rerunning `train.py` after this ticket landed) validates the fix end to
end; probing an old checkpoint with the new formula validates the formula,
not the training outcome.

Sanity-run against Run 5's checkpoint (pre-fix training, formula applied
post-hoc) confirms exactly that split: at reward-window starts, `R_ext`
drops from a pre-fix-discount 6.61 mean (max 9.35) to an absorbing 2.19
mean (max 4.41) — the discount change visibly caps the tail, but it's still

> 1, because `target_values` here come from `critic_target` shaped by
> ~23k grad steps of the _unbounded_ objective (v*ext ≈ 6.6 baked in). This
> is the expected shape of evidence from an old checkpoint: the formula bites
> immediately, but criterion 3's ≤ 1 bound is a property of a critic \_trained*
> under the fix, which only the Run 6 resume produces.

## Findings

Run completed at env_steps=100000, grad_steps=49500 (checkpoint counters;
Run 6's span is the ~26.5k grad steps after the ~23k resume point). See
`runs/two_stream_returns/eval_100000_prefix.json` /
`eval_100000_postfix.json` for the raw per-episode data behind the tables
below.

**Verdict: the pre-registered Good column, near-verbatim.** The absorbing
discount deflated the poisoned extrinsic stream and held it bounded for
26k grad steps of continued training; no Bad or Ugly trigger fired.

- **Ext stream deflated cleanly.** One brief resume transient
  (`value_ext_mean` 1.27 → 3.28 and `return_norm_scale_ext` to 9.8 within
  the first ~1k grad steps — the old critic's inflated targets flushing
  through the new bounded objective) then monotone decay: `value_ext_mean`
  0.98 at 26k → 0.36–0.48 tail, `return_norm_scale_ext` 4.08 → 0.79. No
  negative overshoot, no oscillation — pre-registered Bad #2 did not fire.
  `policy/critic_loss` came off Run 5's elevated plateau back to
  ~5e-3–1.3e-2.
- **`dream_score_sum` (the direct farming gauge) never re-crossed 1**
  after the resume transient: range ~0.01–0.67 across the span, tail mean
  0.23, final 0.12 — vs Run 5's checkpoint measuring 1.63 mean / 4.3 max
  per dream.
- **Criterion 3 (the R*ext probe) passes on a critic \_trained* under the
  fix** — the half the Run 5 sanity-run explicitly couldn't show. From
  reward-window starts: absorbing `R_ext` t0 mean 0.737 / max **1.003**
  (uniform starts: 0.384 / 0.990). The same dreams under the pre-fix
  discount give 1.10 mean / 6.27 max, and raw per-dream reward _sums_
  still reach 7.3 — the reward head can still imagine re-triggering
  sequences, but the λ-return now pays for at most one of them. That is
  exactly the designed bound: the incentive is deleted downstream of the
  head, not papered over by retraining it.
- **World model untouched, as designed:** recon ~0.0064, `loss/reward`
  ~4e-6 on real data, `disagreement_mean` alive at 0.002–0.0065 with
  per-game structure, `kl_raw` 0.026–0.066 (see the units correction
  below — this straddles the true per-dim floor of ~0.031, so the KL term
  is active, not inert), final recon/imagination samples sharp with
  rollouts holding the full horizon.
- **Behavior: real scoring recurred, modestly.** In Run 6's span: cd82
  scored in 3 of 7 collection episodes (0 in all of Run 5), lp85 3 of 5
  (1 in Run 5), sp80 0 of 5 (down from Run 5's 1 — though the final
  checkpoint scores 1.00 on sp80 at eval in both modes, so this is
  small-sample collection noise, not a lost skill). Zero wins anywhere,
  still. So pre-registered Bad #1 half-fired: absorption fixed the
  imagination pathology and scoring now recurs _above_ the incidental
  rate on 2 of 3 games, but nothing yet looks like a policy that seeks
  levels. The standing credit-assignment/exploration gap is now cleanly
  the top open problem — with the generalization question (tickets/0009)
  standing right next to it.

**Watch items (off-script — neither pre-registered Good nor Bad):**

- `policy/entropy` runs hot: 2.2–5.3 across the span (tail ~3.7–4.3) vs
  Run 5's settled 1.9–2.4. Consistent with the advantage mix now being
  intrinsic-dominated (`value_int_mean` ~3.3–5.5 vs `value_ext_mean`
  ~0.4): with the ext stream no longer hallucinating dense reward, the
  actor's objective is mostly normalized exploration bonus plus entropy
  regularizer. Not a failure at this stage of learning, but it explains
  the sampled-mode eval churn below, and any future extrinsic-dominated
  phase should pull this back down.
- `return_norm_scale_int` drifted 8.3 → ~13 (peak 15.2) while
  `value_int_mean` and `disagreement_mean` stayed flat — the _spread_ of
  intrinsic returns is widening, not the mean. The normalizer divides it
  out of the advantage, so no action now; carry into the next run as a
  watch item.

**Log correction (applies to Runs 1/3/5 above, recorded here rather than
rewriting their Findings):** those entries compare `loss/kl_raw` — a
per-dimension mean — against the **1.0 total** free-nats budget and
conclude the KL term is "fully clamped and inert". `compute_losses`
applies the floor per-dim (`free_nats / stoch_dim` = 1/32 ≈ 0.031), and
kl_raw has ranged ~0.03–0.07 across runs — at or above the floor much of
the time, so the KL gradient was active all along. No decision taken in
those runs changes (recon health, the Run 3 burn-in verdict, and the
Run 5 farming diagnosis never hinged on KL being inert), but the claim
should not be repeated. Inline corrections added at the two spots.

### tickets/0008 Step 4 — acting-loop alignment fix, before/after eval

tickets/0008 found that `OnlineActor`'s TaskEncoder fold was misaligned by
one step relative to the convention `forward_sequence` actually trains
under (folding `(s_t, a_{t+1}, r_{t+1})` online vs `(s_t, a_t, r_t)` in
training) — every act-time macro-context, at every step of every episode
ever collected or evaluated, was built off-distribution from what the
TaskEncoder was trained to interpret. This run's final checkpoint is the
first one whose _training data collection_ also ran through the misaligned
loop (all of Runs 3–6), so it's the natural checkpoint to measure the
fix's isolated effect on: same weights, same eval protocol (25 games x 5
episodes x greedy+sampled, `--config.timeout_env_steps`-capped), same seed
— the only variable toggled is `training/online_actor.py`'s fold
alignment.

**Pre-registered reading** (stated before running the post-fix table, per
the ticket): the post-fix table should be no worse than the pre-fix table,
and any improvement is a direct measure of how much the misaligned
macro-context was costing the policy at act time. This is a zero-shot
correctness check on the _acting_ loop only — the world model/policy
weights are unchanged between the two runs; nothing here retrains anything.

**Pre-fix** (`training/online_actor.py` reverted to the commit before
tickets/0008's fix landed, restored afterward — see `eval_100000_prefix.json`):

```
game         mode     mean_score max_score win_rate mean_len steps_to_1st
cd82         greedy         0.20         1     0.00    101.4          6.0
lp85         greedy         1.00         1     0.00    600.0          6.6
sp80         greedy         1.00         1     0.00     91.4         11.4
vc33         greedy         0.20         1     0.00    600.0         21.4
  -> games_scored=4 games_won=0 (greedy)  [21 other games: 0.00]
cd82         sampled        0.40         1     0.00    104.0          7.5
lp85         sampled        0.40         1     0.00    269.4          5.5
r11l         sampled        0.00         0     0.00    600.0          inf
sp80         sampled        1.00         1     0.00     55.4         20.8
  -> games_scored=3 games_won=0 (sampled)  [r11l shown at 0.00 for the
     post-fix comparison; 21 other games also 0.00]
```

**Post-fix** (fix restored, `git diff training/online_actor.py` empty
against HEAD — see `eval_100000_postfix.json`):

```
game         mode     mean_score max_score win_rate mean_len steps_to_1st
cd82         greedy         0.40         1     0.00    103.0          5.0
lp85         greedy         1.00         1     0.00    600.0          5.4
sp80         greedy         1.00         1     0.00     67.0         11.8
vc33         greedy         0.40         1     0.00    600.0         17.8
  -> games_scored=4 games_won=0 (greedy)  [21 other games: 0.00]
lp85         sampled        0.80         1     0.00    501.6          6.0
r11l         sampled        0.20         1     0.00    532.8        179.0
sp80         sampled        1.00         1     0.00     71.0         17.2
  -> games_scored=3 games_won=0 (sampled)  [22 other games: 0.00, including cd82/vc33 which scored pre-fix]
```

**Reading against the pre-registration:**

- No regression on the headline aggregate: `games_scored` is unchanged in
  both modes (greedy 4/25, sampled 3/25 — the pre-fix table's original
  "sampled=4" footer double-counted r11l's 0.00 row; corrected against
  the raw JSON), though the sampled game set churns — `r11l` (zero
  pre-fix) starts scoring post-fix
  (mean 0.20, first score at step 179), while `cd82`/`vc33` drop to zero
  under `sampled`'s stochastic action noise. `greedy` (the deterministic,
  lower-variance read of the policy) is unambiguously flat-to-better: `cd82`
  mean_score 0.20→0.40, `vc33` 0.20→0.40, `sp80`/`lp85` unchanged at 1.00,
  and `steps_to_1st` improves on 3 of 4 scoring games (cd82 6.0→5.0, lp85
  6.6→5.4, sp80 11.4→11.8 is the one exception, +0.4 steps).
- No new instability: no NaN/inf, no game that scored pre-fix goes to
  literal zero under `greedy`, mean episode lengths move by single-digit
  percentages except where a game's termination behavior is itself
  stochastic (sp80, the natural-terminal game).
- Still zero wins in both tables — this fix corrects act-time
  distributional alignment, not the separate credit-assignment/exploration
  gap Run 5/6 already diagnosed (incidental-rate scoring, no sustained
  win-seeking policy yet). Consistent with the ticket's own framing: this
  measures what the misaligned `m` was costing the policy, not a fix for
  the scoring-frequency problem.

**Verdict:** matches the pre-registered "no worse, possible improvement"
reading. `sampled` mode's per-game churn (r11l gaining, cd82/vc33 losing)
is within what's expected from a single-seed, 5-episode-per-game sample at
this data volume and shouldn't be read as a regression — `greedy` mode,
which isolates the policy's mode from sampling noise, shows a clean
improvement or parity on every scoring game. The next run (Run 7) trains
online collection through the aligned loop for the first time; per
tickets/0008 Step 5, its `online/*` scalars are not strictly comparable to
Runs 3–6's for that reason.

---

# Run 7 — tickets/0009 held-out-games generalization protocol, from scratch (July 10, 2026)

Runs 1–6 answered "can Thumper learn the games it practices on": world
model healthy (Run 3), two-stream returns wired (Run 5), extrinsic
objective bounded (Run 6), acting loop aligned (tickets/0008). What no run
has measured is the actual competition: performance on a game that
contributed **zero** training data. Every architectural claim of
"generalizes across games" — the TaskEncoder macro-context above all —
is untested on that claim; a macro-context that memorizes 25 game
identities and one that infers rules from transitions are
indistinguishable on training games. This run splits the 25 downloaded
games 20/5 per tickets/0009's pre-registered split and reads zero-shot
transfer directly.

Three run-defining constraints, all from tickets/0009:

- **From scratch** — no `--config.init-from`: every existing checkpoint's
  world model trained on all 25 games, so warm-starting bakes held-out
  dynamics into the weights undetectably. This also makes Run 7 the first
  run whose _entire_ collection goes through the tickets/0008-aligned
  acting loop (Runs 3–6 collected through the misaligned fold), so its
  `online/*` curves double as the clean post-fix baseline — at the price
  that they are not strictly comparable to Runs 3–6's (per 0008 Step 5).
- **Fresh output dir** — the resume guard would (correctly) refuse
  `runs/two_stream_returns` anyway: its checkpoint's `train_games=[]`.
- **The split is fixed in advance** (tickets/0009 Design principle 2):
  held out `cd82, r11l, ft09, sk48, wa30`. cd82/r11l have demonstrated
  score reachability (Run 3), so held-out scoring on them is achievable —
  the eval is informative, not vacuously zero. lp85 (live extrinsic
  signal) and sp80 (the only natural terminator; the continue head's real
  terminals) deliberately stay in training. ft09/sk48/wa30 are
  arbitrary-but-fixed unknowns for the harder zero-shot case.

```sh
# From scratch (NO --config.init-from), fresh output dir, 20-game training
# round-robin (~5k steps/game at 100k total vs 4k/game in 25-game runs),
# periodic zero-shot eval on the 5 held-out games every 10k env steps
# (1 episode/game to keep the cadence affordable; the final full sweep
# below is the real measurement).
uv run python train.py \
  --config.output-dir runs/held_out_v1 \
  --config.train-games ar25 bp35 cn04 dc22 g50t ka59 lf52 lp85 ls20 m0r0 re86 s5i5 sb26 sc25 sp80 su15 tn36 tr87 tu93 vc33 \
  --config.eval-every 10000 \
  --config.eval-games cd82 r11l ft09 sk48 wa30 \
  --config.eval-episodes-per-game 1

# After the run: the full pre-registered protocol, all 25 games, both modes.
uv run python eval.py --checkpoint runs/held_out_v1/latest.pt
```

Budget: 100k env steps at Run 3+ throughput (~5.5 env steps/s) ≈ 5h.

## What to Look for

`uv run tensorboard --logdir runs/held_out_v1/tb`. Read the final sweep as
tickets/0009's three numbers: (1) train-set score — did losing 5 games
cost learning?; (2) held-out cd82/r11l — zero-shot transfer where scoring
is known reachable, **the headline**; (3) held-out ft09/sk48/wa30 — the
harder unknowns. The honest baseline for (2)/(3) is what _random play_
achieves on those games (Run 1's collection data): "beats random
zero-shot" is the first defensible claim of generalization, pre-registered
here before any number exists.

**Good:**

- From-scratch training rediscovers the Runs 3–6 arc on 20 games with no
  regression in kind: recon < 0.01 with sharp samples by mid-run,
  `dream_score_sum` bounded < 1 throughout (the 0006 objective holding
  from step 0, not just after a resume), `value_ext_mean` sane, scoring
  recurs on lp85/sp80 (the in-train scorers) at ≥ Run 6's rate, and the
  final sweep's train-set table is not worse than Run 6's eval on the
  same 20 games.
- **The headline:** `eval/greedy/score/cd82` or `.../r11l` goes nonzero at
  any eval checkpoint, and the final sweep confirms held-out mean_score >
  0 on either — zero-shot scoring on a never-trained game, above what
  random play produces there. That is the first direct evidence the
  macro-context infers rather than memorizes.
- `online/macro_context_norm` behaves on train games as in Runs 3–6
  (O(1), wandering), and nothing pathological appears in eval on held-out
  games (the eval harness exercising `m` on out-of-distribution dynamics
  for the first time with aligned folds).

**Bad:**

- Train-set learning healthy but held-out **all-zero everywhere** —
  every periodic eval and the final sweep, all 5 games, both modes, while
  random play does better than zero on cd82/r11l. That is a clean negative
  answer: the macro-context (as trained, at this scale/data) memorizes.
  The response is not a bigger run; it is prioritizing test-time
  adaptation (the 0010+ ticket sketched in tickets/0009's non-goals) over
  zero-shot hope.
- Train-set score at 100k clearly below Run 6's on the same games — the
  held-out games mattered to training more than expected (unlikely at
  +25% steps/game; if it happens, suspect the from-scratch/warm-start
  difference first, since Runs 4–6 all rode Run 3's world model).

**Ugly:**

- Any held-out game appears in `online/*` scalars or the buffer, or the
  run starts against a non-empty dir without tripping the resume guard —
  tickets/0009 implementation bug; stop, fix, restart fresh (the
  measurement is unsalvageable once contaminated).
- From-scratch training fails to rediscover Run 3-era basics at matched
  step counts (recon stuck high, imagination collapsing, zero scoring on
  lp85/sp80 by 100k) — would mean Runs 4–6's competence was leaning on
  the warm-start lineage harder than believed, and the from-scratch story
  needs its own investigation before any generalization claim is made.

Watch items carried from Run 6: `policy/entropy` hot (2.2–5.3) and
`return_norm_scale_int` drift (8→13) — log their Run 7 trajectories in
Findings either way; a from-scratch run tells us whether they are
resume artifacts or steady-state properties of the current objective.

## Findings

(Filled in July 10, evening, from the run's TB scalars, samples, and
checkpoint after completion. The pre-registered **final full 25-game
`eval.py` sweep has not been run yet** — the three tickets/0009 numbers
below are read from the 9 periodic 1-episode-per-game evals and are
interim until that sweep lands. A stray `eval_60000.json` in the run dir
is a manual single-episode smoke run of `eval.py` against the 60k
checkpoint (one cd82 greedy episode, 20 steps, 0 score), not a protocol
sweep.)

**Run completed:** env_steps=100000, grad_steps=49,500, ~5.6 env steps/s
(≈5h compute across 4 kill/resume cycles — 5 TB event files with
continuous counters, no discontinuities). **No contamination:** only the
20 train games ever appear in `online/*` and `wm/disagreement_by_game/*`;
the 5 held-out games appear only under `eval/*`. Ugly #1 did not fire.

**Headline (interim): zero-shot transfer is essentially zero — but the
pre-registered Bad-#1 "memorizes" verdict cannot be cleanly issued,
because its premise (train-set learning healthy) failed on the scoring
axis.** Two findings, in tension:

- **Held-out eval: all-zero on final score, every checkpoint.**
  `eval/{greedy,sampled}/score/*` is 0.00 for all 5 held-out games at all
  9 eval points (10k–95k), both modes; zero wins. One nuance the scalars
  almost hide: `eval/sampled/games_scored` reads **1** at 85k and 95k
  while every per-game score reads 0 — `games_scored` counts a finite
  `steps_to_first_score` (any transient nonzero-reward event) while
  `score/<game>` logs _final_ `levels_completed`, so one held-out game
  had a scoring event under sampled play late in training and then lost
  the level again before the episode ended. TB doesn't record which game.
  At 1 episode/game this is at noise level; the full sweep must settle it.
- **The from-scratch train-set arc did not rediscover Runs 4–6's scoring
  — pre-registered Ugly #2, half-fired.** Exactly **one** scoring episode
  in the entire run: sp80 at env ~40.5k. lp85 — a live scorer in Runs
  5–6 (3-of-5 episodes in Run 6's span) — never scored in 12 collection
  episodes; nothing else scored either, and `online/win_rate/*` stayed 0.
  So Runs 4–6's scoring competence was leaning on the warm-start lineage
  (Run 3's world model + accumulated on-objective buffer) harder than
  believed. Until a from-scratch run scores on _training_ games at
  Run 6's rate, "held-out all-zero" is evidence about exploration/sample
  budget, not yet about macro-context memorization vs. inference.

**The world model itself is healthy from scratch** — Ugly #2's
model-side clauses did not fire: `loss/recon` median 0.057 → 0.0066 by
45k+ (final logged batch 0.0022), recon samples near pixel-perfect,
imagination samples from 15k on hold game identity and structure across
the full horizon on visually distinct games. `kl_raw` 0.045 → 0.019
(straddling the ~0.031 per-dim floor), `train/grad_norm` settling ~0.2,
no NaN/inf. `wm/disagreement_mean` 0.012 → 0.0022 with per-game
structure — alive, same depletion shape as Runs 3–6.

**The tickets/0006 objective held from step 0, not just post-resume**
(the Good-#1 clause it owned): `policy/dream_score_sum` bounded below 1
for ~99% of grad steps (tail median ~0.5; brief single-batch transients,
max 4.48, always immediately recovered), `value_ext_mean` never above
0.6, `return_norm_scale_ext` settling ~1.0. The stratified sampler
engaged off a **single** real event: `train/reward_windows_in_batch`
went 0 → 4 at grad step 19,750, immediately after sp80's lone scoring
episode completed (env 40,553 ≈ grad ~19.8k), and held 4/16 for the
remaining ~30k grad steps — the whole late-run ext stream (scale_ext
0.10 → 1.03, value_ext 0.01 → ~0.38) is powered by one episode.

**Watch items from Run 6, now with a from-scratch data point:**

- `policy/entropy`: ran ~2.9–3.1 early, dipped hard exactly when ext
  advantages arrived (~grad 20k; 8 logged points under 0.5, min 0.32),
  self-recovered to ~2.0–2.3, ending ~2.7. The <0.3-sustained trigger
  never fired; no action-type frac above 0.25. Run 6's "hot" 3.7–4.3
  tail looks like a resume artifact; ~2 nats is the steady state.
- `return_norm_scale_int`: rose 1.3 → ~9.1 by 15k, then _decayed_ to
  ~5 — Run 6's upward drift is not a from-scratch property and not
  runaway. Keep as a passive watch item.
- New: `online/macro_context_norm` runs much hotter from scratch —
  median ~12 over the first 15k env steps (max 67 at env ~2.3k) before
  decaying to Run 3's ~1.7–2 band by 60k+. Early-training `m` is
  large and unregularized exactly when the held-out eval exercises it
  out-of-distribution; worth remembering when reading early eval points.

**Decision (interim, pending the final sweep):** neither pre-registered
exit fires cleanly. The zero-shot answer so far is "no transfer
observed, one ambiguous transient" — but the run's own train-set scoring
regression (Ugly #2, half-fired) means the memorize-vs-infer question is
confounded by exploration/sample budget. Both readings point the same
way as Bad #1's prescribed response anyway: **prioritize test-time
adaptation (tickets/0010, `adapt.py` — already in flight) over zero-shot
hope**, and treat the from-scratch scoring shortfall as its own open
question (warm-start lineage vs. budget) rather than rerunning this
protocol bigger. Before closing this entry: run the pre-registered
`uv run python eval.py --checkpoint runs/held_out_v1/latest.pt` and
append its 25-game table here.

### Final full sweep (July 10, evening — `eval_100000.json`, 250 episodes)

The full 25-game × 5-episode × both-modes protocol, closing this entry.
Scoring rows only (all 42 other game×mode rows are 0.00 across the board,
zero wins anywhere):

```
game         mode     mean_score max_score win_rate mean_len steps_to_1st
sp80         greedy         0.00         0     0.00    600.0         18.6
  -> games_scored=1 games_won=0 (greedy)
cd82         sampled        0.40         1     0.00    196.2         79.5
r11l         sampled        0.00         0     0.00    600.0        229.0
sp80         sampled        0.00         0     0.00    600.0         16.2
  -> games_scored=3 games_won=0 (sampled)
```

Three refinements to the interim reading above:

- **The one nonzero final score in the whole table is on a held-out
  game.** cd82 (never trained on) completes a level and _keeps_ it in
  2 of 5 sampled episodes (first scores at steps 20 and 139). r11l
  (also held-out) has transient scoring events in 2 of 5 sampled
  episodes (~steps 222/236) but ends both at 0. This resolves the
  85k/95k `games_scored=1` mystery — it was cd82/r11l's transient-or-
  retained sampled scoring, invisible in the final-levels scalar.
  **Caveat before calling it transfer:** cd82/r11l were held out
  precisely because incidental play can score them (Run 3 precedent),
  greedy scores 0 on both, and the "beats random play" bar
  pre-registered above hasn't been checked against a random baseline —
  so this is a glimmer, not a claim.
- **The train-set regression is confirmed and worse than the collection
  scalars suggested — pre-registered Bad #2 fired.** Zero final score
  on all 20 train games in both modes (Run 6's checkpoint: 4 games
  scored greedy, lp85/sp80 at mean 1.00). sp80 scores a level within
  ~18 steps in _all 10_ of its episodes, both modes, then loses it and
  rides the 600-step cap every time (it also no longer terminates
  naturally — Runs 1–6's only natural terminator). lp85: nothing, 10/10.
- Net: the from-scratch/warm-start confound now dominates the
  tickets/0009 question. The zero-shot table exists (all-but-zero, one
  sampled-mode glimmer on cd82), which is exactly the precondition
  tickets/0009's non-goals set for prioritizing test-time adaptation.
  **Proceed to Run 8 (tickets/0010, `adapt.py`) on this checkpoint**,
  whose frozen-eval cells double as a reproduction check of this table.

---

# Run 8 — tickets/0010 test-time adaptation on the held-out games (July 10, 2026, overnight)

Runs on Run 7's checkpoint: its zero-shot table (`eval_100000.json`,
appended above) is the baseline this run exists to beat. Per game, per
tickets/0010: eval the frozen checkpoint (carry off/on), warm-start a
fresh single-game `Trainer` from its **full** weights
(`init_from_full` — world model + policy + both critics; fresh
optimizers/normalizers/buffer), train for a fixed 20k-env-step budget
(500 random prefill included), eval the adapted weights (carry off/on),
and write `<output_dir>/<game>/adapt_report.json` with all four cells
plus `env_steps_to_first_score`. The novelty guard reads the source
checkpoint's `train_games` and refuses any game in it — all five targets
here are the held-out set, so it must pass silently.

Two pre-registered questions (tickets/0010 Step 7): **(1)** does 20k
steps of test-time training beat zero-shot where scoring is known
reachable (cd82/r11l — the headline; cd82's frozen sampled 0.40 is the
number to beat)? **(2)** does carried `m` (Arm A) help or hurt, and does
the answer differ between frozen and adapted weights?

```sh
# Ticket 0010 Step 7's pre-registered command, as written. Defaults:
# budget=20000, prefill=500, eval-every=5000, episodes-per-game=5.
# Resume is automatic per game: rerunning picks each <game>/ dir up
# from its latest.pt, so an interrupted overnight run just continues.
uv run python adapt.py \
  --checkpoint runs/held_out_v1/latest.pt \
  --games cd82 r11l ft09 sk48 wa30 \
  --output-dir runs/adapt_v1
```

Budget arithmetic: 5 games × 20k adaptation steps ≈ 100k env steps ≈ 5h
at Run 3+'s ~5.5 steps/s, plus eval cells (per game: 4 cells × 2 modes ×
5 episodes × ≤600 steps ≈ up to 24k interactions) and the in-adaptation
`eval_every=5000` curve — call it 8–12h wall-clock. An overnight run;
if it must be cut short, completed games have full reports (per-game
loop), so partial results are still readable.

## What to Look for

Read each game's `adapt_report.json` as a 2×2 (frozen/adapted × carry
off/on); TB per game under `runs/adapt_v1/<game>/tb`.

**Good:**

- **Sanity anchor first:** each game's frozen, carry-off cell reproduces
  Run 7's `eval_100000.json` rows for that game (same weights, protocol,
  seed). Any large disagreement is an `init_from_full`/plumbing bug and
  invalidates the rest of that game's report — check this before reading
  anything else.
- **The headline (Q1):** adapted beats frozen on cd82 and/or r11l —
  greedy going nonzero (frozen greedy is 0.00 everywhere), sampled
  mean_score above cd82's 0.40, `env_steps_to_first_score` well inside
  the 20k budget, and the `eval_every=5000` in-adaptation curve rising
  rather than flat. Any of these on ft09/sk48/wa30 (no known scoring
  path) would be a stronger, unexpected win.
- **Q2 answered either way:** carry-on cells measurably differ from
  carry-off (helping _or_ hurting is informative; Arm A is a measured
  ablation, pre-registered as such). No crashes/NaN from `m` running
  out-of-distribution across episode resets — note Run 7's finding that
  early-training `m` norms run hot.
- **Adaptation internals healthy on a single game:** recon falling
  further from the warm start, `dream_score_sum` bounded < 1,
  fresh normalizer scales re-estimating within a few hundred grad steps,
  no NaN/inf. Novelty guard silent on all five games; each game's buffer
  contains only that game.

**Bad (report as negative result, don't knob-tune):**

- Adapted ≈ frozen across all four cells on cd82/r11l — 20k steps is too
  small a budget or single-game data too thin for the current objective;
  the next lever is budget or an adaptation-specific objective
  (intrinsic_scale annealing), decided in a follow-up ticket, not here.
- Adaptation _degrades_ cd82's frozen sampled 0.40 — plausible failure:
  the intrinsic-dominated objective steers the policy away from the
  scoring behavior it already had. Would directly motivate down-weighting
  intrinsic reward at test time; capture the per-cell numbers as the
  evidence.
- ft09/sk48/wa30 all-zero in every cell — expected-ish (no known scoring
  path); log it as the difficulty read on the harder unknowns.

**Ugly (stop and investigate):**

- The frozen-cell sanity anchor fails (see Good #1) — `init_from_full`
  is not faithfully restoring the agent; nothing downstream is readable.
- The novelty guard fails open (runs a trained game without
  `--allow-trained-game`) or any adaptation buffer contains another
  game's episodes — contamination; the run is unsalvageable as designed.
- NaN/instability shortly after adaptation starts — suspect the fresh
  return normalizers meeting the warm critics (the one deliberately
  fresh/warm seam in `init_from_full`).

Decision this run should produce, pre-committed: if adapted > frozen on
either known-reachable game, tickets/0010 validates and test-time
adaptation becomes the default eval story (follow-up: budget scaling,
intrinsic annealing). If flat everywhere, the negative result plus
Run 7's from-scratch scoring regression together say the _training-time_
policy/exploration gap is the binding constraint — go back there before
buying more adaptation machinery.

## Findings

(Filled in July 11, morning after the overnight run. All five games
completed — per-game reports written 21:25 July 10 (cd82) through 01:10
July 11 (wa30). Condensed four-cell table below; full per-episode data in
each `runs/adapt_v1/<game>/adapt_report.json`, per-game TB under
`runs/adapt_v1/<game>/tb`.)

**Verdict: tickets/0010 validates — the pre-committed "adapted > frozen
on a known-reachable game" exit fires on cd82.** Adaptation turned a
policy that never trained on cd82 from greedy 0.00 / sampled 0.20 into
greedy 0.80 / sampled 1.00, and the in-adaptation `eval_every=5000`
curve rose rather than spiking only at the end (sampled 0 → 0.40 → 0.20
→ 1.00 at 5k/10k/15k/20k; greedy 0 → 0.80 at 20k). First real score at
env step 6,763 of the 20k budget; collection episodes at ~19.4k–19.9k
score 1.0 back-to-back. Per the pre-committed decision: test-time
adaptation becomes the default eval story.

```
game  cell                 greedy  sampled   env_steps_to_first_score
cd82  frozen,  carry=off     0.00     0.20   6763
cd82  frozen,  carry=on      0.00     0.40
cd82  adapted, carry=off     0.80     1.00
cd82  adapted, carry=on      0.80     1.00
r11l  all four cells         0.00     0.00   1103
ft09  all four cells         0.00     0.00   None
sk48  all four cells         0.00     0.00   None
wa30  all four cells         0.00     0.00   None
```

(mean_levels_completed; win_rate 0.00 everywhere.)

**Sanity anchor: passed in the only sense it could — and the
pre-registration was flawed, not the plumbing.** The frozen cells cannot
reproduce Run 7's *sampled* rows exactly: `evaluate` reseeds once per
**mode** and all games in a sweep share the RNG stream, so a single-game
eval puts that game at a different stream position than the 25-game
sweep did (cd82 frozen sampled 0.20 here vs 0.40 there — one episode of
five at n=5, RNG offset, not weights drift). The deterministic check —
greedy, which doesn't consume the sampling stream — matches exactly
(0.00 = 0.00 on every game). No `init_from_full` fault indicated; the
Ugly anchor-failure trigger did not fire. (Follow-up if exact sampled
reproduction ever matters: reseed per (game, mode) in `evaluate`.)

**r11l: the pre-registered "adaptation degrades" Bad case fired in
substance, and it's the most actionable finding of the run.**
Adaptation scored once, very early (env step 1,103 — faster than cd82)
and never again in 20k steps. Worse, the adapted policy *lost* the
frozen policy's transient scoring events: frozen sampled had reward
events at steps 359/501 (ending at 0, as in Run 7); adapted has zero
events in any cell. TB shows the mechanism: `policy/entropy` collapsed
from ~7.4 early to ~0.58 at the tail (vs cd82's healthy ~1.5),
`value_ext_mean` inflated to ~0.76 anchored by that single event, and
adapted greedy switched from frozen's 62-step deaths to riding the
600-step cap — 20k steps of intrinsic-dominated training taught it
survival and disagreement-farming, not scoring. This is the direct
evidence for the follow-up the pre-registration named: down-weight or
anneal `intrinsic_scale` during test-time adaptation (ticket it; the
next adapt run should carry an `--intrinsic-scale` arm).

**ft09/sk48/wa30: all-zero in every cell** — the pre-registered
expected-ish read on the no-known-scoring-path games; sk48/wa30 ride
the cap in every cell, before and after. One oddity worth keeping:
ft09's *adapted* policy terminates every episode at ~32 steps (frozen
greedy varied 36–251) — consistent with learning to reach a fast reset
to farm the novelty around episode starts. Same intrinsic-objective
signature as r11l, different surface behavior; more fuel for the
annealing ticket.

**Q2 (carried macro-context): essentially neutral — answered, as
pre-registered, just not excitingly.** Frozen cd82: carry-on sampled
0.40 vs carry-off 0.20 — one extra scoring episode at n=5, noise-level.
Adapted cells: carry-on and carry-off are byte-identical on cd82 and
ft09 (the adapted, low-entropy policy takes the same actions either
way; frozen cd82's cells *do* diverge at episode 5, so the flag
demonstrably reaches the actor). No crashes/NaN from out-of-distribution
`m`. Arm A neither helps nor hurts measurably at this budget.

**Adaptation internals healthy on all five games** — no Ugly trigger
fired: no NaN/inf anywhere; `policy/dream_score_sum` tails bounded well
under 1 (brief single-batch transients to ~6, immediately recovered —
Run 7's shape); recon fell to ~0.0015 on single-game data (better than
any multi-game run, as expected); the novelty guard passed silently on
all five held-out games; and each game's `online/*` scalars contain
only that game — no contamination.

**Housekeeping loss discovered while writing this up:**
`runs/held_out_v1/eval_100000.json` — Run 7's pre-registered 250-episode
sweep artifact — was overwritten (July 10, 21:59) by a 4-episode
plan-mode smoke test from tickets/0011's implementation check (`eval.py`
names its output `eval_<env_steps>.json`, so every sweep against the
same checkpoint collides). Run 7's summary tables survive in its entry
above; the per-episode JSON is gone. The smoke test itself carries a
signal worth keeping: **plan mode scored cd82 zero-shot in 13 steps
under a 15-step cap** (1 of 2 episodes), where frozen greedy scores 0 —
an encouraging pre-signal for Run 9. Rename discipline added to Run 9's
launch notes below.

Decision: tickets/0010 closes validated. Follow-ups, in order: (1)
Run 9 (already pre-registered below) — the planning sweep, whose Q2 is
now genuinely live given cd82's adapted checkpoint and the smoke-test
signal; (2) a test-time intrinsic-annealing ticket with r11l/ft09's
numbers as the evidence; (3) budget scaling only after those two.

---

# Run 9 — tickets/0011 decision-time planning, measurement sweep (July 10, 2026)

No training in this entry — `eval.py` only, per tickets/0011 Step 7 (same
shape as tickets/0007: measurement, not a training run). tickets/0011
implemented policy-guided shooting (sampling-based MPC,
`training/planner.py`): from the actor's current latent state, roll `N`
policy-sampled dreams plus one always-present greedy dream through
`Thumper.dream`, score each with the exact TD(λ) return math
`training/actor_critic.py` trains on (same γ/λ, the tickets/0006 absorbing
extrinsic discount, the target-critic bootstrap), and execute the first
action of the best-scoring rollout — replanning from scratch every step.
Exposed as a third `EvalProtocol` mode, `"plan"`, alongside the existing
`greedy`/`sampled`.

Run 7 (tickets/0009, from-scratch, held-out split) and Run 8 (tickets/0010,
per-game test-time adaptation, assumed passing above) are the two existing
checkpoints/report lineages this ticket was written to sit next to:
tickets/0010 Step 7 pre-registered a null adaptation result as the trigger
for this mechanism, but Run 8's Findings (this entry, assumed-Good) show
adaptation _did_ help on cd82 — so the question this run answers is not
"is planning the fallback" but the ticket's own pre-registered one: does
decision-time search add anything **on top of** either the frozen
zero-shot policy or the adapted one, independent of how Run 8 came out.

```sh
# (1) Zero-shot: greedy vs plan on the from-scratch held-out checkpoint,
# every game (all 25) x 5 episodes, so train-set and held-out both get a
# reading in one sweep -- planner defaults throughout (num_candidates=64,
# horizon=15, intrinsic_scale=0.0, matching TrainerConfig.dream_horizon).
uv run python eval.py --checkpoint runs/held_out_v1/latest.pt \
  --protocol.modes greedy plan

# (2) Does planning stack with test-time training? Per adapted per-game
# checkpoint from Run 8 (cd82/r11l -- the two with a demonstrated scoring
# path -- first; ft09/sk48/wa30 after if time allows), frozen+plan vs
# adapted+plan against that same game only. adapt_report.json's four-cell
# schema is not widened (tickets/0011 Non-goals) -- these are separate
# eval.py invocations against the adapted latest.pt files.
for game in cd82 r11l ft09 sk48 wa30; do
  uv run python eval.py --checkpoint runs/adapt_v1/$game/latest.pt \
    --protocol.modes greedy plan --protocol.games $game \
    --protocol.episodes-per-game 5
done
```

Wall-time note (tickets/0011 Step 7): `plan` costs
~`(num_candidates + 1) x horizon` world-model steps per env step — at
defaults (64 + 1) x 15 = 975x a reactive step's dream cost. Sweep (1) is
25 games x 5 episodes x ≤600 steps x 2 modes, with only the `plan` mode
paying that multiplier (`greedy` is the existing cheap reactive path) —
budget accordingly and consider `--protocol.max-steps` / a smaller
`--protocol.planner.num-candidates` if wall-clock is prohibitive before
cutting `--protocol.games`/`--protocol.episodes-per-game` (losing games
from the sweep loses the per-game comparison this run exists to make).

## What to Look for

Both pre-registered questions from tickets/0011 Step 7, read directly off
`eval.py`'s printed summary table / `eval_report.json` — no TensorBoard
archaeology, per the ticket's acceptance criterion 5.

**Good:**

- **Q1 — plan beats greedy somewhere, and _where_ is the finding.**
  `mean_levels_completed` and/or `win_rate` higher under `plan` than
  `greedy` on at least one game — the strongest version of this result is
  it showing up on the **held-out** games (ft09/sk48/wa30, or cd82/r11l
  read zero-shot from `held_out_v1` rather than the adapted checkpoints),
  since Missing Feature #2 specifically predicted search matters most
  where the reactive policy's habits don't transfer. A gain confined to
  training games (lp85/sp80, where the reward head is best-anchored per
  tickets/0011's own framing) is real but a weaker result — the reward
  head has more real data to rank futures by there regardless of transfer.
- **The floor holds in this sweep too:** spot-checking
  `--protocol.planner.num-candidates 0` against plain `greedy` on a couple
  of games reproduces identical rows (already verified once against this
  same checkpoint at implementation time — see tickets/0011's acceptance
  criterion 2 — but worth reconfirming here since this is the first sweep
  run at production `num_candidates=64`, a different code path's N).
- **Q2 — planning stacks with adaptation.** `adapted+plan` beats both
  `adapted+greedy` (Run 8's cell) and `frozen+plan`, i.e. the two
  mechanisms' gains are at least partially additive rather than one
  subsuming the other — cd82 (Run 8's clearest win) is the best-powered
  test of this.
- `greedy_chosen` rate (log manually if debugging demands it, per the
  ticket) sits meaningfully below 1.0 on any game `plan` improves — the
  scorer is overriding the policy's own action at least sometimes, which
  is the mechanistic signature behind any score gain. A gain with
  `greedy_chosen` ≈ 1.0 would be suspicious (nothing should differ from
  greedy if the greedy rollout is always winning the argmax).

**Bad (the honest expectation, not a failure — tune or investigate before
concluding the mechanism is broken):**

- **`plan` ≈ `greedy` everywhere** — the pre-registered honest-expectation
  outcome per tickets/0011's own text: "with a weak reward head, plan ≈
  greedy (the designed floor)". Genuine gains require the reward/continue
  heads to rank imagined futures better than the policy's own habits, and
  nothing about landing the mechanism changes how well-anchored those
  heads are today (Run 6/7's reward head was validated on real transitions,
  not on the _policy-steered imagined_ ones a wide shooting search
  explores — a different distribution than either training loop has
  stress-tested). If this fires, it is itself the finding: search adds
  nothing until the world model's predictive quality is the binding
  constraint, and the response is improving the world model (more data,
  more capacity, or the ensemble/uncertainty-aware scoring tickets/0011's
  Non-goals deferred), not re-tuning `num_candidates`/`horizon` first.
- `greedy_chosen` near 1.0 with no score gain — confirms the "weak
  reward head, uninformative scorer" reading directly (per tickets/0011
  Step 7): the search is happening but the scorer never disagrees with
  the policy's own greedy choice.
- `greedy_chosen` near 0.0 with no score gain — the opposite and more
  concerning shape: the scorer _is_ overriding the policy but not toward
  anything that helps real play, i.e. the reward/continue heads are
  hallucinating plannable futures the same way Run 5 diagnosed for
  imagination-trained returns (tickets/0006's original failure mode),
  just surfaced at decision time instead of training time. Would motivate
  revisiting whether `score_rollouts`'s absorbing-extrinsic bound is
  enough protection at search time the way it was for training.

**Ugly (stop and investigate — implementation bug, not a modeling
result):**

- The `num_candidates=0` floor check disagrees with plain `greedy` at
  production settings — something about batching N=64 candidate rows
  differs from the N=0 code path in a way tests/test_planner.py's
  small-Thumper check didn't catch at real model scale.
- NaN/inf in any `plan`-mode episode, or a crash inside `score_rollouts`/
  `plan` — the real checkpoint's tensors exercise shapes/ranges the
  shrunken test config doesn't.
- Two runs of the same sweep command are not byte-identical
  `eval_report.json`s (already spot-checked once at implementation time
  for a smaller sweep — this is the production-scale reconfirmation).

Decision this run should produce, pre-committed: if Q1 fires (plan beats
greedy anywhere, ideally held-out), decision-time planning is validated
as a usable eval-time mechanism and the natural next step is feeding it
back during collection (explicitly out of this ticket's scope, its own
future ticket per the Non-goals) or escalating to CEM/MCTS if the
shooting signal is strong enough to amplify. If Bad fires cleanly
(plan ≈ greedy everywhere), the honest reading is that the world model's
predictive quality, not the lack of search, is the binding constraint —
next work goes back to world-model data/capacity rather than planner
tuning.

### Pre-launch amendments (July 11 — after Run 8's Findings landed)

The entry above was written while Run 8 was still assumed-passing;
Run 8's actual Findings sharpen it in three ways, recorded here before
launch so the pre-registration stays honest:

- **Q2 is reframed by Run 8's split result.** cd82's adapted checkpoint
  (greedy 0.80 / sampled 1.00) is the best-powered test of
  "planning stacks with adaptation," as already noted. But r11l's
  adapted checkpoint is now a *different* question and arguably the more
  interesting one: its reward head trained on a real scoring event
  (env step 1,103) while its policy was steered away from scoring by
  the intrinsic-dominated objective (entropy ~0.58, zero eval events).
  If planning's value comes from the reward/continue heads overriding
  the policy's habits, r11l is the cleanest place for `plan` to beat
  both `greedy` *and* the adapted policy's own behavior — plan > greedy
  on adapted-r11l would directly demonstrate search recovering
  competence the reactive policy has but won't express.
- **Pre-signal on file:** tickets/0011's implementation-time smoke test
  (now sitting in `runs/held_out_v1/eval_100000.json`, see below)
  scored cd82 zero-shot in 13 steps under a 15-step cap in plan mode,
  where frozen greedy scores 0.00. Q1 is live, not just hopeful.
- **Artifact discipline — `eval.py` output naming collides.** It writes
  `eval_<env_steps>.json` next to the checkpoint, which is how Run 7's
  250-episode sweep JSON was silently clobbered by the smoke test.
  Before sweep (1), preserve the smoke file; after every `eval.py`
  invocation against the same checkpoint, rename the fresh output
  before the next one:

  ```sh
  mv runs/held_out_v1/eval_100000.json \
     runs/held_out_v1/eval_100000_plan_smoke.json   # before launch
  # after sweep (1):
  mv runs/held_out_v1/eval_100000.json runs/held_out_v1/eval_100000_greedy_plan.json
  # the num-candidates=0 floor spot-check gets its own rename too.
  ```

  The adapted checkpoints write `eval_20000.json` in their own
  `runs/adapt_v1/<game>/` dirs — no existing file, no collision.
- **Launch order: run part (2) first.** The five per-game evals against
  the adapted checkpoints are far cheaper than the 25-game sweep (5
  games × 5 episodes × ≤600 steps × 2 modes, only `plan` paying the
  ~975× dream multiplier) and contain the two most informative
  questions (cd82 stacking, r11l rescue). Sweep (1) follows if part
  (2)'s wall-clock confirms the budget is tolerable; shrink
  `--protocol.planner.num-candidates` before cutting games/episodes,
  per the wall-time note above.

## Findings (July 11, 2026 — part (2) complete; sweep (1) not run, disposition below)

**Verdict: the pre-registered Bad branch fired, and on cd82 it fired
*harder* than Bad — plan is worse than greedy on the one game where the
policy is competent.** Decision-time planning as implemented adds nothing
on top of adaptation at current reward-head quality. The run's real
product is a clean mechanistic account of *why*, and both halves of that
account point at tickets/0012.

Per-game (adapted `runs/adapt_v1/<game>/latest.pt` checkpoints, that
game only, 5 episodes/mode, planner defaults; outputs preserved as
`runs/adapt_v1/<game>/eval_20000_plan_run9.json` per the rename
discipline):

| game | greedy mean | plan mean | shape |
|------|------------|-----------|-------|
| cd82 | **1.00** (5/5, scores at steps 9–11) | **0.40** (2/5, scores at 4 and 103) | plan *hurts* |
| r11l | 0.00 (all 600-cap) | 0.00 (all 600-cap) | rescue refuted |
| ft09 | 0.00 (all exactly 32 steps) | 0.00 (32/33/100/103/104) | habit broken, no reward |
| sk48 | 0.00 (600-cap) | 0.00 (600-cap) | unchanged |
| wa30 | 0.00 (600-cap) | 0.00 (600-cap) | unchanged |

**cd82 — the sharp negative.** Greedy scores within 9–11 steps in every
episode; plan scores in 2 of 5. The three scoreless plan episodes all
end at *exactly* length 100 — together with greedy's ~110-step episodes
(≈100 steps after scoring at ~10) this reads as a ~100-step per-level
timer, i.e. plan wandered away the entire level-1 clock. This is not a
floor violation: the greedy rollout is always candidate 0, but the floor
only guarantees `num_candidates=0` ≡ greedy — at N=64 plan executes the
first action of whichever rollout the TD(λ) scorer ranks best, and once
it deviates it replans from off-greedy states it steered itself into.
So plan < greedy is direct evidence for pre-registered Bad shape 3: the
reward/continue heads rank some imagined non-greedy futures above the
*true* scoring path — hallucinated plannable futures, tickets/0006's
training-time failure mode resurfacing at search time. The flip side is
real too: when the scorer agreed with reality, plan scored at step 4,
faster than greedy ever does — the search can shortcut when the head is
right; it just isn't right reliably. (Minor note: greedy here is 1.00
vs Run 8's adapted-greedy cell of 0.80 — one episode at n=5, and cd82's
env is visibly stochastic across resets (frozen sampled lengths
102/315/600/223 in Run 8), so greedy is not bit-repeatable across
invocations. Noise, not weights drift.)

**r11l — the pre-launch-amendment question ("search recovers suppressed
competence") is refuted at this budget, with a mechanism that matters.**
All 10 episodes, both modes, ride the 600-step cap with zero reward
events. The amendment hoped the reward head — trained on the real
step-1,103 event — could out-vote the collapsed policy. It structurally
cannot in policy-guided shooting: candidates are sampled *from* the
collapsed policy (entropy ~0.58), so the 64 sampled rollouts barely
diverge from the greedy one and the scorer is never shown the scoring
path in the first place. Entropy collapse doesn't just kill reactive
play; it kills the planner's search width. This upgrades tickets/0012
from "fix the actor's objective" to *the enabling condition for
decision-time search on adapted checkpoints* — pre-registered as a
Run 10 secondary question below.

**ft09 — plan demonstrably overrides habits, just not toward reward.**
Greedy repeats the 32-step fast-reset loop in all 5 episodes; plan
breaks out of it in 3 of 5 (episodes of 100–104 steps — the ~100 shape
again), still scoreless. Mechanistically encouraging (the scorer *does*
override the policy, so `greedy_chosen` < 1 in practice), outcome-
neutral. sk48/wa30: nothing in any cell, as in every prior measurement.

**No Ugly trigger fired** — no NaN/inf or crash in any plan episode.
The production-scale `num-candidates 0` floor spot-check was *not* run;
given cd82's plan < greedy it is now the cheapest falsification of the
"scorer misleads" reading (if the floor is broken, that result is a bug,
not a finding) — ~5 minutes on adapted cd82, folded into Run 10
pre-work.

**Sweep (1) not run — deliberate disposition, not an omission.**
Wall-clock from part (2): a 600-cap game costs ~30 min in plan mode
(r11l 06:18→06:47, sk48 06:51→07:23, wa30 07:23→07:56 by file mtimes),
so the 25-game sweep is 10h+, dominated by cap-riders. With Q2 answered
negative, the only still-live sweep question is Q1 zero-shot — and it
*is* still live: the frozen `held_out_v1` policy never went through 20k
steps of intrinsic-dominated adaptation, so its entropy (and therefore
the planner's proposal diversity) is intact, which is exactly the
configuration where the smoke test scored cd82 in 13 steps while frozen
greedy scores 0.00. A scoped 5-held-out-game version is Run 10's
pre-work; the full 25-game sweep only happens if the scoped one shows a
gain worth mapping across training games.

Decision, per the pre-commit: do **not** feed planning back into
collection, and do not tune `num_candidates`/`horizon` — the binding
constraints are reward/continue-head quality on policy-steered imagined
futures (cd82) and proposal diversity under entropy collapse (r11l),
not search breadth. tickets/0011 closes as "mechanism landed, works as
implemented, negative at current model quality"; the planner stays
eval-only. Next: Run 10.

---

# Run 10 — tickets/0012 test-time intrinsic annealing (pre-registered July 11, 2026)

Two parts. Pre-work needs no new code and can run **while tickets/0012
is being implemented**; the main arm launches once 0012 lands and its
tests are green.

## Pre-work (before 0012 lands — `eval.py` only)

Run 9's five outputs were already renamed to
`runs/adapt_v1/<game>/eval_20000_plan_run9.json`, so the collision path
is clear. Rename each fresh output before the next invocation against
the same checkpoint, as below.

```sh
# (a) ~5 min — production-scale floor check, on the game where plan lost
# to greedy. Expect the plan rows to be identical to the greedy rows;
# any divergence is Run 9's Ugly trigger (tickets/0011 bug) and Run 9's
# cd82 conclusion is void until it's resolved.
uv run python eval.py --checkpoint runs/adapt_v1/cd82/latest.pt \
  --protocol.modes greedy plan --protocol.games cd82 \
  --protocol.planner.num-candidates 0
mv runs/adapt_v1/cd82/eval_20000.json \
   runs/adapt_v1/cd82/eval_20000_floor_check.json

# (b) ~2.5h — the scoped remainder of Run 9's sweep (1): Q1 zero-shot on
# the frozen checkpoint, held-out games only. The frozen policy's entropy
# is intact, so policy-guided shooting has real proposal diversity here
# (unlike adapted r11l), and the smoke pre-signal (cd82 plan scoring in
# 13 steps where frozen greedy scores 0.00) is exactly this arm.
uv run python eval.py --checkpoint runs/held_out_v1/latest.pt \
  --protocol.modes greedy plan \
  --protocol.games cd82 r11l ft09 sk48 wa30
mv runs/held_out_v1/eval_100000.json \
   runs/held_out_v1/eval_100000_plan_heldout.json
```

Pre-registered read for (b): **Good** = plan > greedy on any held-out
game (cd82 the live candidate) — tickets/0011's Q1 fires where it was
predicted to matter most, and the 25-game sweep gets scheduled to map
the gain. **Bad** = plan ≈ greedy everywhere — Q1 closes negative at
current reward-head quality, consistent with Run 9's cd82 diagnosis,
and no further planner sweeps run until the world model improves.
Either way tickets/0011's last open question is answered at ~2.5h
instead of 10h+.

## Main arm (after 0012 lands): annealed test-time adaptation

Exactly tickets/0012 Step 6's pre-registered command — one new arm
against Run 8's existing baseline (`runs/adapt_v1`, constant
`intrinsic_scale=1.0`, **do not rerun it**), same checkpoint, seed,
budget, and eval protocol; only the schedule differs:

```sh
uv run python adapt.py \
  --checkpoint runs/held_out_v1/latest.pt \
  --games r11l cd82 ft09 \
  --output-dir runs/adapt_v2_anneal \
  --intrinsic-scale-final 0.0
```

Launch gate: `uv run pytest` green including 0012's four new tests, and
the ticket's acceptance criteria hold (notably: no `--intrinsic-scale-
final` flag ⇒ bit-identical behavior to today). Wall-clock ~3h (Run 8
measured ~57 min/game; three games).

## What to Look for

**Good:**

- **r11l — the headline.** Any adapted cell > 0.00 (baseline: all four
  cells 0.00 despite `env_steps_to_first_score=1103`), and adapted play
  *retaining* sampled-mode scoring events rather than erasing frozen's
  transients (Run 8's steps-359/501 events). Mechanistic TB checks:
  `policy/intrinsic_scale` ramping linearly 1.0 → 0.0 over the 20k
  budget; `policy/entropy` tail meaningfully above Run 8's ~0.58;
  `value_ext_mean` not inflating past Run 8's ~0.76 on the same
  single-event diet.
- **cd82 — do-no-harm control.** Adapted cells at or near Run 8's
  greedy 0.80 / sampled 1.00 (allow one-episode noise — see Run 9's
  1.00-vs-0.80 note; cd82's env is stochastic across resets), and
  `env_steps_to_first_score` not much worse than 6,763. The linear
  schedule still has scale ≈ 0.66 at step 6,763, so early exploration
  is barely touched — a large regression here would be surprising and
  informative.
- **ft09 — degeneracy gone.** Adapted greedy episode lengths no longer
  locked at ~32 (the fast-reset novelty-farming loop). A score is not
  expected; losing the degenerate attractor is the win condition.
- **Run 9's mechanism hook (secondary, pre-registered):** after the arm
  completes, rerun the plan eval on annealed r11l —

  ```sh
  uv run python eval.py --checkpoint runs/adapt_v2_anneal/r11l/latest.pt \
    --protocol.modes greedy plan --protocol.games r11l
  mv runs/adapt_v2_anneal/r11l/eval_20000.json \
     runs/adapt_v2_anneal/r11l/eval_20000_plan.json
  ```

  If annealing preserves entropy, policy-guided shooting regains the
  proposal diversity Run 9 showed it loses under collapse — plan >
  greedy here would recover Run 9's refuted rescue and tie 0011 and
  0012 together mechanistically. Secondary signal only, not the exit
  criterion.

**Bad (informative, not failure):**

- r11l entropy stays healthy but every cell is still 0.00 — intrinsic
  dominance was not the (only) binding constraint; the residual
  suspects are budget (one reward event in 20k steps is a thin diet for
  the extrinsic stream) and plain reward sparsity. The response is
  budget scaling (Run 8's deferred follow-up: 2–3× on r11l), not
  schedule tuning.
- cd82 regresses meaningfully (adapted greedy well below 0.80) — late
  intrinsic signal was still buying useful exploration on cd82 at this
  budget. The documented response is the nonzero-floor follow-up arm
  (`--intrinsic-scale-final 0.1`, per tickets/0012), not abandoning the
  schedule.
- `policy/entropy` pins at its maximum late in adaptation on the
  zero-signal games (tickets/0012's pre-registered secondary risk: with
  both advantage streams near zero, the entropy bonus is unopposed) —
  same 0.1-floor follow-up arm.

**Ugly (stop and investigate — implementation bug):**

- `policy/intrinsic_scale` is not a clean linear 1.0 → 0.0 ramp in env
  steps (schedule wired to the wrong counter, or applied to the critic
  losses — tickets/0012 Non-goals say actor mix only).
- NaN/inf anywhere, or the novelty guard behaving differently from
  Run 8 on identical inputs (the arm changes nothing upstream of it).
- Baseline drift: rerunning any `runs/adapt_v1` eval cell disagrees
  with Run 8's report beyond cd82-style reset noise — would mean 0012's
  changes weren't inert with the flag off, contradicting its acceptance
  criterion 4.

Decision this run should produce, pre-committed: if r11l retains
scoring, tickets/0012 validates and budget scaling on r11l is next. If
r11l stays zero *with* healthy entropy, annealing is kept as a
default-off arm and the next lever is budget, not schedules. A split
(cd82 harmed + r11l helped) routes to the 0.1-floor arm before any
conclusion is drawn.

## Findings

(to be filled after both parts)
