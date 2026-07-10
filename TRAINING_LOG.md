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
   term is fully clamped and inert. This is the benign case — recon kept
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
they gather and sample *identical* data, so samples and scalars are
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
  *equal or lower* than Run 1's (~0.09–0.14 vs ~0.10–0.18); ensemble loss
  noisier but falling. Only `train/grad_norm` differed in shape, running ~2×
  Run 1's throughout (3–5 vs 1.5–3).
- **Imagination collapsed.** At matched grad steps on identical batches,
  Run 1's dreams stay on-game with minor artifacts; this run's dreams
  (9k/11k/14k/17k samples) disintegrate within 2–6 imagined frames, often
  morphing into *other games'* textures.
- **Root-cause A/B (script preserved in tickets/0004):** from the same
  burned-in posterior start state and real action sequence, dreams with the
  burned-in macro-context vs a zeroed one degrade about equally (burned-in
  sometimes worse) — so the regression is not the qualitative check's
  freeze-at-zero query; the prior itself is weaker no matter what context
  it's given.
- **Mechanism:** during training, the prior at step t is conditioned on
  `m_t`, freshly built every step from the *real* transitions up to t−1
  (including detached posterior samples, which carry observation
  information). `m` is therefore a second, teacher-forced memory stream
  that always carries fresh ground-truth trajectory info at training time —
  the prior learns to lean on it (which is *why* `kl_raw` beat Run 1's) —
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
prior/posterior conditioning is now the same *kind* of frozen, real-data
context that `Thumper.dream` and `imagine_with_burn_in` use, so the prior
can no longer lean on a per-step teacher-forced context that goes stale the
instant a dream starts. Same defaults as Runs 1/2 otherwise (100k env
steps, `train_every=2`, `prefill_steps=1_000`, batch 16, seq_len 16, lr
3e-4), fresh output dir, `seed=0` against the deterministic offline env —
so this run gathers and samples data *identical* to Runs 1/2, keeping all
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
replay against Run 2's *unfixed* checkpoint — it does not exercise this run's
code path, only confirms the pre-0004 mechanism), then the full rerun's
scalars/samples against both Run 1 (the bar to meet) and Run 2 (the failure
to beat), per tickets/0004's acceptance criteria 3–4.

**A/B script — Good:**

- The three `context_ab_*.png` rows (real / burned-in-`m` dream / zeroed-`m`
  dream) confirm Run 2's finding again: burned-in and zeroed context degrade
  about equally on Run 2's checkpoint. This isn't a pass/fail gate on its
  own — it's a repeat of the diagnosis, confirming nothing has silently
  changed about *why* Run 2 failed before judging whether the fix (a
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

_Pending — fill in after the run._
