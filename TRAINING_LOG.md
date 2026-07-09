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
  show *many different games*, not one game's look — 25 games round-robin
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
  expected; any game where random play *does* win is worth noting for
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
  a trainer bug. Also watch for the opposite: a game whose episodes *never*
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

_Pending — run not yet launched._
