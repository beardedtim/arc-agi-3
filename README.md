# arc-agi-3 Agent

## Goal

This is my entry into the ARC-AGI 3 challenge. It aims to be able to
generalize across all games and be able to beat me at playing them.

## What Challenge?

[ARC-AGI 3](https://arcprize.org/arc-agi/3)

## Thumper?

Like `Theo` but a lot more basic

## Technical Limitations

- Must train on a single GPU with ~16gb VRAM and ~32gb of RAM
  - Only validated on CUDA/NVIDIA GPU

## Idea

> Taken straight from [the docs](https://docs.arcprize.org/):
>
> Traditionally, to measure AI, static benchmarks have been the yardstick. These work well for evaluating LLMs and AI reasoning systems. However, to evaluate
> frontier AI agent systems, we need new tools that measure:
>
> - Exploration
> - Percept → Plan → Action
> - Memory
> - Goal Acquisition
> - Alignment
>
> By building agents that can play ARC-AGI-3, you’re directly contributing to the frontier of AI research.

I think the following is a valid approach to solving this problem:

- Having a `World Model` that can ~accurately predict _future states_ that
  Thumper will be in.
  - We are training the Encoder/RSSM and a debugging Decoder at the same time
    so that we can "see" what Thumper is "imagining"

- Having a `Planner` that can ~accurately _plan goals_ that Thumper will go after
  - This planning will be done inside of the `World Model` where Thumper _plans_
    his next move

- Having a way to _train both parts during execution_. Thumper, as he plays a game for
  the first time, should be able to _update his world model_ if he needs to or upgrade his
  _ability to plan_ while he is playing, not just during `training`
  - This is a future goal. Let's see if we can get a World Model and Planner working.

## Current Progress

See `tickets/` for the design history and `TRAINING_LOG.md` for every run's
pre-registered expectations and findings.

- [x] Training Basic World Model: _**Done, validated**_ (tickets 0001, 0004)
  - Dreamer-style world model (`model/world_model.py`): symbol-embedding conv
    encoder → RSSM → decoder + reward/continue/internal-state heads, trained
    online across all 25 downloaded games at once (`train.py`).
  - Run 3 validated it end-to-end: imagination holds over the full dream
    horizon (after the ticket 0004 burn-in fix — the earlier per-step
    macro-context design collapsed imagination, Run 2), and the reward head
    nails scoring transitions (±1.0→±1.1) despite them being ~0.015% of steps.

- [x] Memory: _**Done**_ (ticket 0002)
  - Hierarchical slow-fast split: the RSSM's GRU is the fast frame-to-frame
    memory; a `TaskEncoder` (`model/task_encoder.py`) folds completed
    transitions into a slow macro-context vector `m` — Thumper's evolving
    belief about the current game's rules — which conditions the RSSM
    prior/posterior and the exploration ensemble.
  - `m` is built over a burn-in prefix and frozen for the loss window /
    dream rollout (ticket 0004), so training and imagination see the same
    kind of context. Known gap: online play accumulates `m` over a whole
    episode, longer than training's burn-in horizon.

- [x] Exploration: _**Done, driving play**_ (ticket 0003)
  - Plan2Explore-style transition ensemble; its disagreement is the intrinsic
    reward stream for an actor-critic trained entirely in imagination
    (`Thumper.dream` + `training/actor_critic.py`). Intrinsic-driven play
    found the first real scoring episodes (Run 5).

- [ ] Planner (actor-critic) actually scoring: _**In Progress**_
    (tickets 0005, 0006 — Run 6 in flight)
  - Ticket 0005 split extrinsic and intrinsic returns into two streams
    (separate critics, separate return normalizers) after Run 3 showed the
    intrinsic stream drowning score by construction; also fixed
    truncation-vs-termination and added reward-stratified replay sampling.
  - Run 5 exposed reward farming: in dreams the policy learned to re-trigger
    the reward head repeatedly for a level completion that pays once. Ticket
    0006 makes extrinsic returns absorbing at predicted scores; Run 6
    (resuming `runs/two_stream_returns`) is validating that the farming
    incentive is gone and scoring recurs in real play.

- [ ] Train during execution: _**Not started**_ — still the future goal.
