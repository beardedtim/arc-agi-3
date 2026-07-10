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

- [ ] Training Basic World Model: _**In Progress**_
  - I think it's important to have _some level_ of World Model
    validated and trained before we integrate another piece that
    depends on it, even if they need to be trained together from
    scratch

- [ ] Exploration: _**Scoping**_
  - There needs to be some way for when Thumper is _planning_ that he _wants_
    to explore, knowing that he's going to fail a few times before he's really
    going to be able to have a solid plan.

- [ ] Memory: _**Scoping**_
  - Thumper needs a way to "remember" things. LSTM/GRU/Transformer/idk. This should
    probably be a part of his World Model _and_ his Planning stage.
    - I use my memory to decide what _type of state_ I am in
    - I use my memory to decide how to get to my _future state_
