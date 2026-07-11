"""
Test-time adaptation harness (tickets/0010) :: "does giving Thumper a
budget of interactions with a novel game actually help?" made measurable.

ARC-AGI-3's premise is learning efficiency on unseen games, but `eval.py`
alone only ever measures frozen zero-shot transfer (tickets/0009's baseline).
This harness adds the cheapest real adaptation mechanism -- test-time
training -- and reports it against that baseline: for one held-out game,
copy a trained checkpoint's *full* weights (world model + policy + critics)
into a fresh single-game `Trainer` run with a fixed env-step budget, then
`evaluate` before vs. after. Per game, in its own output dir, with its own
fresh replay buffer (Design principle 1: nothing adapted on game A is reused
for game B, so catastrophic forgetting is unobservable here by
construction).

Each game's report contains four eval cells -- frozen vs. adapted weights,
crossed with carried-macro-context off/on (tickets/0010 Arm A, measured as
an ablation on both) -- plus `env_steps_to_first_score`, the closest analog
of the competition's actual metric. Read a game's four cells together, not
any one cell in isolation: the headline comparison is frozen-carry-off vs.
adapted-carry-off (does test-time training beat zero-shot?), and the
carry-off-vs-on columns are the separate Arm A question (does carrying `m`
across episode resets help or hurt, and does that answer differ before vs.
after adaptation?).

    uv run python adapt.py --checkpoint runs/held_out_v1/latest.pt \\
        --games cd82 r11l --output-dir runs/adapt_v1

Refuses to adapt on a game the source checkpoint already trained on
(Design principle 5) -- override with --allow-trained-game for debugging.
Resume is automatic: rerunning against the same output dir picks each
per-game run back up from its own latest.pt/buffer.pt (Trainer's ordinary
resume semantics, tickets/0009's guard still applies unchanged since
train_games=[game] never varies across a rerun).
"""
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from env.env import Env
from model.thumper import Thumper
from training.evaluate import EvalProtocol, EvalReport, evaluate
from training.replay_buffer import ReplayBuffer
from training.trainer import Trainer, TrainerConfig


@dataclass
class Args:
    checkpoint: str
    """Source checkpoint (e.g. runs/held_out_v1/latest.pt) -- must carry a
    trainer_config with train_games so the novelty guard can check it."""
    games: list[str]
    """Games to adapt on, one independent run each."""
    output_dir: str = "runs/adapt"
    """Per-game runs land in <output_dir>/<game>/."""
    budget: int = 20_000
    """Env steps per game, prefill included."""
    prefill_steps: int = 500
    eval_every: int = 5_000
    """In-adaptation eval curve cadence (env steps)."""
    episodes_per_game: int = 5
    """Episodes per eval-protocol call (before/after/curve)."""
    allow_trained_game: bool = False
    """Override the novelty guard (Design principle 5) for debugging."""
    seed: int = 0


def check_novelty(game: str, train_games: list[str], allow_trained_game: bool) -> None:
    """Raise if `game` is not a genuine holdout of the checkpoint that
    produced `train_games` (its saved trainer_config.train_games): an empty
    list means the checkpoint trained on every downloaded game, which also
    fails the guard. `allow_trained_game=True` downgrades to a printed
    warning instead of raising, for debugging only."""
    contaminated = not train_games or game in train_games
    if not contaminated:
        return
    message = (
        f"adapt.py novelty guard: game {game!r} is not a held-out game for this checkpoint "
        f"(checkpoint trainer_config.train_games={train_games!r}; an empty list means it trained "
        f"on every downloaded game). Adapting on a trained game measures continued training, not "
        f"test-time adaptation."
    )
    if allow_trained_game:
        warnings.warn(message)
    else:
        raise ValueError(message)


def env_steps_to_first_score(buffer: ReplayBuffer) -> int | None:
    """Walk the buffer's episodes in insertion order, summing episode
    lengths until the first step with nonzero reward; None if the run never
    scored. Assumes the buffer holds one contiguous adaptation run (no FIFO
    eviction reordering it -- true whenever the run's total steps stay under
    buffer_capacity, as the default 20k-step budget does against the 200k
    default capacity)."""
    steps = 0
    for episode in buffer.episodes:
        for reward in episode.rewards:
            if reward != 0.0:
                return steps
            steps += 1
    return None


def _eval_cell(thumper: Thumper, env: Env, game: str, args: Args, carry_macro_context: bool) -> EvalReport:
    protocol = EvalProtocol(
        games=[game],
        episodes_per_game=args.episodes_per_game,
        seed=args.seed,
        carry_macro_context=carry_macro_context,
    )
    return evaluate(thumper, env, protocol)


def adapt_game(checkpoint: str, game: str, args: Args) -> dict:
    """Run the full tickets/0010 protocol for one game: novelty guard,
    pre-adaptation (frozen) eval x2 carry settings, adaptation via Trainer,
    post-adaptation (adapted) eval x2 carry settings, env_steps_to_first_score,
    and the written adapt_report.json. Returns the report dict."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    train_games = getattr(payload.get("trainer_config"), "train_games", [])
    check_novelty(game, train_games, args.allow_trained_game)

    env = Env()

    print(f"[{game}] pre-adaptation (frozen) eval...")
    frozen_thumper = Thumper.load(checkpoint)
    frozen_carry_off = _eval_cell(frozen_thumper, env, game, args, carry_macro_context=False)
    frozen_carry_on = _eval_cell(frozen_thumper, env, game, args, carry_macro_context=True)

    print(f"[{game}] adapting: {args.budget} env steps (prefill {args.prefill_steps})...")
    game_output_dir = str(Path(args.output_dir) / game)
    trainer_config = TrainerConfig(
        train_games=[game],
        init_from=checkpoint,
        init_from_full=True,
        output_dir=game_output_dir,
        total_env_steps=args.budget,
        prefill_steps=args.prefill_steps,
        eval_every=args.eval_every,
        eval_games=[game],
        eval_episodes_per_game=args.episodes_per_game,
        seed=args.seed,
    )
    trainer = Trainer(trainer_config)
    trainer.train()

    print(f"[{game}] post-adaptation (adapted) eval...")
    adapted_thumper = trainer.thumper
    adapted_carry_off = _eval_cell(adapted_thumper, env, game, args, carry_macro_context=False)
    adapted_carry_on = _eval_cell(adapted_thumper, env, game, args, carry_macro_context=True)

    first_score = env_steps_to_first_score(trainer.buffer)

    report = {
        "checkpoint": checkpoint,
        "checkpoint_env_steps": payload.get("env_steps"),
        "game": game,
        "budget": args.budget,
        "prefill_steps": args.prefill_steps,
        "seed": args.seed,
        "env_steps_to_first_score": first_score,
        "eval": {
            "frozen_carry_off": frozen_carry_off.to_json(),
            "frozen_carry_on": frozen_carry_on.to_json(),
            "adapted_carry_off": adapted_carry_off.to_json(),
            "adapted_carry_on": adapted_carry_on.to_json(),
        },
    }

    out_path = Path(game_output_dir) / "adapt_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[{game}] wrote {out_path}")

    print(f"\n{game} summary (mean_levels_completed / win_rate, greedy & sampled):")
    cells = [
        ("frozen,  carry=off", frozen_carry_off),
        ("frozen,  carry=on ", frozen_carry_on),
        ("adapted, carry=off", adapted_carry_off),
        ("adapted, carry=on ", adapted_carry_on),
    ]
    header = f"  {'cell':<20} {'mode':<8} {'mean_score':>10} {'win_rate':>8}"
    print(header)
    for label, cell_report in cells:
        for mode in ("greedy", "sampled"):
            stats_by_game = cell_report.per_game(mode)
            stats = stats_by_game.get(game)
            if stats is None:
                continue
            print(f"  {label:<20} {mode:<8} {stats.mean_levels_completed:>10.2f} {stats.win_rate:>8.2f}")
    print(f"  env_steps_to_first_score = {first_score}")

    return report


def main() -> None:
    args = tyro.cli(Args)
    for game in args.games:
        adapt_game(args.checkpoint, game, args)


if __name__ == "__main__":
    main()
