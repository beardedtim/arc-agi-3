"""
Entry point: run the per-game evaluation harness (tickets/0007) against a
checkpoint -- "which games can Thumper play?" as a repeatable protocol
instead of training-time anecdote.

    uv run python eval.py --checkpoint runs/world_model/latest.pt
    uv run python eval.py --checkpoint runs/two_stream_returns/latest.pt \
        --protocol.episodes-per-game 3 --protocol.games '["lp85","sp80","cd82"]'

Prints a per-game x per-mode summary table and writes `eval_report.json`
next to the checkpoint (suffixed with the checkpoint's env-step count when
available, so a run directory accumulates its own eval history).
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro

from env.env import Env
from model.thumper import Thumper
from training.evaluate import EvalProtocol, evaluate


@dataclass
class Args:
    checkpoint: str = "runs/world_model/latest.pt"
    protocol: EvalProtocol = field(default_factory=EvalProtocol)
    """Eval protocol fields (games, episodes_per_game, max_steps, modes, seed)."""


def main() -> None:
    args = tyro.cli(Args)

    # Trainer checkpoints carry env_steps alongside the Thumper.load-compatible
    # 'config'/'state_dict' payload; a bare Thumper.save() checkpoint doesn't,
    # so the filename falls back to the un-suffixed default.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    thumper = Thumper.load(args.checkpoint)
    env = Env()

    print(f"Evaluating {args.checkpoint} on {args.protocol.games or env.games()}")
    report = evaluate(thumper, env, args.protocol)
    print(report.summary_table())

    checkpoint_dir = Path(args.checkpoint).parent
    env_steps = payload.get("env_steps")
    filename = f"eval_{env_steps}.json" if env_steps is not None else "eval_report.json"
    out_path = checkpoint_dir / filename
    out_path.write_text(report.to_json())
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
