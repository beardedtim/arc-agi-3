"""
Entry point: train Thumper's world model against every downloaded
ARC-AGI-3 game.

    uv run python train.py
    uv run python train.py --config.lr 1e-4 --config.batch-size 32
    uv run python train.py --config.output-dir runs/world_model_v2

Resume is automatic: rerunning against the same --config.output-dir picks
up from its latest.pt/buffer.pt (disable with --config.no-resume). Use
--config.init-from <checkpoint> to seed a *new* run's world-model weights
from prior work instead. TensorBoard logs live under <output_dir>/tb,
qualitative samples under <output_dir>/samples.
"""
from dataclasses import dataclass, field

import tyro

from training.trainer import Trainer, TrainerConfig


@dataclass
class Args:
    config: TrainerConfig = field(default_factory=TrainerConfig)
    """Training hyperparameters -- every TrainerConfig field is exposed as a flag."""


def main() -> None:
    args = tyro.cli(Args)
    trainer = Trainer(args.config)

    print(f"device={trainer.config.device} games={trainer.env.games()}")
    print(trainer.thumper.summary())

    trainer.train()


if __name__ == "__main__":
    main()
