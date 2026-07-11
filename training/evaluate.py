"""
Evaluation harness :: "can Thumper play this game?" made measurable
(tickets/0007).

Every scalar the online collector logs is a training-time side effect: a
stochastic, exploration-biased policy, round-robin across games, sampled
whenever the rotation happens to land on a game. That answers "is the world
model learning", not "which games can the policy actually play". This module
runs a **fixed, repeatable protocol** instead -- same checkpoint, same game
list, same episode count, same seed, in and sampled -- so runs are
comparable across tickets and training changes.

`evaluate` is the one function all three surfaces (the standalone `eval.py`
CLI, an optional periodic in-training hook, and any future automation) call
into. It shares its acting loop with the online collector via
`training/online_actor.py`'s `OnlineActor`, so an evaluation measures
exactly the agent being trained -- never a second, subtly-diverged copy.
Eval episodes are measurement only: nothing here writes to the replay
buffer or otherwise perturbs training state (see Non-goals in tickets/0007).
"""
import json
import random
from dataclasses import asdict, dataclass, field

import torch

from env.env import Env
from model.actions import ACTION6
from model.thumper import Thumper
from training.online_actor import OnlineActor


@dataclass
class EvalProtocol:
    games: list[str] | None = None
    """None -> every game Env.games() reports (all downloaded games)."""
    episodes_per_game: int = 5
    max_steps: int = 600
    modes: tuple[str, ...] = ("greedy", "sampled")
    seed: int = 0
    carry_macro_context: bool = False
    """Reuse one OnlineActor per (game, mode), carrying the macro-context
    `m` across that game's episodes instead of resetting it every episode
    (tickets/0010 Arm A) -- measures whether the task belief built in
    episode 1 helps episode 2+. Off by default, which preserves 0007/0009
    semantics exactly (a fresh actor per episode)."""


@dataclass
class EvalEpisode:
    game: str
    mode: str
    levels_completed: int
    won: bool
    length: int
    steps_to_first_score: int | None


@dataclass
class GameStats:
    game: str
    mode: str
    mean_levels_completed: float
    max_levels_completed: int
    win_rate: float
    mean_length: float
    mean_steps_to_first_score: float | None
    """None when no episode of this game/mode ever scored."""


@dataclass
class EvalReport:
    episodes: list[EvalEpisode] = field(default_factory=list)

    def per_game(self, mode: str) -> dict[str, GameStats]:
        games = sorted({ep.game for ep in self.episodes if ep.mode == mode})
        stats = {}
        for game in games:
            eps = [ep for ep in self.episodes if ep.mode == mode and ep.game == game]
            scored = [ep.steps_to_first_score for ep in eps if ep.steps_to_first_score is not None]
            stats[game] = GameStats(
                game=game,
                mode=mode,
                mean_levels_completed=sum(ep.levels_completed for ep in eps) / len(eps),
                max_levels_completed=max(ep.levels_completed for ep in eps),
                win_rate=sum(ep.won for ep in eps) / len(eps),
                mean_length=sum(ep.length for ep in eps) / len(eps),
                mean_steps_to_first_score=(sum(scored) / len(scored)) if scored else None,
            )
        return stats

    def games_scored(self, mode: str) -> int:
        return sum(
            1 for stats in self.per_game(mode).values() if stats.mean_steps_to_first_score is not None
        )

    def games_won(self, mode: str) -> int:
        return sum(1 for stats in self.per_game(mode).values() if stats.win_rate > 0.0)

    def to_json(self) -> str:
        return json.dumps({"episodes": [asdict(ep) for ep in self.episodes]}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "EvalReport":
        payload = json.loads(text)
        return cls(episodes=[EvalEpisode(**ep) for ep in payload["episodes"]])

    def summary_table(self) -> str:
        modes = sorted({ep.mode for ep in self.episodes})
        lines = [
            f"{'game':<12} {'mode':<8} {'mean_score':>10} {'max_score':>9} "
            f"{'win_rate':>8} {'mean_len':>8} {'steps_to_1st':>12}"
        ]
        for mode in modes:
            for game, stats in sorted(self.per_game(mode).items()):
                steps_str = (
                    f"{stats.mean_steps_to_first_score:.1f}"
                    if stats.mean_steps_to_first_score is not None
                    else "inf"
                )
                lines.append(
                    f"{stats.game:<12} {stats.mode:<8} {stats.mean_levels_completed:>10.2f} "
                    f"{stats.max_levels_completed:>9d} {stats.win_rate:>8.2f} "
                    f"{stats.mean_length:>8.1f} {steps_str:>12}"
                )
            lines.append(
                f"  -> games_scored={self.games_scored(mode)} games_won={self.games_won(mode)} ({mode})"
            )
        return "\n".join(lines)


def evaluate(thumper: Thumper, env: Env, protocol: EvalProtocol) -> EvalReport:
    """Roll `thumper`'s current policy for `protocol.episodes_per_game`
    episodes on every game in `protocol.games` (or every downloaded game),
    once per mode in `protocol.modes`. Deterministic given (thumper weights,
    protocol): torch/random are reseeded from `protocol.seed` so the sampled
    mode is reproducible run to run.

    Runs the whole sweep under `thumper.eval()` + no_grad, restoring the
    prior training mode afterward. Writes nothing to any replay buffer --
    eval episodes are measurement, not experience (tickets/0007 Non-goals).
    """
    games = protocol.games if protocol.games is not None else env.games()
    device = next(thumper.parameters()).device

    was_training = thumper.training
    thumper.eval()
    report = EvalReport()
    try:
        with torch.no_grad():
            for mode in protocol.modes:
                greedy = mode == "greedy"
                random.seed(protocol.seed)
                torch.manual_seed(protocol.seed)
                for game in games:
                    # One actor per (game, mode): with carry off this is
                    # behaviorally identical to a fresh actor per episode
                    # (begin_episode fully resets state either way); with
                    # carry on, m persists across this game's episodes but
                    # can never leak into another game or mode, since the
                    # actor itself doesn't either (tickets/0010 Design
                    # principle 3 -- structural, not conditional).
                    actor = OnlineActor(thumper, str(device))
                    for _ in range(protocol.episodes_per_game):
                        report.episodes.append(
                            _run_episode(
                                actor, env, game, mode, greedy, protocol.max_steps,
                                carry_macro_context=protocol.carry_macro_context,
                            )
                        )
    finally:
        thumper.train(was_training)
    return report


def _run_episode(
    actor: OnlineActor, env: Env, game: str, mode: str, greedy: bool, max_steps: int,
    carry_macro_context: bool = False,
) -> EvalEpisode:
    result = env.reset(game)
    actor.begin_episode(result.frame, carry_macro_context=carry_macro_context)

    steps_to_first_score: int | None = None
    length = 0
    for step in range(max_steps):
        action_type, coords, _mask = actor.act(result.available_actions, greedy=greedy)
        x, y = (coords if action_type == ACTION6 else (None, None))
        result = env.step(action_type, x=x, y=y)
        actor.observe(action_type, coords, float(result.reward), result.frame)
        length += 1
        if steps_to_first_score is None and result.reward != 0:
            steps_to_first_score = step
        if result.done:
            break

    return EvalEpisode(
        game=game,
        mode=mode,
        levels_completed=result.levels_completed,
        won=result.won,
        length=length,
        steps_to_first_score=steps_to_first_score,
    )
