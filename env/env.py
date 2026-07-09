"""
Env :: thin wrapper over arc_agi.Arcade (offline mode)

Speaks tensors/ints outward (frames as torch.long grids, action types as
model.actions ints, rewards as the levels_completed delta) and
GameAction/data-dicts inward. No training logic here -- that's
training/trainer.py's job.
"""
from dataclasses import dataclass

import torch
from arc_agi import Arcade, OperationMode
from arcengine import GameAction
from arcengine.enums import FrameDataRaw, GameState

from model.actions import ACTION6, NUM_ACTION_TYPES, RESET

# model.actions ints (RESET=0, ACTION1..6) map 1:1 onto arcengine's
# GameAction enum values (via from_id) -- ACTION7 exists in the engine but is
# outside the model's action space (see model/actions.py), so it's simply
# never used.
_ACTION_BY_TYPE = {i: GameAction.from_id(i) for i in range(NUM_ACTION_TYPES)}


@dataclass
class StepResult:
    """One env step/reset, tensors/ints outward."""

    frame: torch.Tensor
    """(H, W) int64 grid of cell symbols 0-15."""
    reward: int
    """levels_completed delta since the previous step (0 most steps)."""
    done: bool
    """True once the episode has finished (win or game over)."""
    won: bool
    """True when the episode finished in GameState.WIN (subset of done)."""
    available_actions: list[int]
    """Action-type ints (model.actions space) legal in this state."""
    levels_completed: int
    """Raw cumulative score, for callers wanting the un-differenced signal."""


class Env:
    """One game at a time; call `reset(game)` to switch games."""

    def __init__(self):
        self.arc = Arcade(operation_mode=OperationMode.OFFLINE)
        self.env = None
        self._game: str | None = None
        self._prev_levels_completed = 0

    def games(self) -> list[str]:
        """Short names (e.g. 'ls20') of every downloaded game, suitable for
        `reset`. Derived from `get_environments()` so new games dropped into
        environment_files/ are picked up with no code change. Sorted so the
        order (which keys the trainer's per-game episode ids) is stable
        across processes, not directory-scan order."""
        return sorted(info.game_id.rsplit("-", 1)[0] for info in self.arc.get_environments())

    def reset(self, game: str) -> StepResult:
        """(Re)start `game` (a short name from `games()`) and return its
        first frame."""
        self.env = self.arc.make(game)
        if self.env is None:
            raise ValueError(f"unknown game {game!r}; available: {self.games()}")
        self._game = game
        self._prev_levels_completed = 0
        frame_data = self.env.reset()
        return self._to_step_result(frame_data)

    def step(self, action_type: int, x: int | None = None, y: int | None = None) -> StepResult:
        """Take one action (model.actions int type; x/y required iff
        action_type == ACTION6, in [0, grid_size))."""
        if self.env is None:
            raise RuntimeError("call reset(game) before step")
        action = _ACTION_BY_TYPE[action_type]
        data = {"x": x, "y": y} if action_type == ACTION6 else None
        frame_data = self.env.step(action, data=data)
        return self._to_step_result(frame_data)

    def _to_step_result(self, frame_data: FrameDataRaw) -> StepResult:
        frame = torch.tensor(frame_data.frame[0], dtype=torch.long)
        reward = frame_data.levels_completed - self._prev_levels_completed
        self._prev_levels_completed = frame_data.levels_completed
        done = frame_data.state not in (GameState.NOT_PLAYED, GameState.NOT_FINISHED)
        # RESET is always legal (it's how you start/restart a game) but the
        # engine's available_actions list only ever names ACTION1-7.
        available_actions = [RESET] + [a for a in frame_data.available_actions if a < NUM_ACTION_TYPES]
        return StepResult(
            frame=frame,
            reward=reward,
            done=done,
            won=frame_data.state == GameState.WIN,
            available_actions=available_actions,
            levels_completed=frame_data.levels_completed,
        )
