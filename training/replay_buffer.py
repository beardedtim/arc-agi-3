"""
Episodic replay buffer for Thumper's world-model training.

Stores raw per-step frames (not pre-stacked) and assembles the
`(B, T, K, H, W)` frame-stack windows `WorldModel.forward_sequence` /
`compute_losses` need at sample time -- cheaper than storing every stack.

Step convention (matches `WorldModel.forward_sequence`'s docstring): step t's
`action_type`/`coords` are the action that *produced* `frame[t]` (the
prev_action-at-t convention), so an episode's first step has a placeholder
action and `is_first[0] = True`.
"""
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

from model.actions import NUM_ACTION_TYPES


@dataclass
class Episode:
    game_id: int = -1
    """Stable small-int id of the game this episode was collected from (the
    trainer's enumeration order); -1 when unknown. Keys per-game telemetry
    (e.g. ensemble-disagreement breakdowns), so it only needs to be stable
    within a run lineage."""
    frames: list[torch.Tensor] = field(default_factory=list)
    """(H, W) int64 grids, one per step."""
    action_types: list[int] = field(default_factory=list)
    """Action that produced frames[t]; frames[0]'s entry is a placeholder (RESET)."""
    coords: list[tuple[int, int]] = field(default_factory=list)
    """(x, y) click coords; placeholder (0, 0) on non-ACTION6 steps."""
    rewards: list[float] = field(default_factory=list)
    terminateds: list[bool] = field(default_factory=list)
    internal_states: list[float] = field(default_factory=list)
    """Normalized score (levels_completed) at each step."""
    available_actions: list[torch.Tensor] = field(default_factory=list)
    """(num_action_types,) bool mask per step: the actions legal *at* that
    observed state (i.e. alongside frames[t], same indexing)."""

    def __len__(self) -> int:
        return len(self.frames)

    def append(
        self,
        frame: torch.Tensor,
        action_type: int,
        coords: tuple[int, int],
        reward: float,
        terminated: bool,
        internal_state: float,
        available_actions: torch.Tensor,
    ) -> None:
        self.frames.append(frame)
        self.action_types.append(action_type)
        self.coords.append(coords)
        self.rewards.append(reward)
        self.terminateds.append(terminated)
        self.internal_states.append(internal_state)
        self.available_actions.append(available_actions)


class ReplayBuffer:
    """FIFO-over-whole-episodes buffer capped at `capacity` total steps."""

    def __init__(
        self,
        capacity: int = 200_000,
        frame_stack: int = 4,
        internal_state_dim: int = 1,
        num_action_types: int = NUM_ACTION_TYPES,
    ):
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.internal_state_dim = internal_state_dim
        self.num_action_types = num_action_types
        self.episodes: list[Episode] = []
        self.total_steps = 0

    def start_episode(self, game_id: int = -1) -> Episode:
        episode = Episode(game_id=game_id)
        self.episodes.append(episode)
        self._evict()
        return episode

    def add_step(
        self,
        episode: Episode,
        frame: torch.Tensor,
        action_type: int,
        coords: tuple[int, int],
        reward: float,
        terminated: bool,
        internal_state: float,
        available_actions: torch.Tensor | None = None,
    ) -> None:
        if available_actions is None:
            available_actions = torch.ones(self.num_action_types, dtype=torch.bool)
        episode.append(frame, action_type, coords, reward, terminated, internal_state, available_actions)
        self.total_steps += 1
        self._evict()

    def _evict(self) -> None:
        """FIFO-evict whole episodes (oldest first) while over capacity.
        Never evicts down to nothing (an episode being actively collected
        into must survive), so capacity is a soft cap under one episode's
        length of slop."""
        while self.total_steps > self.capacity and len(self.episodes) > 1:
            dropped = self.episodes.pop(0)
            self.total_steps -= len(dropped)

    def can_sample(self) -> bool:
        return any(len(ep) >= 1 for ep in self.episodes)

    def _stack(self, episode: Episode, t: int) -> torch.Tensor:
        """(K, H, W): frames[t-K+1 .. t], repeating frame 0 to fill the stack
        at episode starts."""
        K = self.frame_stack
        frames = [episode.frames[max(i, 0)] for i in range(t - K + 1, t + 1)]
        return torch.stack(frames, dim=0)

    def sample(self, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
        """Returns tensors shaped for `WorldModel.forward_sequence` /
        `compute_losses`:
          observations: (B, T, K, H, W) int64
          action_types: (B, T) int64
          coords: (B, T, 2) int64
          is_first: (B, T) bool
          rewards: (B, T) float32
          terminateds: (B, T) bool
          internal_states: (B, T, internal_state_dim) float32
          game_ids: (B, T) int64 (constant along T; the episode's game)
          available_actions: (B, T, num_action_types) bool
        Each sequence is a contiguous window from one (randomly chosen,
        length-weighted) episode, clipped to stay in-bounds -- episodes
        shorter than seq_len are padded by repeating their last step
        (is_first correctly marks only the true start, so padded steps are
        just harmless repeats of the terminal transition).
        """
        eligible = [ep for ep in self.episodes if len(ep) >= 1]
        if not eligible:
            raise ValueError("cannot sample from an empty buffer")

        weights = [len(ep) for ep in eligible]
        obs_batch, type_batch, coord_batch = [], [], []
        first_batch, reward_batch, term_batch, state_batch, game_batch = [], [], [], [], []
        avail_batch = []

        for _ in range(batch_size):
            episode = random.choices(eligible, weights=weights, k=1)[0]
            n = len(episode)
            span = min(seq_len, n)
            start = random.randint(0, n - span)
            idxs = list(range(start, start + span))
            if span < seq_len:
                # short episode: pad by repeating the last step (safe --
                # padded steps sit past all real content and are just
                # extra transitions the loss trains on, no leakage).
                idxs += [idxs[-1]] * (seq_len - span)

            obs_batch.append(torch.stack([self._stack(episode, t) for t in idxs], dim=0))
            type_batch.append(torch.tensor([episode.action_types[t] for t in idxs], dtype=torch.long))
            coord_batch.append(torch.tensor([episode.coords[t] for t in idxs], dtype=torch.long))
            first_batch.append(torch.tensor([t == 0 for t in idxs], dtype=torch.bool))
            reward_batch.append(torch.tensor([episode.rewards[t] for t in idxs], dtype=torch.float32))
            term_batch.append(torch.tensor([episode.terminateds[t] for t in idxs], dtype=torch.bool))
            state_batch.append(
                torch.tensor([episode.internal_states[t] for t in idxs], dtype=torch.float32).unsqueeze(-1)
            )
            game_batch.append(torch.full((seq_len,), episode.game_id, dtype=torch.long))
            avail_batch.append(torch.stack([episode.available_actions[t] for t in idxs], dim=0))

        return {
            "observations": torch.stack(obs_batch, dim=0),
            "action_types": torch.stack(type_batch, dim=0),
            "coords": torch.stack(coord_batch, dim=0),
            "is_first": torch.stack(first_batch, dim=0),
            "rewards": torch.stack(reward_batch, dim=0),
            "terminateds": torch.stack(term_batch, dim=0),
            "internal_states": torch.stack(state_batch, dim=0),
            "game_ids": torch.stack(game_batch, dim=0),
            "available_actions": torch.stack(avail_batch, dim=0),
        }

    def save(self, path: str | Path) -> None:
        """Persist every episode so a resumed run keeps its collected
        experience. Frames are packed to one uint8 tensor per episode
        (symbols fit 0-16), ~8x smaller on disk than the in-memory int64."""
        payload = []
        for episode in self.episodes:
            if len(episode) == 0:
                continue
            payload.append(
                {
                    "game_id": episode.game_id,
                    "frames": torch.stack(episode.frames, dim=0).to(torch.uint8),
                    "action_types": torch.tensor(episode.action_types, dtype=torch.int16),
                    "coords": torch.tensor(episode.coords, dtype=torch.int16),
                    "rewards": torch.tensor(episode.rewards, dtype=torch.float32),
                    "terminateds": torch.tensor(episode.terminateds, dtype=torch.bool),
                    "internal_states": torch.tensor(episode.internal_states, dtype=torch.float32),
                    "available_actions": torch.stack(episode.available_actions, dim=0),
                }
            )
        torch.save({"episodes": payload}, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        capacity: int = 200_000,
        frame_stack: int = 4,
        internal_state_dim: int = 1,
        num_action_types: int = NUM_ACTION_TYPES,
    ) -> "ReplayBuffer":
        """Rebuild a buffer written by `save`. Loading into a smaller
        `capacity` FIFO-evicts oldest episodes as usual.

        Backward compat: a pre-0003 `buffer.pt` has no "available_actions"
        key -- fall back to all-True masks (what "no mask" already means to
        `Policy.act`), so `--config.init-from` workflows against older runs
        don't crash."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        buffer = cls(
            capacity=capacity,
            frame_stack=frame_stack,
            internal_state_dim=internal_state_dim,
            num_action_types=num_action_types,
        )
        for ep in payload["episodes"]:
            episode = buffer.start_episode(game_id=int(ep["game_id"]))
            episode.frames = list(ep["frames"].long().unbind(0))
            episode.action_types = [int(a) for a in ep["action_types"]]
            episode.coords = [(int(x), int(y)) for x, y in ep["coords"]]
            episode.rewards = [float(r) for r in ep["rewards"]]
            episode.terminateds = [bool(t) for t in ep["terminateds"]]
            episode.internal_states = [float(s) for s in ep["internal_states"]]
            if "available_actions" in ep:
                episode.available_actions = list(ep["available_actions"].bool().unbind(0))
            else:
                episode.available_actions = [
                    torch.ones(num_action_types, dtype=torch.bool) for _ in range(len(episode.frames))
                ]
            buffer.total_steps += len(episode)
            buffer._evict()
        return buffer
