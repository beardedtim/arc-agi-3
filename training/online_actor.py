"""
OnlineActor :: the shared real-time acting loop for Thumper.

Both the online collector (`training/trainer.py`) and the evaluation harness
(`training/evaluate.py`, tickets/0007) need to drive Thumper against a live
`Env` one step at a time: maintain the frame-stack deque, step the RSSM
posterior, update the TaskEncoder's macro-context on real transitions, and
mask/sample an action. That bookkeeping is subtle (the is_first convention
that mirrors `WorldModel.forward_sequence`, the rule that the TaskEncoder
*does* step online even though dreaming freezes it -- tickets/0002/0003) and
must exist in exactly one place: a divergence between a collector copy and
an evaluator copy would silently make eval measure a different agent than
the one being trained. `Trainer` and the evaluator are both just callers of
this class.
"""
from collections import deque

import torch

from model.thumper import Thumper


class OnlineActor:
    """Per-episode latent state + acting loop for one live game.

    One instance drives one episode at a time; call `begin_episode` again to
    reset onto a new one (a fresh instance is just as valid -- state is
    reconstructed from a first frame, not carried across episodes).
    """

    def __init__(self, thumper: Thumper, device: str):
        self.thumper = thumper
        self.device = device
        self._frame_stack: deque[torch.Tensor] | None = None
        self._deter: torch.Tensor | None = None
        self._stoch: torch.Tensor | None = None
        self._macro_context: torch.Tensor | None = None
        self._prev_action_onehot: torch.Tensor | None = None

    def begin_episode(self, first_frame: torch.Tensor) -> None:
        """Reset the per-episode latent state from an episode's first frame:
        frame-stack deque (K copies of it), RSSM initial_state, TaskEncoder
        initial_state, and a zeroed previous action -- mirrors
        forward_sequence's is_first convention (no real action produced the
        first observation)."""
        wm = self.thumper.world_model
        device = self.device
        K = wm.config.frame_stack
        self._frame_stack = deque([first_frame] * K, maxlen=K)
        self._deter, self._stoch = wm.rssm.initial_state(1, device)
        self._macro_context = wm.task_encoder.initial_state(1, device)
        self._prev_action_onehot = torch.zeros(1, wm.config.action_dim, device=device)

    @torch.no_grad()
    def act(
        self, available_actions: list[int], greedy: bool = False
    ) -> tuple[int, tuple[int, int], torch.Tensor]:
        """One online step of the policy: encode the current frame stack,
        step the RSSM posterior, and sample (or, if greedy, argmax) an
        action -- mirrors forward_sequence's per-step ordering exactly
        (encode -> observe_step -> act). Returns (action_type, coords, mask)
        so the caller can store the mask the action was chosen under
        alongside the resulting step."""
        wm = self.thumper.world_model
        device = self.device
        stack = torch.stack(list(self._frame_stack), dim=0).unsqueeze(0).to(device)
        embed = wm.encode(stack)
        self._deter, self._stoch = wm.rssm.observe_step(
            self._deter, self._stoch, self._prev_action_onehot, embed, self._macro_context
        )
        mask = self._mask_from_available(available_actions)
        out = self.thumper.act(
            self._deter, self._stoch, self._macro_context, mask.unsqueeze(0).to(device), greedy=greedy
        )
        action_type = int(out["action_type"].item())
        coords = tuple(out["coords"][0].tolist())
        return action_type, coords, mask

    def observe(
        self, action_type: int, coords: tuple[int, int], reward: float, frame: torch.Tensor
    ) -> None:
        """Fold a completed real transition into the online latent state:
        advance the frame stack and update macro_context (the TaskEncoder
        *does* step online, on real transitions -- the imagination freeze
        rule only applies to dreamed ones, see tickets/0003)."""
        wm = self.thumper.world_model
        device = self.device
        action_type_t = torch.tensor([action_type], device=device)
        coords_t = torch.tensor([coords], device=device)
        action_onehot = wm.encode_actions(action_type_t, coords_t)
        reward_t = torch.tensor([[reward]], device=device)
        with torch.no_grad():
            self._macro_context = wm.task_encoder(
                self._macro_context, self._deter, self._stoch, action_onehot, reward_t
            )
        self._prev_action_onehot = action_onehot
        self._frame_stack.append(frame)

    def _mask_from_available(self, available_actions: list[int]) -> torch.Tensor:
        """API's `available_actions` (int list) -> (num_action_types,) bool mask."""
        num_types = self.thumper.world_model.config.num_action_types
        mask = torch.zeros(num_types, dtype=torch.bool)
        for a in available_actions:
            if a < num_types:
                mask[a] = True
        return mask

    @property
    def macro_context_norm(self) -> float:
        return self._macro_context.norm().item()
