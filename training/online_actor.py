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

TaskEncoder folds use `forward_sequence`'s arrival-state convention (the
only convention the TaskEncoder is ever trained under, see
model/world_model.py's burn-in loop): a fold consumes the transition that
*arrived at* the state it's called with -- (deter_t, stoch_t) at observation
t, the action that produced t, and the reward that arrived with t. Online,
that arrival state for transition t->t+1 only exists once the *next* `act()`
call has advanced the RSSM onto t+1, so `observe()` merely stashes the
pending (action, reward) and `act()` performs the fold right after its
`observe_step`, before the policy reads features -- per-step ordering
becomes exactly forward_sequence's: step -> fold -> act (tickets/0008).
"""
from collections import deque

import torch

from model.thumper import Thumper


class OnlineActor:
    """Per-episode latent state + acting loop for one live game.

    One instance drives one episode at a time; call `begin_episode` again to
    reset onto a new one (a fresh instance is just as valid -- state is
    reconstructed from a first frame, not carried across episodes) unless
    `begin_episode(carry_macro_context=True)` is used, in which case the
    macro-context `m` survives the reset while everything else (frame stack,
    RSSM state, pending fold) still resets -- an opt-in eval-only ablation
    (tickets/0010 Arm A) scoped by the caller to one (game, mode) block; a
    fresh instance's first `begin_episode` behaves identically regardless of
    the flag, since there is no prior `m` to carry.
    """

    def __init__(self, thumper: Thumper, device: str):
        self.thumper = thumper
        self.device = device
        self._frame_stack: deque[torch.Tensor] | None = None
        self._deter: torch.Tensor | None = None
        self._stoch: torch.Tensor | None = None
        self._macro_context: torch.Tensor | None = None
        self._prev_action_onehot: torch.Tensor | None = None
        self._pending_action_onehot: torch.Tensor | None = None
        self._pending_reward: torch.Tensor | None = None

    def begin_episode(self, first_frame: torch.Tensor, carry_macro_context: bool = False) -> None:
        """Reset the per-episode latent state from an episode's first frame:
        frame-stack deque (K copies of it), RSSM initial_state, TaskEncoder
        initial_state, and a zeroed previous action -- mirrors
        forward_sequence's is_first convention (no real action produced the
        first observation). The pending TaskEncoder fold is seeded zeroed
        too, so the first `act()` performs the same zero-fold on the first
        observation that training's burn-in performs at an is_first step.

        carry_macro_context: when True and a previous episode already ran
        (self._macro_context is not None), leave the macro-context as-is
        instead of resetting it to task_encoder.initial_state -- the slow
        task belief survives the reset while the frame stack, RSSM state,
        and pending fold (all episode-scoped by definition) still reset
        (tickets/0010 Arm A). A first-ever call behaves identically to
        False, since there is no prior `m` to carry."""
        wm = self.thumper.world_model
        device = self.device
        K = wm.config.frame_stack
        self._frame_stack = deque([first_frame] * K, maxlen=K)
        self._deter, self._stoch = wm.rssm.initial_state(1, device)
        if not (carry_macro_context and self._macro_context is not None):
            self._macro_context = wm.task_encoder.initial_state(1, device)
        self._prev_action_onehot = torch.zeros(1, wm.config.action_dim, device=device)
        self._pending_action_onehot = torch.zeros(1, wm.config.action_dim, device=device)
        self._pending_reward = torch.zeros(1, 1, device=device)

    @torch.no_grad()
    def act(
        self, available_actions: list[int], greedy: bool = False
    ) -> tuple[int, tuple[int, int], torch.Tensor]:
        """One online step of the policy: encode the current frame stack,
        step the RSSM posterior, fold the pending transition into the
        macro-context, and sample (or, if greedy, argmax) an action --
        mirrors forward_sequence's per-step ordering exactly (step -> fold ->
        act, both consuming the same arrival state; tickets/0008). Returns
        (action_type, coords, mask) so the caller can store the mask the
        action was chosen under alongside the resulting step."""
        wm = self.thumper.world_model
        device = self.device
        stack = torch.stack(list(self._frame_stack), dim=0).unsqueeze(0).to(device)
        embed = wm.encode(stack)
        self._deter, self._stoch = wm.rssm.observe_step(
            self._deter, self._stoch, self._prev_action_onehot, embed, self._macro_context
        )
        # Arrival-state convention (forward_sequence's burn-in loop): fold
        # the transition that just arrived at (self._deter, self._stoch)
        # using the action/reward that produced it, before the policy reads
        # features -- not the outgoing action about to be chosen.
        self._macro_context = wm.task_encoder(
            self._macro_context, self._deter, self._stoch, self._pending_action_onehot, self._pending_reward
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
        """Record a completed real transition: advance the frame stack and
        stash (action, reward) as the pending TaskEncoder fold. The fold
        itself doesn't happen here -- the arrival state it needs to consume
        alongside this action/reward (deter_{t+1}, stoch_{t+1}) doesn't exist
        yet; it's produced by the *next* `act()` call's `observe_step`, which
        performs the fold right after (see `act`, tickets/0008)."""
        wm = self.thumper.world_model
        device = self.device
        action_type_t = torch.tensor([action_type], device=device)
        coords_t = torch.tensor([coords], device=device)
        action_onehot = wm.encode_actions(action_type_t, coords_t)
        reward_t = torch.tensor([[reward]], device=device)
        # _prev_action_onehot (RSSM's prev-action input) and
        # _pending_action_onehot (TaskEncoder's next-fold input) hold the
        # same value between calls but serve different consumers under
        # different conventions -- keep them separate named fields so the
        # two stay legible rather than collapsing into one field with two
        # meanings.
        self._prev_action_onehot = action_onehot
        self._pending_action_onehot = action_onehot
        self._pending_reward = reward_t
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
