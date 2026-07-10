"""
Thumper's World Model

Responsible for transforming vision/past states into some latent state
for the rest of Thumper to care about
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.actions import ACTION6, GRID_SIZE, NUM_ACTION_TYPES, NUM_SYMBOLS, PAD_SYMBOL
from model.rssm import RSSM, ImageDecoder, ImageDecoderConfig, RSSMConfig
from model.task_encoder import TaskEncoder, TaskEncoderConfig
from model.vision import Vision, VisionConfig


@dataclass
class EnsembleConfig:
    """Config for the Plan2Explore-style transition-disagreement ensemble:
    K small MLP heads predict the next posterior stoch mean from
    (deter, stoch, action); their variance is an intrinsic "what don't I
    know yet" signal. ARC-AGI-3 gives no instructions and sparse score, so
    disagreement-driven exploration is how the agent discovers each game's
    mechanics in the first place."""
    ensemble_size: int = 5
    """K heads; a knob (paper uses 8), not a decision -- splits paper cost/K."""
    ensemble_hidden_dim: int = 256
    """Hidden width of each head's MLP; matches head_hidden_dim by default."""
    ensemble_loss_scale: float = 1.0
    """Weight on the ensemble's MSE term in the world model's total_loss."""


@dataclass
class WorldModelConfig:
    """
    The configuration for Thumper's World Model, with sane defaults
    """
    num_action_types: int = NUM_ACTION_TYPES
    """Discrete action types: RESET + ACTION1-5 + ACTION6 (see model.actions)"""
    grid_size: int = GRID_SIZE
    """Height/width the ARC-AGI-3 grid is padded to before Vision sees it,
    and the resolution the decoder reconstructs; must be 4 * 2**n (decoder
    stage math). The ARC-AGI-3 API caps grids at 64x64 and ACTION6
    coordinates at [0, 63], so this doubles as the click-coordinate range."""
    num_symbols: int = NUM_SYMBOLS
    """Cell vocabulary: 16 ARC colors + 1 padding symbol"""
    frame_stack: int = 4
    """K most recent frames per observation (oldest first, settled frame
    last). At episode start, replicate the first frame to fill the stack.
    The decoder reconstructs only the stack's *last* frame -- the settled
    state the next action acts on; earlier frames exist to expose dynamics
    to the encoder, not to be reconstructed."""
    internal_state_dim: int = 1
    """Size of the auxiliary internal-state target the internal_state_head
    predicts. For ARC-AGI-3 the natural target is the API's score (0-254),
    normalized by the caller."""
    head_hidden_dim: int = 256
    """Hidden width of the reward/continue/internal_state MLP heads"""
    kl_weight: float = 0.2  # tune against posterior collapse (too low) vs blurry recon (too high)
    kl_balance: float = 0.8  # weight on training the prior toward sg(posterior); 1-this trains posterior toward sg(prior)
    # KL floor in *total* nats across the whole stochastic latent (Dreamer's
    # free bits): stop pushing KL to zero once it's already small. Total, not
    # per-dim, because the KL term is the prior head's only gradient source --
    # a floor above the KL the model actually reaches zeroes that gradient and
    # leaves imagination untrained.
    free_nats: float = 1.0
    # Extra weight on large-magnitude reward targets in the reward head's MSE:
    # per-step weight is 1 + reward_loss_balance * |r|. ARC-AGI-3 reward is
    # almost entirely zero -- the score only ticks on rare events like
    # completing a level -- so under plain mean MSE the head regresses to ~0
    # everywhere and level completions never reach imagination. 0 recovers
    # plain MSE.
    reward_loss_balance: float = 10.0
    vision: VisionConfig = field(default_factory=VisionConfig)
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    task_encoder: TaskEncoderConfig = field(default_factory=TaskEncoderConfig)

    @property
    def action_dim(self) -> int:
        """Width of the encoded action vector fed to the RSSM/ensemble:
        one-hot type ++ one-hot click x ++ one-hot click y (coordinate
        one-hots zeroed unless the type is ACTION6). One-hot coordinates
        rather than 2 normalized scalars so the dynamics model gets a
        spatially crisp signal about *where* a click landed."""
        return self.num_action_types + 2 * self.grid_size

    def __post_init__(self):
        # Keep the pieces wired together instead of trusting three configs to
        # agree by hand: the RSSM must consume the embedding Vision actually
        # produces, share this model's single action_dim, and Vision must be
        # sized for the grid/vocabulary `preprocess` actually feeds it.
        self.rssm.embed_dim = self.vision.out_dim
        self.rssm.action_dim = self.action_dim
        self.rssm.macro_context_dim = self.task_encoder.context_dim
        self.vision.input_size = self.grid_size
        self.vision.num_symbols = self.num_symbols
        self.vision.frame_stack = self.frame_stack


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig | None = None):
        super().__init__()
        self.config = config or WorldModelConfig()
        c = self.config
        self.encoder = Vision(c.vision)
        self.rssm = RSSM(c.rssm)
        self.decoder = ImageDecoder(
            ImageDecoderConfig(
                deter_dim=c.rssm.deter_dim,
                stoch_dim=c.rssm.stoch_dim,
                out_channels=c.num_symbols,
                out_size=c.grid_size,
            )
        )
        # Slow memory: folds completed (deter, stoch, action, reward)
        # transitions into a macro-context belief about the current game's
        # rules -- see model/task_encoder.py.
        self.task_encoder = TaskEncoder(
            c.task_encoder, deter_dim=c.rssm.deter_dim, stoch_dim=c.rssm.stoch_dim, action_dim=c.action_dim
        )
        # Dreamer heads on the RSSM latent (deter ++ stoch ++ macro_context).
        # All three train alongside the reconstruction/KL objective whenever
        # batch targets are passed to compute_losses; imagination-time
        # actor-critic reads the reward and continue heads (continue outputs
        # a *logit*).
        feature_dim = c.rssm.deter_dim + c.rssm.stoch_dim + c.task_encoder.context_dim
        self.reward_head = _mlp_head(feature_dim, c.head_hidden_dim, 1)
        self.continue_head = _mlp_head(feature_dim, c.head_hidden_dim, 1)
        self.internal_state_head = _mlp_head(feature_dim, c.head_hidden_dim, c.internal_state_dim)
        # Disagreement ensemble: rides the same batches, reads
        # the latent without shaping it -- see compute_losses' detach.
        self.ensemble = TransitionEnsemble(
            c.rssm.deter_dim, c.rssm.stoch_dim, c.action_dim, c.task_encoder.context_dim, c.ensemble
        )

    def features(self, deter: Tensor, stoch: Tensor, macro_context: Tensor) -> Tensor:
        return torch.cat([deter, stoch, macro_context], dim=-1)

    def predict_heads(self, features: Tensor) -> dict[str, Tensor]:
        """features: (..., deter_dim + stoch_dim + macro_context_dim).
        Returns reward (...,), continue_logit (...,), internal_state
        (..., internal_state_dim)."""
        return {
            "reward": self.reward_head(features).squeeze(-1),
            "continue_logit": self.continue_head(features).squeeze(-1),
            "internal_state": self.internal_state_head(features),
        }

    def disagreement(self, deter: Tensor, stoch: Tensor, action: Tensor, macro_context: Tensor) -> Tensor:
        """(..., dim) -> (...,) variance across the ensemble's K predicted
        next-stoch means."""
        return self.ensemble.disagreement(deter, stoch, action, macro_context)

    def preprocess(self, grids: Tensor) -> Tensor:
        """(..., H, W) integer cell values -> (..., grid_size, grid_size)
        int64; works on any leading dims (batch, frame stack, or both).

        Grids smaller than 64x64 are padded bottom/right with PAD_SYMBOL --
        a dedicated vocabulary entry, not color 0, so the model can tell
        "outside the playfield" from "black cell". Shared by `encode`
        (Vision's actual input) and `compute_losses` (the reconstruction
        target), so the encoder and decoder always agree on resolution no
        matter what size the ARC-AGI-3 API hands us.
        """
        g = grids.long()
        size = self.config.grid_size
        pad_h, pad_w = size - g.shape[-2], size - g.shape[-1]
        if pad_h or pad_w:
            g = F.pad(g, (0, pad_w, 0, pad_h), value=PAD_SYMBOL)
        return g

    def encode(self, grids: Tensor) -> Tensor:
        """(B, K, H, W) raw ARC-AGI-3 frame stack -> (B, embed_dim) for the RSSM."""
        return self.encoder(self.preprocess(grids))

    def encode_actions(self, action_types: Tensor, coords: Tensor | None = None) -> Tensor:
        """Factored ARC-AGI-3 action -> flat vector for the RSSM/ensemble.

        action_types: (...,) int64 in [0, num_action_types) (see model.actions).
        coords: (..., 2) int64 click (x, y) in [0, grid_size), or None when the
            batch contains no ACTION6. Only consulted where the type is
            ACTION6; other steps get zeroed coordinate one-hots, so stale or
            placeholder coords on non-click steps are harmless.

        Returns (..., action_dim) float: one-hot type ++ one-hot x ++ one-hot y.
        """
        c = self.config
        type_onehot = F.one_hot(action_types, num_classes=c.num_action_types).float()
        if coords is None:
            coord_onehot = type_onehot.new_zeros(*action_types.shape, 2 * c.grid_size)
        else:
            x_onehot = F.one_hot(coords[..., 0], num_classes=c.grid_size).float()
            y_onehot = F.one_hot(coords[..., 1], num_classes=c.grid_size).float()
            is_click = (action_types == ACTION6).unsqueeze(-1).float()
            coord_onehot = torch.cat([x_onehot, y_onehot], dim=-1) * is_click
        return torch.cat([type_onehot, coord_onehot], dim=-1)

    def forward_sequence(
        self,
        observations: Tensor,
        action_types: Tensor,
        coords: Tensor | None = None,
        is_first: Tensor | None = None,
        rewards: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run the RSSM over a batch of sequences with posterior teacher-forcing.

        observations: (B, T, K, H, W) integer ARC-AGI-3 frame stacks at each
            step t: the K most recent frames, oldest first, settled frame
            last. When the API returns several animation frames for one
            action they fill the stack; at episode start, replicate the
            first frame. K must equal config.frame_stack.
        action_types: (B, T) int64, the action type that *produced*
            `observations[t]` (the buffer's prev_action-at-t convention),
            so step t's entry is exactly the prev_action of the RSSM
            transition that lands on step t.
        coords: (B, T, 2) int64 click (x, y), consulted only where the type
            is ACTION6 (see `encode_actions`). None if no clicks in batch.
        is_first: (B, T) bool, True where step t is an episode's first frame.
            Its placeholder action is zeroed rather than consumed -- no real
            action produced an episode's first observation. None (e.g. a
            caller without episode bookkeeping) skips the masking.
        rewards: (B, T) or (B, T, 1) float, the reward that *produced*
            `observations[t]` (same prev-action-at-t convention as
            action_types) -- fed to the TaskEncoder alongside the completed
            transition. None defaults to all zeros. Steps where is_first is
            True are placeholders, zeroed the same way placeholder actions
            are.

        Returns per-timestep (B, T, ...) tensors: prior_mean, prior_std,
        post_mean, post_std, recon (per-cell symbol *logits*,
        (B, T, num_symbols, grid_size, grid_size), decoded from the
        posterior latent), and macro_context (the slow-memory belief used
        to produce each step's outputs, see model/task_encoder.py).
        """
        B, T = observations.shape[:2]
        device = observations.device
        action_onehot = self.encode_actions(action_types, coords)
        if is_first is not None:
            action_onehot = action_onehot * (~is_first).unsqueeze(-1).float()

        if rewards is None:
            rewards = torch.zeros(B, T, 1, dtype=torch.float, device=device)
        elif rewards.ndim == 2:
            rewards = rewards.unsqueeze(-1)
        if is_first is not None:
            rewards = rewards * (~is_first).unsqueeze(-1).float()

        deter, stoch = self.rssm.initial_state(B, device)
        macro_context = self.task_encoder.initial_state(B, device)

        # Encode every frame in one batched call: Vision has no cross-timestep
        # dependency (unlike the RSSM step below), so folding B*T into one
        # forward pass avoids T separate small kernel launches.
        embeds = self.encode(observations.reshape(B * T, *observations.shape[2:])).reshape(B, T, -1)

        prior_means, prior_stds, post_means, post_stds, recons = [], [], [], [], []
        deters, stochs, macro_contexts = [], [], []
        for t in range(T):
            embed = embeds[:, t]
            deter, prior_stats, post_stats, stoch = self.rssm.step(
                deter, stoch, action_onehot[:, t], embed, macro_context
            )

            prior_mean, prior_std = RSSM.dist_params(prior_stats)
            post_mean, post_std = RSSM.dist_params(post_stats)
            recon = self.decoder(deter, stoch)

            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            post_means.append(post_mean)
            post_stds.append(post_std)
            recons.append(recon)
            deters.append(deter)
            stochs.append(stoch)
            macro_contexts.append(macro_context)

            # Fold in the completed transition (t) to build the context
            # step t+1's outputs will see -- after storing m_t above, so
            # rewards[:, t] never leaks into step t's own reward-head target.
            macro_context = self.task_encoder(
                macro_context, deter.detach(), stoch.detach(), action_onehot[:, t], rewards[:, t]
            )

        deter_seq = torch.stack(deters, dim=1)
        stoch_seq = torch.stack(stochs, dim=1)
        macro_context_seq = torch.stack(macro_contexts, dim=1)
        outputs = {
            "prior_mean": torch.stack(prior_means, dim=1),
            "prior_std": torch.stack(prior_stds, dim=1),
            "post_mean": torch.stack(post_means, dim=1),
            "post_std": torch.stack(post_stds, dim=1),
            "recon": torch.stack(recons, dim=1),
            "deter": deter_seq,
            "stoch": stoch_seq,
            "action_onehot": action_onehot,
            "macro_context": macro_context_seq,
        }
        outputs.update(self.predict_heads(self.features(deter_seq, stoch_seq, macro_context_seq)))
        return outputs

    def compute_losses(
        self,
        outputs: dict[str, Tensor],
        target_observations: Tensor,
        batch: dict[str, Tensor] | None = None,
        train_reward_head: bool = True,
        kl_weight: float | None = None,
    ) -> dict[str, Tensor]:
        """target_observations: (B, T, K, H, W) integer frame stacks, same as
        fed to forward_sequence; only each stack's last (settled) frame is
        the reconstruction target.

        kl_weight overrides `self.config.kl_weight` when given (e.g. for KL
        warmup schedules) without mutating the model's config in place.

        When `batch` (a rollout-buffer sequence dict) is given, the
        reward / continue / internal-state heads train at the *arriving*
        state (Dreamer convention): the head at features of step t is
        regressed against batch["rewards"][t]/terminateds[t], the
        reward/terminal status of arriving at observations[t] itself -- this
        is what imagination's `heads["reward"][:, 1:]` already assumes.
          - reward: MSE against batch["rewards"], only when `train_reward_head`
            (reward shares the RSSM trunk with reconstruction, so a sparse/noisy
            reward signal can otherwise disturb the learned dynamics). Each
            step is weighted 1 + reward_loss_balance * |r| so rare terminal
            rewards aren't regressed to the near-zero mean (see config).
            Steps where is_first is True are masked out: their reward/prev_action
            are placeholders, not a real transition into that state.
          - continue: BCE against `not terminated`, masked out at is_first
            steps for the same reason.
          - internal_state: MSE against batch["internal_states"] (for
            ARC-AGI-3, the normalized score).
        Callers without rollout targets omit `batch` and get the pure
        recon+KL objective.
        """
        B, T = target_observations.shape[:2]
        # Reconstruct only the settled frame (stack index -1): earlier frames
        # in the stack are context for the encoder, not decoder targets.
        target = self.preprocess(
            target_observations.reshape(B * T, *target_observations.shape[2:])[:, -1]
        )
        recon = outputs["recon"].reshape(B * T, *outputs["recon"].shape[2:])
        # Cells are categorical symbols, so reconstruction is per-cell
        # 17-way classification (cross-entropy over decoder logits), not MSE.
        recon_loss = F.cross_entropy(recon, target)

        # KL balancing (DreamerV2): the same KL, computed twice with stop-grads
        # on opposite sides. The heavily-weighted term trains the prior to
        # match a frozen posterior (better imagination); the lightly-weighted
        # term regularizes the posterior toward a frozen prior.
        post_mean, post_std = outputs["post_mean"], outputs["post_std"]
        prior_mean, prior_std = outputs["prior_mean"], outputs["prior_std"]
        stoch_dim = post_mean.shape[-1]

        def kl_per_dim_mean(mq: Tensor, sq: Tensor, mp: Tensor, sp: Tensor) -> Tensor:
            return (kl_divergence_diag_gaussian(mq, sq, mp, sp) / stoch_dim).mean()

        kl_prior = kl_per_dim_mean(post_mean.detach(), post_std.detach(), prior_mean, prior_std)
        kl_post = kl_per_dim_mean(post_mean, post_std, prior_mean.detach(), prior_std.detach())
        kl_raw = kl_per_dim_mean(post_mean, post_std, prior_mean, prior_std)

        # free-bits floor is applied per term on the mean, so it still allows
        # individual dims/timesteps to vary; only the aggregate is floored.
        # free_nats is in total nats, the kl_* terms are per-dim means, so the
        # floor is converted to per-dim units here.
        # NOTE: kl_loss is what's optimized, but once a term drops below the
        # floor the clamp makes it a constant (zero gradient) -- always check
        # kl_raw, not kl_loss, to see whether the posterior is actually
        # collapsing (and whether the prior is receiving any gradient).
        if self.config.free_nats > 0:
            floor = self.config.free_nats / stoch_dim
            kl_prior = torch.clamp(kl_prior, min=floor)
            kl_post = torch.clamp(kl_post, min=floor)
        alpha = self.config.kl_balance
        kl_loss = alpha * kl_prior + (1 - alpha) * kl_post

        weight = self.config.kl_weight if kl_weight is None else kl_weight
        total_loss = recon_loss + weight * kl_loss
        losses = {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_raw": kl_raw,
        }

        # Disagreement ensemble: each head at step t predicts the *next*
        # posterior stoch mean, i.e. post_mean[t+1], from the transition
        # (deter[t], stoch[t], action[t+1]) -- action[t+1] is exactly the
        # prev_action the RSSM consumed to land on step t+1. Both the
        # (deter, stoch) input and the post_mean target are detached: the
        # ensemble reads the trunk's latent without shaping it, so this
        # loss is zero-risk to recon/KL. macro_context is taken at t+1 too
        # (built from transitions up to t, the same belief the RSSM prior
        # sees when producing step t+1) -- not t, which would leak nothing
        # extra but misalign with what the prior actually conditioned on.
        if T > 1:
            deter_in = outputs["deter"][:, :-1].detach()
            stoch_in = outputs["stoch"][:, :-1].detach()
            action_in = outputs["action_onehot"][:, 1:]
            macro_context_in = outputs["macro_context"][:, 1:].detach()
            target = outputs["post_mean"][:, 1:].detach()
            preds = self.ensemble(deter_in, stoch_in, action_in, macro_context_in)  # (K, B, T-1, stoch_dim)
            per_step_mse = (preds - target.unsqueeze(0)).pow(2).mean(dim=-1)  # (K, B, T-1)

            if batch is not None:
                # is_first at t+1 means action[t+1] is a zeroed placeholder
                # and post_mean[t+1] isn't a real transition target -- same
                # boundary exclusion as the KL/recon terms rely on upstream.
                mask = (~batch["is_first"][:, 1:]).float()
                ensemble_loss = (per_step_mse * mask.unsqueeze(0)).sum() / (
                    mask.sum().clamp(min=1.0) * self.ensemble.k
                )
            else:
                ensemble_loss = per_step_mse.mean()
            total_loss = total_loss + self.config.ensemble.ensemble_loss_scale * ensemble_loss
            losses["ensemble_loss"] = ensemble_loss

        if batch is not None:
            internal_state_loss = F.mse_loss(outputs["internal_state"], batch["internal_states"])

            terminated = batch["terminateds"].float()
            continue_target = 1.0 - terminated
            # mask is_first steps: their reward/terminated fields are
            # placeholders (no real prev_action produced this observation),
            # not a transition the heads should be trained against.
            mask = (~batch["is_first"]).float()
            continue_bce = F.binary_cross_entropy_with_logits(
                outputs["continue_logit"], continue_target, reduction="none"
            )
            continue_loss = (continue_bce * mask).sum() / mask.sum().clamp(min=1.0)

            total_loss = total_loss + continue_loss + internal_state_loss
            losses.update(continue_loss=continue_loss, internal_state_loss=internal_state_loss)

            if train_reward_head:
                reward_weight = mask * (1.0 + self.config.reward_loss_balance * batch["rewards"].abs())
                reward_loss = (
                    reward_weight * (outputs["reward"] - batch["rewards"]).pow(2)
                ).sum() / mask.sum().clamp(min=1.0)
                total_loss = total_loss + reward_loss
                losses["reward_loss"] = reward_loss

        losses["total_loss"] = total_loss
        return losses

    @torch.no_grad()
    def imagine_from_first_frame(
        self, first_observation: Tensor, action_types: Tensor, coords: Tensor | None = None
    ) -> Tensor:
        """Imagination rollout sanity check.

        first_observation: (B, K, H, W) integer frame stack -- only the very
            first real observation (at episode start, the first frame
            replicated K times).
        action_types: (B, T) int64 -- action sequence to imagine forward with, no
            further real observations are used after t=0. `action_types[t]`
            produces imagined frame t+1, so under the buffer's
            prev_action-at-t convention a caller comparing against real
            frames obs[0..T] must pass the buffer's actions[1..T] slice,
            not actions[0..T-1].
        coords: (B, T, 2) int64 click (x, y) for ACTION6 steps, or None.

        Returns decoded per-cell symbol logits
        (B, T + 1, num_symbols, grid_size, grid_size): the real first frame's
        posterior reconstruction, followed by T imagined frames. Argmax over
        dim 2 recovers integer grids.

        The macro-context belief m is initialized at t=0 and held frozen for
        the whole rollout (the "imagination freeze rule"): game rules don't
        change during a short latent dream, and stepping the TaskEncoder on
        self-predicted (not real) transitions would corrupt task belief.
        """
        B, T = action_types.shape
        device = first_observation.device
        action_onehot = self.encode_actions(action_types, coords)

        deter, stoch = self.rssm.initial_state(B, device)
        macro_context = self.task_encoder.initial_state(B, device)
        prev_action = torch.zeros(B, self.config.action_dim, device=device)
        embed = self.encode(first_observation)
        deter, stoch = self.rssm.observe_step(deter, stoch, prev_action, embed, macro_context)

        frames = [self.decoder(deter, stoch)]
        for t in range(T):
            deter, stoch = self.rssm.imagine_step(deter, stoch, action_onehot[:, t], macro_context)
            frames.append(self.decoder(deter, stoch))

        return torch.stack(frames, dim=1)


def _mlp_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class TransitionEnsemble(nn.Module):
    """Plan2Explore-style disagreement ensemble (Sekar et al. 2020): K
    independently initialized MLP heads, each predicting the next posterior
    stoch *mean* from (deter, stoch, action, macro_context) at the current
    step. Variance across heads at a state-action is the intrinsic "what
    don't I know yet" signal -- high on game mechanics the agent hasn't
    tried yet (an unclicked object, an unexplored room), low on mastered
    ones. That is the exploration currency ARC-AGI-3 demands: no
    instructions, so the agent has to seek out the transitions it can't yet
    predict. Conditioning on macro_context lets disagreement be
    task-relative: what's "known" can differ per belief about which game's
    rules are in play.

    The K heads share one `_mlp_head`-shaped architecture (2 hidden ELU
    layers) but batch their weights into single (K, ...) parameter tensors
    and run via `torch.vmap` rather than a Python loop over K modules, so
    the extra cost stays proportional to one small MLP's FLOPs times K, not
    K separate kernel launches.
    """

    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        action_dim: int,
        macro_context_dim: int,
        config: EnsembleConfig | None = None,
    ):
        super().__init__()
        self.config = config or EnsembleConfig()
        k = self.config.ensemble_size
        in_dim = deter_dim + stoch_dim + action_dim + macro_context_dim
        hidden_dim = self.config.ensemble_hidden_dim
        self.stoch_dim = stoch_dim
        self.k = k

        # K independent copies of a 2-hidden-layer ELU MLP, stacked so vmap
        # can batch them: each parameter tensor gets a leading K dim, built
        # by stacking K freshly (independently) initialized single heads --
        # independent inits are what give the variance signal any meaning
        # (identical copies would agree everywhere, ~0 disagreement).
        heads = [_mlp_head(in_dim, hidden_dim, stoch_dim) for _ in range(k)]
        self.params = nn.ParameterDict()
        self._param_names: list[str] = []
        for name, p in heads[0].named_parameters():
            stacked = torch.stack([dict(h.named_parameters())[name] for h in heads], dim=0)
            safe_name = name.replace(".", "__")
            self.params[safe_name] = nn.Parameter(stacked)
            self._param_names.append(safe_name)
        self._template = heads[0]

    def _functional_forward(self, params: dict, x: Tensor) -> Tensor:
        restored = {name.replace("__", "."): params[name] for name in self._param_names}
        return torch.func.functional_call(self._template, restored, (x,))

    def forward(self, deter: Tensor, stoch: Tensor, action: Tensor, macro_context: Tensor) -> Tensor:
        """deter/stoch/action/macro_context: (..., dim). Returns
        (K, ..., stoch_dim), each head's predicted next posterior stoch mean."""
        x = torch.cat([deter, stoch, action, macro_context], dim=-1)
        flat_x = x.reshape(-1, x.shape[-1])
        params = {name: self.params[name] for name in self._param_names}
        preds = torch.vmap(self._functional_forward, in_dims=(0, None))(params, flat_x)
        return preds.reshape(self.k, *x.shape[:-1], self.stoch_dim)

    def disagreement(self, deter: Tensor, stoch: Tensor, action: Tensor, macro_context: Tensor) -> Tensor:
        """Variance across the K heads' predicted means, averaged over stoch
        dims -- one scalar per state-action. no_grad-safe for imagination
        use."""
        preds = self.forward(deter, stoch, action, macro_context)
        return preds.var(dim=0, unbiased=False).mean(dim=-1)


def kl_divergence_diag_gaussian(mean_q: Tensor, std_q: Tensor, mean_p: Tensor, std_p: Tensor) -> Tensor:
    """KL(q || p) for diagonal Gaussians, summed over the last dim.

    q = posterior (uses real observation), p = prior (predicted from past
    only) -- matches the Dreamer convention of pulling the posterior toward
    the prior's prediction.
    """
    var_q, var_p = std_q.pow(2), std_p.pow(2)
    kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p).pow(2)) / (2 * var_p) - 0.5
    return kl.sum(dim=-1)

