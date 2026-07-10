"""
Qualitative sample tracking for world-model training.

Renders the symbol grids the model sees and predicts as PNGs so a run can be
eyeballed, not just judged by falling scalars:

  - ``recon_step_*.png``: real settled frames (top row) vs their posterior
    reconstructions (bottom row) -- is the model learning the game's
    appearance at all.
  - ``imagine_step_*.png``: real frames (top) vs an imagination rollout
    (bottom) that burns in on a prefix of real steps (warming the RSSM
    state and macro-context, tickets/0004), then rolls forward from the
    burned-in start state with no further observations -- the actual test
    of learned dynamics rather than single-frame compression.

Files land under ``<output_dir>/samples/`` and are mirrored to TensorBoard
when a writer is given. The decoder is used here for exactly its intended
purpose: validating the world model, never driving behavior.
"""
from pathlib import Path

import numpy as np
import torch
from arc_agi.rendering import COLOR_MAP
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from model.actions import NUM_SYMBOLS, PAD_SYMBOL
from model.world_model import WorldModel

# Cell symbols 0-15 use the official ARC-AGI-3 palette ("#RRGGBBAA" strings);
# PAD_SYMBOL gets a color of its own (dark navy, not in the game palette) so
# "outside the playfield" is visibly distinct from any real cell.
_PAD_RGB = (18, 18, 48)

_UPSCALE = 4
"""Nearest-neighbor upscale factor so 64x64 grids are legible in a viewer."""


def _build_palette() -> np.ndarray:
    palette = np.zeros((NUM_SYMBOLS, 3), dtype=np.uint8)
    for symbol, hex_color in COLOR_MAP.items():
        palette[symbol] = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    palette[PAD_SYMBOL] = _PAD_RGB
    return palette


_PALETTE = _build_palette()


def grid_to_image(grid: torch.Tensor) -> Image.Image:
    """(H, W) integer symbol grid -> upscaled RGB PIL image."""
    arr = _PALETTE[grid.clamp(0, NUM_SYMBOLS - 1).cpu().numpy()]
    img = Image.fromarray(arr)
    return img.resize((img.width * _UPSCALE, img.height * _UPSCALE), Image.NEAREST)


def _row_of_frames(frames: list[Image.Image]) -> Image.Image:
    w, h = frames[0].size
    row = Image.new("RGB", (w * len(frames), h))
    for i, frame in enumerate(frames):
        row.paste(frame, (i * w, 0))
    return row


def _stack_rows(rows: list[Image.Image]) -> Image.Image:
    w, h = rows[0].size
    grid = Image.new("RGB", (w, h * len(rows)))
    for i, row in enumerate(rows):
        grid.paste(row, (0, i * h))
    return grid


def _pil_to_chw_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _save(grid: Image.Image, out_dir: Path, name: str, tag: str, step: int, writer: SummaryWriter | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.save(out_dir / name)
    if writer is not None:
        writer.add_image(tag, _pil_to_chw_tensor(grid), step)


def save_recon_check(
    world_model: WorldModel,
    batch: dict[str, torch.Tensor],
    burn_in: int,
    step: int,
    out_dir: Path,
    writer: SummaryWriter | None = None,
    num_frames: int = 8,
) -> None:
    """Decode posterior reconstructions for one sequence and save real vs.
    reconstructed settled frames side by side.

    batch is the full burn_in-prefixed sample (tickets/0004): burn_in real
    steps warm the RSSM/macro-context, then up to num_frames loss-window
    steps are reconstructed and shown."""
    num_frames = min(num_frames, batch["observations"].shape[1] - burn_in)
    with torch.no_grad():
        end = burn_in + num_frames
        outputs = world_model.forward_sequence(
            batch["observations"][:1, :end],
            batch["action_types"][:1, :end],
            batch["coords"][:1, :end],
            batch["is_first"][:1, :end],
            rewards=batch["rewards"][:1, :end],
            burn_in=burn_in,
        )
        # Only each stack's last (settled) frame is reconstructed -- compare
        # against exactly that.
        target = world_model.preprocess(batch["observations"][0, burn_in:end])[:, -1]
        recon = outputs["recon"][0].argmax(dim=1)

    real_row = _row_of_frames([grid_to_image(f) for f in target])
    recon_row = _row_of_frames([grid_to_image(f) for f in recon])
    grid = _stack_rows([real_row, recon_row])
    _save(grid, out_dir, f"recon_step_{step:07d}.png", "qualitative/recon_vs_real", step, writer)


def save_imagination_check(
    world_model: WorldModel,
    batch: dict[str, torch.Tensor],
    burn_in: int,
    step: int,
    out_dir: Path,
    horizon: int,
    writer: SummaryWriter | None = None,
) -> None:
    """Burn in on the batch's leading burn_in real steps, then roll the RSSM
    forward with no further observations, and compare imagined frames to the
    real ones (tickets/0004): this answers "given a warmed-up belief, does
    the prior hold?" -- the question the actor-critic's dreams actually
    pose, rather than the pre-0004 cold m=0 start."""
    total = batch["observations"].shape[1]
    horizon = min(horizon, total - burn_in - 1)
    with torch.no_grad():
        end = burn_in + 1 + horizon
        imagined = world_model.imagine_with_burn_in(
            batch["observations"][:1, : burn_in + 1],
            batch["action_types"][:1, :end],
            batch["coords"][:1, :end],
            batch["is_first"][:1, : burn_in + 1],
            batch["rewards"][:1, : burn_in + 1],
            burn_in=burn_in,
            horizon=horizon,
        )[0].argmax(dim=1)
        target = world_model.preprocess(batch["observations"][0, burn_in:end])[:, -1]

    real_row = _row_of_frames([grid_to_image(f) for f in target])
    imagined_row = _row_of_frames([grid_to_image(f) for f in imagined])
    grid = _stack_rows([real_row, imagined_row])
    _save(grid, out_dir, f"imagine_step_{step:07d}.png", "qualitative/imagine_vs_real", step, writer)
