"""Strict checkpoint utilities for bilingual JP-Extra models."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    skip_optimizer: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, float, int]:
    """Load a checkpoint and fail on any model ABI mismatch."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint_dict = torch.load(path, map_location=device)
    try:
        iteration = int(checkpoint_dict["iteration"])
        learning_rate = float(checkpoint_dict["learning_rate"])
        saved_state_dict = checkpoint_dict["model"]
    except KeyError as exc:
        raise KeyError(f"Checkpoint {path} is missing required key: {exc.args[0]}") from exc

    if not isinstance(saved_state_dict, dict):
        raise TypeError(f"Checkpoint {path} key 'model' must be a state_dict")

    logger.info("Loading model checkpoint at iteration %s from %s", iteration, path)

    if not skip_optimizer:
        if optimizer is None:
            raise ValueError("optimizer is required when skip_optimizer=False")
        optimizer_state = checkpoint_dict.get("optimizer")
        if optimizer_state is None:
            raise KeyError(f"Checkpoint {path} is missing required key: optimizer")
        optimizer.load_state_dict(optimizer_state)

    target_model = model.module if hasattr(model, "module") else model  # type: ignore[reportAttributeAccessIssue]
    target_model.load_state_dict(saved_state_dict, strict=True)  # type: ignore[reportAttributeAccessIssue]

    logger.info("Loaded '%s' (iteration %s)", path, iteration)

    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    iteration: int,
    checkpoint_path: str | Path,
) -> None:
    """Save a full model and optimizer checkpoint."""

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target_model = model.module if hasattr(model, "module") else model  # type: ignore[reportAttributeAccessIssue]
    logger.info("Saving model checkpoint at iteration %s to %s", iteration, path)
    torch.save(
        {
            "model": target_model.state_dict(),  # type: ignore[reportAttributeAccessIssue]
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        path,
    )


def get_latest_checkpoint_path(model_dir_path: str | Path, pattern: str = "G_*.pth") -> str:
    """Return the latest checkpoint by numeric step embedded in the file name."""

    path = Path(model_dir_path)
    candidates = sorted(path.glob(pattern), key=lambda p: _checkpoint_step(p))
    if not candidates:
        raise ValueError(f"No checkpoint found in {path} with pattern {pattern}")
    return str(candidates[-1])


def is_resuming(model_dir_path: str | Path) -> bool:
    """Return whether a generator checkpoint is available for resume."""

    return any(Path(model_dir_path).glob("G_*.pth"))


def clean_checkpoints(
    model_dir_path: str | Path,
    n_ckpts_to_keep: int,
    sort_by_time: bool = True,
) -> None:
    """Prune training checkpoints while preserving initial *_0.pth files."""

    if n_ckpts_to_keep <= 0:
        return
    path = Path(model_dir_path)
    for prefix in ("G_", "D_", "DUR_", "WD_"):
        files = [p for p in path.glob(f"{prefix}*.pth") if not p.name.endswith("_0.pth")]
        if sort_by_time:
            files.sort(key=lambda p: p.stat().st_mtime)
        else:
            files.sort(key=_checkpoint_step)
        for checkpoint in files[:-n_ckpts_to_keep]:
            logger.info("Deleting expired checkpoint %s", checkpoint)
            checkpoint.unlink()


def get_steps(model_path: str | Path) -> int:
    """Extract the final numeric step from a checkpoint path."""

    return _checkpoint_step(Path(model_path))


def _checkpoint_step(path: Path) -> int:
    matches = re.findall(r"\d+", path.name)
    if not matches:
        raise ValueError(f"Checkpoint path does not contain a numeric step: {path}")
    return int(matches[-1])
