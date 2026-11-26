"""Strict safetensors loading utility for bilingual JP-Extra models."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def load_safetensors(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, int | None]:
    """Load a safetensors model checkpoint and fail on any ABI mismatch."""

    tensors: dict[str, torch.Tensor] = {}
    iteration: int | None = None
    path = Path(checkpoint_path)
    safe_device = str(device)
    with safe_open(str(path), framework="pt", device=safe_device) as f:  # type: ignore
        for key in f.keys():
            if key == "iteration":
                iteration = f.get_tensor(key).item()
                continue
            tensors[key] = f.get_tensor(key)

    target_model = model.module if hasattr(model, "module") else model  # type: ignore[reportAttributeAccessIssue]
    target_model.load_state_dict(tensors, strict=True)  # type: ignore[reportAttributeAccessIssue]

    if iteration is None:
        logger.info("Loaded '%s'", path)
    else:
        logger.info("Loaded '%s' (iteration %s)", path, iteration)

    return model, iteration


def save_safetensors(
    model: torch.nn.Module,
    iteration: int,
    checkpoint_path: str | Path,
    is_half: bool = False,
) -> None:
    """Save a full model state_dict as safetensors."""

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target_model = model.module if hasattr(model, "module") else model  # type: ignore[reportAttributeAccessIssue]
    state_dict = target_model.state_dict()  # type: ignore[reportAttributeAccessIssue]
    tensors = {key: value.half() for key, value in state_dict.items()} if is_half else dict(state_dict)
    tensors["iteration"] = torch.LongTensor([iteration])
    save_file(tensors, path)
    logger.info("Saved safetensors to %s", path)
