"""Pretrained import, resumable checkpoints, and frozen ternary export."""

from __future__ import annotations

import random
from collections.abc import Mapping
import copy
from pathlib import Path
from typing import Any

import torch
from ternarylayers.bit import convert_to_ternary
from torch import nn

TRAINING_FORMAT = "ternary-dinov3-training-v1"
BEST_FORMAT = "ternary-dinov3-best-v1"
TERNARY_FORMAT = "ternary-dinov3-ternary-v1"


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a checkpoint mapping")
    return payload


def _find_state_dict(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    for key in ("state_dict", "model", "teacher", "student"):
        value = payload.get(key)
        if isinstance(value, Mapping) and value and all(
            isinstance(item, torch.Tensor) for item in value.values()
        ):
            return value
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload  # type: ignore[return-value]
    raise ValueError("could not find a tensor state_dict in the checkpoint")


def _strip_prefixes(key: str) -> str:
    prefixes = ("module.", "model.", "teacher.", "student.", "backbone.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def load_pretrained_backbone(
    backbone: nn.Module,
    checkpoint: Path | Mapping[str, Any],
    *,
    strict: bool = True,
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load plain, official-style, or project weights into a backbone."""
    payload = load_checkpoint(checkpoint) if isinstance(checkpoint, Path) else checkpoint
    state = {_strip_prefixes(key): value for key, value in _find_state_dict(payload).items()}
    expected = backbone.state_dict()
    state = {key: value for key, value in state.items() if key in expected}
    return backbone.load_state_dict(state, strict=strict)


@torch.no_grad()
def freeze_ternary_weights(model: nn.Module) -> nn.Module:
    """Clone a QAT model and replace Bit layers with frozen inference layers."""
    if hasattr(model, "clone") and callable(getattr(model, "clone")):
        clone: nn.Module = model.clone()
        clone.load_state_dict(model.state_dict(), strict=True)
    else:
        clone = copy.deepcopy(model)
    clone = convert_to_ternary(clone)
    clone.eval()

    dev = next(model.parameters()).device
    return clone.to(dev)


def save_weights(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float],
    frozen: bool = False,
    class_names: list[str] | None = None,
) -> None:
    output_model = freeze_ternary_weights(model) if frozen else model
    _atomic_save(
        {
            "format": TERNARY_FORMAT if frozen else BEST_FORMAT,
            "epoch": epoch,
            "state_dict": output_model.state_dict(),
            "model_config": model.init_config,
            "metrics": metrics,
            "class_names": class_names or [],
        },
        path,
    )


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_accuracy: float,
    metrics: dict[str, float],
    class_names: list[str],
) -> None:
    payload: dict[str, Any] = {
        "format": TRAINING_FORMAT,
        "epoch": epoch,
        "best_accuracy": best_accuracy,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model.init_config,
        "metrics": metrics,
        "class_names": class_names,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    _atomic_save(payload, path)
