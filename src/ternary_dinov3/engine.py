"""Classification training and evaluation loops."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def select_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_total = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch = targets.numel()
            loss_total += float(loss.detach()) * batch
            correct += int((logits.argmax(dim=1) == targets).sum())
            count += batch
    if count == 0:
        raise ValueError("the data loader produced no examples")
    return {"loss": loss_total / count, "accuracy": correct / count}

