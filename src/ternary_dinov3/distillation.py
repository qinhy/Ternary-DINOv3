"""Direct output distillation for models with matching tensor outputs."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def run_distillation_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    log_interval: int = 25,
    log_prefix: str = "",
) -> dict[str, float]:
    """Match ``student(images)`` to a frozen teacher using mean squared error.

    Both models must return tensors of exactly the same shape. Dataset labels
    are ignored. The caller places both models on ``device`` and constructs an
    optimizer containing only student parameters. Omitting it evaluates MSE.
    Log the first, last, and every ``log_interval`` batches; zero disables
    progress output. ``log_prefix`` can identify the current epoch or run.
    """
    if log_interval < 0:
        raise ValueError("log_interval must be non-negative")
    teacher.requires_grad_(False)
    teacher.eval()
    student.train(optimizer is not None)
    loss_total = 0.0
    count = 0
    total_batches = len(loader)
    phase = "train" if optimizer is not None else "val"
    label = f"{log_prefix} {phase}".strip()
    started = time.perf_counter()
    if log_interval:
        print(f"{label} starting batches={total_batches} device={device}", flush=True)
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch_index, (images, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                target = teacher(images)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = student(images)
            if not isinstance(output, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise ValueError("student and teacher must return tensors")
            if output.shape != target.shape:
                raise ValueError(
                    f"student output {tuple(output.shape)} and teacher output "
                    f"{tuple(target.shape)} must have the same shape"
                )
            loss = F.mse_loss(output.float(), target.float())
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch = images.shape[0]
            batch_loss = float(loss.detach())
            loss_total += batch_loss * batch
            count += batch
            if log_interval and (
                batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches
            ):
                elapsed = time.perf_counter() - started
                rate = count / max(elapsed, 1e-9)
                remaining = elapsed / batch_index * (total_batches - batch_index)
                lr = f" lr={optimizer.param_groups[0]['lr']:.3e}" if optimizer else ""
                shapes = (
                    f" input={tuple(images.shape)} output={tuple(output.shape)}"
                    if batch_index == 1
                    else ""
                )
                print(
                    f"{label} batch={batch_index}/{total_batches} samples={count} "
                    f"mse={batch_loss:.6f} avg_mse={loss_total / count:.6f}"
                    f"{lr} elapsed={elapsed:.1f}s eta={remaining:.1f}s "
                    f"images/s={rate:.1f}{shapes}",
                    flush=True,
                )
    if count == 0:
        raise ValueError("the data loader produced no examples")
    return {"loss": loss_total / count}
