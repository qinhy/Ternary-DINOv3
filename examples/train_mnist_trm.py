"""Train a ternary DINOv3 ViT-Femto TRM classifier on MNIST (v2).

This script keeps the dataset, checkpoint, optimizer, and frozen-ternary
comparison flow from ``train_mnist.py`` while adding the TRM training behavior

* recurrent ``solution`` and ``latent`` states;
* randomized latent/refinement depth;
* truncated deep supervision with intermediate optimizer updates;
* a confidence/correctness head for adaptive early-exit inference.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from ternary_dinov3.checkpoints import (
    freeze_ternary_weights,
    load_checkpoint,
    save_training_checkpoint,
    save_weights,
)
from ternary_dinov3.engine import select_device
from ternarylayers.dinov3.layers.bitlayers import Linear as BitLinear
from ternarylayers.dinov3.models.vision_transformer import (
    DinoVisionTransformerTRM,
    vit_femto,
)


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    confidence_accuracy: float = 0.0
    average_supervision_steps: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument(
        "--save", type=Path, default=Path("checkpoints/mnist-femto-trm-v2.pt")
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--scale-op", choices=("mean", "median"), default="median")
    parser.add_argument("--seed", type=int, default=7)

    # TRM controls. The uploaded example uses supervision=16, n in [0, 4],
    # and T in [1, 8]. The defaults below are lighter for an initial MNIST run.
    parser.add_argument("--supervision-steps", type=int, default=4)
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=2)
    parser.add_argument("--T-min", dest="t_min", type=int, default=1)
    parser.add_argument("--T-max", dest="t_max", type=int, default=3)
    parser.add_argument("--eval-n", type=int, default=1)
    parser.add_argument("--eval-T", dest="eval_t", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--confidence-loss-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Print training progress every N batches; use 0 to disable batch logs.",
    )
    parser.add_argument(
        "--verbose-supervision",
        action="store_true",
        help="Include per-supervision active/exit counts in batch progress logs.",
    )
    parser.add_argument(
        "--track-latent-grads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep gradients through internal TRM latent refinement steps.",
    )

    parser.add_argument(
        "--limit-train", type=int, help="Use only N training examples for a quick trial"
    )
    parser.add_argument(
        "--limit-test", type=int, help="Use only N test examples for a quick trial"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.supervision_steps <= 0:
        raise ValueError("--supervision-steps must be positive")
    if args.n_min < 0 or args.n_max < args.n_min:
        raise ValueError("require 0 <= --n-min <= --n-max")
    if args.t_min <= 0 or args.t_max < args.t_min:
        raise ValueError("require 0 < --T-min <= --T-max")
    if args.eval_n < 0 or args.eval_t <= 0:
        raise ValueError("require --eval-n >= 0 and --eval-T > 0")
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be in (0, 1]")
    if args.confidence_loss_weight < 0.0:
        raise ValueError("--confidence-loss-weight must be non-negative")
    if args.log_interval < 0:
        raise ValueError("--log-interval must be non-negative")


def mnist_dataset(root: Path, *, train: bool) -> Dataset:
    # Preserve the preprocessing used by the original train_mnist.py so the
    # TRM model sees 32x32 RGB-shaped tensors and can use patch_size=8.
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,) * 3, (0.3081,) * 3),
        ]
    )
    return datasets.MNIST(root=root, train=train, transform=transform, download=True)


def maybe_limit(dataset: Dataset, count: int | None) -> Dataset:
    if count is None:
        return dataset
    if count <= 0:
        raise ValueError("dataset limits must be positive")
    return Subset(dataset, range(min(count, len(dataset))))


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


class TernaryDINOv3TRMClassifier(nn.Module):
    """ViT-Femto TRM backbone with classification and confidence heads."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 8,
        in_chans: int = 3,
        dropout: float = 0.1,
        scale_op: str = "median",
        track_latent_grads: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.dropout = dropout
        self.scale_op = scale_op
        self.track_latent_grads = track_latent_grads

        # Checkpoint helpers serialize ``model.init_config`` alongside the
        # state dict. Keep this dictionary limited to this class's constructor
        # arguments so the TRM classifier can be recreated exactly.
        self.init_config = {
            "num_classes": num_classes,
            "image_size": image_size,
            "patch_size": patch_size,
            "in_chans": in_chans,
            "dropout": dropout,
            "scale_op": scale_op,
            "track_latent_grads": track_latent_grads,
        }

        # The factory's ``cls`` parameter is the intended hook for replacing
        # ordinary blocks with DinoVisionTransformerTRM.
        self.backbone = vit_femto(
            patch_size=patch_size,
            cls=DinoVisionTransformerTRM,
            img_size=image_size,
            in_chans=in_chans,
            no_conv=True,
            drop_path_rate=0.0,
        )
        self.backbone.init_weights()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            BitLinear(self.backbone.embed_dim,
                    num_classes,
                    bias=True,
                    scale_op=scale_op,)
        )
        self.q_head = nn.Sequential(
            nn.Dropout(dropout),
            BitLinear(self.backbone.embed_dim,
                1,
                bias=True,
                scale_op=scale_op,)
        )
        self._init_head(self.head[1])
        self._init_head(self.q_head[1])

    @staticmethod
    def _init_head(layer: nn.Module) -> None:
        weight = getattr(layer, "weight", None)
        bias = getattr(layer, "bias", None)
        if weight is not None:
            nn.init.normal_(weight, std=0.02)
        if bias is not None:
            nn.init.zeros_(bias)

    def clone(self) -> "TernaryDINOv3TRMClassifier":
        """Support checkpoint utilities that freeze a cloned ternary model."""
        return self.__class__(
            num_classes=self.num_classes,
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            dropout=self.dropout,
            scale_op=self.scale_op,
            track_latent_grads=self.track_latent_grads,
        )

    def forward(
        self,
        x: Tensor,
        *,
        solution: Optional[Tensor] = None,
        latent: Optional[Tensor] = None,
        n: int = 1,
        T: int = 1,
        full_output: bool = False,
    ):
        out = self.backbone.forward_features(
            x,
            solution=solution,
            latent=latent,
            n=n,
            T=T,
            track_latent_grads=self.track_latent_grads,
        )
        cls = out["x_norm_clstoken"]
        logits = self.head(cls)
        q_logits = self.q_head(cls)

        if full_output:
            return out["x_prenorm"], out["z_latent"], logits, q_logits
        return logits

    @torch.no_grad()
    def thinking(
        self,
        x: Tensor,
        *,
        n: int,
        T: int,
        max_supervision: int,
        confidence: float,
    ) -> tuple[Tensor, Tensor]:
        """Adaptive inference, returning logits and steps used per sample."""
        if max_supervision <= 0:
            raise ValueError("max_supervision must be positive")

        logits_parts: list[Tensor] = []
        index_parts: list[Tensor] = []
        step_parts: list[Tensor] = []

        indices = torch.arange(x.size(0), device=x.device)
        solution: Optional[Tensor] = None
        latent: Optional[Tensor] = None

        for step in range(1, max_supervision + 1):
            solution, latent, logits, q_logits = self(
                x,
                solution=solution,
                latent=latent,
                n=n,
                T=T,
                full_output=True,
            )

            q_probability = torch.sigmoid(q_logits).flatten()
            done = q_probability >= confidence
            if step == max_supervision:
                done = torch.ones_like(done, dtype=torch.bool)
            keep = ~done

            if done.any():
                logits_parts.append(logits[done])
                index_parts.append(indices[done])
                step_parts.append(
                    torch.full(
                        (int(done.sum().item()),),
                        step,
                        dtype=torch.long,
                        device=x.device,
                    )
                )

            if not keep.any():
                break

            x = x[keep]
            indices = indices[keep]
            solution = solution[keep]
            latent = latent[keep]

        all_logits = torch.cat(logits_parts, dim=0)
        all_indices = torch.cat(index_parts, dim=0)
        all_steps = torch.cat(step_parts, dim=0)
        order = all_indices.argsort()
        return all_logits[order], all_steps[order]


def optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: Tensor,
    grad_clip: float,
) -> None:
    loss.backward()
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _format_duration(seconds: float) -> str:
    """Format elapsed wall-clock time compactly for training logs."""
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def train_epoch(
    model: TernaryDINOv3TRMClassifier,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    total_epochs: int,
    supervision_steps: int,
    n_min: int,
    n_max: int,
    t_min: int,
    t_max: int,
    confidence: float,
    confidence_loss_weight: float,
    grad_clip: float,
    log_interval: int,
    verbose_supervision: bool,
) -> EpochMetrics:
    """Train one epoch and emit periodic, dependency-free progress logs.

    Running accuracy is measured when a sample exits the recurrent supervision
    loop. Loss and confidence accuracy include every active sample at every
    supervision step, matching the optimization workload.
    """
    model.train()

    loss_sum = 0.0
    classification_loss_sum = 0.0
    confidence_loss_sum = 0.0
    loss_weight = 0
    correct_sum = 0
    sample_sum = 0
    q_correct_sum = 0
    q_sample_sum = 0
    supervision_sum = 0
    optimizer_updates = 0
    original_sample_sum = 0

    epoch_start = time.perf_counter()
    total_batches = len(loader)
    total_dataset_samples = len(loader.dataset)
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"[train] epoch {epoch:02d}/{total_epochs:02d} start | "
        f"batches={total_batches} samples={total_dataset_samples} "
        f"lr={current_lr:.3e} supervision={supervision_steps} "
        f"n=[{n_min},{n_max}] T=[{t_min},{t_max}] conf={confidence:.3f} "
        f"q_weight={confidence_loss_weight:g}",
        flush=True,
    )

    for batch_index, (images, targets) in enumerate(loader, start=1):
        batch_start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        original_batch_size = targets.numel()
        original_sample_sum += original_batch_size

        n = random.randint(n_min, n_max)
        T = random.randint(t_min, t_max)

        solution: Optional[Tensor] = None
        latent: Optional[Tensor] = None
        images_cur = images
        targets_cur = targets

        batch_loss_sum = 0.0
        batch_classification_loss_sum = 0.0
        batch_confidence_loss_sum = 0.0
        batch_loss_weight = 0
        batch_correct_sum = 0
        batch_finished_sum = 0
        batch_q_correct_sum = 0
        batch_q_sample_sum = 0
        batch_supervision_sum = 0
        active_counts: list[int] = []
        exit_counts: list[int] = []

        optimizer.zero_grad(set_to_none=True)

        for step in range(1, supervision_steps + 1):
            active_count = targets_cur.numel()
            active_counts.append(active_count)

            solution, latent, logits, q_logits = model(
                images_cur,
                solution=solution,
                latent=latent,
                n=n,
                T=T,
                full_output=True,
            )

            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(targets_cur)
            q_target = correct.to(dtype=q_logits.dtype).unsqueeze(-1)

            classification_loss = F.cross_entropy(logits, targets_cur)
            confidence_loss = F.binary_cross_entropy_with_logits(q_logits, q_target)
            loss = classification_loss + confidence_loss_weight * confidence_loss

            detached_loss = float(loss.detach())
            detached_classification_loss = float(classification_loss.detach())
            detached_confidence_loss = float(confidence_loss.detach())

            loss_sum += detached_loss * active_count
            classification_loss_sum += detached_classification_loss * active_count
            confidence_loss_sum += detached_confidence_loss * active_count
            loss_weight += active_count

            batch_loss_sum += detached_loss * active_count
            batch_classification_loss_sum += detached_classification_loss * active_count
            batch_confidence_loss_sum += detached_confidence_loss * active_count
            batch_loss_weight += active_count

            q_predictions = torch.sigmoid(q_logits).ge(0.5)
            q_correct = int(q_predictions.eq(q_target.bool()).sum().item())
            q_correct_sum += q_correct
            q_sample_sum += active_count
            batch_q_correct_sum += q_correct
            batch_q_sample_sum += active_count

            done = torch.sigmoid(q_logits).flatten().ge(confidence)
            if step == supervision_steps:
                done = torch.ones_like(done, dtype=torch.bool)
            keep = ~done

            finished = int(done.sum().item())
            exit_counts.append(finished)
            if finished:
                finished_correct = int(correct[done].sum().item())
                correct_sum += finished_correct
                sample_sum += finished
                supervision_sum += step * finished
                batch_correct_sum += finished_correct
                batch_finished_sum += finished
                batch_supervision_sum += step * finished

            # This standalone trainer owns every optimizer update, including the
            # final supervision step. Recurrent states are detached before a
            # subsequent step to implement truncated deep supervision.
            optimizer_step(model, optimizer, loss, grad_clip)
            optimizer_updates += 1

            if not keep.any():
                break

            images_cur = images_cur[keep]
            targets_cur = targets_cur[keep]
            solution = solution.detach()[keep]
            latent = latent.detach()[keep]

        should_log = (
            log_interval > 0
            and (
                batch_index == 1
                or batch_index % log_interval == 0
                or batch_index == total_batches
            )
        )
        if should_log:
            elapsed = time.perf_counter() - epoch_start
            batch_elapsed = time.perf_counter() - batch_start
            progress = 100.0 * batch_index / max(total_batches, 1)
            running_loss = loss_sum / max(loss_weight, 1)
            running_classification_loss = classification_loss_sum / max(loss_weight, 1)
            running_confidence_loss = confidence_loss_sum / max(loss_weight, 1)
            running_accuracy = correct_sum / max(sample_sum, 1)
            running_q_accuracy = q_correct_sum / max(q_sample_sum, 1)
            running_steps = supervision_sum / max(sample_sum, 1)
            samples_per_second = original_sample_sum / max(elapsed, 1e-9)

            batch_loss = batch_loss_sum / max(batch_loss_weight, 1)
            batch_classification_loss = (
                batch_classification_loss_sum / max(batch_loss_weight, 1)
            )
            batch_confidence_loss = (
                batch_confidence_loss_sum / max(batch_loss_weight, 1)
            )
            batch_accuracy = batch_correct_sum / max(batch_finished_sum, 1)
            batch_q_accuracy = batch_q_correct_sum / max(batch_q_sample_sum, 1)
            batch_steps = batch_supervision_sum / max(batch_finished_sum, 1)
            current_lr = optimizer.param_groups[0]["lr"]

            detail = ""
            if verbose_supervision:
                active_text = "->".join(str(value) for value in active_counts)
                exit_text = ",".join(str(value) for value in exit_counts)
                detail = f" active={active_text} exits=[{exit_text}]"

            print(
                f"[train] epoch {epoch:02d}/{total_epochs:02d} "
                f"batch {batch_index:04d}/{total_batches:04d} ({progress:5.1f}%) | "
                f"batch loss={batch_loss:.4f} "
                f"(ce={batch_classification_loss:.4f}, q={batch_confidence_loss:.4f}) "
                f"acc={batch_accuracy:.4f} q_acc={batch_q_accuracy:.4f} "
                f"steps={batch_steps:.2f} n={n} T={T} | "
                f"avg loss={running_loss:.4f} "
                f"(ce={running_classification_loss:.4f}, q={running_confidence_loss:.4f}) "
                f"acc={running_accuracy:.4f} q_acc={running_q_accuracy:.4f} "
                f"steps={running_steps:.2f} | lr={current_lr:.3e} "
                f"speed={samples_per_second:.1f} img/s "
                f"batch_time={batch_elapsed:.2f}s elapsed={_format_duration(elapsed)}"
                f"{detail}",
                flush=True,
            )

    elapsed = time.perf_counter() - epoch_start
    metrics = EpochMetrics(
        loss=loss_sum / max(loss_weight, 1),
        accuracy=correct_sum / max(sample_sum, 1),
        confidence_accuracy=q_correct_sum / max(q_sample_sum, 1),
        average_supervision_steps=supervision_sum / max(sample_sum, 1),
    )
    print(
        f"[train] epoch {epoch:02d}/{total_epochs:02d} done | "
        f"loss={metrics.loss:.4f} "
        f"ce={classification_loss_sum / max(loss_weight, 1):.4f} "
        f"q_loss={confidence_loss_sum / max(loss_weight, 1):.4f} "
        f"acc={metrics.accuracy:.4f} q_acc={metrics.confidence_accuracy:.4f} "
        f"steps={metrics.average_supervision_steps:.2f} "
        f"updates={optimizer_updates} speed={original_sample_sum / max(elapsed, 1e-9):.1f} img/s "
        f"elapsed={_format_duration(elapsed)}",
        flush=True,
    )
    return metrics


@torch.no_grad()
def evaluate(
    model: TernaryDINOv3TRMClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    n: int,
    T: int,
    max_supervision: int,
    confidence: float,
) -> EpochMetrics:
    model.eval()

    loss_sum = 0.0
    correct_sum = 0
    sample_sum = 0
    supervision_sum = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, steps = model.thinking(
            images,
            n=n,
            T=T,
            max_supervision=max_supervision,
            confidence=confidence,
        )
        batch_size = targets.numel()
        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
        correct_sum += int(logits.argmax(dim=-1).eq(targets).sum().item())
        sample_sum += batch_size
        supervision_sum += int(steps.sum().item())

    return EpochMetrics(
        loss=loss_sum / max(sample_sum, 1),
        accuracy=correct_sum / max(sample_sum, 1),
        average_supervision_steps=supervision_sum / max(sample_sum, 1),
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)

    train_data = maybe_limit(mnist_dataset(args.data_dir, train=True), args.limit_train)
    test_data = maybe_limit(mnist_dataset(args.data_dir, train=False), args.limit_test)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    model = TernaryDINOv3TRMClassifier(
        num_classes=10,
        image_size=32,
        patch_size=8,
        in_chans=3,
        dropout=args.dropout,
        scale_op=args.scale_op,
        track_latent_grads=args.track_latent_grads,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01,
    )

    start_epoch = 0
    best_accuracy = float("-inf")

    if args.resume is not None:
        payload = load_checkpoint(args.resume, device)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"])
        best_accuracy = float(payload.get("best_accuracy", best_accuracy))

    args.save.parent.mkdir(parents=True, exist_ok=True)
    best_path = args.save.with_name(args.save.stem + "-best" + args.save.suffix)
    ternary_path = args.save.with_name(
        args.save.stem + "-best-ternary" + args.save.suffix
    )
    class_names = [str(index) for index in range(10)]

    print(
        f"training vit_femto TRM on {device}: "
        f"{len(train_data)} train / {len(test_data)} test; "
        f"supervision={args.supervision_steps}, "
        f"n=[{args.n_min},{args.n_max}], T=[{args.t_min},{args.t_max}]"
    )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            supervision_steps=args.supervision_steps,
            n_min=args.n_min,
            n_max=args.n_max,
            t_min=args.t_min,
            t_max=args.t_max,
            confidence=args.confidence,
            confidence_loss_weight=args.confidence_loss_weight,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            verbose_supervision=args.verbose_supervision,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            n=args.eval_n,
            T=args.eval_t,
            max_supervision=args.supervision_steps,
            confidence=args.confidence,
        )

        ternary_model = freeze_ternary_weights(model).to(device)
        ternary_metrics = evaluate(
            ternary_model,
            test_loader,
            device,
            n=args.eval_n,
            T=args.eval_t,
            max_supervision=args.supervision_steps,
            confidence=args.confidence,
        )
        del ternary_model

        scheduler.step()

        metrics = {
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "train_confidence_accuracy": train_metrics.confidence_accuracy,
            "train_average_supervision_steps": train_metrics.average_supervision_steps,
            "test_loss": test_metrics.loss,
            "test_accuracy": test_metrics.accuracy,
            "test_average_supervision_steps": test_metrics.average_supervision_steps,
            "ternary_test_loss": ternary_metrics.loss,
            "ternary_test_accuracy": ternary_metrics.accuracy,
            "ternary_average_supervision_steps": ternary_metrics.average_supervision_steps,
        }

        print(
            f"epoch={epoch:02d}/{args.epochs} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"train_acc={metrics['train_accuracy']:.4f} "
            f"q_acc={metrics['train_confidence_accuracy']:.4f} "
            f"train_steps={metrics['train_average_supervision_steps']:.2f} "
            f"test_loss={metrics['test_loss']:.4f} "
            f"test_acc={metrics['test_accuracy']:.4f} "
            f"test_steps={metrics['test_average_supervision_steps']:.2f} "
            f"ternary_loss={metrics['ternary_test_loss']:.4f} "
            f"ternary_acc={metrics['ternary_test_accuracy']:.4f} "
            f"ternary_steps={metrics['ternary_average_supervision_steps']:.2f}"
        )

        if metrics["test_accuracy"] >= best_accuracy:
            best_accuracy = metrics["test_accuracy"]
            save_weights(
                best_path,
                model=model,
                epoch=epoch,
                metrics=metrics,
                class_names=class_names,
            )
            save_weights(
                ternary_path,
                model=model,
                epoch=epoch,
                metrics=metrics,
                frozen=True,
                class_names=class_names,
            )

        save_training_checkpoint(
            args.save,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_accuracy=best_accuracy,
            metrics=metrics,
            class_names=class_names,
        )


if __name__ == "__main__":
    main()
