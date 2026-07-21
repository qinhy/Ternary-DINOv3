"""Fine-tune a ternary DINOv3 classifier on ImageFolder data."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch

from .checkpoints import (
    TERNARY_FORMAT,
    load_checkpoint,
    load_pretrained_backbone,
    save_training_checkpoint,
    save_weights,
)
from .data import SyntheticClassificationDataset, create_loader, imagefolder
from .engine import run_epoch, select_device
from .model import ARCHITECTURES, TernaryDINOv3Classifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, help="ImageFolder training root")
    parser.add_argument("--val-dir", type=Path, help="ImageFolder validation root")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="vit_small")
    parser.add_argument("--num-classes", type=int, help="Inferred from --train-dir when omitted")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--scale-op", choices=("mean", "median"), default="median")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pretrained", type=Path, help="DINOv3 backbone checkpoint")
    parser.add_argument("--resume", type=Path, help="Project training checkpoint")
    parser.add_argument("--save", type=Path, default=Path("checkpoints/ternary-dinov3.pt"))
    parser.add_argument("--smoke-test", action="store_true", help="Run one tiny synthetic epoch")
    return parser.parse_args(argv)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    if args.smoke_test:
        args.architecture = "vit_femto"
        args.image_size = 32
        args.patch_size = 8
        args.batch_size = 2
        args.epochs = 1
        args.workers = 0
        num_classes = args.num_classes or 4
        train_data = SyntheticClassificationDataset(4, args.image_size, num_classes, args.seed)
        val_data = SyntheticClassificationDataset(4, args.image_size, num_classes, args.seed + 1)
        class_names = [f"class-{index}" for index in range(num_classes)]
    else:
        if args.train_dir is None:
            raise SystemExit("--train-dir is required unless --smoke-test is used")
        train_data = imagefolder(args.train_dir, args.image_size, training=True)
        val_data = imagefolder(args.val_dir or args.train_dir, args.image_size, training=False)
        class_names = list(train_data.classes)
        if val_data.classes != train_data.classes:
            raise ValueError("training and validation class names differ")
        num_classes = args.num_classes or len(class_names)
        if num_classes != len(class_names):
            raise ValueError("--num-classes must match the number of ImageFolder classes")

    train_loader = create_loader(
        train_data, batch_size=args.batch_size, workers=args.workers, training=True
    )
    val_loader = create_loader(
        val_data, batch_size=args.batch_size, workers=args.workers, training=False
    )
    model = TernaryDINOv3Classifier(
        args.architecture,
        num_classes=num_classes,
        image_size=args.image_size,
        patch_size=args.patch_size,
        scale_op=args.scale_op,
        dropout=args.dropout,
    )
    if args.pretrained:
        load_pretrained_backbone(model.backbone, args.pretrained)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_accuracy = float("-inf")
    if args.resume:
        payload = load_checkpoint(args.resume, device)
        if payload.get("format") == TERNARY_FORMAT:
            raise ValueError("a frozen ternary checkpoint cannot be resumed")
        model.load_state_dict(payload["state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            _optimizer_to_device(optimizer, device)
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            start_epoch = int(payload.get("epoch", 0))
            best_accuracy = float(payload.get("best_accuracy", best_accuracy))

    best_path = args.save.with_name(args.save.stem + "-best" + args.save.suffix)
    ternary_path = args.save.with_name(args.save.stem + "-best-ternary" + args.save.suffix)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        scheduler.step()
        metrics = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        print(
            f"epoch={epoch}/{args.epochs} train_loss={metrics['train_loss']:.4f} "
            f"train_acc={metrics['train_accuracy']:.4f} val_loss={metrics['val_loss']:.4f} "
            f"val_acc={metrics['val_accuracy']:.4f}"
        )
        if metrics["val_accuracy"] >= best_accuracy:
            best_accuracy = metrics["val_accuracy"]
            save_weights(
                best_path, model=model, epoch=epoch, metrics=metrics, class_names=class_names
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
