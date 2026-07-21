"""Train a ternary DINOv3 ViT-Femto classifier on MNIST."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from ternary_dinov3.checkpoints import (
    freeze_ternary_weights,
    load_checkpoint,
    save_training_checkpoint,
    save_weights,
)
from ternary_dinov3.engine import run_epoch, select_device
from ternary_dinov3.model import TernaryDINOv3Classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--save", type=Path, default=Path("checkpoints/mnist-femto.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--scale-op", choices=("mean", "median"), default="median")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--limit-train", type=int, help="Use only N training examples for a quick trial"
    )
    parser.add_argument(
        "--limit-test", type=int, help="Use only N test examples for a quick trial"
    )
    return parser.parse_args()


def mnist_dataset(root: Path, *, train: bool) -> Dataset:
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
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    train_data = maybe_limit(mnist_dataset(args.data_dir, train=True), args.limit_train)
    test_data = maybe_limit(mnist_dataset(args.data_dir, train=False), args.limit_test)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    model = TernaryDINOv3Classifier(
        "vit_femto",
        num_classes=10,
        image_size=32,
        patch_size=8,
        scale_op=args.scale_op,
        backbone_kwargs={"no_conv":True}, # no any cnn layer
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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

    best_path = args.save.with_name(args.save.stem + "-best" + args.save.suffix)
    ternary_path = args.save.with_name(args.save.stem + "-best-ternary" + args.save.suffix)
    class_names = [str(index) for index in range(10)]
    print(f"training vit_femto on {device}: {len(train_data)} train / {len(test_data)} test")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        test_metrics = run_epoch(model, test_loader, device)
        ternary_model = freeze_ternary_weights(model).to(device)
        ternary_metrics = run_epoch(ternary_model, test_loader, device)
        del ternary_model
        scheduler.step()
        metrics = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "ternary_test_loss": ternary_metrics["loss"],
            "ternary_test_accuracy": ternary_metrics["accuracy"],
        }
        print(
            f"epoch={epoch:02d}/{args.epochs} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"train_acc={metrics['train_accuracy']:.4f} "
            f"test_loss={metrics['test_loss']:.4f} "
            f"test_acc={metrics['test_accuracy']:.4f} "
            f"ternary_loss={metrics['ternary_test_loss']:.4f} "
            f"ternary_acc={metrics['ternary_test_accuracy']:.4f}"
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
