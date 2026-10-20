"""Distill official DINOv3 ViT-S/16 embeddings into a ternary ViT-S/16.
uv run python examples/distill_dinov3.py \
  --teacher-repo ../dinov3 \
  --teacher-weights ../data/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --train-dir data/distill/train \
  --val-dir data/distill/val \
  --device cuda
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from ternary_dinov3.checkpoints import (
    _atomic_save,
    freeze_ternary_weights,
    load_checkpoint,
    load_pretrained_backbone,
)
from ternary_dinov3.data import UnlabeledImageDataset, create_loader
from ternary_dinov3.distillation import run_distillation_epoch
from ternary_dinov3.engine import select_device
from ternary_dinov3.model import build_backbone
from ternary_dinov3.train import _optimizer_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-repo", type=Path, required=True, help="Official DINOv3 clone")
    parser.add_argument(
        "--teacher-weights", type=Path, required=True, help="Official ViT-S/16 weights"
    )
    parser.add_argument(
        "--train-dir", type=Path, required=True, help="Training images (flat or nested folders)"
    )
    parser.add_argument(
        "--val-dir", type=Path, required=True, help="Validation images (flat or nested folders)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Resize to a square; by default preserve original size and pad to multiples of 16",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Print progress every N batches, plus first and last; 0 disables batch logs",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--scale-op", choices=("mean", "median"), default="median")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save", type=Path, default=Path("checkpoints/distilled-vits16.pt"))
    args = parser.parse_args()
    if args.image_size is not None and (args.image_size <= 0 or args.image_size % 16):
        parser.error("--image-size must be a positive multiple of 16")
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers must be non-negative")
    if args.log_interval < 0:
        parser.error("--log-interval must be non-negative")
    if not (args.teacher_repo / "hubconf.py").is_file():
        parser.error("--teacher-repo must contain the official DINOv3 hubconf.py")
    if not args.teacher_weights.is_file():
        parser.error("--teacher-weights must be a local official ViT-S/16 checkpoint")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    teacher = (
        torch.hub.load(
            str(args.teacher_repo.resolve()),
            "dinov3_vits16",
            source="local",
            pretrained=True,
            weights=str(args.teacher_weights.resolve()),
        )
        .eval()
        .requires_grad_(False)
    )

    # Match the official small model, including settings absent from factory defaults.
    config = {
        "architecture": "vit_small",
        "image_size": args.image_size or 224,
        "patch_size": 16,
        "scale_op": args.scale_op,
        "pos_embed_rope_base": 100.0,
        "pos_embed_rope_normalize_coords": "separate",
        "pos_embed_rope_rescale_coords": 2.0,
        "pos_embed_rope_dtype": "fp32",
        "layerscale_init": 1e-5,
        "norm_layer": "layernormbf16",
        "n_storage_tokens": 4,
        "mask_k_bias": True,
    }
    payload = load_checkpoint(args.resume) if args.resume else None
    input_config = {"image_size": args.image_size, "patch_size": 16}
    if payload is not None:
        if payload.get("format") != "ternary-dinov3-distillation-v1":
            raise ValueError("resume requires a distillation training checkpoint")
        if payload["backbone_config"] != config:
            raise ValueError("resume backbone configuration differs from the requested options")
    student = build_backbone(**config)
    if payload is None:
        load_pretrained_backbone(student, teacher.state_dict())
    else:
        student.load_state_dict(payload["state_dict"])
    student.to(device)
    teacher.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch, best_loss = 0, float("inf")
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        start_epoch, best_loss = payload["epoch"], payload["best_loss"]
        previous_input = payload.get(
            "input_config", {"image_size": config["image_size"], "patch_size": 16}
        )
        if previous_input != input_config:
            best_loss = float("inf")
            print("input preprocessing changed; resetting best validation MSE", flush=True)
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scheduler.T_max = args.epochs
        # Recompute the current rate when extending a completed cosine schedule.
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs, strict=True):
            group["lr"] = (
                scheduler.eta_min
                + (base_lr - scheduler.eta_min)
                * (1 + math.cos(math.pi * start_epoch / args.epochs))
                / 2
            )

    train_loader = create_loader(
        UnlabeledImageDataset(args.train_dir, args.image_size, training=True),
        batch_size=args.batch_size,
        workers=args.workers,
        training=True,
    )
    val_loader = create_loader(
        UnlabeledImageDataset(args.val_dir, args.image_size, training=False),
        batch_size=args.batch_size,
        workers=args.workers,
        training=False,
    )
    print(f"distilling official ViT-S/16 into ternary vit_small on {device}")
    print(
        "input: original resolution, minimal right/bottom padding to patch size 16"
        if args.image_size is None
        else f"input: resize to {args.image_size}x{args.image_size}",
        flush=True,
    )
    print(
        f"images: train={len(train_loader.dataset)} val={len(val_loader.dataset)}",
        flush=True,
    )
    for epoch in range(start_epoch + 1, args.epochs + 1):
        log_options = {
            "log_interval": args.log_interval,
            "log_prefix": f"epoch={epoch}/{args.epochs}",
        }
        train = run_distillation_epoch(
            student, teacher, train_loader, device, optimizer=optimizer, **log_options
        )
        val = run_distillation_epoch(student, teacher, val_loader, device, **log_options)
        scheduler.step()
        metrics = {"train_loss": train["loss"], "val_loss": val["loss"]}
        print(
            f"epoch={epoch}/{args.epochs} train_mse={train['loss']:.6f} val_mse={val['loss']:.6f}"
        )
        weights = {
            "format": "ternary-dinov3-distillation-weights-v1",
            "epoch": epoch,
            "backbone_config": config,
            "input_config": input_config,
            "state_dict": student.state_dict(),
            "metrics": metrics,
        }
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best = args.save.with_name(args.save.stem + "-best" + args.save.suffix)
            _atomic_save(weights, best)
            frozen = freeze_ternary_weights(student)
            _atomic_save(
                {
                    **weights,
                    "format": "ternary-dinov3-distillation-ternary-v1",
                    "state_dict": frozen.state_dict(),
                },
                args.save.with_name(args.save.stem + "-best-ternary" + args.save.suffix),
            )
            del frozen
        _atomic_save(
            {
                **weights,
                "format": "ternary-dinov3-distillation-v1",
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
            },
            args.save,
        )


if __name__ == "__main__":
    main()
