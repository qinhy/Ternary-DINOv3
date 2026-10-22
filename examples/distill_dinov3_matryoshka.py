"""Distill official DINOv3 CLS and patch tokens into shared Matryoshka subnets.

Example:
    uv run python examples/distill_dinov3_matryoshka.py \\
      --teacher-repo ../dinov3 --teacher-weights path/to/vits16.pth \\
      --train-dir data/distill/train --val-dir data/distill/val --device cuda

Run without images or official teacher weights using --smoke-test --device cpu.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler

from ternary_dinov3.checkpoints import _atomic_save, freeze_ternary_weights, load_checkpoint
from ternary_dinov3.data import (
    SyntheticClassificationDataset,
    UnlabeledImageDataset,
    collate_image_groups,
    create_loader,
)
from ternary_dinov3.distillation_losses import TeacherCLSPCA, fit_teacher_cls_pca
from ternary_dinov3.engine import select_device
from ternary_dinov3.matryoshka import (
    MATRYOSHKA_ARCHITECTURES,
    build_exported_backbone,
    build_matryoshka_backbone,
    export_matryoshka_subnetwork,
    exported_backbone_config,
    initialize_from_teacher,
    run_matryoshka_distillation_epoch,
    selected_subnets,
)
from ternary_dinov3.train import _optimizer_to_device

TRAINING_FORMAT = "ternary-dinov3-matryoshka-distillation-v2"
LEGACY_TRAINING_FORMAT = "ternary-dinov3-matryoshka-distillation-v1"
WEIGHTS_FORMAT = "ternary-dinov3-matryoshka-distillation-weights-v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-repo", type=Path, help="Official DINOv3 clone")
    parser.add_argument("--teacher-weights", type=Path, help="Local official teacher checkpoint")
    parser.add_argument(
        "--teacher-model",
        choices=("dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"),
        default="dinov3_vits16",
    )
    parser.add_argument("--train-dir", type=Path, help="Training images, flat or nested")
    parser.add_argument("--val-dir", type=Path, help="Validation images, flat or nested")
    parser.add_argument(
        "--architecture", choices=MATRYOSHKA_ARCHITECTURES, default="vit_matryoshka_small"
    )
    parser.add_argument(
        "--subnets", nargs="+", help="Names to train/export; default: all architecture splits"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Square resize; default: preserve resolution and pad to multiples of 16",
    )
    parser.add_argument(
        "--pca-samples",
        type=int,
        default=8192,
        help="Maximum training images for fixed CLS PCA calibration",
    )
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--patch-loss-weight", type=float, default=1.0)
    parser.add_argument("--patch-temperature", type=float, default=0.1)
    parser.add_argument(
        "--patch-query-chunk-size",
        type=int,
        default=256,
        help="Similarity query rows per chunk; 0 computes the full matrix",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--scale-op", choices=("mean", "median"), default="median")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save", type=Path, default=Path("checkpoints/distilled-matryoshka.pt"))
    parser.add_argument(
        "--smoke-test", action="store_true", help="One synthetic Pico/Femto epoch; no downloads"
    )
    args = parser.parse_args(argv)
    if args.image_size is not None and (args.image_size <= 0 or args.image_size % 16):
        parser.error("--image-size must be a positive multiple of 16")
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0 or args.log_interval < 0:
        parser.error("epochs/batch size must be positive; workers/log interval non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative and finite")
    if args.pca_samples < 2:
        parser.error("--pca-samples must be at least 2")
    if args.patch_query_chunk_size < 0:
        parser.error("--patch-query-chunk-size must be non-negative")
    if not math.isfinite(args.patch_temperature) or args.patch_temperature <= 0:
        parser.error("--patch-temperature must be positive and finite")
    if any(not math.isfinite(w) or w < 0 for w in (args.cls_loss_weight, args.patch_loss_weight)):
        parser.error("loss weights must be finite and non-negative")
    if args.cls_loss_weight + args.patch_loss_weight == 0:
        parser.error("at least one loss weight must be positive")
    if not args.smoke_test:
        if args.teacher_repo is None or not (args.teacher_repo / "hubconf.py").is_file():
            parser.error("--teacher-repo must contain the official DINOv3 hubconf.py")
        if args.teacher_weights is None or not args.teacher_weights.is_file():
            parser.error("--teacher-weights must be a local official teacher checkpoint")
        if args.train_dir is None or args.val_dir is None:
            parser.error("--train-dir and --val-dir are required")
    return args


def _output_path(save: Path, suffix: str) -> Path:
    return save.with_name(save.stem + suffix + save.suffix)


def export_subnets(
    student: torch.nn.Module,
    save: Path,
    *,
    names: list[str],
    scale_op: str,
    metadata: dict[str, Any],
) -> None:
    """Write learned and frozen ordinary backbones, one subnet at a time."""
    for name in names:
        compact = export_matryoshka_subnetwork(student, name, scale_op=scale_op).eval()
        config = exported_backbone_config(compact, scale_op=scale_op)
        weights = {
            **metadata,
            "format": "ternary-dinov3-matryoshka-subnetwork-weights-v1",
            "name": name,
            "backbone_config": config,
            "state_dict": compact.state_dict(),
        }
        _atomic_save(weights, _output_path(save, f"-best-{name}"))
        frozen = freeze_ternary_weights(compact)
        _atomic_save(
            {
                **weights,
                "format": "ternary-dinov3-matryoshka-subnetwork-ternary-v1",
                "state_dict": frozen.state_dict(),
            },
            _output_path(save, f"-best-{name}-ternary"),
        )
        print(f"exported {name}: {sum(p.numel() for p in compact.parameters()):,} QAT parameters")
        del compact, frozen, weights


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    if args.smoke_test:
        args.architecture = "vit_matryoshka_pico"
        args.image_size, args.batch_size, args.epochs, args.workers = 32, 2, 1, 0
    config = {
        "architecture": args.architecture,
        "image_size": args.image_size or 224,
        "patch_size": 16,
        "scale_op": args.scale_op,
        "no_conv": True,
        "pos_embed_rope_base": 100.0,
        "pos_embed_rope_normalize_coords": "separate",
        "pos_embed_rope_rescale_coords": 2.0,
        "pos_embed_rope_dtype": "fp32",
        "layerscale_init": 1e-5,
        "norm_layer": "layernormbf16",
        "n_storage_tokens": 4,
        "mask_k_bias": True,
    }
    input_config = {"image_size": args.image_size, "patch_size": 16}
    loss_config = {
        "cls_target": "centered_teacher_pca",
        "patch_target": "relational_kl_batchmean",
        "cls_loss_weight": args.cls_loss_weight,
        "patch_loss_weight": args.patch_loss_weight,
        "patch_temperature": args.patch_temperature,
    }
    if args.smoke_test:
        teacher_config = {"model": "synthetic-pico", "seed": args.seed}
        teacher = build_exported_backbone(
            img_size=32,
            patch_size=16,
            embed_dim=64,
            depth=6,
            num_heads=1,
            ffn_ratio=4.0,
            n_storage_tokens=4,
            layerscale_init=1e-5,
            norm_layer="layernormbf16",
            mask_k_bias=True,
            pos_embed_rope_dtype="fp32",
        )
    else:
        teacher_config = {
            "model": args.teacher_model,
            "repo": str(args.teacher_repo.resolve()),
            "weights": str(args.teacher_weights.resolve()),
        }
        teacher = torch.hub.load(
            teacher_config["repo"],
            args.teacher_model,
            source="local",
            pretrained=True,
            weights=teacher_config["weights"],
        )
    teacher.eval().requires_grad_(False)
    student = build_matryoshka_backbone(**config)
    names = selected_subnets(student, args.subnets)
    if student.embed_dim > teacher.embed_dim:
        raise ValueError("Choose a teacher at least as wide as the largest student architecture")
    payload = load_checkpoint(args.resume) if args.resume else None
    start_epoch, best_loss = 0, float("inf")
    if payload is None:
        initialize_from_teacher(student, teacher)
    else:
        if payload.get("format") not in (TRAINING_FORMAT, LEGACY_TRAINING_FORMAT):
            raise ValueError("resume requires a Matryoshka distillation training checkpoint")
        previous_config = dict(payload["backbone_config"])
        # Image size is nominal model metadata; preprocessing is compared separately.
        previous_config["image_size"] = config["image_size"]
        if previous_config != config or payload["subnets"] != names:
            raise ValueError("resume student configuration or selected subnets differ")
        if payload["teacher_config"] != teacher_config:
            raise ValueError("resume teacher configuration differs")
        student.load_state_dict(payload["state_dict"], strict=True)
        start_epoch, best_loss = payload["epoch"], payload["best_loss"]
        if payload.get("loss_config") != loss_config:
            best_loss = float("inf")
            print("distillation objective changed; resetting best validation loss", flush=True)
        if payload["input_config"] != input_config:
            best_loss = float("inf")
            print("input preprocessing changed; resetting best validation loss", flush=True)
    student.to(device)
    teacher.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scheduler.T_max = args.epochs
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs, strict=True):
            group["lr"] = (
                scheduler.eta_min
                + (base_lr - scheduler.eta_min)
                * (1 + math.cos(math.pi * start_epoch / args.epochs))
                / 2
            )
        random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if args.smoke_test:
        train_data = SyntheticClassificationDataset(4, 32, 1, args.seed)
        val_data = SyntheticClassificationDataset(2, 32, 1, args.seed + 1)
    else:
        train_data = UnlabeledImageDataset(args.train_dir, args.image_size, training=True)
        val_data = UnlabeledImageDataset(args.val_dir, args.image_size, training=False)
    train_loader = create_loader(
        train_data, batch_size=args.batch_size, workers=args.workers, training=True
    )
    val_loader = create_loader(
        val_data, batch_size=args.batch_size, workers=args.workers, training=False
    )
    if payload is not None and "cls_pca" in payload:
        cls_pca = TeacherCLSPCA.from_state_dict(payload["cls_pca"])
        if cls_pca.mean.numel() != teacher.embed_dim or cls_pca.components.shape[1] < max(
            student.splits[name][0] for name in names
        ):
            raise ValueError("saved CLS PCA does not fit the teacher or selected student widths")
        print(f"restored fixed CLS PCA from {int(cls_pca.sample_count)} training images")
    else:
        calibration_data = (
            train_data
            if args.smoke_test
            else UnlabeledImageDataset(args.train_dir, args.image_size, training=False)
        )
        # A separate generator keeps calibration sampling out of the training RNG stream.
        calibration_loader = DataLoader(
            calibration_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=RandomSampler(
                calibration_data, generator=torch.Generator().manual_seed(args.seed)
            ),
            collate_fn=collate_image_groups,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"fitting fixed CLS PCA on up to {args.pca_samples} training images", flush=True)
        cls_pca = fit_teacher_cls_pca(
            teacher,
            calibration_loader,
            device,
            max_samples=args.pca_samples,
            output_dim=max(student.splits[name][0] for name in names),
        )
        print(f"fitted CLS PCA using {int(cls_pca.sample_count)} images", flush=True)
    cls_pca.to(device)
    print(f"distilling {teacher_config['model']} into {args.architecture} on {device}")
    print(f"subnets: { {name: student.splits[name] for name in names} }")
    print(f"images: train={len(train_data)} val={len(val_data)}", flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        options = {
            "subnets": names,
            "cls_pca": cls_pca,
            "cls_loss_weight": args.cls_loss_weight,
            "patch_loss_weight": args.patch_loss_weight,
            "patch_temperature": args.patch_temperature,
            "patch_query_chunk_size": args.patch_query_chunk_size or None,
            "log_interval": args.log_interval,
            "log_prefix": f"epoch={epoch}/{args.epochs}",
        }
        train = run_matryoshka_distillation_epoch(
            student, teacher, train_loader, device, optimizer=optimizer, **options
        )
        val = run_matryoshka_distillation_epoch(student, teacher, val_loader, device, **options)
        scheduler.step()
        metrics = {
            "train_loss": train["loss"],
            "val_loss": val["loss"],
            "train_subnet_losses": train["subnet_losses"],
            "val_subnet_losses": val["subnet_losses"],
            "train_cls_loss": train["cls_loss"],
            "val_cls_loss": val["cls_loss"],
            "train_patch_loss": train["patch_loss"],
            "val_patch_loss": val["patch_loss"],
            "train_subnet_cls_losses": train["subnet_cls_losses"],
            "val_subnet_cls_losses": val["subnet_cls_losses"],
            "train_subnet_patch_losses": train["subnet_patch_losses"],
            "val_subnet_patch_losses": val["subnet_patch_losses"],
        }
        print(f"epoch={epoch} train_loss={train['loss']:.6f} val_loss={val['loss']:.6f}")
        for name in names:
            print(
                f"  {name}: val_loss={val['subnet_losses'][name]:.6f} "
                f"cls_mse={val['subnet_cls_losses'][name]:.6f} "
                f"patch_kl={val['subnet_patch_losses'][name]:.6f}"
            )
        metadata = {
            "epoch": epoch,
            "input_config": input_config,
            "teacher_config": teacher_config,
            "metrics": metrics,
            "loss_config": loss_config,
            "cls_pca": {
                key: value.detach().cpu().clone() for key, value in cls_pca.state_dict().items()
            },
        }
        weights = {
            **metadata,
            "format": WEIGHTS_FORMAT,
            "backbone_config": config,
            "subnets": names,
            "splits": dict(student.splits),
            "state_dict": student.state_dict(),
        }
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            _atomic_save(weights, _output_path(args.save, "-best"))
            export_subnets(
                student, args.save, names=names, scale_op=args.scale_op, metadata=metadata
            )
        checkpoint = {
            **weights,
            "format": TRAINING_FORMAT,
            "best_loss": best_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        _atomic_save(checkpoint, args.save)


if __name__ == "__main__":
    main()

# from pathlib import Path
# from ternary_dinov3.checkpoints import (
#     freeze_ternary_weights,
#     load_checkpoint,
# )
# from ternary_dinov3.gui import DinoPatchSimilarityViewer
# from ternary_dinov3.matryoshka import build_exported_backbone

# def load_ternary_checkpoint(p="./distilled-matryoshka-best-tiny-ternary.pt"):
#     checkpoint = load_checkpoint(Path(p),device="cpu",)

#     model = build_exported_backbone(
#         **checkpoint["backbone_config"],initialize=False,)
#     model = freeze_ternary_weights(model)
#     model.load_state_dict(checkpoint["state_dict"], strict=True)
#     model.eval()
#     return model

# images = sorted(Path("your/images").rglob("*.jpg"))

# DinoPatchSimilarityViewer.show_images(
#     model,
#     images,
#     size=224,
#     batch_size=1,
# )