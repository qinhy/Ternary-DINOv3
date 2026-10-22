"""Matryoshka backbone construction, prefix distillation, and compact export."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import torch
from ternarylayers.dinov3.models.vision_transformer import DinoVisionTransformer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .distillation_losses import TeacherCLSPCA, relational_kd_kl
from .model import _fix_masked_projections, _set_scale_op

MATRYOSHKA_ARCHITECTURES = tuple(
    f"vit_matryoshka_{name}"
    for name in ("femto", "pico", "micro", "nano", "tiny", "small", "base", "large")
)


def build_matryoshka_backbone(
    architecture: str = "vit_matryoshka_small",
    *,
    image_size: int = 224,
    patch_size: int = 16,
    scale_op: str = "median",
    initialize: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Build a shared QAT backbone using the TernaryCNN Matryoshka factories."""
    if architecture not in MATRYOSHKA_ARCHITECTURES:
        raise ValueError(f"unknown Matryoshka architecture {architecture!r}")
    try:
        from ternarylayers.dinov3.models import vision_transformer_matryoshka as models
    except ImportError as error:
        raise ImportError("Install a TernaryCNN revision with Matryoshka support") from error
    model = getattr(models, architecture)(img_size=image_size, patch_size=patch_size, **kwargs)
    _fix_masked_projections(model)
    _set_scale_op(model, scale_op)
    if initialize:
        model.init_weights()
    return model


@torch.no_grad()
def initialize_from_teacher(student: nn.Module, teacher: nn.Module) -> None:
    """Copy a compatible full model or a width/depth prefix of a larger teacher.

    Q/K/V row groups are sliced separately. A convolutional teacher patch
    projector is flattened when the student uses a no-convolution projector.
    """
    teacher_width = teacher.embed_dim
    if student.embed_dim > teacher_width:
        raise ValueError("student width cannot exceed teacher width")
    if student.head_dim != teacher_width // teacher.num_heads:
        raise ValueError("student and teacher must use the same attention head dimension")
    source = teacher.state_dict()
    copied = {}
    for key, destination in student.state_dict().items():
        if key not in source:
            raise ValueError(f"teacher is missing student state key {key!r}")
        value = source[key]
        if key == "patch_embed.proj.weight" and value.ndim == 4 and destination.ndim == 2:
            value = value.flatten(1)
        if value.shape == destination.shape:
            copied[key] = value.clone()
            continue
        if value.ndim != destination.ndim or any(
            size < needed for size, needed in zip(value.shape, destination.shape, strict=True)
        ):
            raise ValueError(
                f"incompatible teacher state {key}: {tuple(value.shape)} -> "
                f"{tuple(destination.shape)}"
            )
        if ".attn.qkv." in key and value.shape[0] == 3 * teacher_width:
            groups = value.reshape(3, teacher_width, *value.shape[1:])
            slices = (slice(None), slice(0, student.embed_dim)) + tuple(
                slice(0, size) for size in destination.shape[1:]
            )
            value = groups[slices].reshape(destination.shape)
        else:
            value = value[tuple(slice(0, size) for size in destination.shape)]
        copied[key] = value.clone()
    student.load_state_dict(copied, strict=True)


def selected_subnets(student: nn.Module, subnets: Sequence[str] | None = None) -> list[str]:
    names = list(student.splits if subnets is None else subnets)
    if not names or len(names) != len(set(names)):
        raise ValueError("subnets must be nonempty and contain no duplicates")
    unknown = set(names).difference(student.splits)
    if unknown:
        raise ValueError(f"unknown subnets {sorted(unknown)}; available: {tuple(student.splits)}")
    return names


def run_matryoshka_distillation_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    cls_pca: TeacherCLSPCA,
    optimizer: torch.optim.Optimizer | None = None,
    subnets: Sequence[str] | None = None,
    cls_loss_weight: float = 1.0,
    patch_loss_weight: float = 1.0,
    patch_temperature: float = 0.1,
    patch_query_chunk_size: int | None = 256,
    log_interval: int = 25,
    log_prefix: str = "",
) -> dict[str, Any]:
    """Match CLS to fixed teacher PCA scores, and patches by relational KL.

    Average the weighted sum of these losses over selected subnets and images.
    Patch relations use all teacher channels; CLS targets use the leading PCA
    components at the active width. A resolution group needs one teacher pass.
    Gradients accumulate across subnets/groups into one step per loader batch.
    """
    if log_interval < 0:
        raise ValueError("log_interval must be non-negative")
    if any(not math.isfinite(w) or w < 0 for w in (cls_loss_weight, patch_loss_weight)):
        raise ValueError("loss weights must be finite and non-negative")
    if cls_loss_weight + patch_loss_weight == 0:
        raise ValueError("at least one loss weight must be positive")
    if not math.isfinite(patch_temperature) or patch_temperature <= 0:
        raise ValueError("patch_temperature must be positive and finite")
    if patch_query_chunk_size is not None and patch_query_chunk_size <= 0:
        raise ValueError("patch_query_chunk_size must be positive or None")
    names = selected_subnets(student, subnets)
    teacher.requires_grad_(False).eval()
    cls_pca.to(device).eval().requires_grad_(False)
    student.train(optimizer is not None)
    totals = {key: dict.fromkeys(names, 0.0) for key in ("loss", "cls", "patch")}
    count = 0
    total_batches = len(loader)
    phase = "train" if optimizer is not None else "val"
    label = f"{log_prefix} {phase}".strip()
    started = time.perf_counter()
    if log_interval:
        print(
            f"{label} starting batches={total_batches} subnets={names} device={device}", flush=True
        )
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch_index, (images, _) in enumerate(loader, start=1):
            groups = [images] if isinstance(images, torch.Tensor) else images
            batch = sum(group.shape[0] for group in groups)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            for group in groups:
                group = group.to(device, non_blocking=True)
                with torch.no_grad():
                    target = teacher.forward_features(group)
                for name in names:
                    width, depth = student.splits[name]
                    with torch.no_grad():
                        expected_cls = cls_pca(target["x_norm_clstoken"], width)
                    output = student.forward_features(group, width=width, depth=depth)
                    actual_cls = output["x_norm_clstoken"].float()
                    if actual_cls.shape != expected_cls.shape:
                        raise ValueError(f"subnet {name!r}: student and PCA CLS shapes must match")
                    cls_loss = F.mse_loss(actual_cls, expected_cls)
                    patch_loss = relational_kd_kl(
                        output["x_norm_patchtokens"],
                        target["x_norm_patchtokens"],
                        T=patch_temperature,
                        drop_cls=False,
                        mask_diag=True,
                        query_chunk_size=patch_query_chunk_size,
                    )
                    loss = cls_loss_weight * cls_loss + patch_loss_weight * patch_loss
                    if optimizer is not None:
                        (loss * (group.shape[0] / batch / len(names))).backward()
                    for key, value in (("loss", loss), ("cls", cls_loss), ("patch", patch_loss)):
                        totals[key][name] += float(value.detach()) * group.shape[0]
                    del output, expected_cls, actual_cls, loss, cls_loss, patch_loss
                del target
            if optimizer is not None:
                optimizer.step()
            count += batch
            if log_interval and (
                batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches
            ):
                elapsed = time.perf_counter() - started
                averages = {
                    key: sum(values.values()) / len(names) / count for key, values in totals.items()
                }
                eta = elapsed / batch_index * (total_batches - batch_index)
                lr = f" lr={optimizer.param_groups[0]['lr']:.3e}" if optimizer is not None else ""
                print(
                    f"{label} batch={batch_index}/{total_batches} samples={count} "
                    f"loss={averages['loss']:.6f} cls_mse={averages['cls']:.6f} "
                    f"patch_kl={averages['patch']:.6f}{lr} elapsed={elapsed:.1f}s "
                    f"eta={eta:.1f}s images/s={count / max(elapsed, 1e-9):.1f}",
                    flush=True,
                )
    if count == 0:
        raise ValueError("the data loader produced no examples")
    components = {
        key: {name: value / count for name, value in values.items()}
        for key, values in totals.items()
    }
    return {
        "loss": sum(components["loss"].values()) / len(names),
        "cls_loss": sum(components["cls"].values()) / len(names),
        "patch_loss": sum(components["patch"].values()) / len(names),
        "subnet_losses": components["loss"],
        "subnet_cls_losses": components["cls"],
        "subnet_patch_losses": components["patch"],
    }


def export_matryoshka_subnetwork(
    student: nn.Module, name: str, *, scale_op: str = "median"
) -> nn.Module:
    """Extract before freezing; preserve scale selection and masked K biases."""
    compact = student.export_subnetwork(name)
    _fix_masked_projections(compact)
    _set_scale_op(compact, scale_op)
    return compact


def exported_backbone_config(backbone: nn.Module, *, scale_op: str = "median") -> dict[str, Any]:
    """Describe the ordinary exported ViT, including all non-default options."""
    names = (
        "img_size",
        "patch_size",
        "in_chans",
        "embed_dim",
        "depth",
        "num_heads",
        "ffn_ratio",
        "pos_embed_rope_base",
        "pos_embed_rope_min_period",
        "pos_embed_rope_max_period",
        "pos_embed_rope_normalize_coords",
        "pos_embed_rope_shift_coords",
        "pos_embed_rope_jitter_coords",
        "pos_embed_rope_rescale_coords",
        "pos_embed_rope_dtype",
        "qkv_bias",
        "drop_path_rate",
        "layerscale_init",
        "norm_layer",
        "ffn_layer",
        "ffn_bias",
        "proj_bias",
        "n_storage_tokens",
        "mask_k_bias",
        "untie_cls_and_patch_norms",
        "untie_global_and_local_cls_norm",
        "no_conv",
    )
    config = {name: getattr(backbone, name) for name in names}
    config["scale_op"] = scale_op
    for name, component in (("qkv", "attn.qkv"), ("fc1", "mlp.fc1"), ("fc2", "mlp.fc2")):
        config[f"sattn_{name}_quant_dim"] = [
            block.get_submodule(component).w_q.dim for block in backbone.blocks
        ]
    return config


def build_exported_backbone(
    *, scale_op: str = "median", initialize: bool = True, **kwargs: Any
) -> nn.Module:
    """Reconstruct an ordinary export from its saved backbone_config."""
    model = DinoVisionTransformer(**kwargs)
    _fix_masked_projections(model)
    _set_scale_op(model, scale_op)
    if initialize:
        model.init_weights()
    return model
