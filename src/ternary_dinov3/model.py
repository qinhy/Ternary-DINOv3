"""Ternary DINOv3 backbone and classification model factories."""

from __future__ import annotations

from typing import Any, Literal

import torch
from ternarylayers import Bit
from ternarylayers.dinov3.models import vision_transformer as vits
from torch import Tensor, nn

Architecture = Literal[
    "vit_femto",
    "vit_pico",
    "vit_micro",
    "vit_nano",
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_so400m",
    "vit_huge2",
    "vit_giant2",
    "vit_7b",
]

ARCHITECTURES: tuple[str, ...] = (
    "vit_femto",
    "vit_pico",
    "vit_micro",
    "vit_nano",
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_so400m",
    "vit_huge2",
    "vit_giant2",
    "vit_7b",
)


def _set_scale_op(module: nn.Module, scale_op: str) -> None:
    if scale_op not in {"mean", "median"}:
        raise ValueError("scale_op must be 'mean' or 'median'")
    for child in module.modules():
        if isinstance(child, (Bit.Linear, Bit.Conv2d)):
            child.scale_op = scale_op
            child.w_q.scale_op = scale_op


def build_backbone(
    architecture: Architecture | str = "vit_small",
    *,
    image_size: int = 224,
    patch_size: int = 16,
    scale_op: str = "median",
    initialize: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Build a DINOv3-compatible vision transformer with ternary projections."""
    if architecture not in ARCHITECTURES:
        choices = ", ".join(ARCHITECTURES)
        raise ValueError(f"unknown architecture {architecture!r}; choose one of: {choices}")
    factory = getattr(vits, architecture)
    backbone = factory(img_size=image_size, patch_size=patch_size, **kwargs)
    _set_scale_op(backbone, scale_op)
    if initialize:
        backbone.init_weights()
    return backbone


class TernaryDINOv3Classifier(nn.Module):
    """A ternary DINOv3 backbone with a trainable ternary classification head."""

    def __init__(
        self,
        architecture: Architecture | str = "vit_small",
        *,
        num_classes: int = 1000,
        image_size: int = 224,
        patch_size: int = 16,
        scale_op: str = "median",
        dropout: float = 0.0,
        initialize: bool = True,
        backbone_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if image_size <= 0 or patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        self.init_config = {
            "architecture": architecture,
            "num_classes": num_classes,
            "image_size": image_size,
            "patch_size": patch_size,
            "scale_op": scale_op,
            "dropout": dropout,
            "initialize": False,
            "backbone_kwargs": dict(backbone_kwargs or {}),
        }
        self.backbone = build_backbone(
            architecture,
            image_size=image_size,
            patch_size=patch_size,
            scale_op=scale_op,
            initialize=initialize,
            **(backbone_kwargs or {}),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = Bit.Linear(self.backbone.embed_dim, num_classes, scale_op=scale_op)
        if initialize:
            torch.nn.init.trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward_features(self, images: Tensor) -> Tensor:
        """Return normalized DINOv3 class-token embeddings."""
        return self.backbone.forward_features(images)["x_norm_clstoken"]

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.dropout(self.forward_features(images)))

    def clone(self) -> TernaryDINOv3Classifier:
        clone = type(self)(**self.init_config)
        clone.load_state_dict(self.state_dict())
        return clone
