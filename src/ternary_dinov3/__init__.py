"""Public API for Ternary-DINOv3."""

from .checkpoints import freeze_ternary_weights, load_pretrained_backbone
from .distillation import run_distillation_epoch
from .model import ARCHITECTURES, TernaryDINOv3Classifier, build_backbone

__all__ = [
    "ARCHITECTURES",
    "TernaryDINOv3Classifier",
    "build_backbone",
    "freeze_ternary_weights",
    "load_pretrained_backbone",
    "run_distillation_epoch",
]

__version__ = "0.1.0"
