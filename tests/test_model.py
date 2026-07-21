import pytest
import torch
from ternarylayers import Bit

from ternary_dinov3.model import TernaryDINOv3Classifier, build_backbone


def test_femto_classifier_forward_and_features() -> None:
    model = TernaryDINOv3Classifier(
        "vit_femto", num_classes=5, image_size=32, patch_size=8
    )
    images = torch.randn(2, 3, 32, 32)
    assert model(images).shape == (2, 5)
    assert model.forward_features(images).shape == (2, 64)
    assert any(isinstance(module, Bit.Linear) for module in model.modules())
    assert any(isinstance(module, Bit.Conv2d) for module in model.modules())


def test_backbone_supports_non_square_multiple_of_patch() -> None:
    model = build_backbone("vit_femto", image_size=32, patch_size=8)
    features = model.forward_features(torch.randn(1, 3, 32, 48))
    assert features["x_norm_patchtokens"].shape == (1, 24, 64)


def test_invalid_architecture() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        build_backbone("not-a-vit")

