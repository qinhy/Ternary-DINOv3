from pathlib import Path

import torch
from ternarylayers import Bit

from ternary_dinov3.checkpoints import freeze_ternary_weights, load_pretrained_backbone
from ternary_dinov3.model import TernaryDINOv3Classifier, build_backbone


def test_load_prefixed_backbone_state() -> None:
    source = build_backbone("vit_femto", image_size=32, patch_size=8)
    target = build_backbone("vit_femto", image_size=32, patch_size=8)
    payload = {"teacher": {f"backbone.{key}": value for key, value in source.state_dict().items()}}
    incompatible = load_pretrained_backbone(target, payload)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.equal(target.cls_token, source.cls_token)


def test_frozen_model_uses_inference_layers(tmp_path: Path) -> None:
    model = TernaryDINOv3Classifier(
        "vit_femto", num_classes=3, image_size=32, patch_size=8
    ).eval()
    frozen = freeze_ternary_weights(model)
    assert not any(isinstance(module, (Bit.Linear, Bit.Conv2d)) for module in frozen.modules())
    assert any(isinstance(module, Bit.LinearInfer) for module in frozen.modules())
    with torch.inference_mode():
        output = frozen(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 3)

