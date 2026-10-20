from pathlib import Path

import pytest
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
    model = TernaryDINOv3Classifier("vit_femto", num_classes=3, image_size=32, patch_size=8).eval()
    frozen = freeze_ternary_weights(model)
    assert not any(isinstance(module, (Bit.Linear, Bit.Conv2d)) for module in frozen.modules())
    assert any(isinstance(module, Bit.LinearInfer) for module in frozen.modules())
    with torch.inference_mode():
        output = frozen(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 3)


@pytest.mark.parametrize("scale_op", ["mean", "median"])
def test_masked_backbone_qat_matches_frozen_roundtrip(tmp_path: Path, scale_op: str) -> None:
    model = build_backbone(
        "vit_femto",
        image_size=32,
        patch_size=8,
        mask_k_bias=True,
        n_storage_tokens=4,
        scale_op=scale_op,
    ).eval()
    projection = model.blocks[0].attn.qkv
    with torch.no_grad():
        projection.bias.fill_(0.3)  # Export must apply the mask, not just copy the bias.
    images = torch.randn(2, 3, 32, 32)
    features = torch.randn(2, 5, projection.in_features)
    expected = torch.nn.functional.linear(
        features, projection.w_q(projection.weight), projection.bias * projection.bias_mask
    )
    torch.testing.assert_close(projection(features), expected)
    frozen = freeze_ternary_weights(model)
    path = tmp_path / "backbone.pt"
    torch.save(frozen.state_dict(), path)
    frozen.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(model(images), frozen(images), rtol=1e-4, atol=1e-4)
