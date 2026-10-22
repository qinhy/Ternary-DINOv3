from __future__ import annotations

import copy

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ternary_dinov3.checkpoints import freeze_ternary_weights, load_checkpoint
from ternary_dinov3.data import collate_image_groups
from ternary_dinov3.distillation_losses import TeacherCLSPCA
from ternary_dinov3.matryoshka import (
    build_exported_backbone,
    build_matryoshka_backbone,
    export_matryoshka_subnetwork,
    exported_backbone_config,
    initialize_from_teacher,
    run_matryoshka_distillation_epoch,
)


class TokenModel(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(3, width)
        self.splits = {"full": (width, 2), "half": (width // 2, 1)}
        self.calls = 0

    def forward_features(self, images, *, width=None, depth=None):
        self.calls += 1
        patches = self.proj(images.flatten(2).transpose(1, 2))
        if width is not None:
            patches = patches[..., :width] / depth
        return {"x_norm_clstoken": patches.mean(1), "x_norm_patchtokens": patches}


def fixed_pca(width=8):
    generator = torch.Generator().manual_seed(29)
    basis, _ = torch.linalg.qr(torch.randn(width, width, generator=generator))
    return TeacherCLSPCA(
        torch.linspace(-0.2, 0.2, width), basis, torch.arange(width, 0, -1).float(), 100
    )


def full_patch_kl(student, teacher, temperature=0.2):
    # Independent, unchunked reference matching the requested KL reduction.
    teacher = F.normalize(teacher.detach().float(), dim=-1)
    student = F.normalize(student.float(), dim=-1)
    teacher_sim = teacher @ teacher.transpose(-1, -2)
    student_sim = student @ student.transpose(-1, -2)
    diag = torch.eye(student.shape[1], dtype=torch.bool)[None]
    return (
        F.kl_div(
            F.log_softmax(student_sim.masked_fill(diag, -1e9) / temperature, dim=-1),
            F.softmax(teacher_sim.masked_fill(diag, -1e9) / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


def test_mixed_resolution_loss_and_gradient_match_per_image_reference():
    torch.manual_seed(7)
    student, teacher, pca = TokenModel(), TokenModel(), fixed_pca()
    samples = [(torch.randn(3, h, w), 0) for h, w in [(2, 3), (3, 2), (2, 3)]]
    loader = DataLoader(samples, batch_size=3, collate_fn=collate_image_groups)
    expected, expected_cls, expected_patch = {}, {}, {}
    for name, (width, depth) in student.splits.items():
        cls_losses, patch_losses = [], []
        for image, _ in samples:
            output = student.forward_features(image[None], width=width, depth=depth)
            target = teacher.forward_features(image[None])
            cls_losses.append(
                F.mse_loss(output["x_norm_clstoken"], pca(target["x_norm_clstoken"], width))
            )
            patch_losses.append(
                full_patch_kl(output["x_norm_patchtokens"], target["x_norm_patchtokens"])
            )
        expected_cls[name] = sum(cls_losses) / len(samples)
        expected_patch[name] = sum(patch_losses) / len(samples)
        expected[name] = 0.7 * expected_cls[name] + 1.3 * expected_patch[name]
    (sum(expected.values()) / len(expected)).backward()
    gradient = student.proj.weight.grad.clone()
    student.zero_grad(set_to_none=True)
    before = student.proj.weight.detach().clone()
    teacher_before = copy.deepcopy(teacher.state_dict())
    teacher.calls = 0
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    result = run_matryoshka_distillation_epoch(
        student,
        teacher,
        loader,
        torch.device("cpu"),
        cls_pca=pca,
        optimizer=optimizer,
        cls_loss_weight=0.7,
        patch_loss_weight=1.3,
        patch_temperature=0.2,
        patch_query_chunk_size=2,
        log_interval=0,
    )
    assert result["loss"] == pytest.approx(sum(v.item() for v in expected.values()) / 2)
    for name, value in expected.items():
        assert result["subnet_losses"][name] == pytest.approx(value.item())
        assert result["subnet_cls_losses"][name] == pytest.approx(expected_cls[name].item())
        assert result["subnet_patch_losses"][name] == pytest.approx(expected_patch[name].item())
    torch.testing.assert_close(student.proj.weight, before - 0.1 * gradient)
    assert teacher.calls == 2
    assert not teacher.training
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert all(p.grad is None for p in pca.parameters())
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key])


def test_evaluation_and_subnet_selection():
    student, teacher, pca = TokenModel(), TokenModel(), fixed_pca()
    images = torch.randn(3, 3, 2, 2)
    loader = DataLoader(TensorDataset(images, torch.zeros(3)), batch_size=2)
    before = copy.deepcopy(student.state_dict())
    result = run_matryoshka_distillation_epoch(
        student,
        teacher,
        loader,
        torch.device("cpu"),
        cls_pca=pca,
        subnets=["half"],
        log_interval=0,
    )
    assert set(result["subnet_losses"]) == {"half"}
    output = student.forward_features(images, width=4, depth=1)
    target = teacher.forward_features(images)
    expected = F.mse_loss(output["x_norm_clstoken"], pca(target["x_norm_clstoken"], 4))
    expected = expected + full_patch_kl(
        output["x_norm_patchtokens"], target["x_norm_patchtokens"], 0.1
    )
    assert result["loss"] == pytest.approx(expected.item())
    assert not student.training
    assert all(p.grad is None for p in student.parameters())
    for key, value in student.state_dict().items():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize("subnets", [[], ["missing"], ["full", "full"]])
def test_invalid_subnet_selection(subnets):
    loader = DataLoader(TensorDataset(torch.randn(1, 3, 2, 2), torch.zeros(1)))
    with pytest.raises(ValueError):
        run_matryoshka_distillation_epoch(
            TokenModel(),
            TokenModel(),
            loader,
            torch.device("cpu"),
            cls_pca=fixed_pca(),
            subnets=subnets,
        )


def test_empty_loader_and_teacher_width():
    empty = DataLoader(TensorDataset(torch.empty(0, 3, 2, 2), torch.empty(0)))
    with pytest.raises(ValueError, match="no examples"):
        run_matryoshka_distillation_epoch(
            TokenModel(), TokenModel(), empty, torch.device("cpu"), cls_pca=fixed_pca()
        )
    loader = DataLoader(TensorDataset(torch.randn(1, 3, 2, 2), torch.zeros(1)))
    with pytest.raises(ValueError, match="PCA input dimension"):
        run_matryoshka_distillation_epoch(
            TokenModel(),
            TokenModel(4),
            loader,
            torch.device("cpu"),
            cls_pca=fixed_pca(),
            log_interval=0,
        )


def test_teacher_initialization_slices_groups_and_flattens_patch_projection():
    teacher = build_exported_backbone(
        img_size=16,
        patch_size=8,
        embed_dim=128,
        depth=6,
        num_heads=2,
        ffn_ratio=4.0,
        mask_k_bias=True,
        n_storage_tokens=1,
        pos_embed_rope_dtype="fp32",
    )
    student = build_matryoshka_backbone(
        "vit_matryoshka_pico",
        image_size=16,
        patch_size=8,
        no_conv=True,
        mask_k_bias=True,
        n_storage_tokens=1,
        pos_embed_rope_dtype="fp32",
    )
    initialize_from_teacher(student, teacher)
    torch.testing.assert_close(
        student.patch_embed.proj.weight, teacher.patch_embed.proj.weight[:64].flatten(1)
    )
    expected = teacher.blocks[0].attn.qkv.weight.reshape(3, 128, 128)[:, :64, :64]
    torch.testing.assert_close(student.blocks[0].attn.qkv.weight, expected.reshape(192, 64))
    assert set(student.state_dict()) == set(teacher.state_dict())


@pytest.mark.parametrize("scale_op", ["mean", "median"])
def test_export_reloads_and_frozen_output_matches(scale_op):
    student = build_matryoshka_backbone(
        "vit_matryoshka_pico",
        image_size=16,
        patch_size=8,
        no_conv=True,
        mask_k_bias=True,
        n_storage_tokens=1,
        layerscale_init=0.5,
        pos_embed_rope_dtype="fp32",
        norm_layer="layernormbf16",
        scale_op=scale_op,
    ).eval()
    for block in student.blocks:
        with torch.no_grad():
            block.attn.qkv.bias.fill_(0.2)
    images = torch.randn(2, 3, 16, 16)
    for name, (width, depth) in student.splits.items():
        compact = export_matryoshka_subnetwork(student, name, scale_op=scale_op).eval()
        config = exported_backbone_config(compact, scale_op=scale_op)
        rebuilt = build_exported_backbone(**config).eval()
        rebuilt.load_state_dict(compact.state_dict(), strict=True)
        frozen = freeze_ternary_weights(compact)
        with torch.no_grad():
            expected = student(images, width=width, depth=depth)
            torch.testing.assert_close(compact(images), expected, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(rebuilt(images), expected, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(frozen(images), expected, rtol=1e-4, atol=1e-4)


def test_example_trains_resumes_and_writes_reloadable_exports(tmp_path, monkeypatch, capsys):
    from examples.distill_dinov3_matryoshka import TRAINING_FORMAT, main

    teacher = build_exported_backbone(
        img_size=16,
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
    teacher_before = copy.deepcopy(teacher.state_dict())
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: copy.deepcopy(teacher))
    repo = tmp_path / "teacher"
    repo.mkdir()
    (repo / "hubconf.py").touch()
    weights = tmp_path / "teacher.pt"
    weights.touch()
    for name in ("train", "val"):
        folder = tmp_path / name
        folder.mkdir()
        for index in range(2):
            Image.new("RGB", (16, 16), (50 + index * 40, 100, 150)).save(folder / f"{index}.png")
    save = tmp_path / "shared.pt"
    args = [
        "--teacher-repo",
        str(repo),
        "--teacher-weights",
        str(weights),
        "--train-dir",
        str(tmp_path / "train"),
        "--val-dir",
        str(tmp_path / "val"),
        "--architecture",
        "vit_matryoshka_pico",
        "--image-size",
        "16",
        "--workers",
        "0",
        "--batch-size",
        "2",
        "--log-interval",
        "0",
        "--device",
        "cpu",
        "--save",
        str(save),
    ]
    main([*args, "--epochs", "1"])
    checkpoint = load_checkpoint(save)
    assert checkpoint["format"] == TRAINING_FORMAT
    assert checkpoint["epoch"] == 1
    assert checkpoint["splits"] == {"pico": (64, 6), "femto": (64, 3)}
    assert (tmp_path / "shared-best.pt").is_file()
    for name in ("pico", "femto"):
        for suffix in ("", "-ternary"):
            payload = load_checkpoint(tmp_path / f"shared-best-{name}{suffix}.pt")
            restored = build_exported_backbone(**payload["backbone_config"])
            if suffix:
                restored = freeze_ternary_weights(restored)
            restored.load_state_dict(payload["state_dict"], strict=True)
            with torch.no_grad():
                assert torch.isfinite(restored.eval()(torch.randn(1, 3, 16, 16))).all()
    pca_before = copy.deepcopy(checkpoint["cls_pca"])
    import examples.distill_dinov3_matryoshka as example

    original_fit = example.fit_teacher_cls_pca

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("resume must reuse the fixed PCA basis")

    monkeypatch.setattr(example, "fit_teacher_cls_pca", forbidden_fit)
    main([*args, "--epochs", "2", "--resume", str(save)])
    checkpoint = load_checkpoint(save)
    assert checkpoint["epoch"] == 2
    assert checkpoint["scheduler_state_dict"]["T_max"] == 2
    for key, value in checkpoint["cls_pca"].items():
        torch.testing.assert_close(value, pca_before[key])
    assert checkpoint["loss_config"]["cls_target"] == "centered_teacher_pca"
    assert "val_patch_loss" in checkpoint["metrics"]
    # Existing MSE checkpoints retain weights/optimizer but recalibrate PCA and best loss.
    checkpoint["format"] = example.LEGACY_TRAINING_FORMAT
    checkpoint.pop("cls_pca")
    checkpoint.pop("loss_config")
    checkpoint["best_loss"] = -1.0
    torch.save(checkpoint, save)
    monkeypatch.setattr(example, "fit_teacher_cls_pca", original_fit)
    main([*args, "--epochs", "3", "--resume", str(save)])
    migrated = load_checkpoint(save)
    assert migrated["epoch"] == 3 and migrated["format"] == TRAINING_FORMAT
    assert migrated["best_loss"] >= 0
    assert "objective changed" in capsys.readouterr().out
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key])
