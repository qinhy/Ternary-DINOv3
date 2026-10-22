from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ternary_dinov3.distillation_losses import (
    TeacherCLSPCA,
    cosine_sim_matrix,
    fit_teacher_cls_pca,
    relational_kd_kl,
)


class CLSTeacher(nn.Module):
    def forward_features(self, values):
        return {"x_norm_clstoken": values}


def test_streaming_pca_matches_svd_and_nested_scores():
    generator = torch.Generator().manual_seed(9)
    rotation, _ = torch.linalg.qr(torch.randn(5, 5, generator=generator))
    samples = (
        torch.randn(200, 5, generator=generator) * torch.tensor([8.0, 4.0, 2.0, 1.0, 0.2])
    ) @ rotation
    samples = samples + torch.arange(5)
    dataset = TensorDataset(samples, torch.zeros(len(samples)))
    pca = fit_teacher_cls_pca(
        CLSTeacher(),
        DataLoader(dataset, batch_size=13),
        torch.device("cpu"),
        max_samples=137,
        output_dim=4,
    )
    subset = samples[:137].double()
    mean = subset.mean(0)
    _, singular, axes = torch.linalg.svd(subset - mean, full_matrices=False)
    torch.testing.assert_close(pca.mean, mean.float())
    torch.testing.assert_close(pca.explained_variance, (singular[:4] ** 2 / 136).float())
    expected_space = axes[:4].T @ axes[:4]
    torch.testing.assert_close(pca.components @ pca.components.T, expected_space.float())
    assert pca.sample_count == 137
    assert torch.all(pca.explained_variance[:-1] >= pca.explained_variance[1:])
    torch.testing.assert_close(pca(samples, 2), pca(samples, 4)[:, :2])
    torch.testing.assert_close(pca(samples, 2), (samples - pca.mean) @ pca.components[:, :2])
    restored = TeacherCLSPCA.from_state_dict(pca.state_dict())
    torch.testing.assert_close(restored(samples, 4), pca(samples, 4))
    alternative = fit_teacher_cls_pca(
        CLSTeacher(),
        DataLoader(dataset, batch_size=31),
        torch.device("cpu"),
        max_samples=137,
        output_dim=4,
    )
    torch.testing.assert_close(alternative.components, pca.components)


def test_pca_requires_calibration_data_and_valid_width():
    data = TensorDataset(torch.randn(1, 4), torch.zeros(1))
    with pytest.raises(ValueError, match="at least two"):
        fit_teacher_cls_pca(CLSTeacher(), DataLoader(data), torch.device("cpu"))
    data = TensorDataset(torch.randn(3, 4), torch.zeros(3))
    with pytest.raises(ValueError, match="output_dim"):
        fit_teacher_cls_pca(CLSTeacher(), DataLoader(data), torch.device("cpu"), output_dim=5)
    pca = TeacherCLSPCA(torch.zeros(4), torch.eye(4)[:, :2], torch.ones(2), 3)
    with pytest.raises(ValueError, match="output dimension"):
        pca(torch.randn(1, 4), 3)


@pytest.mark.parametrize("drop_cls", [False, True])
@pytest.mark.parametrize("mask_diag", [False, True])
def test_relational_kl_matches_reference_and_chunked_gradients(drop_cls, mask_diag):
    torch.manual_seed(41)
    student = torch.randn(2, 7, 4, requires_grad=True)
    teacher = torch.randn(2, 7, 9, requires_grad=True)
    temperature = 0.17
    expected_student = student[:, 1:] if drop_cls else student
    expected_teacher = teacher[:, 1:] if drop_cls else teacher
    ks, kt = cosine_sim_matrix(expected_student), cosine_sim_matrix(expected_teacher).detach()
    if mask_diag:
        diag = torch.eye(ks.shape[-1], dtype=torch.bool)[None]
        ks, kt = ks.masked_fill(diag, -1e9), kt.masked_fill(diag, -1e9)
    reference = (
        F.kl_div(
            F.log_softmax(ks / temperature, dim=-1),
            F.softmax(kt / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )
    (gradient,) = torch.autograd.grad(reference, student)
    actual = relational_kd_kl(
        student, teacher, T=temperature, drop_cls=drop_cls, mask_diag=mask_diag, query_chunk_size=2
    )
    actual.backward()
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(student.grad, gradient)
    assert teacher.grad is None
    cached = relational_kd_kl(
        student.detach(),
        K_t=cosine_sim_matrix(teacher),
        T=temperature,
        drop_cls=drop_cls,
        mask_diag=mask_diag,
        query_chunk_size=3,
    )
    torch.testing.assert_close(cached, reference)


def test_patch_relations_are_invariant_to_teacher_channel_order_and_dimension():
    torch.manual_seed(23)
    teacher, student = torch.randn(2, 5, 11), torch.randn(2, 5, 3)
    expected = relational_kd_kl(student, teacher, drop_cls=False)
    rotation, _ = torch.linalg.qr(torch.randn(11, 11))
    actual = relational_kd_kl(student, teacher @ rotation, drop_cls=False)
    torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("count,drop_cls", [(1, False), (1, True), (0, False)])
def test_degenerate_patch_counts_have_finite_zero_loss_and_gradient(count, drop_cls):
    student = torch.randn(2, count, 3, requires_grad=True)
    teacher = torch.randn(2, count, 5)
    loss = relational_kd_kl(student, teacher, drop_cls=drop_cls)
    loss.backward()
    assert loss.item() == 0
    assert torch.isfinite(student.grad).all()


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_relational_temperature(temperature):
    with pytest.raises(ValueError, match="T must"):
        relational_kd_kl(torch.randn(1, 3, 4), torch.randn(1, 3, 8), T=temperature)
