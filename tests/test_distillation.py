import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ternary_dinov3.distillation import run_distillation_epoch


def test_distillation_updates_only_student_and_reduces_error() -> None:
    torch.manual_seed(7)
    student = nn.Linear(4, 3)
    teacher = nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.9))
    images = torch.randn(16, 4)
    loader = DataLoader(TensorDataset(images, torch.zeros(16)), batch_size=16)
    device = torch.device("cpu")
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    student_before = student.weight.detach().clone()
    before = run_distillation_epoch(student, teacher, loader, device)["loss"]
    optimizer = torch.optim.SGD(student.parameters(), lr=0.2)
    for _ in range(10):
        teacher.train()  # The loop must reset a stochastic teacher to eval each time.
        run_distillation_epoch(student, teacher, loader, device, optimizer=optimizer)
    after = run_distillation_epoch(student, teacher, loader, device)["loss"]
    assert after < before
    assert not torch.equal(student.weight, student_before)
    assert not teacher.training
    assert all(param.grad is None and not param.requires_grad for param in teacher.parameters())
    assert all(
        torch.equal(value, teacher_before[key]) for key, value in teacher.state_dict().items()
    )


def test_distillation_rejects_broadcasting() -> None:
    loader = DataLoader(TensorDataset(torch.randn(2, 4), torch.zeros(2)), batch_size=2)
    with pytest.raises(ValueError, match="same shape"):
        run_distillation_epoch(nn.Linear(4, 3), nn.Linear(4, 1), loader, torch.device("cpu"))


def test_distillation_empty_loader() -> None:
    loader = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0)))
    with pytest.raises(ValueError, match="no examples"):
        run_distillation_epoch(nn.Linear(4, 3), nn.Linear(4, 3), loader, torch.device("cpu"))


def test_distillation_evaluation_leaves_student_unchanged() -> None:
    student, teacher = nn.Linear(4, 3), nn.Linear(4, 3)
    images = torch.randn(5, 4)
    loader = DataLoader(TensorDataset(images, torch.zeros(5)), batch_size=2)
    expected = torch.nn.functional.mse_loss(student(images), teacher(images)).item()
    before = student.weight.detach().clone()
    result = run_distillation_epoch(student, teacher, loader, torch.device("cpu"))
    assert result["loss"] == pytest.approx(expected)
    assert torch.equal(student.weight, before)
    assert not student.training
    assert all(param.grad is None for param in student.parameters())
