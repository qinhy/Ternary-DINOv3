"""Fixed teacher CLS PCA targets and dimension-independent patch relations."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader


class TeacherCLSPCA(nn.Module):
    """A fixed centered PCA transform, ordered by decreasing training variance.

    Leading columns provide nested targets for smaller student widths. These
    are PCA scores, without whitening or per-batch normalization.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        explained_variance: torch.Tensor,
        sample_count: int,
    ) -> None:
        super().__init__()
        if mean.ndim != 1 or components.ndim != 2 or components.shape[0] != mean.numel():
            raise ValueError("PCA mean/components must have shapes [D] and [D, K]")
        if explained_variance.shape != (components.shape[1],) or components.shape[1] < 1:
            raise ValueError("PCA must contain at least one component and its variance")
        if not all(torch.isfinite(t).all() for t in (mean, components, explained_variance)):
            raise ValueError("PCA statistics must be finite")
        self.register_buffer("mean", mean.detach().float().clone())
        self.register_buffer("components", components.detach().float().clone())
        self.register_buffer("explained_variance", explained_variance.detach().float().clone())
        self.register_buffer("sample_count", torch.tensor(sample_count, dtype=torch.int64))

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> TeacherCLSPCA:
        return cls(
            state["mean"],
            state["components"],
            state["explained_variance"],
            int(state["sample_count"].item()),
        )

    def forward(self, features: torch.Tensor, width: int) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.mean.numel():
            raise ValueError("teacher CLS tokens must be [B, PCA input dimension]")
        if not 1 <= width <= self.components.shape[1]:
            raise ValueError("student width exceeds the fitted PCA output dimension")
        return (features.detach().float() - self.mean) @ self.components[:, :width]


@torch.no_grad()
def fit_teacher_cls_pca(
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_samples: int = 8192,
    output_dim: int | None = None,
) -> TeacherCLSPCA:
    """Fit once from calibration images, accumulating CPU float64 covariance.

    Memory depends on teacher width, not the number of images. The caller must
    provide training images with deterministic preprocessing, not validation
    data. At least two images are needed; diverse samples well above the output
    dimension are recommended to estimate its leading components reliably.
    """
    if max_samples < 2:
        raise ValueError("PCA calibration requires max_samples >= 2")
    teacher.requires_grad_(False).eval()
    count, mean, scatter = 0, None, None
    for images, _ in loader:
        groups = [images] if isinstance(images, torch.Tensor) else images
        for group in groups:
            group = group[: max_samples - count].to(device, non_blocking=True)
            values = teacher.forward_features(group)["x_norm_clstoken"].detach().double().cpu()
            if values.ndim != 2 or not torch.isfinite(values).all():
                raise ValueError("PCA needs finite teacher CLS tokens of shape [B, D]")
            size = values.shape[0]
            batch_mean = values.mean(0)
            centered = values - batch_mean
            batch_scatter = centered.T @ centered
            if count == 0:
                mean, scatter = batch_mean, batch_scatter
            else:
                delta = batch_mean - mean
                scatter += batch_scatter + torch.outer(delta, delta) * (
                    count * size / (count + size)
                )
                mean += delta * (size / (count + size))
            count += size
            if count >= max_samples:
                break
        if count >= max_samples:
            break
    if count < 2:
        raise ValueError("PCA calibration requires at least two training images")
    width = mean.numel()
    output_dim = width if output_dim is None else output_dim
    if not 1 <= output_dim <= width:
        raise ValueError("PCA output_dim must be between 1 and teacher width")
    eigenvalues, eigenvectors = torch.linalg.eigh(scatter / (count - 1))
    components = eigenvectors[:, -output_dim:].flip(1)
    # Fix the arbitrary eigenvector sign for reproducible saved coordinate axes.
    rows = components.abs().argmax(0)
    signs = components[rows, torch.arange(output_dim)].sign()
    components *= signs
    variance = eigenvalues[-output_dim:].flip(0).clamp_min(0)
    return TeacherCLSPCA(mean, components, variance, count)


def cosine_sim_matrix(tokens: torch.Tensor) -> torch.Tensor:
    """Cosine relations across tokens, using all feature channels in float32."""
    normalized = F.normalize(tokens.float(), dim=-1)
    return normalized @ normalized.transpose(-1, -2)


def relational_kd_kl(
    z_s: torch.Tensor,
    z_t: torch.Tensor | None = None,
    K_t: torch.Tensor | None = None,
    T: float = 0.1,
    drop_cls: bool = True,
    mask_diag: bool = True,
    *,
    query_chunk_size: int | None = None,
) -> torch.Tensor:
    """KL(teacher patch relations || student relations), with batchmean * T**2.

    Matches the full BxNxN KL, summing query/key entries and dividing by B.
    Chunking keeps all keys for each query, preserving that exact reduction.
    Training checkpoints each query chunk to bound saved similarity activations.
    Set drop_cls=False for x_norm_patchtokens, which already excludes CLS.
    """
    if not math.isfinite(T) or T <= 0:
        raise ValueError("T must be positive and finite")
    if query_chunk_size is not None and query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive or None")
    if z_s.ndim != 3 or z_s.shape[0] < 1:
        raise ValueError("student tokens must have shape [B, N, D] with B > 0")
    batch, count, _ = z_s.shape
    if K_t is None:
        if z_t is None or z_t.ndim != 3 or z_t.shape[:2] != z_s.shape[:2]:
            raise ValueError("teacher and student must have matching batch/token counts")
        teacher_tokens = z_t.detach()
        if drop_cls:
            teacher_tokens = teacher_tokens[:, 1:]
        teacher_tokens = F.normalize(teacher_tokens.float(), dim=-1)
    else:
        if K_t.shape != (batch, count, count):
            raise ValueError("K_t must have shape [B, N, N] matching the student tokens")
        K_t = K_t.detach().float()
        if drop_cls:
            K_t = K_t[:, 1:, 1:]
        teacher_tokens = None
    if drop_cls:
        z_s = z_s[:, 1:]
    count = z_s.shape[1]
    # With no alternative keys there are no spatial relations to supervise.
    if count <= 1:
        return z_s.float().sum() * 0.0
    student_tokens = F.normalize(z_s.float(), dim=-1)
    chunk_size = count if query_chunk_size is None else min(query_chunk_size, count)

    def chunk_loss(student_all, teacher_all, teacher_matrix, start, stop):
        student_logits = student_all[:, start:stop] @ student_all.transpose(-1, -2)
        with torch.no_grad():
            teacher_logits = (
                teacher_all[:, start:stop] @ teacher_all.transpose(-1, -2)
                if teacher_matrix is None
                else teacher_matrix[:, start:stop]
            )
        if mask_diag:
            diag = (
                torch.arange(start, stop, device=z_s.device)[:, None]
                == torch.arange(count, device=z_s.device)[None, :]
            )
            student_logits = student_logits.masked_fill(diag[None], -1e9)
            teacher_logits = teacher_logits.masked_fill(diag[None], -1e9)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)

    loss = student_tokens.new_zeros(())
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        if torch.is_grad_enabled() and student_tokens.requires_grad and chunk_size < count:
            loss = loss + checkpoint(
                chunk_loss, student_tokens, teacher_tokens, K_t, start, stop, use_reentrant=False
            )
        else:
            loss = loss + chunk_loss(student_tokens, teacher_tokens, K_t, start, stop)
    return loss
