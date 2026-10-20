"""Labeled, unlabeled, and deterministic synthetic image datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader


class SyntheticClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, count: int, image_size: int, num_classes: int, seed: int = 7) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(count, 3, image_size, image_size, generator=generator)
        self.targets = torch.arange(count) % num_classes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.targets[index])


def image_transform(image_size: int | None, *, training: bool) -> transforms.Compose:
    operations: list[object] = []
    if image_size is not None:
        operations.append(transforms.Resize((image_size, image_size)))
    if training:
        operations.append(transforms.RandomHorizontalFlip())
        if image_size is not None:
            operations.append(transforms.RandAugment())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)


def imagefolder(path: Path, image_size: int, *, training: bool) -> datasets.ImageFolder:
    if not path.is_dir():
        raise FileNotFoundError(f"ImageFolder directory does not exist: {path}")
    return datasets.ImageFolder(path, transform=image_transform(image_size, training=training))


class UnlabeledImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Read images recursively, including flat COCO-style directories.

    Return a dummy label to share the existing distillation batch interface.
    No class folders or annotation files are required.
    """

    def __init__(
        self, path: Path, image_size: int | None = None, *, training: bool, patch_size: int = 16
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.image_size = image_size
        self.patch_size = patch_size
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {path}")
        self.samples = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.samples:
            raise FileNotFoundError(f"No supported image files found in: {path}")
        self.transform = image_transform(image_size, training=training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = default_loader(str(self.samples[index]))
        image = self.transform(image)
        if self.image_size is None:
            height, width = image.shape[-2:]
            # Zero after normalization represents the ImageNet mean color.
            image = F.pad(image, (0, -width % self.patch_size, 0, -height % self.patch_size))
        return image, 0


def collate_image_groups(
    samples: list[tuple[torch.Tensor, int]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Stack equal shapes without adding any extra batch padding."""
    groups: dict[tuple[int, ...], list[torch.Tensor]] = {}
    for image, _ in samples:
        groups.setdefault(tuple(image.shape), []).append(image)
    return [torch.stack(images) for images in groups.values()], torch.tensor(
        [label for _, label in samples]
    )


def create_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    training: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training and len(dataset) >= batch_size,
        collate_fn=(
            collate_image_groups
            if isinstance(dataset, UnlabeledImageDataset) and dataset.image_size is None
            else None
        ),
    )
