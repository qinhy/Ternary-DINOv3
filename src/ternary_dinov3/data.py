"""Labeled, unlabeled, and deterministic synthetic image datasets."""

from __future__ import annotations

from pathlib import Path

import torch
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


def image_transform(image_size: int, *, training: bool) -> transforms.Compose:
    operations: list[object] = [transforms.Resize((image_size, image_size))]
    if training:
        operations.extend([transforms.RandomHorizontalFlip(), transforms.RandAugment()])
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

    def __init__(self, path: Path, image_size: int, *, training: bool) -> None:
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
        return self.transform(image), 0


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
    )
