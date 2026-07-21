"""ImageFolder and deterministic synthetic datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


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

