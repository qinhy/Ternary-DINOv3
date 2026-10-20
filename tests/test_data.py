from pathlib import Path

import pytest
import torch
from PIL import Image

from ternary_dinov3.data import UnlabeledImageDataset, create_loader


def test_unlabeled_flat_coco_directory(tmp_path: Path) -> None:
    Image.new("RGB", (20, 30), "red").save(tmp_path / "000001.jpg")
    Image.new("L", (30, 20), 128).save(tmp_path / "000002.JPG")
    (tmp_path / "annotations.json").write_text("{}")
    dataset = UnlabeledImageDataset(tmp_path, 32, training=False)
    loader = create_loader(dataset, batch_size=2, workers=0, training=False)
    images, labels = next(iter(loader))
    assert len(dataset) == 2
    assert images.shape == (2, 3, 32, 32)
    assert torch.isfinite(images).all()
    assert labels.tolist() == [0, 0]
    # RGB conversion and the same ImageNet normalization used for teacher inputs.
    expected = torch.tensor([128 / 255] * 3) - torch.tensor([0.485, 0.456, 0.406])
    expected /= torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(images[1, :, 0, 0], expected)


def test_unlabeled_nested_and_root_images(tmp_path: Path) -> None:
    folder = tmp_path / "nested" / "images"
    folder.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(tmp_path / "root.png")
    Image.new("RGB", (32, 32)).save(folder / "nested.png")
    dataset = UnlabeledImageDataset(tmp_path, 32, training=True)
    assert len(dataset) == 2
    assert all(dataset[index][0].shape == (3, 32, 32) for index in range(2))


def test_unlabeled_missing_or_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        UnlabeledImageDataset(tmp_path / "missing", 32, training=False)
    with pytest.raises(FileNotFoundError, match="No supported image"):
        UnlabeledImageDataset(tmp_path, 32, training=False)
