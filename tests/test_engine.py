import torch

from ternary_dinov3.data import SyntheticClassificationDataset, create_loader
from ternary_dinov3.engine import run_epoch
from ternary_dinov3.model import TernaryDINOv3Classifier


def test_training_smoke() -> None:
    model = TernaryDINOv3Classifier(
        "vit_femto", num_classes=4, image_size=32, patch_size=8
    )
    dataset = SyntheticClassificationDataset(2, 32, 4)
    loader = create_loader(dataset, batch_size=2, workers=0, training=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metrics = run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer)
    assert metrics["loss"] > 0
    assert 0 <= metrics["accuracy"] <= 1

