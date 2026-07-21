"""Classify one image with a Ternary-DINOv3 project checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .checkpoints import TERNARY_FORMAT, load_checkpoint
from .data import image_transform
from .engine import select_device
from .model import TernaryDINOv3Classifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    payload = load_checkpoint(args.checkpoint, device)
    config = dict(payload["model_config"])
    config["initialize"] = False
    model = TernaryDINOv3Classifier(**config)
    if payload.get("format") == TERNARY_FORMAT:
        from ternarylayers.bit import convert_to_ternary

        model = convert_to_ternary(model)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    image = Image.open(args.image).convert("RGB")
    tensor = image_transform(config["image_size"], training=False)(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    class_names = payload.get("class_names", [])
    count = min(args.top_k, probabilities.numel())
    values, indices = probabilities.topk(count)
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        label = class_names[index] if index < len(class_names) else str(index)
        print(f"{label}\t{value:.6f}")


if __name__ == "__main__":
    main()

