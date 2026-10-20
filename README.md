# Ternary-DINOv3

A standalone training and inference project for DINOv3-compatible vision
transformers whose patch embedding, attention projections, MLP projections,
and classification head use TernaryCNN's trainable 1.58-bit weights.

The project intentionally depends on
[TernaryCNN](https://github.com/qinhy/TernaryCNN) instead of copying its DINOv3
port. This keeps one implementation of the backbone and preserves its upstream
DINOv3 license boundary.

## What is included

- femto through 7B DINOv3 ViT factories;
- QAT-style ternary weights with per-output-channel mean or median scaling;
- a reusable backbone API and a ternary classification head;
- import of plain, prefixed, and teacher-style DINOv3 state dictionaries;
- ImageFolder fine-tuning with resumable AdamW/cosine checkpoints;
- best QAT and frozen int8 ternary checkpoint export;
- single-image top-k inference;
- CPU-friendly unit and end-to-end smoke tests.

Normalization, class/storage tokens, RoPE, activations, and biases remain in
floating point. The frozen checkpoint stores ternary kernels as int8 together
with one floating-point scale per output channel.

## Setup

For local development, place this repository beside TernaryCNN:

```text
parent/
├── TernaryCNN/
└── Ternary-DINOv3/
```

Then install and verify it:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

The editable sibling path under `[tool.uv.sources]` is a local convenience.
Remove that table in a standalone clone to use the Git dependency declared in
`project.dependencies`.

## Fast end-to-end check

This uses a 64-dimensional, three-block model and generated 32 px images. It
runs training, validation, checkpointing, and frozen ternary conversion without
downloading a dataset:

```bash
uv run train-ternary-dinov3 \
  --smoke-test \
  --device cpu \
  --save checkpoints/smoke.pt
```

It produces `smoke.pt` (resumable state), `smoke-best.pt` (QAT weights), and
`smoke-best-ternary.pt` (frozen inference weights).

## ImageFolder fine-tuning

Use the standard torchvision layout with matching class directories:

```text
data/my-dataset/
├── train/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

```bash
uv run train-ternary-dinov3 \
  --train-dir data/my-dataset/train \
  --val-dir data/my-dataset/val \
  --architecture vit_small \
  --image-size 224 \
  --patch-size 16 \
  --batch-size 64 \
  --epochs 100 \
  --learning-rate 1e-4 \
  --weight-decay 0.05 \
  --device cuda \
  --save checkpoints/vits16.pt
```

Resume by adding `--resume checkpoints/vits16.pt`. `--epochs` is the desired
final epoch, not the number of additional epochs.

## MNIST example

The [`examples/train_mnist.py`](examples/train_mnist.py) script trains
`vit_femto` on MNIST using 32×32 three-channel inputs and 8×8 patches:

```bash
uv run python examples/train_mnist.py --device cuda
```

For a quick CPU trial:

```bash
uv run python examples/train_mnist.py \
  --device cpu --epochs 1 --limit-train 512 --limit-test 128 --workers 0
```

## Starting from DINOv3 weights

Pass a local PyTorch checkpoint using `--pretrained`. The loader accepts a
plain backbone state dictionary as well as common `state_dict`, `model`,
`teacher`, or `student` containers and strips repeated wrapper prefixes.

```bash
uv run train-ternary-dinov3 \
  --train-dir data/my-dataset/train \
  --val-dir data/my-dataset/val \
  --architecture vit_small \
  --pretrained checkpoints/dinov3_vits16.pth \
  --device cuda
```

Architecture, patch size, storage-token count, FFN type, and structural
settings must match the source checkpoint. Full-precision weights initialize
the real-valued latent weights, after which every forward pass applies ternary
fake quantization. Accuracy should therefore be recovered with QAT or
distillation rather than assumed to match the source immediately.

For special configurations, use the Python API's `backbone_kwargs`, for
example `ffn_layer="swiglu"`, `n_storage_tokens=4`, or `norm_layer="rmsnorm"`.

## Inference

```bash
uv run infer-ternary-dinov3 image.jpg \
  --checkpoint checkpoints/vits16-best-ternary.pt \
  --device cuda \
  --top-k 5
```

## Python API

```python
import torch
from ternary_dinov3 import TernaryDINOv3Classifier, freeze_ternary_weights

model = TernaryDINOv3Classifier(
    "vit_small", num_classes=1000, image_size=224, patch_size=16
)
logits = model(torch.randn(2, 3, 224, 224))
frozen_model = freeze_ternary_weights(model)
```

## Scope

This project supports supervised fine-tuning and simple embedding distillation
from a frozen official DINOv3 ViT-S/16 teacher. See the
[distillation example](examples/README.md#distillation-from-official-dinov3-vit-s16)
for training and loading an embedding backbone. The complete multi-crop
DINO/iBOT training stack remains in TernaryCNN's `ternarylayers.dinov3` package
and is not duplicated here.
