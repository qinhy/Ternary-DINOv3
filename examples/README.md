# Examples

## Distillation from official DINOv3 ViT-S/16

`distill_dinov3.py` trains a ternary `vit_small` to reproduce the official
full-precision ViT-S/16 class-token embeddings. Both backbones receive the
exact same augmented, ImageNet-normalized RGB tensor and return `[B, 384]`.
The loss is `MSE(student(images), teacher(images))`. The teacher stays frozen
in evaluation mode; only student parameters are optimized. No classification
head, labels, multi-crop DINO objective, or projection adapter is needed.

Use a local clone of [official DINOv3](https://github.com/facebookresearch/dinov3)
and its `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` weights. The student starts
from those same weights with ternary quantization applied during each forward
pass. The example matches the official storage-token, LayerScale, norm, and
RoPE configuration, which differs from this project's bare `vit_small` defaults.

Place images directly in `data/distill/train/` and `data/distill/val/`, or in
subfolders. Images are discovered recursively; class folders and annotations
are unnecessary. Flat COCO `train2017` and `val2017` directories work directly.
Use distinct training and validation images.

```bash
uv run python examples/distill_dinov3.py \
  --teacher-repo ../dinov3 \
  --teacher-weights ../BitNetCNN/data/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --train-dir data/distill/train \
  --val-dir data/distill/val \
  --device cuda \
  --epochs 20
```

The script reports training and validation embedding MSE. It writes a resumable
`checkpoints/distilled-vits16.pt`, best QAT weights `distilled-vits16-best.pt`,
and frozen ternary weights `distilled-vits16-best-ternary.pt`. Resume with
`--resume checkpoints/distilled-vits16.pt` and the same model options;
`--epochs` is the final epoch. Lower validation MSE selects the best model;
this metric does not measure classification accuracy.

Progress logs show the first, last, and every 25th batch of training and
validation, including current and average MSE, learning rate during training,
elapsed time, estimated time remaining, and images per second. The first batch
also shows input and output shapes. Use `--log-interval 1` to log every batch,
or `--log-interval 0` to keep only the epoch summaries.

Load the best embedding backbone with:

```python
import torch
from ternary_dinov3 import build_backbone, freeze_ternary_weights

checkpoint = torch.load("checkpoints/distilled-vits16-best.pt", weights_only=True)
student = build_backbone(**checkpoint["backbone_config"], initialize=False)
student.load_state_dict(checkpoint["state_dict"])
student.eval()
with torch.no_grad():
    embeddings = student(images)  # images: ImageNet-normalized [B, 3, H, W]
    patches = student.forward_features(images)["x_norm_patchtokens"]
```

For the frozen checkpoint, call `freeze_ternary_weights(student)` before
loading its state dictionary. These are backbone checkpoints; the classifier
inference CLI expects a different checkpoint format. This example supervises
the class-token output only; patch embeddings are available but are not
directly matched by this loss. The MNIST femto model has 64-dimensional features
and ten class logits, so it cannot directly match the official 384-dimensional
output without an adapter or architecture change.

## MNIST with DINOv3 ViT-Femto

The script downloads MNIST, resizes each digit to 32×32, repeats grayscale into
three channels, and trains the ternary `vit_femto` backbone with 8×8 patches.

```bash
uv run python examples/train_mnist.py --device cuda
```

For a short CPU check before the full 60,000-image run:

```bash
uv run python examples/train_mnist.py \
  --device cpu \
  --epochs 1 \
  --limit-train 512 \
  --limit-test 128 \
  --workers 0 \
  --save checkpoints/mnist-femto-quick.pt
```

Resume a run with:

```bash
uv run python examples/train_mnist.py \
  --resume checkpoints/mnist-femto.pt \
  --epochs 20 \
  --device cuda
```

## MNIST with DINOv3 ViT-Femto TRM

`train_mnist_trm.py` trains the same 32×32, three-channel MNIST input with a
`DinoVisionTransformerTRM` backbone. The model carries recurrent `solution`
and `latent` states between supervision steps and adds a confidence head for
adaptive early-exit inference.

The default TRM configuration uses four supervision steps, samples `n` from
0–2 and `T` from 1–3 during training, and evaluates with `n=1` and `T=2`.

```bash
uv run python examples/train_mnist_trm.py --device cuda
```

For a short CPU check:

```bash
uv run python examples/train_mnist_trm.py \
  --device cpu \
  --epochs 1 \
  --limit-train 512 \
  --limit-test 128 \
  --workers 0 \
  --supervision-steps 2 \
  --n-max 1 \
  --T-max 2 \
  --save checkpoints/mnist-femto-trm-quick.pt
```

To use the deeper recurrent settings from the TRM training example:

```bash
uv run python examples/train_mnist_trm.py \
  --device cuda \
  --epochs 20 \
  --supervision-steps 16 \
  --n-min 0 \
  --n-max 4 \
  --T-min 1 \
  --T-max 8 \
  --eval-n 1 \
  --eval-T 2
```

Resume TRM training with:

```bash
uv run python examples/train_mnist_trm.py \
  --resume checkpoints/mnist-femto-trm.pt \
  --epochs 20 \
  --device cuda
```

The trainer reports classification accuracy, confidence-head accuracy, and the
average number of supervision steps used. It writes the latest training
checkpoint to `checkpoints/mnist-femto-trm.pt` by default, plus
`mnist-femto-trm-best.pt` and `mnist-femto-trm-best-ternary.pt` when validation
accuracy improves.

Keeping gradients through internal TRM latent refinement is enabled by default.
Disable it to reduce memory use:

```bash
uv run python examples/train_mnist_trm.py \
  --device cuda \
  --no-track-latent-grads
```
