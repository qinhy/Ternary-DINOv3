# Examples

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

