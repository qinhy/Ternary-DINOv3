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
