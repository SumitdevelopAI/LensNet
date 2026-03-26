# LensNet
GSOC 2026

## Gravitational Lens Image Classification with Fourier Neural Operators

LensNet classifies gravitational lens images into three morphological classes
using two complementary model architectures:

| Class | Description |
|-------|-------------|
| `no_sub` | Smooth lens with no detectable substructure |
| `cdm_sub` | CDM-type compact subhalo substructure |
| `wdm_sub` | WDM-type arc/vortex substructure |

---

## Architecture Strategy

### Common Test I Baseline – CNN (ResNet-18)
A standard ResNet-18 convolutional backbone with a custom classification head.
CNNs learn spatially *local* features via sliding k×k filters.  Each layer
captures patterns within a limited receptive field, requiring many stacked
layers to model global image structure.

### FNO Classifier – Fourier Neural Operator
The FNO replaces the spatial convolutions with **spectral convolutions** that
operate entirely in the Fourier frequency domain:

1. **Lifting layer** – pointwise Conv2d maps input channels → `hidden_channels`.
2. **N × FNO blocks** – each block applies:
   - `SpectralConv2d`: 2-D real FFT → learnable complex weight multiplication
     on the lowest `modes × modes` frequencies → inverse FFT.
   - Bypass pointwise Conv2d (skip connection).
   - GELU activation.
3. **Projection** – 1×1 Conv2d + global average pooling.
4. **MLP head** – dropout + fully-connected layers → class logits.

#### Why FNO for lens classification?

| Property | Standard CNN | FNO |
|----------|-------------|-----|
| Receptive field | Local (k×k) | **Global** (full image via FFT) |
| Long-range correlations | Many layers needed | Captured in single FNO block |
| Parameter efficiency | Grows with image size | Independent of spatial resolution |
| Inductive bias | Spatial locality | Spectral smoothness (low-pass) |
| Computational complexity | O(N · k²) | O(N log N) via FFT |

Gravitational lens images are dominated by *global*, *smooth*, arc-shaped
features.  The FNO's spectral inductive bias – retaining only the lowest
Fourier modes – is naturally aligned with this structure, making it a strong
alternative to CNNs for this task.

---

## Repository Structure

```
lensnet/
├── models/
│   ├── spectral_layers.py   # SpectralConv2d + FNOBlock2d (core FNO ops)
│   ├── fno_classifier.py    # FNO-based classifier
│   └── cnn_baseline.py      # ResNet-18 CNN baseline (Common Test I)
├── data/
│   └── dataset.py           # LensDataset + synthetic data generator
└── training/
    └── trainer.py           # Training loop, early stopping, metrics
train.py                     # Train a single model
evaluate.py                  # Train and compare both models side-by-side
tests/
└── test_lensnet.py          # Unit + integration tests
requirements.txt
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the FNO classifier (synthetic data)

```bash
python train.py --model fno --synthetic --epochs 30
```

### Train the CNN baseline (synthetic data)

```bash
python train.py --model cnn --synthetic --epochs 30
```

### Train on real data

Organise your dataset as:
```
data/
├── no_sub/    *.npy
├── cdm_sub/   *.npy
└── wdm_sub/   *.npy
```

Then:
```bash
python train.py --model fno --data data/ --epochs 30 --save checkpoints/fno.pt
python train.py --model cnn --data data/ --epochs 30 --save checkpoints/cnn.pt
```

### Compare both models

```bash
# Train from scratch and compare:
python evaluate.py --data data/ --epochs 30

# Or load pre-trained checkpoints:
python evaluate.py --cnn-checkpoint checkpoints/cnn.pt \
                   --fno-checkpoint checkpoints/fno.pt
```

Results, learning-curve plots, confusion matrices, and an AUC comparison chart
are saved to `results/` by default.

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hidden-channels` | 64 | FNO internal feature width |
| `--modes` | 12 | Fourier modes kept per spatial dimension |
| `--n-fno-blocks` | 4 | Number of FNO residual blocks |
| `--image-size` | 64 | Spatial resolution after resize |
| `--epochs` | 30 | Maximum training epochs |
| `--lr` | 1e-3 | Initial Adam learning rate |
| `--patience` | 10 | Early-stopping patience (epochs) |
