# Gravitational Lens Classification — CNN and FNO

Classifying strong gravitational lensing images into three substructure classes using a CNN baseline and a Fourier Neural Operator (FNO), built with PyTorch for the ML4SCI program.

This gives FNO a global receptive field in a single layer.

Advantages:
- Captures long-range dependencies  
- Produces smoother training curves  
- Uses fewer parameters (~3× smaller than CNN)  

Limitation:
- Less effective at capturing fine local textures needed for class separation  

---

### 3. Class-wise Performance

| Class | CNN Accuracy | FNO Accuracy |
|-------|:------------:|:------------:|
| no | 96.0% | 89.6% |
| vort | 81.6% | 67.7% |
| sphere | 76.0% | 63.1% |

Interpretation:
- Both models perform best on `no` (simplest class)  
- Major confusion occurs between `vort` and `sphere`  
- These classes have visually similar lensing patterns, making them harder to distinguish  

---

### 4. Why CNN Performs Better

- Image resolution (150×150): CNN already captures global context with stacked layers  
- Feature type: Task requires local edge and texture detection  
- Model capacity: FNO (width=32, modes=24) has lower expressive power  
- Task nature: Classification favors spatial feature extraction over function mapping  

---

## Why Use FNO?

Despite lower accuracy, FNO offers important advantages:

- Physics-aware modeling: Works in frequency space, aligning with physical processes  
- Parameter efficiency: ~150K vs ~500K (CNN)  
- Global receptive field: Captures full-image dependencies instantly  
- Scalability: Expected to perform better on high-resolution and simulation-heavy tasks  

---

## CNN vs FNO Summary

| Aspect | CNN | FNO |
|--------|-----|-----|
| Feature focus | Local | Global |
| Best for | Texture, edges | Smooth structures |
| Accuracy | Higher | Lower |
| Parameters | Higher | Lower |
| Training | Noisier | Smoother |

---

## Choice of Architecture

We selected Fourier Neural Operator (FNO) over alternatives such as DeepONet because:

- The dataset consists of regular grid images (150×150)  
- FNO efficiently applies spectral convolution via FFT  
- DeepONet is more suitable for irregular domains and continuous query-based problems  

---

## Strategy

1. Built a CNN baseline to establish performance  
2. Replaced convolution layers with SpectralConv2d (FNO)  
3. Kept the same dataset, augmentations, and training pipeline  
4. Evaluated using accuracy, ROC-AUC, and confusion matrices  

---

## Conclusion

- CNN achieves the best performance for this classification task  
- FNO demonstrates strong potential but requires:
  - Higher capacity (more channels)  
  - More Fourier modes  
  - Higher-resolution inputs  

Key Insight:  
CNNs are better suited for low-resolution classification with local features, while FNOs are promising for physics-driven and high-resolution problems.

---

## Future Work

- Increase FNO width (64 / 128)  
- Tune number of Fourier modes  
- Develop hybrid CNN + FNO model  
- Train on higher-resolution lensing data  

---

## Dataset

- **Total samples** : 30,000 `.npy` images, shape `(1, 150, 150)`, single channel
- **Classes** : `no` (no substructure) · `vort` (vortex) · `sphere` (subhalo)
- **Normalisation** : min-max per sample, then scaled to `[-1, 1]`
- **Split** : 90% training / 10% held-out test (from `train/`) · `val/` folder for validation

```
data/
├── train/
│   ├── no/
│   ├── vort/
│   └── sphere/
└── val/
    ├── no/
    ├── vort/
    └── sphere/
```

---

## Project Structure

```
gravitational-lens-classification/
│
├── cnn_lens_classifier.ipynb       # CNN — full pipeline
├── fno_lens_classifier.ipynb       # FNO — full pipeline
├── roc_auc_cells.ipynb             # Standalone ROC/AUC cells
├── fno_prediction_viz.ipynb        # FNO prediction visualisation
├── cnn_lens.py                     # CNN standalone script
└── README.md
```

---

## Models

### CNN (Baseline)

Five convolutional blocks with BatchNorm and ReLU, each followed by MaxPool, then a fully connected classification head.

```
Input (1, 150, 150)
  Conv(1→16)   + BN + ReLU → MaxPool →  75×75
  Conv(16→32)  + BN + ReLU → MaxPool →  37×37
  Conv(32→64)  + BN + ReLU → MaxPool →  18×18
  Conv(64→128) + BN + ReLU → MaxPool →   9×9
  Conv(128→256)+ BN + ReLU
  AdaptiveAvgPool → FC(256→128) → FC(128→3)
```

### FNO (Fourier Neural Operator)

Replaces local spatial convolutions with global spectral convolutions via FFT. Each FNO block has a full-image receptive field in a single operation.

```
Input (1, 150, 150)
  Lift: Conv2d(1 → 32)
  FNOBlock × 4:
    SpectralConv2d → FFT → filter 24 modes → iFFT   [global]
    Conv2d(1×1)                                      [local residual]
    → BN → GELU
  AdaptiveAvgPool → FC(32→128) → FC(128→64) → FC(64→3)
```

**SpectralConv2d core operation:**
```python
x_ft  = torch.fft.rfft2(x)                      # spatial → frequency
out_ft[:, :, :modes, :modes] = x_ft @ W         # learned filter on low freqs
out   = torch.fft.irfft2(out_ft, s=(H, W))      # frequency → spatial
```

---

## Training Configuration

| Setting | CNN | FNO |
|---------|-----|-----|
| Batch size | 64 | 64 |
| Max epochs | 40 | 40 |
| Optimiser | AdamW | AdamW |
| Learning rate | 3e-4 | 3e-4 |
| Weight decay | 1e-4 | 1e-4 |
| LR schedule | Warm-up (3ep) + Cosine | Warm-up (3ep) + Cosine |
| Label smoothing | 0.05 | 0.05 |
| Dropout | 0.4 | 0.3 |
| Early stopping | patience=10 | patience=10 |
| FNO modes | — | 24 × 24 |
| FNO width | — | 32 |
| FNO blocks | — | 4 |

**Augmentation (training only):** horizontal flip (p=0.5), vertical flip (p=0.5), random 90° rotation.

---

## Results

| Model | Test Accuracy | AUC Macro | AUC Micro | Parameters |
|-------|:-------------:|:---------:|:---------:|:----------:|
| CNN   | **84.70%**    | **0.9535** | **0.9570** | ~500K |
| FNO   | 73.63%        | 0.8904    | 0.8954    | ~150K |

### Per-Class Results

**CNN**

| Class | Accuracy | AUC |
|-------|:--------:|:---:|
| no | 96.0% | 0.9717 |
| vort | 81.6% | 0.9534 |
| sphere | 76.0% | 0.9348 |

**FNO**

| Class | Accuracy | AUC |
|-------|:--------:|:---:|
| no | 89.6% | 0.9280 |
| vort | 67.7% | 0.8799 |
| sphere | 63.1% | 0.8626 |

---

## Training Curves

**CNN**<img width="2384" height="593" alt="training_curves" src="https://github.com/user-attachments/assets/460ce40c-97b5-430e-8a3e-0d9e83542850" />

**FNO**<img width="2384" height="593" alt="fno_training_curves" src="https://github.com/user-attachments/assets/3fd5ee00-2b28-4a66-955c-177dfc68208f" />


The CNN validation loss is noisier but converges to a lower value. The FNO trains more smoothly but plateaus at a higher loss, suggesting it needs more capacity (wider channels or more blocks) for this image size.

---

## Confusion Matrices

**CNN — Test Accuracy: 84.70%**<img width="1875" height="740" alt="confusion_matrix" src="https://github.com/user-attachments/assets/4e57d561-55d6-4ce6-9619-e81653d24e21" />


**FNO — Test Accuracy: 73.63%**<img width="1875" height="740" alt="fno_confusion_matrix" src="https://github.com/user-attachments/assets/178b6807-c01a-4054-b2bb-41982c3d7eed" />



Both models classify `no` most reliably. The primary confusion in both is between `vort` and `sphere` — two classes whose lensing signatures are visually similar. The CNN separates them more effectively (81.6% vs 76.0%) than the FNO (67.7% vs 63.1%).

---

## ROC Curves and AUC

**CNN**
<img width="2235" height="889" alt="roc_curves" src="https://github.com/user-attachments/assets/ff74067a-f4d1-4b29-bc23-560c84e779d0" />

<img width="2235" height="742" alt="roc_zoomed_and_auc_bar" src="https://github.com/user-attachments/assets/a075c547-5927-46fd-9369-d7d571f50521" />

**FNO**
<img width="2235" height="889" alt="fno_roc_curves" src="https://github.com/user-attachments/assets/8b7fb8e0-962d-4a39-8632-e87669ece51d" />

<img width="2235" height="742" alt="fno_roc_zoomed_auc_bar" src="https://github.com/user-attachments/assets/259400f5-f3ac-4ae2-a92c-0e5346550c02" />




The CNN achieves AUC > 0.93 on all three classes. The FNO achieves AUC > 0.86 on all classes — both models are well above random (0.5). The `no` class is the most separable in both models. `sphere` is the hardest class for both.

---

## Prediction Samples

**CNN — 20 test samples (green = correct, red = wrong)**

<img width="2010" height="1674" alt="predictions" src="https://github.com/user-attachments/assets/fef68eb8-5e28-47ad-8485-0a6950c8fe31" />

**FNO — 20 test samples (green = correct, red = wrong)**

<img width="2171" height="1871" alt="fno_predictions" src="https://github.com/user-attachments/assets/47c62e8e-4849-4d45-8d3a-c1136fed4372" />



---

## CNN vs FNO — Discussion

| Aspect | CNN | FNO |
|--------|-----|-----|
| Receptive field | Grows with depth (local 3×3) | Full image in one layer |
| Domain | Spatial (pixel space) | Frequency (Fourier space) |
| Test accuracy | **84.70%** | 73.63% |
| AUC macro | **0.9535** | 0.8904 |
| Training stability | Noisier validation | Smoother convergence |
| Parameters | ~500K | ~150K |

The CNN outperforms the FNO at this scale. This is likely because:

1. **Channel width** — FNO uses `width=32`. Increasing to 64 or 128 would add capacity.
2. **Image resolution** — FNO's advantage grows with higher-resolution inputs where local kernels struggle to build global context. At 150×150, CNNs with 5 blocks already achieve a large effective receptive field.
3. **Modes** — `modes=24` retains ~16% of spatial frequencies. Tuning this may help.

The FNO uses 3× fewer parameters and trains more stably, making it a strong candidate for further tuning.

---

## Setup

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn jupyter
```

Edit `TRAIN_DIR` and `VAL_DIR` in Cell 2 of each notebook, then run all cells top to bottom.

---

## References

- Li et al. (2020) — [Fourier Neural Operator for Parametric PDEs](https://arxiv.org/abs/2010.08895)
- Lu et al. (2019) — [DeepONet](https://arxiv.org/abs/1910.03193)
- [ML4SCI Program](https://ml4sci.org/)
