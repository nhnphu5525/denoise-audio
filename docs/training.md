# Training Guide

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Training Pipeline Overview](#2-training-pipeline-overview)
3. [Loss Functions](#3-loss-functions)
   - 3.1 [Spectral L1 Loss](#31-spectral-l1-loss)
   - 3.2 [Spectral Convergence Loss](#32-spectral-convergence-loss)
   - 3.3 [Combined Spectral Loss (default)](#33-combined-spectral-loss-default)
4. [Evaluation Metrics](#4-evaluation-metrics)
   - 4.1 [SI-SNR](#41-si-snr)
   - 4.2 [SI-SNRi](#42-si-snri)
   - 4.3 [PESQ](#43-pesq)
   - 4.4 [STOI](#44-stoi)
5. [Optimizer and Schedule](#5-optimizer-and-schedule)
6. [Callbacks](#6-callbacks)
7. [Hyper-parameters](#7-hyper-parameters)
8. [Running Training](#8-running-training)
9. [Monitoring with TensorBoard](#9-monitoring-with-tensorboard)

---

## 1. Prerequisites

Complete the data pipeline first:

```bash
pip install -r requirements.txt
pip install pesq pystoi              # optional — perceptual metrics

python scripts/download_dataset.py
python -m src.data.preprocessing.audio --split all --pairs-per-clean 3
```

Verify that `data/processed/train/` contains `.npz` files before proceeding.

---

## 2. Training Pipeline Overview

```
data/processed/train/*.npz
        │
        ▼  tf.data.Dataset  (dataset.py — TODO)
  (noisy_batch, clean_batch)   shape: (B, 256, 32, 1)
        │
        ▼  UNet_IRM_Denoise.forward()
  clean_estimate               shape: (B, 256, 32, 1)
        │
        ▼  CombinedSpectralLoss
  total_loss  =  α · L₁_spectral + (1−α) · SpectralConvergence
        │
        ▼  Adam(lr=3e-4, clipnorm=1.0)
        ▼  ReduceLROnPlateau / EarlyStopping / ModelCheckpoint
```

---

## 3. Loss Functions

All losses are defined in `src/training/metrics.py`.

### 3.1 Spectral L1 Loss

Mean absolute error on log-magnitude spectrograms:

$$\mathcal{L}_{L1} = \frac{1}{F \cdot T} \sum_{k,t} \left| \hat{S}[k,t] - S[k,t] \right|$$

where $\hat{S}$ is the model output and $S$ is the clean target (both in log₁p space).

```python
from src.training.metrics import SpectralL1Loss

loss = SpectralL1Loss()
```

### 3.2 Spectral Convergence Loss

Normalised Frobenius-norm difference:

$$\mathcal{L}_{SC} = \frac{\|\hat{S} - S\|_F}{\|S\|_F}$$

Scale-invariant — penalises relative spectral error rather than absolute values.

```python
from src.training.metrics import SpectralConvergenceLoss

loss = SpectralConvergenceLoss()
```

### 3.3 Combined Spectral Loss (default)

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{L1} + (1-\alpha) \cdot \mathcal{L}_{SC}, \quad \alpha = 0.7$$

```python
from src.training.metrics import CombinedSpectralLoss

loss = CombinedSpectralLoss(alpha=0.7)   # default
```

The weighting $\alpha=0.7$ emphasises absolute magnitude accuracy ($\mathcal{L}_{L1}$)
while still penalising shape distortion via $\mathcal{L}_{SC}$.

---

## 4. Evaluation Metrics

Computed post-training on waveforms (not spectrograms). Require converting model
output back to audio via the postprocessing module.

### 4.1 SI-SNR

**Scale-Invariant Signal-to-Noise Ratio** — measures signal quality independently
of amplitude scaling:

$$\text{SI-SNR} = 10 \cdot \log_{10} \frac{\|\mathbf{s}_{target}\|^2}{\|\mathbf{e}_{noise}\|^2}$$

where:

$$\mathbf{s}_{target} = \frac{\langle \hat{s}, s \rangle}{\|s\|^2} \cdot s, \quad \mathbf{e}_{noise} = \hat{s} - \mathbf{s}_{target}$$

Higher is better. Typical good denoising achieves > 15 dB.

### 4.2 SI-SNRi

**SI-SNR Improvement** — improvement over the unprocessed noisy baseline:

$$\text{SI-SNRi} = \text{SI-SNR}(\hat{s}, s) - \text{SI-SNR}(y, s)$$

A positive value means the model improves upon the noisy input.

### 4.3 PESQ

**Perceptual Evaluation of Speech Quality** (ITU-T P.862).
Scores range from −0.5 (bad) to 4.5 (excellent). Requires the `pesq` package:

```bash
pip install pesq
```

### 4.4 STOI

**Short-Time Objective Intelligibility** (Taal et al., 2011).
Scores range from 0 to 1. Values > 0.65 are typically considered intelligible. Requires:

```bash
pip install pystoi
```

### Computing all metrics

```python
from src.training.metrics import compute_metrics

scores = compute_metrics(
    clean=clean_waveform,      # (N,) float32
    estimate=denoised_wave,    # (N,) float32
    noisy=noisy_waveform,      # (N,) float32
    sr=16000,
)
# returns: {"si_snr": float, "si_snr_improve": float, "pesq": float|None, "stoi": float|None}
```

---

## 5. Optimizer and Schedule

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate = 3e-4,
    clipnorm      = 1.0,      # gradient clipping prevents exploding gradients
)
```

**Learning rate scheduler:** `ReduceLROnPlateau`

| Parameter | Value | Meaning |
|---|---|---|
| `monitor` | `val_loss` | Watches validation loss |
| `patience` | 5 | Reduce LR after 5 epochs without improvement |
| `factor` | 0.5 | Multiply LR by 0.5 |
| `min_lr` | 1e-6 | Floor on learning rate |

---

## 6. Callbacks

| Callback | Config key | Purpose |
|---|---|---|
| `EarlyStopping` | `callbacks.early_stopping` | Stop when `val_loss` stalls for 15 epochs |
| `ModelCheckpoint` | `callbacks.model_checkpoint` | Save best weights to `models/checkpoints/` |
| `TensorBoard` | `callbacks.tensorboard` | Log metrics to `models/logs/` |
| `ReduceLROnPlateau` | `training.lr_*` | Adaptive LR schedule |
| `DenoiseMetricsCallback` | `src/training/metrics.py` | Log SI-SNR, PESQ, STOI per epoch |

---

## 7. Hyper-parameters

All training hyper-parameters live in `configs/train_config.yaml`:

```yaml
training:
  batch_size:       16
  epochs:           100
  learning_rate:    0.0003
  lr_scheduler:     reduce_on_plateau
  lr_patience:      5
  lr_factor:        0.5
  min_lr:           1.0e-6
  gradient_clip_norm: 1.0

loss:
  alpha: 0.7         # weight for SpectralL1; (1-alpha) for SpectralConvergence
```

---

## 8. Running Training

```bash
# Training script (to be implemented — see docs/contributing.md)
python scripts/train.py --config configs/train_config.yaml

# With custom overrides
python scripts/train.py --config configs/train_config.yaml \
    --epochs 50 \
    --batch-size 32
```

**Minimal training loop (manual):**

```python
import tensorflow as tf
from src.model.unet import build_unet_denoise
from src.training.metrics import CombinedSpectralLoss

model = build_unet_denoise()
model.compile(
    optimizer = tf.keras.optimizers.Adam(3e-4, clipnorm=1.0),
    loss      = CombinedSpectralLoss(alpha=0.7),
)

# load train/val datasets via src/data/dataset.py (TODO)
model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs          = 100,
    callbacks       = [...],
)
```

---

## 9. Monitoring with TensorBoard

```bash
tensorboard --logdir models/logs
```

Open `http://localhost:6006` to view:
- Training and validation loss curves
- Learning rate over time
- Per-epoch SI-SNR / PESQ / STOI (via `DenoiseMetricsCallback`)
