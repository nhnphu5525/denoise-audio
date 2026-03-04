# Model Architecture

> **File:** `src/model/unet.py`  
> **Class:** `UNetDenoiser`  
> **Factory:** `build_unet_denoise(input_shape=(256,32,1), base_filters=64, dropout_rate=0.3)`

---

## Table of Contents

1. [The IRM Approach](#1-the-irm-approach)
2. [Input Representation](#2-input-representation)
3. [Asymmetric Pooling — The Real-Time Design Choice](#3-asymmetric-pooling--the-real-time-design-choice)
4. [Full Architecture Diagram](#4-full-architecture-diagram)
5. [Building Blocks](#5-building-blocks)
   - 5.1 [ConvBlock](#51-convblock)
   - 5.2 [Encoder Block](#52-encoder-block)
   - 5.3 [Decoder Block](#53-decoder-block)
   - 5.4 [Squeeze-and-Excitation Block](#54-squeeze-and-excitation-block)
   - 5.5 [Output Layer](#55-output-layer)
6. [Feature Map Dimensions](#6-feature-map-dimensions)
7. [Parameter Count](#7-parameter-count)
8. [Model Variants](#8-model-variants)
9. [Configuration Reference](#9-configuration-reference)

---

## 1. The IRM Approach

Instead of mapping noisy → clean spectrograms directly, the model predicts a **soft mask** $M$ and applies it to the noisy input:

$$\hat{S} = M(|Y|) \cdot |Y|$$

| Symbol | Description |
|---|---|
| $Y$ | STFT of the noisy waveform |
| $\|Y\|$ | Noisy log₁p-magnitude spectrogram — model **input** |
| $M$ | IRM mask (sigmoid), values $\in (0,1)$ |
| $\hat{S}$ | Estimated clean magnitude spectrogram — model **output** |

**Why IRM?**

- Mask is naturally bounded in $(0,1)$ — sigmoid is a perfect fit; no output clipping needed
- The output **cannot exceed** the noisy spectrogram magnitude (physically correct)
- Noisy phase is reused for waveform reconstruction — no phase estimator required
- Training is more stable than direct spectrogram regression

---

## 2. Input Representation

```
WAV (16 kHz, mono)
  → STFT              n_fft=512, hop=128, win=512, hann, center=True
  → |magnitude|       (257, T)   complex → float32
  → Nyquist crop      (256, T)   drop bin 256 → F = 2⁸ = 256
  → log1p             log(1 + |mag| + ε),  ε = 1e-8
  → slice T-axis      (256, 32, 1)  per training segment
```

| Parameter | Value | Derivation |
|---|---|---|
| `F` (freq bins) | 256 | `n_fft // 2` after Nyquist crop |
| `T` (time frames) | 32 | sliding buffer → 256 ms latency |
| `C` (channels) | 1 | magnitude only |
| Frame duration | 8 ms | `hop / sr = 128 / 16000` |

---

## 3. Asymmetric Pooling — The Real-Time Design Choice

Standard U-Net uses `(2,2)` max-pooling, halving both frequency ($F$) and time ($T$) at every encoder stage. With $T=32$:

```
After 4 stages:  T = 32 → 16 → 8 → 4 → 2   ← nearly no time context left
```

This project uses **asymmetric `(2,1)` pooling** — downsample $F$ only, preserve $T$:

```
After 4 stages:  T = 32 → 32 → 32 → 32 → 32   ← full time context preserved
                 F = 256 → 128 → 64 → 32 → 16
```

| Stage | Output shape | Pooling |
|---|---|---|
| Input | (256, 32, 1) | — |
| After enc1 | (128, 32, 64) | MaxPool(2,1) |
| After enc2 | (64, 32, 128) | MaxPool(2,1) |
| After enc3 | (32, 32, 256) | MaxPool(2,1) |
| After enc4 | (16, 32, 512) | MaxPool(2,1) |
| Bottleneck | (16, 32, 1024) | — |

The same `(2,1)` stride is used in the decoder's `Conv2DTranspose` upsampling steps.

---

## 4. Full Architecture Diagram

```
noisy_spectrogram  (256, 32, 1)
        │
        ▼
 ╔══════════════════════════════╗
 ║         ENCODER              ║
 ║                              ║
 ║  enc1: ConvBlock(64)         ║──── skip s1 (256, 32,  64) ───────────────────┐
 ║        MaxPool(2,1)          ║                                               │
 ║        → (128, 32,  64)      ║                                               │
 ║                              ║                                               │
 ║  enc2: ConvBlock(128)        ║──── skip s2 (128, 32, 128) ───────────────┐   │
 ║        MaxPool(2,1)          ║                                           │   │
 ║        → ( 64, 32, 128)      ║                                           │   │
 ║                              ║                                           │   │
 ║  enc3: ConvBlock(256)        ║──── skip s3 ( 64, 32, 256) ───────────┐   │   │
 ║        MaxPool(2,1)          ║                                       │   │   │
 ║        → ( 32, 32, 256)      ║                                       │   │   │
 ║                              ║                                       │   │   │
 ║  enc4: ConvBlock(512)        ║──── skip s4 ( 32, 32, 512) ───────┐   │   │   │
 ║        MaxPool(2,1)          ║                                   │   │   │   │
 ║        → ( 16, 32, 512)      ║                                   │   │   │   │
 ╚══════════════════════════════╝                                   │   │   │   │
        │                                                           │   │   │   │
        ▼                                                           │   │   │   │
 ╔══════════════════════════════╗                                   │   │   │   │
 ║       BOTTLENECK             ║                                   │   │   │   │
 ║                              ║                                   │   │   │   │
 ║  ConvBlock(1024)             ║                                   │   │   │   │
 ║  SE Block (ratio=16)         ║                                   │   │   │   │
 ║  → (16, 32, 1024)            ║                                   │   │   │   │
 ╚══════════════════════════════╝                                   │   │   │   │
        │                                                           │   │   │   │
        ▼                                                           │   │   │   │
 ╔══════════════════════════════╗                                   │   │   │   │
 ║         DECODER              ║                                   │   │   │   │
 ║                              ║                                   │   │   │   │
 ║  dec4: TransposeConv(2,1)    ║←──────────────── concat ─────────┘   │   │   │
 ║        ConvBlock(512)        ║  (32, 32,  512+512) → (32, 32, 512)  │   │   │
 ║                              ║                                       │   │   │
 ║  dec3: TransposeConv(2,1)    ║←──────────────── concat ─────────────┘   │   │
 ║        ConvBlock(256)        ║  (64, 32,  256+256) → (64, 32, 256)       │   │
 ║                              ║                                           │   │
 ║  dec2: TransposeConv(2,1)    ║←──────────────── concat ───────────────────┘   │
 ║        ConvBlock(128)        ║  (128, 32, 128+128) → (128, 32, 128)           │
 ║                              ║                                               │
 ║  dec1: TransposeConv(2,1)    ║←──────────────── concat ─────────────────────┘
 ║        ConvBlock(64)         ║  (256, 32,  64+64) → (256, 32, 64)
 ╚══════════════════════════════╝
        │
        ▼
  Conv2D(1, 1×1, sigmoid) → mask  (256, 32, 1)
        │
        ▼  Multiply
  mask × noisy_spectrogram → clean_estimate  (256, 32, 1)
```

---

## 5. Building Blocks

### 5.1 ConvBlock

```
Conv2D(F, 3×3, padding=same) → BN → ReLU
Conv2D(F, 3×3, padding=same) → BN → ReLU
SpatialDropout2D(rate=0.3)
```

Two stacked convolutions expand the receptive field. `SpatialDropout2D` drops
entire feature maps (vs individual pixels in standard Dropout), which is more
effective for 2-D spatial data.

### 5.2 Encoder Block

```
ConvBlock(F)  →  skip connection
MaxPooling2D(pool_size=(2,1))  →  next encoder stage
```

The skip connection preserves fine-grained frequency-time features that would
otherwise be lost through pooling.

### 5.3 Decoder Block

```
Conv2DTranspose(F, kernel=(2,1), strides=(2,1)) → upsampled
Concatenate([upsampled, skip])
ConvBlock(F)
```

`Conv2DTranspose` with `strides=(2,1)` doubles the frequency axis back to its
original size. Concatenating the skip connection (from the mirrored encoder stage)
reintroduces spatial detail.

### 5.4 Squeeze-and-Excitation Block

Placed at the bottleneck. Recalibrates channel responses using global context:

$$\mathbf{s} = \sigma\!\left(W_2 \cdot \delta\!\left(W_1 \cdot \text{GAP}(X)\right)\right)$$
$$\tilde{X} = \mathbf{s} \odot X$$

| Step | Operation | Shape |
|---|---|---|
| Input | bottleneck feature map | $(16, 32, 1024)$ |
| GAP | `GlobalAveragePooling2D` | $(1024,)$ |
| FC1 | `Dense(64, relu)` (ratio=16) | $(64,)$ |
| FC2 | `Dense(1024, sigmoid)` | $(1024,)$ |
| Reshape | | $(1, 1, 1024)$ |
| Scale | `Multiply([X, s])` | $(16, 32, 1024)$ |

### 5.5 Output Layer

```
Conv2D(1, 1×1, activation="sigmoid") → mask ∈ (0,1)
Multiply([mask, noisy_input])        → clean_estimate
```

The $1 \times 1$ convolution acts as a per-pixel channel mixer to collapse
1024 feature channels down to the single-channel mask.

---

## 6. Feature Map Dimensions

| Layer | Shape | Filters |
|---|---|---|
| Input | (256, 32, 1) | — |
| enc1 output | (256, 32, 64) | 64 |
| enc1 pooled | (128, 32, 64) | 64 |
| enc2 output | (128, 32, 128) | 128 |
| enc2 pooled | (64, 32, 128) | 128 |
| enc3 output | (64, 32, 256) | 256 |
| enc3 pooled | (32, 32, 256) | 256 |
| enc4 output | (32, 32, 512) | 512 |
| enc4 pooled | (16, 32, 512) | 512 |
| Bottleneck | (16, 32, 1024) | 1024 |
| dec4 output | (32, 32, 512) | 512 |
| dec3 output | (64, 32, 256) | 256 |
| dec2 output | (128, 32, 128) | 128 |
| dec1 output | (256, 32, 64) | 64 |
| Output mask | (256, 32, 1) | 1 |

---

## 7. Parameter Count

| Component | Parameters (approx.) |
|---|---|
| Encoder (4 blocks) | ~7.2 M |
| Bottleneck ConvBlock | ~18.9 M |
| SE Block | ~132 K |
| Decoder (4 blocks) | ~8.4 M |
| Output 1×1 Conv | ~65 |
| **Total** | **~34.7 M** |

---

## 8. Model Variants

The `UNetDenoiser` class exposes two build methods:

```python
from src.model.unet import build_unet_denoise, UNetDenoiser

# Standard: returns clean_estimate = mask × noisy
model = build_unet_denoise(input_shape=(256, 32, 1), base_filters=64, dropout_rate=0.3)

# Mask-only: returns raw sigmoid mask for visualisation / debugging
mask_model = UNetDenoiser(input_shape=(256, 32, 1)).build_mask_only()
```

---

## 9. Configuration Reference

```yaml
# configs/train_config.yaml → [model]
model:
  input_shape:  [256, 32, 1]   # [F, T, C]
  base_filters: 64             # filters in enc1; doubles at each stage
  dropout_rate: 0.3            # SpatialDropout2D rate

# configs/train_config.yaml → [stft]
stft:
  n_fft:       512
  hop_length:  128
  win_length:  512
  window:      hann
  center:      true
```
