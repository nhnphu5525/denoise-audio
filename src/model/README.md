# UNet IRM Denoiser — Technical Reference

**File:** `unet.py`  
**Main class:** `UNetDenoiser`  
**Factory function:** `build_unet_denoise()`

---

## 1. Overview — Ideal Ratio Mask (IRM) Approach

Rather than directly predicting a clean spectrogram, the model predicts a **soft mask** $M$ with values in $(0, 1)$ and multiplies it element-wise with the noisy magnitude spectrogram to estimate the clean one:

$$
\hat{S} = M(|Y|) \cdot |Y|
$$

| Symbol | Description |
|---|---|
| $Y$ | STFT of the noisy input signal |
| $\|Y\|$ | Noisy magnitude spectrogram — model input |
| $M$ | IRM mask, sigmoid output, $M \in (0, 1)$ |
| $\hat{S}$ | Estimated clean magnitude spectrogram |

**Advantages of IRM over direct spectrogram mapping:**
- The mask is naturally bounded in $[0, 1]$ → sigmoid activation is a perfect fit
- The noisy phase is reused as-is — no phase prediction required
- Training is more stable: the output can never be negative or exceed the noisy magnitude

---

## 2. Input Preprocessing Pipeline

Before being fed into the model, raw audio is transformed through the following steps:

```
WAV (16 kHz, mono)
  → STFT  (n_fft=512, hop=128, win=512, hann window)
  → |magnitude|         shape: (257, T)
  → crop Nyquist bin    shape: (256, T)    [257 → 256, power-of-2]
  → log1p compression   log1p(|mag| + ε)
  → slice T-axis        shape: (256, 32, 1) per segment
```

**Why log1p compression?**  
Spectrograms follow a heavy-tailed distribution: a few frequency bins carry very large values while most are near zero. $\log(1+x)$ compresses the dynamic range and balances gradient magnitudes during training.

**Why crop the Nyquist bin?**  
STFT with `n_fft=512` produces 257 frequency bins (DC to Nyquist). The Nyquist bin (index 256) typically carries negligible energy. Removing it yields $F = 256 = 2^8$, which divides evenly across 4 asymmetric pooling stages (256 → 128 → 64 → 32 → 16).

---

## 3. Input Shape and Latency

| Parameter | Value | Notes |
|---|---|---|
| `n_fft` | 512 | FFT window size |
| `hop_length` | 128 | 8 ms per frame @ 16 kHz |
| `F` (freq bins) | 256 | `n_fft // 2`, after Nyquist crop |
| `T` (time frames) | 32 | sliding buffer window length |
| `C` (channels) | 1 | magnitude only, no phase |
| **Input shape** | **(256, 32, 1)** | |
| **Algorithmic latency** | **256 ms** | $32 \times \frac{128}{16000}$ |

---

## 4. Full Architecture Diagram

```
noisy_spectrogram (256, 32, 1)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  ENCODER                                                          │
│                                                                   │
│  enc1: ConvBlock(64)  ──────────────────────────── skip s1        │
│        MaxPool(2,1)   → (128, 32, 64)                             │
│  enc2: ConvBlock(128) ─────────────────────────── skip s2        │
│        MaxPool(2,1)   → ( 64, 32, 128)                            │
│  enc3: ConvBlock(256) ─────────────────────────── skip s3        │
│        MaxPool(2,1)   → ( 32, 32, 256)                            │
│  enc4: ConvBlock(512) ─────────────────────────── skip s4        │
│        MaxPool(2,1)   → ( 16, 32, 512)                            │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  BOTTLENECK                                                       │
│                                                                   │
│  ConvBlock(1024)       → (16, 32, 1024)                           │
│  SE Block              → channel recalibration (16, 32, 1024)    │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  DECODER                                                          │
│                                                                   │
│  dec4: ConvTranspose(2,1) → (32, 32, 512) + concat(s4)           │
│        ConvBlock(512)                                             │
│  dec3: ConvTranspose(2,1) → (64, 32, 256) + concat(s3)           │
│        ConvBlock(256)                                             │
│  dec2: ConvTranspose(2,1) → (128, 32, 128) + concat(s2)          │
│        ConvBlock(128)                                             │
│  dec1: ConvTranspose(2,1) → (256, 32, 64)  + concat(s1)          │
│        ConvBlock(64)                                              │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
  Conv2D(1, 1×1) + sigmoid  →  mask  (256, 32, 1)
        │
        ▼
  Multiply([mask, noisy])   →  clean_estimate  (256, 32, 1)
```

---

## 5. Component Details

### 5.1 ConvBlock

Each ConvBlock consists of two stacked Conv2D layers, enabling the network to learn more complex patterns than a single convolution:

```
Conv2D(F, 3×3, padding=same) → ReLU
BatchNormalization
Conv2D(F, 3×3, padding=same) → ReLU
BatchNormalization
SpatialDropout2D(0.3)
```

- **3×3 kernel**: captures both local frequency relationships and short-range temporal context simultaneously
- **BatchNorm**: normalises activations across the batch, stabilising training and allowing higher learning rates
- **SpatialDropout2D(0.3)**: drops entire feature map channels rather than individual pixels — better suited for spatially correlated data like spectrograms; critical for regularisation given that VIVOS contains only ~15 hours of speech

### 5.2 Asymmetric Pooling (2,1)

This is the key design decision enabling real-time inference:

| Pooling strategy | F axis | T axis | Problem |
|---|---|---|---|
| Standard `(2,2)` | ÷2 | ÷2 | T=32 → T=2 at bottleneck ❌ |
| Asymmetric `(2,1)` | ÷2 | unchanged | T=32 at every stage ✅ |

Feature map dimensions across all 4 encoder stages with `(2,1)` pooling:

| Stage | F | T | Channels |
|---|---|---|---|
| Input | 256 | 32 | 1 |
| After enc1 | 128 | 32 | 64 |
| After enc2 | 64 | 32 | 128 |
| After enc3 | 32 | 32 | 256 |
| After enc4 | 16 | 32 | 512 |
| Bottleneck | 16 | 32 | 1024 |

The time axis $T=32$ is **fully preserved** — the model sees the complete 256 ms context at every depth level.

### 5.3 Skip Connections

Skip connections are the defining property of U-Net, addressing the loss of fine-grained frequency detail as features pass through the bottleneck:

```
skip (from encoder stage N)  ──┐
                                ├── Concatenate → ConvBlock
upsampled decoder output     ──┘
```

At `dec1`, skip `s1` from `enc1` carries high-resolution frequency features that have never been downsampled. Concatenating them with the decoder output ensures the final mask has full frequency resolution.

### 5.4 Squeeze-and-Excitation Block (Bottleneck)

The SE block at the bottleneck performs **channel-wise attention**: not all feature maps are equally informative, and SE learns to re-weight them adaptively:

```
Input (16, 32, 1024)
  → GlobalAveragePooling2D      → (1024,)          [squeeze]
  → Dense(1024 // 16 = 64, relu) → (64,)
  → Dense(1024, sigmoid)         → (1024,)          [excite]
  → Reshape(1, 1, 1024)
  → Multiply with input          → (16, 32, 1024)   [scale]
```

The reduction ratio of 16 forces the intermediate FC layer to learn a compact representation of inter-channel relationships before expanding back to the full channel dimension.

### 5.5 Output Layer and IRM Application

```python
mask           = Conv2D(1, (1,1), activation="sigmoid")(d1)  # (256, 32, 1), ∈ (0,1)
clean_estimate = Multiply()([mask, noisy_input])              # (256, 32, 1)
```

- **Conv2D(1, 1×1)**: projects 64 channels down to a single-channel mask
- **sigmoid**: constrains mask ∈ $(0, 1)$; mask ≈ 1 preserves the noisy bin (speech-dominated); mask ≈ 0 suppresses it (noise-dominated)
- **Multiply inside the model**: the loss function operates directly on `clean_estimate` vs `clean_target` — no external mask application needed during training

---

## 6. Parameter Count

With default `base_filters=64`:

| Component | Output channels | Estimated parameters |
|---|---|---|
| enc1 ConvBlock | 64 | ~74K |
| enc2 ConvBlock | 128 | ~295K |
| enc3 ConvBlock | 256 | ~1.2M |
| enc4 ConvBlock | 512 | ~4.7M |
| Bottleneck ConvBlock | 1024 | ~18.9M |
| SE Block | 1024 | ~66K |
| dec4 ConvBlock | 512 | ~7.1M |
| dec3 ConvBlock | 256 | ~1.8M |
| dec2 ConvBlock | 128 | ~443K |
| dec1 ConvBlock | 64 | ~111K |
| Output Conv | 1 | ~65 |
| **Total** | | **~34.7M** |

> Run `python -m src.model.unet` to get the exact parameter count from `model.summary()`.

---

## 7. Real-Time Inference — Sliding Buffer

```
t=0ms:  buffer = [f1 ... f32]   → model → emit f32
t=8ms:  buffer = [f2 ... f33]   → model → emit f33
t=16ms: buffer = [f3 ... f34]   → model → emit f34
...
```

Each step: shift the buffer by 1 frame (8 ms), run a single inference pass, extract only the last output frame (`output_frame_idx=-1`) and send it to the audio sink.

**Total end-to-end latency:**

$$
\text{latency} = \underbrace{256\text{ ms}}_{\text{buffer fill}} + \underbrace{t_{\text{inference}}}_{\text{GPU/CPU}}
$$

The postprocessing counterpart is `AudioPostprocessor.reconstruct_frame()` in `src/data/postprocessing/audio.py`.

---

## 8. Model Variants

| Factory / Method | Output | Use when |
|---|---|---|
| `build_unet_denoise()` | `clean_estimate = mask × noisy` | Standard training and inference |
| `UNetDenoiser.build_mask_only()` | raw mask $M \in (0,1)$ | Inspecting mask values, visualisation, or manual application |

---

## 9. Configuration Reference

See [configs/train_config.yaml](../../configs/train_config.yaml):

```yaml
stft:
  n_fft: 512
  hop_length: 128

model:
  input_shape: [256, 32, 1]
  base_filters: 64
  dropout_rate: 0.3

realtime:
  buffer_frames: 32
  step_frames: 1
  output_frame_idx: -1
```
