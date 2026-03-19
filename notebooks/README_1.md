# HybridDeepFilterNet — Full Technical Documentation

> **A hybrid speech enhancement system** combining a MobileOne convolutional encoder,
> Conformer Lite sequence modelling, Time-Frequency (TF) attention, GRU temporal
> refinement, and a band-split DeepFilter complex FIR stage — all trained end-to-end
> from raw waveforms.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Full Pipeline Overview](#2-full-pipeline-overview)
3. [Module-by-Module Breakdown](#3-module-by-module-breakdown)
   - 3.1 [Utilities](#31-utilities)
   - 3.2 [Augmentation & Data Mixing](#32-augmentation--data-mixing)
   - 3.3 [Dataset](#33-dataset)
   - 3.4 [MobileOne Encoder](#34-mobileone-encoder)
   - 3.5 [Conformer Lite](#35-conformer-lite)
   - 3.6 [TF Attention Block](#36-tf-attention-block)
   - 3.7 [GRU Refinement](#37-gru-refinement)
   - 3.8 [Magnitude Mask Head](#38-magnitude-mask-head)
   - 3.9 [Tap Head — Deep Filter](#39-tap-head--deep-filter)
   - 3.10 [Band-Split Filtering (Forward Pass)](#310-band-split-filtering-forward-pass)
   - 3.11 [Loss Functions](#311-loss-functions)
   - 3.12 [Training Infrastructure](#312-training-infrastructure)
   - 3.13 [Evaluation & Metrics](#313-evaluation--metrics)
   - 3.14 [Export](#314-export)
4. [Key Design Decisions & Novelties](#4-key-design-decisions--novelties)
5. [Dataset & Preprocessing](#5-dataset--preprocessing)
6. [Training Configuration](#6-training-configuration)
7. [Evaluation Protocol](#7-evaluation-protocol)
8. [Incorrect or Misleading Comments Found in the Code](#8-incorrect-or-misleading-comments-found-in-the-code)
9. [Information Gaps to Fill for a Paper](#9-information-gaps-to-fill-for-a-paper)
10. [Suggested Paper Outline](#10-suggested-paper-outline)

---

## 1. Project Summary

**HybridDeepFilterNet** is an end-to-end single-channel speech enhancement (denoising)
system. Its goal is to remove background noise, additive interference, and reverb from a
noisy audio waveform and recover the dry clean speech — all in near-real-time on
consumer hardware (NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM).

The system operates entirely in the complex STFT domain, applies a multi-branch neural
network to estimate both a **coarse real-valued magnitude mask** and a set of
**complex FIR filter taps** (DeepFilter), then recombines them via a novel
**band-split** strategy before inverting the spectrogram back to a waveform.

| Property | Value |
|---|---|
| Sample rate | 16 000 Hz |
| STFT window | Hann, 400-sample (25 ms) |
| STFT hop | 160 samples (10 ms) |
| FFT size | 512 → 257 freq bins |
| Segment length | 3.0 s |
| Model parameters | ~8.29 M (with `enc_stages=(64,128,256)`) |
| Target hardware | NVIDIA RTX 4060 Laptop (8.6 GB VRAM) |
| Framework | PyTorch 2.5.1 + CUDA 12.1 |

---

## 2. Full Pipeline Overview

```
Noisy Waveform  (B, T)
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  STFT  — n_fft=512, hop=160, win=400, Hann window               │
│  → Complex spectrogram  (B, 257, F)                             │
│  → Log-power: 20·log10(|X|)                                     │
│  → Per-sample Z-normalization  → feats  (B, 1, 257, T)          │
└──────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  MobileOne Encoder                                               │
│  in_proj: Conv2d(1 → 32/64) + BN + SiLU                         │
│  Stage 0 (32→64 ch):  2× MobileOneBlock + 1× DSConv2d           │
│  Stage 1 (64→128 ch): 2× MobileOneBlock + 1× DSConv2d           │
│  Stage 2 (128→256 ch):2× MobileOneBlock + 1× DSConv2d           │
│  Output: (B, C=256, F=257, T)                                    │
└──────────────────────────────────────────────────────────────────┘
  │
  ▼  mean-pool freq dim → (B, T, C=256)
┌──────────────────────────────────────────────────────────────────┐
│  Conformer Lite × 2   [d=256, heads=4, ffn_dim=512, conv_k=31]  │
│  Each block: FFN₁ → MHA → DepthwiseConv → FFN₂ → LayerNorm      │
│  Output projected via Linear(C→C)                               │
│  Broadcast back to 2-D:  (B, C, F=257, T)  via unsqueeze+add    │
└──────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  TFAttentionBlock                                                │
│  • Channel SE (squeeze-excite, ratio 8)                          │
│  • Temporal 1-D depthwise conv gate  (kernel=15)                 │
│  • Frequency 1-D depthwise conv gate (kernel=15)                 │
│  • GroupNorm(8) + residual                                       │
│  Output: (B, C, F, T)                                           │
└──────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  GRU Refinement                                                  │
│  mean-pool freq → (B, T, C) → Linear(C→256) → GRU(256, layers=2)│
│  Linear(256→C) + LayerNorm → broadcast add back to (B, C, F, T) │
└──────────────────────────────────────────────────────────────────┘
  │
  ├─────────────────────────────────┐
  ▼                                 ▼
[Magnitude Mask Head]          [Tap Head]
Conv2d(C→C/2)+SiLU             Conv2d(C→C/2)+SiLU
Conv2d(C/2→1)+Sigmoid          Conv2d(C/2→2K)   (K=5 taps, real+imag)
× 2.0  → mag_mask (B,F,T)     tanh-bounded magnitude normalization
                                → taps_real, taps_imag  (B,F,K,T)
  │                                 │
  ▼──────── Band-Split Fusion ──────┘
  │
  │  LOW BAND (bins 0–159 → 0–5 kHz):
  │    inject mag_mask into identity tap position of taps_real_low
  │    apply complex FIR:  (spec_real + j·spec_imag) * (tr + j·ti)
  │    causal pad: K-1 zeros prepended on time axis
  │
  │  HIGH BAND (bins 160–256 → 5–8 kHz):
  │    multiply spectrogram by mag_mask (amplitude scale, keep noisy phase)
  │
  │  Concatenate → denoised_spec (B, 257, T)  [complex]
  │
  ▼
ISTFT → denoised waveform  (B, T)
```

---

## 3. Module-by-Module Breakdown

### 3.1 Utilities

**File origin:** `src/utils.py` (inlined)

| Function | Purpose | Notes |
|---|---|---|
| `seed_all(seed)` | Deterministic seeding for reproducibility | Seeds Python `random`, NumPy, PyTorch CPU+CUDA |
| `compute_rms(x)` | Root-mean-square energy | Float64 to prevent underflow |
| `scale_noise_to_snr(clean, noise, snr_db)` | Scales noise array to a target SNR relative to clean | Used during mixing |
| `soft_clip(x, threshold)` | Tanh-based soft saturation | Simulates microphone ADC clipping |
| `snr_db(clean, test)` | Computes output/improvement SNR | Evaluation metric |
| `load_audio_mono(path, sr)` | Loads any audio format → float32 mono, resamples if needed | Falls back from soundfile → scipy; avoids librosa/numba in workers |
| `crop_or_repeat(x, length, rng)` | Random crop if long; tile if short | Ensures fixed-length segments for batching |
| `peak_normalize(x, peak)` | Scales waveform to a peak amplitude | Pre-processing step |
| `save_manifest / load_manifest` | JSONL file I/O for validation sets | Ensures reproducible val/test splits |
| `generate_val_manifest` | Builds fixed deterministic noisy-clean pairs for validation | Seeds each item individually; allows 1–3 noise sources per sample |

**How it connects to the rest:** All dataset I/O, mixing, and augmentation functions depend on these primitives. The RIR cache builder and `NoiseSuppressionDataset` both call `load_audio_mono`.

---

### 3.2 Augmentation & Data Mixing

**File origin:** `src/augment.py` (inlined)

This module synthesizes noisy training data on-the-fly from pairs of clean speech and
noise recordings. It performs a rich chain of stochastic transformations.

#### Noise mixing chain (`mix_clean_with_noises`)

```
Clean segment
    │
    ├── (optionally) convolve with Room Impulse Response → clean_reverb
    │         clean_proc (the training TARGET) stays DRY — no reverb
    │
    ├── Load 1–3 noise files
    │     ├── _maybe_time_or_pitch_perturb   (35% chance: ±5% time-stretch)
    │     ├── Bandpass filter                (35% chance)
    │     └── Random gain ±6 dB
    │
    ├── Sum noise sources → noise_sum
    │
    ├── Scale to target SNR:
    │     Static:      triangular(0, 15, 40) dB     [DNS Challenge-style]
    │     Time-varying: 2–6 random segments, each with independent SNR
    │
    ├── _add_hum                 (25% chance: 50 or 60 Hz mains hum)
    ├── _apply_device_effects
    │     ├── bandlimit           (35% chance: simulate telephone/codec)
    │     ├── hard clip + LPF     (12% chance)
    │     └── soft_clip           (25% chance)
    ├── Final global gain: uniform(−3, +3) dB
    ├── Peak normalize to 0.90–0.99          (90% chance)
    └── Post soft_clip                        (25% chance)
```

#### Room Impulse Response (RIR) generation

Three sources, in priority order:
1. **pyroomacoustics** — shoebox room simulation, random T60 in [0.08, 0.8] s
2. **On-disk RIR dataset** — looks for `.wav/.flac` files in `data/rirs/`
3. **Synthetic RIR** — exponential decay envelope with white noise

**RIR Cache:** At training startup, 500 RIRs are pre-generated into a Python list
(`build_rir_cache`). During training, each sample picks from the cache randomly
in O(1) — no compute penalty during the hot data-loading path.

**Critical design decision — dry target:** The RIR is applied only to
`clean_reverb` (which is mixed with noise to produce the noisy input). The
training target `clean_proc` is always the **dry, anechoic** clean signal.
This trains the model to dereverberate as well as denoise.

**How it connects:** `NoiseSuppressionDataset.__getitem__` calls
`mix_clean_with_noises` for every training sample. The validation set uses
fixed seeds via manifests to ensure reproducible metrics.

---

### 3.3 Dataset

**File origin:** `src/dataset.py` (inlined)

`NoiseSuppressionDataset(Dataset)` handles both training and validation modes with
different sampling strategies:

| Mode | Length | Mixing | Seed strategy |
|---|---|---|---|
| `train` | `len(clean_list)` | On-the-fly, fresh per epoch (seed = base_seed + idx) | Per-item deterministic within epoch |
| `val` | `len(manifest)` | Pre-fixed via JSONL manifest | Per-item fixed seed stored in manifest |

Each `__getitem__` returns:
```python
{
  'clean': torch.FloatTensor (T,),  # dry clean target
  'noisy': torch.FloatTensor (T,),  # noisy + reverbed input
  'meta':  dict,                    # SNR, augmentation metadata
}
```

The `collate_audio` function zero-pads to the maximum length within the batch and stacks into `(B, T)` tensors.

---

### 3.4 MobileOne Encoder

**File origin:** `src/mobileone.py` (inlined)

The encoder is built from **MobileOneBlock** — a re-parameterizable multi-branch
convolutional block inspired by the Apple MobileOne architecture.

#### Training-time structure (per block)

```
Input
  ├── rbr_conv[0]:  3×3 Conv + BN        (branch 0)
  ├── rbr_conv[1]:  3×3 Conv + BN        (branch 1)
  ├── rbr_scale:    1×1 Conv + BN         (scale branch)
  └── rbr_identity: BN only               (if stride=1, in_ch=out_ch)
  │
  sum → SiLU activation → Output
```

#### Inference-time structure (after reparameterization)

All branches are **algebraically fused** into a single `3×3 Conv + bias` via
`get_equivalent_kernel_bias()`:
- `fuse_conv_bn` folds BN scale/bias into conv weights
- `_pad_1x1_to_3x3` zero-pads 1×1 kernels to 3×3
- `_fuse_identity_bn` converts identity shortcut into a diagonal 3×3 kernel

This means at inference there is **zero branching overhead** — exact same
mathematical result, half the memory reads.

#### Encoder architecture

```python
in_proj: Conv2d(1 → enc_stages[0]=64) + BN + SiLU
Stage 0: MobileOneBlock(64→128) × 2 + DSConv2d(128)
Stage 1: MobileOneBlock(128→256) × 2 + DSConv2d(256)
# enc_stages = (64, 128, 256) in the final config
```

`DSConv2d` (Depthwise-Separable Conv2d) acts as a final mixer within each stage:
depthwise 3×3 → BN → SiLU → pointwise 1×1 → BN → SiLU.

**Input/output:** The encoder takes the normalized log-power spectrogram
`(B, 1, F=257, T)` and outputs feature maps `(B, C=256, F=257, T)`.

**How it connects:** Encoder features feed the conformer, TF attention, and GRU
blocks. The 2-D feature map is never downsampled spatially — every frequency bin
and every time frame is preserved at full resolution throughout.

---

### 3.5 Conformer Lite

**File origin:** Part of `model.py` (inlined)

Two stacked `ConformerLiteBlock` modules capture **temporal sequential
dependencies** across the compressed frequency-pooled representation.

#### ConformerLiteBlock structure

```
Input (B, T, C=256)
  ▼
FFN₁  (half-step: 0.5 × residual scale)
  LayerNorm → Linear(C→ffn_dim=512) → SiLU → Dropout → Linear(512→C) → Dropout
  ▼
Multi-Head Self-Attention  (heads=4, batch_first=True)
  LayerNorm → MHA → Dropout + residual
  ▼
Convolutional Module (depthwise along time)
  LayerNorm → GLU gating (pointwise 1×1, split into x·σ(g)) →
  DepthwiseConv1d(kernel=31, causal via padding=k//2) → BN → SiLU →
  pointwise 1×1 → Dropout + residual
  ▼
FFN₂  (same as FFN₁)
  ▼
LayerNorm
  ▼
Output (B, T, C=256)
```

The conformer operates on the **frequency-pooled** feature sequence
`x.mean(dim=2)` — i.e., a single vector per time step — then adds the result
back into the 2-D (freq × time) feature tensor via broadcasting. This is a
deliberately lightweight approach: global temporal context without the quadratic
cost of 2-D attention.

**Why Conformer Lite?** Full conformer operates on long sequences and is
memory-heavy. This "lite" variant:
- Operates on C-dim vectors, not raw spectrograms
- Uses standard SDPA (not relative positional encoding)
- Is limited to 2 blocks to stay within 8 GB VRAM

---

### 3.6 TF Attention Block

**File origin:** Part of `model.py` (inlined)

`TFAttentionBlock` is a **custom 2-D attention module** that simultaneously
models channel importance (SE), temporal patterns, and frequency patterns.

```
Input x (B, C, F, T)
  │
  ├── SE gate: AdaptiveAvgPool2d(1) → FC(C→C/8→C) → Sigmoid → channel reweighting
  │
  ├── Temporal gate:
  │     x.mean(dim=2)  →  (B, C, T)
  │     DepthwiseConv1d(kernel=15) → BN → SiLU
  │     unsqueeze(2) → (B, C, 1, T) gate
  │     x + pointwise_gate(x · temporal_gate)
  │
  ├── Frequency gate:
  │     x.mean(dim=3)  →  (B, C, F)
  │     DepthwiseConv1d(kernel=15) → BN → SiLU
  │     unsqueeze(3) → (B, C, F, 1) gate
  │     x + pointwise_gate(x · freq_gate)
  │
  └── GroupNorm(8, C) + full residual → Output (B, C, F, T)
```

**Interactions:** Temporal gate focuses the model on salient speech frames; frequency
gate focuses it on voiced formant bands. The SE gate additionally up-weights
channels that consistently carry speech energy. Together, these three axes of attention
help the network distinguish speech from noise without flattening the TF structure.

---

### 3.7 GRU Refinement

**File origin:** Part of `model.py` (inlined)

After spatial attention, a `GRURefinement` module provides **causal temporal
recurrence** — critical for streaming/real-time operation.

```
Input x (B, C=256, F=257, T)
  │
  mean over freq → (B, C, T) → transpose → (B, T, C)
  │
  Linear(C → hidden=512) → GRU(512, 512, layers=2, batch_first=True)
  │
  Linear(512 → C) → LayerNorm(C)
  │
  transpose → (B, C, T) → unsqueeze(2) → (B, C, 1, T)
  │
  x + context_broadcast  → Output (B, C, F, T)
```

The hidden state `h` is passed in and returned (stateful), enabling
**chunk-by-chunk streaming inference** without recomputing history.

**How it connects:** The GRU output feeds directly to both mask heads.
Its hidden state encodes recent speech/noise context, which helps the filter
adapt to non-stationary noise conditions.

---

### 3.8 Magnitude Mask Head

**File origin:** Part of `model.py` (inlined)

```python
mask_mag_head = Sequential(
    Conv2d(C, C//2, 1, bias=False),
    SiLU(),
    Conv2d(C//2, 1, 1),
    Sigmoid(),      # output ∈ (0, 1)
)
# Scaled: mag_mask = mask_mag_head(x).squeeze(1) * 2.0
# Final range: (0, 2.0) — allows slight boosting of soft speech regions
```

The magnitude mask `mag_mask ∈ (0, 2)` is a **real-valued scalar per
time-frequency bin** that controls global noise suppression strength. Values
below 1.0 suppress; values above 1.0 can slightly amplify — useful for boosting
soft phonemes that risk being over-suppressed.

This mask is used in two ways:
1. Directly applied to the **high-frequency band** (bins 160–256, 5–8 kHz)
2. Injected into the **identity tap** of the deep filter for the low-frequency band

---

### 3.9 Tap Head — Deep Filter

**File origin:** Part of `model.py` (inlined)

```python
tap_head = Sequential(
    Conv2d(C, C//2, 1, bias=False),
    SiLU(),
    Conv2d(C//2, K * 2, 1),   # K=5 taps, outputs 10 channels: K real + K imag
)
```

The raw tap predictions `taps_raw` are normalized with a custom
**tanh-based magnitude bounding**:

```python
tap_mag = sqrt(taps_real² + taps_imag² + ε)
scale   = (2.0 * tanh(tap_mag)) / tap_mag
taps_real = taps_raw[:, :, :K, :] * scale
taps_imag = taps_raw[:, :, K:, :] * scale
```

This ensures the **total complex magnitude of each tap is bounded** by 2.0
(since `tanh(·) ∈ (-1, 1)`, the maximum L2 norm of the complex tap is `2 * tanh(∞) → 2`),
preventing filter instability without clipping gradients.

**Output shape:** `taps_real, taps_imag` are both `(B, F, K=5, T)`.

---

### 3.10 Band-Split Filtering (Forward Pass)

This is the **core novel contribution** of the architecture. Rather than applying the
same operation to all frequency bins, the model uses a band-split strategy:

#### Low band (bins 0–159 → 0 to 5 kHz): Complex FIR

```python
# Inject magnitude mask into the "current" tap position (K-1)
identity_low = zeros_like(taps_real[:, :df_bins, :, :])
identity_low[:, :, K-1, :] = mag_mask[:, :df_bins, :]

taps_real_low = taps_real[:, :df_bins, :, :] + identity_low
taps_imag_low = taps_imag[:, :df_bins, :, :]

low_spec = _deep_filter(spec[:, :df_bins, :], taps_real_low, taps_imag_low)
```

`_deep_filter` implements a **causal complex FIR convolution**:

```
For each frequency bin f:
  X_real[f, t-(K-1):t]  ← K past frames of real part
  X_imag[f, t-(K-1):t]  ← K past frames of imaginary part
  
  Y_real[f, t] = Σ (X_real[k] · W_real[k] − X_imag[k] · W_imag[k])
  Y_imag[f, t] = Σ (X_real[k] · W_imag[k] + X_imag[k] · W_real[k])
```

This is the standard complex multiplication rule `(a+jb)(c+jd) = (ac−bd) + j(ad+bc)`,
applied across K time lags. The `K-1` zero-pad on the **left** enforces causality —
no future frames are seen.

#### High band (bins 160–256 → 5 to 8 kHz): Magnitude-only masking

```python
high_spec = spec[:, df_bins:, :] * mag_mask[:, df_bins:, :]
```

Above 5 kHz, the spectral **phase is perceptually unreliable** (fine phase structure
matters less for intelligibility). The model preserves the noisy phase and only
scales the magnitude. This avoids the model fitting noise to random high-frequency
phase patterns.

#### Final output

```python
denoised_spec = torch.cat([low_spec, high_spec], dim=1)  # (B, 257, T)
denoised_wav  = istft(denoised_spec, length=L)
```

---

### 3.11 Loss Functions

#### `HybridLoss` — the actual training criterion

`HybridLoss` operates in the **complex compressed spectral domain**:

```
pred_wav, clean_wav → STFT → P_spec, C_spec

c = 0.3   (power compression exponent)
P_mag = |P_spec|.clamp(1e-8)
C_mag = |C_spec|.clamp(1e-8)

--- Asymmetric Compressed Magnitude Loss ---
mag_diff = P_mag^c - C_mag^c
mag_loss = mean(
    |mag_diff| * 2.0   if mag_diff < 0   (model under-estimated → penalize harder)
    |mag_diff| * 1.0   otherwise
)

--- Compressed Complex Loss ---
P_comp = P_mag^c · exp(j · ∠P_spec)
C_comp = C_mag^c · exp(j · ∠C_spec)
comp_loss = L1(P_comp.real, C_comp.real) + L1(P_comp.imag, C_comp.imag)

--- Tap Regularization ---
tap_reg = mean(|taps|)   (L1 on filter tap magnitudes)

--- Total ---
total = mag_loss + comp_loss + 1e-4 · tap_reg
```

**Asymmetric magnitude penalty:** The 2× penalty for under-estimation directly
combats the common failure mode of speech muffling — where the model suppresses
both noise and speech together. This forces the network to err on the side of
allowing too much through rather than suppressing too much.

**Power compression (c=0.3):** Compresses the dynamic range of the spectrogram
before computing loss. This prevents loud frames from dominating the gradient and
ensures the model also fits quiet speech passages (whispers, fricatives).

**Tap regularization:** Prevents the K=5 complex FIR taps from growing arbitrarily
large. Encourages sparse, near-identity filtering rather than complex deconvolution.

#### `MultiResoSTFTLoss` — defined but unused

This class computes L1 log-magnitude loss across three STFT resolutions
`(512, 1024, 2048)`. It is defined in the notebook but **never called during
training or evaluation** — `HybridLoss` is the sole criterion used.

---

### 3.12 Training Infrastructure

#### Optimizer & scheduler

```python
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                              patience=4, min_lr=1e-6)
# scheduler.step(cur_pesq)  ← steps on PESQ, not on loss
```

The scheduler monitors PESQ in `max` mode — it halves LR if PESQ does not
improve for 4 consecutive epochs.

#### Gradient accumulation + AMP

```python
BATCH     = 4
GRAD_ACCUM = 4     # effective batch = 16
# Loss is divided by GRAD_ACCUM before .backward()
# scaler.step() is called every GRAD_ACCUM steps
# Gradient norm clipping: max_norm=5.0
```

Mixed precision (`torch.amp.autocast + GradScaler`) keeps VRAM usage low enough
to fit `batch=4 × seg=3.0s` inside 8 GB VRAM.

#### Early stopping

```python
EarlyStopping(patience=15, min_delta=1e-4)
# Monitors: -PESQ  (so it stops when PESQ stops improving)
# Restores best model state on trigger
```

#### ThreadedPrefetcher — Windows-safe data pipeline

On Windows, `DataLoader(num_workers > 0)` requires spawning new processes,
incurring 3–8 minutes of startup overhead per fold due to reimporting
torch/scipy/numpy. The `ThreadedPrefetcher` works around this by using Python
**threads** (which share memory) instead of processes:

```
DataLoader(num_workers=0)   ← single-process data fetching
    ↓
ThreadedPrefetcher(n_threads=2, queue_size=1)
    ↓
  Thread 0 & Thread 1 pull from shared iter(loader)
  GPU transfer via .to(device, non_blocking=True)
  Queued batches ready before GPU step completes
```

numpy/scipy release the Python GIL, enabling real parallel CPU-side augmentation
without spawn overhead.

#### 5-Fold Cross-Validation

```python
KFold(n_splits=5, shuffle=True, random_state=SEED)
```

Each fold:
- **Train set:** 80% of `train_val` split (~6,389 clean files)
- **Val set:** 20% of `train_val` split (~1,597 files) — fixed manifest
- **Test set:** holdout fold (~2,484 files) — fixed manifest
- Per-fold checkpoint saved to `checkpoints/fold_{i}/best.pth`
- Best fold selected by PESQ score

---

### 3.13 Evaluation & Metrics

`compute_val_metrics` evaluates up to `max_batches=20` validation batches and
computes the following metrics on the **first sample of each batch**:

| Metric | Library | What it measures |
|---|---|---|
| **PESQ** (WB, 1–4.5) | `pesq` | Perceptual quality (ITU-T P.862) |
| **DNSMOS OVR/SIG/BAK** | `speechmos` | DNS Challenge P.835 MOS estimates |
| **STOI** (0–1) | `pystoi` | Short-time objective intelligibility |
| **Cosine Similarity** | scipy/numpy | Normalized spectral vector similarity |
| **Pearson Correlation** | numpy | Linear spectral correlation |
| **Spectral SNR** (dB) | custom | Signal-to-noise in log-power spectrogram |
| **Spectral Distortion** | custom | MSE of log-power spectrogram |
| **MAE** | custom | Mean absolute error of log-power spectrogram |
| **HybridLoss** components | custom | mag, comp_complex, tap_reg, total |

All spectral metrics are computed on log-power spectrograms (`20·log10|STFT|`)
using `(n_fft=512, hop=160, win=400)`.

**Benchmark section** additionally measures:
- **Per-sample GPU latency** (ms) — measured with CUDA sync before/after
- **Real-Time Factor (RTF)** — inference time / audio duration; RTF < 1.0 = real-time

---

### 3.14 Export

```python
# Reparameterize MobileOne blocks first (fuse branches → single 3×3 conv)
model.reparameterize()
model = model.to(device)

# TorchScript trace
ts = torch.jit.trace(ExportWrapper(model), (example_wav,))
ts.save('hybrid_denoiser_best.ts')
```

The `ExportWrapper` strips the tap and hidden-state outputs so the traced model
has a clean `noisy_wav → denoised_wav` interface. Reparameterization must happen
**before** moving the model to GPU — the branch-fusion math runs on whichever
device the parameters are currently on.

---

## 4. Key Design Decisions & Novelties

### 4.1 Band-Split Deep Filter

The separation of the 257-bin spectrogram into:
- **Low band (0–5 kHz): complex FIR filtering** — models phase relationships critical for speech intelligibility
- **High band (5–8 kHz): magnitude-only masking** — avoids overfitting noisy phase above 5 kHz

This is adapted from DeepFilterNet's band-split idea but implemented here as a
continuous, differentiable, K=5 tap causal complex FIR filter without a
separate ERB filterbank.

### 4.2 Magnitude Mask Injection into Identity Tap

Rather than having two completely separate output heads, the magnitude mask is
**injected into the FIR identity tap position (index K-1)** of the low-band
taps. This means the coarse mask provides a strong initialization for the deep
filter, and the network can learn to refine it with the other K-1 taps.
This architectural coupling ensures both heads are always jointly optimized.

### 4.3 MobileOne Re-parameterization

Using MobileOneBlocks allows training with a rich multi-branch structure
(identity shortcut + scale branch + 2 conv branches) but deploying as a single
merged 3×3 convolution. This gives training stability without inference overhead.

### 4.4 Dry Target for Dereverberation

The training target is always the **anechoic dry signal**, but the noisy
input contains speech convolved with a room impulse response. This forces the
model to perform **joint dereverberation and denoising** without any explicit
dereverberation label or separate training stage.

### 4.5 Asymmetric Loss

The 2× penalty for under-prediction in the magnitude loss is a practical fix
for the well-known **over-suppression problem** in neural speech enhancement,
where networks learn that the safest strategy is to suppress everything and
accept quiet outputs. The asymmetric loss breaks this dynamic.

### 4.6 SNR Sampling Distribution

Training SNR is sampled from `Triangular(min=0, mode=15, max=40)` dB,
matching the DNS Challenge 4 distribution. This biases training toward
moderate-SNR conditions (most common in real recordings) while still covering
very noisy (0 dB) and nearly-clean (40 dB) conditions.

---

## 5. Dataset & Preprocessing

### Clean speech

- Source: custom collection, stored at `data/clean_16k/`
- Total files: **12,420 utterances**
- Format: 16 kHz, mono, 16-bit PCM WAV

### Noise

- Source: custom noise collection + UrbanSound8K filtered classes (0, 1, 3, 4, 5, 7, 8)
  - Excluded classes: children playing (2), dog bark handled separately, street music (6), drilling
- Total files: **9,288 noise recordings**
- Format: 16 kHz, mono, 16-bit PCM WAV

### Preprocessing script

A utility function `resample_dataset_fast` (commented out in notebook) converts
any supported format (`.wav/.flac/.ogg/.mp3/.m4a`) to 16 kHz mono WAV using
librosa, preserving folder structure. The UrbanSound variant additionally
filters by class ID embedded in the filename (`filename.split('-')[1]`).

---

## 6. Training Configuration

| Hyperparameter | Value | Note |
|---|---|---|
| Sample rate | 16 000 Hz | |
| Segment length | 3.0 s | ≥ 2.5 s required for reliable PESQ WB |
| Batch size | 4 | |
| Gradient accumulation | 4 | Effective batch = 16 |
| Steps per epoch | 625 | Not epoch over full dataset; fixed budget |
| Max epochs | 100 | Early stopping typically fires ~30–60 |
| Early stop patience | 15 epochs | Monitors −PESQ |
| Optimizer | AdamW | |
| Learning rate | From `configs/train.yaml` | |
| Weight decay | From `configs/train.yaml` | |
| LR schedule | ReduceLROnPlateau (max, ×0.5, pat=4) | Steps on PESQ |
| Gradient clip | `max_norm=5.0` | |
| AMP | Yes (torch.amp, CUDA) | |
| K-Folds | 5 | KFold, shuffle, seed=42 |
| RIR probability | From `configs/train.yaml` (`p_rir`) | ~0.35 |
| Time-varying SNR probability | From `configs/train.yaml` (`p_vary`) | ~0.40 |
| RIR cache size | 500 rooms | Pre-built at startup |
| Workers | 0 (Windows) + ThreadedPrefetcher(threads=2) | |

---

## 7. Evaluation Protocol

### Per-fold validation

After each epoch, up to 20 validation batches are evaluated using
`compute_val_metrics`. Checkpoints are saved whenever PESQ improves.
The learning rate scheduler also steps on the per-epoch PESQ.

### Cross-fold test evaluation

After all folds complete, each fold's `best.pth` checkpoint is re-loaded and
evaluated on its **held-out test split** (a separate manifest from the val split).
Results include mean and standard deviation across all 5 folds for all 10 metrics.

### Speed benchmark

200 test samples from fold 0 are processed with the reparameterized (fused) model.
GPU latency and RTF are measured with `torch.cuda.synchronize()` bracketing.

---

## 8. Incorrect or Misleading Comments Found in the Code

The following discrepancies were found between code comments and actual implementation:

---

**1. Pipeline diagram says "Complex Mask Head [tanh real+imag]"**

The actual `mask_mag_head` uses `nn.Sigmoid()` — not `tanh`. The output is a
single real-valued mask per bin, not a complex (real+imag) mask. The comment
incorrectly implies the mask is complex. The tap head (a separate module) is
where complex values are produced.

---

**2. Comment says "Boost complex loss by 10x so it fights evenly against magnitude loss"**

```python
# Boost complex loss by 10x so it fights evenly against the magnitude loss!
total = mag_loss + comp_loss + (self.tap_reg_weight * tap_reg)
```

There is **no 10× multiplier applied to `comp_loss`** in the actual computation.
The comment is a leftover from a previous experiment. The two losses are added
with equal weight (1.0 each).

---

**3. `_maybe_time_or_pitch_perturb` — claimed pitch shift, only time-stretch implemented**

The docstring says "Scipy-based time stretch" (correct), but the function
initializes the meta dict with `{"time_stretch": None, "pitch_shift": None}`
and **only ever sets `time_stretch`**. No pitch shifting is implemented anywhere.
The `pitch_shift` key is always `None` and the function name implies an operation
that does not exist in the code.

---

**4. `log_line` references `metrics['mrstft']` — key never exists**

```python
def log_line(fold_idx, epoch, metrics, lr, elapsed):
    return (f"... mrstft {metrics['mrstft']:.4f} ...")
```

`compute_val_metrics` never computes or returns an `'mrstft'` key.
`MultiResoSTFTLoss` is defined in the notebook but never used in training or
validation. If `log_line` were called directly, it would raise a `KeyError`.
(It is never directly called in the actual training loop — `epoch_bar.set_postfix`
is used instead — so this does not crash training, but it is dead/broken code.)

---

**5. `MultiResoSTFTLoss` is defined but never used**

The class is fully implemented and tested in isolation, but the actual training
loop exclusively uses `HybridLoss`. `MultiResoSTFTLoss` is effectively dead code
in this notebook version.

---

**6. Pipeline description says "DeepFilter K=5 Taps [tanh + L1-norm, causal]"**

The normalization is not L1. The code computes:
```python
tap_mag = sqrt(real² + imag² + ε)   # L2 complex magnitude
scale   = (2 × tanh(tap_mag)) / tap_mag
```
This is **tanh-bounded L2-magnitude normalization** (complex modulus). L1 norm
would be `sum(|real| + |imag|)` — that is not what is implemented.

---

## 9. Information Gaps to Fill for a Paper

The following information was not present in the code and would need to be
provided to write a complete research paper:

| Section | Missing Information Needed |
|---|---|
| **Abstract / Intro** | Target application domain (video calls, hearing aids, broadcast, etc.) |
| **Related Work** | Citation list: DeepFilterNet (Schröter et al.), MobileOne (Vasu et al.), Conformer (Gulati et al.), DNS Challenge baseline, SEGAN, DCCRN, FullSubNet |
| **Dataset** | Full name and citation for the clean speech dataset (LibriSpeech? DNS? Custom?) |
| **Dataset** | Full name and citation for the noise dataset (beyond UrbanSound8K; 9,288 files total) |
| **Dataset** | Any licenses or access restrictions |
| **Experiments** | Full numeric results table (PESQ, STOI, DNSMOS per fold + mean/std) |
| **Experiments** | Comparison against baselines (e.g., DeepFilterNet2, FullSubNet, classic Wiener filter) |
| **Experiments** | Ablation study: what happens when you remove each module (Conformer, TF attention, GRU, band-split)? |
| **Experiments** | Training time per fold (hours) and total compute |
| **Experiments** | Actual PESQ / STOI / DNSMOS scores achieved (only "1.664 PESQ" from fold 0 mentioned in the audio playback cell) |
| **Config** | Exact values of `lr` and `weight_decay` from `configs/train.yaml` |
| **Config** | Exact values of `p_rir` and `p_vary` from config |
| **Model** | Inference latency and RTF numbers from the benchmark run |
| **Conclusion** | Failure modes / qualitative analysis of when the model performs poorly |

---

## 10. Suggested Paper Outline

Given the information available, the following structure is recommended.

### Abstract
- Problem: speech enhancement for real-time applications
- Approach: hybrid MobileOne + Conformer + GRU + band-split DeepFilter
- Key contribution: band-split complex FIR, asymmetric compressed loss, dry-target dereverberation
- Results: PESQ=X.XX ± Y, STOI=X.XX ± Y, DNSMOS=X.XX ± Y on custom test set; RTF < 1.0

### 1. Introduction
- Speech enhancement problem statement
- Real-time constraint and consumer hardware motivation
- Gap: existing systems either too large (Conformer-heavy) or lack complex domain filtering
- Contributions list: (1) band-split DeepFilter; (2) asymmetric loss; (3) dry-target joint dereverberation; (4) MobileOne reparameterization for inference efficiency

### 2. Related Work
- Classical: Wiener filter, MMSE-STSA
- Deep masking: SEGAN, Conv-TasNet, FullSubNet
- Complex domain: DCCRN, DPCRN, DeepFilterNet 1/2
- Conformer-based: CTS-Net, Efficient Conformer
- Architecture efficiency: MobileOne

### 3. Methodology
- 3.1 System overview + full pipeline diagram (Section 2 of this document)
- 3.2 STFT front-end and feature normalization
- 3.3 MobileOne Encoder + reparameterization
- 3.4 Conformer Lite sequence model
- 3.5 TF Attention Block
- 3.6 GRU Refinement
- 3.7 Band-split complex FIR deep filter (key novelty)
- 3.8 Magnitude mask head + injection into identity tap
- 3.9 HybridLoss: asymmetric compressed magnitude + compressed complex + tap regularization

### 4. Experimental Setup
- Dataset details (clean + noise sources, counts, split strategy)
- Data augmentation: RIR, SNR distribution, device effects, hum, time-stretch
- 5-fold cross-validation protocol
- Training hyperparameters table
- Evaluation metrics: PESQ WB, STOI, DNSMOS P.835, SNR, RTF

### 5. Results
- Cross-fold metrics table (mean ± std)
- Comparison vs baselines
- Ablation study
- Speed benchmark (latency, RTF)
- Spectrogram visualizations

### 6. Conclusion
- Summary of contributions
- Limitations: single-channel only; moderate PESQ (if ~1.664); Windows-specific training workarounds
- Future work: multi-channel, causal streaming deployment, quantization for edge devices
