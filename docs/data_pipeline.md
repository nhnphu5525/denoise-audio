# Data Pipeline

> **Modules:** `src/data/preprocessing/` and `src/data/postprocessing/`  
> For the full mathematical derivations of each step, see [`src/data/TECHNICAL.md`](../src/data/TECHNICAL.md).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
   - 2.1 [PreprocessConfig](#21-preprocessconfig)
   - 2.2 [AudioPreprocessor API](#22-audiopreprocessor-api)
   - 2.3 [Step-by-Step Pipeline](#23-step-by-step-pipeline)
3. [Postprocessing Pipeline](#3-postprocessing-pipeline)
   - 3.1 [PostprocessConfig](#31-postprocessconfig)
   - 3.2 [AudioPostprocessor API](#32-audiopostprocessor-api)
   - 3.3 [Reconstruction Methods](#33-reconstruction-methods)
4. [Dataset Builder](#4-dataset-builder)
5. [Data Shapes Reference](#5-data-shapes-reference)
6. [Configuration Reference](#6-configuration-reference)

---

## 1. Overview

```
                ┌────────────── TRAINING ──────────────────┐
                │                                           │
 clean.wav ─────┤                                  Y (clean segments)
                │   load → normalise → mix at SNR           │
 noise.wav ─────┤   → STFT → |mag| → log1p → slice  ──► model
                │                                  X (noisy segments)
                └──────────────────────────────────────────┘

                ┌────────────── REAL-TIME ─────────────────┐
                │                                           │
 microphone ───►│   sliding buffer (32 frames)              │
                │   compute_magnitude()                     │
                │         ↓  model.predict()                │
                │   reconstruct_frame()  ←  model output    │
                │         ↓                                 │
 speaker ◄──────│   PCM (128 samples, 8 ms)                 │
                └──────────────────────────────────────────┘
```

The preprocessing module converts raw audio pairs into `(noisy, clean)` spectrogram segments
saved as `.npz` files. The postprocessing module converts model outputs back to PCM audio
**in-memory** (no file writing) for real-time streaming.

---

## 2. Preprocessing Pipeline

### 2.1 PreprocessConfig

All parameters are loaded from the YAML configs. Use `PreprocessConfig.from_configs()` to
read them automatically:

```python
from src.data.preprocessing import PreprocessConfig

cfg = PreprocessConfig.from_configs(
    data_cfg_path="configs/data_config.yaml",
    train_cfg_path="configs/train_config.yaml",
)
```

| Field | Default | Source |
|---|---|---|
| `sample_rate` | 16000 | `data_config.yaml → preprocessing.target_sample_rate` |
| `n_fft` | 512 | `train_config.yaml → stft.n_fft` |
| `hop_length` | 128 | `train_config.yaml → stft.hop_length` |
| `win_length` | 512 | `train_config.yaml → stft.win_length` |
| `window` | `"hann"` | `train_config.yaml → stft.window` |
| `snr_range_db` | `(-5.0, 20.0)` | `data_config.yaml → preprocessing.snr_range_db` |
| `segment_frames` | 32 | `train_config.yaml → model.input_shape[1]` |
| `segment_step` | 16 | hardcoded (50% overlap) |
| `freq_bins` | 256 | derived: `n_fft // 2` |

### 2.2 AudioPreprocessor API

```python
from src.data.preprocessing import AudioPreprocessor

pre = AudioPreprocessor.from_configs()

# --- Individual steps ---
audio        = pre.load("path/to/file.wav")          # (N,) float32
audio_norm   = pre.peak_normalize(audio)
noisy, snr   = pre.random_mix(clean, noise)
log_mag, stft_cx = pre.compute_magnitude(noisy)       # (256,T), (257,T)
segments     = pre.slice_spectrogram(log_mag)         # (N, 256, 32, 1)

# --- End-to-end pair ---
noisy_segs, clean_segs = pre.process_pair("speech.wav", "noise.wav")
# → both shapes (N, 256, 32, 1)

# --- Specific SNR ---
noisy_segs, clean_segs = pre.process_pair("speech.wav", "noise.wav", snr_db=5.0)
```

### 2.3 Step-by-Step Pipeline

| Step | Method | Input → Output |
|---|---|---|
| 1 — Load | `load(path)` | WAV file → `(N,)` float32 @ 16 kHz |
| 2 — Normalise | `peak_normalize(audio)` | `(N,)` → `(N,)`, peak=1 |
| 3 — Mix | `mix_at_snr(clean, noise, snr_db)` | two `(N,)` → `(N,)` noisy |
| 4 — STFT | `stft(audio)` | `(N,)` → `(257, T)` complex |
| 5 — Magnitude + crop | `compute_magnitude(audio)` | `(N,)` → `(256, T)` log-mag + `(257, T)` STFT |
| 6 — Slice | `slice_spectrogram(spec)` | `(256, T)` → `(N_segs, 256, 32, 1)` |

**SNR mixing math:**

$$\text{scale} = \frac{\text{RMS}_{clean}}{10^{\,\text{SNR}_{dB}/20} \cdot \text{RMS}_{noise}}, \quad y = x_{clean} + \text{scale} \cdot x_{noise}$$

**log1p compression:**

$$\hat{M}[k,t] = \log\!\bigl(1 + |X[k,t]| + \varepsilon\bigr), \quad \varepsilon = 10^{-8}$$

---

## 3. Postprocessing Pipeline

The postprocessor is **stateless** and performs all operations in memory — no file I/O.

### 3.1 PostprocessConfig

```python
from src.data.postprocessing import PostprocessConfig

cfg = PostprocessConfig.from_configs()
```

Mirrors the same STFT parameters as `PreprocessConfig` (must stay in sync).

### 3.2 AudioPostprocessor API

```python
from src.data.postprocessing import AudioPostprocessor

post = AudioPostprocessor.from_configs()
```

### 3.3 Reconstruction Methods

Three reconstruction strategies are provided:

#### `reconstruct(clean_estimate, noisy_stft)` — Standard (offline + streaming)

```python
# clean_estimate : (256, 32)  log1p-compressed magnitude from model
# noisy_stft     : (257, 32)  complex STFT from AudioPreprocessor.compute_magnitude()
waveform = post.reconstruct(clean_estimate, noisy_stft)   # (N,) float32
```

**Steps internally:**
1. `decompress`: `expm1(log_mag) − ε`, clip ≥ 0 → linear magnitude
2. `restore_nyquist`: mirror last bin → `(257, T)`
3. Phase substitution: `Ŝ = |mag| · exp(j·∠noisy_stft)`
4. iSTFT (overlap-add) → waveform

#### `reconstruct_frame(clean_estimate, noisy_stft, output_frame_idx=-1)` — Real-time

```python
# Emits only the last hop (8 ms) for streaming
pcm_frame = post.reconstruct_frame(clean_est, noisy_stft)   # (128,) float32
```

Reconstructs the full 256 ms window internally but returns only `hop_length=128`
samples (8 ms) — the newest denoised frame.

#### `reconstruct_griffin_lim(clean_estimate, n_iter=32)` — Phase-free fallback

```python
# Use when noisy_stft is unavailable (offline evaluation from .npz files)
waveform = post.reconstruct_griffin_lim(clean_est, n_iter=32)
```

Uses the Griffin-Lim iterative phase estimation algorithm.
Quality increases with `n_iter` but so does compute cost.

---

## 4. Dataset Builder

`build_dataset()` processes all audio pairs and writes `.npz` files to disk:

```bash
# Build all three splits
python -m src.data.preprocessing.audio --split all --pairs-per-clean 3

# Single split
python -m src.data.preprocessing.audio --split train --pairs-per-clean 5
```

```python
# Programmatic usage
from src.data.preprocessing import build_dataset

build_dataset(split="train", pairs_per_clean=3, seed=42)
```

**Output format:**

```
data/processed/train/batch_000000.npz
    noisy  →  (N, 256, 32, 1)  float32   model input
    clean  →  (N, 256, 32, 1)  float32   training target
```

**Augmentation strategy:** For each clean speech file, sample `pairs_per_clean`
noise files at random. Total training pairs = `n_clean_files × pairs_per_clean`.

**Train/val split:** VIVOS only provides `train/` and `test/` subsets.
The `train/` files are further split by ratio from `data_config.yaml`:

$$N_{val} = \left\lfloor N_{train} \cdot \frac{r_{val}}{r_{train} + r_{val}} \right\rfloor$$

---

## 5. Data Shapes Reference

### Preprocessing — shape at each stage

```
raw WAV                           (N_samples,)       float32
after load() + resample           (N_16k,)           float32
after peak_normalize()            (N_16k,)           float32
after mix_at_snr()                (N_16k,)           float32
after stft()                      (257, T)           complex64
after |magnitude| + Nyquist crop  (256, T)           float32
after log1p()                     (256, T)           float32
after slice_spectrogram()         (N, 256, 32, 1)    float32
```

### Postprocessing — shape at each stage

```
model output                      (256, 32)          float32
after decompress()                (256, 32)          float32
after restore_nyquist()           (257, 32)          float32
after phase_substitution          (257, 32)          complex64
after istft()                     (N_samples,)       float32
after reconstruct_frame()         (128,)             float32
```

---

## 6. Configuration Reference

```yaml
# configs/data_config.yaml
preprocessing:
  target_sample_rate: 16000
  mono: true
  normalize: true
  snr_range_db: [-5, 20]        # SNR range for noise mixing (dB)
  clip_duration_sec: 3.0

split:
  train: 0.80
  val:   0.10
  test:  0.10
  seed:  42

# configs/train_config.yaml
stft:
  n_fft:       512              # Must match on both pre and post sides
  hop_length:  128
  win_length:  512
  window:      hann
  center:      true

model:
  input_shape: [256, 32, 1]     # [F, T, C]
```

> **Warning:** `n_fft` and `hop_length` must be identical between `AudioPreprocessor`
> and `AudioPostprocessor`. Changing them requires rebuilding all `.npz` files.
