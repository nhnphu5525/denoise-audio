# Denoise Audio

> **Real-time speech denoising** using an IRM U-Net — Vietnamese speech, trained on VIVOS + DEMAND + MUSAN.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [1. Install dependencies](#1-install-dependencies)
  - [2. Download datasets](#2-download-datasets)
  - [3. Build preprocessed dataset](#3-build-preprocessed-dataset)
  - [4. Train](#4-train)
  - [5. Real-time inference](#5-real-time-inference)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## Overview

`denoise-audio` is a deep-learning pipeline that removes background noise from speech in real time.
The model processes **256 ms** audio windows at a time and emits a single denoised hop (8 ms) per
inference step, making it suitable for live microphone streaming.

| Property | Value |
|---|---|
| Model | IRM U-Net with asymmetric pooling |
| Input | log₁p-magnitude spectrogram (256 × 32 × 1) |
| Sample rate | 16 000 Hz |
| Algorithmic latency | **256 ms** |
| Output hop latency | **8 ms** per step |
| Language focus | Vietnamese (VIVOS corpus) + any noise from DEMAND/MUSAN |
| Framework | TensorFlow / Keras |

---

## Architecture

The model predicts an **Ideal Ratio Mask (IRM)** $M \in (0,1)$ and applies it directly to the noisy magnitude spectrogram:

$$\hat{S} = M\!\left(|Y|\right) \cdot |Y|$$

Key design decisions:

| Feature | Reason |
|---|---|
| Asymmetric pooling `(2,1)` | Preserves the time axis T=32 at every encoder depth |
| Squeeze-and-Excitation block | Channel-wise attention at bottleneck |
| `SpatialDropout2D(0.3)` | Regularisation on small Vietnamese dataset |
| Mask applied inside model | Loss computed on clean estimate, not raw mask |

See [docs/architecture.md](docs/architecture.md) for the full architecture diagram and parameter count.

---

## Project Structure

```
denoise-audio/
├── configs/
│   ├── data_config.yaml          # Dataset paths, SNR range, split ratios
│   └── train_config.yaml         # STFT params, model hyper-params, callbacks
│
├── data/
│   ├── raw/
│   │   ├── clean/vivos/          # VIVOS Vietnamese speech WAVs
│   │   └── noise/
│   │       ├── demand/           # DEMAND noise WAVs
│   │       └── musan/            # MUSAN music + noise WAVs
│   └── processed/
│       ├── train/                # .npz files: keys "noisy", "clean" (B, 256, 32, 1)
│       ├── val/
│       └── test/
│
├── docs/                         # Extended technical documentation
│   ├── architecture.md
│   ├── data_pipeline.md
│   ├── training.md
│   ├── realtime.md
│   └── contributing.md
│
├── models/
│   └── checkpoints/              # Saved .keras model weights
│
├── notebooks/
│   └── architecture_experiment.ipynb
│
├── scripts/
│   └── download_dataset.py       # Kaggle API dataset downloader
│
└── src/
    ├── audio/                    # Low-level audio I/O utilities
    ├── data/
    │   ├── preprocessing/        # WAV → (noisy, clean) spectrogram segments
    │   │   └── audio.py          ← AudioPreprocessor, build_dataset
    │   ├── postprocessing/       # Model output → PCM audio (real-time)
    │   │   └── audio.py          ← AudioPostprocessor
    │   └── dataset.py            # tf.data.Dataset loader (TODO)
    ├── model/
    │   └── unet.py               # UNetDenoiser, build_unet_denoise()
    ├── realtime/                 # Streaming inference utilities (TODO)
    └── training/
        └── metrics.py            # CombinedSpectralLoss, SI-SNR, PESQ, STOI
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Optional — perceptual evaluation metrics
pip install pesq pystoi
```

> Requires Python ≥ 3.10 and TensorFlow ≥ 2.15.

### 2. Download datasets

Requires a [Kaggle API token](https://www.kaggle.com/docs/api) saved at `~/.kaggle/kaggle.json`.

```bash
# Download all datasets defined in configs/data_config.yaml
python scripts/download_dataset.py

# Download individual dataset
python scripts/download_dataset.py --dataset vivos
python scripts/download_dataset.py --dataset demand
python scripts/download_dataset.py --dataset musan
```

| Dataset | Purpose | Size |
|---|---|---|
| [VIVOS](https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr) | Clean Vietnamese speech | ~700 MB |
| [DEMAND](https://www.kaggle.com/datasets/chrisfilo/demand) | Environmental noise (15 categories) | ~8 GB |
| [MUSAN](https://www.kaggle.com/datasets/dogrose/musan-dataset) | Music + noise (optional) | ~11 GB |

### 3. Build preprocessed dataset

Converts raw WAV pairs into `.npz` segment files used by the training loop:

```bash
# All splits (train / val / test)
python -m src.data.preprocessing.audio --split all --pairs-per-clean 3

# Single split
python -m src.data.preprocessing.audio --split train
```

Each `.npz` file contains:
- `noisy` — shape `(N, 256, 32, 1)` — model input
- `clean` — shape `(N, 256, 32, 1)` — training target

### 4. Train

```bash
# (training script coming soon — see docs/training.md)
python scripts/train.py --config configs/train_config.yaml
```

Default hyper-parameters (`configs/train_config.yaml`):

| Parameter | Value |
|---|---|
| Batch size | 16 |
| Epochs | 100 |
| Learning rate | 3 × 10⁻⁴ (Adam) |
| LR scheduler | ReduceLROnPlateau (patience=5) |
| Early stopping | patience=15 |
| Loss | 0.7 × SpectralL1 + 0.3 × SpectralConvergence |

### 5. Real-time inference

```python
import collections
import numpy as np
from src.data.preprocessing import AudioPreprocessor
from src.data.postprocessing import AudioPostprocessor
import tensorflow as tf

model = tf.keras.models.load_model("models/checkpoints/best.keras")
pre   = AudioPreprocessor.from_configs()
post  = AudioPostprocessor.from_configs()

buffer = collections.deque(maxlen=32)   # 32-frame ring buffer

for hop_samples in audio_stream:        # stream 128 samples (8 ms) at a time
    buffer.append(hop_samples)
    if len(buffer) < 32:
        continue

    waveform         = np.concatenate(buffer)
    log_mag, noisy_stft = pre.compute_magnitude(waveform)

    clean_est = model.predict(
        log_mag[np.newaxis, ..., np.newaxis], verbose=0
    )[0, ..., 0]                        # (256, 32)

    pcm_frame = post.reconstruct_frame(clean_est, noisy_stft)  # (128,) float32
    speaker.write(pcm_frame)
```

See [docs/realtime.md](docs/realtime.md) for detailed streaming integration.

---

## Datasets

| Dataset | ID | Categories | SR | License |
|---|---|---|---|---|
| VIVOS | `vivos` | Vietnamese read speech | 16 kHz | CC BY-SA 4.0 |
| DEMAND | `demand` | 15 noise environments (transport, office, outdoor, …) | 16 / 48 kHz | CC BY-SA 3.0 |
| MUSAN | `musan` | Music, speech, noise | 16 kHz | CC BY 4.0 |

All datasets are downloaded via the Kaggle API and placed under `data/raw/` according to
paths defined in `configs/data_config.yaml`.

---

## Configuration

All hyper-parameters are centralised in two YAML files:

| File | Controls |
|---|---|
| [`configs/data_config.yaml`](configs/data_config.yaml) | Dataset paths, SNR range `[-5, 20]` dB, train/val/test split ratios |
| [`configs/train_config.yaml`](configs/train_config.yaml) | STFT params, model shape, optimizer, LR schedule, callbacks |

> **Important:** `stft.*` values in `train_config.yaml` must stay in sync between preprocessing and
> postprocessing. Changing `n_fft` or `hop_length` requires rebuilding the dataset.

---

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | U-Net architecture, SE block, asymmetric pooling, parameter counts |
| [docs/data_pipeline.md](docs/data_pipeline.md) | Full math for STFT, SNR mixing, log1p, segmentation, iSTFT |
| [docs/training.md](docs/training.md) | Training loop, loss functions, metrics (SI-SNR, PESQ, STOI) |
| [docs/realtime.md](docs/realtime.md) | Sliding-buffer streaming design, latency analysis |
| [docs/contributing.md](docs/contributing.md) | How to implement `# TODO` blocks, code style, PR workflow |
| [src/data/TECHNICAL.md](src/data/TECHNICAL.md) | Detailed math reference for preprocessing / postprocessing |
| [src/model/README.md](src/model/README.md) | Model architecture technical reference |

---

## Contributing

This repository uses skeleton `# TODO` stubs — see [docs/contributing.md](docs/contributing.md)
for the implementation guide and development workflow.

```
src/data/preprocessing/audio.py   — 11 TODOs (AudioPreprocessor + build_dataset)
src/data/postprocessing/audio.py  —  6 TODOs (AudioPostprocessor)
src/data/dataset.py               — tf.data.Dataset loader
src/realtime/                     — streaming inference utilities
scripts/train.py                  — training entry-point
```
