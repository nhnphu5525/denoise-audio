# Contributing Guide

Welcome to the `denoise-audio` project. This guide explains how to implement
the `# TODO` skeleton stubs and contribute to the codebase.

---

## Table of Contents

1. [Development Setup](#1-development-setup)
2. [Repository Structure](#2-repository-structure)
3. [Work Assignment — TODO Map](#3-work-assignment--todo-map)
   - [Module A — Preprocessing (11 TODOs)](#module-a--preprocessing-11-todos)
   - [Module B — Postprocessing (6 TODOs)](#module-b--postprocessing-6-todos)
   - [Module C — Dataset Loader (new file)](#module-c--dataset-loader-new-file)
   - [Module D — Training Script (new file)](#module-d--training-script-new-file)
4. [How to Implement a TODO Block](#4-how-to-implement-a-todo-block)
5. [Testing Your Implementation](#5-testing-your-implementation)
6. [Code Style](#6-code-style)
7. [Pull Request Checklist](#7-pull-request-checklist)

---

## 1. Development Setup

```bash
# Clone
git clone https://github.com/<your-org>/denoise-audio.git
cd denoise-audio

# Create isolated environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pesq pystoi          # optional perceptual metrics

# Download data (requires ~/.kaggle/kaggle.json)
python scripts/download_dataset.py

# Verify setup — should raise NotImplementedError (expected)
python -c "from src.data.preprocessing import AudioPreprocessor; print('imports OK')"
```

---

## 2. Repository Structure

```
src/
├── data/
│   ├── TECHNICAL.md              ← Math reference — READ THIS FIRST
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── audio.py              ← Module A  (11 TODOs)
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   └── audio.py              ← Module B  (6 TODOs)
│   └── dataset.py                ← Module C  (new file)
├── model/
│   ├── README.md                 ← Architecture reference
│   └── unet.py                   ← Complete (do not modify)
├── realtime/                     ← Future work
└── training/
    └── metrics.py                ← Complete (do not modify)

docs/
├── architecture.md               ← Model architecture guide
├── data_pipeline.md              ← Data pipeline guide
├── training.md                   ← Training guide
├── realtime.md                   ← Streaming inference guide
└── contributing.md               ← This file

configs/
├── data_config.yaml              ← Dataset paths & preprocessing params
└── train_config.yaml             ← STFT params, model shape, training config
```

---

## 3. Work Assignment — TODO Map

### Module A — Preprocessing (11 TODOs)

**File:** `src/data/preprocessing/audio.py`  
**Reference:** [`src/data/TECHNICAL.md`](../src/data/TECHNICAL.md) §2, [`docs/data_pipeline.md`](data_pipeline.md) §2

| # | Method | Key concept |
|---|---|---|
| A1 | `AudioPreprocessor.load()` | `librosa.load()` with resample |
| A2 | `AudioPreprocessor.peak_normalize()` | Scale to unit peak amplitude |
| A3 | `AudioPreprocessor.mix_at_snr()` | SNR formula: `scale = RMS_clean / (10^(SNR/20) × RMS_noise)` |
| A4 | `AudioPreprocessor.random_mix()` | Sample SNR from `snr_range_db`, call `mix_at_snr` |
| A5 | `AudioPreprocessor.stft()` | `librosa.stft()` with config params |
| A6 | `AudioPreprocessor.compute_magnitude()` | `|stft|` → Nyquist crop → `log1p` |
| A7 | `AudioPreprocessor.slice_spectrogram()` | Sliding window → `(N, F, T, 1)` |
| A8 | `AudioPreprocessor.process_pair()` | Compose all steps A1→A7 |
| A9 | `AudioPreprocessor._rms()` | `sqrt(mean(x²) + 1e-10)` |
| A10 | `AudioPreprocessor._fit_length()` | Trim or tile noise to target length |
| A11 | `build_dataset()` | File I/O loop → save `.npz` files |

### Module B — Postprocessing (6 TODOs)

**File:** `src/data/postprocessing/audio.py`  
**Reference:** [`src/data/TECHNICAL.md`](../src/data/TECHNICAL.md) §3, [`docs/data_pipeline.md`](data_pipeline.md) §3

| # | Method | Key concept |
|---|---|---|
| B1 | `AudioPostprocessor.decompress()` | `np.expm1(log_mag) − ε`, clip ≥ 0 |
| B2 | `AudioPostprocessor.restore_nyquist()` | `vstack([mag, mag[-1:,:]])` → `(257, T)` |
| B3 | `AudioPostprocessor.reconstruct()` | Phase substitution → `librosa.istft()` |
| B4 | `AudioPostprocessor.reconstruct_frame()` | Call B3, emit last hop only |
| B5 | `AudioPostprocessor.reconstruct_griffin_lim()` | `librosa.griffinlim()` |
| B6 | `AudioPostprocessor._peak_normalize()` | Same formula as A2 |

### Module C — Dataset Loader (new file)

**File:** `src/data/dataset.py`  
**Reference:** [`docs/training.md`](training.md) §2

Implement a `tf.data.Dataset` loader that reads `.npz` files from `data/processed/<split>/`
and produces `(noisy, clean)` tensor pairs for the training loop.

**Expected API:**

```python
from src.data.dataset import build_tf_dataset

train_ds = build_tf_dataset(split="train", batch_size=16, shuffle=True)
val_ds   = build_tf_dataset(split="val",   batch_size=16, shuffle=False)

# Each element: (noisy, clean) — both shape (B, 256, 32, 1), dtype float32
```

**Implementation hints:**

```python
import tensorflow as tf
from pathlib import Path

def build_tf_dataset(split, batch_size=16, shuffle=True, prefetch=tf.data.AUTOTUNE):
    npz_files = sorted(Path(f"data/processed/{split}").glob("*.npz"))

    def load_npz(path):
        # tf.numpy_function wrapping np.load
        noisy, clean = tf.numpy_function(
            lambda p: _load_pair(p.numpy().decode()),
            [path], (tf.float32, tf.float32),
        )
        return noisy, clean

    ds = tf.data.Dataset.from_tensor_slices([str(f) for f in npz_files])
    ds = ds.flat_map(lambda p: tf.data.Dataset.from_tensors(load_npz(p)))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    return ds.batch(batch_size).prefetch(prefetch)
```

### Module D — Training Script (new file)

**File:** `scripts/train.py`  
**Reference:** [`docs/training.md`](training.md) §8

Wire together the model, loss, optimizer, and callbacks into a complete `model.fit()` loop.

**Expected usage:**

```bash
python scripts/train.py --config configs/train_config.yaml
```

**Skeleton:**

```python
import argparse
import yaml
import tensorflow as tf
from src.model.unet import build_unet_denoise
from src.training.metrics import CombinedSpectralLoss
from src.data.dataset import build_tf_dataset

def train(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = build_unet_denoise(
        input_shape  = tuple(cfg["model"]["input_shape"]),
        base_filters = cfg["model"]["base_filters"],
        dropout_rate = cfg["model"]["dropout_rate"],
    )

    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate = cfg["training"]["learning_rate"],
            clipnorm      = cfg["training"]["gradient_clip_norm"],
        ),
        loss = CombinedSpectralLoss(alpha=cfg["loss"]["alpha"]),
    )

    train_ds = build_tf_dataset("train", batch_size=cfg["training"]["batch_size"])
    val_ds   = build_tf_dataset("val",   batch_size=cfg["training"]["batch_size"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(**cfg["callbacks"]["early_stopping"]),
        tf.keras.callbacks.ModelCheckpoint(**cfg["callbacks"]["model_checkpoint"]),
        tf.keras.callbacks.TensorBoard(**cfg["callbacks"]["tensorboard"]),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            patience = cfg["training"]["lr_patience"],
            factor   = cfg["training"]["lr_factor"],
            min_lr   = cfg["training"]["min_lr"],
        ),
    ]

    model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = cfg["training"]["epochs"],
        callbacks       = callbacks,
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    train(ap.parse_args().config)
```

---

## 4. How to Implement a TODO Block

Each `# TODO` comment contains a numbered list of steps. Follow them precisely.

**Before you start:**

1. Read the math in [`src/data/TECHNICAL.md`](../src/data/TECHNICAL.md) for the function you are implementing.
2. Check the docstring directly above the `# TODO` — it documents the expected input/output shapes and types.
3. Look at the test hints in §5 below.

**Example — implementing `peak_normalize` (TODO A2):**

```python
# BEFORE (skeleton)
@staticmethod
def peak_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale waveform so max absolute amplitude = 1."""
    # TODO: Scale the waveform so that its peak absolute amplitude equals 1.
    #   Formula:  x_norm = x / (max|x| + ε)
    raise NotImplementedError

# AFTER (your implementation)
@staticmethod
def peak_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale waveform so max absolute amplitude = 1."""
    peak = np.max(np.abs(audio))
    return audio / (peak + eps)
```

**Rules:**
- ✅ Remove `raise NotImplementedError` only when the entire function is implemented
- ✅ Do not change function signatures (name, parameters, return type)
- ✅ Do not import new packages without updating `requirements.txt`
- ❌ Do not modify `src/model/unet.py` or `src/training/metrics.py`

---

## 5. Testing Your Implementation

No formal test framework is configured yet. Use these manual smoke tests:

### Test A — Preprocessing single pair

```python
from src.data.preprocessing import AudioPreprocessor

pre = AudioPreprocessor.from_configs()

# Requires at least one clean and one noise file to be downloaded
noisy_segs, clean_segs = pre.process_pair(
    clean_path = "data/raw/clean/vivos/vivos/train/waves/VIVOSBK01/VIVOSBK01_1001.wav",
    noise_path = "data/raw/noise/demand/DKITCHEN_16k/DKITCHEN/ch01.wav",
)

assert noisy_segs.shape[1:] == (256, 32, 1), f"Wrong shape: {noisy_segs.shape}"
assert noisy_segs.dtype == np.float32
assert np.all(np.isfinite(noisy_segs)), "NaN or Inf in noisy_segs"
print(f"OK — {noisy_segs.shape[0]} segments")
```

### Test B — Round-trip (pre → post)

```python
import numpy as np
from src.data.preprocessing  import AudioPreprocessor
from src.data.postprocessing import AudioPostprocessor

pre  = AudioPreprocessor.from_configs()
post = AudioPostprocessor.from_configs()

audio = pre.load("data/raw/clean/vivos/vivos/train/waves/VIVOSBK01/VIVOSBK01_1001.wav")
log_mag, noisy_stft = pre.compute_magnitude(audio)

# Simulate model output = pass-through (identity mask ≈ 1)
fake_model_out = log_mag[:, :32]    # first segment only

waveform = post.reconstruct(fake_model_out, noisy_stft[:, :32])
assert len(waveform) > 0
assert np.all(np.isfinite(waveform))
print(f"OK — reconstructed {len(waveform)} samples ({len(waveform)/16000:.2f} s)")
```

### Test C — Dataset builder

```python
from src.data.preprocessing import build_dataset

build_dataset(split="val", pairs_per_clean=1)

from pathlib import Path
files = list(Path("data/processed/val").glob("*.npz"))
assert len(files) > 0, "No .npz files written"

import numpy as np
d = np.load(files[0])
assert "noisy" in d and "clean" in d
assert d["noisy"].shape[1:] == (256, 32, 1)
print(f"OK — {len(files)} .npz files, each {d['noisy'].shape}")
```

---

## 6. Code Style

- **Python ≥ 3.10** — use `str | Path` union types, `match` where appropriate
- **Type hints** — all public functions must have full annotations
- **Docstrings** — Google-style, include `Parameters` and `Returns` sections
- **Formatting** — run `black src/` before committing
- **Imports** — standard library → third-party → local; sorted within each group
- **Logging** — use `logger = logging.getLogger(__name__)`, not `print()`

---

## 7. Pull Request Checklist

Before opening a PR:

- [ ] All implemented TODOs have `raise NotImplementedError` removed
- [ ] Smoke tests A, B, C pass (or whichever apply to your module)
- [ ] No new `print()` statements — use `logger.info/debug/warning`
- [ ] `requirements.txt` updated if new packages were added
- [ ] Self-reviewed diff for unintended changes to other files
- [ ] Branch name follows convention: `feat/<module>-<description>` or `fix/<description>`
