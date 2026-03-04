# Real-Time Inference

---

## Table of Contents

1. [Overview](#1-overview)
2. [Latency Budget](#2-latency-budget)
3. [Sliding Buffer Design](#3-sliding-buffer-design)
4. [Complete Streaming Example](#4-complete-streaming-example)
5. [Frame-Rate Analysis](#5-frame-rate-analysis)
6. [Integration Notes](#6-integration-notes)

---

## 1. Overview

The model is designed for **causal, low-latency** streaming:

- Input: a sliding window of **32 spectrogram frames** = 256 ms
- Output: **1 denoised PCM frame** = 128 samples = 8 ms, emitted per inference step
- The window shifts by **1 frame** (8 ms) each step

This means the model always has 256 ms of audio context but delivers output with
only 8 ms additional latency beyond the inherent 256 ms algorithmic delay.

---

## 2. Latency Budget

| Component | Duration | Formula |
|---|---|---|
| Frame duration (1 hop) | 8 ms | `hop_length / sr = 128 / 16000` |
| Buffer window | 256 ms | `buffer_frames × hop_duration = 32 × 8 ms` |
| **Algorithmic latency** | **256 ms** | Buffer fill time (one-time startup cost) |
| **Per-step output latency** | **8 ms** | One hop emitted per inference step |
| Model inference time | depends on hardware | Typically 2–10 ms on CPU |

```
Timeline:
─────────────────────────────────────────────────────────────────────────►  time
│← 8ms →│← 8ms →│← 8ms →│ ···  (32 hops = 256 ms) ···  │← 8ms →│← 8ms →│
  hop 0   hop 1   hop 2  ···                              hop 31   hop 32

At step t=32:  buffer = [hop_1, hop_2, ..., hop_32]
               model infers → emit denoised hop_32 (last frame)

At step t=33:  buffer = [hop_2, hop_3, ..., hop_33]
               model infers → emit denoised hop_33
```

---

## 3. Sliding Buffer Design

```python
import collections
import numpy as np

HOP      = 128          # samples per frame
BUFFER_T = 32           # frames in the sliding window
SR       = 16000

buffer = collections.deque(maxlen=BUFFER_T)
```

The `deque` with `maxlen=BUFFER_T` automatically **discards the oldest frame**
when a new one is appended — no manual index management needed.

**Buffer → waveform:**

```python
waveform = np.concatenate(list(buffer))   # (32 × 128,) = (4096,) samples
```

**Why keep the full noisy STFT?**

The postprocessor needs the noisy STFT's phase for reconstruction
(`reconstruct_frame` uses phase substitution). `compute_magnitude` returns both
the log-magnitude (model input) and the complex STFT (phase source).

---

## 4. Complete Streaming Example

```python
import collections
import numpy as np
import tensorflow as tf
from src.data.preprocessing  import AudioPreprocessor
from src.data.postprocessing import AudioPostprocessor

# --- Setup (done once) ---
model = tf.keras.models.load_model("models/checkpoints/best.keras")
pre   = AudioPreprocessor.from_configs()
post  = AudioPostprocessor.from_configs()

HOP      = pre.cfg.hop_length    # 128
BUFFER_T = 32                    # must equal model input_shape[1]

buffer = collections.deque(maxlen=BUFFER_T)

# --- Per-frame streaming loop ---
def process_hop(raw_hop: np.ndarray) -> np.ndarray:
    """
    Process one incoming audio frame (128 samples, 8 ms).

    Parameters
    ----------
    raw_hop : (128,) float32  — new audio from microphone

    Returns
    -------
    pcm : (128,) float32  — denoised audio, or silence if buffer not filled yet
    """
    buffer.append(raw_hop.astype(np.float32))

    if len(buffer) < BUFFER_T:
        return np.zeros(HOP, dtype=np.float32)   # silence during startup

    # 1. Build waveform from buffer
    waveform = np.concatenate(list(buffer))       # (4096,) float32

    # 2. Preprocessing — magnitude spectrogram + retain noisy phase
    log_mag, noisy_stft = pre.compute_magnitude(waveform)
    # log_mag    : (256, 32)  float32
    # noisy_stft : (257, 32)  complex64

    # 3. Model inference — predict clean magnitude estimate
    x         = log_mag[np.newaxis, ..., np.newaxis]           # (1, 256, 32, 1)
    clean_est = model.predict(x, verbose=0)[0, ..., 0]         # (256, 32)

    # 4. Postprocessing — emit only the last hop (most recently denoised 8 ms)
    pcm = post.reconstruct_frame(
        clean_est,
        noisy_stft,
        output_frame_idx=-1,    # last frame = newest audio
    )
    return pcm   # (128,) float32
```

---

## 5. Frame-Rate Analysis

To sustain real-time operation, the model must process one 256 ms window **before**
the next 8 ms hop arrives. The required throughput:

$$\text{Required inference time} < 8\,\text{ms} \quad \text{(one hop duration)}$$

Alternatively, inference can be pipelined with audio I/O on separate threads.

**Practical throughput estimates:**

| Hardware | Approx. inference time | Real-time? |
|---|---|---|
| Modern CPU (i7/i9) | ~5–15 ms | Marginal — use threading |
| CPU + TF-Lite quantised | ~2–5 ms | Yes |
| GPU (RTX 3060+) | < 1 ms | Yes |
| Raspberry Pi 4 | ~80–200 ms | No — use smaller model |

**Reducing latency:**
- Set `buffer_frames` < 32 (trade-off: less context → potentially worse quality)
- Quantise the model with TensorFlow Lite (`float16` or `int8`)
- Use `model(x, training=False)` instead of `model.predict()` in the hot path

---

## 6. Integration Notes

### Realtime config block

```yaml
# configs/train_config.yaml → [realtime]
realtime:
  buffer_frames:    32     # must equal model input_shape[1]
  step_frames:      1      # hops shifted per step (1 = every new 8 ms)
  output_frame_idx: -1     # -1 = last (newest) frame
```

### Input normalization

The preprocessing module **peak-normalises during training** but for real-time
streaming you may want to apply a **running RMS normaliser** instead to handle
microphone level drift:

```python
rms   = np.sqrt(np.mean(raw_hop ** 2) + 1e-8)
scale = 0.1 / (rms + 1e-8)     # target RMS = 0.1
normalised_hop = raw_hop * np.clip(scale, 0.1, 10.0)
```

### Thread-safe streaming

For production use, run inference in a dedicated worker thread:

```python
import threading, queue

audio_in  = queue.Queue(maxsize=4)   # raw hops from microphone callback
audio_out = queue.Queue(maxsize=4)   # denoised hops to speaker callback

def inference_worker():
    while True:
        hop = audio_in.get()
        pcm = process_hop(hop)
        audio_out.put(pcm)

threading.Thread(target=inference_worker, daemon=True).start()
```
