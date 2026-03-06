# MobileDeepFilterNet — Real-time Speech Denoising

> Lightweight, causal speech enhancement using a MobileOne-like encoder + GRU temporal module + hybrid Mask + Deep-Filter decoder. Designed for real-time, low-latency inference on CPU/mobile.

---

## Table of Contents

1. Overview
2. Architecture (diagram + detailed blocks)
3. Shapes and recommended hyperparameters
4. Data pipeline and training
5. Streaming inference & deployment
6. Performance targets and evaluation
7. Experiments / Ablations to try
8. File layout and how to run
9. Contribution & license

---

## 1. Overview

**MobileDeepFilterNet** is a modified DeepFilterNet tailored for low-latency, real-time speech denoising on CPU/embedded hardware. Key ideas:

* Use STFT as front-end (complex spectrogram representation).
* Use a MobileOne-inspired lightweight convolutional encoder to reduce computation and memory bandwidth.
* Use a causal temporal module (GRU by default) to capture speech continuity.
* Output two decoder heads: a coarse spectral mask and per-frequency deep-filter taps (hybrid approach).
* Apply causal deep filtering per frequency over a small number of past frames and multiply by mask for robust suppression.

This README documents the architecture, training guidance, streaming inference loop, and tips for deployment and ablation experiments.

---

## 2. Architecture

### ASCII diagram (causal, realtime)

```
AUDIO IN (16 kHz)
      │
      ▼
Frameing (320 samples = 20 ms; hop = 160 samples = 10 ms)
      │
      ▼
STFT per hop (n_fft=320 -> freq_bins=161)  ---> complex X(f,t)
      │
      ▼
Feature extraction & ring buffer (K_ctx frames, causal)
  - log-power magnitude per bin
  - optional ERB pooled bands
      │
      ▼
MobileDeepFilterNet model (causal)
  ┌────────────────────────────────────────────────────────────┐
  │ Encoder (MobileOne-like blocks)                            │
  │  - Depthwise separable convs + pointwise convs             │
  │  - BN + activation                                          │
  │                                                            │
  │ Temporal Module (causal GRU / or TCN)                      │
  │  - learns inter-frame dependencies                         │
  │                                                            │
  │ Decoder heads (two outputs)                                │
  │  1) Mask head -> spectral mask (per-freq)                  │
  │  2) Deep-filter head -> K_tap taps per-frequency (real/imag if complex)   │
  └────────────────────────────────────────────────────────────┘
      │
      ▼
Apply filtering per frequency (causal):
  Y(f,t) = mask(f,t) * sum_{k=0..K_tap-1} W(f,k,t) * X(f,t-k)
      │
      ▼
iSTFT + overlap-add -> OUTPUT (real-time)
```

### Why hybrid (mask + deep-filter)?

* The mask provides robust, coarse suppression of stationary/noisy energy.
* Deep filtering combines multiple past frames to resolve overlap cases (speech+noise in same bin) and improves speech naturalness.
* Hybrid improves both intelligibility and perceptual quality versus mask-only or filter-only approaches in many conditions.

### Block details

* **Encoder**: MobileOne-like blocks. Depthwise convs keep MACs low; pointwise convs increase channels. Use BN or GroupNorm and lightweight activations (ReLU/SiLU).
* **Temporal**: A 1-layer causal GRU (hidden size 96..256 depending on model size) balances capacity and latency. Alternative: small causal TCN for parallelism.
* **Decoder**: Two heads:

  * Mask head: 1×1 conv -> sigmoid -> output shape: `[freq_bins]` (per frame)
  * Deep-filter head: 1×1 conv -> reshape -> `freq_bins × K_tap × (1 or 2)` (1 for real taps, 2 for real+imag if modeling complex taps)

---

## 3. Shapes and recommended hyperparameters

**Front-end**

* sample_rate = 16000
* frame_len = 320 samples (20 ms)
* hop = 160 samples (10 ms)
* n_fft = 320 -> freq_bins = 161 (0..160)

**Model input**

* K_ctx (context frames) = 5 (t-4..t)
* Input patch (batch=1): shape for Conv2D = `[B=1, C=1, T=K_ctx, F=freq_bins]` where C is channel (mag or stacked features)

**Filter taps**

* K_tap = 3..7. Typical: 5

**Two suggested configs**

* **Tiny** (embedded/mobile):

  * Encoder channels: `[16, 32, 48]`
  * GRU hidden: `96`
  * K_ctx = 5, K_tap = 3
  * Params: ~120k–250k
  * Objective: realtime on ARM mobile CPU

* **Medium** (edge/desktop):

  * Encoder channels: `[24, 48, 96]`
  * GRU hidden: `192`
  * K_ctx = 5, K_tap = 5
  * Params: ~600k–1.5M
  * Objective: higher quality, still realtime with optimization

**Tensor shapes** (example medium)

* Input buffer: `(K_ctx, 161)` magnitude
* After encoder: `(channels, T_reduce, F_reduce)` → flattened/pooled → sequence length=`T_reduce` → GRU input
* Mask head output: `(161,)` per frame
* Deep-filter head output: `(161, K_tap, 2)` for complex taps

---

## 4. Data pipeline & training

**Data sources**

* Clean speech: LibriSpeech / VCTK / DNS dataset (clean split)
* Noise: MUSAN, DNS real noises, environmental recordings, traffic, babble
* RIRs: optional for reverberation augmentation

**Mixing & augmentation**

* On-the-fly mixing: pick SNR uniformly between -5 and +20 dB (or as needed)
* Random RIR convolution with prob p (e.g., p=0.3)
* Level jitter, clipping simulation, codec/noise distortions (optional)

**Losses**

* Primary: SI-SDR (time-domain) — encourages perceptual fidelity.
* Auxiliary:

  * Complex spectral MSE (between enhanced and clean complex spectrogram)
  * Multi-resolution STFT loss (optional)
  * Mask regularization (L2 on mask gradients to smooth)

**Training tips**

* Compute STFT on GPU if possible to accelerate training.
* Batch size: 16–64 patches depending on GPU memory.
* Learning rate schedule: AdamW with cosine annealing or step decay; LR warm-up helpful.
* Use teacher-student distillation: train a larger teacher (higher K_ctx/K_tap) and distill to tiny model.

---

## 5. Streaming inference & deployment

**Streaming loop (per incoming hop)**

1. Acquire `hop` samples (160 samples).
2. Window and compute STFT frame (complex X(f,t)).
3. Compute magnitude / log-power and push into ring buffer.
4. If buffer length >= K_ctx, build input patch and run model forward.
5. Retrieve mask(t) and filter coefficients W(f,k,t).
6. For each frequency bin `f`, compute:

   `Y(f,t) = mask(f,t) * sum_{k=0..K_tap-1} W(f,k,t) * X(f,t-k)`

   Note: store past complex frames to compute this causal sum.
7. iSTFT the enhanced frame and overlap-add to output buffer.
8. Emit the earliest available output samples (accounting for frame length/hop).

**Optimization tips**

* Precompute FFT windowing and twiddle factors; use optimized FFT library (FFTW, kissFFT, oneAPI, or platform-provided DSP).
* Fuse 1x1 conv + BN for faster inference.
* Quantize to int8 with calibration (post-training quantization) for mobile.
* Use platform-accelerated libraries (oneDNN, TFLite, CoreML, NNAPI) when deploying.

---

## 6. Performance targets & evaluation

**Objective metrics**

* SI-SDR improvement (dB)
* PESQ / POLQA (if available)
* STOI
* DNSMOS or other no-reference metrics

**Real-time metrics**

* Latency: aim for end-to-end ≤ 30 ms (20 ms frame + compute + iSTFT)
* Real-time factor (RTF) on target device: target < 1.0 (preferably << 1)
* Memory and CPU usage: measure peak RAM and MACs

---

## 7. Experiments & Ablations (suggested)

* Encoder: compare MobileOne-like vs standard conv encoder
* Temporal: GRU vs TCN vs causal Transformer
* Decoder: mask-only vs deep-filter-only vs hybrid
* Frequency grouping: full 161 bins vs ERB 32 bands
* Adaptive K_tap by band (low-freq more taps)
* Distillation: teacher (large) -> student (tiny)

---

## 8. File layout (suggestion)

```
mobile_deepfilternet/
├─ README.md            # this file
├─ configs/
│  ├─ tiny.yaml
│  └─ medium.yaml
├─ data/
│  └─ scripts for mixing and augmentation
├─ models/
│  └─ model_def.py
├─ training/
│  └─ train.py
├─ inference/
│  └─ stream_infer.py
├─ scripts/
│  └─ evaluate.py
└─ experiments/
   └─ ablation_*.ipynb
```

---

## 9. How to run (quick start)

**Train (example)**

```bash
python training/train.py --config configs/medium.yaml --gpus 1
```

**Streaming inference local (example)**

```bash
python inference/stream_infer.py --model checkpoints/medium_ckpt.pt --sr 16000
```

**Convert/Quantize for mobile**

* Export to ONNX or TFLite / CoreML.
* Calibrate with representative dataset and apply post-training quantization.

---

## 10. Contribution & license

Contributions are welcome — PRs for:

* more efficient encoder blocks
* better training recipes
* optimized inference kernels for embedded platforms

Suggested license: MIT / Apache-2.0 (pick per project policy).

---

## Contact / Author

Implementation & design notes by: Project team — MobileDeepFilterNet

---

*End of README*
