# Data Pipeline — Technical Documentation

> **Scope** — `src/data/preprocessing/` and `src/data/postprocessing/`
>
> This document covers every mathematical operation in the two modules.
> Read this before implementing any `# TODO` block.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
   - 2.1 [Loading and Resampling](#21-loading-and-resampling)
   - 2.2 [Peak Normalisation](#22-peak-normalisation)
   - 2.3 [SNR-Controlled Noise Mixing](#23-snr-controlled-noise-mixing)
   - 2.4 [Short-Time Fourier Transform (STFT)](#24-short-time-fourier-transform-stft)
   - 2.5 [Nyquist Bin Cropping](#25-nyquist-bin-cropping)
   - 2.6 [log1p Compression](#26-log1p-compression)
   - 2.7 [Spectrogram Segmentation](#27-spectrogram-segmentation)
   - 2.8 [Dataset Builder](#28-dataset-builder)
3. [Postprocessing Pipeline](#3-postprocessing-pipeline)
   - 3.1 [Inverse log1p (Decompression)](#31-inverse-log1p-decompression)
   - 3.2 [Nyquist Bin Restoration](#32-nyquist-bin-restoration)
   - 3.3 [Phase Substitution & iSTFT](#33-phase-substitution--istft)
   - 3.4 [Real-Time Frame Extraction](#34-real-time-frame-extraction)
   - 3.5 [Griffin-Lim Reconstruction](#35-griffin-lim-reconstruction)
4. [Data Shapes at Each Stage](#4-data-shapes-at-each-stage)
5. [Configuration Reference](#5-configuration-reference)
6. [Common Mistakes](#6-common-mistakes)

---

## 1. Architecture Overview

```
                 ┌─────────────────── TRAINING ──────────────────────┐
                 │                                                     │
  clean.wav ─────┤                                              Y ─── Loss
                 │  load → norm → mix → STFT → |·| → crop             │
  noise.wav ─────┤             (SNR)       → log1p → slice     X ──► UNet ──► Ŷ
                 │                                                     │
                 └─────────────────────────────────────────────────────┘

                 ┌─────────────────── REAL-TIME ─────────────────────┐
                 │                                                     │
  microphone ───►│  sliding buffer (32 frames)                        │
                 │  compute_magnitude()   →  log_mag (256,32)         │
                 │  [noisy_stft retained for phase]                   │
                 │                                 ↓ model.predict()  │
                 │  reconstruct_frame()   ←  clean_est (256,32)       │
                 │                                 ↓                  │
  speaker ◄──────│  PCM (128 samples = 8 ms)                          │
                 └─────────────────────────────────────────────────────┘
```

---

## 2. Preprocessing Pipeline

### 2.1 Loading and Resampling

All audio is loaded at a fixed sample rate `sr = 16 000 Hz` and downmixed to mono.

If the original file has sample rate $f_s^{orig} \ne f_s$, librosa applies **polyphase resampling**:

$$x_{resampled}[n] = \sum_{k} x_{orig}[k] \cdot h\!\left[n - k \cdot \frac{f_s^{orig}}{f_s}\right]$$

where $h$ is a low-pass anti-aliasing FIR filter.

**Implementation hint** — `librosa.load(path, sr=16000, mono=True, dtype=np.float32)`.  
Return value: 1-D `float32` array of shape $(N,)$.

---

### 2.2 Peak Normalisation

Before mixing, each waveform is scaled to unit peak amplitude:

$$x_{norm}[n] = \frac{x[n]}{\max_n |x[n]| + \varepsilon}, \quad \varepsilon = 10^{-8}$$

**Why?** Ensures SNR mixing (Section 2.3) is meaningful regardless of the recording loudness
of the source files.

**Implementation hint** — `np.max(np.abs(audio))`.

---

### 2.3 SNR-Controlled Noise Mixing

**Signal-to-Noise Ratio (dB)** is defined as:

$$\text{SNR}_{dB} = 20 \cdot \log_{10}\!\left(\frac{\text{RMS}_{clean}}{\text{RMS}_{noise}}\right)$$

where

$$\text{RMS}(x) = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$

To achieve a *target* $\text{SNR}_{dB}^{*}$, the noise must be scaled by:

$$\text{scale} = \frac{\text{RMS}_{clean}}{10^{\,\text{SNR}_{dB}^{*}/20} \cdot \text{RMS}_{noise}}$$

The mixed signal is:

$$y[n] = x_{clean}[n] + \text{scale} \cdot x_{noise}[n]$$

**Random mixing** draws the target SNR from a uniform distribution:

$$\text{SNR}_{dB}^{*} \sim \mathcal{U}(-5, 20)$$

covering very noisy (−5 dB) to mildly noisy (+20 dB) conditions.

**Length alignment** — if `len(noise) < len(clean)`, tile the noise array:

```
noise_tiled = np.tile(noise, ceil(len(clean)/len(noise)))[:len(clean)]
```

If `len(noise) >= len(clean)`, take a random contiguous crop.

---

### 2.4 Short-Time Fourier Transform (STFT)

The STFT decomposes a signal into time-frequency bins:

$$X[k, m] = \sum_{n=0}^{N-1} x[n + m \cdot H] \cdot w[n] \cdot e^{-j 2\pi k n / N}$$

| Symbol | Value | Meaning |
|--------|-------|---------|
| $N$ | 512 (`n_fft`) | FFT size — frequency resolution $\Delta f = f_s / N = 31.25$ Hz |
| $H$ | 128 (`hop_length`) | Frame shift — time resolution $\Delta t = H / f_s = 8$ ms |
| $L$ | 512 (`win_length`) | Window length (equal to $N$ here) |
| $w$ | Hann | Window function (reduces spectral leakage) |
| $K$ | $N/2 + 1 = 257$ | Number of unique frequency bins (real-valued input) |

**Hann window:**

$$w[n] = 0.5 \left(1 - \cos\!\left(\frac{2\pi n}{L - 1}\right)\right), \quad n = 0, \ldots, L-1$$

**Output shape:** complex `(257, T)` where $T = \lceil N_{samples} / H \rceil$ (with `center=True` padding).

**Implementation hint** — `librosa.stft(audio, n_fft=512, hop_length=128, win_length=512, window="hann", center=True)`.

---

### 2.5 Nyquist Bin Cropping

The STFT returns $N/2 + 1 = 257$ frequency bins:
- Bins $0 \ldots 255$ — baseband (DC to 7 968.75 Hz)
- Bin $256$ — **Nyquist frequency** $f_s / 2 = 8$ kHz

The Nyquist bin is always **real-valued** and carries no phase information.
We discard it to obtain $F = N/2 = 256$ bins:

$$|X|_{cropped}[k, m] = |X[k, m]|, \quad k = 0, \ldots, 255$$

**Benefits of cropping:**
- $F = 256 = 2^8$ — power of two, divides cleanly through all 4 U-Net encoder levels with asymmetric `(2,1)` pooling: $256 \to 128 \to 64 \to 32 \to 16$.
- Slightly reduces compute per forward pass.

**Implementation hint** — `mag = mag[:self.cfg.freq_bins, :]` where `freq_bins = n_fft // 2 = 256`.

---

### 2.6 log1p Compression

Speech spectrograms span a very large dynamic range (~60 dB).  
We apply `log1p` compression to map this to a compact range suitable for neural network training:

$$\hat{M}[k, m] = \log_1p(|X[k, m]| + \varepsilon) = \log\!\bigl(1 + |X[k, m]| + \varepsilon\bigr)$$

where $\varepsilon = 10^{-8}$ prevents $\log(0) = -\infty$ for silent bins.

**Properties:**
- $\hat{M}[k,m] = 0$ when $|X| = 0$ (no offset, correct zero-point)
- Approximately logarithmic for large $|X|$, approximately linear for small $|X|$
- Differentiable everywhere (important if gradients flow through the input)

**Compared to alternatives:**

| Transform | Formula | Issues |
|-----------|---------|--------|
| none | $\|X\|$ | Very large range, poor gradients |
| $\log(|\|X\| + \varepsilon)$ | $\log(\|X\|+\varepsilon)$ | Negative values; offsets sensitivity at zero |
| **log1p** | $\log(1+\|X\|+\varepsilon)$ | ✅ Non-negative, smooth, compact range |
| Power-law $x^{1/3}$ | $\|X\|^{0.333}$ | Harsher compression; less interpretable |

**Implementation hint** — `np.log1p(mag + self.cfg.eps)`.

---

### 2.7 Spectrogram Segmentation

The model accepts fixed-size windows of shape $(F=256,\, T=32,\, C=1)$.

A full utterance spectrogram of shape $(256, T_{total})$ is sliced into
overlapping windows with **50% overlap** ($step = 16$):

```
   T_total frames
   ┌────────────────────────────────────────────────────────┐
   │ w0: frames [0,  32) │
   │         w1: frames [16, 48) │
   │                  w2: frames [32, 64) │
   └────────────────────────────────────────────────────────┘
   step = 16  →  ~50% overlap during training
```

Number of windows: $N = \lfloor (T_{total} - T_{seg}) / step \rfloor + 1$

**Short-signal padding** — if $T_{total} < T_{seg}$, right-pad with zeros:

$$spec' = [spec \mid \mathbf{0}_{F \times (T_{seg} - T_{total})}]$$

**Output shape:** $(N,\, F=256,\, T_{seg}=32,\, C=1)$

**Implementation hint:**
```python
starts = range(0, T_total - T_seg + 1, step)
segments = np.stack([spec[:, s:s+T_seg] for s in starts])   # (N, F, T)
return segments[..., np.newaxis]                             # (N, F, T, 1)
```

---

### 2.8 Dataset Builder

`build_dataset(split)` writes `.npz` files to `data/processed/<split>/`:

```
data/processed/
├── train/
│   ├── batch_000000.npz   # keys: "noisy" (B,256,32,1), "clean" (B,256,32,1)
│   └── ...
├── val/
└── test/
```

**Train/val split strategy** (VIVOS only has `train/` and `test/` subdivisions):

$$N_{val} = \left\lfloor N_{train\_total} \cdot \frac{r_{val}}{r_{train} + r_{val}} \right\rfloor$$

where $r_{train}, r_{val}$ come from `data_config.yaml → split`.

Files are shuffled (with `seed`) before splitting to avoid leakage.

---

## 3. Postprocessing Pipeline

### 3.1 Inverse log1p (Decompression)

Exact inverse of Section 2.6:

$$|X|_{est}[k,m] = \text{expm1}(\hat{M}[k,m]) - \varepsilon$$

where `expm1(x) = exp(x) - 1`, implemented via `np.expm1()` for numerical precision.

**Why `expm1` instead of `exp(x) - 1`?**  
For small $x$: `exp(x) ≈ 1`, so `exp(x) - 1` suffers catastrophic cancellation.
`expm1(x)` uses a Taylor expansion at $x=0$ for precision.

Clip to non-negative: $|X|_{est} \leftarrow \max(|X|_{est},\, 0)$

---

### 3.2 Nyquist Bin Restoration

`librosa.istft` requires shape $(257, T)$. We re-attach the missing Nyquist bin
by mirroring the highest valid bin (row index 255):

$$|X|_{full}[256, m] = |X|_{full}[255, m]$$

This is perceptually acceptable because:
- Speech energy above ~7.9 kHz is minimal.
- Any error in the Nyquist bin has negligible audible impact.

**Implementation hint** — `np.vstack([mag, mag[-1:, :]])` → shape goes from $(256, T)$ to $(257, T)$.

---

### 3.3 Phase Substitution & iSTFT

The model outputs a **magnitude-only** estimate. To reconstruct a complex spectrogram,
we borrow the phase of the **noisy** STFT:

$$\hat{S}[k,m] = |\hat{X}|_{full}[k,m] \cdot e^{j \cdot \angle X_{noisy}[k,m]}$$

where $\angle X_{noisy}[k,m] = \arg\!\bigl(X_{noisy}[k,m]\bigr) = \arctan\!\left(\frac{\text{Im}(X_{noisy})}{\text{Re}(X_{noisy})}\right)$

**Implementation hint** — `np.angle(noisy_stft)` then `mag_full * np.exp(1j * phase)`.

The waveform is then recovered via the **inverse STFT** (overlap-add synthesis):

$$\hat{x}[n] = \frac{\sum_m \hat{s}_m[n - mH] \cdot w[n - mH]}{\sum_m w^2[n - mH]}$$

where $\hat{s}_m$ is the $m$-th synthesised frame and the denominator is the OLA normalisation.

**Implementation hint** — `librosa.istft(stft_est, hop_length=128, win_length=512, window="hann", center=True)`.

---

### 3.4 Real-Time Frame Extraction

The sliding-buffer loop processes **32 frames** at each step but emits only **1 frame (8 ms)**:

```
Buffer at time t:   [f_{t-31}, f_{t-30}, ..., f_{t-1}, f_t]    (32 frames = 256 ms)
                                                          ↑
                                                  newest frame → emit
```

The returned PCM frame is:

$$\hat{x}_{out}[n] = \hat{x}[n],\quad n \in [(T_{seg}-1) \cdot H,\; T_{seg} \cdot H)$$

i.e., `full[-1 * hop_length :]` — the last `hop_length = 128` samples.

**Latency** = 1 hop = $H / f_s = 128 / 16000 = 8\,\text{ms}$

---

### 3.5 Griffin-Lim Reconstruction

When the original noisy STFT is unavailable (e.g. evaluating from stored `.npz` files),
use the Griffin-Lim algorithm (Griffin & Lim, 1984) for phase estimation.

**Algorithm — alternating projections:**

Let $|M|$ be the target magnitude. Define two sets:
- $\mathcal{A}$ — STFTs that are consistent with the time-domain signal (every iSTFT→STFT round-trip lies here)
- $\mathcal{B}$ — spectrograms with magnitude $= |M|$

Griffin-Lim alternates projections onto these sets:

$$X_0 = |M| \cdot e^{j \phi_0}\qquad \text{(random initial phase)}$$

$$\text{for } i = 0, 1, \ldots, n_{iter}-1:$$
$$x_i = \text{iSTFT}(X_i) \quad\Rightarrow\quad X_{i+1}^{(\mathcal{A})} = \text{STFT}(x_i) \quad\Rightarrow\quad X_{i+1} = |M| \cdot e^{j \angle X_{i+1}^{(\mathcal{A})}}$$

**Convergence criterion:**

$$\bigl\| |X_i| - |M| \bigr\|_F^2$$

is monotonically non-increasing. Empirically, $n_{iter} = 32$ gives good quality for speech.

**Implementation hint** — `librosa.griffinlim(mag_full, n_iter=32, n_fft=512, hop_length=128, win_length=512, window="hann", center=True)`.

---

## 4. Data Shapes at Each Stage

### Preprocessing

| Step | Operation | Shape | Dtype |
|------|-----------|-------|-------|
| Input WAV | raw audio | $(N_{samples},)$ | float32 |
| After `load()` | resampled, mono | $(N_{16k},)$ | float32 |
| After `peak_normalize()` | unit peak | $(N_{16k},)$ | float32 |
| After `mix_at_snr()` | noisy waveform | $(N_{16k},)$ | float32 |
| After `stft()` | complex spectrogram | $(257,\, T)$ | complex64 |
| After magnitude + crop | linear magnitude | $(256,\, T)$ | float32 |
| After `log1p` | compressed magnitude | $(256,\, T)$ | float32 |
| After `slice_spectrogram()` | segmented | $(N,\, 256,\, 32,\, 1)$ | float32 |

### Postprocessing

| Step | Operation | Shape | Dtype |
|------|-----------|-------|-------|
| Model output | log-magnitude estimate | $(256,\, 32)$ | float32 |
| After `decompress()` | linear magnitude | $(256,\, 32)$ | float32 |
| After `restore_nyquist()` | full magnitude | $(257,\, 32)$ | float32 |
| After phase substitution | complex spectrogram | $(257,\, 32)$ | complex64 |
| After `librosa.istft()` | waveform | $(N_{samples},)$ | float32 |
| After `reconstruct_frame()` | single hop | $(128,)$ | float32 |

---

## 5. Configuration Reference

Both modules share STFT parameters from `configs/train_config.yaml` (section `stft`).

```yaml
# configs/train_config.yaml  →  [stft]
stft:
  n_fft:       512      # FFT size  →  freq resolution = sr/N = 31.25 Hz
  hop_length:  128      # frame shift  →  time resolution = H/sr = 8 ms
  win_length:  512      # window length (= n_fft here)
  window:      hann     # window function
  center:      true     # symmetric padding at signal boundaries

# [model]
model:
  input_shape: [256, 32, 1]   # [F, T, C]  →  256 × 32 × 8ms = 256 ms buffer
```

```yaml
# configs/data_config.yaml  →  [preprocessing]
preprocessing:
  target_sample_rate: 16000
  mono: true
  normalize: true
  snr_range_db: [-5, 20]      # dB range for random SNR mixing
```

### Latency Table

| Parameter | Formula | Value |
|-----------|---------|-------|
| Frame duration | $H / f_s$ | 8 ms |
| Buffer length | $T \cdot H / f_s$ | 256 ms |
| Output latency (streaming) | $1 \cdot H / f_s$ | 8 ms |
| Frequency resolution | $f_s / N$ | 31.25 Hz |

---

## 6. Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Using `np.log` instead of `np.log1p` | Silent bins get $-\infty$ → NaN loss | Always use `np.log1p(mag + eps)` |
| Not cropping Nyquist bin in preprocessing | Shape mismatch `(257,…)` vs model input `(256,…)` | `mag = mag[:256, :]` |
| Forgetting to restore Nyquist before `istft` | `librosa.istft` expects shape `(257,T)` → ValueError | Call `restore_nyquist()` before iSTFT |
| Using `np.exp(x) - 1` instead of `np.expm1` | Numerical cancellation for small magnitudes | Always use `np.expm1()` |
| Cropping T axis with `(2,2)` pooling | T=32 collapses to T=2 at bottleneck | Model uses asymmetric `(2,1)` pooling — do not change |
| Processing noisy and clean with different SNR | Spectrogram frames are misaligned | Mix first, then STFT both waveforms from the same mixed `noisy` |
| Saving the complex STFT for clean (not noisy) | Phase substitution uses noisy phase — wrong phase if you save clean's STFT | Only retain `noisy_stft` from `compute_magnitude(noisy)` |
| `segment_step > segment_frames` | Windows do not overlap → gaps in coverage | Keep `step ≤ T_seg` (default `step = T_seg // 2 = 16`) |

---

*Last updated: 2026-03-04*
