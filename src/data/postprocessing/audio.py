"""
src/data/postprocessing/audio.py
=================================
Real-time postprocessing: converts UNet_IRM_Denoise model outputs back into
PCM audio frames.  Designed for in-memory streaming — no file I/O.

Model output contract (from unet.py)
--------------------------------------
    The model returns  clean_estimate = mask × noisy_spectrogram
    shape : (B, F=256, T=32, C=1)   log1p-compressed magnitude
    mask  : sigmoid output, values ∈ (0, 1) — already multiplied inside model

    To recover audio we need to:
        1. Extract the estimated magnitude from the output.
        2. Undo log1p compression  →  linear magnitude.
        3. Restore the Nyquist bin cropped during preprocessing (256 → 257).
        4. Combine with the noisy phase (phase substitution).
        5. iSTFT  →  waveform.
        6. Emit only the last hop (8 ms) for real-time streaming.

Real-time sliding-buffer loop
-------------------------------
    pre  = AudioPreprocessor.from_configs()
    post = AudioPostprocessor.from_configs()

    buffer = collections.deque(maxlen=32)   # 32-frame ring buffer

    for hop in audio_stream:                # new 8 ms block of samples
        buffer.append(hop)
        if len(buffer) < 32:
            continue

        log_mag, noisy_stft = pre.compute_magnitude(frames_to_wave(buffer))
        clean_est = model.predict(log_mag[None, ..., None])[0, ..., 0]  # (256, 32)

        pcm = post.reconstruct_frame(clean_est, noisy_stft)  # (128,) float32
        speaker.write(pcm)
"""

from __future__ import annotations

import logging
from typing import Optional

import librosa
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class PostprocessConfig:
    """
    STFT reconstruction parameters — must mirror configs/train_config.yaml [stft].
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        window: str = "hann",
        center: bool = True,
        eps: float = 1e-8,
        normalize_output: bool = True,
    ) -> None:
        self.sample_rate      = sample_rate
        self.n_fft            = n_fft
        self.hop_length       = hop_length
        self.win_length       = win_length
        self.window           = window
        self.center           = center
        self.eps              = eps
        self.normalize_output = normalize_output
        self.freq_bins        = n_fft // 2   # 256 — matches preprocessed F axis

    @classmethod
    def from_configs(
        cls,
        data_cfg_path: str = "configs/data_config.yaml",
        train_cfg_path: str = "configs/train_config.yaml",
    ) -> "PostprocessConfig":
        with open(data_cfg_path) as f:
            dcfg = yaml.safe_load(f)
        with open(train_cfg_path) as f:
            tcfg = yaml.safe_load(f)

        stft = tcfg.get("stft", {})
        dp   = dcfg.get("preprocessing", {})

        return cls(
            sample_rate  = dp.get("target_sample_rate", 16000),
            n_fft        = stft.get("n_fft", 512),
            hop_length   = stft.get("hop_length", 128),
            win_length   = stft.get("win_length", 512),
            window       = stft.get("window", "hann"),
            center       = stft.get("center", True),
        )


# ---------------------------------------------------------------------------
# Core postprocessor
# ---------------------------------------------------------------------------

class AudioPostprocessor:
    """
    Converts model output (clean_estimate spectrogram) back to PCM audio.

    All operations are in-memory and return numpy arrays — suitable for
    real-time streaming pipelines.
    """

    def __init__(self, config: Optional[PostprocessConfig] = None) -> None:
        self.cfg = config or PostprocessConfig()

    @classmethod
    def from_configs(
        cls,
        data_cfg: str = "configs/data_config.yaml",
        train_cfg: str = "configs/train_config.yaml",
    ) -> "AudioPostprocessor":
        return cls(PostprocessConfig.from_configs(data_cfg, train_cfg))

    # ------------------------------------------------------------------
    # Step 1 — Undo log1p compression
    # ------------------------------------------------------------------

    def decompress(self, log_mag: np.ndarray) -> np.ndarray:
        """
        Invert the log1p compression applied during preprocessing.

            forward : log_mag = log1p(|mag| + eps)
            inverse : mag     = expm1(log_mag) - eps

        Parameters
        ----------
        log_mag : (F=256, T)  compressed magnitude from model output

        Returns
        -------
        mag : (256, T)  linear magnitude ≥ 0
        """
        # TODO: Invert the log1p compression applied during preprocessing.
        #
        #   Forward pass (preprocessing):
        #       log_mag = log1p( |stft| + ε )
        #
        #   Inverse pass (here):
        #       mag = expm1( log_mag ) − ε
        #
        #   Use np.expm1() rather than np.exp(x)-1 for numerical precision near 0:
        #       expm1 avoids catastrophic cancellation when log_mag is small.
        #
        #   Clip the result to [0, +∞) because magnitudes cannot be negative.
        #   Cast to float32 and return shape (256, T).
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Step 2 — Restore Nyquist bin (256 → 257)
    # ------------------------------------------------------------------

    def restore_nyquist(self, mag: np.ndarray) -> np.ndarray:
        """
        Re-add the Nyquist bin that was cropped during preprocessing.

        (F=256, T)  →  (F=257, T)  by mirroring the last frequency bin.
        librosa.istft expects (n_fft//2 + 1, T) = (257, T).
        """
        # TODO: Re-attach the Nyquist bin that was cropped in preprocessing.
        #
        #   Preprocessing removed bin index 256 (the Nyquist frequency = sr/2 = 8 kHz)
        #   to obtain shape (256, T). librosa.istft requires (n_fft//2 + 1, T) = (257, T).
        #
        #   Strategy: mirror the last valid frequency bin to fill the missing row.
        #       result = np.vstack([mag, mag[-1:, :]])    → (257, T)
        #
        #   Note: the exact value of the Nyquist bin has negligible perceptual impact
        #   for speech (content above 7.9 kHz is minimal), so mirroring is acceptable.
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Step 3a — Full waveform reconstruction (phase substitution)
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        clean_estimate: np.ndarray,
        noisy_stft: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct a waveform from the model's clean_estimate output using
        noisy-phase substitution (no phase prediction needed).

        Parameters
        ----------
        clean_estimate : (F=256, T)  model output — log1p clean magnitude
                         (already mask-multiplied inside the model)
        noisy_stft     : (257, T)    complex STFT of the noisy buffer
                         returned by AudioPreprocessor.compute_magnitude()

        Returns
        -------
        waveform : (N,)  float32  reconstructed audio segment
        """
        # TODO: Reconstruct a waveform from the model output via noisy-phase substitution.
        #
        #   Step 1 — Decompress log-magnitude:
        #       mag = self.decompress(clean_estimate)        → (256, T)  linear
        #
        #   Step 2 — Restore Nyquist:
        #       mag_full = self.restore_nyquist(mag)         → (257, T)
        #
        #   Step 3 — Phase substitution (polar form reconstruction):
        #       Extract instantaneous phase from the noisy STFT:
        #           φ[k,t] = ∠ noisy_stft[k,t]  =  np.angle(noisy_stft)
        #
        #       Combine estimated magnitude with noisy phase:
        #           Ŝ[k,t] = mag_full[k,t] · exp( j·φ[k,t] )         → (257, T) complex
        #
        #       Justification: the IRM mask suppresses noise in magnitude;
        #       phase distortion from the noisy STFT is perceptually tolerable
        #       and avoids the cost of a separate phase estimator.
        #
        #   Step 4 — Inverse STFT (overlap-add):
        #       waveform = librosa.istft(Ŝ, hop_length, win_length, window, center)
        #
        #   Step 5 — Optional peak normalisation; cast to float32 and return.
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Step 3b — Single-hop output for real-time streaming
    # ------------------------------------------------------------------

    def reconstruct_frame(
        self,
        clean_estimate: np.ndarray,
        noisy_stft: np.ndarray,
        output_frame_idx: int = -1,
    ) -> np.ndarray:
        """
        Real-time variant: reconstruct the full segment internally but emit
        only a single hop_length-sized PCM frame.

        Used with the sliding-buffer strategy:
            • infer on a 32-frame buffer
            • emit only the last frame (most recently denoised 8 ms hop)
            • shift the buffer by 1 frame and repeat

        Parameters
        ----------
        clean_estimate   : (F=256, T=32)  model output for the current buffer
        noisy_stft       : (257, T=32)    complex STFT of the same buffer
        output_frame_idx : which hop frame to emit; -1 = last (default)

        Returns
        -------
        pcm_frame : (hop_length=128,)  float32  — 8 ms of denoised audio
        """
        # TODO: Real-time variant — reconstruct the full segment, emit only one hop frame.
        #
        #   Step 1 — reconstruct the full waveform for the 32-frame buffer:
        #       full = self.reconstruct(clean_estimate, noisy_stft)   → (N,) samples
        #
        #   Step 2 — partition the waveform into hop_length-sized frames:
        #       n_hops = len(full) // hop_length
        #       frames[i] = full[i * hop : (i+1) * hop]
        #
        #   Step 3 — return frames[output_frame_idx] (default −1 = last frame).
        #
        #   Latency rationale:
        #       Sliding buffer has 32 frames. At each step, 1 new frame arrives.
        #       The model runs on the full 32-frame context but we only emit the
        #       newest denoised frame (−1) → net latency = 1 hop = 8 ms.
        #
        #   Edge case: if n_hops == 0, return the entire `full` array.
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Step 3c — Griffin-Lim fallback (no phase available)
    # ------------------------------------------------------------------

    def reconstruct_griffin_lim(
        self,
        clean_estimate: np.ndarray,
        n_iter: int = 32,
    ) -> np.ndarray:
        """
        Phase-free reconstruction via Griffin-Lim algorithm.

        Use when the noisy STFT phase is unavailable (e.g. the model is
        evaluated from stored log-magnitude files without the original audio).

        Parameters
        ----------
        clean_estimate : (F=256, T)  model output log-magnitude
        n_iter         : Griffin-Lim iterations (more → higher quality, slower)

        Returns
        -------
        waveform : (N,)  float32
        """
        # TODO: Phase-free reconstruction via the Griffin-Lim algorithm.
        #
        #   Use when the original noisy STFT is unavailable (e.g. offline evaluation
        #   from stored .npz log-magnitude files).
        #
        #   Algorithm (Griffin & Lim, 1984 — alternating projections):
        #
        #       Initialise:  X₀ = |M| · exp(j·φ₀)   (random phase φ₀)
        #
        #       for i = 0, 1, …, n_iter−1:
        #           x_i    = iSTFT(X_i)                   → time domain
        #           X_i⁺¹ = STFT(x_i)                   → back to freq domain
        #           X_i⁺¹ = |M| · exp( j · ∠X_i⁺¹ )   → replace magnitude with M
        #
        #       waveform = iSTFT(X_{n_iter})
        #
        #   Convergence: error ||X_i − M||_F monotonically decreases with iterations.
        #   More iterations → lower phase inconsistency → better quality (slower).
        #
        #   Decompress and restore Nyquist before calling librosa.griffinlim().
        #   Optionally peak-normalise and cast to float32.
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _peak_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        # TODO: x_norm = x / (max|x| + ε)    (same as AudioPreprocessor.peak_normalize)
        raise NotImplementedError
