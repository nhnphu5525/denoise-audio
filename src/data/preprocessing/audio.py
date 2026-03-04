"""
src/data/preprocessing/audio.py
================================
Converts raw WAV files into (noisy, clean) spectrogram segment pairs
ready to be consumed by the UNet_IRM_Denoise model.

Model input contract (from unet.py)
-------------------------------------
    shape : (F=256, T=32, C=1)
    dtype : float32
    values: log1p-compressed magnitude spectrogram

    F=256  ← n_fft//2, Nyquist bin cropped (512//2 = 256)
    T=32   ← sliding buffer length  →  32 × (128/16000) = 256 ms latency
    C=1    ← single-channel magnitude (no phase)

Pipeline
--------
    WAV  →  resample (16 kHz)  →  mono  →  peak-normalise
         →  mix clean + noise at random SNR
         →  STFT  →  |magnitude|  →  crop Nyquist (257→256)
         →  log1p  →  slice into (256, 32, 1) windows

Usage
-----
    from src.data.preprocessing import AudioPreprocessor, build_dataset

    pre = AudioPreprocessor.from_configs()
    noisy_segs, clean_segs = pre.process_pair("speech.wav", "noise.wav")
    # → shape (N, 256, 32, 1)

    build_dataset(split="train")
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """All parameters consumed by the preprocessing pipeline."""

    # Audio
    sample_rate: int = 16000
    mono: bool = True
    normalize: bool = True

    # STFT — must stay in sync with configs/train_config.yaml [stft]
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    window: str = "hann"
    center: bool = True

    # Compression
    eps: float = 1e-8            # floor added before log1p to avoid log(0)

    # Noise mixing
    snr_range_db: Tuple[float, float] = (-5.0, 20.0)

    # Segmentation — T must match model input_shape[1]
    segment_frames: int = 32     # T frames per window
    segment_step: int = 16       # stride (16 = 50 % overlap during training)

    # Paths
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    # Derived
    freq_bins: int = field(init=False)   # F = n_fft // 2

    def __post_init__(self) -> None:
        self.freq_bins = self.n_fft // 2  # 256 after Nyquist crop

    # ------------------------------------------------------------------
    @classmethod
    def from_configs(
        cls,
        data_cfg_path: str = "configs/data_config.yaml",
        train_cfg_path: str = "configs/train_config.yaml",
    ) -> "PreprocessConfig":
        """Load parameters from both YAML config files."""
        with open(data_cfg_path) as f:
            dcfg = yaml.safe_load(f)
        with open(train_cfg_path) as f:
            tcfg = yaml.safe_load(f)

        dp    = dcfg.get("preprocessing", {})
        stft  = tcfg.get("stft", {})
        model = tcfg.get("model", {})
        paths = dcfg.get("paths", {})
        snr   = dp.get("snr_range_db", [-5, 20])

        return cls(
            sample_rate    = dp.get("target_sample_rate", 16000),
            mono           = dp.get("mono", True),
            normalize      = dp.get("normalize", True),
            n_fft          = stft.get("n_fft", 512),
            hop_length     = stft.get("hop_length", 128),
            win_length     = stft.get("win_length", 512),
            window         = stft.get("window", "hann"),
            center         = stft.get("center", True),
            snr_range_db   = (float(snr[0]), float(snr[1])),
            segment_frames = model.get("input_shape", [256, 32, 1])[1],
            raw_dir        = paths.get("raw_dir", "data/raw"),
            processed_dir  = paths.get("processed_dir", "data/processed"),
        )


# ---------------------------------------------------------------------------
# Core preprocessor
# ---------------------------------------------------------------------------

class AudioPreprocessor:
    """
    Transforms raw (clean, noise) audio pairs into matched spectrogram
    segment arrays shaped for the UNet_IRM_Denoise model.

    Each output segment has shape (F=256, T=32, C=1) — a 256 ms window of
    log1p-compressed magnitude spectrogram, matching the model's input contract.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.cfg = config or PreprocessConfig()

    @classmethod
    def from_configs(
        cls,
        data_cfg: str = "configs/data_config.yaml",
        train_cfg: str = "configs/train_config.yaml",
    ) -> "AudioPreprocessor":
        return cls(PreprocessConfig.from_configs(data_cfg, train_cfg))

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> np.ndarray:
        """
        Load a WAV file, resample to target sample rate, convert to mono.

        Returns
        -------
        audio : np.ndarray  shape (N,)  float32
        """
        # TODO: Load the WAV file at `path` using librosa.load().
        #   - Resample to self.cfg.sample_rate (16 kHz)
        #   - Downmix to mono if self.cfg.mono is True
        #   - Keep dtype as float32
        #   - Return a 1-D ndarray of shape (N,)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Step 2 — Normalise
    # ------------------------------------------------------------------

    @staticmethod
    def peak_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Scale waveform so max absolute amplitude = 1."""
        # TODO: Scale the waveform so that its peak absolute amplitude equals 1.
        #
        #   Formula:  x_norm = x / (max|x| + ε)
        #
        #   The ε term (default 1e-8) prevents division by zero for silent clips.
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Step 3 — SNR mixing
    # ------------------------------------------------------------------

    def mix_at_snr(
        self,
        clean: np.ndarray,
        noise: np.ndarray,
        snr_db: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale noise to achieve the target SNR, then add to clean.

        SNR (dB) = 20 · log10(RMS_clean / RMS_noise_scaled)

        Returns (noisy_waveform, scaled_noise).
        """
        # TODO: Add scaled noise to clean speech to achieve the target SNR.
        #
        #   Step 1 — align lengths:
        #       noise = self._fit_length(noise, len(clean))
        #
        #   Step 2 — compute RMS of both signals:
        #       RMS(x) = sqrt( mean(x²) )   (use self._rms)
        #
        #   Step 3 — derive the scale factor from the SNR definition:
        #
        #       SNR_dB = 20 · log10( RMS_clean / RMS_noise_scaled )
        #
        #       Solving for scale (such that RMS_noise_scaled = RMS_noise · scale):
        #
        #           scale = RMS_clean / ( 10^(SNR_dB / 20) · RMS_noise )
        #
        #   Step 4 — apply and return:
        #       noisy = clean + scale · noise
        #       return (noisy, scale · noise)
        #
        #   Edge case: if RMS_noise < 1e-10, return (clean.copy(), noise.copy())
        raise NotImplementedError

    def random_mix(
        self,
        clean: np.ndarray,
        noise: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Mix at a uniformly sampled SNR from snr_range_db. Returns (noisy, snr_db)."""
        # TODO: Sample a random SNR value from self.cfg.snr_range_db
        #   using random.uniform(low, high), then call self.mix_at_snr().
        #   Return (noisy_waveform, snr_db_used).
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Step 4 — STFT → log-magnitude (model input format)
    # ------------------------------------------------------------------

    def stft(self, audio: np.ndarray) -> np.ndarray:
        """Complex STFT, shape (n_fft//2 + 1, T)."""
        # TODO: Compute the Short-Time Fourier Transform using librosa.stft().
        #
        #   STFT definition:
        #       X[k, m] = Σ_n  x[n] · w[n − m·H] · exp(−j·2π·k·n / N)
        #
        #   Parameters:
        #       N = self.cfg.n_fft          FFT size → frequency resolution Δf = sr / N
        #       H = self.cfg.hop_length     frame shift → time resolution Δt = H / sr
        #       win_length = self.cfg.win_length   (zero-padded to n_fft if < n_fft)
        #       window     = self.cfg.window       (e.g. "hann")
        #       center     = self.cfg.center       (True → pad signal at both ends)
        #
        #   Return the complex matrix of shape (n_fft//2 + 1, T) = (257, T).
        raise NotImplementedError

    def compute_magnitude(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute log1p-compressed magnitude spectrogram.

        Steps:
            1. STFT                     → complex (257, T)
            2. |magnitude|              → float32 (257, T)
            3. Crop Nyquist bin         → (256, T)   [F = n_fft // 2]
            4. log1p compression        → (256, T)   log1p(mag + eps)

        Returns
        -------
        log_mag : (F=256, T)   float32  — model input after slicing
        stft_cx : (257, T)     complex  — full STFT retained for phase recovery
                                          in postprocessing
        """
        # TODO: Build the log1p-compressed magnitude spectrogram.
        #
        #   Step 1 — compute STFT:
        #       stft_cx = self.stft(audio)               → complex (257, T)
        #
        #   Step 2 — extract magnitude:
        #       mag = |stft_cx|                          → float32 (257, T)
        #
        #   Step 3 — Nyquist crop (discard bin index 256):
        #       mag = mag[:self.cfg.freq_bins, :]        → (256, T)
        #
        #       Why crop? Bin 256 = sr/2 = 8 kHz (Nyquist). It is real-valued,
        #       carries no extra phase info, and dropping it makes F=256=2⁸,
        #       which divides cleanly through the 4-level asymmetric U-Net encoder.
        #
        #   Step 4 — log1p compression:
        #       log_mag = log1p(mag + ε)                 → (256, T)
        #
        #       Motivation: speech spectrogram values span ~60 dB of dynamic range.
        #       log1p ≈ log(1 + x) maps [0, ∞) → [0, ∞) monotonically, compresses
        #       large values, and satisfies log1p(0) = 0 (no offset needed).
        #       Adding ε before log avoids log(0) = −∞ for silent bins.
        #
        #   Return (log_mag, stft_cx).
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Step 5 — Slice into (256, 32, 1) windows
    # ------------------------------------------------------------------

    def slice_spectrogram(
        self,
        spec: np.ndarray,
        step: Optional[int] = None,
    ) -> np.ndarray:
        """
        Slice a (F, T_total) spectrogram into overlapping T-frame windows.

        Parameters
        ----------
        spec : (F, T_total)
        step : stride between windows; defaults to self.cfg.segment_step

        Returns
        -------
        segments : (N, F, T_seg, 1)  — batch of model-ready inputs
        """
        # TODO: Slice a (F, T_total) spectrogram into overlapping (F, T_seg) windows.
        #
        #   Parameters:
        #       T_seg = self.cfg.segment_frames   (= 32  →  256 ms at 16 kHz / 128-hop)
        #       step  = self.cfg.segment_step     (= 16  →  50 % overlap, more diversity)
        #
        #   Step 1 — short-signal padding:
        #       If T_total < T_seg, zero-pad on the right:
        #           pad = zeros((F, T_seg − T_total))
        #           spec = hstack([spec, pad])
        #
        #   Step 2 — generate window start indices:
        #       starts = [0, step, 2·step, …]  while  start + T_seg ≤ T_total
        #
        #   Step 3 — extract, stack, add channel dim:
        #       segments[i] = spec[:, starts[i] : starts[i] + T_seg]
        #       result = np.stack(segments)             → (N, F, T_seg)
        #       return result[..., np.newaxis]          → (N, F, T_seg, 1)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Step 6 — Full pair processing
    # ------------------------------------------------------------------

    def process_pair(
        self,
        clean_path: str | Path,
        noise_path: str | Path,
        snr_db: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        End-to-end preprocessing for one (clean_file, noise_file) pair.

        Returns
        -------
        noisy_segs : (N, 256, 32, 1)  float32 — model input
        clean_segs : (N, 256, 32, 1)  float32 — training target
        """
        # TODO: Execute the full preprocessing pipeline for one (clean, noise) pair.
        #
        #   Pipeline:
        #     1. Load clean waveform  →  self.load(clean_path)
        #     2. Load noise waveform  →  self.load(noise_path)
        #     3. Peak-normalise both if self.cfg.normalize is True.
        #     4. Mix at snr_db; if None, call self.random_mix() to sample randomly.
        #     5. Log the mix details with logger.debug().
        #     6. Compute log-magnitude spectrograms for BOTH noisy and clean signals
        #        via self.compute_magnitude() (discard the returned stft_cx for clean).
        #     7. Slice both spectrograms into segment windows.
        #     8. Return (noisy_segs, clean_segs), each of shape (N, 256, 32, 1).
        #
        #   These arrays become (X, Y) pairs fed to the U-Net during training:
        #       X = noisy_segs  (model input)
        #       Y = clean_segs  (supervision target for the IRM loss)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rms(audio: np.ndarray) -> float:
        # TODO: RMS(x) = sqrt( mean(x²) + ε )  where ε = 1e-10
        raise NotImplementedError

    @staticmethod
    def _fit_length(noise: np.ndarray, target: int) -> np.ndarray:
        """Trim or tile noise to exactly `target` samples."""
        # TODO: Return a noise array of exactly `target` samples.
        #   - If len(noise) >= target: random-crop a contiguous window of length target.
        #   - If len(noise) < target:  tile (np.tile) noise enough times, then crop.
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dataset builder — writes processed .npz files to disk for training
# ---------------------------------------------------------------------------

def build_dataset(
    split: str = "train",
    data_cfg: str = "configs/data_config.yaml",
    train_cfg: str = "configs/train_config.yaml",
    pairs_per_clean: int = 3,
    seed: int = 42,
) -> None:
    """
    Build and persist a processed dataset split.

    For each clean speech file:
      • sample `pairs_per_clean` noise files at random
      • process each pair into segments
      • save as  data/processed/<split>/batch_<idx>.npz
            keys: noisy (B, 256, 32, 1)
                  clean (B, 256, 32, 1)

    Parameters
    ----------
    split           : "train" | "val" | "test"
    pairs_per_clean : noise augmentations per clean file
    seed            : reproducibility seed
    """
    # TODO: Build and persist a processed dataset split to disk.
    #
    #   Step 1 — seed RNGs for reproducibility:
    #       random.seed(seed);  np.random.seed(seed)
    #
    #   Step 2 — load data_config.yaml; construct AudioPreprocessor.
    #
    #   Step 3 — resolve & create output directory:
    #       out_dir = Path(dcfg["paths"]["processed_dir"]) / split
    #       out_dir.mkdir(parents=True, exist_ok=True)
    #
    #   Step 4 — collect clean WAV files:
    #       Root = dcfg["clean"][0]["local_path"] / dcfg["clean"][0]["splits"][split_key]
    #       Find all *.wav recursively. Raise FileNotFoundError if empty.
    #       For split=="train" or "val", use key "train" from VIVOS; split the list:
    #           val_cut = int(N * val_ratio / (train_ratio + val_ratio))
    #           val files   = clean_files[:val_cut]
    #           train files = clean_files[val_cut:]
    #
    #   Step 5 — collect noise WAV files from all entries in dcfg["noise"].
    #       Raise FileNotFoundError if empty.
    #
    #   Step 6 — main loop:
    #       for each clean_path:
    #           sample `pairs_per_clean` noise files with random.sample()
    #           for each noise_path:
    #               noisy_segs, clean_segs = pre.process_pair(clean_path, noise_path)
    #               save to out_dir/batch_{idx:06d}.npz  (keys: "noisy", "clean")
    #               use np.savez_compressed(); wrap in try/except; log warnings.
    #
    #   Step 7 — log total file count.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(description="Build preprocessed dataset splits.")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    ap.add_argument("--pairs-per-clean", type=int, default=3, metavar="N")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits:
        logger.info("=== %s ===", s)
        build_dataset(split=s, pairs_per_clean=args.pairs_per_clean, seed=args.seed)
