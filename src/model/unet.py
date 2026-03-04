"""
UNet-based spectrogram mask estimator for speech denoising.

Mask-based approach (Ideal Ratio Mask — IRM):
    1. Input  : noisy magnitude spectrogram  |Y|  shape (F, T, 1)
    2. Output : soft mask  M ∈ (0, 1)        shape (F, T, 1)
    3. Estimate: clean spectrogram  |S_hat| = M * |Y|

Input shape convention (real-time)
------------------------------------
    F = n_fft // 2 = 256  (Nyquist bin cropped, n_fft=512)
    T = 32 frames  → latency = 32 × (hop_length / sample_rate)
                             = 32 × (128 / 16000) = 256 ms
    C = 1  (magnitude, single channel)
    → default input_shape = (256, 32, 1)

    At inference: maintain a sliding buffer of 32 frames; shift by 1 frame
    (8 ms) each step and output only the last denoised frame.

Asymmetric pooling for real-time
----------------------------------
    Standard (2,2) pooling halves both F and T axes — with T=32 this leaves
    only T=2 at the bottleneck after 4 encoder stages.

    Instead we use (2,1) pooling: downsample only the frequency axis F,
    keeping the time axis T intact throughout the network.

        After enc1 : (128, 32, ...)
        After enc2 : ( 64, 32, ...)
        After enc3 : ( 32, 32, ...)
        After enc4 : ( 16, 32, ...)
        Bottleneck  : ( 16, 32, base_filters×16)

Architecture improvements
--------------------------
    • Asymmetric pool (2,1) / upsample (2,1) → real-time compatible
    • SpatialDropout2D in every conv block   → regularisation
    • Squeeze-and-Excitation at bottleneck   → channel-wise attention
    • Mask × noisy inside model              → loss computed on clean estimate
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class UNetDenoiser:
    """
    U-Net denoiser that predicts an IRM and returns the masked spectrogram.

    Parameters
    ----------
    input_shape   : (F, T, 1) magnitude spectrogram shape after padding/crop.
    base_filters  : number of filters in the first encoder block (doubles each level).
    dropout_rate  : SpatialDropout2D rate applied after every conv block.
    se_ratio      : squeeze ratio for the Squeeze-and-Excitation bottleneck block.
    """

    def __init__(
        self,
        input_shape: tuple = (256, 32, 1),
        base_filters: int = 64,
        dropout_rate: float = 0.3,
        se_ratio: int = 16,
    ):
        self.input_shape = input_shape
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate
        self.se_ratio = se_ratio

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _conv_block(self, x, filters: int, name: str):
        """Two Conv2D-BN-ReLU layers followed by SpatialDropout2D."""
        x = layers.Conv2D(
            filters, 3, padding="same", activation="relu", name=f"{name}_conv1"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Conv2D(
            filters, 3, padding="same", activation="relu", name=f"{name}_conv2"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.SpatialDropout2D(self.dropout_rate, name=f"{name}_drop")(x)
        return x

    def _encoder_block(self, x, filters: int, name: str):
        """
        Conv block + asymmetric (2,1) max-pool.
        Downsamples the frequency axis (F) only; time axis (T) is preserved.
        Returns (skip, pooled).
        """
        skip = self._conv_block(x, filters, name)
        pooled = layers.MaxPooling2D((2, 1), name=f"{name}_pool")(skip)
        return skip, pooled

    def _decoder_block(self, x, skip, filters: int, name: str):
        """
        Asymmetric (2,1) transposed conv up-sample → concatenate skip → conv block.
        Upsamples the frequency axis (F) only; time axis (T) stays unchanged.
        """
        x = layers.Conv2DTranspose(
            filters, (2, 1), strides=(2, 1), padding="same", name=f"{name}_up"
        )(x)
        x = layers.Concatenate(name=f"{name}_concat")([x, skip])
        x = self._conv_block(x, filters, name)
        return x

    def _se_block(self, x, filters: int, name: str):
        """
        Squeeze-and-Excitation channel attention.
        Recalibrates channel responses via global average pooling + FC gates.
        """
        se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
        se = layers.Dense(
            max(filters // self.se_ratio, 1), activation="relu", name=f"{name}_fc1"
        )(se)
        se = layers.Dense(filters, activation="sigmoid", name=f"{name}_fc2")(se)
        se = layers.Reshape((1, 1, filters), name=f"{name}_reshape")(se)
        return layers.Multiply(name=f"{name}_scale")([x, se])

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def build(self) -> tf.keras.Model:
        """
        Build the full IRM U-Net with asymmetric pooling.

        Input  : noisy magnitude spectrogram  shape=(F, T, 1)  e.g. (256, 32, 1)
        Output : estimated clean spectrogram  shape=(F, T, 1)
                 computed as  mask (sigmoid) × noisy input

        Latency: T × hop_length / sample_rate  →  32 × 128/16000 = 256 ms
        """
        noisy = layers.Input(shape=self.input_shape, name="noisy_spectrogram")

        # ---- Encoder ----
        s1, p1 = self._encoder_block(noisy, self.base_filters * 1, "enc1")
        s2, p2 = self._encoder_block(p1,    self.base_filters * 2, "enc2")
        s3, p3 = self._encoder_block(p2,    self.base_filters * 4, "enc3")
        s4, p4 = self._encoder_block(p3,    self.base_filters * 8, "enc4")

        # ---- Bottleneck + SE attention ----
        b = self._conv_block(p4, self.base_filters * 16, "bottleneck")
        b = self._se_block(b,   self.base_filters * 16, "bottleneck_se")

        # ---- Decoder ----
        d4 = self._decoder_block(b,  s4, self.base_filters * 8, "dec4")
        d3 = self._decoder_block(d4, s3, self.base_filters * 4, "dec3")
        d2 = self._decoder_block(d3, s2, self.base_filters * 2, "dec2")
        d1 = self._decoder_block(d2, s1, self.base_filters * 1, "dec1")

        # ---- IRM output: mask in (0, 1) ----
        mask = layers.Conv2D(1, (1, 1), activation="sigmoid", name="mask")(d1)

        # ---- Apply mask → estimated clean spectrogram ----
        clean_estimate = layers.Multiply(name="clean_estimate")([mask, noisy])

        return models.Model(noisy, clean_estimate, name="UNet_IRM_Denoise")

    def build_mask_only(self) -> tf.keras.Model:
        """
        Variant that returns the raw mask instead of the masked spectrogram.
        Useful when you want direct access to mask values during inference.
        """
        noisy = layers.Input(shape=self.input_shape, name="noisy_spectrogram")

        s1, p1 = self._encoder_block(noisy, self.base_filters * 1, "enc1")
        s2, p2 = self._encoder_block(p1,    self.base_filters * 2, "enc2")
        s3, p3 = self._encoder_block(p2,    self.base_filters * 4, "enc3")
        s4, p4 = self._encoder_block(p3,    self.base_filters * 8, "enc4")

        b = self._conv_block(p4, self.base_filters * 16, "bottleneck")
        b = self._se_block(b,   self.base_filters * 16, "bottleneck_se")

        d4 = self._decoder_block(b,  s4, self.base_filters * 8, "dec4")
        d3 = self._decoder_block(d4, s3, self.base_filters * 4, "dec3")
        d2 = self._decoder_block(d3, s2, self.base_filters * 2, "dec2")
        d1 = self._decoder_block(d2, s1, self.base_filters * 1, "dec1")

        mask = layers.Conv2D(1, (1, 1), activation="sigmoid", name="mask")(d1)

        return models.Model(noisy, mask, name="UNet_Mask_Only")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_unet_denoise(
    input_shape: tuple = (256, 32, 1),
    base_filters: int = 64,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Return the full IRM U-Net (clean_estimate = mask × noisy).
    Default input_shape=(256, 32, 1) → 256 ms latency at 16 kHz / hop 128.
    """
    return UNetDenoiser(input_shape, base_filters, dropout_rate).build()


if __name__ == "__main__":
    model = build_unet_denoise()
    model.summary(expand_nested=True)
