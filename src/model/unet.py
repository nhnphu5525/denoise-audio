import tensorflow as tf
from tensorflow.keras import layers, models

class UNetDenoiser:
    def __init__(self, input_shape=(256, 256, 1), base_filters=64):
        self.input_shape = input_shape
        self.base_filters = base_filters

    def _conv_block(self, x, filters, name):
        x = layers.Conv2D(
            filters, 3, padding="same", activation="relu", name=f"{name}_conv1"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Conv2D(
            filters, 3, padding="same", activation="relu", name=f"{name}_conv2"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        return x

    def _encoder_block(self, x, filters, name):
        f = self._conv_block(x, filters, name)
        p = layers.MaxPooling2D((2, 2), name=f"{name}_pool")(f)
        return f, p

    def _decoder_block(self, x, skip, filters, name):
        x = layers.Conv2DTranspose(
            filters, (2, 2), strides=2, padding="same", name=f"{name}_up"
        )(x)
        x = layers.Concatenate(name=f"{name}_concat")([x, skip])
        x = self._conv_block(x, filters, name)
        return x

    def build(self):
        inputs = layers.Input(shape=self.input_shape, name="noisy_spectrogram")

        # Encoder
        s1, p1 = self._encoder_block(inputs, self.base_filters, "enc1")
        s2, p2 = self._encoder_block(p1, self.base_filters * 2, "enc2")
        s3, p3 = self._encoder_block(p2, self.base_filters * 4, "enc3")
        s4, p4 = self._encoder_block(p3, self.base_filters * 8, "enc4")

        # Bottleneck
        b1 = self._conv_block(p4, self.base_filters * 16, "bottleneck")

        # Decoder
        d4 = self._decoder_block(b1, s4, self.base_filters * 8, "dec4")
        d3 = self._decoder_block(d4, s3, self.base_filters * 4, "dec3")
        d2 = self._decoder_block(d3, s2, self.base_filters * 2, "dec2")
        d1 = self._decoder_block(d2, s1, self.base_filters, "dec1")

        # Output
        outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", name="mask")(d1)

        model = models.Model(inputs, outputs, name="UNet_Denoise")
        return model

def build_unet_denoise(input_shape=(256, 256, 1), base_filters=64):
    return UNetDenoiser(input_shape, base_filters).build()

if __name__ == "__main__":
    model = build_unet_denoise()
    model.summary()
