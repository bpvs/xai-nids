"""
1D-CNN model for NSL-KDD intrusion detection.

Three convolutional blocks (Conv → BN → ReLU → Pooling) followed by
a dense classifier head. Optimized for Apple Silicon via the Metal plugin.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from .base import ModelWrapper


def configure_gpu():
    """Detect and configure GPU (Apple Metal, CUDA, or CPU fallback)."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPU detected: {gpus[0].name}")
        tf.keras.backend.set_floatx("float32")
    else:
        print("  No GPU detected — running on CPU")
    return len(gpus) > 0


class CNNModel(ModelWrapper):
    """1D-CNN binary classifier."""

    name = "cnn"

    def build(self, n_features):
        self.n_features = n_features
        self.model = keras.Sequential([
            layers.Conv1D(64, 3, padding="same", input_shape=(n_features, 1)),
            layers.BatchNormalization(), layers.ReLU(), layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, padding="same"),
            layers.BatchNormalization(), layers.ReLU(), layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, padding="same"),
            layers.BatchNormalization(), layers.ReLU(), layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"), layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None, fast_mode=False):
        epochs = 10 if fast_mode else 30
        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        history = self.model.fit(
            X_train.reshape(-1, X_train.shape[1], 1), y_train,
            validation_data=(X_val.reshape(-1, X_val.shape[1], 1), y_val),
            epochs=epochs, batch_size=256, callbacks=cbs, verbose=1,
        )
        # Training history plot
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
        a1.plot(history.history["loss"], label="Train"); a1.plot(history.history["val_loss"], label="Val")
        a1.set_title("Loss"); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(history.history["accuracy"], label="Train"); a2.plot(history.history["val_accuracy"], label="Val")
        a2.set_title("Accuracy"); a2.legend(); a2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/training_history.png", dpi=150)
        plt.close()
        return self

    def predict_proba(self, X):
        """Return P(attack) — uses model() directly for GPU acceleration."""
        X_3d = tf.constant(X.reshape(-1, X.shape[-1], 1), dtype=tf.float32)
        return self.model(X_3d, training=False).numpy().flatten()

    @property
    def supports_gradients(self) -> bool:
        return True

    def get_input_gradients(self, X):
        """∂output/∂input — used by saliency map."""
        x_tensor = tf.constant(X.reshape(-1, X.shape[-1], 1), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            pred = self.model(x_tensor, training=False)
        grads = tape.gradient(pred, x_tensor).numpy()
        return grads.reshape(grads.shape[0], grads.shape[1])  # (n_samples, n_features)

    def get_conv_grad_model(self):
        """Build a sub-model exposing the last Conv1D layer's activations.
        Used by Grad-CAM."""
        conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, layers.Conv1D):
                conv_layer = layer
                break
        if conv_layer is None:
            return None

        inp = keras.Input(shape=(self.n_features, 1))
        x = inp
        conv_out = None
        for layer in self.model.layers:
            x = layer(x)
            if layer == conv_layer:
                conv_out = x
        return keras.Model(inputs=inp, outputs=[conv_out, x]), conv_layer.name