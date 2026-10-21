"""
Phase 2 — Model architectures and efficiency helpers.

Variants (all return an uncompiled keras.Model):
  - build_baseline()   : standard Conv2D blocks            (~0.1M params)
  - build_separable()  : depthwise-separable Conv blocks   (fewer params)
  - build_mobilenet()  : tiny MobileNetV2 (optional, named baseline)

Efficiency helpers:
  - count_params(model)
  - measure_cpu_latency(model)  : mean single-image inference time
  - model_file_size(path)

Design notes (cite these in the report):
  * GlobalAveragePooling instead of Flatten keeps the dense head tiny.
  * BatchNorm after every conv stabilizes training at lr=1e-3.
  * Dropout 0.3 before the classifier controls the train/val gap.
  * Target: well under 1M params, sub-10 ms CPU per-character inference.
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 26
INPUT_SHAPE = (28, 28, 1)


# Architectures
def _conv_block(x, filters, separable=False):
    Conv = layers.SeparableConv2D if separable else layers.Conv2D
    x = Conv(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    return x


def _head(x, n_classes, dropout=0.3, dense_units=128):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(n_classes, activation="softmax")(x)
    return x


def build_baseline(n_classes=NUM_CLASSES, filters=(32, 64, 128),
                   dropout=0.3, dense_units=128) -> keras.Model:
    """Standard convolutions. Tweak `filters` / `dropout` to experiment."""
    inp = layers.Input(INPUT_SHAPE)
    x = inp
    for f in filters:
        x = _conv_block(x, f, separable=False)
    out = _head(x, n_classes, dropout, dense_units)
    return keras.Model(inp, out, name="baseline_cnn")


def build_separable(n_classes=NUM_CLASSES, filters=(32, 64, 128),
                    dropout=0.3, dense_units=128) -> keras.Model:
    """Depthwise-separable convolutions (MobileNet-style efficiency)."""
    inp = layers.Input(INPUT_SHAPE)
    # first block standard: separable convs on 1-channel input save nothing
    x = _conv_block(inp, filters[0], separable=False)
    for f in filters[1:]:
        x = _conv_block(x, f, separable=True)
    out = _head(x, n_classes, dropout, dense_units)
    return keras.Model(inp, out, name="separable_cnn")


def build_mobilenet(n_classes=NUM_CLASSES, alpha=0.35) -> keras.Model:
    """
    Optional named baseline: tiny MobileNetV2.
    EMNIST 28x28x1 is upsampled to 32x32x3 (MobileNet's minimum input).
    Overkill for this task — included for the efficiency comparison only.
    """
    inp = layers.Input(INPUT_SHAPE)
    x = layers.Resizing(32, 32)(inp)
    x = layers.Concatenate()([x, x, x])  # 1 -> 3 channels
    base = keras.applications.MobileNetV2(
        input_shape=(32, 32, 3), alpha=alpha,
        include_top=False, weights=None)  # train from scratch
    x = base(x)
    out = _head(x, n_classes)
    return keras.Model(inp, out, name=f"mobilenetv2_{alpha}")


VARIANTS = {
    "baseline": build_baseline,
    "separable": build_separable,
    "mobilenet": build_mobilenet,
}


# Compile / train helpers
def compile_model(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def default_callbacks(name: str, artifacts_dir: Path, logs_dir: Path):
    """EarlyStopping + best-checkpoint + TensorBoard (one log dir per run)."""
    run_dir = logs_dir / name
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            str(artifacts_dir / f"{name}.keras"),
            monitor="val_loss", save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=str(run_dir)),
    ]


# Efficiency measurement
def measure_cpu_latency(model: keras.Model, n_warmup: int = 10,
                        n_runs: int = 200) -> float:
    """
    Mean single-image CPU inference latency in milliseconds.
    Uses model(x) directly (not .predict) to avoid per-call pipeline overhead,
    which is the honest proxy for per-character inference on device.
    """
    x = tf.random.uniform((1, *INPUT_SHAPE))
    infer = tf.function(lambda t: model(t, training=False))
    for _ in range(n_warmup):
        infer(x)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        infer(x)
    return (time.perf_counter() - t0) / n_runs * 1000.0


def model_file_size_kb(path: Path) -> float:
    return Path(path).stat().st_size / 1024.0


def efficiency_row(name: str, model: keras.Model, keras_path: Path,
                   test_acc: float, macro_f1: float) -> dict:
    return {
        "variant": name,
        "params": model.count_params(),
        "size_kb": round(model_file_size_kb(keras_path), 1),
        "test_acc": round(test_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "cpu_latency_ms": round(measure_cpu_latency(model), 3),
    }
