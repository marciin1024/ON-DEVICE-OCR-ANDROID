"""
Phase 1 — EMNIST data pipeline.

Single place that defines:
  - loading EMNIST/letters via tensorflow_datasets
  - the orientation fix (raw EMNIST is transposed/flipped!)
  - normalization to [0,1], relabel 1..26 -> 0..25
  - train/val split (90/10), test split untouched
  - augmentation (TRAIN ONLY)

Usage:
    from src.data import get_datasets, NUM_CLASSES
    ds_train, ds_val, ds_test, info = get_datasets()
"""

import json
import string
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

NUM_CLASSES = 26
BATCH_SIZE = 128
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10  # carved from tfds 'train'; tfds 'test' is broken (see get_datasets)
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# Orientation fix — DO NOT CHANGE without re-running the visual sanity check
# (notebooks/01_data.ipynb). Raw EMNIST images are stored transposed; the
# standard fix is: transpose == rot90(k=3) + flip_left_right.
def fix_orientation(img: tf.Tensor) -> tf.Tensor:
    """img: (28, 28, 1). Returns the image readable in normal orientation."""
    img = tf.image.rot90(img, k=3)
    img = tf.image.flip_left_right(img)
    return img


def normalize(img: tf.Tensor) -> tf.Tensor:
    """uint8 [0,255] -> float32 [0,1], shape (28,28,1)."""
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (28, 28, 1))
    return img

# Augmentation — applied ONLY to the training set. Narrows the domain gap
# between handwritten EMNIST and printed camera text.
def build_augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            layers.RandomRotation(0.04, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.10, 0.10, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.10, fill_mode="constant", fill_value=0.0),
        ],
        name="augmenter",
    )


def add_noise(img: tf.Tensor, stddev: float = 0.05) -> tf.Tensor:
    """Light Gaussian noise; output clipped back to [0,1]."""
    noise = tf.random.normal(tf.shape(img), stddev=stddev)
    return tf.clip_by_value(img + noise, 0.0, 1.0)


def random_thickness(img: tf.Tensor) -> tf.Tensor:
    """
    Random morphological dilation/erosion via max/min pooling (pure TF, so it
    stays inside tf.data). p=1/3 dilate, 1/3 erode, 1/3 unchanged.
    """
    r = tf.random.uniform([], 0, 1)
    x = tf.expand_dims(img, 0)  # (1,28,28,1)
    dilated = tf.nn.max_pool2d(x, ksize=2, strides=1, padding="SAME")
    eroded = -tf.nn.max_pool2d(-x, ksize=2, strides=1, padding="SAME")
    x = tf.cond(r < 1 / 3, lambda: dilated,
                lambda: tf.cond(r < 2 / 3, lambda: eroded, lambda: x))
    return tf.squeeze(x, 0)


# Mapping functions
def _prep_eval(img, lab):
    """Deterministic preprocessing for val/test: orient + normalize + relabel."""
    img = fix_orientation(img)
    img = normalize(img)
    return img, lab - 1  # 1..26 -> 0..25


_aug = None  # lazy singleton, created inside graph context


def _prep_train(img, lab):
    global _aug
    if _aug is None:
        _aug = build_augmenter()
    img = fix_orientation(img)
    img = normalize(img)
    img = _aug(tf.expand_dims(img, 0), training=True)[0]
    img = random_thickness(img)
    img = add_noise(img)
    return img, lab - 1


# Public entry point
def get_datasets(batch_size: int = BATCH_SIZE):
    """
    Returns (ds_train, ds_val, ds_test, info), all batched + prefetched.
    Train: augmented + shuffled. Val/test: deterministic.

    IMPORTANT — tfds 'emnist/letters' test split is BROKEN: it contains only
    classes a-r (800 each) + s (400); letters t-z are entirely absent
    (tfds ships 14,800 test examples vs 20,800 in the original EMNIST,
    truncated after sorting by class). We therefore IGNORE the tfds test
    split and carve val (10%) and test (10%) from the tfds train split,
    which is complete and balanced (~3,415 per class, verified in 01_data).
    The carved test split is used exactly once, for final evaluation.
    """
    ds_train_raw, info = tfds.load(
        "emnist/letters",
        split="train",
        as_supervised=True,
        with_info=True,
    )

    n_total = info.splits["train"].num_examples
    n_val = int(n_total * VAL_FRACTION)
    n_test = int(n_total * TEST_FRACTION)

    # Deterministic carve-out (tfds on-disk order is fixed and class-shuffled):
    # [0, n_val) -> val | [n_val, n_val+n_test) -> test | rest -> fit
    ds_val_raw = ds_train_raw.take(n_val)
    ds_test_raw = ds_train_raw.skip(n_val).take(n_test)
    ds_fit_raw = ds_train_raw.skip(n_val + n_test)

    ds_train = (
        ds_fit_raw.map(_prep_train, num_parallel_calls=AUTOTUNE)
        .shuffle(10_000, seed=SEED)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    ds_val = (
        ds_val_raw.map(_prep_eval, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    ds_test = (
        ds_test_raw.map(_prep_eval, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return ds_train, ds_val, ds_test, info


def write_label_artifacts(artifacts_dir: Path = ARTIFACTS) -> None:
    """Writes label_map.json (index -> char) and labels.txt (one char/line)."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    label_map = {i: c for i, c in enumerate(string.ascii_lowercase)}
    with open(artifacts_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(artifacts_dir / "labels.txt", "w") as f:
        f.write("\n".join(string.ascii_lowercase) + "\n")


def dataset_stats(info) -> dict:
    """Stats to cite in the report (Phase 1 gotcha #4)."""
    n_total = info.splits["train"].num_examples
    n_val = int(n_total * VAL_FRACTION)
    n_test = int(n_total * TEST_FRACTION)
    return {
        "dataset": "emnist/letters (tfds train split only; tfds test is broken)",
        "num_classes": NUM_CLASSES,
        "source_total": n_total,
        "train_fit": n_total - n_val - n_test,
        "val": n_val,
        "test": n_test,
        "image_shape": [28, 28, 1],
        "label_range_raw": "1..26 (relabeled to 0..25)",
        "note": "tfds 'test' split excluded: contains only classes a-s "
                "(t-z absent); val/test carved 10%/10% from tfds 'train'.",
    }


def check_split_integrity(ds, name: str, num_classes: int = NUM_CLASSES):
    """Asserts a batched eval split contains all classes. Returns bincounts."""
    import numpy as np
    ys = np.concatenate([y.numpy() for _, y in ds])
    counts = np.bincount(ys, minlength=num_classes)
    missing = [i for i, c in enumerate(counts) if c == 0]
    assert not missing, f"{name}: missing classes {missing}!"
    print(f"{name}: {len(ys)} examples, per-class min={counts.min()} "
          f"max={counts.max()} -> all {num_classes} classes present")
    return counts


if __name__ == "__main__":
    write_label_artifacts()
    print("label_map.json + labels.txt written to", ARTIFACTS)
