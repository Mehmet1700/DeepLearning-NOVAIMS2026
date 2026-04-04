"""Dataset loading and cache management for WikiArt splits."""

import hashlib
import json
import re
from pathlib import Path

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "outputs" / "cache"
CACHE_SEED = 42
CACHE_VERSION = 1


# For future flexibility, we can define augmentation parameters in the config file
def build_augmentation_pipeline(config):
    """Return a Sequential augmentation model for training."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(config.get("rotation_range", 0.1)),
        tf.keras.layers.RandomZoom(config.get("zoom_range", 0.1)),
        tf.keras.layers.RandomContrast(config.get("contrast_range", 0.1)),
    ])


def _resolve_split_dir(split_dir):
    """Return the absolute split directory path."""
    return Path(split_dir).resolve()


def _split_file_signature(split_dir):
    """Return a deterministic fingerprint for the current split contents."""
    file_records = []
    for file_path in sorted(path for path in split_dir.rglob("*") if path.is_file()):
        stat = file_path.stat()
        file_records.append({
            "path": file_path.relative_to(split_dir).as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })

    payload = json.dumps(file_records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_cache_label(split_dir):
    """Return a readable cache label derived from the split directory name."""
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", split_dir.name).strip("_")
    return sanitized or "split"


def _build_cache_path(split_dir, img_size, batch_size, shuffle, seed=CACHE_SEED):
    """Return the on-disk cache path for a split and loader configuration."""
    resolved_split_dir = _resolve_split_dir(split_dir)
    cache_payload = {
        "cache_version": CACHE_VERSION,
        "split_dir": str(resolved_split_dir),
        "img_size": list(img_size),
        "batch_size": batch_size,
        "shuffle": bool(shuffle),
        "seed": seed,
        "fingerprint": _split_file_signature(resolved_split_dir),
    }
    cache_digest = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    label = _split_cache_label(resolved_split_dir)
    return CACHE_ROOT / f"{label}_{img_size[0]}x{img_size[1]}_b{batch_size}_shuffle{int(bool(shuffle))}_{cache_digest}"


def load_split(split_dir, img_size, batch_size, augment=False, config=None):
    """
    Build a tf.data.Dataset from a pre-split folder.

    Args:
        split_dir:   Path to the split folder (e.g. "data/train").
                     Must contain one sub-folder per class.
        img_size:    Tuple (height, width).
        batch_size:  Batch size.
        augment:     Whether to apply augmentation (training only).
        config:      Optional dict with augmentation parameters.

    Returns:
        tf.data.Dataset yielding (image_tensor, label) batches,
        and the list of class names inferred from the folder structure.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=augment,   # only shuffle for training
        seed=CACHE_SEED,
    )

    class_names = ds.class_names

    # Raw pixels [0, 255] are passed through — each model handles its own
    # normalization internally (Rescaling layer for baseline, preprocess_input
    # Lambda for transfer learning backbones).

    # Cache resized and batched tensors before augmentation so later epochs
    # skip repeated disk reads and image decoding work.
    # Disabled for large resolutions (>300px) to avoid OOM — decoded float32
    # tensors are ~46GB at 512x512 for the full training set.
    if img_size[0] <= 300:
        cache_path = _build_cache_path(split_dir, img_size, batch_size, shuffle=augment)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(cache_path))

    # Apply augmentation on training set only
    if augment:
        aug = build_augmentation_pipeline(config or {})
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Prefetch for performance(load data while the model is training on the current batch)
    return ds.prefetch(tf.data.AUTOTUNE), class_names
