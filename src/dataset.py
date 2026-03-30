"""
WikiArt dataset loader and augmentation pipeline.

Expects data to be pre-split into folders by src/split_dataset.py:
    data/
      train/<artist_name>/<image>.jpg
      validation/<artist_name>/<image>.jpg
      test/<artist_name>/<image>.jpg
"""

import tensorflow as tf


# For future flexibility, we can define augmentation parameters in the config file
def build_augmentation_pipeline(config):
    """Return a Sequential augmentation model for training."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(config.get("rotation_range", 0.1)),
        tf.keras.layers.RandomZoom(config.get("zoom_range", 0.1)),
        tf.keras.layers.RandomContrast(config.get("contrast_range", 0.1)),
    ])


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
        seed=42,
    )

    class_names = ds.class_names

    # Raw pixels [0, 255] are passed through — each model handles its own
    # normalization internally (Rescaling layer for baseline, preprocess_input
    # Lambda for transfer learning backbones).

    # Cache after normalization so images are only read from disk once.
    # Disabled for large resolutions (>300px) to avoid OOM — decoded float32
    # tensors are ~46GB at 512x512 for the full training set.
    if img_size[0] <= 300:
        ds = ds.cache()

    # Apply augmentation on training set only
    if augment:
        aug = build_augmentation_pipeline(config or {})
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Prefetch for performance(load data while the model is training on the current batch)
    return ds.prefetch(tf.data.AUTOTUNE), class_names
