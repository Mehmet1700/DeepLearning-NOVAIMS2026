"""Dataset loading and cache management for WikiArt splits."""

import hashlib
import json
import re
from pathlib import Path

import tensorflow as tf

from src import model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "outputs" / "cache"
CACHE_SEED = 42
CACHE_VERSION = 1


def _coerce_float(value, default=0.0):
    """Return a float config value, treating nulls as defaults."""
    if value is None:
        return float(default)
    return float(value)


def _coerce_range(value, default):
    """Return a two-item float range from a scalar or YAML list."""
    if value is None:
        return tuple(float(v) for v in default)
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a two-item range, got: {value!r}")
        return float(value[0]), float(value[1])
    magnitude = float(value)
    return -magnitude, magnitude


class RandomResizedCrop(tf.keras.layers.Layer):
    """Randomly crop a smaller view and resize it back to the input size."""

    def __init__(self, scale_range=(0.8, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.scale_range = _coerce_range(scale_range, (0.8, 1.0))

    def get_config(self):
        config = super().get_config()
        config.update({"scale_range": list(self.scale_range)})
        return config

    def call(self, images, training=None):
        if training is None:
            training = False
        if training is False:
            return images

        images = tf.cast(images, tf.float32)
        batch_size = tf.shape(images)[0]
        target_size = tf.shape(images)[1:3]
        min_scale, max_scale = self.scale_range

        scales = tf.random.uniform(
            shape=[batch_size],
            minval=min_scale,
            maxval=max_scale,
            dtype=tf.float32,
        )
        max_offsets = 1.0 - scales
        offset_y = tf.random.uniform([batch_size], 0.0, 1.0) * max_offsets
        offset_x = tf.random.uniform([batch_size], 0.0, 1.0) * max_offsets

        # crop_and_resize expects normalized box coordinates per image.
        boxes = tf.stack(
            [
                offset_y,
                offset_x,
                offset_y + scales,
                offset_x + scales,
            ],
            axis=1,
        )
        box_indices = tf.range(batch_size)
        return tf.image.crop_and_resize(images, boxes, box_indices, target_size)


class ArtColorJitter(tf.keras.layers.Layer):
    """Apply mild color perturbations on normalized RGB images."""

    def __init__(
        self,
        brightness_range=0.0,
        contrast_range=0.0,
        saturation_range=0.0,
        color_jitter_strength=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.brightness_range = _coerce_float(brightness_range)
        self.contrast_range = _coerce_float(contrast_range)
        self.saturation_range = _coerce_float(saturation_range)
        self.color_jitter_strength = _coerce_float(color_jitter_strength)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "brightness_range": self.brightness_range,
                "contrast_range": self.contrast_range,
                "saturation_range": self.saturation_range,
                "color_jitter_strength": self.color_jitter_strength,
            }
        )
        return config

    def call(self, images, training=None):
        if training is None:
            training = False
        if training is False:
            return images

        images = tf.cast(images, tf.float32)
        batch_size = tf.shape(images)[0]

        if self.brightness_range > 0.0:
            delta = tf.random.uniform(
                [batch_size, 1, 1, 1],
                minval=-self.brightness_range,
                maxval=self.brightness_range,
            )
            images = images + delta

        if self.contrast_range > 0.0:
            factors = tf.random.uniform(
                [batch_size, 1, 1, 1],
                minval=1.0 - self.contrast_range,
                maxval=1.0 + self.contrast_range,
            )
            mean = tf.reduce_mean(images, axis=[1, 2], keepdims=True)
            images = (images - mean) * factors + mean

        if self.saturation_range > 0.0:
            factors = tf.random.uniform(
                [batch_size, 1, 1, 1],
                minval=1.0 - self.saturation_range,
                maxval=1.0 + self.saturation_range,
            )
            grayscale = tf.image.rgb_to_grayscale(images)
            images = grayscale + (images - grayscale) * factors

        if self.color_jitter_strength > 0.0:
            channel_gains = tf.random.uniform(
                [batch_size, 1, 1, 3],
                minval=1.0 - self.color_jitter_strength,
                maxval=1.0 + self.color_jitter_strength,
            )
            images = images * channel_gains

        return tf.clip_by_value(images, 0.0, 1.0)


class RandomSharpness(tf.keras.layers.Layer):
    """Sharpen or slightly blur images with a small random factor."""

    def __init__(self, sharpness_range=0.0, **kwargs):
        super().__init__(**kwargs)
        self.sharpness_range = _coerce_float(sharpness_range)
        kernel = (
            tf.constant(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=tf.float32,
            )
            / 9.0
        )
        self.blur_kernel = tf.reshape(kernel, [3, 3, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"sharpness_range": self.sharpness_range})
        return config

    def call(self, images, training=None):
        if training is None:
            training = False
        if training is False or self.sharpness_range <= 0.0:
            return images

        images = tf.cast(images, tf.float32)
        channels = tf.shape(images)[-1]
        blur_kernel = tf.tile(self.blur_kernel, [1, 1, channels, 1])
        padded = tf.pad(images, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        blurred = tf.nn.depthwise_conv2d(
            padded, blur_kernel, strides=[1, 1, 1, 1], padding="VALID"
        )
        factors = tf.random.uniform(
            [tf.shape(images)[0], 1, 1, 1],
            minval=1.0 - self.sharpness_range,
            maxval=1.0 + self.sharpness_range,
        )
        sharpened = blurred + factors * (images - blurred)
        return tf.clip_by_value(sharpened, 0.0, 1.0)


class RandomErasing(tf.keras.layers.Layer):
    """Randomly erase a small rectangular region from an image."""

    def __init__(self, probability=0.0, area_range=(0.02, 0.10), **kwargs):
        super().__init__(**kwargs)
        self.probability = _coerce_float(probability)
        self.area_range = _coerce_range(area_range, (0.02, 0.10))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "probability": self.probability,
                "area_range": list(self.area_range),
            }
        )
        return config

    def _erase_image(self, image):
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        total_area = tf.cast(height * width, tf.float32)

        erase_fraction = tf.random.uniform(
            shape=[],
            minval=self.area_range[0],
            maxval=self.area_range[1],
            dtype=tf.float32,
        )
        aspect_ratio = tf.exp(
            tf.random.uniform(
                shape=[],
                minval=tf.math.log(0.5),
                maxval=tf.math.log(2.0),
                dtype=tf.float32,
            )
        )
        erase_area = erase_fraction * total_area
        erase_height = tf.cast(
            tf.clip_by_value(
                tf.round(tf.sqrt(erase_area * aspect_ratio)),
                1.0,
                tf.cast(height, tf.float32),
            ),
            tf.int32,
        )
        erase_width = tf.cast(
            tf.clip_by_value(
                tf.round(tf.sqrt(erase_area / aspect_ratio)),
                1.0,
                tf.cast(width, tf.float32),
            ),
            tf.int32,
        )

        max_offset_y = tf.maximum(height - erase_height, 0)
        max_offset_x = tf.maximum(width - erase_width, 0)
        offset_y = tf.random.uniform([], 0, max_offset_y + 1, dtype=tf.int32)
        offset_x = tf.random.uniform([], 0, max_offset_x + 1, dtype=tf.int32)

        # Build a single rectangular mask and replace that region with noise.
        mask = tf.pad(
            tf.ones([erase_height, erase_width, 1], dtype=tf.float32),
            [
                [offset_y, height - erase_height - offset_y],
                [offset_x, width - erase_width - offset_x],
                [0, 0],
            ],
        )
        fill = tf.random.uniform(tf.shape(image), 0.0, 1.0, dtype=tf.float32)
        return image * (1.0 - mask) + fill * mask

    def _maybe_erase_image(self, image):
        return tf.cond(
            tf.random.uniform([]) < self.probability,
            lambda: self._erase_image(image),
            lambda: image,
        )

    def call(self, images, training=None):
        if training is None:
            training = False
        if training is False or self.probability <= 0.0:
            return images

        images = tf.cast(images, tf.float32)
        return tf.map_fn(
            self._maybe_erase_image, images, fn_output_signature=tf.float32
        )


class ClippedGaussianNoise(tf.keras.layers.Layer):
    """Add Gaussian noise and clamp normalized pixels back to [0, 1]."""

    def __init__(self, stddev=0.0, **kwargs):
        super().__init__(**kwargs)
        self.stddev = _coerce_float(stddev)

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config

    def call(self, images, training=None):
        if training is None:
            training = False
        if training is False or self.stddev <= 0.0:
            return images

        images = tf.cast(images, tf.float32)
        noisy = images + tf.random.normal(tf.shape(images), stddev=self.stddev)
        return tf.clip_by_value(noisy, 0.0, 1.0)


def build_augmentation_pipeline(config):
    """Return a config-driven Sequential augmentation model for training."""
    layers = [tf.keras.layers.Rescaling(1.0 / 255.0)]

    if config.get("random_crop", False):
        layers.append(RandomResizedCrop(config.get("crop_scale_range", (0.8, 1.0))))

    horizontal_flip = bool(config.get("horizontal_flip", True))
    vertical_flip = bool(config.get("vertical_flip", False))
    flip_mode = None
    if horizontal_flip and vertical_flip:
        flip_mode = "horizontal_and_vertical"
    elif horizontal_flip:
        flip_mode = "horizontal"
    elif vertical_flip:
        flip_mode = "vertical"
    if flip_mode is not None:
        layers.append(tf.keras.layers.RandomFlip(flip_mode))

    rotation_range = _coerce_float(config.get("rotation_range"), 0.0)
    if rotation_range > 0.0:
        layers.append(
            tf.keras.layers.RandomRotation(rotation_range, fill_mode="reflect")
        )

    zoom_range = config.get("zoom_range")
    if zoom_range is not None:
        zoom_factor = _coerce_range(zoom_range, (0.0, 0.0))
        if zoom_factor != (0.0, 0.0):
            layers.append(
                tf.keras.layers.RandomZoom(
                    height_factor=zoom_factor,
                    width_factor=zoom_factor,
                    fill_mode="reflect",
                )
            )

    translation_height_range = _coerce_float(
        config.get("translation_height_range"), 0.0
    )
    translation_width_range = _coerce_float(config.get("translation_width_range"), 0.0)
    if translation_height_range > 0.0 or translation_width_range > 0.0:
        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=translation_height_range,
                width_factor=translation_width_range,
                fill_mode="reflect",
            )
        )

    layers.append(
        ArtColorJitter(
            brightness_range=config.get("brightness_range", 0.0),
            contrast_range=config.get("contrast_range", 0.0),
            saturation_range=config.get("saturation_range", 0.0),
            color_jitter_strength=config.get("color_jitter_strength", 0.0),
        )
    )
    layers.append(RandomSharpness(config.get("sharpness_range", 0.0)))
    layers.append(
        RandomErasing(
            probability=config.get("random_erasing_prob", 0.0),
            area_range=config.get("random_erasing_area", (0.02, 0.10)),
        )
    )
    layers.append(ClippedGaussianNoise(config.get("gaussian_noise_std", 0.0)))
    layers.append(tf.keras.layers.Rescaling(255.0))

    return tf.keras.Sequential(layers)


def _resolve_split_dir(split_dir):
    """Return the absolute split directory path."""
    return Path(split_dir).resolve()


def _split_file_signature(split_dir):
    """Return a deterministic fingerprint for the current split contents."""
    file_records = []
    for file_path in sorted(path for path in split_dir.rglob("*") if path.is_file()):
        stat = file_path.stat()
        file_records.append(
            {
                "path": file_path.relative_to(split_dir).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

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
    return (
        CACHE_ROOT
        / f"{label}_{img_size[0]}x{img_size[1]}_b{batch_size}_shuffle{int(bool(shuffle))}_{cache_digest}"
    )


def _ensure_categorical_targets(labels, num_classes):
    """Return labels as one-hot or soft float32 class distributions."""
    labels = tf.convert_to_tensor(labels)
    if (
        labels.shape.rank is not None
        and labels.shape.rank > 1
        and labels.shape[-1] != 1
    ):
        return tf.cast(labels, tf.float32)

    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    return tf.one_hot(labels, depth=num_classes, dtype=tf.float32)


def _encode_categorical_targets(
    labels, num_classes, label_smoothing=0.0, class_prior=None
):
    """Convert sparse integer labels to one-hot or prior-aware soft targets."""
    one_hot_labels = _ensure_categorical_targets(labels, num_classes)

    epsilon = _coerce_float(label_smoothing, 0.0)
    if epsilon == 0.0:
        return one_hot_labels
    if class_prior is None:
        raise ValueError("class_prior is required when label_smoothing is enabled.")

    # Prior-aware smoothing is applied explicitly here because Keras'
    # built-in label_smoothing only supports uniform smoothing.
    prior_tensor = tf.reshape(
        tf.convert_to_tensor(class_prior, dtype=tf.float32), [1, num_classes]
    )
    return ((1.0 - epsilon) * one_hot_labels) + (epsilon * prior_tensor)


def _apply_target_mode(
    ds, num_classes, target_mode, label_smoothing=0.0, class_prior=None
):
    """Return a dataset with labels converted to the requested target mode."""
    if target_mode == "sparse":
        return ds
    if target_mode != "categorical":
        raise ValueError(f"Unsupported target_mode: {target_mode!r}")

    return ds.map(
        lambda images, labels: (
            images,
            _encode_categorical_targets(
                labels,
                num_classes,
                label_smoothing=label_smoothing,
                class_prior=class_prior,
            ),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def _sample_beta_distribution(batch_size, alpha):
    """Sample per-example mixing coefficients from Beta(alpha, alpha)."""
    if alpha <= 0.0:
        raise ValueError("alpha must be greater than 0.0 when batch mixing is enabled.")

    gamma_1 = tf.random.gamma(shape=[batch_size], alpha=alpha, dtype=tf.float32)
    gamma_2 = tf.random.gamma(shape=[batch_size], alpha=alpha, dtype=tf.float32)
    denominator = tf.maximum(gamma_1 + gamma_2, tf.keras.backend.epsilon())
    return gamma_1 / denominator


def mixup_batch(images, labels, alpha, num_classes):
    """Mix a batch with a shuffled view of itself using soft labels."""
    images = tf.cast(images, tf.float32)
    labels = _ensure_categorical_targets(labels, num_classes)

    batch_size = tf.shape(images)[0]

    def _mix():
        partner_indices = tf.random.shuffle(tf.range(batch_size))
        partner_images = tf.gather(images, partner_indices)
        partner_labels = tf.gather(labels, partner_indices)

        mixup_lambda = _sample_beta_distribution(batch_size, alpha)
        image_lambda = tf.reshape(mixup_lambda, [-1, 1, 1, 1])
        label_lambda = tf.reshape(mixup_lambda, [-1, 1])

        mixed_images = (image_lambda * images) + ((1.0 - image_lambda) * partner_images)
        mixed_labels = (label_lambda * labels) + ((1.0 - label_lambda) * partner_labels)
        return mixed_images, mixed_labels

    return tf.cond(
        tf.less_equal(batch_size, 1),
        lambda: (images, labels),
        _mix,
    )


def cutmix_batch(images, labels, alpha, num_classes):
    """Replace a random patch per image and mix labels by the kept-area ratio."""
    images = tf.cast(images, tf.float32)
    labels = _ensure_categorical_targets(labels, num_classes)

    batch_size = tf.shape(images)[0]

    def _cutmix():
        partner_indices = tf.random.shuffle(tf.range(batch_size))
        partner_images = tf.gather(images, partner_indices)
        partner_labels = tf.gather(labels, partner_indices)

        image_height = tf.shape(images)[1]
        image_width = tf.shape(images)[2]
        cutmix_lambda = _sample_beta_distribution(batch_size, alpha)
        cut_ratio = tf.sqrt(1.0 - cutmix_lambda)

        cut_heights = tf.cast(
            tf.round(cut_ratio * tf.cast(image_height, tf.float32)),
            tf.int32,
        )
        cut_widths = tf.cast(
            tf.round(cut_ratio * tf.cast(image_width, tf.float32)),
            tf.int32,
        )

        center_y = tf.random.uniform([batch_size], 0, image_height, dtype=tf.int32)
        center_x = tf.random.uniform([batch_size], 0, image_width, dtype=tf.int32)
        half_cut_heights = cut_heights // 2
        half_cut_widths = cut_widths // 2

        y1 = tf.clip_by_value(center_y - half_cut_heights, 0, image_height)
        y2 = tf.clip_by_value(
            center_y + (cut_heights - half_cut_heights),
            0,
            image_height,
        )
        x1 = tf.clip_by_value(center_x - half_cut_widths, 0, image_width)
        x2 = tf.clip_by_value(
            center_x + (cut_widths - half_cut_widths),
            0,
            image_width,
        )

        y_coords = tf.reshape(tf.range(image_height, dtype=tf.int32), [1, -1, 1])
        x_coords = tf.reshape(tf.range(image_width, dtype=tf.int32), [1, 1, -1])
        within_y = tf.logical_and(
            y_coords >= y1[:, tf.newaxis, tf.newaxis],
            y_coords < y2[:, tf.newaxis, tf.newaxis],
        )
        within_x = tf.logical_and(
            x_coords >= x1[:, tf.newaxis, tf.newaxis],
            x_coords < x2[:, tf.newaxis, tf.newaxis],
        )
        patch_mask = tf.cast(tf.logical_and(within_y, within_x), tf.float32)[
            ..., tf.newaxis
        ]

        mixed_images = (1.0 - patch_mask) * images + patch_mask * partner_images
        patch_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
        image_area = tf.cast(image_height * image_width, tf.float32)
        lambda_adjusted = 1.0 - (
            patch_area / tf.maximum(image_area, tf.keras.backend.epsilon())
        )
        label_lambda = tf.reshape(lambda_adjusted, [-1, 1])
        mixed_labels = (label_lambda * labels) + ((1.0 - label_lambda) * partner_labels)
        return mixed_images, mixed_labels

    return tf.cond(
        tf.less_equal(batch_size, 1),
        lambda: (images, labels),
        _cutmix,
    )


def apply_batch_mixing(
    images, labels, mix_strategy, mixup_alpha, cutmix_alpha, num_classes
):
    """Apply exactly one batch-mixing policy to a training batch."""
    if mix_strategy == "none":
        return tf.cast(images, tf.float32), _ensure_categorical_targets(
            labels, num_classes
        )
    if mix_strategy == "mixup":
        return mixup_batch(images, labels, alpha=mixup_alpha, num_classes=num_classes)
    if mix_strategy == "cutmix":
        return cutmix_batch(images, labels, alpha=cutmix_alpha, num_classes=num_classes)
    if mix_strategy == "mixup_or_cutmix":
        return tf.cond(
            tf.random.uniform([], 0.0, 1.0, dtype=tf.float32) < 0.5,
            lambda: mixup_batch(
                images, labels, alpha=mixup_alpha, num_classes=num_classes
            ),
            lambda: cutmix_batch(
                images, labels, alpha=cutmix_alpha, num_classes=num_classes
            ),
        )
    raise ValueError(f"Unsupported mix_strategy: {mix_strategy!r}")


def _apply_batch_mixing(
    ds,
    num_classes,
    mix_strategy=None,
    mixup_alpha=0.0,
    cutmix_alpha=0.0,
):
    """Return a dataset with the configured batch-mixing policy applied."""
    mix_config = model.resolve_mix_configuration(
        {
            "mix_strategy": mix_strategy,
            "mixup_alpha": mixup_alpha,
            "cutmix_alpha": cutmix_alpha,
        }
    )
    if not mix_config["mixing_enabled"]:
        return ds

    return ds.map(
        lambda images, labels: apply_batch_mixing(
            images,
            labels,
            mix_strategy=mix_config["effective_mix_strategy"],
            mixup_alpha=mix_config["mixup_alpha"],
            cutmix_alpha=mix_config["cutmix_alpha"],
            num_classes=num_classes,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def load_split(
    split_dir,
    img_size,
    batch_size,
    augment=False,
    config=None,
    target_mode="sparse",
    label_smoothing=0.0,
    class_prior=None,
    mix_strategy=None,
    mixup_alpha=0.0,
    cutmix_alpha=0.0,
):
    """
    Build a tf.data.Dataset from a pre-split folder.

    Args:
        split_dir:   Path to the split folder (e.g. "data/train").
                     Must contain one sub-folder per class.
        img_size:    Tuple (height, width).
        batch_size:  Batch size.
        augment:     Whether to apply augmentation (training only).
        config:      Optional dict with augmentation parameters.
        target_mode: Either "sparse" for integer labels or "categorical" for
                     one-hot / soft targets.
        label_smoothing: Prior-aware smoothing epsilon applied in
                         categorical mode.
        class_prior: Optional training-set prior vector used for
                     prior-aware smoothing.
        mix_strategy: Mixing policy for training batches. When omitted,
                      legacy alpha-only configs are inferred automatically.
        mixup_alpha: Positive alpha enables MixUp on the returned dataset.
        cutmix_alpha: Positive alpha enables CutMix on the returned dataset.

    Returns:
        tf.data.Dataset yielding (image_tensor, label) batches,
        and the list of class names inferred from the folder structure.
    """
    mix_config = model.resolve_mix_configuration(
        {
            "mix_strategy": mix_strategy,
            "mixup_alpha": mixup_alpha,
            "cutmix_alpha": cutmix_alpha,
        }
    )
    shuffle = augment or mix_config["mixing_enabled"]

    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=CACHE_SEED,
        label_mode="int",
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
        cache_path = _build_cache_path(split_dir, img_size, batch_size, shuffle=shuffle)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(cache_path))

    # Apply augmentation on training set only
    if augment:
        aug = build_augmentation_pipeline(config or {})
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = _apply_target_mode(
        ds,
        num_classes=len(class_names),
        target_mode=target_mode,
        label_smoothing=label_smoothing,
        class_prior=class_prior,
    )
    ds = _apply_batch_mixing(
        ds,
        num_classes=len(class_names),
        mix_strategy=mix_config["effective_mix_strategy"],
        mixup_alpha=mix_config["mixup_alpha"],
        cutmix_alpha=mix_config["cutmix_alpha"],
    )

    # Prefetch for performance(load data while the model is training on the current batch)
    return ds.prefetch(tf.data.AUTOTUNE), class_names
