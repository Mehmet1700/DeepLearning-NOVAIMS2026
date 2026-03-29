"""
Model definitions for artist classification.

Baseline: a simple 3-block CNN trained from scratch.
"""

import tensorflow as tf

OPTIMIZERS = {
    "adam":     tf.keras.optimizers.Adam,
    "sgd":      tf.keras.optimizers.SGD,
    "rmsprop":  tf.keras.optimizers.RMSprop,
}


def build_model(backbone: str, num_classes: int, img_size=(300, 300), freeze_base=True, config=None):
    """
    Build a baseline CNN model for classification.

    Args:
        backbone:     Kept for API compatibility — currently unused (baseline only).
        num_classes:  Number of output classes.
        img_size:     Spatial dimensions (height, width).
        freeze_base:  Kept for API compatibility — currently unused.
        config:       Dict with compile settings (optimizer, learning_rate, loss, metrics).

    Returns:
        Compiled tf.keras.Model.
    """
    cfg = config or {}

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*img_size, 3)),

        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        # Classifier head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    optimizer_name = cfg.get("optimizer", "adam").lower()
    learning_rate  = cfg.get("learning_rate", 1e-3)
    loss           = cfg.get("loss", "sparse_categorical_crossentropy")
    metrics        = cfg.get("metrics", ["accuracy"])

    optimizer_cls = OPTIMIZERS.get(optimizer_name, tf.keras.optimizers.Adam)

    model.compile(
        optimizer=optimizer_cls(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model
