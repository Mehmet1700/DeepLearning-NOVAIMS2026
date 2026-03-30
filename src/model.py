"""
Model definitions for artist classification.

Supported backbones:
  - baseline      : simple 3-block CNN trained from scratch (fast, no pretrained weights)
  - resnet50      : ResNet50 pretrained on ImageNet
  - efficientnetb3: EfficientNetB3 pretrained on ImageNet
  - vgg16         : VGG16 pretrained on ImageNet
  - mobilenetv3   : MobileNetV3Large pretrained on ImageNet
  - densenet121   : DenseNet121 pretrained on ImageNet

Two-phase training (transfer learning only):
  Phase 1 - freeze_base=True : only the classification head is trained
  Phase 2 - freeze_base=False: the full network is fine-tuned at a lower learning rate
  Set fine_tune_epochs in config to trigger Phase 2 automatically from train.py.
"""

import tensorflow as tf

OPTIMIZERS = {
    "adam":     tf.keras.optimizers.Adam,
    "sgd":      tf.keras.optimizers.SGD,
    "rmsprop":  tf.keras.optimizers.RMSprop,
}

BACKBONES = {
    "resnet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet.preprocess_input,
    ),
    "efficientnetb3": (
        tf.keras.applications.EfficientNetB3,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
    "vgg16": (
        tf.keras.applications.VGG16,
        tf.keras.applications.vgg16.preprocess_input,
    ),
    "mobilenetv3": (
        tf.keras.applications.MobileNetV3Large,
        tf.keras.applications.mobilenet_v3.preprocess_input,
    ),
    "densenet121": (
        tf.keras.applications.DenseNet121,
        tf.keras.applications.densenet.preprocess_input,
    ),
}


class SparseCategoricalF1Score(tf.keras.metrics.F1Score):
    """F1Score that accepts sparse integer labels instead of one-hot floats."""
    def __init__(self, num_classes, **kwargs):
        super().__init__(average="macro", **kwargs)
        self._num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true = tf.cast(tf.one_hot(y_true, self._num_classes), self.dtype)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self._num_classes
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _resolve_metrics(metric_names, num_classes):
    """Convert metric name strings to Keras metric objects where needed."""
    resolved = []
    for m in metric_names:
        if m == "f1_score":
            resolved.append(SparseCategoricalF1Score(num_classes, name="f1_score"))
        else:
            resolved.append(m)
    return resolved


def _compile(model, config, num_classes=None):
    """Compile model using settings from config dict."""
    cfg           = config or {}
    optimizer_cls = OPTIMIZERS.get(cfg.get("optimizer", "adam").lower(), tf.keras.optimizers.Adam)
    learning_rate = cfg.get("learning_rate", 1e-3)
    loss          = cfg.get("loss", "sparse_categorical_crossentropy")
    metrics       = _resolve_metrics(cfg.get("metrics", ["f1_score"]), num_classes)
    model.compile(
        optimizer=optimizer_cls(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )


def build_baseline(num_classes, img_size, config):
    """Simple 3-block CNN trained from scratch."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*img_size, 3)),

        # Normalize [0, 255] → [0, 1]
        tf.keras.layers.Rescaling(1.0 / 255.0),

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
    _compile(model, config, num_classes)
    return model


def build_transfer_model(backbone_name, num_classes, img_size, freeze_base, config):
    """Transfer learning model with pretrained backbone + custom head."""
    backbone_cls, preprocess_fn = BACKBONES[backbone_name]

    inputs = tf.keras.Input(shape=(*img_size, 3))

    # Apply backbone-specific preprocessing
    x = tf.keras.layers.Lambda(preprocess_fn)(inputs)

    base = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base.trainable = not freeze_base

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    _compile(model, config, num_classes)
    return model


def build_model(backbone: str, num_classes: int, img_size=(224, 224), freeze_base=True, config=None):
    """
    Build and compile a model for artist classification.

    Args:
        backbone:     "baseline", "resnet50", "efficientnetb3", "vgg16",
                      "mobilenetv3", or "densenet121".
        num_classes:  Number of output classes.
        img_size:     Spatial dimensions (height, width).
        freeze_base:  For transfer learning — freeze backbone weights in Phase 1.
        config:       Dict with compile settings from YAML config.

    Returns:
        Compiled tf.keras.Model.
    """
    backbone = backbone.lower()

    if backbone == "baseline":
        return build_baseline(num_classes, img_size, config)

    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone: '{backbone}'. Choose from: baseline, {', '.join(BACKBONES)}")

    return build_transfer_model(backbone, num_classes, img_size, freeze_base, config)
