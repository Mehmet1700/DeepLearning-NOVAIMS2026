"""Model-building utilities for artist classification."""

import tensorflow as tf

SERIALIZATION_PACKAGE = "deeplearning_novaims2026"


@tf.keras.utils.register_keras_serializable(package=SERIALIZATION_PACKAGE)
def resnet50_preprocess_input(inputs):
    """Apply ResNet50 preprocessing inside a serializable Lambda wrapper."""
    return tf.keras.applications.resnet.preprocess_input(inputs)


@tf.keras.utils.register_keras_serializable(package=SERIALIZATION_PACKAGE)
def efficientnetb3_preprocess_input(inputs):
    """Apply EfficientNetB3 preprocessing inside a serializable Lambda wrapper."""
    return tf.keras.applications.efficientnet.preprocess_input(inputs)


@tf.keras.utils.register_keras_serializable(package=SERIALIZATION_PACKAGE)
def vgg16_preprocess_input(inputs):
    """Apply VGG16 preprocessing inside a serializable Lambda wrapper."""
    return tf.keras.applications.vgg16.preprocess_input(inputs)


@tf.keras.utils.register_keras_serializable(package=SERIALIZATION_PACKAGE)
def mobilenetv3_preprocess_input(inputs):
    """Apply MobileNetV3 preprocessing inside a serializable Lambda wrapper."""
    return tf.keras.applications.mobilenet_v3.preprocess_input(inputs)


@tf.keras.utils.register_keras_serializable(package=SERIALIZATION_PACKAGE)
def densenet121_preprocess_input(inputs):
    """Apply DenseNet121 preprocessing inside a serializable Lambda wrapper."""
    return tf.keras.applications.densenet.preprocess_input(inputs)


OPTIMIZERS = {
    "adam":     tf.keras.optimizers.Adam,
    "sgd":      tf.keras.optimizers.SGD,
    "rmsprop":  tf.keras.optimizers.RMSprop,
}

BACKBONE_PREPROCESSORS = {
    "resnet50": resnet50_preprocess_input,
    "efficientnetb3": efficientnetb3_preprocess_input,
    "vgg16": vgg16_preprocess_input,
    "mobilenetv3": mobilenetv3_preprocess_input,
    "densenet121": densenet121_preprocess_input,
}

BACKBONES = {
    "resnet50": (
        tf.keras.applications.ResNet50,
        BACKBONE_PREPROCESSORS["resnet50"],
    ),
    "efficientnetb3": (
        tf.keras.applications.EfficientNetB3,
        BACKBONE_PREPROCESSORS["efficientnetb3"],
    ),
    "vgg16": (
        tf.keras.applications.VGG16,
        BACKBONE_PREPROCESSORS["vgg16"],
    ),
    "mobilenetv3": (
        tf.keras.applications.MobileNetV3Large,
        BACKBONE_PREPROCESSORS["mobilenetv3"],
    ),
    "densenet121": (
        tf.keras.applications.DenseNet121,
        BACKBONE_PREPROCESSORS["densenet121"],
    ),
}

BACKBONE_LAYER_NAMES = {
    "resnet50": "resnet50",
    "efficientnetb3": "efficientnetb3",
    "vgg16": "vgg16",
    "mobilenetv3": "MobilenetV3large",
    "densenet121": "densenet121",
}


class SparseCategoricalF1Score(tf.keras.metrics.F1Score):
    """F1Score that accepts sparse integer labels instead of one-hot floats."""
    def __init__(self, num_classes, **kwargs):
        # Remove 'average' from kwargs if present to avoid duplicate argument
        kwargs.pop('average', None)
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


def get_model_custom_objects(backbone_name=None):
    """Return the custom_objects mapping needed to deserialize saved models."""
    custom_objects = {"SparseCategoricalF1Score": SparseCategoricalF1Score}
    if backbone_name is None:
        return custom_objects

    backbone_key = backbone_name.lower()
    if backbone_key in {"baseline", "perceptron"}:
        return custom_objects
    if backbone_key not in BACKBONE_PREPROCESSORS:
        raise ValueError(
            f"Unsupported backbone for model loading: '{backbone_name}'. "
            f"Choose from: baseline, {', '.join(BACKBONE_PREPROCESSORS)}."
        )

    preprocess_fn = BACKBONE_PREPROCESSORS[backbone_key]
    custom_objects[preprocess_fn.__name__] = preprocess_fn
    custom_objects["preprocess_input"] = preprocess_fn
    return custom_objects


def _resolve_metrics(metric_names, num_classes):
    """Convert metric name strings to Keras metric objects where needed."""
    resolved = []
    for m in metric_names:
        if m == "f1_score":
            # Custom metric for sparse categorical labels
            resolved.append(SparseCategoricalF1Score(num_classes, name="f1_score"))
        
        elif m == "auc":
            # AUC is complex with sparse labels; compute it in evaluation instead
            # Skip adding it here to avoid training errors
            pass
        else:
            resolved.append(m)
    return resolved


def _compile(model, config, num_classes=None, learning_rate=None):
    """Compile model using settings from config dict."""
    cfg           = config or {}
    optimizer_cls = OPTIMIZERS.get(cfg.get("optimizer", "adam").lower(), tf.keras.optimizers.Adam)
    learning_rate = cfg.get("learning_rate", 1e-3) if learning_rate is None else learning_rate
    loss          = cfg.get("loss", "sparse_categorical_crossentropy")
    metrics       = _resolve_metrics(cfg.get("metrics", ["f1_score"]), num_classes)
    model.compile(
        optimizer=optimizer_cls(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )


def _get_backbone_layer_name(backbone_name):
    """Return the nested Keras layer name for a configured backbone key."""
    backbone_key = backbone_name.lower()
    if backbone_key not in BACKBONE_LAYER_NAMES:
        raise ValueError(
            f"Unsupported backbone: '{backbone_name}'. Choose from: {', '.join(BACKBONE_LAYER_NAMES)}"
        )
    return BACKBONE_LAYER_NAMES[backbone_key]


def configure_fine_tuning(model, backbone_name, unfrozen_layers):
    """Unfreeze the last N backbone layers and keep the classification head trainable."""
    if isinstance(unfrozen_layers, str):
        if unfrozen_layers != "all":
            raise ValueError(
                "fine_tune_unfrozen_layers must be a positive integer or 'all'."
            )
    elif isinstance(unfrozen_layers, bool) or not isinstance(unfrozen_layers, int):
        raise ValueError(
            "fine_tune_unfrozen_layers must be a positive integer or 'all'."
        )

    backbone_layer_name = _get_backbone_layer_name(backbone_name)
    try:
        backbone = model.get_layer(backbone_layer_name)
    except ValueError as exc:
        raise ValueError(
            f"Could not locate the '{backbone_layer_name}' backbone in the model."
        ) from exc

    total_layers = len(backbone.layers)
    backbone.trainable = True
    if unfrozen_layers == "all":
        for layer in backbone.layers:
            layer.trainable = True
    else:
        if unfrozen_layers <= 0:
            raise ValueError(
                "fine_tune_unfrozen_layers must be a positive integer or 'all'."
            )
        if unfrozen_layers > total_layers:
            raise ValueError(
                f"fine_tune_unfrozen_layers={unfrozen_layers} exceeds the backbone "
                f"layer count ({total_layers})."
            )
        frozen_until = total_layers - unfrozen_layers
        for index, layer in enumerate(backbone.layers):
            layer.trainable = index >= frozen_until

    for layer in model.layers:
        if layer is not backbone:
            layer.trainable = True

    return total_layers


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

    # Apply backbone-specific preprocessing.
    # MobileNetV3 has include_preprocessing=True by default (applies preprocess_input
    # internally), so we disable it and apply our own Lambda to stay consistent.
    x = tf.keras.layers.Lambda(preprocess_fn)(inputs)

    extra_kwargs = {}
    if backbone_name == "mobilenetv3":
        extra_kwargs["include_preprocessing"] = False

    base = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3),
        **extra_kwargs,
    )
    base.trainable = not freeze_base

    # Keep the application model nested so fine-tuning can target it explicitly.
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
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
