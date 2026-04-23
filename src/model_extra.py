"""Model-building utilities for artist classification."""

import tensorflow as tf

SERIALIZATION_PACKAGE = "deeplearning_novaims2026"
SPARSE_TARGET_MODE = "sparse"
CATEGORICAL_TARGET_MODE = "categorical"


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
    "adam": tf.keras.optimizers.Adam,
    "sgd": tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
}

MIX_STRATEGIES = {"none", "mixup", "cutmix", "mixup_or_cutmix"}

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
    """F1Score that accepts sparse integers and dense label targets."""

    def __init__(self, num_classes, **kwargs):
        # Remove 'average' from kwargs if present to avoid duplicate argument
        kwargs.pop("average", None)
        super().__init__(average="macro", **kwargs)
        self._num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Metrics still need hard labels even when the loss uses prior-aware
        # soft targets, so recover the original class ids via argmax first.
        y_true = tf.convert_to_tensor(y_true)
        if (
            y_true.shape.rank is not None
            and y_true.shape.rank > 1
            and y_true.shape[-1] != 1
        ):
            y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        else:
            y_true = tf.cast(tf.round(tf.reshape(y_true, [-1])), tf.int32)
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
    for metric_name in metric_names:
        if metric_name == "f1_score":
            resolved.append(SparseCategoricalF1Score(num_classes, name="f1_score"))
        elif metric_name == "auc":
            # AUC is complex with sparse labels; compute it in evaluation instead.
            pass
        else:
            resolved.append(metric_name)
    return resolved


def _coerce_non_negative_float(value, config_key, default=0.0):
    """Return a non-negative float config value or the provided default."""
    if value is None:
        return default

    numeric_value = float(value)
    if numeric_value < 0:
        raise ValueError(f"{config_key} must be a non-negative float.")
    return numeric_value


def _coerce_mix_strategy(value):
    """Return a normalized mix strategy or None when it is unset."""
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized not in MIX_STRATEGIES:
        raise ValueError(
            f"mix_strategy must be one of: {', '.join(sorted(MIX_STRATEGIES))}."
        )
    return normalized


def _infer_mix_strategy(mixup_alpha, cutmix_alpha):
    """Infer the legacy mixing strategy from configured alpha values."""
    if mixup_alpha > 0.0 and cutmix_alpha > 0.0:
        return "mixup_or_cutmix"
    if mixup_alpha > 0.0:
        return "mixup"
    if cutmix_alpha > 0.0:
        return "cutmix"
    return "none"


def resolve_mix_configuration(config):
    """Resolve mixing strategy settings, preserving legacy alpha-only configs."""
    cfg = config or {}
    mixup_alpha = _coerce_non_negative_float(
        cfg.get("mixup_alpha", 0.0),
        "mixup_alpha",
    )
    cutmix_alpha = _coerce_non_negative_float(
        cfg.get("cutmix_alpha", 0.0),
        "cutmix_alpha",
    )
    configured_mix_strategy = _coerce_mix_strategy(cfg.get("mix_strategy"))
    effective_mix_strategy = (
        _infer_mix_strategy(mixup_alpha, cutmix_alpha)
        if configured_mix_strategy is None
        else configured_mix_strategy
    )

    if effective_mix_strategy == "mixup" and mixup_alpha <= 0.0:
        raise ValueError("mix_strategy='mixup' requires mixup_alpha > 0.0.")
    if effective_mix_strategy == "cutmix" and cutmix_alpha <= 0.0:
        raise ValueError("mix_strategy='cutmix' requires cutmix_alpha > 0.0.")
    if effective_mix_strategy == "mixup_or_cutmix" and (
        mixup_alpha <= 0.0 or cutmix_alpha <= 0.0
    ):
        raise ValueError(
            "mix_strategy='mixup_or_cutmix' requires both mixup_alpha > 0.0 "
            "and cutmix_alpha > 0.0."
        )

    mixing_enabled = effective_mix_strategy != "none"
    return {
        "configured_mix_strategy": configured_mix_strategy,
        "mix_strategy": effective_mix_strategy,
        "effective_mix_strategy": effective_mix_strategy,
        "mixup_alpha": mixup_alpha,
        "cutmix_alpha": cutmix_alpha,
        "mixing_enabled": mixing_enabled,
        "mixup_enabled": effective_mix_strategy in {"mixup", "mixup_or_cutmix"},
        "cutmix_enabled": effective_mix_strategy in {"cutmix", "mixup_or_cutmix"},
    }


def resolve_training_targets(config):
    """Return the configured label representation, smoothing, and mixing settings."""
    cfg = config or {}
    label_smoothing = _coerce_non_negative_float(
        cfg.get("label_smoothing", 0.0),
        "label_smoothing",
    )
    if label_smoothing >= 1.0:
        raise ValueError("label_smoothing must be less than 1.0.")

    prior_aware = bool(cfg.get("prior_aware_label_smoothing", False))
    mix_config = resolve_mix_configuration(cfg)
    return {
        "target_mode": (
            CATEGORICAL_TARGET_MODE
            if prior_aware or mix_config["mixing_enabled"]
            else SPARSE_TARGET_MODE
        ),
        "prior_aware_label_smoothing": prior_aware,
        "label_smoothing": label_smoothing,
        **mix_config,
    }


def resolve_loss_name(config):
    """Return the effective compile-time loss name for the target mode."""
    target_config = resolve_training_targets(config)
    if target_config["target_mode"] == CATEGORICAL_TARGET_MODE:
        return "categorical_crossentropy"
    return "sparse_categorical_crossentropy"


def resolve_loss(config):
    """Return the loss object implied by the configured target mode."""
    target_config = resolve_training_targets(config)
    if target_config["target_mode"] == CATEGORICAL_TARGET_MODE:
        # Sparse categorical crossentropy expects integer class ids. Once the
        # dataset emits one-hot / soft targets for prior-aware smoothing or
        # MixUp, categorical crossentropy must be used instead.
        #
        # Built-in Keras label_smoothing is uniform across non-target classes,
        # so it cannot express epsilon * class_prior and stays disabled here.
        return tf.keras.losses.CategoricalCrossentropy(
            name="categorical_crossentropy",
        )
    return tf.keras.losses.SparseCategoricalCrossentropy(
        name="sparse_categorical_crossentropy",
    )


def _get_l2_regularizer(config):
    """Build the configured kernel regularizer for project-defined layers."""
    cfg = config or {}
    l2_value = _coerce_non_negative_float(
        cfg.get("l2_regularization", 0.0),
        "l2_regularization",
    )
    if l2_value == 0.0:
        return None
    return tf.keras.regularizers.L2(l2_value)


def resolve_optimizer_settings(config, learning_rate=None, weight_decay=None):
    """Resolve the optimizer class and numeric settings from config values."""
    cfg = config or {}
    configured_optimizer = str(cfg.get("optimizer", "adam")).lower()
    optimizer_name = (
        configured_optimizer if configured_optimizer in OPTIMIZERS else "adam"
    )
    optimizer_cls = OPTIMIZERS.get(
        optimizer_name,
        tf.keras.optimizers.Adam,
    )
    resolved_learning_rate = _coerce_non_negative_float(
        cfg.get("learning_rate", 1e-3) if learning_rate is None else learning_rate,
        "learning_rate",
        default=1e-3,
    )
    resolved_weight_decay = _coerce_non_negative_float(
        cfg.get("weight_decay", 0.0) if weight_decay is None else weight_decay,
        "weight_decay",
    )
    if optimizer_name != "adam" and resolved_weight_decay > 0:
        raise ValueError("weight_decay is only supported when optimizer is 'adam'.")

    return {
        "optimizer_name": optimizer_name,
        "optimizer_class": optimizer_cls,
        "learning_rate": resolved_learning_rate,
        "weight_decay": resolved_weight_decay,
    }


def _compile(model, config, num_classes=None, learning_rate=None, weight_decay=None):
    """Compile model using settings from config dict."""
    cfg = config or {}
    optimizer_settings = resolve_optimizer_settings(
        cfg,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    loss = resolve_loss(cfg)
    metrics = _resolve_metrics(cfg.get("metrics", ["f1_score"]), num_classes)
    optimizer_kwargs = {"learning_rate": optimizer_settings["learning_rate"]}
    if (
        optimizer_settings["optimizer_name"] == "adam"
        and optimizer_settings["weight_decay"] > 0
    ):
        optimizer_kwargs["weight_decay"] = optimizer_settings["weight_decay"]

    model.compile(
        optimizer=optimizer_settings["optimizer_class"](**optimizer_kwargs),
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
    classifier_regularizer = _get_l2_regularizer(config)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(*img_size, 3)),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=classifier_regularizer,
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=classifier_regularizer,
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=classifier_regularizer,
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
                dtype="float32",
                kernel_regularizer=classifier_regularizer,
            ),
        ]
    )
    _compile(model, config, num_classes)
    return model


def build_transfer_model(backbone_name, num_classes, img_size, freeze_base, config):
    """Transfer learning model with pretrained backbone + custom head."""
    backbone_cls, preprocess_fn = BACKBONES[backbone_name]
    classifier_regularizer = _get_l2_regularizer(config)

    inputs = tf.keras.Input(shape=(*img_size, 3))
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

    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        kernel_regularizer=classifier_regularizer,
    )(x)

    model = tf.keras.Model(inputs, outputs)
    _compile(model, config, num_classes)
    return model


def build_model(
    backbone: str,
    num_classes: int,
    img_size=(224, 224),
    freeze_base=True,
    config=None,
):
    """Build and compile a model for artist classification."""
    backbone = backbone.lower()

    if backbone == "baseline":
        return build_baseline(num_classes, img_size, config)

    if backbone not in BACKBONES:
        raise ValueError(
            f"Unsupported backbone: '{backbone}'. Choose from: baseline, {', '.join(BACKBONES)}"
        )

    return build_transfer_model(backbone, num_classes, img_size, freeze_base, config)
