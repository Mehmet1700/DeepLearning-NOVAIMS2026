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
def mobilenetv2_preprocess_input(inputs):
    """Apply MobileNetV2 preprocessing inside a serializable Lambda wrapper."""
    return tf.keras.applications.mobilenet_v2.preprocess_input(inputs)


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
    "mobilenetv2": mobilenetv2_preprocess_input,
    "mobilenetv3": mobilenetv3_preprocess_input,
    "densenet121": densenet121_preprocess_input,
    "vit": lambda x: x,
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
    "mobilenetv2": (
        tf.keras.applications.MobileNetV2,
        BACKBONE_PREPROCESSORS["mobilenetv2"],
    ),
    "mobilenetv3": (
        tf.keras.applications.MobileNetV3Large,
        BACKBONE_PREPROCESSORS["mobilenetv3"],
    ),
    "densenet121": (
        tf.keras.applications.DenseNet121,
        BACKBONE_PREPROCESSORS["densenet121"],
    ),
    "vit": (None, None),
}

BACKBONE_LAYER_NAMES = {
    "resnet50": "resnet50",
    "efficientnetb3": "efficientnetb3",
    "vgg16": "vgg16",
    "mobilenetv2": "mobilenetv2_1.00_224",
    "mobilenetv3": "MobilenetV3large",
    "densenet121": "densenet121",
    "vit": "vit"
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
    if backbone_key in {"baseline", "perceptron", "vit", "vit_tfhub", "vit_pretrained"}:
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
    if backbone_name in {"vit_pretrained", "vit_tfhub"}:
        print(f"Unfreezing full {backbone_name} model for fine-tuning")
        model.trainable = True
        return len(model.layers)

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
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((3, 3)),

        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((3, 3)),

        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((3, 3)),

        # Classifier head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32"),
    ])
    _compile(model, config, num_classes)
    return model


def build_perceptron(num_classes, img_size, config):
    """Single-layer perceptron (fully-connected network) trained from scratch."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*img_size, 3)),

        # Normalize [0, 255] → [0, 1]
        tf.keras.layers.Rescaling(1.0 / 255.0),

        # Flatten to 1D vector
        tf.keras.layers.Flatten(),

        # Single output layer
        tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32"),
    ])
    
    _compile(model, config, num_classes)
    return model


def build_transfer_model(backbone_name, num_classes, img_size, freeze_base, config):
    """Transfer learning model with pretrained backbone + custom head."""
    backbone_cls, preprocess_fn = BACKBONES[backbone_name]
    cfg = config or {}
    hidden_layer_sizes = cfg.get("hidden_layer_sizes", [])
    dropout_rate = cfg.get("dropout_rate", 0.4)
    use_batch_norm = cfg.get("batch_norm", True)
    pooling_type = cfg.get("pooling_type", "global_average_pooling2d").lower()
    l2_lambda = cfg.get("l2_regularization", 0.0)
    regularizer = tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None

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
    # training=False forces BatchNorm to use ImageNet running statistics instead
    # of computing noisy batch statistics from small WikiArt batches (~3 images
    # per class). Without this, the backbone features are meaningless even with
    # trainable=False. Required for both Phase 1 (frozen) and Phase 2 (fine-tuning).
    # See: https://www.tensorflow.org/guide/keras/transfer_learning
    x = base(x, training=False)
    
    # Apply pooling layer based on configuration
    if pooling_type == "flatten":
        x = tf.keras.layers.Flatten()(x)
    elif pooling_type == "global_max_pooling2d":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    elif pooling_type == "global_average_pooling2d":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        raise ValueError(
            f"Unsupported pooling_type: '{pooling_type}'. "
            f"Choose from: flatten, global_max_pooling2d, global_average_pooling2d"
        )
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    for units in hidden_layer_sizes:
        x = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=regularizer)(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)
    _compile(model, config, num_classes)
    return model

def build_vit_model(num_classes, img_size, config):
    """ViT-Base architecture trained from scratch using pure TF/Keras.

    Defaults match the ViT-Base/16 paper configuration:
      projection_dim=768, num_heads=12, transformer_layers=12, mlp_dim=3072.
    Override via config keys: patch_size, projection_dim, num_heads,
    transformer_layers, mlp_dim, dropout_rate.
    """
    cfg = config or {}
    patch_size         = cfg.get("patch_size", 16)
    projection_dim     = cfg.get("projection_dim", 768)
    num_heads          = cfg.get("num_heads", 12)
    transformer_layers = cfg.get("transformer_layers", 12)
    mlp_dim            = cfg.get("mlp_dim", 3072)
    dropout_rate       = cfg.get("dropout_rate", 0.1)

    h, w = img_size
    num_patches = (h // patch_size) * (w // patch_size)   # 196 for 224×224, patch=16

    inputs = tf.keras.Input(shape=(*img_size, 3))

    # 1. Normalize [0, 255] → [0, 1]
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    # 2. Patch embedding — Conv2D extracts & projects patches in one step
    x = tf.keras.layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
    )(x)
    x = tf.keras.layers.Reshape((num_patches, projection_dim))(x)

    # 3. Learned positional embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = tf.keras.layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    x = x + pos_embedding
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # 4. Transformer encoder blocks
    for _ in range(transformer_layers):
        # Self-attention branch
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=dropout_rate,
        )(x1, x1)
        x1 = tf.keras.layers.Dropout(dropout_rate)(x1)
        x = tf.keras.layers.Add()([x, x1])

        # MLP branch
        x2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x2 = tf.keras.layers.Dense(mlp_dim, activation="gelu")(x2)
        x2 = tf.keras.layers.Dropout(dropout_rate)(x2)
        x2 = tf.keras.layers.Dense(projection_dim)(x2)
        x2 = tf.keras.layers.Dropout(dropout_rate)(x2)
        x = tf.keras.layers.Add()([x, x2])

    # 5. Classification head
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)
    _compile(model, cfg, num_classes)
    return model

@tf.keras.utils.register_keras_serializable(package=SERIALIZATION_PACKAGE)
class ViTHubLayer(tf.keras.layers.Layer):
    """TF Hub ViT-Base/16 feature extractor compatible with Keras 3 Functional API.

    hub.KerasLayer is not compatible with Keras 3 because it tries to call the
    underlying tf.function during symbolic graph construction, which fails when
    the input is a KerasTensor (no numeric value). This class fixes it by
    overriding compute_output_shape so Keras 3 can determine the output shape
    symbolically without ever calling the hub model. The hub model is only
    invoked during actual forward passes with real tensors.
    """

    def __init__(self, hub_url, **kwargs):
        super().__init__(**kwargs)
        self.hub_url = hub_url
        self._hub_module = None

    def build(self, input_shape):
        try:
            import tensorflow_hub as hub
        except ImportError as exc:
            raise ImportError(
                "ViTHubLayer requires tensorflow-hub. "
                "Install with: pip install tensorflow-hub"
            ) from exc
        # Explicitly load on GPU:0 if available, otherwise falls back to CPU
        device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        with tf.device(device):
            self._hub_module = hub.load(self.hub_url)
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Called only with real tensors during model.fit() / model.predict().
        # The hub model signature only accepts inputs (no training argument).
        return self._hub_module(inputs)

    def compute_output_shape(self, input_shape):
        # ViT-Base/16 outputs a 768-dimensional feature vector per image.
        # Keras 3 calls this during Functional API graph building instead of
        # calling call(), which avoids the KerasTensor incompatibility.
        return (input_shape[0], 768)

    def get_config(self):
        return {**super().get_config(), "hub_url": self.hub_url}


def build_vit_tfhub(num_classes, img_size, freeze_base=True, config=None):
    """Pretrained ViT-Base/16 (ImageNet21k) via TensorFlow Hub.

    Requires: pip install tensorflow-hub

    Before the first HPC job, download weights on the login node:
        export TFHUB_CACHE_DIR=$(pwd)/.tfhub_cache
        python -c 'import tensorflow_hub as hub; hub.load("https://tfhub.dev/sayakpaul/vit_b16_fe/1")'
    The SLURM job script already exports TFHUB_CACHE_DIR automatically.
    """
    cfg = config or {}
    hub_url = cfg.get("vit_hub_url", "https://tfhub.dev/sayakpaul/vit_b16_fe/1")
    dropout_rate = cfg.get("dropout_rate", 0.1)

    inputs = tf.keras.Input(shape=(*img_size, 3))
    # Scale [0, 255] → [0, 1]; the hub model applies ImageNet normalisation internally
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = ViTHubLayer(hub_url, trainable=not freeze_base, name="vit_b16_fe")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)
    _compile(model, cfg, num_classes)
    return model


def build_vit_pretrained(num_classes, img_size, config):
    """Pretrained ViT from KerasHub. Requires keras_hub + Keras 3 (TF 2.16+)."""
    try:
        import keras_hub
    except ImportError as exc:
        raise ImportError(
            "build_vit_pretrained requires keras_hub with Keras 3 (TF 2.16+). "
            "Install with: pip install keras-hub. "
            "With TF 2.15 use backbone='vit' (custom ViT) instead."
        ) from exc

    preset = "vit_large_patch16_224_imagenet21k"

    # Load pretrained backbone
    backbone = keras_hub.models.Backbone.from_preset(preset)

    # Preprocessing (VERY IMPORTANT)
    preprocessor = keras_hub.models.ViTImageClassifierPreprocessor.from_preset(preset)

    # Build classifier
    model = keras_hub.models.ViTImageClassifier(
        backbone=backbone,
        num_classes=num_classes,
        preprocessor=preprocessor,
    )

    # Freeze backbone (Phase 1)
    if config.get("freeze_base", True):
        backbone.trainable = False

    _compile(model, config, num_classes)

    return model

def build_model(backbone: str, num_classes: int, img_size=(224, 224), freeze_base=True, config=None):
    """
    Build and compile a model for artist classification.

    Args:
        backbone:     "baseline", "perceptron", "resnet50", "efficientnetb3", "vgg16",
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

    if backbone == "perceptron":
        return build_perceptron(num_classes, img_size, config)
    
    if backbone == "vit":
        return build_vit_model(num_classes, img_size, config)

    if backbone == "vit_tfhub":
        return build_vit_tfhub(num_classes, img_size, freeze_base, config)

    if backbone == "vit_pretrained":
        return build_vit_pretrained(num_classes, img_size, config)

    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone: '{backbone}'. Choose from: baseline, perceptron, {', '.join(BACKBONES)}")

    return build_transfer_model(backbone, num_classes, img_size, freeze_base, config)
