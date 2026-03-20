# CNN Generalization Strategy Guide

This note captures practical changes to try when the current CNN starts memorizing the training set instead of generalizing to new images.

## Observed Training Behavior

The current training curves show severe overfitting.

- Training accuracy rises rapidly toward 100%.
- Training loss falls close to zero.
- Validation accuracy stalls around 30-33%.
- Validation loss starts increasing after the first epochs.

This pattern means the model is fitting the training data too closely and is not learning robust visual features that transfer to unseen images.

## Current Architecture Risk

The current CNN is described as:

- Two convolutional layers with 24 and 48 filters
- Two max-pooling layers
- One `Flatten()` layer
- One final dense classification layer with 23 outputs

Even though this looks like a small model, the `Flatten() -> Dense(...)` transition can create a very large number of trainable parameters. That makes it easy for the classifier head to memorize the training set.

## Recommended Strategies

### 1. Add L2 Weight Regularization

L2 regularization, also called weight decay, penalizes large weights and encourages smoother solutions.

Conceptually:

```text
L_total = L_data + lambda * sum(w^2)
```

Where:

- `L_data` is the normal training loss
- `lambda` controls regularization strength
- `w` represents model weights

Keras example:

```python
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D

Conv2D(
    24,
    (3, 3),
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
)
```

Recommended starting range:

- `1e-4`
- `5e-4`

### 2. Add Dropout

Dropout randomly disables part of the network during training, forcing the model to rely less on specific activations.

Keras example:

```python
from tensorflow.keras.layers import Dropout

Dropout(0.5)
```

Good places to try it:

- after the convolutional blocks
- before the final classification layer

Recommended starting values:

- `0.3` for earlier layers
- `0.5` before the classifier

### 3. Replace `Flatten()` With `GlobalAveragePooling2D()`

This is likely the highest-impact architectural change for the current model.

Instead of flattening the full feature map:

```python
Flatten()
Dense(23, activation="softmax")
```

Use:

```python
from tensorflow.keras.layers import GlobalAveragePooling2D

GlobalAveragePooling2D()
Dense(23, activation="softmax")
```

Why this helps:

- drastically reduces parameter count
- weakens the tendency of the dense layer to memorize
- usually improves generalization on image classification tasks

### 4. Use Data Augmentation

Data augmentation makes the training data more diverse and helps the network learn features that are stable under small visual changes.

Keras example:

```python
import tensorflow as tf

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ]
)
```

Good augmentation choices for this project:

- horizontal flips
- small rotations
- small zoom changes
- small translations

Use moderate transformations. If the augmentation is too strong, the model may start training on unrealistic images.

### 5. Add Early Stopping

If validation performance stops improving early, training longer usually just increases memorization.

Keras example:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)
```

This keeps the best weights instead of the last weights and avoids training far past the point where validation performance peaks.

## Suggested Order To Try

Use a controlled sequence instead of changing everything at once.

1. Replace `Flatten()` with `GlobalAveragePooling2D()`.
2. Add data augmentation.
3. Add L2 regularization to the convolution layers.
4. Add dropout before the final classifier, then optionally after convolution blocks.
5. Use early stopping and compare the best validation epoch with previous runs.

This order makes it easier to see which change actually improves validation performance.

## What To Monitor

For each experiment, compare:

- training accuracy
- validation accuracy
- training loss
- validation loss
- the gap between training and validation accuracy
- the epoch where validation loss is lowest

The main goal is not to push training accuracy to 100%. The goal is to reduce the gap between training and validation performance and improve validation results.

## Practical Next Session Plan

For the next training session, a sensible sequence is:

1. Keep the same dataset split and baseline settings.
2. Replace `Flatten()` with `GlobalAveragePooling2D()`.
3. Add light augmentation.
4. Add `kernel_regularizer=l2(1e-4)`.
5. Add `Dropout(0.5)` before the last dense layer.
6. Train with early stopping and compare the new validation curve against the previous run.

## Summary

The current CNN has enough capacity to memorize the training images, especially because of the large classifier created by `Flatten()`.

The strongest interventions to try next are:

- replacing `Flatten()` with `GlobalAveragePooling2D()`
- adding data augmentation
- adding L2 regularization
- adding dropout
- using early stopping

These changes should improve generalization and reduce the overfitting visible in the current training curves.
