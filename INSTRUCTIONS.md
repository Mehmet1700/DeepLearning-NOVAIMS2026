# Project Instructions — src/ Guide

This document explains every file in the `src/` folder, how they connect to each other, and how to run the full pipeline from raw data to evaluation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Step-by-Step Pipeline](#3-step-by-step-pipeline)
4. [File Reference](#4-file-reference)
   - [split_dataset.py](#41-split_datasetpy)
   - [dataset.py](#42-datasetpy)
   - [model.py](#43-modelpy)
   - [train.py](#44-trainpy)
   - [evaluate.py](#45-evaluatepy)
   - [utils.py](#46-utilspy)
5. [Configuration File](#5-configuration-file)
6. [Outputs](#6-outputs)
7. [Common Errors](#7-common-errors)

---

## 1. Project Overview

This project trains a Convolutional Neural Network (CNN) to classify paintings by artist using the WikiArt dataset. The model looks at a painting and predicts which of the 23 artists painted it.

The pipeline has three main stages:

```
Raw images  →  Split into folders  →  Train model  →  Evaluate model
(data/wikiart)   (split_dataset.py)    (train.py)     (evaluate.py)
```

---

## 2. Folder Structure

```
DeepLearning-NOVAIMS2026/
├── data/
│   ├── wikiart/              ← original images, one folder per artist (DO NOT TOUCH)
│   │   ├── Claude_Monet/
│   │   ├── Pablo_Picasso/
│   │   └── ...
│   ├── train/                ← created by split_dataset.py (70% of images)
│   ├── validation/           ← created by split_dataset.py (15% of images)
│   └── test/                 ← created by split_dataset.py (15% of images)
├── src/
│   ├── split_dataset.py      ← Step 1: split raw images into train/val/test
│   ├── dataset.py            ← loads images from folders into TensorFlow datasets
│   ├── model.py              ← defines the CNN architecture
│   ├── train.py              ← Step 2: trains the model
│   ├── evaluate.py           ← Step 3: evaluates the trained model
│   └── utils.py              ← helper functions for EDA notebooks
├── configs/
│   └── config_local.yaml     ← all training settings live here
└── outputs/
    ├── checkpoints/          ← saved model weights after training
    └── logs/                 ← TensorBoard logs
```

---

## 3. Step-by-Step Pipeline

### Prerequisites

Make sure the virtual environment is activated:
```bash
source .venv/bin/activate
```

---

### Step 1 — Split the dataset

**Run once.** Copies images from `data/wikiart/` into `data/train/`, `data/validation/`, and `data/test/`.

```bash
python src/split_dataset.py
```

Expected output:
```
Source: /path/to/data/wikiart
Output: /path/to/data
Ratios: train=0.70, validation=0.15, test=0.15
Seed: 73

Per-class counts:
  Albrecht_Durer: train=..., validation=..., test=...
  ...

Totals: train=9326, validation=1992, test=1992
```

> **Important:** Only run this once. If `data/train/` already exists, the script will refuse to run to avoid overwriting your split. If you need to re-split, delete the `data/train/`, `data/validation/`, and `data/test/` folders first.

---

### Step 2 — Train the model

```bash
python src/train.py
```

By default this uses `configs/config_local.yaml`. To use a different config:
```bash
python src/train.py --config configs/config_hpc.yaml
```

You will see progress per batch:
```
Epoch 1/5
292/292 ━━━━━━━━━━━━━━━━━━━━ 120s - accuracy: 0.12 - val_accuracy: 0.15
```

- `292` = number of batches per epoch (total images ÷ batch size)
- Training saves the best model automatically to `outputs/checkpoints/best_model.keras`

---

### Step 3 — Evaluate the model

```bash
python src/evaluate.py \
  --config configs/config_local.yaml \
  --weights outputs/checkpoints/best_model.keras
```

Output includes:
- A **classification report** with precision, recall, and F1-score per artist
- A **confusion matrix**
- Overall **Top-1 Accuracy**

---

## 4. File Reference

### 4.1 `split_dataset.py`

**Purpose:** Copies images from `data/wikiart/` into physical train, validation, and test folders.

**Key settings** (defined at the top of the file):

| Constant | Value | Meaning |
|---|---|---|
| `TRAIN_RATIO` | `0.70` | 70% of images go to training |
| `VALIDATION_RATIO` | `0.15` | 15% go to validation |
| `TEST_RATIO` | `0.15` | 15% go to testing |
| `SEED` | `73` | Ensures the same split every time |
| `IMAGE_SUFFIXES` | `{".jpg"}` | Only `.jpg` files are included |

**What it does internally:**
1. Reads all artist subfolders from `data/wikiart/`
2. For each artist, shuffles their images deterministically using the seed
3. Splits them according to the ratios
4. Copies (not moves) the files into the output folders

**Why copy instead of move?** The original `data/wikiart/` stays intact. If you need to re-split with different ratios, you still have the originals.

---

### 4.2 `dataset.py`

**Purpose:** Loads images from the split folders into TensorFlow datasets that the model can train on. This file is not run directly — it is imported by `train.py` and `evaluate.py`.

**Two functions:**

#### `build_augmentation_pipeline(config)`
Creates a pipeline of random transformations applied to training images to make the model more robust:

| Transformation | What it does | Config key |
|---|---|---|
| `RandomFlip` | Randomly mirrors image left ↔ right | — |
| `RandomRotation` | Rotates by up to ±10% of 360° | `rotation_range` |
| `RandomZoom` | Zooms in/out by up to ±10% | `zoom_range` |
| `RandomContrast` | Adjusts contrast by up to ±10% | `contrast_range` |

Augmentation only runs on the training set. Validation and test images are always loaded clean.

#### `load_split(split_dir, img_size, batch_size, augment, config)`
Builds a `tf.data.Dataset` from a folder. The pipeline runs in this order:

```
Read images from disk
        ↓
Resize to img_size (e.g. 128×128)
        ↓
Normalize pixels from [0–255] to [0.0–1.0]
        ↓
Cache in memory  ← epoch 1 is slow, epoch 2+ are fast
        ↓
Apply augmentation (training only, random each epoch)
        ↓
Prefetch next batch while GPU trains on current batch
```

**Returns:** `(dataset, class_names)` — the dataset and a list of artist names in alphabetical order (e.g. `["Albrecht_Durer", "Boris_Kustodiev", ...]`).

---

### 4.3 `model.py`

**Purpose:** Defines and compiles the CNN architecture. Imported by `train.py`.

**Current architecture — Baseline CNN:**

```
Input (128×128×3)
        ↓
Conv2D(32 filters, 3×3) + ReLU
MaxPooling  →  64×64
        ↓
Conv2D(64 filters, 3×3) + ReLU
MaxPooling  →  32×32
        ↓
Conv2D(128 filters, 3×3) + ReLU
MaxPooling  →  16×16
        ↓
GlobalAveragePooling2D  →  flattens to 1D
Dropout(0.3)
Dense(23, softmax)  →  probability for each artist
```

**Why this architecture for a baseline?**
- Simple enough to train quickly on CPU/GPU
- Enough capacity to learn basic visual patterns (colours, shapes)
- Not expected to get high accuracy — it is a starting point to verify the pipeline works end-to-end

**Compile settings** are read from `config_local.yaml` (not hardcoded):
- `optimizer` — `adam`, `sgd`, or `rmsprop`
- `learning_rate` — e.g. `0.001`
- `loss` — `sparse_categorical_crossentropy`
- `metrics` — e.g. `[accuracy]`

---

### 4.4 `train.py`

**Purpose:** Runs the full training loop. This is the main script your team will run most often.

**What it does step by step:**
1. Reads all settings from the config YAML
2. Loads the training and validation datasets via `dataset.py`
3. Builds the model via `model.py`
4. Sets up 4 automatic callbacks:

| Callback | What it does |
|---|---|
| `ModelCheckpoint` | Saves `best_model.keras` whenever validation accuracy improves |
| `TensorBoard` | Writes loss/accuracy curves to `outputs/logs/` for visualisation |
| `EarlyStopping` | Stops training if validation loss does not improve for `patience` epochs |
| `ReduceLROnPlateau` | Halves the learning rate if validation loss stalls for 3 epochs |

5. Calls `model.fit()` to run the training loop
6. Saves `final_model.keras` (the last epoch) alongside `best_model.keras` (the best epoch)

**Run:**
```bash
python src/train.py                              # uses config_local.yaml by default
python src/train.py --config configs/other.yaml  # override config
```

---

### 4.5 `evaluate.py`

**Purpose:** Loads a saved model and runs it against the test set to measure performance.

**What it outputs:**
- **Classification report** — precision, recall, F1-score per artist
- **Confusion matrix** — which artists get confused with each other
- **Top-1 Accuracy** — overall percentage of correct predictions

**Run:**
```bash
python src/evaluate.py \
  --config configs/config_local.yaml \
  --weights outputs/checkpoints/best_model.keras
```

> Always use `best_model.keras` (not `final_model.keras`) for evaluation. The best model is saved at the epoch with the highest validation accuracy, which is not necessarily the last epoch.

---

### 4.6 `utils.py`

**Purpose:** Helper functions used in the EDA notebooks. Not part of the training pipeline.

| Function | What it does |
|---|---|
| `get_md5_hash(image_path)` | Returns an MD5 hash of an image file — useful for finding exact duplicate images |
| `get_perceptual_hash(image_path)` | Returns a perceptual hash (pHash) — useful for finding visually similar images even if files differ |

---

## 5. Configuration File

All training settings are in `configs/config_local.yaml`. You should never need to edit Python files to change training behaviour — change the config instead.

```yaml
# --- Data ---
train_dir: data/train        # folder created by split_dataset.py
val_dir:   data/validation
test_dir:  data/test

# --- Model ---
num_classes: 23              # number of artists
img_size:    [128, 128]      # images are resized to this before being fed to the model

# --- Training ---
batch_size: 128              # number of images processed at once
epochs:     5                # maximum training epochs (EarlyStopping may stop earlier)
patience:   5                # epochs to wait before EarlyStopping triggers

# --- Augmentation ---
rotation_range:  0.1         # max rotation (fraction of 360°)
zoom_range:      0.1         # max zoom
contrast_range:  0.1         # max contrast adjustment

# --- Compile ---
optimizer:     adam          # adam | sgd | rmsprop
learning_rate: 0.001
loss:          sparse_categorical_crossentropy
metrics:
  - accuracy

# --- Output paths ---
checkpoint_dir: outputs/checkpoints
log_dir:        outputs/logs
```

---

## 6. Outputs

After training you will find:

| File | Description |
|---|---|
| `outputs/checkpoints/best_model.keras` | Best model weights — use this for evaluation and deployment |
| `outputs/checkpoints/final_model.keras` | Weights from the last epoch |
| `outputs/logs/` | TensorBoard logs — visualise with `tensorboard --logdir outputs/logs` |

---

## 7. Common Errors

**`ModuleNotFoundError: No module named 'src'`**
You are running the script with `python3 -m src.train` from outside the project root, or using a Python interpreter outside the virtual environment. Make sure you activate the venv and run from the project root:
```bash
source .venv/bin/activate
python src/train.py
```

**`ValueError: Output directory already contains split folders`**
You already ran `split_dataset.py` before. The split already exists — you do not need to run it again. If you genuinely want to re-split, delete the folders first:
```bash
rm -rf data/train data/validation data/test
python src/split_dataset.py
```

**`train.py: error: the following arguments are required: --config`**
You ran `evaluate.py` without the required arguments. See [Step 3](#step-3--evaluate-the-model) for the correct command.

**`tensorflow-metal` crash on import**
Version mismatch between TensorFlow and tensorflow-metal. Fix with:
```bash
pip install tensorflow-metal --upgrade
```
