# Project Instructions — src/ Guide

This document explains the key pipeline files in `src/`, how they connect to each other, and how to run the project from raw data cleanup to evaluation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Step-by-Step Pipeline](#3-step-by-step-pipeline)
4. [File Reference](#4-file-reference)
   - [remove_duplicates.py](#41-remove_duplicatespy)
   - [split_dataset.py](#42-split_datasetpy)
   - [dataset.py](#43-datasetpy)
   - [model.py](#44-modelpy)
   - [checkpoints.py](#45-checkpointspy)
   - [train.py](#46-trainpy)
   - [evaluate.py](#47-evaluatepy)
   - [utils.py](#48-utilspy)
5. [Configuration File](#5-configuration-file)
6. [Outputs](#6-outputs)
7. [Common Errors](#7-common-errors)

---

## 1. Project Overview

This project trains a Convolutional Neural Network (CNN) to classify paintings by artist using the WikiArt dataset. The model looks at a painting and predicts which of the 23 artists painted it.

The pipeline has four main stages:

```
Raw images  →  Remove duplicates  →  Split into folders  →  Train / Evaluate
(data/wikiart)   (remove_duplicates.py)  (split_dataset.py)   (train.py / evaluate.py)
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
│   ├── checkpoints.py        ← checkpoint path helpers and run-id resolution
│   ├── dataset.py            ← loads images from folders into TensorFlow datasets
│   ├── model.py              ← defines the CNN architecture
│   ├── train.py              ← Step 2: trains the model
│   ├── evaluate.py           ← Step 3: evaluates the trained model
│   ├── preprocessing/
│   │   ├── remove_duplicates.py  ← removes known duplicate raw images
│   │   └── split_dataset.py      ← splits raw images into train/val/test
│   └── utils.py              ← helper functions for EDA notebooks
├── main.py                   ← convenience entrypoint for duplicate cleanup + splitting
├── configs/
│   └── config_local.yaml     ← all training settings live here
└── outputs/
    ├── checkpoints/          ← run-scoped model weights after training
    ├── cache/                ← TensorFlow dataset caches for compatible runs
    └── logs/                 ← SLURM job logs and reserved log directory
```

---

## 3. Step-by-Step Pipeline

### Prerequisites

Make sure the virtual environment is activated:
```bash
source .venv/bin/activate
```

---

### Step 1 — Prepare the dataset

**Run once.** The standard entrypoint removes known duplicate raw images and then creates `data/train/`, `data/validation/`, and `data/test/`.

```bash
python main.py
```

If you only want to rebuild the split folders without running duplicate cleanup:

```bash
python src/preprocessing/split_dataset.py
```

> **Important:** The split script refuses to run if `data/train/`, `data/validation/`, or `data/test/` already exist. Remove those folders first if you genuinely want to rebuild the split.

---

### Step 2 — Train the model

```bash
python src/train.py
```

By default this uses `configs/config_local.yaml`. To use a different config:
```bash
python src/train.py --config configs/config_resnet50.yaml
```

You will see progress per batch:
```
Epoch 1/5
292/292 ━━━━━━━━━━━━━━━━━━━━ 120s - f1_score: 0.12 - val_f1_score: 0.15
```

- `292` = number of batches per epoch (total images ÷ batch size)
- Training saves checkpoints under `<checkpoint_dir>/<run_id>/` locally
- On HPC, training saves checkpoints under `<checkpoint_dir>/<SLURM_JOB_ID>__<run_id>/`
- In `configs/config_local.yaml`, this resolves to `outputs/checkpoints/local/<run_id>/`

---

### Step 3 — Evaluate the model

```bash
python src/evaluate.py \
  --config configs/config_local.yaml \
  --run-id <run_id>
```

On HPC, `--run-id` should be the combined folder name, for example `123456__abcd1234`.

If you already know the exact checkpoint path, you can override run-based lookup:

```bash
python src/evaluate.py \
  --config configs/config_local.yaml \
  --weights outputs/checkpoints/local/<run_id>/best_model.keras
```

Output includes:
- A **classification report** with precision, recall, and F1-score per artist
- A **confusion matrix**
- Overall **Top-1 Accuracy**

---

## 4. File Reference

### 4.1 `remove_duplicates.py`

**Purpose:** Deletes known duplicate files from `data/wikiart/` before you build the train, validation, and test folders.

**How it works:**
1. Loads the fixed duplicate path list from `src/preprocessing/images_to_remove.json`
2. Maps those JSON paths into the local `data/wikiart/` directory
3. Deletes files that still exist
4. Prints a short summary of removed, missing, and invalid entries

This script is usually run through `main.py`, so you normally do not need to call it directly.

---

### 4.2 `split_dataset.py`

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

### 4.3 `dataset.py`

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
Build an on-disk cache in outputs/cache/ for img_size <= 300
        ↓
Apply augmentation (training only, random each epoch)
        ↓
Prefetch next batch while GPU trains on current batch
```

**Returns:** `(dataset, class_names)` — the dataset and a list of artist names in alphabetical order (e.g. `["Albrecht_Durer", "Boris_Kustodiev", ...]`).
For larger image sizes, caching is disabled to avoid excessive memory and storage pressure.

---

### 4.4 `model.py`

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
- `metrics` — e.g. `[f1_score]`

---

### 4.5 `checkpoints.py`

**Purpose:** Centralizes checkpoint naming, run-id resolution, and evaluation path lookup.

**What it handles:**
- builds the run-scoped checkpoint layout under:
  - local: `<checkpoint_dir>/<run_id>/`
  - HPC: `<checkpoint_dir>/<SLURM_JOB_ID>__<run_id>/`
- resolves the logical child `run_id` from `--run-id` or the active MLflow run id
- prefixes the checkpoint folder name with `SLURM_JOB_ID` on HPC
- keeps Phase 1 and Phase 2 checkpoint paths separate
- resolves evaluation checkpoints from `--weights`, `--run-id`, a single discovered run folder, or the legacy flat layout

This file is the reason concurrent runs and different training phases no longer overwrite one another.

---

### 4.6 `train.py`

**Purpose:** Runs the full training loop. This is the main script your team will run most often.

**What it does step by step:**
1. Reads all settings from the config YAML
2. Loads the training and validation datasets via `dataset.py`
3. Builds the model via `model.py`
4. Sets up 3 automatic callbacks:

| Callback | What it does |
|---|---|
| `ModelCheckpoint` | Saves `phase1/best_model.keras` or `phase2/best_model.keras` whenever validation F1 improves |
| `EarlyStopping` | Stops training if validation loss does not improve for `patience` epochs |
| `ReduceLROnPlateau` | Halves the learning rate if validation loss stalls for 3 epochs |

5. Calls `model.fit()` to run the training loop
6. Saves a run-root `best_model.keras` copied from the best overall phase, plus phase-specific checkpoint files

**Run:**
```bash
python src/train.py                              # uses config_local.yaml by default
python src/train.py --config configs/other.yaml  # override config
python src/train.py --config configs/other.yaml --run-id local-debug
```

---

### 4.7 `evaluate.py`

**Purpose:** Loads a saved model and runs it against the test set to measure performance.

**What it outputs:**
- **Classification report** — precision, recall, F1-score per artist
- **Confusion matrix** — which artists get confused with each other
- **Top-1 Accuracy** — overall percentage of correct predictions

**Run:**
```bash
python src/evaluate.py \
  --config configs/config_local.yaml \
  --run-id <run_id>
```

> Prefer the run-root `best_model.keras` for evaluation. It is copied from whichever phase achieved the highest validation F1, so a stronger Phase 1 model is preserved even after Phase 2 finishes.

---

### 4.8 `utils.py`

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
backbone: baseline            # local config uses the baseline CNN
num_classes: 23              # number of artists
img_size:    [128, 128]      # images are resized to this before being fed to the model

# --- Training ---
batch_size: 128              # number of images processed at once
epochs:     5                # maximum training epochs (EarlyStopping may stop earlier)
patience:   5                # epochs to wait before EarlyStopping triggers
fine_tune_epochs: 0          # Phase 2 disabled in the local baseline config
fine_tune_lr: 0.00001
fine_tune_unfrozen_layers: all

# --- Augmentation ---
augment: false

# --- Compile ---
optimizer:     adam          # adam | sgd | rmsprop
learning_rate: 0.001
loss:          sparse_categorical_crossentropy
metrics:
  - f1_score

# --- Output paths ---
# Base directory for run-scoped checkpoints (<checkpoint_dir>/<run_id>/...)
checkpoint_dir: outputs/checkpoints/local
log_dir:        outputs/logs/local
```

---

## 6. Outputs

After training you will find:

Here `output_run_id` means:
- local: `run_id`
- HPC: `SLURM_JOB_ID__run_id`

| File or folder | Description |
|---|---|
| local: `<checkpoint_dir>/<run_id>/best_model.keras` | Best overall model weights across all executed phases |
| HPC: `<checkpoint_dir>/<SLURM_JOB_ID>__<run_id>/best_model.keras` | Best overall model weights across all executed phases |
| `<checkpoint_dir>/<output_run_id>/phase1/best_model.keras` | Best Phase 1 checkpoint |
| `<checkpoint_dir>/<output_run_id>/phase2/best_model.keras` | Best Phase 2 checkpoint, when fine-tuning runs |
| `<checkpoint_dir>/<output_run_id>/phase1/final_model.keras` | Final model when training stops after Phase 1 |
| `<checkpoint_dir>/<output_run_id>/phase2/final_model.keras` | Final model when Phase 2 runs |
| `outputs/cache/` | TensorFlow dataset cache files for compatible runs |
| `outputs/logs/` | SLURM job logs and the configured project log directory |

---

## 7. Common Errors

**`ModuleNotFoundError: No module named 'src'`**
You are running the script with `python3 -m src.train` from outside the project root, or using a Python interpreter outside the virtual environment. Make sure you activate the venv and run from the project root:
```bash
source .venv/bin/activate
python src/train.py
```

**`ValueError: Output directory already contains split folders`**
You already ran the split step before. The split already exists, so the script refuses to overwrite it. If you genuinely want to re-split, delete the folders first:
```bash
rm -rf data/train data/validation data/test
python main.py
```

**`Multiple run directories found ... Pass --run-id or --weights explicitly.`**
This happens when a config's `checkpoint_dir` contains more than one run folder and evaluation cannot guess which one you want. Pass either:
```bash
python src/evaluate.py --config configs/config_local.yaml --run-id <run_id>
python src/evaluate.py --config configs/config_local.yaml --weights <path-to-checkpoint>
```
On HPC, the `--run-id` value is the combined folder name, e.g. `123456__abcd1234`.

**`tensorflow-metal` crash on import**
Version mismatch between TensorFlow and tensorflow-metal. Fix with:
```bash
uv pip install tensorflow-metal --upgrade
```
