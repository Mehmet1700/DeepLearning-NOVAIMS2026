# Deep Learning Project

Dataset preparation, exploration, and baseline CNN experimentation for WikiArt artist classification.

## Current Status

- `src/train.py` is the main ML entrypoint; its direct runtime dependencies now live in the `ml` dependency group.
- `src/dataset.py` now writes data-aware TensorFlow disk caches under `outputs/cache/` for runs with `img_size[0] <= 300`, keyed by split path, loader settings, and split file metadata.
- Transfer-learning fine-tuning now keeps the pretrained application backbone as a named nested Keras submodel, so Phase 2 layer unfreezing works for the transfer backbones.
- Training checkpoints are now run-scoped under each config's `checkpoint_dir`, so concurrent HPC jobs and Phase 1/Phase 2 best models no longer overwrite one another.
- `src/evaluate.py` now resolves checkpoints from explicit weights, `--run-id`, a single discovered run folder, or the legacy flat layout for older artifacts.
- Transfer checkpoints now load the correct backbone preprocessing function during evaluation, and newly saved checkpoints use project-registered preprocessing wrappers instead of a bare `preprocess_input` Lambda.
- `src/evaluate.py` and `src/compare_runs.py` build on the training stack and additionally need plotting dependencies from the `dev` group.
- `src/preprocessing/split_dataset.py` builds deterministic train, validation, and test splits from `data/wikiart/`.
- `tests/test_dataset_cache.py` now covers both cache-key invalidation and real TensorFlow disk-cache materialization for small-image runs.
- `tests/test_checkpoint_paths.py` contains regression coverage for run-id precedence, run-scoped checkpoint layout, overall-best selection, and evaluation checkpoint resolution.
- `tests/test_train_evaluate_smoke.py` adds entrypoint-level smoke coverage for `src/train.py` and `src/evaluate.py` using temporary configs and stubbed runtime dependencies.
- `src/utils.py` contains image-hashing helpers used in duplicate-analysis and EDA workflows; those dependencies live in the `preprocessing` group.
- Notebooks and exploratory analysis live behind the `dev` group instead of the core runtime path.
- `data/` and `outputs/` are local-only and ignored by Git, so generated splits, checkpoints, logs, and dataset caches stay out of version control.

## Getting Started

Create and activate the virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Sync only the dependency groups you need:

```bash
# Only the src/train.py runtime
uv sync --only-group ml

# Train, evaluate, compare runs, and open notebooks
uv sync --group ml

# Duplicate-analysis / hashing helpers plus notebooks
uv sync --group preprocessing

# Everything
uv sync --all-groups
```

Dependency-group overview:

- `preprocessing`: image hashing and PIL-based helpers used by `src/utils.py` and duplicate-analysis notebooks.
- `ml`: libraries required to run `src/train.py`.
- `dev`: notebooks, plotting, benchmarking, and optional Vision Transformer notebook dependencies. This is `uv`'s default dependency group.

`requirements.txt` is kept as a legacy snapshot, but the primary workflow is now driven by `pyproject.toml` and `uv` groups.

Prepare the local raw dataset:

1. Create `data/wikiart/` in the project root.
2. Add one subdirectory per artist.
3. Put `.jpg` files directly inside each artist directory.

Generate the split dataset:

```bash
uv run python main.py
```

Open notebooks after syncing the default `dev` group. The duplicate-analysis notebooks also need the `preprocessing` group.

## Training Configuration

Transfer-learning runs use `src/train.py` with a YAML config file:

```bash
uv run --only-group ml python src/train.py --config configs/config_resnet50.yaml
uv run --only-group ml python src/train.py --config configs/config_resnet50.yaml --run-id local-resnet50-debug
```

Evaluation and run-comparison commands need both `ml` and `dev`:

```bash
uv run --group ml python src/evaluate.py --config configs/config_resnet50.yaml --run-id 123456__abcd1234
uv run --group ml python src/evaluate.py --config configs/config_resnet50.yaml --weights outputs/checkpoints/resnet50/123456__abcd1234/best_model.keras
uv run --group ml python src/compare_runs.py --outputs_dir outputs
```

Training stores artifacts under:

- local: `<checkpoint_dir>/<run_id>/`
- HPC: `<checkpoint_dir>/<SLURM_JOB_ID>__<run_id>/`

The logical child `run_id` comes from `--run-id` when provided, otherwise from the active MLflow run id.
On HPC, `SLURM_JOB_ID` is added as a prefix so the checkpoint folder can be matched directly to the SLURM log files.
The run root always contains the overall-best checkpoint across all executed phases.

```text
local:
<checkpoint_dir>/
└── <run_id>/
    ├── best_model.keras
    ├── phase1/
    │   └── best_model.keras
    └── phase2/
        ├── best_model.keras
        └── final_model.keras

HPC:
<checkpoint_dir>/
└── <slurm_job_id>__<run_id>/
    ├── best_model.keras
    ├── phase1/
    │   └── best_model.keras
    └── phase2/
        ├── best_model.keras
        └── final_model.keras
```

When `--weights` is omitted, `src/evaluate.py` loads the run-root `best_model.keras` for the selected folder.
On HPC, the `--run-id` value is the combined folder name, e.g. `123456__abcd1234`.
Without `--run-id`, evaluation auto-selects only when exactly one run folder exists; otherwise it fails fast and asks for `--run-id` or `--weights`.
Legacy flat `<checkpoint_dir>/best_model.keras` checkpoints are still supported for older runs.
For transfer backbones, the evaluation config's `backbone` must match the checkpoint so the correct preprocessing function is registered at load time.

Transfer-learning fine-tuning configs such as `configs/config_resnet50.yaml` support:

- `fine_tune_epochs`: number of Phase 2 epochs.
- `fine_tune_lr`: learning rate used for the lower-LR Phase 2 pass.
- `fine_tune_unfrozen_layers`: number of backbone tail layers to unfreeze in Phase 2. Use `all` for a fully unfrozen model.

## Dataset Cache

For runs with `img_size[0] <= 300`, `src/dataset.py` stores TensorFlow cache files under `outputs/cache/`.

- The cache key includes the resolved split path, image size, batch size, shuffle state, fixed seed, and a fingerprint of the current split files based on relative path, file size, and modification time.
- Changing the split contents or those loader settings creates a new cache path automatically, which avoids reusing stale cached tensors from an older run.
- Augmentation still runs after the cache step, so only the decoded and resized base images are reused across epochs and compatible reruns.

## Project Tree

Core project files:

```text
DeepLearning-NOVAIMS2026/
├── configs/
│   ├── config_template.yaml
│   ├── config_densenet121.yaml
│   ├── config_efficientnetb3.yaml
│   ├── config_local.yaml
│   ├── config_mobilenetv3.yaml
│   ├── config_resnet50.yaml
│   └── config_vgg16.yaml
├── documents/
│   └── Deep_Learning_Project.pdf
├── jobs/
│   ├── evaluate_hpc.slurm
│   └── train_hpc.slurm
├── notebooks/
│   ├── Benchmarking.ipynb
│   ├── Data Understanding - Group 8.ipynb
│   ├── EDA/
│   │   ├── EDA.ipynb
│   │   └── ...
│   ├── NN.ipynb
│   ├── Visual Transformer.ipynb
│   ├── exploration.ipynb
│   └── explore_wikiart.ipynb
├── src/
│   ├── __init__.py
│   ├── checkpoints.py
│   ├── compare_runs.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── preprocessing/
│   │   ├── images_to_remove.json
│   │   ├── remove_duplicates.py
│   │   └── split_dataset.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_checkpoint_paths.py
│   ├── test_dataset_cache.py
│   └── test_train_evaluate_smoke.py
├── HPC_SETUP.md
├── INSTRUCTIONS.md
├── README.md
├── main.py
├── pyproject.toml
├── report(04-04).md
├── requirements.txt
└── uv.lock
```

- `configs/`: YAML configuration files for local runs and backbone-specific training jobs.
- `documents/Deep_Learning_Project.pdf`: project brief and reference material.
- `jobs/evaluate_hpc.slurm`: SLURM job definition for HPC evaluation runs.
- `jobs/train_hpc.slurm`: SLURM job definition for HPC training runs.
- `notebooks/`: exploratory analysis, benchmarking, baseline CNN work, and Vision Transformer experiments.
- `src/__init__.py`: package marker for the shared training and preprocessing modules.
- `src/checkpoints.py`: shared helpers for run-scoped checkpoint layout, run-id selection, and evaluation path resolution.
- `src/compare_runs.py`: aggregates saved evaluation summaries into comparison plots and tables.
- `src/dataset.py`: dataset-loading helpers shared by the training and evaluation code, including the data-aware TensorFlow cache path builder.
- `src/evaluate.py`: evaluation entrypoint for trained models and saved checkpoints.
- `src/model.py`: model construction and fine-tuning utilities for the classification pipeline.
- `src/preprocessing/images_to_remove.json`: curated list of duplicate or invalid WikiArt files to delete from the raw dataset.
- `src/preprocessing/remove_duplicates.py`: cleanup script that removes known duplicate raw WikiArt images.
- `src/preprocessing/split_dataset.py`: dataset splitter rooted at `data/wikiart` and writing splits under `data/`.
- `src/train.py`: training entrypoint for the image classification models.
- `src/utils.py`: image hashing helpers used for duplicate-image analysis workflows.
- `tests/test_checkpoint_paths.py`: regression tests for run-scoped checkpoint layout and evaluation checkpoint selection.
- `tests/test_dataset_cache.py`: regression tests for dataset cache-key stability, invalidation, cache location, and real on-disk cache file creation.
- `tests/test_train_evaluate_smoke.py`: smoke tests for `train.train()` and `evaluate.evaluate()` checkpoint output and checkpoint-loading behavior.
- `HPC_SETUP.md`: notes for running the project on the target HPC environment.
- `INSTRUCTIONS.md`: project workflow notes and runbook-style guidance for the dataset and training pipeline.
- `README.md`: project overview, setup steps, data layout, and workflow notes.
- `main.py`: root entrypoint that removes duplicates and then generates the train, validation, and test split folders.
- `pyproject.toml`: project metadata and grouped dependency configuration for the `uv` workflow.
- `report(04-04).md`: checkpoint-fix report summarizing the run-scoped checkpoint corrections made on April 4, 2026.
- `requirements.txt`: legacy dependency snapshot retained for compatibility with older workflows.
- `uv.lock`: locked dependency set for reproducible `uv` environments.

## Local Data Layout

Expected raw dataset layout:

```text
data/
└── wikiart/
    ├── artist_1/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    ├── artist_2/
    └── ...
```

Generated split layout:

```text
data/
├── wikiart/
│   ├── artist_1/
│   ├── artist_2/
│   └── ...
├── train/
│   ├── artist_1/
│   ├── artist_2/
│   └── ...
├── validation/
│   ├── artist_1/
│   ├── artist_2/
│   └── ...
└── test/
    ├── artist_1/
    ├── artist_2/
    └── ...
```

- `data/wikiart/`: raw input dataset used by the split script and EDA notebooks.
- `data/train/`, `data/validation/`, and `data/test/`: generated split output consumed by the training, evaluation, and notebook workflows.

## Split Script Behavior

Default configuration in [`src/preprocessing/split_dataset.py`](src/preprocessing/split_dataset.py):

```python
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RAW_DATASET_DIR_NAME = "wikiart"
SOURCE_DIR = PROJECT_ROOT / "data" / RAW_DATASET_DIR_NAME
OUTPUT_DIR = PROJECT_ROOT / "data"
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 73
```

The script:

- resolves paths from the file location, so it targets the repository root even though the script lives under `src/preprocessing/`
- reads non-hidden class directories from `data/wikiart/`
- copies only `.jpg` files found directly inside each class directory
- writes a fresh split dataset under `data/train`, `data/validation`, and `data/test`
- uses deterministic per-class shuffling with seed `73`
- preserves file metadata via `shutil.copy2`

Validation rules:

- The ratios must sum to `1.0`.
- The source directory must exist and contain class subdirectories.
- The output directory may contain the raw `data/wikiart/` source folder.
- The output directory cannot already contain `train`, `validation`, or `test`.
- The output directory still cannot equal the source directory or sit inside it.
- Each class must have enough images to keep all three splits non-empty under the configured ratios.
- Re-running requires removing or renaming the existing split folders first.
