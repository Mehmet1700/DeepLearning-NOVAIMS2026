# Deep Learning Project

Dataset preparation, exploration, and baseline CNN experimentation for WikiArt artist classification.

## Current Status

- `src/train.py` is the main ML entrypoint; its direct runtime dependencies now live in the `ml` dependency group.
- Transfer-learning fine-tuning now keeps the pretrained application backbone as a named nested Keras submodel, so Phase 2 layer unfreezing works for the transfer backbones.
- `src/evaluate.py` and `src/compare_runs.py` build on the training stack and additionally need plotting dependencies from the `dev` group.
- `src/preprocessing/split_dataset.py` builds deterministic train, validation, and test splits from `data/wikiart/`.
- `tests/test_model.py` contains regression coverage for transfer-model backbone lookup and `fine_tune_unfrozen_layers` validation.
- `src/utils.py` contains image-hashing helpers used in duplicate-analysis and EDA workflows; those dependencies live in the `preprocessing` group.
- Notebooks and exploratory analysis live behind the `dev` group instead of the core runtime path.
- `data/` is local-only and ignored by Git, so raw images and generated splits stay out of version control.

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
```

Evaluation and run-comparison commands need both `ml` and `dev`:

```bash
uv run --group ml python src/evaluate.py --config configs/config_local.yaml --weights outputs/checkpoints/best_model.keras
uv run --group ml python src/compare_runs.py --outputs_dir outputs
```

Transfer-learning fine-tuning configs such as `configs/config_resnet50.yaml` support:

- `fine_tune_epochs`: number of Phase 2 epochs.
- `fine_tune_lr`: learning rate used for the lower-LR Phase 2 pass.
- `fine_tune_unfrozen_layers`: number of backbone tail layers to unfreeze in Phase 2. Use `all` for a fully unfrozen model.

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
│   └── test_model.py
├── HPC_SETUP.md
├── INSTRUCTIONS.md
├── README.md
├── main.py
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

- `configs/`: YAML configuration files for local runs and backbone-specific training jobs.
- `documents/Deep_Learning_Project.pdf`: project brief and reference material.
- `jobs/evaluate_hpc.slurm`: SLURM job definition for HPC evaluation runs.
- `jobs/train_hpc.slurm`: SLURM job definition for HPC training runs.
- `notebooks/`: exploratory analysis, benchmarking, baseline CNN work, and Vision Transformer experiments.
- `src/__init__.py`: package marker for the shared training and preprocessing modules.
- `src/compare_runs.py`: aggregates saved evaluation summaries into comparison plots and tables.
- `src/dataset.py`: dataset-loading helpers shared by the training and evaluation code.
- `src/evaluate.py`: evaluation entrypoint for trained models and saved checkpoints.
- `src/model.py`: model construction and fine-tuning utilities for the classification pipeline.
- `src/preprocessing/images_to_remove.json`: curated list of duplicate or invalid WikiArt files to delete from the raw dataset.
- `src/preprocessing/remove_duplicates.py`: cleanup script that removes known duplicate raw WikiArt images.
- `src/preprocessing/split_dataset.py`: dataset splitter rooted at `data/wikiart` and writing splits under `data/`.
- `src/train.py`: training entrypoint for the image classification models.
- `src/utils.py`: image hashing helpers used for duplicate-image analysis workflows.
- `tests/test_model.py`: regression tests covering transfer-learning backbone lookup and Phase 2 fine-tuning validation.
- `HPC_SETUP.md`: notes for running the project on the target HPC environment.
- `INSTRUCTIONS.md`: project workflow notes and runbook-style guidance for the dataset and training pipeline.
- `README.md`: project overview, setup steps, data layout, and workflow notes.
- `main.py`: root entrypoint that removes duplicates and then generates the train, validation, and test split folders.
- `pyproject.toml`: project metadata and grouped dependency configuration for the `uv` workflow.
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
- `data/train/`, `data/validation/`, and `data/test/`: generated split output consumed by `notebooks/NN.ipynb`.

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
