# Deep Learning Project

Dataset preparation, exploration, and baseline CNN experimentation for WikiArt artist classification.

## Current Status

- `src/preprocessing/split_dataset.py` builds deterministic train, validation, and test splits from `data/wikiart/`.
- `notebooks/NN.ipynb` loads `data/train`, `data/validation`, and `data/test` for baseline TensorFlow training.
- `notebooks/EDA/EDA.ipynb` and `notebooks/explore_wikiart.ipynb` inspect the raw dataset under `data/wikiart/`.
- `cnn_generalization_strategy_guide.md` captures follow-up regularization and architecture ideas.
- `main.py` runs duplicate cleanup and dataset splitting in sequence.
- `data/` is local-only and ignored by Git, so raw images and generated splits stay out of version control.

## Getting Started

Install the environment with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Prepare the local raw dataset:

1. Create `data/wikiart/` in the project root.
2. Add one subdirectory per artist.
3. Put `.jpg` files directly inside each artist directory.

Generate the split dataset:

```bash
uv run python -m src.preprocessing.split_dataset
```

Open the notebooks in your preferred Jupyter environment after installing Jupyter in that environment.

## Repository Tree

Tracked repository files:

```text
DeepLearning-NOVAIMS2026/
├── .gitignore
├── HPC_SETUP.md
├── INSTRUCTIONS.md
├── README.md
├── cnn_generalization_strategy_guide.md
├── configs/
│   ├── config_local.yaml
│   └── ...
├── documents/
│   └── Deep_Learning_Project.pdf
├── jobs/
│   └── train_hpc.slurm
├── main.py
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
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── compare_runs.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── preprocess.py
│   ├── preprocessing/
│   │   ├── images_to_remove.json
│   │   ├── remove_duplicates.py
│   │   └── split_dataset.py
│   ├── train.py
│   └── utils.py
└── uv.lock
```

- `.gitignore`: ignores local datasets, virtual environments, caches, and training artifacts.
- `HPC_SETUP.md`: notes for running the project on the target HPC environment.
- `INSTRUCTIONS.md`: project workflow notes and runbook-style guidance for the dataset and training pipeline.
- `README.md`: project overview, setup steps, data layout, and workflow notes.
- `cnn_generalization_strategy_guide.md`: recommendations for improving CNN generalization and reducing overfitting.
- `configs/config_local.yaml`: local training configuration template; sibling config files cover model-specific runs.
- `documents/Deep_Learning_Project.pdf`: project brief and reference material.
- `jobs/train_hpc.slurm`: SLURM job definition for HPC training runs.
- `main.py`: root entrypoint that removes duplicates and then generates the train, validation, and test split folders.
- `notebooks/Benchmarking.ipynb`: notebook for comparing model runs and inspecting benchmark results.
- `notebooks/Data Understanding - Group 8.ipynb`: exploratory notebook covering early dataset understanding work.
- `notebooks/EDA/EDA.ipynb`: exploratory data analysis notebook for raw WikiArt images.
- `notebooks/NN.ipynb`: baseline training and evaluation notebook that consumes the generated split dataset.
- `notebooks/Visual Transformer.ipynb`: notebook for Vision Transformer experiments on the generated splits.
- `notebooks/exploration.ipynb`: notebook for general raw-data and class exploration.
- `notebooks/explore_wikiart.ipynb`: notebook for inspecting dataset availability and raw image coverage.
- `pyproject.toml`: project metadata and Python packaging configuration for the `uv` workflow.
- `requirements.txt`: Python dependency list for the local environment.
- `src/compare_runs.py`: utilities for comparing saved experiment outputs.
- `src/dataset.py`: dataset-loading helpers shared by the training and evaluation code.
- `src/evaluate.py`: evaluation entrypoint for trained models and saved checkpoints.
- `src/model.py`: model construction utilities for the classification pipeline.
- `src/preprocess.py`: preprocessing helpers used before training and evaluation.
- `src/preprocessing/images_to_remove.json`: curated list of duplicate or invalid WikiArt files to delete from the raw dataset.
- `src/preprocessing/remove_duplicates.py`: cleanup script that removes known duplicate raw WikiArt images.
- `src/preprocessing/split_dataset.py`: dataset splitter rooted at `data/wikiart` and writing splits under `data/`.
- `src/train.py`: training entrypoint for the image classification models.
- `src/utils.py`: image hashing helpers used for duplicate-image analysis workflows.
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
