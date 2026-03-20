# Deep Learning Project

Dataset preparation and baseline CNN experimentation for WikiArt artist classification.

## Current Scope

- `split_dataset.py` builds deterministic train, validation, and test splits from a local `wikiart/` dataset.
- `NN.ipynb` loads the generated `data/` splits and trains a baseline TensorFlow CNN.
- `cnn_generalization_strategy_guide.md` captures follow-up architecture and regularization improvements.
- `main.py` is still a minimal placeholder entrypoint.
- Python dependencies and commands are managed with `uv`.

## Getting Started

Install the environment:

```bash
uv sync
```

Prepare the local dataset:

1. Create a `wikiart/` directory in the project root.
2. Add one subdirectory per artist.
3. Put `.jpg` files directly inside each artist directory.

Generate the dataset split:

```bash
uv run python split_dataset.py
```

Open the notebook for training and evaluation:

```bash
uv run jupyter lab
```

`data/` and `wikiart/` are intentionally ignored by Git because they are local dataset directories and can make the repository unnecessarily large.

## Repository Tree

```text
deep_learning_project/
├── .gitignore
├── .python-version
├── cnn_generalization_strategy_guide.md
├── Deep_Learning_Project.pdf
├── NN.ipynb
├── README.md
├── main.py
├── pyproject.toml
├── split_dataset.py
└── uv.lock
```

- `.gitignore`: excludes local datasets, caches, notebook checkpoints, and common training artifacts.
- `.python-version`: pins the local Python interpreter version.
- `cnn_generalization_strategy_guide.md`: notes for improving CNN validation performance and reducing overfitting.
- `Deep_Learning_Project.pdf`: project brief and reference material.
- `NN.ipynb`: notebook that loads `data/`, trains a baseline CNN, plots curves, and evaluates on the test split.
- `README.md`: project overview, setup instructions, and dataset workflow notes.
- `main.py`: placeholder CLI entrypoint.
- `pyproject.toml`: project metadata and dependency configuration for `uv`.
- `split_dataset.py`: deterministic dataset split utility for local image folders.
- `uv.lock`: locked dependency versions for reproducible environments.

## Local Data Layout

Expected raw dataset layout:

```text
wikiart/
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

- `wikiart/`: local raw dataset used as input to the split script.
- `data/`: generated split output consumed by `NN.ipynb`.

## Split Script Behavior

Default configuration in [`split_dataset.py`](/Users/alexandre/Documents/deep_learning_project/split_dataset.py):

```python
SOURCE_DIR = Path("wikiart")
OUTPUT_DIR = Path("data")
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
```

The script:

- reads non-hidden class directories from `wikiart/`
- copies only `.jpg` files found directly inside each class directory
- writes a fresh split dataset under `data/`
- uses deterministic per-class shuffling with seed `42`
- preserves file metadata via `shutil.copy2`

Validation rules:

- The ratios must sum to `1.0`.
- The source directory must exist and contain class subdirectories.
- The output directory cannot already contain `train`, `validation`, or `test`.
- Each class must have enough images to keep all three splits non-empty under the configured ratios.
- Re-running requires removing or renaming the existing `data/` split folders first.
