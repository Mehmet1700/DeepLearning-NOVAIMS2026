# Deep Learning Project

Dataset preparation, exploration, and baseline CNN experimentation for WikiArt artist classification.

## Current Status

- `src/split_dataset.py` builds deterministic train, validation, and test splits from `data/wikiart/`.
- `notebooks/NN.ipynb` loads `data/train`, `data/validation`, and `data/test` for baseline TensorFlow training.
- `notebooks/EDA/EDA.ipynb` and `notebooks/explore_wikiart.ipynb` inspect the raw dataset under `data/wikiart/`.
- `cnn_generalization_strategy_guide.md` captures follow-up regularization and architecture ideas.
- `src/main.py` remains a minimal placeholder entrypoint.
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
uv run python src/split_dataset.py
```

Open the notebooks in your preferred Jupyter environment after installing Jupyter in that environment.

## Repository Tree

Tracked repository files:

```text
deep_learning_project/
├── .gitignore
├── HPC_SETUP.md
├── README.md
├── cnn_generalization_strategy_guide.md
├── documents/
│   └── Deep_Learning_Project.pdf
├── notebooks/
│   ├── Data Understanding - Group 8.ipynb
│   ├── EDA/
│   │   ├── EDA.ipynb
│   │   ├── images_per_artist.png
│   │   ├── pixel_intensity_boxplot.png
│   │   ├── pixel_intensity_boxplot_rgb.png
│   │   ├── pixel_intensity_by_artist.png
│   │   ├── pixel_intensity_rgb_by_artist.png
│   │   └── shape_combinations.png
│   ├── NN.ipynb
│   └── explore_wikiart.ipynb
├── requirements.txt
└── src/
    ├── main.py
    ├── split_dataset.py
    └── utils.py
```

- `.gitignore`: ignores local datasets, virtual environments, caches, and training artifacts.
- `HPC_SETUP.md`: notes for running the project on the target HPC environment.
- `README.md`: project overview, setup steps, data layout, and workflow notes.
- `cnn_generalization_strategy_guide.md`: recommendations for improving CNN generalization and reducing overfitting.
- `documents/Deep_Learning_Project.pdf`: project brief and reference material.
- `notebooks/Data Understanding - Group 8.ipynb`: exploratory notebook covering early dataset understanding work.
- `notebooks/EDA/EDA.ipynb`: exploratory data analysis notebook for raw WikiArt images.
- `notebooks/EDA/images_per_artist.png`: saved chart of image counts by artist.
- `notebooks/EDA/pixel_intensity_boxplot.png`: saved grayscale intensity distribution chart.
- `notebooks/EDA/pixel_intensity_boxplot_rgb.png`: saved RGB intensity boxplot chart.
- `notebooks/EDA/pixel_intensity_by_artist.png`: saved grayscale intensity chart split by artist.
- `notebooks/EDA/pixel_intensity_rgb_by_artist.png`: saved RGB intensity chart split by artist.
- `notebooks/EDA/shape_combinations.png`: saved chart of image shape combinations.
- `notebooks/NN.ipynb`: baseline training and evaluation notebook that consumes the generated split dataset.
- `notebooks/explore_wikiart.ipynb`: notebook for inspecting dataset availability and raw image coverage.
- `requirements.txt`: Python dependency list for the local environment.
- `src/main.py`: placeholder CLI entrypoint.
- `src/split_dataset.py`: dataset splitter rooted at `data/wikiart` and writing splits under `data/`.
- `src/utils.py`: image hashing helpers used for duplicate-image analysis workflows.

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

Default configuration in [`src/split_dataset.py`](/Users/alexandre/Documents/deep_learning_project/src/split_dataset.py):

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATASET_DIR_NAME = "wikiart"
SOURCE_DIR = PROJECT_ROOT / "data" / RAW_DATASET_DIR_NAME
OUTPUT_DIR = PROJECT_ROOT / "data"
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
```

The script:

- resolves paths from the file location, so it works when launched from the repository root or from `src/`
- reads non-hidden class directories from `data/wikiart/`
- copies only `.jpg` files found directly inside each class directory
- writes a fresh split dataset under `data/train`, `data/validation`, and `data/test`
- uses deterministic per-class shuffling with seed `42`
- preserves file metadata via `shutil.copy2`

Validation rules:

- The ratios must sum to `1.0`.
- The source directory must exist and contain class subdirectories.
- The output directory may contain the raw `data/wikiart/` source folder.
- The output directory cannot already contain `train`, `validation`, or `test`.
- The output directory still cannot equal the source directory or sit inside it.
- Each class must have enough images to keep all three splits non-empty under the configured ratios.
- Re-running requires removing or renaming the existing split folders first.
