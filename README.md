# Deep Learning Project

Dataset preparation, exploratory analysis, and baseline CNN experimentation for WikiArt artist classification.

## Current Scope

- `src/split_dataset.py` builds deterministic train, validation, and test splits from the raw dataset in `data/wikiart/`.
- `notebooks/NN.ipynb` trains and evaluates a baseline TensorFlow CNN on `data/train`, `data/validation`, and `data/test`.
- `notebooks/explore_wikiart.ipynb`, `notebooks/EDA/EDA.ipynb`, and `notebooks/Data Understanding - Group 8.ipynb` inspect the raw WikiArt dataset.
- `cnn_generalization_strategy_guide.md` captures follow-up ideas for improving generalization.
- `src/main.py` is still a minimal placeholder entrypoint.
- Python commands in this repository use `uv`.

## Repository Tree

```text
deep_learning_project/
├── .gitignore
├── HPC_SETUP.md
├── README.md
├── cnn_generalization_strategy_guide.md
├── documents/
│   └── Deep_Learning_Project.pdf
├── data/
│   ├── wikiart/
│   ├── train/
│   ├── validation/
│   └── test/
├── notebooks/
│   ├── Data Understanding - Group 8.ipynb
│   ├── NN.ipynb
│   ├── explore_wikiart.ipynb
│   └── EDA/
│       ├── EDA.ipynb
│       ├── images_per_artist.png
│       ├── pixel_intensity_boxplot.png
│       ├── pixel_intensity_boxplot_rgb.png
│       ├── pixel_intensity_by_artist.png
│       ├── pixel_intensity_rgb_by_artist.png
│       └── shape_combinations.png
├── requirements.txt
├── src/
│   ├── main.py
│   ├── split_dataset.py
│   └── utils.py
└── tests/
    └── test_split_dataset.py
```

- `.gitignore`: keeps `data/wikiart` tracked while ignoring generated splits and local artifacts.
- `HPC_SETUP.md`: setup and execution notes for running the project on the Deucalion HPC cluster.
- `README.md`: project overview, setup instructions, and the current data-tracking policy.
- `cnn_generalization_strategy_guide.md`: notes for improving CNN validation performance and reducing overfitting.
- `documents/Deep_Learning_Project.pdf`: project brief and supporting reference material.
- `data/wikiart/`: tracked raw WikiArt dataset organized by artist.
- `data/train/`: generated training split created from `data/wikiart`; ignored by Git.
- `data/validation/`: generated validation split created from `data/wikiart`; ignored by Git.
- `data/test/`: generated test split created from `data/wikiart`; ignored by Git.
- `notebooks/Data Understanding - Group 8.ipynb`: exploratory notebook for inspecting the raw dataset and image statistics.
- `notebooks/NN.ipynb`: baseline TensorFlow training and evaluation notebook for the generated dataset splits.
- `notebooks/explore_wikiart.ipynb`: notebook for browsing and summarizing the raw dataset under `data/wikiart`.
- `notebooks/EDA/EDA.ipynb`: notebook that builds an image-level DataFrame for exploratory analysis.
- `src/main.py`: minimal placeholder CLI entrypoint.
- `src/split_dataset.py`: deterministic dataset splitter that copies images from `data/wikiart` into train, validation, and test folders.
- `src/utils.py`: helper functions for duplicate detection and perceptual hashing in the analysis notebooks.
- `tests/test_split_dataset.py`: regression tests for split-dataset path resolution, validation rules, and split generation.

## Data Layout

Tracked raw dataset layout:

```text
data/wikiart/
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

## Split Script Behavior

Default configuration in `src/split_dataset.py`:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "wikiart"
OUTPUT_DIR = DATA_DIR
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
```

The script:

- reads non-hidden class directories from `data/wikiart/`
- copies only `.jpg` files found directly inside each class directory
- writes a fresh split dataset under `data/train`, `data/validation`, and `data/test`
- uses deterministic per-class shuffling with seed `42`
- preserves file metadata via `shutil.copy2`

Validation rules:

- The ratios must sum to `1.0`.
- The source directory must exist and contain class subdirectories.
- The output directory cannot already contain `train`, `validation`, or `test`.
- The source directory may live inside the output root when it is outside the split folder names, such as `data/wikiart`.
- The source directory cannot be one of the split directories or nested inside them.
- Each class must have enough images to keep all three splits non-empty under the configured ratios.
- Re-running requires removing or renaming the existing split folders first.

## Notebook Path Conventions

- The notebooks resolve paths from the project root, so they work whether Jupyter is launched from the repository root or from `notebooks/`.
- Raw-data notebooks read from `data/wikiart/`.
- The training notebook reads from `data/train/`, `data/validation/`, and `data/test/`.
