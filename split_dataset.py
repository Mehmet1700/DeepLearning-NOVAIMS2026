"""Script for splitting image datasets into train/validation/test folders."""

from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

SOURCE_DIR = Path("wikiart")
OUTPUT_DIR = Path("data")
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

SPLIT_NAMES = ("train", "validation", "test")
IMAGE_SUFFIXES = {".jpg"}


@dataclass(frozen=True)
class SplitRatios:
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class SplitCounts:
    train: int
    validation: int
    test: int


def calculate_split_counts(total_files: int, ratios: SplitRatios) -> SplitCounts:
    """Return raw split counts for one class without applying minimum-size checks."""
    train_count = math.floor(total_files * ratios.train)
    validation_count = math.floor(total_files * ratios.validation)
    test_count = total_files - train_count - validation_count
    return SplitCounts(train=train_count, validation=validation_count, test=test_count)


def minimum_files_for_non_empty_splits(ratios: SplitRatios) -> int:
    """Return the smallest class size that yields non-empty train, validation, and test splits."""
    total_files = len(SPLIT_NAMES)
    while True:
        counts = calculate_split_counts(total_files, ratios)
        if all(
            getattr(counts, split_name) > 0
            for split_name in SPLIT_NAMES
        ):
            return total_files
        total_files += 1


def resolve_ratios() -> SplitRatios:
    """Validate and return the configured split ratios."""
    ratios = SplitRatios(
        train=TRAIN_RATIO,
        validation=VALIDATION_RATIO,
        test=TEST_RATIO,
    )
    total = ratios.train + ratios.validation + ratios.test
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("The train, validation, and test ratios must sum to 1.0.")
    if any(value <= 0 for value in (ratios.train, ratios.validation, ratios.test)):
        raise ValueError("All split ratios must be greater than 0.")
    return ratios


def validate_paths(source: Path, output: Path) -> None:
    """Ensure the source and output directories are valid for splitting."""
    if not source.exists() or not source.is_dir():
        raise ValueError(
            f"Source directory does not exist or is not a directory: {source}"
        )
    if output.exists() and any(
        (output / split_name).exists() for split_name in SPLIT_NAMES
    ):
        raise ValueError(
            f"Output directory already contains split folders: {output}. Remove it or choose another output path.",
        )

    source_resolved = source.resolve()
    output_resolved = output.resolve(strict=False)
    if output_resolved == source_resolved:
        raise ValueError(
            "Output directory must be different from the source directory."
        )
    if source_resolved in output_resolved.parents:
        raise ValueError("Output directory cannot be inside the source directory.")
    if output_resolved in source_resolved.parents:
        raise ValueError("Output directory cannot contain the source directory.")


def list_class_directories(source: Path) -> list[Path]:
    """Return sorted non-hidden class directories from the source dataset."""
    class_dirs = [
        entry
        for entry in sorted(source.iterdir(), key=lambda path: path.name.lower())
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    if not class_dirs:
        raise ValueError(f"No class directories found in source directory: {source}")
    return class_dirs


def list_image_files(class_dir: Path) -> list[Path]:
    """Return supported image files for a single class directory."""
    return [
        entry
        for entry in sorted(class_dir.iterdir(), key=lambda path: path.name.lower())
        if entry.is_file()
        and not entry.name.startswith(".")
        and entry.suffix.lower() in IMAGE_SUFFIXES
    ]


def collect_class_files(source: Path, ratios: SplitRatios) -> dict[str, list[Path]]:
    """Collect image files for each class and enforce the minimum class size."""
    minimum_files = minimum_files_for_non_empty_splits(ratios)
    class_files: dict[str, list[Path]] = {}
    for class_dir in list_class_directories(source):
        image_files = list_image_files(class_dir)
        if len(image_files) < minimum_files:
            raise ValueError(
                f"Class '{class_dir.name}' has fewer than {minimum_files} supported image files: {len(image_files)}.",
            )
        class_files[class_dir.name] = image_files
    return class_files


def compute_split_counts(total_files: int, ratios: SplitRatios) -> SplitCounts:
    """Compute train, validation, and test counts for one class."""
    minimum_files = minimum_files_for_non_empty_splits(ratios)
    if total_files < minimum_files:
        raise ValueError(
            "Class does not contain enough images to keep all split buckets non-empty: "
            f"{total_files} files provided, {minimum_files} required.",
        )

    return calculate_split_counts(total_files, ratios)


def split_files_for_class(
    class_name: str, files: list[Path], ratios: SplitRatios, seed: int
) -> dict[str, list[Path]]:
    """Shuffle one class deterministically and partition files by split."""
    counts = compute_split_counts(len(files), ratios)
    shuffled_files = list(files)
    random.Random(f"{seed}:{class_name}").shuffle(shuffled_files)

    train_end = counts.train
    validation_end = train_end + counts.validation
    return {
        "train": shuffled_files[:train_end],
        "validation": shuffled_files[train_end:validation_end],
        "test": shuffled_files[validation_end:],
    }


def copy_split_files(
    output: Path, class_name: str, split_files: dict[str, list[Path]]
) -> dict[str, int]:
    """Copy split files into the output directory and return written counts."""
    written_counts: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        destination_dir = output / split_name / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for file_path in split_files[split_name]:
            shutil.copy2(file_path, destination_dir / file_path.name)
        written_counts[split_name] = len(split_files[split_name])
    return written_counts


def build_split_dataset(
    source: Path, output: Path, ratios: SplitRatios, seed: int
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Create the split dataset and return per-class and total counts."""
    class_summaries: dict[str, dict[str, int]] = {}
    total_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    class_files = collect_class_files(source, ratios)

    for class_name, image_files in class_files.items():
        split_files = split_files_for_class(class_name, image_files, ratios, seed)
        written_counts = copy_split_files(output, class_name, split_files)
        class_summaries[class_name] = written_counts

        for split_name in SPLIT_NAMES:
            total_counts[split_name] += written_counts[split_name]

    return class_summaries, total_counts


def print_summary(
    source: Path,
    output: Path,
    ratios: SplitRatios,
    seed: int,
    class_summaries: dict[str, dict[str, int]],
    total_counts: dict[str, int],
) -> None:
    """Print a summary of the dataset split configuration and results."""
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(
        "Ratios: "
        f"train={ratios.train:.2f}, validation={ratios.validation:.2f}, test={ratios.test:.2f}"
    )
    print(f"Seed: {seed}")
    print()
    print("Per-class counts:")
    for class_name in sorted(class_summaries):
        counts = class_summaries[class_name]
        print(
            f"  {class_name}: "
            f"train={counts['train']}, validation={counts['validation']}, test={counts['test']}"
        )
    print()
    print(
        "Totals: "
        f"train={total_counts['train']}, validation={total_counts['validation']}, test={total_counts['test']}"
    )


def main() -> None:
    """Run the dataset split using the constants defined at the top of the file."""
    try:
        ratios = resolve_ratios()
        validate_paths(SOURCE_DIR, OUTPUT_DIR)
        class_summaries, total_counts = build_split_dataset(
            SOURCE_DIR, OUTPUT_DIR, ratios, SEED
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error

    print_summary(
        SOURCE_DIR, OUTPUT_DIR, ratios, SEED, class_summaries, total_counts
    )


if __name__ == "__main__":
    main()
