"""Remove duplicate WikiArt images from the tracked raw dataset."""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WIKIART_DIR = REPO_ROOT / "data" / "wikiart"
IMAGES_TO_REMOVE_PATH = SCRIPT_DIR / "images_to_remove.json"
JSON_PATH_PREFIX = "DeepLearning-NOVAIMS2026/data/wikiart/"


def _load_images_to_remove(json_path: Path) -> list[object]:
    """Load the duplicate image list from JSON."""
    with json_path.open("r", encoding="utf-8") as file_handle:
        images_to_remove = json.load(file_handle)

    if not isinstance(images_to_remove, list):
        raise ValueError(f"{json_path} must contain a JSON array of image paths")

    return images_to_remove


def _build_target_path(image_path: str) -> Path:
    """Map one fixed JSON entry to its file under data/wikiart."""
    if not image_path.startswith(JSON_PATH_PREFIX):
        raise ValueError(f"Skipped invalid path: {image_path}")

    relative_path = image_path.removeprefix(JSON_PATH_PREFIX)
    if not relative_path:
        raise ValueError(f"Skipped invalid path: {image_path}")

    return WIKIART_DIR / relative_path


def _remove_duplicate_images() -> dict[str, int]:
    """Delete duplicate images listed in the bundled JSON file."""
    images_to_remove = _load_images_to_remove(IMAGES_TO_REMOVE_PATH)
    stats = {"removed": 0, "missing": 0, "skipped_invalid": 0}

    for image_path in images_to_remove:
        if not isinstance(image_path, str):
            print(f"Skipped invalid path entry: {image_path!r}")
            stats["skipped_invalid"] += 1
            continue

        try:
            target_path = _build_target_path(image_path)
        except ValueError:
            print(f"Skipped invalid path: {image_path}")
            stats["skipped_invalid"] += 1
            continue

        if target_path.exists():
            target_path.unlink()
            stats["removed"] += 1
        else:
            print(f"File already deleted: {target_path}")
            stats["missing"] += 1

    return stats


def remove_duplicate_images() -> None:
    """Remove duplicate WikiArt images using the bundled JSON path list."""
    if not WIKIART_DIR.is_dir():
        raise FileNotFoundError(f"WikiArt dataset directory not found: {WIKIART_DIR}")

    stats = _remove_duplicate_images()
    print(f"Removed {stats['removed']} duplicate images")
    print(f"Missing files: {stats['missing']}")
    print(f"Skipped invalid paths: {stats['skipped_invalid']}")


if __name__ == "__main__":
    remove_duplicate_images()
