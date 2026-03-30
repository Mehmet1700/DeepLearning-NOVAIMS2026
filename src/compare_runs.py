"""
Compare evaluation results across multiple model runs.

Reads all summary.txt files from outputs/run_*/
and produces a bar chart + markdown table ranking models by accuracy.

Usage:
    python src/compare_runs.py
    python src/compare_runs.py --outputs_dir outputs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np


def parse_summary(summary_path):
    """Extract accuracy and per-class F1 from a summary.txt file."""
    result = {"run": summary_path.parent.name, "accuracy": None, "f1_scores": {}}
    with open(summary_path) as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Top-1 Accuracy:"):
            result["accuracy"] = float(line.split(":")[1].strip())
        # Parse per-class F1 from classification report
        parts = line.split()
        if len(parts) == 5:  # class_name precision recall f1 support
            try:
                result["f1_scores"][parts[0]] = float(parts[3])
            except ValueError:
                pass
    return result


def collect_runs(outputs_dir):
    """Find all summary.txt files under outputs/run_*/ folders."""
    outputs_dir = Path(outputs_dir)
    summaries = sorted(outputs_dir.glob("run_*/summary.txt"))
    if not summaries:
        print(f"No run folders found in {outputs_dir}. Run evaluate.py first.")
        return []
    return [parse_summary(s) for s in summaries]


def save_accuracy_chart(runs, output_dir):
    """Bar chart comparing Top-1 Accuracy across runs."""
    names     = [r["run"] for r in runs]
    accuracies = [r["accuracy"] for r in runs]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(runs)))

    fig, ax = plt.subplots(figsize=(max(10, len(runs) * 1.5), 6))
    bars = ax.bar(names, accuracies, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Model Comparison — Top-1 Accuracy on Test Set")
    plt.xticks(rotation=30, ha="right", fontsize=9)

    # Annotate bars with accuracy value
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.4f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    path = output_dir / "comparison_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def save_markdown_table(runs, output_dir):
    """Markdown table ranking models by accuracy."""
    runs_sorted = sorted(runs, key=lambda r: r["accuracy"], reverse=True)
    path = output_dir / "comparison_table.md"
    with open(path, "w") as f:
        f.write("# Model Comparison\n\n")
        f.write("| Rank | Run | Top-1 Accuracy |\n")
        f.write("|------|-----|----------------|\n")
        for i, r in enumerate(runs_sorted, 1):
            f.write(f"| {i} | {r['run']} | {r['accuracy']:.4f} |\n")
    print(f"Saved: {path}")


def compare(outputs_dir):
    runs = collect_runs(outputs_dir)
    if not runs:
        return

    valid_runs = [r for r in runs if r["accuracy"] is not None]
    if not valid_runs:
        print("No valid summary files found.")
        return

    output_dir = Path(outputs_dir) / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(valid_runs)} run(s):\n")
    for r in sorted(valid_runs, key=lambda x: x["accuracy"], reverse=True):
        print(f"  {r['run']:<40} accuracy: {r['accuracy']:.4f}")

    save_accuracy_chart(valid_runs, output_dir)
    save_markdown_table(valid_runs, output_dir)

    print(f"\nComparison saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_outputs = str(Path(__file__).resolve().parent.parent / "outputs")
    parser.add_argument("--outputs_dir", default=default_outputs, help="Path to outputs/ directory")
    args = parser.parse_args()
    compare(args.outputs_dir)
