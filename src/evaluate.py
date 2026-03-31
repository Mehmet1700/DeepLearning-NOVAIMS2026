"""
Evaluation and metrics for artist classification.
Saves visual outputs to outputs/<run_timestamp>/
"""

import argparse
import yaml
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import load_split
from src.model import SparseCategoricalF1Score


def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize per row

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (normalized)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def save_classification_report_chart(y_true, y_pred, class_names, output_dir):
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    artists    = class_names
    precision  = [report[a]["precision"] for a in artists]
    recall     = [report[a]["recall"]    for a in artists]
    f1         = [report[a]["f1-score"]  for a in artists]

    x = np.arange(len(artists))
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(x - width, precision, width, label="Precision", color="steelblue")
    ax.bar(x,         recall,    width, label="Recall",    color="darkorange")
    ax.bar(x + width, f1,        width, label="F1-Score",  color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(artists, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall and F1-Score")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "classification_report.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def save_summary(y_true, y_pred, class_names, output_dir):
    accuracy = np.mean(y_true == y_pred)
    report   = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    path = output_dir / "summary.txt"
    with open(path, "w") as f:
        f.write(f"Top-1 Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved: {path}")
    return accuracy


def evaluate(config_path: str, weights_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    img_size   = tuple(cfg["img_size"])
    batch_size = cfg["batch_size"]

    test_ds, class_names = load_split(
        cfg["test_dir"], img_size, batch_size, augment=False
    )

    model = tf.keras.models.load_model(
        weights_path,
        safe_mode=False,
        compile=False,
        custom_objects={"SparseCategoricalF1Score": SparseCategoricalF1Score},
    )

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Create timestamped output folder, prefixed with model name if set
    backbone   = cfg.get("backbone", "baseline")
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name   = f"run_{backbone}_{timestamp}"
    output_dir = Path(cfg.get("checkpoint_dir", "outputs/checkpoints")).parent.parent / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_dir}\n")

    accuracy = save_summary(y_true, y_pred, class_names, output_dir)
    save_confusion_matrix(y_true, y_pred, class_names, output_dir)
    save_classification_report_chart(y_true, y_pred, class_names, output_dir)

    print(f"\nTop-1 Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_config  = str(Path(__file__).resolve().parent.parent / "configs" / "config_local.yaml")
    default_weights = str(Path(__file__).resolve().parent.parent / "outputs" / "checkpoints" / "best_model.keras")
    parser.add_argument("--config",  default=default_config,  help="Path to YAML config file")
    parser.add_argument("--weights", default=default_weights, help="Path to saved model (.keras / .h5)")
    args = parser.parse_args()
    evaluate(args.config, args.weights)
