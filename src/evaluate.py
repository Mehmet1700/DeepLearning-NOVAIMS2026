"""Evaluation entrypoint for artist classification checkpoints."""

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import yaml
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from src import dataset, model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_weights_path(config, weights_path=None):
    """Resolve the checkpoint path from the CLI override or config checkpoint_dir."""
    raw_path = weights_path
    if raw_path is None:
        checkpoint_dir = config.get("checkpoint_dir", "outputs/checkpoints")
        raw_path = str(Path(checkpoint_dir) / "best_model.keras")

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_weights_path(weights_path):
    """Fail fast with a config-aware error when the checkpoint file is missing."""
    if weights_path.is_file():
        return

    raise FileNotFoundError(
        "Model weights not found at "
        f"'{weights_path}'. Pass --weights explicitly or set checkpoint_dir in the "
        "config so evaluation can locate '<checkpoint_dir>/best_model.keras'."
    )


def generate_and_log_confusion_matrix(y_true, y_pred, class_names, artifact_path="test_artifacts"):
    """Generate confusion matrix and log as image to MLflow (temp file only).
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        class_names: List of class names for axis labels
        artifact_path: MLflow artifact path (default: "test_artifacts")
    """
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
    
    # Save to temp file and log to MLflow (file will be cleaned up after logging)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    
    fig.savefig(tmp_path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    os.remove(tmp_path)  # Clean up temp file
    print(f"  Logged confusion matrix to MLflow")


def generate_and_log_classification_report(y_true, y_pred, class_names, artifact_path="test_artifacts"):
    """Generate classification report chart and log as image to MLflow (temp file only).
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        class_names: List of class names for axis labels
        artifact_path: MLflow artifact path (default: "test_artifacts")
    """
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
    
    # Save to temp file and log to MLflow (file will be cleaned up after logging)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    
    fig.savefig(tmp_path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    os.remove(tmp_path)  # Clean up temp file
    print(f"  Logged classification report to MLflow")


def evaluate(config_path: str, weights_path: str | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    img_size   = tuple(cfg["img_size"])
    batch_size = cfg["batch_size"]
    backbone = cfg.get("backbone", "baseline")
    resolved_weights_path = resolve_weights_path(cfg, weights_path)
    validate_weights_path(resolved_weights_path)
    custom_objects = model.get_model_custom_objects(backbone)

    test_ds, class_names = dataset.load_split(
        cfg["test_dir"], img_size, batch_size, augment=False
    )

    loaded_model = tf.keras.models.load_model(
        resolved_weights_path,
        safe_mode=False,
        compile=False,
        custom_objects=custom_objects,
    )

    # Initialize MLflow
    mlflow.set_experiment("Artist_Classification_Test")

    with mlflow.start_run(run_name=f"Eval_{backbone}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}", nested=True):
        # Log YAML configuration parameters
        mlflow.log_params(cfg)
        
        y_true, y_pred, y_pred_proba = [], [], []
        for images, labels in test_ds:
            preds = loaded_model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))
            y_pred_proba.extend(preds)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)

        accuracy = np.mean(y_true == y_pred)
        
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Compute AUC using one-vs-rest strategy
        y_true_binarized = label_binarize(y_true, classes=range(len(class_names)))
        auc_ovr = roc_auc_score(y_true_binarized, y_pred_proba, multi_class='ovr', average='weighted')
        auc_ovo = roc_auc_score(y_true_binarized, y_pred_proba, multi_class='ovo', average='weighted')
        
        # Log metrics to MLflow
        mlflow.log_metric("test_f1_weighted", f1_weighted)
        mlflow.log_metric("test_f1_macro", f1_macro)
        mlflow.log_metric("test_auc_ovr", auc_ovr)
        mlflow.log_metric("test_auc_ovo", auc_ovo)

        # Generate and log visualizations to MLflow (temp files only, no repo artifacts)
        print("\nGenerating evaluation visualizations...")
        generate_and_log_confusion_matrix(y_true, y_pred, class_names, artifact_path="test_artifacts")
        generate_and_log_classification_report(y_true, y_pred, class_names, artifact_path="test_artifacts")

        print(f"\nTop-1 Accuracy: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"AUC (OvR): {auc_ovr:.4f}")
        print(f"AUC (OvO): {auc_ovo:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_config  = str(PROJECT_ROOT / "configs" / "config_local.yaml")
    parser.add_argument("--config",  default=default_config,  help="Path to YAML config file")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to saved model (.keras / .h5). Defaults to <checkpoint_dir>/best_model.keras from the config.",
    )
    args = parser.parse_args()
    evaluate(args.config, args.weights)
