"""Training entrypoint for artist classification models."""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.dataset import load_split
from src.model import _compile, build_model, configure_fine_tuning


def make_callbacks(checkpoint_dir, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_f1_score",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_score", patience=patience, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]


def compute_and_log_auc(model, dataset, num_classes, phase_name, mlflow_step):
    """Compute AUC on the validation set and log to MLflow."""
    y_true, y_pred_proba = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_proba.extend(preds)

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))

    try:
        auc_ovr = roc_auc_score(
            y_true_binarized, y_pred_proba, multi_class="ovr", average="weighted"
        )
        mlflow.log_metric(f"{phase_name}_val_auc_ovr", auc_ovr, step=mlflow_step)
        print(f"  {phase_name} val_auc_ovr: {auc_ovr:.4f}")
    except Exception as e:
        print(f"  Warning: Could not compute AUC: {e}")


def safe_log_params(cfg):
    """Log config to MLflow, converting lists to strings to avoid type errors."""
    safe = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in cfg.items()}
    mlflow.log_params(safe)


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    img_size    = tuple(cfg["img_size"])
    batch_size  = cfg["batch_size"]
    epochs      = cfg["epochs"]
    num_classes = cfg["num_classes"]
    backbone    = cfg.get("backbone", "baseline")
    patience    = cfg.get("patience", 5)

    # fine_tune_unfrozen_layers: "all" or a positive integer
    # Config may use num_unfreeze_layers for backwards compat; -1 means "all"
    raw_unfreeze = cfg.get("fine_tune_unfrozen_layers", cfg.get("num_unfreeze_layers", "all"))
    fine_tune_unfrozen_layers = "all" if raw_unfreeze in (-1, "all", None) else int(raw_unfreeze)

    augment     = cfg.get("augment", False)
    train_ds, _ = load_split(cfg["train_dir"], img_size, batch_size, augment=augment, config=cfg)
    val_ds,   _ = load_split(cfg["val_dir"],   img_size, batch_size, augment=False,   config=cfg)

    checkpoint_dir = cfg.get("checkpoint_dir", "outputs/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    mlflow.set_experiment("Artist_Classification")
    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run(run_name=backbone):
        safe_log_params(cfg)

        # ── Phase 1: train head only, backbone frozen ─────────────────────────
        print(f"\n{'='*60}")
        print(f"Phase 1 — Training head  |  backbone: {backbone}  |  frozen: True")
        print(f"{'='*60}\n")

        model = build_model(backbone, num_classes, img_size, freeze_base=True, config=cfg)
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=make_callbacks(checkpoint_dir, patience),
        )

        best1 = int(np.argmax(history1.history["val_f1_score"]))
        mlflow.log_metrics(
            {f"phase1_best_{k}": v[best1] for k, v in history1.history.items()},
            step=0,
        )
        mlflow.log_param("phase1_best_epoch", best1 + 1)
        print(f"Phase 1 best epoch: {best1 + 1}")
        compute_and_log_auc(model, val_ds, num_classes, "phase1", 0)

        # ── Phase 2: unfreeze backbone, fine-tune at low LR ───────────────────
        fine_tune_epochs = cfg.get("fine_tune_epochs", 0)
        if fine_tune_epochs > 0 and backbone != "baseline":
            total_layers = configure_fine_tuning(model, backbone, fine_tune_unfrozen_layers)
            unfrozen_label = str(total_layers) if fine_tune_unfrozen_layers == "all" else str(fine_tune_unfrozen_layers)

            print(f"\n{'='*60}")
            print(f"Phase 2 — Fine-tuning    |  backbone: {backbone}  |  unfrozen: {unfrozen_label}/{total_layers}")
            print(f"{'='*60}\n")

            # Recompile at a much lower learning rate
            fine_tune_lr = cfg.get("fine_tune_lr", 1e-5)
            _compile(model, cfg, num_classes, learning_rate=fine_tune_lr)

            history2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=fine_tune_epochs,
                callbacks=make_callbacks(checkpoint_dir, patience),
            )

            best2 = int(np.argmax(history2.history["val_f1_score"]))
            mlflow.log_metrics(
                {f"phase2_best_{k}": v[best2] for k, v in history2.history.items()},
                step=1,
            )
            mlflow.log_param("phase2_best_epoch", best2 + 1)
            mlflow.log_param("fine_tune_lr", fine_tune_lr)
            mlflow.log_param("fine_tune_unfrozen_layers", unfrozen_label)
            compute_and_log_auc(model, val_ds, num_classes, "phase2", 1)

        model.save(os.path.join(checkpoint_dir, "final_model.keras"))
        print(f"\nTraining complete. Model saved to: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_config = str(Path(__file__).resolve().parent.parent / "configs" / "config_local.yaml")
    parser.add_argument("--config", default=default_config, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
