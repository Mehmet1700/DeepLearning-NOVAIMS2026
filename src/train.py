"""Training entrypoint for artist classification models."""

import argparse
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight as skl_compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf


from src import checkpoints, dataset, metric_utils, model


def make_callbacks(best_model_path, patience=6):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_f1_score",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_score", patience=patience, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_f1_score", factor=0.1, patience=3, verbose=1,  min_lr=1e-6
        ),
    ]


def compute_and_log_auc(model, dataset, num_classes, phase_name, mlflow_step):
    """Compute AUC on the validation set and log to MLflow."""
    y_true, y_pred_proba = [], []
    for images, labels in dataset:
        preds = metric_utils.prepare_probabilities_for_sklearn(
            model.predict(images, verbose=0)
        )
        y_true.append(labels.numpy())
        y_pred_proba.append(preds)

    y_true = np.concatenate(y_true)
    y_pred_proba = np.concatenate(y_pred_proba)
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


def train(config_path: str, run_id: str | None = None):
    # Global seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

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
    train_ds, _ = dataset.load_split(
        cfg["train_dir"], img_size, batch_size, augment=augment, config=cfg
    )
    val_ds, _ = dataset.load_split(
        cfg["val_dir"], img_size, batch_size, augment=False, config=cfg
    )

    # Compute class weights to handle imbalanced artist counts (optional)
    class_weight_dict = None
    if cfg.get("compute_class_weight", False):
        print("Computing class weights from training labels...")
        all_labels = np.concatenate([y.numpy() for _, y in train_ds])
        classes = np.unique(all_labels)
        weights = skl_compute_class_weight("balanced", classes=classes, y=all_labels)
        class_weight_dict = dict(zip(classes.astype(int), weights))
        print(f"  Class weights — min: {min(weights):.3f}, max: {max(weights):.3f}")

    checkpoint_dir = cfg.get("checkpoint_dir", "outputs/checkpoints")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Artist_Classification")
    mlflow.tensorflow.autolog(log_models=False)

    run_label = run_id or datetime.now().strftime("%d%m%Y_%H%M%S")
    with mlflow.start_run(run_name=f"{backbone}_{run_label}") as mlflow_run:
        run_context = checkpoints.resolve_output_run_context(
            run_id=run_id,
            active_run_id=mlflow_run.info.run_id,
        )
        checkpoint_layout = checkpoints.build_run_checkpoint_layout(
            checkpoint_dir,
            run_context.output_run_id,
        )
        checkpoints.ensure_run_directories(checkpoint_layout, "phase1")

        safe_log_params(cfg)
        mlflow.log_param("logical_run_id", run_context.logical_run_id)
        mlflow.log_param("output_run_id", run_context.output_run_id)
        if run_context.slurm_job_id is not None:
            mlflow.log_param("slurm_job_id", run_context.slurm_job_id)
        mlflow.log_param("output_checkpoint_run_dir", str(checkpoint_layout.run_dir))

        print(f"Run ID          : {run_context.logical_run_id}")
        print(f"Output Run ID   : {run_context.output_run_id}")
        if run_context.slurm_job_id is not None:
            print(f"SLURM Job ID    : {run_context.slurm_job_id}")
        print(f"Checkpoint Path : {checkpoint_layout.run_dir}")

        # ── Phase 1: train head only, backbone frozen ─────────────────────────
        print(f"\n{'='*60}")
        print(f"Phase 1 — Training head  |  backbone: {backbone}  |  frozen: True")
        print(f"{'='*60}\n")

        trained_model = model.build_model(
            backbone, num_classes, img_size, freeze_base=True, config=cfg
        )
        history1 = trained_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=make_callbacks(checkpoint_layout.phase1_best_model_path, patience),
            class_weight=class_weight_dict,
        )

        best1 = int(np.argmax(history1.history["val_f1_score"]))
        phase_scores = [("phase1", float(history1.history["val_f1_score"][best1]))]
        mlflow.log_metrics(
            {f"phase1_best_{k}": v[best1] for k, v in history1.history.items()},
            step=0,
        )
        mlflow.log_param("phase1_best_epoch", best1 + 1)
        print(f"Phase 1 best epoch: {best1 + 1}")
        compute_and_log_auc(trained_model, val_ds, num_classes, "phase1", 0)

        # ── Phase 2: unfreeze backbone, fine-tune at low LR ───────────────────
        fine_tune_epochs = cfg.get("fine_tune_epochs", 0)
        if fine_tune_epochs > 0 and backbone != "baseline":
            checkpoints.ensure_run_directories(checkpoint_layout, "phase2")
            total_layers = model.configure_fine_tuning(
                trained_model, backbone, fine_tune_unfrozen_layers
            )
            unfrozen_label = (
                str(total_layers)
                if fine_tune_unfrozen_layers == "all"
                else str(fine_tune_unfrozen_layers)
            )

            print(f"\n{'='*60}")
            print(
                "Phase 2 — Fine-tuning    |  backbone: "
                f"{backbone}  |  unfrozen: {unfrozen_label}/{total_layers}"
            )
            print(f"{'='*60}\n")

            # Recompile at a much lower learning rate
            fine_tune_lr = cfg.get("fine_tune_lr", 1e-5)
            model._compile(trained_model, cfg, num_classes, learning_rate=fine_tune_lr)

            history2 = trained_model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=fine_tune_epochs,
                callbacks=make_callbacks(checkpoint_layout.phase2_best_model_path, patience),
                class_weight=class_weight_dict,
            )

            best2 = int(np.argmax(history2.history["val_f1_score"]))
            phase_scores.append(("phase2", float(history2.history["val_f1_score"][best2])))
            mlflow.log_metrics(
                {f"phase2_best_{k}": v[best2] for k, v in history2.history.items()},
                step=1,
            )
            mlflow.log_param("phase2_best_epoch", best2 + 1)
            mlflow.log_param("fine_tune_lr", fine_tune_lr)
            mlflow.log_param("fine_tune_unfrozen_layers", unfrozen_label)
            compute_and_log_auc(trained_model, val_ds, num_classes, "phase2", 1)
            trained_model.save(checkpoint_layout.phase2_final_model_path)
        else:
            trained_model.save(checkpoint_layout.phase1_final_model_path)

        best_phase_name, best_phase_score = checkpoints.select_best_phase(phase_scores)
        best_source_path = checkpoint_layout.best_model_path(best_phase_name)
        shutil.copy2(best_source_path, checkpoint_layout.run_best_model_path)

        mlflow.log_param("best_checkpoint_phase", best_phase_name)
        mlflow.log_metric("best_checkpoint_val_f1_score", best_phase_score)

        print(
            "\nOverall best checkpoint: "
            f"{checkpoint_layout.run_best_model_path} "
            f"({best_phase_name}, val_f1_score={best_phase_score:.4f})"
        )
        print(f"Training complete. Model saved to: {checkpoint_layout.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_config = str(Path(__file__).resolve().parent.parent / "configs" / "config_local.yaml")
    parser.add_argument("--config", default=default_config, help="Path to YAML config file")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional logical run ID. On HPC the checkpoint folder becomes <SLURM_JOB_ID>__<run_id>; locally it stays <run_id>.",
    )
    args = parser.parse_args()
    train(args.config, args.run_id)
