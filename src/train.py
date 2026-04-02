"""
Training loop for artist classification.

Supports two-phase training for transfer learning models:
  Phase 1: backbone frozen, only head is trained
  Phase 2: full network fine-tuned at a lower learning rate (fine_tune_epochs in config)
"""

import argparse
import yaml
import os
import sys
from datetime import datetime
from pathlib import Path
import mlflow
import mlflow.tensorflow
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.dataset import load_split
from src.model import build_model, OPTIMIZERS, _resolve_metrics, SparseCategoricalF1Score
from src.evaluate import generate_and_log_confusion_matrix, generate_and_log_classification_report


def make_callbacks(checkpoint_dir, log_dir, patience, phase_name="", timestamp=""):
    """Create training callbacks.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        patience: Early stopping patience
        phase_name: Phase identifier (e.g., "phase1", "phase2") for checkpoint naming
        timestamp: Timestamp string for checkpoint naming
    """
    # Add phase and timestamp to best_model filename if provided
    best_model_filename = f"best_model_{phase_name}_{timestamp}.keras" if phase_name and timestamp else "best_model.keras"
    
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, best_model_filename),
            monitor="val_f1_score",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_score", patience=patience, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_f1_score", factor=0.5, patience=3, verbose=1
        ),
    ]


def compute_class_weights(train_ds, num_classes):
    """Compute class weights from training dataset to balance imbalanced classes.
    
    Args:
        train_ds: Training dataset (tf.data.Dataset).
        num_classes: Number of classes.
    
    Returns:
        dict: Dictionary mapping class indices to weights.
    """
    y_true = []
    for _, labels in train_ds:
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    
    # Compute weights using sklearn's utility
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=y_true
    )
    
    # Convert to dictionary format
    return {i: float(w) for i, w in enumerate(class_weights)}


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility across numpy, TensorFlow, and Python's random module.
    
    Args:
        seed: Random seed value (default: 42)
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    print(f"Random seeds set to: {seed}")


def compute_and_log_auc(model, dataset, num_classes, phase_name, mlflow_step):
    """Compute AUC metrics on a dataset and log to MLflow."""
    y_true, y_pred_proba = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_proba.extend(preds)
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Binarize labels for AUC computation
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))
    
    # Compute AUC (one-vs-rest)
    try:
        auc_ovr = roc_auc_score(y_true_binarized, y_pred_proba, multi_class='ovr', average='weighted')
        mlflow.log_metric(f"{phase_name}_val_auc_ovr", auc_ovr, step=mlflow_step)
        print(f"  {phase_name} val_auc_ovr: {auc_ovr:.4f}")
    except Exception as e:
        print(f"  Warning: Could not compute AUC OvR: {e}")


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Set random seeds for reproducibility
    seed = cfg.get("seed", 42)
    set_seeds(seed)

    img_size    = tuple(cfg["img_size"])
    batch_size  = cfg["batch_size"]
    epochs      = cfg["epochs"]
    num_classes = cfg["num_classes"]
    backbone    = cfg.get("backbone", "baseline")
    patience    = cfg.get("patience", 5)

    augment     = cfg.get("augment", False)
    train_ds, class_names = load_split(cfg["train_dir"], img_size, batch_size, augment=augment, config=cfg)
    val_ds,   _            = load_split(cfg["val_dir"],   img_size, batch_size, augment=False,   config=cfg)

    class_weight = None
    # Compute class weights if specified in config (useful for imbalanced datasets)
    if cfg.get("compute_class_weight", False):
        print("\nComputing class weights from training data...")
        class_weight = compute_class_weights(train_ds, num_classes)
        print(f"Computed class weights: {class_weight}\n")

    checkpoint_dir = cfg.get("checkpoint_dir", "outputs/checkpoints")
    log_dir        = cfg.get("log_dir", "outputs/logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize MLflow
    mlflow.set_experiment("Artist_Classification_Train_Val")

    # Set log_models=True to save the model in MLflow's artifact store
    mlflow.tensorflow.autolog(log_models=True)
    
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    with mlflow.start_run(run_name=f"{backbone}_{timestamp}"):
        # Log your YAML config parameters
        mlflow.log_params(cfg)
        
        # ── Phase 1: train with backbone frozen ──────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Phase 1 — Training head  |  backbone: {backbone}  |  frozen: True")
        print(f"{'='*60}\n")

        model = build_model(backbone, num_classes, img_size, freeze_base=True, config=cfg)
        history_phase1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weight,
            callbacks=make_callbacks(checkpoint_dir, log_dir, patience, phase_name="phase1", timestamp=timestamp),
        )
        
        # Log best epoch metrics (where val_f1_score was highest) to MLflow
        best_epoch_idx = np.argmax(history_phase1.history['val_f1_score'])
        for metric_name, metric_values in history_phase1.history.items():
            mlflow.log_metrics({f"phase1_best_{metric_name}": metric_values[best_epoch_idx]}, step=0)
        mlflow.log_param("phase1_best_epoch", best_epoch_idx + 1)
        print(f"Phase 1 best epoch: {best_epoch_idx + 1}")
        
        # Compute and log AUC for Phase 1 (training and validation)
        compute_and_log_auc(model, train_ds, num_classes, "phase1_train", 0)
        compute_and_log_auc(model, val_ds, num_classes, "phase1_val", 0)
        
        # Generate and log confusion matrices and classification reports
        print("\n  Generating Phase 1 metrics visualizations...")
        
        # Compute predictions for training data
        y_true_train, y_pred_train = [], []
        for images, labels in train_ds:
            preds = model.predict(images, verbose=0)
            y_true_train.extend(labels.numpy())
            y_pred_train.extend(np.argmax(preds, axis=1))
        y_true_train, y_pred_train = np.array(y_true_train), np.array(y_pred_train)
        
        # Compute predictions for validation data
        y_true_val, y_pred_val = [], []
        for images, labels in val_ds:
            preds = model.predict(images, verbose=0)
            y_true_val.extend(labels.numpy())
            y_pred_val.extend(np.argmax(preds, axis=1))
        y_true_val, y_pred_val = np.array(y_true_val), np.array(y_pred_val)
        
        generate_and_log_confusion_matrix(y_true_train, y_pred_train, class_names, artifact_path="phase1_train_artifacts")
        generate_and_log_confusion_matrix(y_true_val, y_pred_val, class_names, artifact_path="phase1_val_artifacts")
        generate_and_log_classification_report(y_true_train, y_pred_train, class_names, artifact_path="phase1_train_artifacts")
        generate_and_log_classification_report(y_true_val, y_pred_val, class_names, artifact_path="phase1_val_artifacts")
        
        # Save Phase 1 final model
        model.save(os.path.join(checkpoint_dir, f"final_model_phase1_{timestamp}.keras"))

        # ── Phase 2: fine-tune full network (transfer learning only) ─────────────
        fine_tune_epochs = cfg.get("fine_tune_epochs", 0)
        if fine_tune_epochs > 0 and backbone != "baseline" and backbone != "perceptron":
            print(f"\n{'='*60}")
            print(f"Phase 2 — Fine-tuning    |  backbone: {backbone}  |  frozen: False")
            print(f"{'='*60}\n")

            # Unfreeze/keep frozen layers according to num_unfreeze_layers configuration
            num_unfreeze_layers = cfg.get("num_unfreeze_layers", -1)
            if num_unfreeze_layers is None or num_unfreeze_layers == -1:
                # Unfreeze all layers
                for layer in model.layers:
                    layer.trainable = True
            else:
                # Unfreeze last N layers from the end
                total_layers = len(model.layers)
                start_unfreeze = total_layers - num_unfreeze_layers
                for i, layer in enumerate(model.layers):
                    layer.trainable = i >= start_unfreeze

            # Recompile at a much lower learning rate
            fine_tune_lr  = cfg.get("fine_tune_lr", 1e-5)
            optimizer_cls = OPTIMIZERS.get(cfg.get("optimizer", "adam").lower(), tf.keras.optimizers.Adam)
            loss          = cfg.get("loss", "sparse_categorical_crossentropy")
            metrics       = _resolve_metrics(cfg.get("metrics", ["f1_score"]), num_classes)
            model.compile(
                optimizer=optimizer_cls(learning_rate=fine_tune_lr),
                loss=loss,
                metrics=metrics,
            )

            history_phase2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=fine_tune_epochs,
                class_weight=class_weight,
                callbacks=make_callbacks(checkpoint_dir, log_dir, patience, phase_name="phase2", timestamp=timestamp),
            )
            
            # Log best epoch metrics for Phase 2 (after recompilation with lower LR)
            best_epoch_idx_2 = np.argmax(history_phase2.history['val_f1_score'])
            for metric_name, metric_values in history_phase2.history.items():
                mlflow.log_metrics({f"phase2_best_{metric_name}": metric_values[best_epoch_idx_2]}, step=1)
            mlflow.log_param("phase2_best_epoch", best_epoch_idx_2 + 1)
            
            # Compute and log AUC for Phase 2 (training and validation)
            compute_and_log_auc(model, train_ds, num_classes, "phase2_train", 1)
            compute_and_log_auc(model, val_ds, num_classes, "phase2_val", 1)
            
            # Generate and log confusion matrices and classification reports
            print("\n  Generating Phase 2 metrics visualizations...")
            
            # Compute predictions for training data
            y_true_train, y_pred_train = [], []
            for images, labels in train_ds:
                preds = model.predict(images, verbose=0)
                y_true_train.extend(labels.numpy())
                y_pred_train.extend(np.argmax(preds, axis=1))
            y_true_train, y_pred_train = np.array(y_true_train), np.array(y_pred_train)
            
            # Compute predictions for validation data
            y_true_val, y_pred_val = [], []
            for images, labels in val_ds:
                preds = model.predict(images, verbose=0)
                y_true_val.extend(labels.numpy())
                y_pred_val.extend(np.argmax(preds, axis=1))
            y_true_val, y_pred_val = np.array(y_true_val), np.array(y_pred_val)
            
            generate_and_log_confusion_matrix(y_true_train, y_pred_train, class_names, artifact_path="phase2_train_artifacts")
            generate_and_log_confusion_matrix(y_true_val, y_pred_val, class_names, artifact_path="phase2_val_artifacts")
            generate_and_log_classification_report(y_true_train, y_pred_train, class_names, artifact_path="phase2_train_artifacts")
            generate_and_log_classification_report(y_true_val, y_pred_val, class_names, artifact_path="phase2_val_artifacts")
            
            # Save Phase 2 final model
            model.save(os.path.join(checkpoint_dir, f"final_model_phase2_{timestamp}.keras"))

        # Log final summary metrics to MLflow
        mlflow.log_param("backbone", backbone)
        mlflow.log_param("total_epochs_phase1", epochs)
        if fine_tune_epochs > 0 and backbone != "baseline" and backbone != "perceptron":
            mlflow.log_param("total_epochs_phase2", fine_tune_epochs)
        
        print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_config = str(Path(__file__).resolve().parent.parent / "configs" / "config_local.yaml")
    parser.add_argument("--config", default=default_config, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
