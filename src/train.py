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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.dataset import load_split
from src.model import build_model, OPTIMIZERS


def make_callbacks(checkpoint_dir, log_dir, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    img_size    = tuple(cfg["img_size"])
    batch_size  = cfg["batch_size"]
    epochs      = cfg["epochs"]
    num_classes = cfg["num_classes"]
    backbone    = cfg.get("backbone", "baseline")
    patience    = cfg.get("patience", 5)

    train_ds, _ = load_split(cfg["train_dir"], img_size, batch_size, augment=True,  config=cfg)
    val_ds,   _ = load_split(cfg["val_dir"],   img_size, batch_size, augment=False, config=cfg)

    checkpoint_dir = cfg.get("checkpoint_dir", "outputs/checkpoints")
    log_dir        = cfg.get("log_dir", "outputs/logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ── Phase 1: train with backbone frozen ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1 — Training head  |  backbone: {backbone}  |  frozen: True")
    print(f"{'='*60}\n")

    model = build_model(backbone, num_classes, img_size, freeze_base=True, config=cfg)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=make_callbacks(checkpoint_dir, log_dir, patience),
    )

    # ── Phase 2: fine-tune full network (transfer learning only) ─────────────
    fine_tune_epochs = cfg.get("fine_tune_epochs", 0)
    if fine_tune_epochs > 0 and backbone != "baseline":
        print(f"\n{'='*60}")
        print(f"Phase 2 — Fine-tuning    |  backbone: {backbone}  |  frozen: False")
        print(f"{'='*60}\n")

        # Unfreeze all layers
        for layer in model.layers:
            layer.trainable = True

        # Recompile at a much lower learning rate
        fine_tune_lr      = cfg.get("fine_tune_lr", 1e-5)
        optimizer_cls     = OPTIMIZERS.get(cfg.get("optimizer", "adam").lower(), tf.keras.optimizers.Adam)
        loss              = cfg.get("loss", "sparse_categorical_crossentropy")
        metrics           = cfg.get("metrics", ["accuracy"])
        model.compile(
            optimizer=optimizer_cls(learning_rate=fine_tune_lr),
            loss=loss,
            metrics=metrics,
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=make_callbacks(checkpoint_dir, log_dir, patience),
        )

    model.save(os.path.join(checkpoint_dir, "final_model.keras"))
    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_config = str(Path(__file__).resolve().parent.parent / "configs" / "config_local.yaml")
    parser.add_argument("--config", default=default_config, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
