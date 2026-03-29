"""
Training loop for artist classification.
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.dataset import load_split
from src.model import build_model


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    img_size    = tuple(cfg["img_size"])
    batch_size  = cfg["batch_size"]
    epochs      = cfg["epochs"]
    num_classes = cfg["num_classes"]
    backbone    = cfg.get("backbone", "efficientnetb3")

    train_ds, _ = load_split(
        cfg["train_dir"], img_size, batch_size, augment=True, config=cfg
    )
    val_ds, _ = load_split(
        cfg["val_dir"], img_size, batch_size, augment=False, config=cfg
    )

    model = build_model(backbone, num_classes, img_size, config=cfg)

    checkpoint_dir = cfg.get("checkpoint_dir", "outputs/checkpoints")
    log_dir        = cfg.get("log_dir", "outputs/logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.get("patience", 5), restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.save(os.path.join(checkpoint_dir, "final_model.keras"))
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_config = str(Path(__file__).resolve().parent.parent / "configs" / "config_local.yaml")
    parser.add_argument("--config", default=default_config, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
