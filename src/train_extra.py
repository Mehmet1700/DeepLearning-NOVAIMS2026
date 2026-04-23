"""Training entrypoint for artist classification models."""

import argparse
import shutil
import sys
from pathlib import Path

import mlflow
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src import (
    checkpoints,
    class_weights,
    dataset,
    metric_utils,
    mlflow_config,
    model,
    runtime_env,
    training_plots,
)


def validate_training_class_names(
    train_class_names: list[str],
    val_class_names: list[str],
    num_classes: int,
) -> list[str]:
    """Validate split class names before training starts."""
    if train_class_names != val_class_names:
        raise ValueError(
            "Training and validation splits expose different class names or ordering. "
            f"train={train_class_names}, val={val_class_names}"
        )

    if len(train_class_names) != num_classes:
        raise ValueError(
            "Configured num_classes does not match the discovered training classes. "
            f"config num_classes={num_classes}, discovered={len(train_class_names)}, "
            f"class_names={train_class_names}"
        )

    return list(train_class_names)


def make_callbacks(best_model_path, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_f1_score",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_score",
            patience=patience,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]


def compute_class_prior(
    class_names: list[str],
    class_count_map: dict[str, int],
) -> np.ndarray:
    """Return the ordered class-prior vector derived from per-class counts."""
    class_counts = np.asarray(
        [class_count_map[class_name] for class_name in class_names],
        dtype=np.float32,
    )
    total_count = float(np.sum(class_counts))
    if total_count <= 0.0:
        raise ValueError("Training-set class counts must sum to a positive value.")
    return class_counts / total_count


def prior_aware_targets_preserve_argmax(
    class_prior: np.ndarray,
    label_smoothing: float,
) -> bool:
    """Return whether smoothed targets still decode back to the source class."""
    if label_smoothing == 0.0:
        return True

    num_classes = int(class_prior.shape[0])
    hard_targets = np.eye(num_classes, dtype=np.float32)
    smoothed_targets = ((1.0 - label_smoothing) * hard_targets) + (
        label_smoothing * class_prior[np.newaxis, :]
    )
    return np.array_equal(np.argmax(smoothed_targets, axis=1), np.arange(num_classes))


def log_class_prior_summary(
    class_names: list[str],
    class_count_map: dict[str, int],
    class_prior: np.ndarray,
) -> None:
    """Print the resolved training-set class prior vector."""
    print("\nTraining-set class priors:")
    for class_index, class_name in enumerate(class_names):
        print(
            f"  [{class_index:>2}] {class_name}: "
            f"count={class_count_map[class_name]} "
            f"prior={class_prior[class_index]:.6f}"
        )


def compute_and_log_auc(model, dataset, num_classes, phase_name, mlflow_step):
    """Compute AUC on the validation set and log to MLflow."""
    y_true, y_pred_proba = [], []
    for images, labels in dataset:
        preds = metric_utils.prepare_probabilities_for_sklearn(
            model.predict(images, verbose=0)
        )
        y_true.append(metric_utils.prepare_labels_for_sklearn(labels.numpy()))
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


def log_phase_optimizer_params(phase_name: str, optimizer_settings: dict) -> None:
    """Log resolved optimizer settings under phase-specific MLflow param keys."""
    mlflow.log_param(f"{phase_name}_optimizer", optimizer_settings["optimizer_name"])
    mlflow.log_param(f"{phase_name}_learning_rate", optimizer_settings["learning_rate"])
    mlflow.log_param(f"{phase_name}_weight_decay", optimizer_settings["weight_decay"])


def log_runtime_context(require_gpu: bool) -> None:
    """Collect, print, and log the runtime environment metadata."""
    try:
        metadata = runtime_env.collect_runtime_metadata(require_gpu=require_gpu)
    except runtime_env.RuntimeEnvironmentError as error:
        print(runtime_env.format_runtime_report(error.metadata))
        runtime_env.log_runtime_metadata(error.metadata)
        raise

    print(runtime_env.format_runtime_report(metadata))
    runtime_env.log_runtime_metadata(metadata)


def log_training_summary_artifact(
    phase1_history,
    phase2_history=None,
    artifact_file: str = "plots/training_summary.png",
) -> None:
    """Log a combined loss and macro-F1 training summary figure to MLflow."""
    train_loss = list(phase1_history.history["loss"])
    val_loss = list(phase1_history.history["val_loss"])
    train_macro_f1 = list(phase1_history.history["f1_score"])
    val_macro_f1 = list(phase1_history.history["val_f1_score"])
    phase_break_epoch = None

    if phase2_history is not None:
        phase_break_epoch = len(train_loss)
        train_loss.extend(phase2_history.history["loss"])
        val_loss.extend(phase2_history.history["val_loss"])
        train_macro_f1.extend(phase2_history.history["f1_score"])
        val_macro_f1.extend(phase2_history.history["val_f1_score"])

    training_plots.log_training_summary_figure(
        train_loss=train_loss,
        val_loss=val_loss,
        train_macro_f1=train_macro_f1,
        val_macro_f1=val_macro_f1,
        artifact_file=artifact_file,
        phase_break_epoch=phase_break_epoch,
    )


def resolve_phase_optimizer_settings(cfg, phase_name: str) -> dict:
    """Resolve the optimizer settings that should be used for a training phase."""
    if phase_name == "phase1":
        return model.resolve_optimizer_settings(cfg)
    if phase_name == "phase2":
        fine_tune_weight_decay = cfg.get("fine_tune_weight_decay")
        if fine_tune_weight_decay is None:
            fine_tune_weight_decay = cfg.get("weight_decay", 0.0)
        return model.resolve_optimizer_settings(
            cfg,
            learning_rate=cfg.get("fine_tune_lr", 1e-5),
            weight_decay=fine_tune_weight_decay,
        )
    raise ValueError(f"Unsupported phase name: {phase_name}")


def train(
    config_path: str,
    run_id: str | None = None,
    require_gpu: bool = False,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

    img_size = tuple(cfg["img_size"])
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    num_classes = cfg["num_classes"]
    backbone = cfg.get("backbone", "baseline")
    patience = cfg.get("patience", 5)
    target_config = model.resolve_training_targets(cfg)
    effective_label_smoothing = (
        target_config["label_smoothing"]
        if target_config["prior_aware_label_smoothing"]
        else 0.0
    )

    # fine_tune_unfrozen_layers: "all" or a positive integer
    # Config may use num_unfreeze_layers for backwards compat; -1 means "all"
    raw_unfreeze = cfg.get(
        "fine_tune_unfrozen_layers", cfg.get("num_unfreeze_layers", "all")
    )
    fine_tune_unfrozen_layers = (
        "all" if raw_unfreeze in (-1, "all", None) else int(raw_unfreeze)
    )

    checkpoint_dir = cfg.get("checkpoint_dir", "outputs/checkpoints")

    try:
        tracking_config = mlflow_config.get_tracking_config()
    except mlflow_config.TrackingConfigurationError as error:
        raise SystemExit(str(error)) from error

    tracking_uri = tracking_config.uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Artist_Classification")

    with mlflow.start_run(run_name=backbone) as mlflow_run:
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
        mlflow.set_tag("tracking_backend_source", tracking_config.source)
        phase1_optimizer_settings = resolve_phase_optimizer_settings(cfg, "phase1")
        log_phase_optimizer_params("phase1", phase1_optimizer_settings)
        mlflow.log_param("training_target_mode", target_config["target_mode"])
        mlflow.log_param("validation_target_mode", target_config["target_mode"])
        mlflow.log_param("effective_loss", model.resolve_loss_name(cfg))
        mlflow.log_param("effective_label_smoothing", effective_label_smoothing)
        mlflow.log_param(
            "configured_mix_strategy",
            target_config["configured_mix_strategy"] or "<inferred_from_alpha>",
        )
        mlflow.log_param(
            "effective_mix_strategy", target_config["effective_mix_strategy"]
        )
        mlflow.log_param("mixup_alpha", target_config["mixup_alpha"])
        mlflow.log_param("cutmix_alpha", target_config["cutmix_alpha"])
        mlflow.log_param("mixing_enabled", target_config["mixing_enabled"])

        log_runtime_context(require_gpu=require_gpu)

        print(f"Run ID          : {run_context.logical_run_id}")
        print(f"Output Run ID   : {run_context.output_run_id}")
        if run_context.slurm_job_id is not None:
            print(f"SLURM Job ID    : {run_context.slurm_job_id}")
        print(f"MLflow URI      : {tracking_uri}")
        print(f"MLflow Backend  : {tracking_config.description}")
        print(f"Checkpoint Path : {checkpoint_layout.run_dir}")
        print(f"Target Mode     : {target_config['target_mode']}")
        print(f"Loss            : {model.resolve_loss_name(cfg)}")
        print(
            "Mix Strategy    : "
            f"{target_config['configured_mix_strategy'] or '<inferred_from_alpha>'}"
        )
        print(f"Active Mix      : {target_config['effective_mix_strategy']}")
        print(f"MixUp Alpha     : {target_config['mixup_alpha']}")
        print(f"CutMix Alpha    : {target_config['cutmix_alpha']}")

        if (
            not target_config["prior_aware_label_smoothing"]
            and target_config["label_smoothing"] > 0.0
        ):
            print(
                "Ignoring label_smoothing because prior_aware_label_smoothing is disabled; "
                "keeping sparse integer labels and SparseCategoricalCrossentropy."
            )

        augment = cfg.get("augment", False)
        train_ds, train_class_names = dataset.load_split(
            cfg["train_dir"], img_size, batch_size, augment=augment, config=cfg
        )
        val_ds, val_class_names = dataset.load_split(
            cfg["val_dir"], img_size, batch_size, augment=False, config=cfg
        )
        class_names = validate_training_class_names(
            train_class_names,
            val_class_names,
            num_classes,
        )

        # ----------------------------------
        # Class weights
        # ----------------------------------
        def log_class_weight_summary(
            class_names: list[str],
            class_count_map: dict[str, int],
            class_weight_map: dict[int, float],
        ) -> None:
            """Print the resolved per-class counts and balanced weights."""
            print("\nBalanced class weights enabled:")
            for class_index, class_name in enumerate(class_names):
                print(
                    f"  [{class_index:>2}] {class_name}: "
                    f"count={class_count_map[class_name]} "
                    f"weight={class_weight_map[class_index]:.6f}"
                )

        train_class_weight = None
        train_class_counts = None
        if cfg.get("enable_class_weight", False):
            train_class_weight, train_class_counts = (
                class_weights.compute_balanced_class_weights(
                    cfg["train_dir"],
                    class_names,
                )
            )
            log_class_weight_summary(
                class_names, train_class_counts, train_class_weight
            )
            if target_config["mixing_enabled"]:
                print(
                    "\nSoft-label mixing + class_weight enabled:"
                    " Keras applies class weights to soft labels via the dominant class."
                )

        train_class_prior = None
        if target_config["prior_aware_label_smoothing"]:
            if train_class_counts is None:
                train_class_counts = class_weights.compute_class_counts(
                    cfg["train_dir"],
                    class_names,
                )
            train_class_prior = compute_class_prior(class_names, train_class_counts)
            if not prior_aware_targets_preserve_argmax(
                train_class_prior,
                effective_label_smoothing,
            ):
                raise ValueError(
                    "Configured prior-aware label smoothing no longer preserves the original "
                    "class as the argmax target. Reduce label_smoothing so metrics and "
                    "class_weight remain aligned with the source class ids."
                )

            mlflow.log_param(
                "train_class_prior",
                np.array2string(train_class_prior, precision=6, separator=","),
            )
            log_class_prior_summary(class_names, train_class_counts, train_class_prior)

        if target_config["target_mode"] == model.CATEGORICAL_TARGET_MODE:
            print(
                "Validation targets stay hard one-hot in categorical mode so "
                "CategoricalCrossentropy remains valid while validation images remain untouched."
            )

            train_ds, _ = dataset.load_split(
                cfg["train_dir"],
                img_size,
                batch_size,
                augment=augment,
                config=cfg,
                target_mode=model.CATEGORICAL_TARGET_MODE,
                label_smoothing=effective_label_smoothing,
                class_prior=train_class_prior,
                mix_strategy=target_config["effective_mix_strategy"],
                mixup_alpha=target_config["mixup_alpha"],
                cutmix_alpha=target_config["cutmix_alpha"],
            )
            val_ds, _ = dataset.load_split(
                cfg["val_dir"],
                img_size,
                batch_size,
                augment=False,
                config=cfg,
                target_mode=model.CATEGORICAL_TARGET_MODE,
            )

        # ── Phase 1: train head only, backbone frozen ─────────────────────────
        print(f"\n{'=' * 60}")
        print(f"Phase 1 — Training head  |  backbone: {backbone}  |  frozen: True")
        print(f"{'=' * 60}\n")

        trained_model = model.build_model(
            backbone, num_classes, img_size, freeze_base=True, config=cfg
        )
        phase1_fit_kwargs = {
            "validation_data": val_ds,
            "epochs": epochs,
            "callbacks": make_callbacks(
                checkpoint_layout.phase1_best_model_path, patience
            ),
        }
        if train_class_weight is not None:
            phase1_fit_kwargs["class_weight"] = train_class_weight
        history1 = trained_model.fit(
            train_ds,
            **phase1_fit_kwargs,
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
        history2 = None
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

            print(f"\n{'=' * 60}")
            print(
                "Phase 2 — Fine-tuning    |  backbone: "
                f"{backbone}  |  unfrozen: {unfrozen_label}/{total_layers}"
            )
            print(f"{'=' * 60}\n")

            # Recompile at a much lower learning rate
            phase2_optimizer_settings = resolve_phase_optimizer_settings(cfg, "phase2")
            model._compile(
                trained_model,
                cfg,
                num_classes,
                learning_rate=phase2_optimizer_settings["learning_rate"],
                weight_decay=phase2_optimizer_settings["weight_decay"],
            )

            phase2_fit_kwargs = {
                "validation_data": val_ds,
                "epochs": fine_tune_epochs,
                "callbacks": make_callbacks(
                    checkpoint_layout.phase2_best_model_path, patience
                ),
            }
            if train_class_weight is not None:
                phase2_fit_kwargs["class_weight"] = train_class_weight
            history2 = trained_model.fit(
                train_ds,
                **phase2_fit_kwargs,
            )

            best2 = int(np.argmax(history2.history["val_f1_score"]))
            phase_scores.append(
                ("phase2", float(history2.history["val_f1_score"][best2]))
            )
            mlflow.log_metrics(
                {f"phase2_best_{k}": v[best2] for k, v in history2.history.items()},
                step=1,
            )
            mlflow.log_param("phase2_best_epoch", best2 + 1)
            log_phase_optimizer_params("phase2", phase2_optimizer_settings)
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
        log_training_summary_artifact(history1, history2)

        print(
            "\nOverall best checkpoint: "
            f"{checkpoint_layout.run_best_model_path} "
            f"({best_phase_name}, val_f1_score={best_phase_score:.4f})"
        )
        print(f"Training complete. Model saved to: {checkpoint_layout.run_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for training runs."""
    parser = argparse.ArgumentParser()
    default_config = str(
        Path(__file__).resolve().parent.parent / "configs" / "config_local.yaml"
    )
    parser.add_argument(
        "--config", default=default_config, help="Path to YAML config file"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional logical run ID. On HPC the checkpoint folder becomes <SLURM_JOB_ID>__<run_id>; locally it stays <run_id>.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail fast when TensorFlow cannot see at least one physical GPU.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the training CLI."""
    args = build_arg_parser().parse_args(argv)
    train(args.config, args.run_id, require_gpu=args.require_gpu)


if __name__ == "__main__":
    main()
