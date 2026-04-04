"""Checkpoint helpers for run-scoped model outputs."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ResolvedRunContext:
    """Resolved logical and output run identifiers for one training job."""

    logical_run_id: str
    output_run_id: str
    slurm_job_id: str | None


@dataclass(frozen=True)
class RunCheckpointLayout:
    """Run-scoped checkpoint paths for the training pipeline."""

    base_dir: Path
    run_id: str
    run_dir: Path
    phase1_dir: Path
    phase2_dir: Path
    run_best_model_path: Path
    phase1_best_model_path: Path
    phase1_final_model_path: Path
    phase2_best_model_path: Path
    phase2_final_model_path: Path

    def phase_dir(self, phase_name: str) -> Path:
        """Return the directory for a training phase."""
        phase_dirs = {
            "phase1": self.phase1_dir,
            "phase2": self.phase2_dir,
        }
        try:
            return phase_dirs[phase_name]
        except KeyError as error:
            raise ValueError(f"Unsupported phase name: {phase_name}") from error

    def best_model_path(self, phase_name: str) -> Path:
        """Return the best-model path for a training phase."""
        phase_paths = {
            "phase1": self.phase1_best_model_path,
            "phase2": self.phase2_best_model_path,
        }
        try:
            return phase_paths[phase_name]
        except KeyError as error:
            raise ValueError(f"Unsupported phase name: {phase_name}") from error

    def final_model_path(self, phase_name: str) -> Path:
        """Return the final-model path for a training phase."""
        phase_paths = {
            "phase1": self.phase1_final_model_path,
            "phase2": self.phase2_final_model_path,
        }
        try:
            return phase_paths[phase_name]
        except KeyError as error:
            raise ValueError(f"Unsupported phase name: {phase_name}") from error


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve a project-relative or absolute path."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_run_id(raw_run_id: str) -> str:
    """Ensure the run ID is safe to use as a single folder name."""
    run_id = str(raw_run_id).strip()
    if not run_id:
        raise ValueError("Run ID cannot be empty.")
    if run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("Run ID must be a single path segment without separators.")
    return run_id


def resolve_logical_run_id(
    run_id: str | None = None,
    active_run_id: str | None = None,
) -> str:
    """Resolve the logical child run ID from CLI or MLflow context."""
    if run_id is not None:
        return validate_run_id(run_id)

    if active_run_id:
        return validate_run_id(active_run_id)

    raise ValueError("Could not determine a run ID. Pass --run-id explicitly.")


def resolve_output_run_context(
    run_id: str | None = None,
    env: dict[str, str] | None = None,
    active_run_id: str | None = None,
) -> ResolvedRunContext:
    """Resolve the logical run ID and the final checkpoint folder name."""
    logical_run_id = resolve_logical_run_id(run_id=run_id, active_run_id=active_run_id)
    environment = os.environ if env is None else env
    raw_slurm_job_id = environment.get("SLURM_JOB_ID")
    slurm_job_id = validate_run_id(raw_slurm_job_id) if raw_slurm_job_id else None
    output_run_id = (
        logical_run_id
        if slurm_job_id is None
        else f"{slurm_job_id}__{logical_run_id}"
    )
    return ResolvedRunContext(
        logical_run_id=logical_run_id,
        output_run_id=output_run_id,
        slurm_job_id=slurm_job_id,
    )


def resolve_output_run_id(
    run_id: str | None = None,
    env: dict[str, str] | None = None,
    active_run_id: str | None = None,
) -> str:
    """Resolve the final checkpoint folder name for one training run."""
    return resolve_output_run_context(
        run_id=run_id,
        env=env,
        active_run_id=active_run_id,
    ).output_run_id


def build_run_checkpoint_layout(checkpoint_dir: str | Path, run_id: str) -> RunCheckpointLayout:
    """Build the run-scoped checkpoint layout from the config base directory."""
    base_dir = resolve_project_path(checkpoint_dir)
    resolved_run_id = validate_run_id(run_id)
    run_dir = base_dir / resolved_run_id
    phase1_dir = run_dir / "phase1"
    phase2_dir = run_dir / "phase2"

    return RunCheckpointLayout(
        base_dir=base_dir,
        run_id=resolved_run_id,
        run_dir=run_dir,
        phase1_dir=phase1_dir,
        phase2_dir=phase2_dir,
        run_best_model_path=run_dir / "best_model.keras",
        phase1_best_model_path=phase1_dir / "best_model.keras",
        phase1_final_model_path=phase1_dir / "final_model.keras",
        phase2_best_model_path=phase2_dir / "best_model.keras",
        phase2_final_model_path=phase2_dir / "final_model.keras",
    )


def ensure_run_directories(layout: RunCheckpointLayout, *phase_names: str) -> None:
    """Create the run root and any requested phase directories."""
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    for phase_name in phase_names:
        layout.phase_dir(phase_name).mkdir(parents=True, exist_ok=True)


def select_best_phase(phase_scores: list[tuple[str, float]]) -> tuple[str, float]:
    """Return the phase with the highest score, keeping the first phase on ties."""
    if not phase_scores:
        raise ValueError("At least one phase score is required.")

    best_phase_name, best_score = phase_scores[0]
    for phase_name, score in phase_scores[1:]:
        if score > best_score:
            best_phase_name = phase_name
            best_score = score
    return best_phase_name, best_score


def list_run_directories(checkpoint_dir: str | Path) -> list[Path]:
    """List run directories that contain run-scoped checkpoint outputs."""
    base_dir = resolve_project_path(checkpoint_dir)
    if not base_dir.exists():
        return []

    run_dirs = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "best_model.keras").is_file() or (child / "phase1").is_dir() or (child / "phase2").is_dir():
            run_dirs.append(child)
    return run_dirs


def resolve_evaluation_weights_path(
    checkpoint_dir: str | Path,
    weights_path: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Resolve evaluation weights from an explicit path or a run-scoped layout."""
    if weights_path is not None:
        return resolve_project_path(weights_path)

    if run_id is not None:
        layout = build_run_checkpoint_layout(checkpoint_dir, run_id)
        return layout.run_best_model_path

    run_dirs = list_run_directories(checkpoint_dir)
    if len(run_dirs) == 1:
        return run_dirs[0] / "best_model.keras"

    if len(run_dirs) > 1:
        available_run_ids = ", ".join(path.name for path in run_dirs)
        base_dir = resolve_project_path(checkpoint_dir)
        raise ValueError(
            f"Multiple run directories found under '{base_dir}': {available_run_ids}. "
            "Pass --run-id or --weights explicitly."
        )

    base_dir = resolve_project_path(checkpoint_dir)
    legacy_best_model_path = base_dir / "best_model.keras"
    return legacy_best_model_path
