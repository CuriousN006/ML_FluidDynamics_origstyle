from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable


DEFAULT_RUN_TAG = "20260310-fluid-rtx3070"
DEFAULT_BRANCH_NAME = f"codex/autoresearch/{DEFAULT_RUN_TAG}"
DEFAULT_SNAPSHOT_DT = 0.2


def discover_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent
    markers = {"pyproject.toml", "CYLINDER_ALL.mat"}
    for candidate in [current, *current.parents]:
        if markers.intersection({item.name for item in candidate.iterdir()}):
            return candidate
    raise FileNotFoundError("Could not discover the project root from the current path.")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = field(default_factory=discover_root)
    run_tag: str = DEFAULT_RUN_TAG

    @property
    def data_file(self) -> Path:
        return self.root / "CYLINDER_ALL.mat"

    @property
    def src_dir(self) -> Path:
        return self.root / "src" / "mlfd"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def linear_dir(self) -> Path:
        return self.output_dir / "linear"

    @property
    def nonlinear_dir(self) -> Path:
        return self.output_dir / "nonlinear"

    @property
    def research_dir(self) -> Path:
        return self.root / "research"

    @property
    def experiment_dir(self) -> Path:
        return self.research_dir / "experiments"

    @property
    def run_dir(self) -> Path:
        return self.research_dir / "runs" / self.run_tag

    @property
    def run_log_dir(self) -> Path:
        return self.output_dir / "autoresearch" / self.run_tag

    @property
    def results_tsv(self) -> Path:
        return self.root / "results.tsv"

    @property
    def state_md(self) -> Path:
        return self.research_dir / "state.md"

    @property
    def final_report_md(self) -> Path:
        return self.research_dir / "final_report.md"

    @property
    def program_md(self) -> Path:
        return self.root / "program.md"

    @property
    def branch_name(self) -> str:
        return f"codex/autoresearch/{self.run_tag}"

    def ensure_directories(self, extra: Iterable[Path] | None = None) -> None:
        directories = [
            self.output_dir,
            self.linear_dir,
            self.nonlinear_dir,
            self.research_dir,
            self.experiment_dir,
            self.run_dir,
            self.run_log_dir,
        ]
        if extra:
            directories.extend(extra)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LinearConfig:
    field_name: str = "VORTALL"
    truncated_rank: int = 10
    leading_modes: int = 5
    mode_movie_count: int = 3
    compare_steps: tuple[int, ...] = (50, 100, 150)
    dmd_ranks: tuple[int, ...] = (5, 10, 15, 20, 30, 50)
    fps: int = 12


@dataclass(frozen=True)
class NonlinearConfig:
    field_name: str = "VORTALL"
    seed: int = 42
    latent_dim: int = 24
    ae_epochs: int = 160
    dyn_epochs: int = 260
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    latent_l1_weight: float = 1e-4
    dyn_l2_weight: float = 1e-5
    rollout_loss_weight: float = 0.15
    dynamics_hidden_dim: int = 64
    dynamics_depth: int = 2
    validation_rollout_weight: float = 0.25
    validation_rollout_horizon: int = 12
    ae_train_ratio: float = 0.9
    dyn_train_ratio: float = 0.8
    early_stopping_patience: int = 20
    compare_steps: tuple[int, ...] = (100, 150)
    num_preview_images: int = 4
    device: str = "auto"
    deterministic: bool = True

    def smoke(self) -> "NonlinearConfig":
        return replace(
            self,
            ae_epochs=2,
            dyn_epochs=2,
            batch_size=8,
            early_stopping_patience=2,
        )


@dataclass(frozen=True)
class AutoresearchConfig:
    run_tag: str = DEFAULT_RUN_TAG
    max_trials: int = 24
    timeout_seconds: int = 12 * 60
    target_runtime_seconds: int = 10 * 60
    comparison_tolerance: float = 0.0025
    vram_reduction_threshold: float = 0.05
    baseline_description: str = "baseline convolutional AE + latent MLP"
