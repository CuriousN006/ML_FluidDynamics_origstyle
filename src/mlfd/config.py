from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable


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

    @property
    def data_file(self) -> Path:
        return self.root / "CYLINDER_ALL.mat"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def linear_dir(self) -> Path:
        return self.output_dir / "linear"

    @property
    def nonlinear_dir(self) -> Path:
        return self.output_dir / "nonlinear"

    def ensure_directories(self, extra: Iterable[Path] | None = None) -> None:
        directories = [self.output_dir, self.linear_dir, self.nonlinear_dir]
        if extra:
            directories.extend(extra)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LinearConfig:
    field_name: str = "VORTALL"
    layout: str = "portrait"
    truncated_rank: int = 15
    dmd_ranks: tuple[int, ...] = (5, 10, 15, 20, 30, 50)
    compare_steps: tuple[int, ...] = (100, 150)
    zoom_crop: tuple[int, int, int, int] = (0, 240, 20, 180)
    error_percentile: float = 99.0

    def smoke(self) -> "LinearConfig":
        return replace(
            self,
            dmd_ranks=(5, 10, 15),
        )


@dataclass(frozen=True)
class NonlinearConfig:
    field_name: str = "VORTALL"
    layout: str = "portrait"
    profile: str = "full"
    campaign: str | None = None
    reference_full_score: float | None = None
    target_primary_score: float | None = None
    target_wall_seconds: float | None = None
    seed: int = 42
    latent_dim: int = 32
    ae_architecture: str = "residual_refine"
    ae_width_mult: float = 1.0
    coordconv: bool = False
    coarse_loss_weight: float = 0.25
    coarse_blur_kernel: int = 9
    coarse_blur_sigma: float = 2.0
    refine_blocks: int = 1
    refine_channels_mult: float = 1.25
    dynamics_model: str = "residual_linear"
    ae_epochs: int = 160
    dyn_epochs: int = 260
    batch_size: int = 16
    ae_learning_rate: float = 1e-3
    dyn_learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    latent_l1_weight: float = 1e-4
    dyn_l2_weight: float = 0.0
    dyn_deviation_from_linear_weight: float = 1e-3
    gradient_loss_weight: float = 0.1
    fft_loss_weight: float = 0.0
    rollout_loss_weight: float = 0.15
    dynamics_hidden_dim: int = 64
    dynamics_depth: int = 2
    ae_scheduler: str = "none"
    dyn_scheduler: str = "none"
    ae_scheduler_factor: float = 0.5
    dyn_scheduler_factor: float = 0.5
    ae_scheduler_patience: int = 8
    dyn_scheduler_patience: int = 12
    ae_min_learning_rate: float = 1e-5
    dyn_min_learning_rate: float = 1e-5
    ae_cosine_t0_epochs: int = 64
    ae_cosine_tmult: int = 2
    ae_cosine_eta_min: float | None = None
    ae_budget_lr_floor_fraction: float = 0.75
    ae_use_swa: bool = False
    ae_swa_start_fraction: float = 0.75
    ae_swa_lr: float = 1e-5
    ae_use_curriculum: bool = False
    ae_curriculum_finish_fraction: float = 0.8
    ae_curriculum_coarse_start: float = 0.40
    ae_curriculum_gradient_start: float = 0.05
    validation_rollout_weight: float = 0.25
    train_rollout_stride: int = 8
    train_rollout_horizon: int = 16
    validation_rollout_stride: int = 4
    validation_rollout_horizon: int = 24
    dyn_residual_gate_max: float = 0.5
    dyn_residual_gate_init: float = 0.05
    dyn_residual_warmup_fraction: float = 0.35
    dyn_residual_warmup_floor: float = 0.1
    dyn_linear_relative_penalty: float = 2.0
    dyn_select_after_warmup: bool = True
    early_stopping_patience: int = 20
    compare_steps: tuple[int, ...] = (100, 150)
    zoom_crop: tuple[int, int, int, int] = (0, 240, 20, 180)
    error_percentile: float = 99.0
    num_preview_images: int = 4
    device: str = "auto"
    use_amp: bool = False
    deterministic: bool = True
    save_checkpoint: bool = False
    ae_train_budget_seconds: float | None = None
    dyn_train_budget_seconds: float | None = None
    ae_cache_dir: str | None = None

    def smoke(self) -> "NonlinearConfig":
        return replace(
            self,
            ae_epochs=2,
            dyn_epochs=2,
            batch_size=8,
            early_stopping_patience=2,
            ae_train_budget_seconds=min(self.ae_train_budget_seconds, 0.05) if self.ae_train_budget_seconds is not None else None,
            dyn_train_budget_seconds=min(self.dyn_train_budget_seconds, 0.05) if self.dyn_train_budget_seconds is not None else None,
        )
