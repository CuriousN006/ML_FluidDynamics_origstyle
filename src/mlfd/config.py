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
    def nonlinear_dir(self) -> Path:
        return self.output_dir / "nonlinear"

    def ensure_directories(self, extra: Iterable[Path] | None = None) -> None:
        directories = [self.output_dir, self.nonlinear_dir]
        if extra:
            directories.extend(extra)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class NonlinearConfig:
    field_name: str = "VORTALL"
    layout: str = "portrait"
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
    validation_rollout_weight: float = 0.25
    train_rollout_stride: int = 8
    train_rollout_horizon: int = 16
    validation_rollout_stride: int = 4
    validation_rollout_horizon: int = 24
    early_stopping_patience: int = 20
    compare_steps: tuple[int, ...] = (100, 150)
    zoom_crop: tuple[int, int, int, int] = (0, 240, 20, 180)
    error_percentile: float = 99.0
    num_preview_images: int = 4
    device: str = "auto"
    deterministic: bool = True
    save_checkpoint: bool = False

    def smoke(self) -> "NonlinearConfig":
        return replace(
            self,
            ae_epochs=2,
            dyn_epochs=2,
            batch_size=8,
            early_stopping_patience=2,
        )
