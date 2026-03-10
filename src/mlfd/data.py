from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .config import DEFAULT_SNAPSHOT_DT, ProjectPaths


FIELD_LABELS = {
    "UALL": "u",
    "VALL": "v",
    "VORTALL": "vorticity",
}


@dataclass(frozen=True)
class FieldBundle:
    field_name: str
    field_label: str
    matrix: np.ndarray
    frames: np.ndarray
    height: int
    width: int
    dt: float = DEFAULT_SNAPSHOT_DT

    @property
    def num_snapshots(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def flattened_size(self) -> int:
        return int(self.matrix.shape[0])

    def summary(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "field_label": self.field_label,
            "matrix_shape": list(self.matrix.shape),
            "frames_shape": list(self.frames.shape),
            "height": self.height,
            "width": self.width,
            "num_snapshots": self.num_snapshots,
            "dt": self.dt,
            "min": float(self.matrix.min()),
            "max": float(self.matrix.max()),
            "mean": float(self.matrix.mean()),
            "std": float(self.matrix.std()),
        }


def _read_meta(mat_path: Path) -> tuple[int, int]:
    meta = loadmat(mat_path, variable_names=["nx", "ny"])
    height = int(np.asarray(meta["nx"]).squeeze())
    width = int(np.asarray(meta["ny"]).squeeze())
    return height, width


def load_field_bundle(field_name: str = "VORTALL", paths: ProjectPaths | None = None) -> FieldBundle:
    paths = paths or ProjectPaths()
    if not paths.data_file.exists():
        raise FileNotFoundError(f"Data file not found: {paths.data_file}")

    height, width = _read_meta(paths.data_file)
    raw = loadmat(paths.data_file, variable_names=[field_name])[field_name]
    matrix = np.asarray(raw, dtype=np.float32)
    flattened_size = height * width
    if matrix.shape[0] != flattened_size and matrix.shape[1] == flattened_size:
        matrix = matrix.T
    if matrix.shape[0] != flattened_size:
        raise ValueError(
            f"{field_name} has shape {matrix.shape}, which does not match {height}x{width} flattened data."
        )
    frames = matrix.T.reshape(matrix.shape[1], height, width)
    return FieldBundle(
        field_name=field_name,
        field_label=FIELD_LABELS.get(field_name, field_name.lower()),
        matrix=matrix,
        frames=frames,
        height=height,
        width=width,
    )


def random_snapshot_split(num_items: int, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_items)
    rng.shuffle(indices)
    split = max(1, min(num_items - 1, int(num_items * train_ratio)))
    train_idx = np.sort(indices[:split])
    test_idx = np.sort(indices[split:])
    return train_idx, test_idx


def sequential_transition_split(num_snapshots: int, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if num_snapshots < 3:
        raise ValueError("Need at least 3 snapshots for transition modeling.")
    transition_indices = np.arange(num_snapshots - 1)
    split = max(1, min(len(transition_indices) - 1, int(len(transition_indices) * train_ratio)))
    return transition_indices[:split], transition_indices[split:]

