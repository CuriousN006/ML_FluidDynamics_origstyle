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

STRICT_TEMPORAL_HOLDOUT_PROTOCOL = "assignment_temporal_holdout_v1"
EXPECTED_NUM_SNAPSHOTS = 151


@dataclass(frozen=True)
class FieldBundle:
    field_name: str
    field_label: str
    layout: str
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

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def summary(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "field_label": self.field_label,
            "layout": self.layout,
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


def _reshape_frames(matrix: np.ndarray, height: int, width: int) -> np.ndarray:
    return matrix.T.reshape(matrix.shape[1], height, width)


def convert_field_layout(field: np.ndarray, height: int, width: int) -> np.ndarray:
    return field.reshape(-1).reshape(height, width)


def load_field_bundle(
    field_name: str = "VORTALL",
    paths: ProjectPaths | None = None,
    layout: str = "landscape",
) -> FieldBundle:
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
    if layout == "landscape":
        frame_height, frame_width = height, width
    elif layout == "portrait":
        frame_height, frame_width = width, height
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    frames = _reshape_frames(matrix, frame_height, frame_width)
    return FieldBundle(
        field_name=field_name,
        field_label=FIELD_LABELS.get(field_name, field_name.lower()),
        layout=layout,
        matrix=matrix,
        frames=frames,
        height=frame_height,
        width=frame_width,
    )


@dataclass(frozen=True)
class TemporalHoldoutSplit:
    protocol: str
    snapshot_train_idx: np.ndarray
    snapshot_val_idx: np.ndarray
    snapshot_test_idx: np.ndarray
    transition_train_idx: np.ndarray
    transition_val_idx: np.ndarray
    transition_test_idx: np.ndarray

    @property
    def snapshot_train_stop(self) -> int:
        return int(self.snapshot_train_idx[-1]) + 1

    @property
    def snapshot_val_stop(self) -> int:
        return int(self.snapshot_val_idx[-1]) + 1

    @property
    def snapshot_test_stop(self) -> int:
        return int(self.snapshot_test_idx[-1]) + 1

    def snapshot_indices(self) -> dict[str, list[int]]:
        return {
            "train": self.snapshot_train_idx.tolist(),
            "val": self.snapshot_val_idx.tolist(),
            "test": self.snapshot_test_idx.tolist(),
        }

    def transition_indices(self) -> dict[str, list[int]]:
        return {
            "train": self.transition_train_idx.tolist(),
            "val": self.transition_val_idx.tolist(),
            "test": self.transition_test_idx.tolist(),
        }


def build_assignment_temporal_holdout(num_snapshots: int) -> TemporalHoldoutSplit:
    if num_snapshots != EXPECTED_NUM_SNAPSHOTS:
        raise ValueError(
            f"{STRICT_TEMPORAL_HOLDOUT_PROTOCOL} expects {EXPECTED_NUM_SNAPSHOTS} snapshots, found {num_snapshots}."
        )

    # Assignment-aligned protocol: keep an explicit ~10% reconstruction test split,
    # reserve a small temporal validation window, and leave t=150 inside held-out future.
    snapshot_train_idx = np.arange(0, 120, dtype=np.int64)
    snapshot_val_idx = np.arange(120, 135, dtype=np.int64)
    snapshot_test_idx = np.arange(135, 151, dtype=np.int64)
    transition_train_idx = np.arange(0, 119, dtype=np.int64)
    transition_val_idx = np.arange(119, 134, dtype=np.int64)
    transition_test_idx = np.arange(134, 150, dtype=np.int64)

    return TemporalHoldoutSplit(
        protocol=STRICT_TEMPORAL_HOLDOUT_PROTOCOL,
        snapshot_train_idx=snapshot_train_idx,
        snapshot_val_idx=snapshot_val_idx,
        snapshot_test_idx=snapshot_test_idx,
        transition_train_idx=transition_train_idx,
        transition_val_idx=transition_val_idx,
        transition_test_idx=transition_test_idx,
    )
