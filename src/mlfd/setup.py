from __future__ import annotations

import importlib
from typing import Sequence

import torch

from .config import ProjectPaths
from .data import load_field_bundle
from .experiments import init_results_file, refresh_run_index, refresh_state


REQUIRED_MODULES: Sequence[str] = (
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "imageio",
    "sklearn",
    "torch",
)


def main() -> None:
    paths = ProjectPaths()
    paths.ensure_directories()
    init_results_file(paths)
    refresh_state(paths)
    refresh_run_index(paths)
    missing = [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(f"Missing required modules: {', '.join(missing)}")
    bundle = load_field_bundle("VORTALL", paths)
    print(f"Root: {paths.root}")
    print(f"Data file: {paths.data_file}")
    print(f"Snapshot matrix shape: {bundle.matrix.shape}")
    print(f"Frame shape: {bundle.frames.shape}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        x = torch.randn(4, 1, 16, 16, device="cuda")
        layer = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1, device="cuda")
        y = layer(x)
        y.mean().backward()
        print("GPU smoke test: passed")


if __name__ == "__main__":
    main()

