from __future__ import annotations

from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_figure(fig: plt.Figure) -> np.ndarray:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return imageio.imread(buffer)


def _symmetric_limits(field: np.ndarray) -> tuple[float, float]:
    limit = float(np.max(np.abs(field)))
    return -limit, limit


def save_field_image(field: np.ndarray, path: Path, title: str, cmap: str = "RdBu_r", symmetric: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 8.0))
    vmin, vmax = _symmetric_limits(field) if symmetric else (float(field.min()), float(field.max()))
    im = ax.imshow(field, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, path)


def save_singular_spectrum(values: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(values) + 1), values, marker="o", linewidth=1.5)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular value spectrum")
    ax.grid(alpha=0.25)
    _save_figure(fig, path)


def save_complex_plane(eigenvalues: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    circle = plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle="--", alpha=0.6)
    ax.add_patch(circle)
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=18, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    _save_figure(fig, path)


def save_comparison_panel(true_field: np.ndarray, pred_field: np.ndarray, path: Path, title: str) -> None:
    error = pred_field - true_field
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    vmin, vmax = _symmetric_limits(np.stack([true_field, pred_field]))
    images = [
        (true_field, "Ground Truth", "RdBu_r", vmin, vmax),
        (pred_field, "Prediction", "RdBu_r", vmin, vmax),
        (error, "Error", "magma", float(error.min()), float(error.max())),
    ]
    for ax, (field, subtitle, cmap, lo, hi) in zip(axes, images, strict=True):
        im = ax.imshow(field, cmap=cmap, origin="upper", vmin=lo, vmax=hi)
        ax.set_title(subtitle)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    _save_figure(fig, path)


def save_series_gif(fields: np.ndarray, path: Path, title_prefix: str, fps: int = 12) -> None:
    frames: list[np.ndarray] = []
    vmin, vmax = _symmetric_limits(fields)
    for index, field in enumerate(fields):
        fig, ax = plt.subplots(figsize=(4.5, 8.0))
        im = ax.imshow(field, cmap="RdBu_r", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title_prefix} | step={index}")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        frames.append(_render_figure(fig))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def save_rank_sweep(rank_to_metric: dict[int, float], path: Path, ylabel: str, title: str) -> None:
    ranks = sorted(rank_to_metric)
    values = [rank_to_metric[rank] for rank in ranks]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ranks, values, marker="o")
    ax.set_xlabel("Rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    _save_figure(fig, path)


def save_training_curves(history: dict[str, list[float]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in history.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, path)

