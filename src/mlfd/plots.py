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


def save_latent_trajectory_pca(
    true_latents: np.ndarray,
    predicted_latents: np.ndarray,
    path: Path,
    title: str,
) -> None:
    centered = true_latents - true_latents.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T if vt.size else np.zeros((true_latents.shape[1], 2), dtype=np.float32)
    if basis.shape[1] < 2:
        basis = np.pad(basis, ((0, 0), (0, 2 - basis.shape[1])))
    true_proj = centered @ basis
    pred_proj = (predicted_latents - true_latents.mean(axis=0, keepdims=True)) @ basis

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    steps = np.arange(true_proj.shape[0])
    scatter = ax.scatter(true_proj[:, 0], true_proj[:, 1], c=steps, cmap="viridis", s=20, label="Encoded latent")
    ax.plot(pred_proj[:, 0], pred_proj[:, 1], color="black", linewidth=1.2, alpha=0.7, label="Rollout latent")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Time step")
    _save_figure(fig, path)


def save_latent_time_series(
    true_latents: np.ndarray,
    predicted_latents: np.ndarray,
    path: Path,
    title: str,
    dims: int = 3,
) -> None:
    num_dims = min(dims, true_latents.shape[1], predicted_latents.shape[1])
    fig, axes = plt.subplots(num_dims, 1, figsize=(8, 2.6 * num_dims), sharex=True)
    if num_dims == 1:
        axes = [axes]
    steps = np.arange(true_latents.shape[0])
    for dim_index, ax in enumerate(axes):
        ax.plot(steps, true_latents[:, dim_index], label="Encoded latent", linewidth=1.6)
        ax.plot(steps, predicted_latents[:, dim_index], label="Rollout latent", linewidth=1.2, linestyle="--")
        ax.set_ylabel(f"z{dim_index + 1}")
        ax.grid(alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time step")
    axes[0].legend()
    _save_figure(fig, path)
