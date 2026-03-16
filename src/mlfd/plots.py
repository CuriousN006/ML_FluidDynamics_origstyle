from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _symmetric_limits(field: np.ndarray) -> tuple[float, float]:
    limit = float(np.max(np.abs(field)))
    return -limit, limit


def _crop_field(field: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return field
    row_start, row_end, col_start, col_end = crop
    return field[row_start:row_end, col_start:col_end]


def _figure_size(field: np.ndarray, portrait_scale: float = 8.0) -> tuple[float, float]:
    aspect = field.shape[0] / max(1, field.shape[1])
    width = max(3.6, portrait_scale / max(1.2, aspect))
    height = max(4.8, width * aspect)
    return width, height


def save_field_image(
    field: np.ndarray,
    path: Path,
    title: str,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    interpolation: str = "bilinear",
) -> None:
    fig, ax = plt.subplots(figsize=_figure_size(field))
    vmin, vmax = _symmetric_limits(field) if symmetric else (float(field.min()), float(field.max()))
    im = ax.imshow(field, cmap=cmap, origin="upper", interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, path)


def save_comparison_panel(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    path: Path,
    title: str,
    *,
    crop: tuple[int, int, int, int] | None = None,
    interpolation: str = "bilinear",
    error_percentile: float | None = None,
) -> None:
    true_field = _crop_field(true_field, crop)
    pred_field = _crop_field(pred_field, crop)
    error = np.abs(pred_field - true_field)
    fig, axes = plt.subplots(1, 3, figsize=(14, 8))
    vmin, vmax = _symmetric_limits(np.stack([true_field, pred_field]))
    if error_percentile is None:
        error_hi = float(error.max())
    else:
        error_hi = float(np.percentile(error, error_percentile))
        error_hi = max(error_hi, 1e-8)
    images = [
        (true_field, "Ground Truth", "RdBu_r", vmin, vmax),
        (pred_field, "Prediction", "RdBu_r", vmin, vmax),
        (error, "Absolute Error", "magma", 0.0, error_hi),
    ]
    for ax, (field, subtitle, cmap, lo, hi) in zip(axes, images, strict=True):
        im = ax.imshow(field, cmap=cmap, origin="upper", interpolation=interpolation, vmin=lo, vmax=hi)
        ax.set_title(subtitle)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
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
