from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LinearConfig, ProjectPaths
from .data import convert_field_layout, load_field_bundle
from .metrics import nrmse, relative_l2, rmse
from .plots import (
    save_comparison_panel,
    save_complex_plane,
    save_field_image,
    save_rank_sweep,
    save_series_gif,
    save_singular_spectrum,
)
from .utils import utc_timestamp, write_json


def _compare_steps(steps: tuple[int, ...], num_snapshots: int) -> tuple[int, ...]:
    return tuple(step for step in steps if 0 <= step < num_snapshots)


def _reconstruct_truncated(u: np.ndarray, s: np.ndarray, vh: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]
    weights = np.diag(s_r) @ vh_r
    reconstruction = u_r @ weights
    return u_r, weights, reconstruction


def _fit_linear_dynamics(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = weights[:, :-1]
    future = weights[:, 1:]
    operator = future @ np.linalg.pinv(current)
    predicted = np.zeros_like(weights)
    predicted[:, 0] = weights[:, 0]
    for step in range(1, weights.shape[1]):
        predicted[:, step] = operator @ predicted[:, step - 1]
    return operator, predicted


def _exact_dmd(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    x1 = matrix[:, :-1]
    x2 = matrix[:, 1:]
    u, s, vh = np.linalg.svd(x1, full_matrices=False)
    rank = min(rank, np.linalg.matrix_rank(x1), len(s))
    u_r = u[:, :rank]
    s_r = s[:rank]
    v_r = vh[:rank, :].T
    inv_sigma = np.diag(1.0 / s_r)
    a_tilde = u_r.T @ x2 @ v_r @ inv_sigma
    eigenvalues, w = np.linalg.eig(a_tilde)
    phi = x2 @ v_r @ inv_sigma @ w
    b = np.linalg.lstsq(phi, matrix[:, 0], rcond=None)[0]
    predictions = []
    for step in range(matrix.shape[1]):
        dynamics = (eigenvalues**step) * b
        snapshot = phi @ dynamics
        predictions.append(np.real_if_close(snapshot))
    predicted_matrix = np.column_stack(predictions).astype(np.float32)
    return eigenvalues, predicted_matrix


def run_linear_pipeline(config: LinearConfig | None = None, paths: ProjectPaths | None = None, output_tag: str = "baseline") -> dict[str, object]:
    config = config or LinearConfig()
    paths = paths or ProjectPaths()
    bundle = load_field_bundle(config.field_name, paths)
    compare_steps = _compare_steps(config.compare_steps, bundle.num_snapshots)
    output_dir = paths.linear_dir / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom_crop = (0, 240, 20, 180)

    matrix = bundle.matrix
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    save_singular_spectrum(s, output_dir / "singular_spectrum.png")
    for mode_idx in range(config.leading_modes):
        field = u[:, mode_idx].reshape(bundle.height, bundle.width)
        save_field_image(field, output_dir / f"mode_{mode_idx + 1}.png", f"Spatial mode {mode_idx + 1}")

    for mode_idx in range(config.mode_movie_count):
        mode_series = (s[mode_idx] * np.outer(u[:, mode_idx], vh[mode_idx, :])).T.reshape(
            bundle.num_snapshots, bundle.height, bundle.width
        )
        save_series_gif(mode_series, output_dir / f"mode_{mode_idx + 1}.gif", f"Mode {mode_idx + 1}", config.fps)

    u_r, weights, truncated_recon = _reconstruct_truncated(u, s, vh, config.truncated_rank)
    truncated_metrics: dict[str, dict[str, float]] = {}
    for step in compare_steps:
        true_snapshot = bundle.frames[step]
        pred_snapshot = truncated_recon[:, step].reshape(bundle.height, bundle.width)
        truncated_metrics[f"step_{step}"] = {
            "rmse": rmse(true_snapshot, pred_snapshot),
            "nrmse": nrmse(true_snapshot, pred_snapshot),
            "relative_l2": relative_l2(true_snapshot, pred_snapshot),
        }
        save_comparison_panel(
            true_snapshot,
            pred_snapshot,
            output_dir / f"truncated_reconstruction_step_{step}.png",
            f"Truncated SVD reconstruction | step {step}",
        )
        save_comparison_panel(
            convert_field_layout(true_snapshot, bundle.width, bundle.height),
            convert_field_layout(pred_snapshot, bundle.width, bundle.height),
            output_dir / f"truncated_reconstruction_step_{step}_portrait_full.png",
            f"Truncated SVD reconstruction portrait | step {step}",
            error_percentile=99.0,
        )
        save_comparison_panel(
            convert_field_layout(true_snapshot, bundle.width, bundle.height),
            convert_field_layout(pred_snapshot, bundle.width, bundle.height),
            output_dir / f"truncated_reconstruction_step_{step}_portrait_wake.png",
            f"Truncated SVD reconstruction wake | step {step}",
            crop=zoom_crop,
            error_percentile=99.0,
        )

    linear_operator, predicted_weights = _fit_linear_dynamics(weights)
    linear_eigs = np.linalg.eigvals(linear_operator)
    save_complex_plane(linear_eigs, output_dir / "linear_dynamics_eigs.png", "Linear dynamics eigenvalues")
    linear_rollout = u_r @ predicted_weights
    linear_rollout_metrics: dict[str, dict[str, float]] = {}
    for step in compare_steps:
        true_snapshot = bundle.frames[step]
        pred_snapshot = linear_rollout[:, step].reshape(bundle.height, bundle.width)
        linear_rollout_metrics[f"step_{step}"] = {
            "rmse": rmse(true_snapshot, pred_snapshot),
            "nrmse": nrmse(true_snapshot, pred_snapshot),
            "relative_l2": relative_l2(true_snapshot, pred_snapshot),
        }
        save_comparison_panel(
            true_snapshot,
            pred_snapshot,
            output_dir / f"linear_rollout_step_{step}.png",
            f"Linear latent rollout | step {step}",
        )
        save_comparison_panel(
            convert_field_layout(true_snapshot, bundle.width, bundle.height),
            convert_field_layout(pred_snapshot, bundle.width, bundle.height),
            output_dir / f"linear_rollout_step_{step}_portrait_full.png",
            f"Linear latent rollout portrait | step {step}",
            error_percentile=99.0,
        )
        save_comparison_panel(
            convert_field_layout(true_snapshot, bundle.width, bundle.height),
            convert_field_layout(pred_snapshot, bundle.width, bundle.height),
            output_dir / f"linear_rollout_step_{step}_portrait_wake.png",
            f"Linear latent rollout wake | step {step}",
            crop=zoom_crop,
            error_percentile=99.0,
        )

    dmd_dir = output_dir / "dmd"
    dmd_dir.mkdir(parents=True, exist_ok=True)
    dmd_summary: list[dict[str, float | int]] = []
    sweep_scores: dict[int, float] = {}
    best_rank = None
    best_score = float("inf")
    best_prediction = None
    for rank in config.dmd_ranks:
        rank = min(rank, matrix.shape[1] - 1)
        if rank < 2:
            continue
        eigenvalues, predicted = _exact_dmd(matrix, rank)
        save_complex_plane(eigenvalues, dmd_dir / f"eigs_rank_{rank}.png", f"DMD eigenvalues | rank {rank}")
        score_values = [nrmse(bundle.frames[step], predicted[:, step].reshape(bundle.height, bundle.width)) for step in compare_steps]
        mean_score = float(np.mean(score_values))
        sweep_scores[rank] = mean_score
        dmd_summary.append(
            {
                "rank": rank,
                "mean_nrmse": mean_score,
                "spurious_eigs_outside_unit_circle": int(np.sum(np.abs(eigenvalues) > 1.05)),
                "dominant_frequency_hz": float(np.max(np.abs(np.angle(eigenvalues))) / (2.0 * np.pi * bundle.dt)),
            }
        )
        if mean_score < best_score:
            best_score = mean_score
            best_rank = rank
            best_prediction = predicted

    best_metrics: dict[str, dict[str, float]] = {}
    if best_rank is not None and best_prediction is not None:
        save_rank_sweep(sweep_scores, dmd_dir / "rank_sweep.png", "Mean NRMSE", "DMD rank sweep")
        pd.DataFrame(dmd_summary).to_csv(dmd_dir / "rank_sweep.csv", index=False)
        for step in compare_steps:
            true_snapshot = bundle.frames[step]
            pred_snapshot = best_prediction[:, step].reshape(bundle.height, bundle.width)
            best_metrics[f"step_{step}"] = {
                "rmse": rmse(true_snapshot, pred_snapshot),
                "nrmse": nrmse(true_snapshot, pred_snapshot),
                "relative_l2": relative_l2(true_snapshot, pred_snapshot),
            }
            save_comparison_panel(
                true_snapshot,
                pred_snapshot,
                dmd_dir / f"best_rank_step_{step}.png",
                f"DMD best rank={best_rank} | step {step}",
            )
            save_comparison_panel(
                convert_field_layout(true_snapshot, bundle.width, bundle.height),
                convert_field_layout(pred_snapshot, bundle.width, bundle.height),
                dmd_dir / f"best_rank_step_{step}_portrait_full.png",
                f"DMD best rank={best_rank} portrait | step {step}",
                error_percentile=99.0,
            )
            save_comparison_panel(
                convert_field_layout(true_snapshot, bundle.width, bundle.height),
                convert_field_layout(pred_snapshot, bundle.width, bundle.height),
                dmd_dir / f"best_rank_step_{step}_portrait_wake.png",
                f"DMD best rank={best_rank} wake | step {step}",
                crop=zoom_crop,
                error_percentile=99.0,
            )

    metrics = {
        "created_at": utc_timestamp(),
        "field_name": bundle.field_name,
        "summary": bundle.summary(),
        "singular_values": s[: min(20, len(s))].tolist(),
        "truncated_rank": config.truncated_rank,
        "truncated_reconstruction": truncated_metrics,
        "linear_dynamics": {
            "eigenvalues_real": np.real(linear_eigs).tolist(),
            "eigenvalues_imag": np.imag(linear_eigs).tolist(),
            "rollout_metrics": linear_rollout_metrics,
        },
        "dmd": {
            "best_rank": best_rank,
            "best_metrics": best_metrics,
            "rank_sweep": dmd_summary,
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
