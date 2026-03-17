from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass

import numpy as np

from .config import LinearConfig, ProjectPaths
from .data import TemporalHoldoutSplit, build_assignment_temporal_holdout, convert_field_layout, load_field_bundle
from .metrics import mse, nrmse, primary_score, rmse
from .plots import save_comparison_panel, save_complex_plane, save_field_image, save_rank_sweep, save_singular_spectrum
from .utils import utc_timestamp, write_json


@dataclass(frozen=True)
class SnapshotMetricSummary:
    mse: float
    rmse: float
    nrmse: float


def _compare_steps(steps: tuple[int, ...], num_snapshots: int) -> tuple[int, ...]:
    return tuple(step for step in steps if 0 <= step < num_snapshots)


def _compare_step_regions(compare_steps: tuple[int, ...], split: TemporalHoldoutSplit) -> dict[str, str]:
    regions: dict[str, str] = {}
    train_set = set(split.snapshot_train_idx.tolist())
    val_set = set(split.snapshot_val_idx.tolist())
    test_set = set(split.snapshot_test_idx.tolist())
    for step in compare_steps:
        if step in train_set:
            regions[f"step_{step}"] = "train_seen_region"
        elif step in val_set:
            regions[f"step_{step}"] = "validation_holdout_region"
        elif step in test_set:
            regions[f"step_{step}"] = "test_holdout_region"
        else:
            regions[f"step_{step}"] = "outside_protocol"
    return regions


def _summarize_snapshot_pairs(truth_pred_pairs: list[tuple[np.ndarray, np.ndarray]]) -> SnapshotMetricSummary:
    if not truth_pred_pairs:
        return SnapshotMetricSummary(mse=0.0, rmse=0.0, nrmse=0.0)
    return SnapshotMetricSummary(
        mse=float(np.mean([mse(true, pred) for true, pred in truth_pred_pairs])),
        rmse=float(np.mean([rmse(true, pred) for true, pred in truth_pred_pairs])),
        nrmse=float(np.mean([nrmse(true, pred) for true, pred in truth_pred_pairs])),
    )


def _compute_step_metrics(
    truth_frames: np.ndarray,
    pred_frames: np.ndarray,
    compare_steps: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for step in compare_steps:
        true_snapshot = truth_frames[step]
        pred_snapshot = pred_frames[step]
        metrics[f"step_{step}"] = {
            "mse": mse(true_snapshot, pred_snapshot),
            "rmse": rmse(true_snapshot, pred_snapshot),
            "nrmse": nrmse(true_snapshot, pred_snapshot),
        }
    return metrics


def _project_onto_basis(basis: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return basis @ (basis.T @ matrix)


def _fit_exact_dmd(x1: np.ndarray, x2: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    u, s, vh = np.linalg.svd(x1, full_matrices=False)
    effective_rank = min(rank, np.linalg.matrix_rank(x1), len(s))
    if effective_rank < 1:
        raise ValueError("DMD requires rank >= 1 after truncation.")
    u_r = u[:, :effective_rank]
    s_r = s[:effective_rank]
    v_r = vh[:effective_rank, :].T
    inv_sigma = np.diag(1.0 / s_r)
    a_tilde = u_r.T @ x2 @ v_r @ inv_sigma
    eigenvalues, eigenvectors = np.linalg.eig(a_tilde)
    phi = x2 @ v_r @ inv_sigma @ eigenvectors
    return eigenvalues, phi


def _rollout_exact_dmd(phi: np.ndarray, eigenvalues: np.ndarray, initial_snapshot: np.ndarray, num_steps: int) -> np.ndarray:
    amplitudes = np.linalg.lstsq(phi, initial_snapshot.astype(np.complex128), rcond=None)[0]
    predictions = np.empty((initial_snapshot.shape[0], num_steps), dtype=np.float64)
    for step in range(num_steps):
        step_dynamics = np.power(eigenvalues, step) * amplitudes
        predictions[:, step] = np.real(phi @ step_dynamics)
    return predictions.astype(np.float32)


def _write_rank_sweep_csv(rows: list[dict[str, float | int]], path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_linear_pipeline(
    config: LinearConfig | None = None,
    paths: ProjectPaths | None = None,
    output_tag: str = "baseline",
) -> dict[str, object]:
    started = time.perf_counter()
    config = config or LinearConfig()
    paths = paths or ProjectPaths()
    bundle = load_field_bundle(config.field_name, paths, layout=config.layout)
    split = build_assignment_temporal_holdout(bundle.num_snapshots)
    compare_steps = _compare_steps(config.compare_steps, bundle.num_snapshots)
    compare_regions = _compare_step_regions(compare_steps, split)

    output_dir = paths.linear_dir / output_tag
    dmd_dir = output_dir / "dmd"
    paths.ensure_directories([output_dir, dmd_dir])

    train_snapshot_matrix = bundle.matrix[:, split.snapshot_train_idx]
    train_transition_matrix = bundle.matrix[:, split.transition_train_idx]
    train_future_matrix = bundle.matrix[:, split.transition_train_idx + 1]
    val_matrix = bundle.matrix[:, split.snapshot_val_idx]
    test_matrix = bundle.matrix[:, split.snapshot_test_idx]

    u_train, s_train, _ = np.linalg.svd(train_snapshot_matrix, full_matrices=False)
    save_singular_spectrum(s_train, output_dir / "train_singular_spectrum.png", "Train singular value spectrum")
    for mode_idx in range(min(3, u_train.shape[1])):
        mode_field = u_train[:, mode_idx].reshape(bundle.height, bundle.width)
        save_field_image(mode_field, output_dir / f"train_mode_{mode_idx + 1}.png", f"Train POD mode {mode_idx + 1}")

    rank_rows: list[dict[str, float | int]] = []
    rank_sweep_scores: dict[int, float] = {}
    best_rank = None
    best_val_score = float("inf")
    best_rollout = None
    best_projection = None
    best_eigenvalues = None

    for candidate_rank in config.dmd_ranks:
        if candidate_rank < 1:
            continue
        eigenvalues, phi = _fit_exact_dmd(train_transition_matrix, train_future_matrix, candidate_rank)
        rollout_matrix = _rollout_exact_dmd(phi, eigenvalues, bundle.matrix[:, 0], bundle.num_snapshots)
        rollout_frames = rollout_matrix.T.reshape(bundle.num_snapshots, bundle.height, bundle.width)

        projection_basis = u_train[:, : min(int(candidate_rank), u_train.shape[1])]
        projected_matrix = _project_onto_basis(projection_basis, bundle.matrix)
        projected_frames = projected_matrix.T.reshape(bundle.num_snapshots, bundle.height, bundle.width)

        val_recon_pairs = [
            (bundle.frames[int(step)], projected_frames[int(step)])
            for step in split.snapshot_val_idx
        ]
        val_rollout_pairs = [
            (bundle.frames[int(step)], rollout_frames[int(step)])
            for step in split.snapshot_val_idx
        ]
        val_recon_summary = _summarize_snapshot_pairs(val_recon_pairs)
        val_rollout_summary = _summarize_snapshot_pairs(val_rollout_pairs)
        rank_sweep_scores[int(candidate_rank)] = val_rollout_summary.nrmse
        rank_rows.append(
            {
                "rank": int(candidate_rank),
                "val_recon_nrmse": val_recon_summary.nrmse,
                "val_rollout_nrmse": val_rollout_summary.nrmse,
                "spurious_eigs_outside_unit_circle": int(np.sum(np.abs(eigenvalues) > 1.05)),
            }
        )

        if val_rollout_summary.nrmse < best_val_score:
            best_rank = int(candidate_rank)
            best_val_score = val_rollout_summary.nrmse
            best_rollout = rollout_frames
            best_projection = projected_frames
            best_eigenvalues = eigenvalues

    if best_rank is None or best_rollout is None or best_projection is None or best_eigenvalues is None:
        raise RuntimeError("No valid DMD rank was evaluated.")

    save_rank_sweep(rank_sweep_scores, dmd_dir / "rank_sweep.png", "Validation rollout NRMSE", "DMD rank sweep")
    _write_rank_sweep_csv(rank_rows, dmd_dir / "rank_sweep.csv")
    save_complex_plane(best_eigenvalues, dmd_dir / "best_rank_eigs.png", f"DMD eigenvalues | rank {best_rank}")

    test_recon_pairs = [
        (bundle.frames[int(step)], best_projection[int(step)])
        for step in split.snapshot_test_idx
    ]
    val_rollout_pairs = [
        (bundle.frames[int(step)], best_rollout[int(step)])
        for step in split.snapshot_val_idx
    ]
    test_rollout_pairs = [
        (bundle.frames[int(step)], best_rollout[int(step)])
        for step in split.snapshot_test_idx
    ]
    recon_summary = _summarize_snapshot_pairs(test_recon_pairs)
    val_rollout_summary = _summarize_snapshot_pairs(val_rollout_pairs)
    test_rollout_summary = _summarize_snapshot_pairs(test_rollout_pairs)
    compare_metrics = _compute_step_metrics(bundle.frames, best_rollout, compare_steps)

    for step in compare_steps:
        true_snapshot = bundle.frames[step]
        pred_snapshot = best_rollout[step]
        save_comparison_panel(
            true_snapshot,
            pred_snapshot,
            dmd_dir / f"best_rank_step_{step}.png",
            f"DMD rollout | rank {best_rank} | step {step}",
            error_percentile=config.error_percentile,
        )
        save_comparison_panel(
            convert_field_layout(true_snapshot, bundle.height, bundle.width),
            convert_field_layout(pred_snapshot, bundle.height, bundle.width),
            dmd_dir / f"best_rank_step_{step}_wake.png",
            f"DMD rollout wake | rank {best_rank} | step {step}",
            crop=config.zoom_crop,
            error_percentile=config.error_percentile,
        )

    for preview_index, step in enumerate(split.snapshot_test_idx[: min(4, len(split.snapshot_test_idx))], start=1):
        true_snapshot = bundle.frames[int(step)]
        pred_snapshot = best_projection[int(step)]
        save_comparison_panel(
            true_snapshot,
            pred_snapshot,
            output_dir / f"test_reconstruction_preview_{preview_index}.png",
            f"Train-subspace reconstruction | step {int(step)}",
            error_percentile=config.error_percentile,
        )

    step_100 = compare_metrics.get("step_100", {"mse": 0.0, "rmse": 0.0, "nrmse": 0.0})
    step_150 = compare_metrics.get("step_150", {"mse": 0.0, "rmse": 0.0, "nrmse": 0.0})
    overall_score = primary_score(recon_summary.nrmse, step_100["nrmse"], step_150["nrmse"])
    wall_seconds = time.perf_counter() - started

    metrics = {
        "created_at": utc_timestamp(),
        "evaluation_protocol": split.protocol,
        "field_name": bundle.field_name,
        "layout": config.layout,
        "summary": bundle.summary(),
        "config": asdict(config),
        "snapshot_split": split.snapshot_indices(),
        "transition_split": split.transition_indices(),
        "compare_step_regions": compare_regions,
        "selection_metric": "mean_validation_rollout_nrmse",
        "best_rank": best_rank,
        "best_rank_validation_rollout_nrmse": best_val_score,
        "rank_sweep": rank_rows,
        "recon_mse": recon_summary.mse,
        "recon_rmse": recon_summary.rmse,
        "recon_nrmse": recon_summary.nrmse,
        "validation_rollout_mse": val_rollout_summary.mse,
        "validation_rollout_rmse": val_rollout_summary.rmse,
        "validation_rollout_nrmse": val_rollout_summary.nrmse,
        "test_rollout_mse": test_rollout_summary.mse,
        "test_rollout_rmse": test_rollout_summary.rmse,
        "test_rollout_nrmse": test_rollout_summary.nrmse,
        "mse_t100": step_100["mse"],
        "rmse_t100": step_100["rmse"],
        "nrmse_t100": step_100["nrmse"],
        "mse_t150": step_150["mse"],
        "rmse_t150": step_150["rmse"],
        "nrmse_t150": step_150["nrmse"],
        "primary_score": overall_score,
        "peak_memory_gb": 0.0,
        "wall_seconds": wall_seconds,
        "artifact_paths": {
            "singular_spectrum": "train_singular_spectrum.png",
            "rank_sweep_plot": "dmd/rank_sweep.png",
            "rank_sweep_csv": "dmd/rank_sweep.csv",
            "best_rank_eigs": "dmd/best_rank_eigs.png",
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
