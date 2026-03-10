from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import NonlinearConfig, ProjectPaths
from .data import FieldBundle, load_field_bundle, random_snapshot_split, sequential_transition_split
from .metrics import mse, nrmse, primary_score, rmse
from .models import build_autoencoder, build_dynamics_model
from .plots import (
    save_comparison_panel,
    save_field_image,
    save_latent_time_series,
    save_latent_trajectory_pca,
    save_training_curves,
)
from .utils import set_seed, utc_timestamp, write_json


def detect_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SnapshotDataset(Dataset[torch.Tensor]):
    def __init__(self, frames: np.ndarray, indices: np.ndarray, mean: float, std: float) -> None:
        normalized = (frames[indices] - mean) / std
        self.tensor = torch.from_numpy(normalized[:, None, :, :].astype(np.float32))

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]


@dataclass
class NormalizationStats:
    mean: float
    std: float


@dataclass
class LatentNormalizationStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class SnapshotMetricSummary:
    mse: float
    rmse: float
    nrmse: float


def _denormalize(x: torch.Tensor | np.ndarray, stats: NormalizationStats) -> torch.Tensor | np.ndarray:
    return (x * stats.std) + stats.mean


def _standardize_latents(latents: np.ndarray, stats: LatentNormalizationStats) -> np.ndarray:
    return (latents - stats.mean) / stats.std


def _restore_latents(latents: np.ndarray, stats: LatentNormalizationStats) -> np.ndarray:
    return (latents * stats.std) + stats.mean


def _regularization_l2(module: nn.Module) -> torch.Tensor:
    penalties = [parameter.pow(2).mean() for parameter in module.parameters() if parameter.requires_grad]
    if not penalties:
        return torch.tensor(0.0)
    return torch.stack(penalties).mean()


def _gradient_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_dx = target[..., :, 1:] - target[..., :, :-1]
    true_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, true_dx) + F.l1_loss(pred_dy, true_dy)


def _build_loader(
    dataset: Dataset[torch.Tensor],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator,
    )


def _build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    factor: float,
    patience: int,
    min_lr: float,
) -> ReduceLROnPlateau | None:
    if name == "none":
        return None
    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    raise ValueError(f"Unsupported scheduler type: {name}")


def _summarize_snapshot_pairs(truth_pred_pairs: list[tuple[np.ndarray, np.ndarray]]) -> SnapshotMetricSummary:
    return SnapshotMetricSummary(
        mse=float(np.mean([mse(true, pred) for true, pred in truth_pred_pairs])),
        rmse=float(np.mean([rmse(true, pred) for true, pred in truth_pred_pairs])),
        nrmse=float(np.mean([nrmse(true, pred) for true, pred in truth_pred_pairs])),
    )


def _rollout_start_indices(indices: np.ndarray, stride: int, num_snapshots: int) -> list[int]:
    starts = [int(index) for index in indices[:: max(1, stride)] if int(index) < num_snapshots - 1]
    if starts:
        return starts
    return [int(indices[0])]


def _latent_rollout_loss(
    model: nn.Module,
    latent_tensor: torch.Tensor,
    start_index: int,
    horizon: int,
) -> torch.Tensor:
    max_horizon = int(latent_tensor.shape[0] - start_index - 1)
    effective_horizon = min(horizon, max_horizon)
    if effective_horizon <= 0:
        return latent_tensor.new_tensor(0.0)
    rollout_state = latent_tensor[start_index : start_index + 1]
    rollout_preds = []
    for _ in range(effective_horizon):
        rollout_state = model(rollout_state)
        rollout_preds.append(rollout_state)
    rollout_pred = torch.cat(rollout_preds, dim=0)
    rollout_target = latent_tensor[start_index + 1 : start_index + 1 + effective_horizon]
    return F.mse_loss(rollout_pred, rollout_target)


def _multi_start_rollout_loss(
    model: nn.Module,
    latent_tensor: torch.Tensor,
    start_indices: list[int],
    horizon: int,
) -> torch.Tensor:
    losses = [_latent_rollout_loss(model, latent_tensor, start_index, horizon) for start_index in start_indices]
    if not losses:
        return latent_tensor.new_tensor(0.0)
    return torch.stack(losses).mean()


def _dynamics_validation_terms(
    model: nn.Module,
    latent_tensor: torch.Tensor,
    val_idx: np.ndarray,
    config: NonlinearConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    val_one_step = F.mse_loss(model(latent_tensor[val_idx]), latent_tensor[val_idx + 1])
    val_starts = _rollout_start_indices(val_idx, config.validation_rollout_stride, int(latent_tensor.shape[0]))
    val_rollout = _multi_start_rollout_loss(
        model,
        latent_tensor,
        start_indices=val_starts,
        horizon=min(config.validation_rollout_horizon, len(val_idx)),
    )
    selection_loss = val_one_step + (config.validation_rollout_weight * val_rollout)
    return val_one_step, val_rollout, selection_loss


def _evaluate_autoencoder(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    stats: NormalizationStats,
    gradient_loss_weight: float,
) -> tuple[float, SnapshotMetricSummary, list[tuple[np.ndarray, np.ndarray]]]:
    model.eval()
    losses = []
    truth_pred_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch) + (gradient_loss_weight * _gradient_l1(recon, batch))
            losses.append(loss.item())
            truth = _denormalize(batch, stats).cpu().numpy()
            pred = _denormalize(recon, stats).cpu().numpy()
            for true_item, pred_item in zip(truth, pred, strict=True):
                truth_pred_pairs.append((true_item[0], pred_item[0]))
    return float(np.mean(losses)), _summarize_snapshot_pairs(truth_pred_pairs), truth_pred_pairs


def _encode_all_frames(
    model: nn.Module,
    bundle: FieldBundle,
    stats: NormalizationStats,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(bundle.num_snapshots)
    dataset = SnapshotDataset(bundle.frames, indices, stats.mean, stats.std)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    latents = []
    reconstructions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, latent = model(batch)
            latents.append(latent.cpu().numpy())
            reconstructions.append(_denormalize(recon, stats).cpu().numpy()[:, 0])
    return np.concatenate(latents, axis=0), np.concatenate(reconstructions, axis=0).astype(np.float32)


def _fit_linear_operator(latents: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    current = latents[train_idx].T
    future = latents[train_idx + 1].T
    return future @ np.linalg.pinv(current)


def _rollout_operator(operator: np.ndarray, initial_state: np.ndarray, num_steps: int) -> np.ndarray:
    rollout = [initial_state]
    state = initial_state
    for _ in range(1, num_steps):
        state = operator @ state
        rollout.append(state)
    return np.stack(rollout).astype(np.float32)


def _compare_steps(steps: tuple[int, ...], num_snapshots: int) -> tuple[int, ...]:
    return tuple(step for step in steps if 0 <= step < num_snapshots)


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


def _save_truth_reference(bundle: FieldBundle, compare_steps: tuple[int, ...], output_dir: Path, config: NonlinearConfig) -> None:
    if not compare_steps:
        return
    step = compare_steps[0]
    truth = bundle.frames[step]
    save_field_image(truth, output_dir / "truth_portrait.png", f"Ground truth portrait | step {step}")
    row_start, row_end, col_start, col_end = config.zoom_crop
    save_field_image(
        truth[row_start:row_end, col_start:col_end],
        output_dir / "truth_wake_zoom.png",
        f"Ground truth wake zoom | step {step}",
    )


def _save_step_artifacts(
    prefix: str,
    truth_frames: np.ndarray,
    pred_frames: np.ndarray,
    compare_steps: tuple[int, ...],
    output_dir: Path,
    config: NonlinearConfig,
) -> dict[str, list[str]]:
    full_paths: list[str] = []
    zoom_paths: list[str] = []
    for step in compare_steps:
        true_snapshot = truth_frames[step]
        pred_snapshot = pred_frames[step]
        full_name = f"{prefix}_step_{step}_full.png"
        zoom_name = f"{prefix}_step_{step}_wake.png"
        save_comparison_panel(
            true_snapshot,
            pred_snapshot,
            output_dir / full_name,
            f"{prefix.replace('_', ' ').title()} | step {step}",
            interpolation="bilinear",
            error_percentile=config.error_percentile,
        )
        save_comparison_panel(
            true_snapshot,
            pred_snapshot,
            output_dir / zoom_name,
            f"{prefix.replace('_', ' ').title()} wake zoom | step {step}",
            crop=config.zoom_crop,
            interpolation="bilinear",
            error_percentile=config.error_percentile,
        )
        full_paths.append(full_name)
        zoom_paths.append(zoom_name)
    return {"full": full_paths, "wake": zoom_paths}


def _train_autoencoder(
    bundle: FieldBundle,
    config: NonlinearConfig,
    output_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, NormalizationStats, dict[str, object], np.ndarray, np.ndarray]:
    train_idx, test_idx = random_snapshot_split(bundle.num_snapshots, config.ae_train_ratio, config.seed)
    train_frames = bundle.frames[train_idx]
    stats = NormalizationStats(mean=float(train_frames.mean()), std=float(train_frames.std() + 1e-6))
    train_dataset = SnapshotDataset(bundle.frames, train_idx, stats.mean, stats.std)
    test_dataset = SnapshotDataset(bundle.frames, test_idx, stats.mean, stats.std)
    train_loader = _build_loader(train_dataset, config.batch_size, True, config.seed)
    test_loader = _build_loader(test_dataset, config.batch_size, False, config.seed + 1)

    model = build_autoencoder((bundle.height, bundle.width), config.latent_dim, config.ae_architecture).to(device)
    optimizer = AdamW(model.parameters(), lr=config.ae_learning_rate, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(
        config.ae_scheduler,
        optimizer,
        config.ae_scheduler_factor,
        config.ae_scheduler_patience,
        config.ae_min_learning_rate,
    )
    history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in tqdm(range(config.ae_epochs), desc="AE", leave=False):
        model.train()
        batch_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            recon, latent = model(batch)
            loss = (
                F.mse_loss(recon, batch)
                + (config.gradient_loss_weight * _gradient_l1(recon, batch))
                + (config.latent_l1_weight * latent.abs().mean())
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        val_loss, _, _ = _evaluate_autoencoder(model, test_loader, device, stats, config.gradient_loss_weight)
        train_loss = float(np.mean(batch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if scheduler is not None:
            scheduler.step(val_loss)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    ae_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)

    _, reconstruction_summary, preview_pairs = _evaluate_autoencoder(model, test_loader, device, stats, config.gradient_loss_weight)
    save_training_curves(
        {key: value for key, value in history.items() if key != "learning_rate"},
        output_dir / "ae_training_curves.png",
        "Autoencoder training",
    )
    preview_artifacts = {"full": [], "wake": []}
    for preview_idx, (true_snapshot, pred_snapshot) in enumerate(preview_pairs[: config.num_preview_images], start=1):
        full_name = f"ae_reconstruction_preview_{preview_idx}_full.png"
        wake_name = f"ae_reconstruction_preview_{preview_idx}_wake.png"
        save_comparison_panel(
            true_snapshot,
            pred_snapshot,
            output_dir / full_name,
            f"AE reconstruction preview {preview_idx}",
            interpolation="bilinear",
            error_percentile=config.error_percentile,
        )
        save_comparison_panel(
            true_snapshot,
            pred_snapshot,
            output_dir / wake_name,
            f"AE reconstruction preview {preview_idx} wake zoom",
            crop=config.zoom_crop,
            interpolation="bilinear",
            error_percentile=config.error_percentile,
        )
        preview_artifacts["full"].append(full_name)
        preview_artifacts["wake"].append(wake_name)

    latents, reconstructed_frames = _encode_all_frames(model, bundle, stats, config.batch_size, device)
    peak_memory_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
    metrics = {
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "recon_mse": reconstruction_summary.mse,
        "recon_rmse": reconstruction_summary.rmse,
        "recon_nrmse": reconstruction_summary.nrmse,
        "history": history,
        "ae_seconds": ae_seconds,
        "peak_memory_gb_after_ae": peak_memory_gb,
        "artifact_paths": {
            "reconstruction_previews_full": preview_artifacts["full"],
            "reconstruction_previews_wake": preview_artifacts["wake"],
            "training_curves": "ae_training_curves.png",
        },
    }
    return model, stats, metrics, latents, reconstructed_frames


def _train_dynamics(
    latents: np.ndarray,
    bundle: FieldBundle,
    autoencoder: nn.Module,
    stats: NormalizationStats,
    config: NonlinearConfig,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    train_idx, val_idx = sequential_transition_split(bundle.num_snapshots, config.dyn_train_ratio)
    latent_stats = LatentNormalizationStats(
        mean=latents[train_idx].mean(axis=0).astype(np.float32),
        std=(latents[train_idx].std(axis=0) + 1e-6).astype(np.float32),
    )
    latents_norm = _standardize_latents(latents, latent_stats)
    linear_operator = _fit_linear_operator(latents_norm, train_idx)
    linear_init = torch.from_numpy(linear_operator.astype(np.float32))
    latent_tensor = torch.from_numpy(latents_norm.astype(np.float32)).to(device)
    model = build_dynamics_model(
        config.dynamics_model,
        config.latent_dim,
        config.dynamics_hidden_dim,
        config.dynamics_depth,
        linear_init=linear_init,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.dyn_learning_rate, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(
        config.dyn_scheduler,
        optimizer,
        config.dyn_scheduler_factor,
        config.dyn_scheduler_patience,
        config.dyn_min_learning_rate,
    )
    history = {"train_loss": [], "val_loss": [], "val_one_step_loss": [], "val_rollout_loss": [], "learning_rate": []}
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 0
    train_starts = _rollout_start_indices(train_idx, config.train_rollout_stride, bundle.num_snapshots)

    start = time.perf_counter()
    for _ in tqdm(range(config.dyn_epochs), desc="Dyn", leave=False):
        model.train()
        prediction = model(latent_tensor[train_idx])
        target = latent_tensor[train_idx + 1]
        one_step_loss = F.mse_loss(prediction, target)
        rollout_loss = _multi_start_rollout_loss(
            model,
            latent_tensor,
            start_indices=train_starts,
            horizon=min(config.train_rollout_horizon, len(train_idx)),
        )
        loss = one_step_loss + (config.rollout_loss_weight * rollout_loss) + (config.dyn_l2_weight * _regularization_l2(model))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_one_step, val_rollout, val_loss = _dynamics_validation_terms(model, latent_tensor, val_idx, config)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_one_step_loss"].append(val_one_step.item())
        history["val_rollout_loss"].append(val_rollout.item())
        if scheduler is not None:
            scheduler.step(val_loss.item())
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    dyn_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    save_training_curves(
        {key: value for key, value in history.items() if key != "learning_rate"},
        output_dir / "dynamics_training_curves.png",
        "Latent dynamics training",
    )

    model.eval()
    rollout_latents_norm = [latents_norm[0]]
    current = latent_tensor[0:1]
    with torch.no_grad():
        for _ in range(1, bundle.num_snapshots):
            current = model(current)
            rollout_latents_norm.append(current.squeeze(0).cpu().numpy())
    rollout_latents_norm_np = np.stack(rollout_latents_norm).astype(np.float32)
    rollout_latents = _restore_latents(rollout_latents_norm_np, latent_stats).astype(np.float32)
    linear_rollout_latents_norm = _rollout_operator(linear_operator, latents_norm[0], bundle.num_snapshots)
    linear_rollout_latents = _restore_latents(linear_rollout_latents_norm, latent_stats).astype(np.float32)

    save_latent_trajectory_pca(
        latents,
        rollout_latents,
        output_dir / "latent_pca_trajectory.png",
        "Latent dynamics trajectory (PCA projection)",
    )
    save_latent_time_series(
        latents,
        rollout_latents,
        output_dir / "latent_time_series.png",
        "Latent coordinates vs time",
    )

    autoencoder = autoencoder.to("cpu")
    model = model.to("cpu")
    if device.type == "cuda":
        del latent_tensor
        torch.cuda.empty_cache()

    with torch.no_grad():
        predicted_recon = autoencoder.decode(torch.from_numpy(rollout_latents.astype(np.float32))).numpy()[:, 0]
        linear_recon = autoencoder.decode(torch.from_numpy(linear_rollout_latents.astype(np.float32))).numpy()[:, 0]
    predicted_frames = _denormalize(predicted_recon, stats).astype(np.float32)
    linear_frames = _denormalize(linear_recon, stats).astype(np.float32)

    compare_steps = _compare_steps(config.compare_steps, bundle.num_snapshots)
    compare_metrics = _compute_step_metrics(bundle.frames, predicted_frames, compare_steps)
    linear_metrics = _compute_step_metrics(bundle.frames, linear_frames, compare_steps)
    rollout_artifacts = _save_step_artifacts("rollout", bundle.frames, predicted_frames, compare_steps, output_dir, config)
    linear_artifacts = _save_step_artifacts(
        "linear_latent",
        bundle.frames,
        linear_frames,
        compare_steps,
        output_dir,
        config,
    )

    peak_memory_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
    return {
        "train_transition_indices": train_idx.tolist(),
        "val_transition_indices": val_idx.tolist(),
        "history": history,
        "dyn_seconds": dyn_seconds,
        "rollout_metrics": compare_metrics,
        "linear_baseline_metrics": linear_metrics,
        "peak_memory_gb_after_dyn": peak_memory_gb,
        "artifact_paths": {
            "training_curves": "dynamics_training_curves.png",
            "latent_pca_trajectory": "latent_pca_trajectory.png",
            "latent_time_series": "latent_time_series.png",
            "rollout_panels_full": rollout_artifacts["full"],
            "rollout_panels_wake": rollout_artifacts["wake"],
            "linear_latent_panels_full": linear_artifacts["full"],
            "linear_latent_panels_wake": linear_artifacts["wake"],
        },
    }


def run_nonlinear_pipeline(
    config: NonlinearConfig | None = None,
    paths: ProjectPaths | None = None,
    output_tag: str = "baseline",
) -> dict[str, object]:
    config = config or NonlinearConfig()
    paths = paths or ProjectPaths()
    paths.ensure_directories()
    output_dir = paths.nonlinear_dir / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed, deterministic=config.deterministic)

    device = detect_device(config.device)
    bundle = load_field_bundle(config.field_name, paths, layout=config.layout)
    compare_steps = _compare_steps(config.compare_steps, bundle.num_snapshots)
    _save_truth_reference(bundle, compare_steps, output_dir, config)

    autoencoder, stats, ae_metrics, latents, reconstructed_frames = _train_autoencoder(bundle, config, output_dir, device)
    ae_floor_metrics = _compute_step_metrics(bundle.frames, reconstructed_frames, compare_steps)
    ae_floor_artifacts = _save_step_artifacts("ae_floor", bundle.frames, reconstructed_frames, compare_steps, output_dir, config)
    dyn_metrics = _train_dynamics(latents, bundle, autoencoder, stats, config, output_dir, device)
    rollout_metrics = dyn_metrics["rollout_metrics"]
    linear_metrics = dyn_metrics["linear_baseline_metrics"]
    step_100 = rollout_metrics.get("step_100", {"mse": 0.0, "nrmse": 0.0, "rmse": 0.0})
    step_150 = rollout_metrics.get("step_150", {"mse": 0.0, "nrmse": 0.0, "rmse": 0.0})
    ae_floor_t100 = ae_floor_metrics.get("step_100", {"rmse": 0.0, "mse": 0.0, "nrmse": 0.0})
    ae_floor_t150 = ae_floor_metrics.get("step_150", {"rmse": 0.0, "mse": 0.0, "nrmse": 0.0})
    linear_t100 = linear_metrics.get("step_100", {"rmse": 0.0, "mse": 0.0, "nrmse": 0.0})
    linear_t150 = linear_metrics.get("step_150", {"rmse": 0.0, "mse": 0.0, "nrmse": 0.0})
    overall_score = primary_score(ae_metrics["recon_nrmse"], step_100["nrmse"], step_150["nrmse"])

    metrics = {
        "created_at": utc_timestamp(),
        "field_name": bundle.field_name,
        "layout": config.layout,
        "summary": bundle.summary(),
        "config": asdict(config),
        "device": str(device),
        "ae": {**ae_metrics, "floor_metrics": ae_floor_metrics},
        "dynamics": dyn_metrics,
        "primary_score": overall_score,
        "recon_mse": ae_metrics["recon_mse"],
        "recon_rmse": ae_metrics["recon_rmse"],
        "recon_nrmse": ae_metrics["recon_nrmse"],
        "mse_t100": step_100["mse"],
        "rmse_t100": step_100["rmse"],
        "nrmse_t100": step_100["nrmse"],
        "mse_t150": step_150["mse"],
        "rmse_t150": step_150["rmse"],
        "nrmse_t150": step_150["nrmse"],
        "ae_floor_rmse_t100": ae_floor_t100["rmse"],
        "ae_floor_rmse_t150": ae_floor_t150["rmse"],
        "linear_latent_rmse_t100": linear_t100["rmse"],
        "linear_latent_rmse_t150": linear_t150["rmse"],
        "peak_memory_gb": max(ae_metrics["peak_memory_gb_after_ae"], dyn_metrics["peak_memory_gb_after_dyn"]),
        "wall_seconds": ae_metrics["ae_seconds"] + dyn_metrics["dyn_seconds"],
        "artifacts": {
            "truth_portrait": "truth_portrait.png",
            "truth_wake_zoom": "truth_wake_zoom.png",
            "ae_training_curves": "ae_training_curves.png",
            "reconstruction_previews_full": ae_metrics["artifact_paths"]["reconstruction_previews_full"],
            "reconstruction_previews_wake": ae_metrics["artifact_paths"]["reconstruction_previews_wake"],
            "ae_floor_full": ae_floor_artifacts["full"],
            "ae_floor_wake": ae_floor_artifacts["wake"],
            "dynamics_training_curves": "dynamics_training_curves.png",
            "latent_pca_trajectory": dyn_metrics["artifact_paths"]["latent_pca_trajectory"],
            "latent_time_series": dyn_metrics["artifact_paths"]["latent_time_series"],
            "rollout_panels_full": dyn_metrics["artifact_paths"]["rollout_panels_full"],
            "rollout_panels_wake": dyn_metrics["artifact_paths"]["rollout_panels_wake"],
            "linear_latent_panels_full": dyn_metrics["artifact_paths"]["linear_latent_panels_full"],
            "linear_latent_panels_wake": dyn_metrics["artifact_paths"]["linear_latent_panels_wake"],
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    torch.save(autoencoder.state_dict(), output_dir / "autoencoder.pt")
    return metrics
