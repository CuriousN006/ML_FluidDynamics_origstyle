from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import NonlinearConfig, ProjectPaths
from .data import FieldBundle, TemporalHoldoutSplit, build_assignment_temporal_holdout, load_field_bundle
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


def _amp_enabled(use_amp: bool, device: torch.device) -> bool:
    return bool(use_amp and device.type == "cuda")


def _autocast_context(use_amp: bool, device: torch.device):
    if _amp_enabled(use_amp, device):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def _make_grad_scaler(use_amp: bool, device: torch.device):
    if not _amp_enabled(use_amp, device):
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type)
        except TypeError:
            return torch.amp.GradScaler()
    return torch.cuda.amp.GradScaler()


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


@dataclass(frozen=True)
class AutoencoderLossWeights:
    gradient_loss_weight: float
    fft_loss_weight: float
    latent_l1_weight: float
    coarse_loss_weight: float


@dataclass(frozen=True)
class SchedulerBundle:
    name: str
    scheduler: ReduceLROnPlateau | CosineAnnealingWarmRestarts | None


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


def _fft_magnitude_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfftn(prediction, dim=(-2, -1))
    target_fft = torch.fft.rfftn(target, dim=(-2, -1))
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - ((kernel_size - 1) / 2.0)
    gaussian = torch.exp(-(coords.pow(2)) / max(2.0 * sigma * sigma, 1e-8))
    gaussian = gaussian / gaussian.sum()
    kernel_2d = gaussian[:, None] * gaussian[None, :]
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def _gaussian_blur(target: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1 or sigma <= 0.0:
        return target
    if kernel_size % 2 == 0:
        raise ValueError("Gaussian blur kernel size must be odd.")
    weight = _gaussian_kernel(kernel_size, sigma, target.device, target.dtype)
    groups = target.shape[1]
    weight = weight.expand(groups, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    return F.conv2d(target, weight, padding=padding, groups=groups)


def _decode_autoencoder(model: nn.Module, latent: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if hasattr(model, "decode_with_aux"):
        return model.decode_with_aux(latent)
    reconstruction = model.decode(latent)  # type: ignore[attr-defined]
    return reconstruction, {"coarse": reconstruction}


def _autoencoder_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latent: torch.Tensor | None,
    config: NonlinearConfig,
    auxiliary_outputs: dict[str, torch.Tensor] | None = None,
    loss_weights: AutoencoderLossWeights | None = None,
) -> torch.Tensor:
    weights = loss_weights or AutoencoderLossWeights(
        gradient_loss_weight=config.gradient_loss_weight,
        fft_loss_weight=config.fft_loss_weight,
        latent_l1_weight=config.latent_l1_weight,
        coarse_loss_weight=config.coarse_loss_weight,
    )
    loss = F.mse_loss(prediction, target)
    if weights.gradient_loss_weight > 0.0:
        loss = loss + (weights.gradient_loss_weight * _gradient_l1(prediction, target))
    if weights.fft_loss_weight > 0.0:
        loss = loss + (weights.fft_loss_weight * _fft_magnitude_l1(prediction, target))
    if latent is not None and weights.latent_l1_weight > 0.0:
        loss = loss + (weights.latent_l1_weight * latent.abs().mean())
    if auxiliary_outputs and weights.coarse_loss_weight > 0.0 and "coarse" in auxiliary_outputs:
        coarse_target = _gaussian_blur(target, config.coarse_blur_kernel, config.coarse_blur_sigma)
        loss = loss + (weights.coarse_loss_weight * F.mse_loss(auxiliary_outputs["coarse"], coarse_target))
    return loss


def compute_ae_score(recon_rmse: float, floor_t100: float, floor_t150: float) -> float:
    return float((0.5 * recon_rmse) + (0.25 * floor_t100) + (0.25 * floor_t150))


def _maybe_save_autoencoder_checkpoint(model: nn.Module, output_dir: Path, *, save_checkpoint: bool) -> None:
    if not save_checkpoint:
        return
    torch.save(model.state_dict(), output_dir / "autoencoder.pt")


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
    *,
    cosine_t0_epochs: int | None = None,
    cosine_tmult: int | None = None,
    cosine_eta_min: float | None = None,
) -> SchedulerBundle:
    if name == "none":
        return SchedulerBundle(name="none", scheduler=None)
    if name == "plateau":
        return SchedulerBundle(
            name="plateau",
            scheduler=ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            ),
        )
    if name == "cosine_restarts":
        return SchedulerBundle(
            name="cosine_restarts",
            scheduler=CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, cosine_t0_epochs or 1),
                T_mult=max(1, cosine_tmult or 1),
                eta_min=min_lr if cosine_eta_min is None else cosine_eta_min,
            ),
        )
    raise ValueError(f"Unsupported scheduler type: {name}")


def _step_scheduler(bundle: SchedulerBundle, *, metric: float | None = None) -> None:
    if bundle.scheduler is None:
        return
    if bundle.name == "plateau":
        if metric is None:
            raise ValueError("Plateau scheduler requires a validation metric.")
        bundle.scheduler.step(metric)
        return
    bundle.scheduler.step()


def _budget_exhausted(start_time: float, budget_seconds: float | None) -> bool:
    return budget_seconds is not None and (time.perf_counter() - start_time) >= budget_seconds


def _summarize_snapshot_pairs(truth_pred_pairs: list[tuple[np.ndarray, np.ndarray]]) -> SnapshotMetricSummary:
    return SnapshotMetricSummary(
        mse=float(np.mean([mse(true, pred) for true, pred in truth_pred_pairs])),
        rmse=float(np.mean([rmse(true, pred) for true, pred in truth_pred_pairs])),
        nrmse=float(np.mean([nrmse(true, pred) for true, pred in truth_pred_pairs])),
    )


def _rollout_start_indices(indices: np.ndarray, stride: int) -> list[int]:
    if len(indices) == 0:
        return []
    starts = [int(index) for index in indices[:: max(1, stride)]]
    if starts:
        return starts
    return [int(indices[0])]


def _latent_rollout_loss(
    model: nn.Module,
    latent_tensor: torch.Tensor,
    start_index: int,
    horizon: int,
    *,
    max_target_exclusive: int | None = None,
) -> torch.Tensor:
    max_horizon = int(latent_tensor.shape[0] - start_index - 1)
    if max_target_exclusive is not None:
        max_horizon = min(max_horizon, max(0, max_target_exclusive - start_index - 1))
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
    *,
    max_target_exclusive: int | None = None,
) -> torch.Tensor:
    losses = [
        _latent_rollout_loss(
            model,
            latent_tensor,
            start_index,
            horizon,
            max_target_exclusive=max_target_exclusive,
        )
        for start_index in start_indices
    ]
    if not losses:
        return latent_tensor.new_tensor(0.0)
    return torch.stack(losses).mean()


def _dynamics_validation_terms(
    model: nn.Module,
    latent_tensor: torch.Tensor,
    val_idx: np.ndarray,
    config: NonlinearConfig,
    *,
    max_target_exclusive: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    val_one_step = F.mse_loss(model(latent_tensor[val_idx]), latent_tensor[val_idx + 1])
    val_starts = _rollout_start_indices(val_idx, config.validation_rollout_stride)
    val_rollout = _multi_start_rollout_loss(
        model,
        latent_tensor,
        start_indices=val_starts,
        horizon=min(config.validation_rollout_horizon, len(val_idx)),
        max_target_exclusive=max_target_exclusive,
    )
    selection_loss = val_one_step + (config.validation_rollout_weight * val_rollout)
    return val_one_step, val_rollout, selection_loss


def _evaluate_autoencoder(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    stats: NormalizationStats,
    config: NonlinearConfig,
    loss_weights: AutoencoderLossWeights | None = None,
) -> tuple[float, SnapshotMetricSummary, list[tuple[np.ndarray, np.ndarray]], SnapshotMetricSummary]:
    model.eval()
    losses = []
    truth_pred_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    coarse_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with _autocast_context(config.use_amp, device):
                latent = model.encode(batch)  # type: ignore[attr-defined]
                recon, auxiliary_outputs = _decode_autoencoder(model, latent)
                loss = _autoencoder_reconstruction_loss(
                    recon,
                    batch,
                    latent=None,
                    config=config,
                    auxiliary_outputs=auxiliary_outputs,
                    loss_weights=loss_weights,
                )
            losses.append(loss.item())
            truth = _denormalize(batch.float(), stats).cpu().numpy()
            pred = _denormalize(recon.float(), stats).cpu().numpy()
            coarse = _denormalize(auxiliary_outputs["coarse"].float(), stats).cpu().numpy()
            for true_item, pred_item in zip(truth, pred, strict=True):
                truth_pred_pairs.append((true_item[0], pred_item[0]))
            for true_item, coarse_item in zip(truth, coarse, strict=True):
                coarse_pairs.append((true_item[0], coarse_item[0]))
    return (
        float(np.mean(losses)),
        _summarize_snapshot_pairs(truth_pred_pairs),
        truth_pred_pairs,
        _summarize_snapshot_pairs(coarse_pairs),
    )


def _progress_fraction(
    *,
    elapsed_seconds: float,
    budget_seconds: float | None,
    epoch_index: int,
    total_epochs: int,
) -> float:
    if budget_seconds is not None and budget_seconds > 0.0:
        return float(min(max(elapsed_seconds / budget_seconds, 0.0), 1.0))
    return float(min(max(epoch_index / max(1, total_epochs), 0.0), 1.0))


def _ae_training_loss_weights(
    config: NonlinearConfig,
    *,
    elapsed_seconds: float,
    budget_seconds: float | None,
    epoch_index: int,
) -> tuple[AutoencoderLossWeights, float]:
    progress = _progress_fraction(
        elapsed_seconds=elapsed_seconds,
        budget_seconds=budget_seconds,
        epoch_index=epoch_index,
        total_epochs=config.ae_epochs,
    )
    if not config.ae_use_curriculum:
        return (
            AutoencoderLossWeights(
                gradient_loss_weight=config.gradient_loss_weight,
                fft_loss_weight=config.fft_loss_weight,
                latent_l1_weight=config.latent_l1_weight,
                coarse_loss_weight=config.coarse_loss_weight,
            ),
            progress,
        )
    curriculum_progress = min(progress / 0.8, 1.0)
    coarse_loss_weight = 0.40 + (curriculum_progress * (0.15 - 0.40))
    gradient_loss_weight = 0.05 + (curriculum_progress * (0.10 - 0.05))
    return (
        AutoencoderLossWeights(
            gradient_loss_weight=gradient_loss_weight,
            fft_loss_weight=0.0,
            latent_l1_weight=config.latent_l1_weight,
            coarse_loss_weight=coarse_loss_weight,
        ),
        progress,
    )


def _ae_validation_loss_weights(config: NonlinearConfig) -> AutoencoderLossWeights:
    if not config.ae_use_curriculum:
        return AutoencoderLossWeights(
            gradient_loss_weight=config.gradient_loss_weight,
            fft_loss_weight=config.fft_loss_weight,
            latent_l1_weight=config.latent_l1_weight,
            coarse_loss_weight=config.coarse_loss_weight,
        )
    return AutoencoderLossWeights(
        gradient_loss_weight=0.10,
        fft_loss_weight=0.0,
        latent_l1_weight=config.latent_l1_weight,
        coarse_loss_weight=0.15,
    )


def _should_start_swa(
    *,
    config: NonlinearConfig,
    elapsed_seconds: float,
    budget_seconds: float | None,
    epoch_count: int,
) -> bool:
    if not config.ae_use_swa:
        return False
    progress = _progress_fraction(
        elapsed_seconds=elapsed_seconds,
        budget_seconds=budget_seconds,
        epoch_index=epoch_count,
        total_epochs=config.ae_epochs,
    )
    return progress >= config.ae_swa_start_fraction


def _encode_all_frames(
    model: nn.Module,
    bundle: FieldBundle,
    stats: NormalizationStats,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(bundle.num_snapshots)
    dataset = SnapshotDataset(bundle.frames, indices, stats.mean, stats.std)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    latents = []
    reconstructions = []
    coarse_reconstructions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with _autocast_context(use_amp, device):
                latent = model.encode(batch)  # type: ignore[attr-defined]
                recon, auxiliary_outputs = _decode_autoencoder(model, latent)
            latents.append(latent.float().cpu().numpy())
            reconstructions.append(_denormalize(recon.float(), stats).cpu().numpy()[:, 0])
            coarse_reconstructions.append(_denormalize(auxiliary_outputs["coarse"].float(), stats).cpu().numpy()[:, 0])
    return (
        np.concatenate(latents, axis=0),
        np.concatenate(reconstructions, axis=0).astype(np.float32),
        np.concatenate(coarse_reconstructions, axis=0).astype(np.float32),
    )


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
    split: TemporalHoldoutSplit,
    config: NonlinearConfig,
    output_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, NormalizationStats, dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    train_idx = split.snapshot_train_idx
    val_idx = split.snapshot_val_idx
    test_idx = split.snapshot_test_idx
    train_frames = bundle.frames[train_idx]
    stats = NormalizationStats(mean=float(train_frames.mean()), std=float(train_frames.std() + 1e-6))
    train_dataset = SnapshotDataset(bundle.frames, train_idx, stats.mean, stats.std)
    val_dataset = SnapshotDataset(bundle.frames, val_idx, stats.mean, stats.std)
    test_dataset = SnapshotDataset(bundle.frames, test_idx, stats.mean, stats.std)
    train_loader = _build_loader(train_dataset, config.batch_size, True, config.seed)
    val_loader = _build_loader(val_dataset, config.batch_size, False, config.seed + 1)
    test_loader = _build_loader(test_dataset, config.batch_size, False, config.seed + 2)

    model = build_autoencoder(
        (bundle.height, bundle.width),
        config.latent_dim,
        config.ae_architecture,
        width_mult=config.ae_width_mult,
        coordconv=config.coordconv,
        refine_blocks=config.refine_blocks,
        refine_channels_mult=config.refine_channels_mult,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.ae_learning_rate, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(
        config.ae_scheduler,
        optimizer,
        config.ae_scheduler_factor,
        config.ae_scheduler_patience,
        config.ae_min_learning_rate,
        cosine_t0_epochs=config.ae_cosine_t0_epochs,
        cosine_tmult=config.ae_cosine_tmult,
        cosine_eta_min=config.ae_cosine_eta_min,
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "coarse_loss_weight": [],
        "gradient_loss_weight": [],
        "curriculum_progress": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 0
    batches_ran = 0
    stop_reason = "epoch_limit"
    eval_loss_weights = _ae_validation_loss_weights(config)
    swa_model = AveragedModel(model) if config.ae_use_swa else None
    swa_started = False
    swa_updates = 0
    swa_val_loss: float | None = None
    selected_model_source = "best_checkpoint"
    selected_val_loss = best_loss

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for epoch_idx in tqdm(range(config.ae_epochs), desc="AE", leave=False):
        model.train()
        batch_losses = []
        time_budget_hit = False
        epoch_elapsed = time.perf_counter() - start
        train_loss_weights, curriculum_progress = _ae_training_loss_weights(
            config,
            elapsed_seconds=epoch_elapsed,
            budget_seconds=config.ae_train_budget_seconds,
            epoch_index=epoch_idx,
        )
        for batch in train_loader:
            batch = batch.to(device)
            latent = model.encode(batch)  # type: ignore[attr-defined]
            recon, auxiliary_outputs = _decode_autoencoder(model, latent)
            loss = _autoencoder_reconstruction_loss(
                recon,
                batch,
                latent=latent,
                config=config,
                auxiliary_outputs=auxiliary_outputs,
                loss_weights=train_loss_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            batches_ran += 1
            if _budget_exhausted(start, config.ae_train_budget_seconds):
                time_budget_hit = True
                break
        if not batch_losses:
            stop_reason = "time_budget" if time_budget_hit else "epoch_limit"
            break
        val_loss, _, _, _ = _evaluate_autoencoder(
            model,
            val_loader,
            device,
            stats,
            config,
            loss_weights=eval_loss_weights,
        )
        train_loss = float(np.mean(batch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        epoch_elapsed = time.perf_counter() - start
        if swa_model is not None and not swa_started and _should_start_swa(
            config=config,
            elapsed_seconds=epoch_elapsed,
            budget_seconds=config.ae_train_budget_seconds,
            epoch_count=len(history["val_loss"]),
        ):
            swa_started = True
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.ae_swa_lr
        if not swa_started:
            _step_scheduler(scheduler, metric=val_loss)
        if swa_model is not None and swa_started:
            swa_model.update_parameters(model)
            swa_updates += 1
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        history["coarse_loss_weight"].append(float(train_loss_weights.coarse_loss_weight))
        history["gradient_loss_weight"].append(float(train_loss_weights.gradient_loss_weight))
        history["curriculum_progress"].append(float(curriculum_progress))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if time_budget_hit:
            stop_reason = "time_budget"
            break
        if patience >= config.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    ae_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    selected_val_loss = best_loss
    if swa_model is not None and swa_updates > 0:
        swa_val_loss, _, _, _ = _evaluate_autoencoder(
            swa_model.module,
            val_loader,
            device,
            stats,
            config,
            loss_weights=eval_loss_weights,
        )
        if swa_val_loss < best_loss:
            model.load_state_dict(swa_model.module.state_dict())
            selected_model_source = "swa"
            selected_val_loss = swa_val_loss
        else:
            selected_model_source = "best_checkpoint"

    _, reconstruction_summary, preview_pairs, coarse_summary = _evaluate_autoencoder(model, test_loader, device, stats, config)
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

    latents, reconstructed_frames, coarse_frames = _encode_all_frames(
        model,
        bundle,
        stats,
        config.batch_size,
        device,
        config.use_amp,
    )
    peak_memory_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
    metrics = {
        "snapshot_indices": split.snapshot_indices(),
        "recon_mse": reconstruction_summary.mse,
        "recon_rmse": reconstruction_summary.rmse,
        "recon_nrmse": reconstruction_summary.nrmse,
        "coarse_recon_mse": coarse_summary.mse,
        "coarse_recon_rmse": coarse_summary.rmse,
        "coarse_recon_nrmse": coarse_summary.nrmse,
        "history": history,
        "ae_seconds": ae_seconds,
        "ae_train_budget_seconds": config.ae_train_budget_seconds,
        "epochs_ran": len(history["val_loss"]),
        "batches_ran": batches_ran,
        "stop_reason": stop_reason,
        "best_val_loss": best_loss,
        "selected_val_loss": selected_val_loss,
        "selected_model_source": selected_model_source,
        "swa_enabled": config.ae_use_swa,
        "swa_started": swa_started,
        "swa_updates": swa_updates,
        "swa_start_fraction": config.ae_swa_start_fraction,
        "swa_lr": config.ae_swa_lr if config.ae_use_swa else None,
        "swa_val_loss": swa_val_loss,
        "curriculum_enabled": config.ae_use_curriculum,
        "stopped_for_time_budget": stop_reason == "time_budget",
        "stopped_for_early_stopping": stop_reason == "early_stopping",
        "stopped_for_epoch_limit": stop_reason == "epoch_limit",
        "peak_memory_gb_after_ae": peak_memory_gb,
        "artifact_paths": {
            "reconstruction_previews_full": preview_artifacts["full"],
            "reconstruction_previews_wake": preview_artifacts["wake"],
            "training_curves": "ae_training_curves.png",
        },
    }
    return model, stats, metrics, latents, reconstructed_frames, coarse_frames


def _train_dynamics(
    latents: np.ndarray,
    bundle: FieldBundle,
    autoencoder: nn.Module,
    stats: NormalizationStats,
    split: TemporalHoldoutSplit,
    config: NonlinearConfig,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    train_idx = split.transition_train_idx
    val_idx = split.transition_val_idx
    latent_stats = LatentNormalizationStats(
        mean=latents[split.snapshot_train_idx].mean(axis=0).astype(np.float32),
        std=(latents[split.snapshot_train_idx].std(axis=0) + 1e-6).astype(np.float32),
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
    if hasattr(model, "linear") and hasattr(model, "residual"):
        optimizer = AdamW(model.residual.parameters(), lr=config.dyn_learning_rate, weight_decay=config.weight_decay)  # type: ignore[attr-defined]
    else:
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
    train_starts = _rollout_start_indices(train_idx, config.train_rollout_stride)
    stop_reason = "epoch_limit"
    batches_ran = 0

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
            max_target_exclusive=split.snapshot_train_stop,
        )
        loss = one_step_loss + (config.rollout_loss_weight * rollout_loss) + (config.dyn_l2_weight * _regularization_l2(model))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batches_ran += 1
        time_budget_hit = _budget_exhausted(start, config.dyn_train_budget_seconds)

        model.eval()
        with torch.no_grad():
            val_one_step, val_rollout, val_loss = _dynamics_validation_terms(
                model,
                latent_tensor,
                val_idx,
                config,
                max_target_exclusive=split.snapshot_val_stop,
            )
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_one_step_loss"].append(val_one_step.item())
        history["val_rollout_loss"].append(val_rollout.item())
        _step_scheduler(scheduler, metric=val_loss.item())
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if time_budget_hit:
            stop_reason = "time_budget"
            break
        if patience >= config.early_stopping_patience:
            stop_reason = "early_stopping"
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
        "transition_indices": split.transition_indices(),
        "history": history,
        "dyn_seconds": dyn_seconds,
        "dyn_train_budget_seconds": config.dyn_train_budget_seconds,
        "epochs_ran": len(history["val_loss"]),
        "batches_ran": batches_ran,
        "stop_reason": stop_reason,
        "stopped_for_time_budget": stop_reason == "time_budget",
        "stopped_for_early_stopping": stop_reason == "early_stopping",
        "stopped_for_epoch_limit": stop_reason == "epoch_limit",
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
    *,
    ae_only: bool = False,
) -> dict[str, object]:
    config = config or NonlinearConfig()
    paths = paths or ProjectPaths()
    paths.ensure_directories()
    output_dir = paths.nonlinear_dir / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed, deterministic=config.deterministic)

    device = detect_device(config.device)
    bundle = load_field_bundle(config.field_name, paths, layout=config.layout)
    split = build_assignment_temporal_holdout(bundle.num_snapshots)
    compare_steps = _compare_steps(config.compare_steps, bundle.num_snapshots)
    compare_step_regions = _compare_step_regions(compare_steps, split)
    _save_truth_reference(bundle, compare_steps, output_dir, config)

    autoencoder, stats, ae_metrics, latents, reconstructed_frames, coarse_frames = _train_autoencoder(
        bundle,
        split,
        config,
        output_dir,
        device,
    )
    ae_floor_metrics = _compute_step_metrics(bundle.frames, reconstructed_frames, compare_steps)
    coarse_floor_metrics = _compute_step_metrics(bundle.frames, coarse_frames, compare_steps)
    ae_floor_artifacts = _save_step_artifacts("ae_floor", bundle.frames, reconstructed_frames, compare_steps, output_dir, config)
    ae_floor_t100 = ae_floor_metrics.get("step_100", {"rmse": 0.0, "mse": 0.0, "nrmse": 0.0})
    ae_floor_t150 = ae_floor_metrics.get("step_150", {"rmse": 0.0, "mse": 0.0, "nrmse": 0.0})
    coarse_t100 = coarse_floor_metrics.get("step_100", {"rmse": 0.0, "mse": 0.0, "nrmse": 0.0})
    coarse_t150 = coarse_floor_metrics.get("step_150", {"rmse": 0.0, "mse": 0.0, "nrmse": 0.0})
    ae_score = compute_ae_score(ae_metrics["recon_rmse"], ae_floor_t100["rmse"], ae_floor_t150["rmse"])
    ae_wall_seconds = ae_metrics["ae_seconds"]
    ae_met_speed_target = bool(
        config.target_primary_score is not None
        and config.target_wall_seconds is not None
        and ae_score <= config.target_primary_score
        and ae_wall_seconds <= config.target_wall_seconds
    )
    if ae_only:
        metrics = {
            "created_at": utc_timestamp(),
            "evaluation_protocol": split.protocol,
            "profile": config.profile,
            "campaign": config.campaign,
            "reference_full_score": config.reference_full_score,
            "target_primary_score": config.target_primary_score,
            "target_wall_seconds": config.target_wall_seconds,
            "met_speed_target": ae_met_speed_target,
            "field_name": bundle.field_name,
            "layout": config.layout,
            "summary": bundle.summary(),
            "config": asdict(config),
            "snapshot_split": split.snapshot_indices(),
            "transition_split": split.transition_indices(),
            "compare_step_regions": compare_step_regions,
            "device": str(device),
            "screening_mode": "ae_only",
            "ae_train_budget_seconds": config.ae_train_budget_seconds,
            "dyn_train_budget_seconds": config.dyn_train_budget_seconds,
            "ae": {**ae_metrics, "floor_metrics": ae_floor_metrics},
            "ae_score": ae_score,
            "primary_score": ae_score,
            "recon_mse": ae_metrics["recon_mse"],
            "recon_rmse": ae_metrics["recon_rmse"],
            "recon_nrmse": ae_metrics["recon_nrmse"],
            "coarse_only_mse_t100": coarse_t100["mse"],
            "coarse_only_rmse_t100": coarse_t100["rmse"],
            "coarse_only_nrmse_t100": coarse_t100["nrmse"],
            "coarse_only_mse_t150": coarse_t150["mse"],
            "coarse_only_rmse_t150": coarse_t150["rmse"],
            "coarse_only_nrmse_t150": coarse_t150["nrmse"],
            "ae_floor_mse_t100": ae_floor_t100["mse"],
            "ae_floor_rmse_t100": ae_floor_t100["rmse"],
            "ae_floor_nrmse_t100": ae_floor_t100["nrmse"],
            "ae_floor_mse_t150": ae_floor_t150["mse"],
            "ae_floor_rmse_t150": ae_floor_t150["rmse"],
            "ae_floor_nrmse_t150": ae_floor_t150["nrmse"],
            "peak_memory_gb": ae_metrics["peak_memory_gb_after_ae"],
            "wall_seconds": ae_wall_seconds,
            "artifacts": {
                "truth_portrait": "truth_portrait.png",
                "truth_wake_zoom": "truth_wake_zoom.png",
                "ae_training_curves": "ae_training_curves.png",
                "reconstruction_previews_full": ae_metrics["artifact_paths"]["reconstruction_previews_full"],
                "reconstruction_previews_wake": ae_metrics["artifact_paths"]["reconstruction_previews_wake"],
                "ae_floor_full": ae_floor_artifacts["full"],
                "ae_floor_wake": ae_floor_artifacts["wake"],
            },
        }
        write_json(output_dir / "metrics.json", metrics)
        _maybe_save_autoencoder_checkpoint(autoencoder, output_dir, save_checkpoint=config.save_checkpoint)
        return metrics

    dyn_metrics = _train_dynamics(latents, bundle, autoencoder, stats, split, config, output_dir, device)
    rollout_metrics = dyn_metrics["rollout_metrics"]
    linear_metrics = dyn_metrics["linear_baseline_metrics"]
    step_100 = rollout_metrics.get("step_100", {"mse": 0.0, "nrmse": 0.0, "rmse": 0.0})
    step_150 = rollout_metrics.get("step_150", {"mse": 0.0, "nrmse": 0.0, "rmse": 0.0})
    linear_t100 = linear_metrics.get("step_100", {"rmse": 0.0, "mse": 0.0, "nrmse": 0.0})
    linear_t150 = linear_metrics.get("step_150", {"rmse": 0.0, "mse": 0.0, "nrmse": 0.0})
    overall_score = primary_score(ae_metrics["recon_nrmse"], step_100["nrmse"], step_150["nrmse"])
    wall_seconds = ae_metrics["ae_seconds"] + dyn_metrics["dyn_seconds"]
    met_speed_target = bool(
        config.target_primary_score is not None
        and config.target_wall_seconds is not None
        and overall_score <= config.target_primary_score
        and wall_seconds <= config.target_wall_seconds
    )

    metrics = {
        "created_at": utc_timestamp(),
        "evaluation_protocol": split.protocol,
        "profile": config.profile,
        "campaign": config.campaign,
        "reference_full_score": config.reference_full_score,
        "target_primary_score": config.target_primary_score,
        "target_wall_seconds": config.target_wall_seconds,
        "met_speed_target": met_speed_target,
        "field_name": bundle.field_name,
        "layout": config.layout,
        "summary": bundle.summary(),
        "config": asdict(config),
        "snapshot_split": split.snapshot_indices(),
        "transition_split": split.transition_indices(),
        "compare_step_regions": compare_step_regions,
        "device": str(device),
        "ae_train_budget_seconds": config.ae_train_budget_seconds,
        "dyn_train_budget_seconds": config.dyn_train_budget_seconds,
        "ae": {**ae_metrics, "floor_metrics": ae_floor_metrics},
        "dynamics": dyn_metrics,
        "ae_score": ae_score,
        "primary_score": overall_score,
        "recon_mse": ae_metrics["recon_mse"],
        "recon_rmse": ae_metrics["recon_rmse"],
        "recon_nrmse": ae_metrics["recon_nrmse"],
        "coarse_only_mse_t100": coarse_t100["mse"],
        "coarse_only_rmse_t100": coarse_t100["rmse"],
        "coarse_only_nrmse_t100": coarse_t100["nrmse"],
        "coarse_only_mse_t150": coarse_t150["mse"],
        "coarse_only_rmse_t150": coarse_t150["rmse"],
        "coarse_only_nrmse_t150": coarse_t150["nrmse"],
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
        "wall_seconds": wall_seconds,
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
    _maybe_save_autoencoder_checkpoint(autoencoder, output_dir, save_checkpoint=config.save_checkpoint)
    return metrics
