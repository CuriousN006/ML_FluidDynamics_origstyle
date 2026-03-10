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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import NonlinearConfig, ProjectPaths
from .data import FieldBundle, load_field_bundle, random_snapshot_split, sequential_transition_split
from .metrics import mse, nrmse, primary_score, rmse
from .models import ConvAutoencoder, LatentDynamicsMLP
from .plots import save_comparison_panel, save_training_curves
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


def _denormalize(x: torch.Tensor | np.ndarray, stats: NormalizationStats) -> torch.Tensor | np.ndarray:
    return (x * stats.std) + stats.mean


def _regularization_l2(module: nn.Module) -> torch.Tensor:
    penalties = [parameter.pow(2).mean() for parameter in module.parameters() if parameter.requires_grad]
    if not penalties:
        return torch.tensor(0.0)
    return torch.stack(penalties).mean()


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


def _latent_rollout_loss(
    model: LatentDynamicsMLP,
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


def _dynamics_validation_terms(
    model: LatentDynamicsMLP,
    latent_tensor: torch.Tensor,
    val_idx: np.ndarray,
    config: NonlinearConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    val_one_step = F.mse_loss(model(latent_tensor[val_idx]), latent_tensor[val_idx + 1])
    val_rollout = _latent_rollout_loss(
        model,
        latent_tensor,
        start_index=int(val_idx[0]),
        horizon=min(config.validation_rollout_horizon, len(val_idx)),
    )
    selection_loss = val_one_step + (config.validation_rollout_weight * val_rollout)
    return val_one_step, val_rollout, selection_loss


def _evaluate_autoencoder(
    model: ConvAutoencoder,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    stats: NormalizationStats,
) -> tuple[float, float, float, list[tuple[np.ndarray, np.ndarray]]]:
    model.eval()
    losses = []
    truth_pred_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            losses.append(F.mse_loss(recon, batch).item())
            truth = _denormalize(batch, stats).cpu().numpy()
            pred = _denormalize(recon, stats).cpu().numpy()
            for true_item, pred_item in zip(truth, pred, strict=True):
                truth_pred_pairs.append((true_item[0], pred_item[0]))
    recon_rmse = float(np.mean([rmse(true, pred) for true, pred in truth_pred_pairs]))
    recon_nrmse = float(np.mean([nrmse(true, pred) for true, pred in truth_pred_pairs]))
    return float(np.mean(losses)), recon_rmse, recon_nrmse, truth_pred_pairs


def _train_autoencoder(
    bundle: FieldBundle,
    config: NonlinearConfig,
    output_dir: Path,
    device: torch.device,
) -> tuple[ConvAutoencoder, NormalizationStats, dict[str, object], np.ndarray]:
    train_idx, test_idx = random_snapshot_split(bundle.num_snapshots, config.ae_train_ratio, config.seed)
    train_frames = bundle.frames[train_idx]
    stats = NormalizationStats(mean=float(train_frames.mean()), std=float(train_frames.std() + 1e-6))
    train_dataset = SnapshotDataset(bundle.frames, train_idx, stats.mean, stats.std)
    test_dataset = SnapshotDataset(bundle.frames, test_idx, stats.mean, stats.std)
    train_loader = _build_loader(train_dataset, config.batch_size, True, config.seed)
    test_loader = _build_loader(test_dataset, config.batch_size, False, config.seed + 1)

    model = ConvAutoencoder((bundle.height, bundle.width), config.latent_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = {"train_loss": [], "val_loss": []}
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
            loss = F.mse_loss(recon, batch) + (config.latent_l1_weight * latent.abs().mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        val_loss, _, _, _ = _evaluate_autoencoder(model, test_loader, device, stats)
        train_loss = float(np.mean(batch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
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

    _, recon_rmse, recon_nrmse, preview_pairs = _evaluate_autoencoder(model, test_loader, device, stats)
    save_training_curves(history, output_dir / "ae_training_curves.png", "Autoencoder training")
    for preview_idx, (true_snapshot, pred_snapshot) in enumerate(preview_pairs[: config.num_preview_images], start=1):
        save_comparison_panel(
            true_snapshot,
            pred_snapshot,
            output_dir / f"ae_reconstruction_preview_{preview_idx}.png",
            f"AE reconstruction preview {preview_idx}",
        )

    full_dataset = SnapshotDataset(bundle.frames, np.arange(bundle.num_snapshots), stats.mean, stats.std)
    full_loader = DataLoader(full_dataset, batch_size=min(32, bundle.num_snapshots), shuffle=False)
    latent_vectors = []
    model.eval()
    with torch.no_grad():
        for batch in full_loader:
            latent_vectors.append(model.encode(batch.to(device)).cpu().numpy())
    latents = np.concatenate(latent_vectors, axis=0)
    peak_memory_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
    metrics = {
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "recon_rmse": recon_rmse,
        "recon_nrmse": recon_nrmse,
        "history": history,
        "ae_seconds": ae_seconds,
        "peak_memory_gb_after_ae": peak_memory_gb,
    }
    return model, stats, metrics, latents


def _train_dynamics(
    latents: np.ndarray,
    bundle: FieldBundle,
    autoencoder: ConvAutoencoder,
    stats: NormalizationStats,
    config: NonlinearConfig,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    train_idx, val_idx = sequential_transition_split(bundle.num_snapshots, config.dyn_train_ratio)
    latent_tensor = torch.from_numpy(latents.astype(np.float32)).to(device)
    model = LatentDynamicsMLP(config.latent_dim, config.dynamics_hidden_dim, config.dynamics_depth).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_one_step_loss": [], "val_rollout_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 0

    start = time.perf_counter()
    for _ in tqdm(range(config.dyn_epochs), desc="Dyn", leave=False):
        model.train()
        prediction = model(latent_tensor[train_idx])
        target = latent_tensor[train_idx + 1]
        loss = F.mse_loss(prediction, target) + (config.dyn_l2_weight * _regularization_l2(model))
        rollout_horizon = min(5, len(train_idx))
        if rollout_horizon > 1:
            loss = loss + (
                config.rollout_loss_weight
                * _latent_rollout_loss(model, latent_tensor, start_index=int(train_idx[0]), horizon=rollout_horizon)
            )
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
    save_training_curves(history, output_dir / "dynamics_training_curves.png", "Latent dynamics training")

    model.eval()
    rollout_latents = [latent_tensor[0].detach().cpu().numpy()]
    current = latent_tensor[0:1]
    with torch.no_grad():
        for _ in range(1, bundle.num_snapshots):
            current = model(current)
            rollout_latents.append(current.squeeze(0).cpu().numpy())
    rollout_latents_np = np.stack(rollout_latents)

    autoencoder = autoencoder.to("cpu")
    model = model.to("cpu")
    if device.type == "cuda":
        del latent_tensor
        torch.cuda.empty_cache()
    with torch.no_grad():
        recon = autoencoder.decode(torch.from_numpy(rollout_latents_np.astype(np.float32))).numpy()[:, 0]
    predicted_frames = _denormalize(recon, stats).astype(np.float32)

    compare_metrics: dict[str, dict[str, float]] = {}
    for step in config.compare_steps:
        if step >= bundle.num_snapshots:
            continue
        true_snapshot = bundle.frames[step]
        pred_snapshot = predicted_frames[step]
        compare_metrics[f"step_{step}"] = {
            "mse": mse(true_snapshot, pred_snapshot),
            "rmse": rmse(true_snapshot, pred_snapshot),
            "nrmse": nrmse(true_snapshot, pred_snapshot),
        }
        save_comparison_panel(
            true_snapshot,
            pred_snapshot,
            output_dir / f"rollout_step_{step}.png",
            f"Latent rollout prediction | step {step}",
        )

    peak_memory_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
    return {
        "train_transition_indices": train_idx.tolist(),
        "val_transition_indices": val_idx.tolist(),
        "history": history,
        "dyn_seconds": dyn_seconds,
        "rollout_metrics": compare_metrics,
        "peak_memory_gb_after_dyn": peak_memory_gb,
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
    bundle = load_field_bundle(config.field_name, paths)

    autoencoder, stats, ae_metrics, latents = _train_autoencoder(bundle, config, output_dir, device)
    dyn_metrics = _train_dynamics(latents, bundle, autoencoder, stats, config, output_dir, device)
    rollout_metrics = dyn_metrics["rollout_metrics"]
    step_100 = rollout_metrics.get("step_100", {"nrmse": 0.0, "rmse": 0.0})
    step_150 = rollout_metrics.get("step_150", {"nrmse": 0.0, "rmse": 0.0})
    overall_score = primary_score(ae_metrics["recon_nrmse"], step_100["nrmse"], step_150["nrmse"])

    metrics = {
        "created_at": utc_timestamp(),
        "field_name": bundle.field_name,
        "summary": bundle.summary(),
        "config": asdict(config),
        "device": str(device),
        "ae": ae_metrics,
        "dynamics": dyn_metrics,
        "primary_score": overall_score,
        "recon_rmse": ae_metrics["recon_rmse"],
        "recon_nrmse": ae_metrics["recon_nrmse"],
        "rmse_t100": step_100["rmse"],
        "rmse_t150": step_150["rmse"],
        "peak_memory_gb": max(ae_metrics["peak_memory_gb_after_ae"], dyn_metrics["peak_memory_gb_after_dyn"]),
        "wall_seconds": ae_metrics["ae_seconds"] + dyn_metrics["dyn_seconds"],
    }
    write_json(output_dir / "metrics.json", metrics)
    torch.save(autoencoder.state_dict(), output_dir / "autoencoder.pt")
    return metrics
