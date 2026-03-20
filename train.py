from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlfd.config import NonlinearConfig, ProjectPaths
from mlfd.nonlinear import run_nonlinear_pipeline


# Main experiment surface. Edit these defaults directly during autoresearch.
FIELD_NAME = "VORTALL"
LAYOUT = "portrait"
LATENT_DIM = 32
AE_ARCHITECTURE = "residual_refine_gated"
AE_WIDTH_MULT = 0.875
COORDCONV = False
COARSE_LOSS_WEIGHT = 0.15
COARSE_BLUR_KERNEL = 9
COARSE_BLUR_SIGMA = 2.0
REFINE_BLOCKS = 1
REFINE_CHANNELS_MULT = 1.25
DYNAMICS_MODEL = "residual_linear"
AE_EPOCHS = 3600
DYN_EPOCHS = 1800
BATCH_SIZE = 16
AE_LEARNING_RATE = 1e-3
DYN_LEARNING_RATE = 5e-4
AE_SCHEDULER = "cosine_budget_floor"
DYN_SCHEDULER = "none"
WEIGHT_DECAY = 1e-5
LATENT_L1_WEIGHT = 1e-4
DYN_L2_WEIGHT = 0.0
DYN_DEVIATION_FROM_LINEAR_WEIGHT = 0.0
GRADIENT_LOSS_WEIGHT = 0.1
FFT_LOSS_WEIGHT = 0.0
ROLLOUT_LOSS_WEIGHT = 0.10
DYNAMICS_HIDDEN_DIM = 32
DYNAMICS_DEPTH = 2
TRAIN_ROLLOUT_STRIDE = 8
TRAIN_ROLLOUT_HORIZON = 24
VALIDATION_ROLLOUT_STRIDE = 4
VALIDATION_ROLLOUT_HORIZON = 24
DYN_RESIDUAL_GATE_MAX = 0.5
DYN_RESIDUAL_GATE_INIT = 0.05
DYN_RESIDUAL_WARMUP_FRACTION = 0.0
DYN_RESIDUAL_WARMUP_FLOOR = 1.0
DYN_LINEAR_RELATIVE_PENALTY = 0.0
DYN_SELECT_AFTER_WARMUP = False
DYN_CALIBRATE_DEPLOY_ALPHA = True
DYN_DEPLOY_ALPHA_GRID = (0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)
DYN_DEPLOY_ALPHA_TAIL_WEIGHT = 0.7
AE_MIN_LEARNING_RATE = 5e-6
EARLY_STOPPING_PATIENCE = 5000
DEVICE = "auto"
USE_AMP = True
DETERMINISTIC = True
SAVE_CHECKPOINT = False
PROXY_AE_TRAIN_BUDGET_SECONDS = 600.0
PROXY_DYN_TRAIN_BUDGET_SECONDS = 180.0
SPEED90_AE_TRAIN_BUDGET_SECONDS = 5100.0
SPEED90_DYN_TRAIN_BUDGET_SECONDS = 300.0
SPEED90_CAMPAIGN = "speed90_same_score_faster_v1"
SPEED90_REFERENCE_FULL_SCORE = 0.000255791203
SPEED90_TARGET_PRIMARY_SCORE = 0.000268580764
SPEED90_TARGET_WALL_SECONDS = 5400.0
AE_COSINE_T0_EPOCHS = 64
AE_COSINE_TMULT = 2
AE_COSINE_ETA_MIN = AE_MIN_LEARNING_RATE
AE_BUDGET_LR_FLOOR_FRACTION = 0.75
AE_USE_SWA = False
AE_SWA_START_FRACTION = 0.75
AE_SWA_LR = 1e-5
AE_USE_CURRICULUM = True
AE_CURRICULUM_FINISH_FRACTION = 0.8
AE_CURRICULUM_COARSE_START = 0.40
AE_CURRICULUM_GRADIENT_START = 0.05


def build_config(*, smoke: bool, device_override: str | None, profile: str, ae_cache_dir: str | None) -> NonlinearConfig:
    ae_train_budget_seconds = None
    dyn_train_budget_seconds = None
    campaign = None
    reference_full_score = None
    target_primary_score = None
    target_wall_seconds = None
    if profile == "proxy":
        ae_train_budget_seconds = PROXY_AE_TRAIN_BUDGET_SECONDS
        dyn_train_budget_seconds = PROXY_DYN_TRAIN_BUDGET_SECONDS
    elif profile == "speed90":
        ae_train_budget_seconds = SPEED90_AE_TRAIN_BUDGET_SECONDS
        dyn_train_budget_seconds = SPEED90_DYN_TRAIN_BUDGET_SECONDS
        campaign = SPEED90_CAMPAIGN
        reference_full_score = SPEED90_REFERENCE_FULL_SCORE
        target_primary_score = SPEED90_TARGET_PRIMARY_SCORE
        target_wall_seconds = SPEED90_TARGET_WALL_SECONDS
    config = NonlinearConfig(
        field_name=FIELD_NAME,
        layout=LAYOUT,
        profile=profile,
        campaign=campaign,
        reference_full_score=reference_full_score,
        target_primary_score=target_primary_score,
        target_wall_seconds=target_wall_seconds,
        latent_dim=LATENT_DIM,
        ae_architecture=AE_ARCHITECTURE,
        ae_width_mult=AE_WIDTH_MULT,
        coordconv=COORDCONV,
        coarse_loss_weight=COARSE_LOSS_WEIGHT,
        coarse_blur_kernel=COARSE_BLUR_KERNEL,
        coarse_blur_sigma=COARSE_BLUR_SIGMA,
        refine_blocks=REFINE_BLOCKS,
        refine_channels_mult=REFINE_CHANNELS_MULT,
        dynamics_model=DYNAMICS_MODEL,
        ae_epochs=AE_EPOCHS,
        dyn_epochs=DYN_EPOCHS,
        batch_size=BATCH_SIZE,
        ae_learning_rate=AE_LEARNING_RATE,
        dyn_learning_rate=DYN_LEARNING_RATE,
        ae_scheduler=AE_SCHEDULER,
        dyn_scheduler=DYN_SCHEDULER,
        ae_cosine_t0_epochs=AE_COSINE_T0_EPOCHS,
        ae_cosine_tmult=AE_COSINE_TMULT,
        ae_cosine_eta_min=AE_COSINE_ETA_MIN,
        ae_budget_lr_floor_fraction=AE_BUDGET_LR_FLOOR_FRACTION,
        ae_use_swa=AE_USE_SWA,
        ae_swa_start_fraction=AE_SWA_START_FRACTION,
        ae_swa_lr=AE_SWA_LR,
        ae_use_curriculum=AE_USE_CURRICULUM,
        ae_curriculum_finish_fraction=AE_CURRICULUM_FINISH_FRACTION,
        ae_curriculum_coarse_start=AE_CURRICULUM_COARSE_START,
        ae_curriculum_gradient_start=AE_CURRICULUM_GRADIENT_START,
        weight_decay=WEIGHT_DECAY,
        latent_l1_weight=LATENT_L1_WEIGHT,
        dyn_l2_weight=DYN_L2_WEIGHT,
        dyn_deviation_from_linear_weight=DYN_DEVIATION_FROM_LINEAR_WEIGHT,
        gradient_loss_weight=GRADIENT_LOSS_WEIGHT,
        fft_loss_weight=FFT_LOSS_WEIGHT,
        rollout_loss_weight=ROLLOUT_LOSS_WEIGHT,
        dynamics_hidden_dim=DYNAMICS_HIDDEN_DIM,
        dynamics_depth=DYNAMICS_DEPTH,
        train_rollout_stride=TRAIN_ROLLOUT_STRIDE,
        train_rollout_horizon=TRAIN_ROLLOUT_HORIZON,
        validation_rollout_stride=VALIDATION_ROLLOUT_STRIDE,
        validation_rollout_horizon=VALIDATION_ROLLOUT_HORIZON,
        dyn_residual_gate_max=DYN_RESIDUAL_GATE_MAX,
        dyn_residual_gate_init=DYN_RESIDUAL_GATE_INIT,
        dyn_residual_warmup_fraction=DYN_RESIDUAL_WARMUP_FRACTION,
        dyn_residual_warmup_floor=DYN_RESIDUAL_WARMUP_FLOOR,
        dyn_linear_relative_penalty=DYN_LINEAR_RELATIVE_PENALTY,
        dyn_select_after_warmup=DYN_SELECT_AFTER_WARMUP,
        dyn_calibrate_deploy_alpha=DYN_CALIBRATE_DEPLOY_ALPHA,
        dyn_deploy_alpha_grid=DYN_DEPLOY_ALPHA_GRID,
        dyn_deploy_alpha_tail_weight=DYN_DEPLOY_ALPHA_TAIL_WEIGHT,
        ae_min_learning_rate=AE_MIN_LEARNING_RATE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        device=device_override or DEVICE,
        use_amp=USE_AMP,
        deterministic=DETERMINISTIC,
        save_checkpoint=SAVE_CHECKPOINT,
        ae_train_budget_seconds=ae_train_budget_seconds,
        dyn_train_budget_seconds=dyn_train_budget_seconds,
        ae_cache_dir=ae_cache_dir,
    )
    return config.smoke() if smoke else config


def print_summary(metrics: dict[str, float]) -> None:
    print("---")
    print(f"primary_score:  {metrics['primary_score']:.6f}")
    print(f"recon_rmse:     {metrics['recon_rmse']:.6f}")
    if "rmse_t100" in metrics:
        print(f"rmse_t100:      {metrics['rmse_t100']:.6f}")
    if "rmse_t150" in metrics:
        print(f"rmse_t150:      {metrics['rmse_t150']:.6f}")
    print(f"peak_memory_gb: {metrics['peak_memory_gb']:.3f}")
    print(f"wall_seconds:   {metrics['wall_seconds']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one nonlinear fluid autoresearch experiment.")
    parser.add_argument("--output-tag", default="run", help="Subdirectory name under output/nonlinear.")
    parser.add_argument("--smoke", action="store_true", help="Run a very short smoke configuration.")
    parser.add_argument("--ae-only", action="store_true", help="Train only the autoencoder and report AE-floor metrics.")
    parser.add_argument("--device", default=None, help="Explicit torch device override.")
    parser.add_argument("--profile", choices=("proxy", "speed90", "full"), default="full", help="Training budget profile.")
    parser.add_argument("--ae-cache-dir", default=None, help="Optional cache directory with AE checkpoint and latent cache for dynamics-only reruns.")
    args = parser.parse_args()

    config = build_config(smoke=args.smoke, device_override=args.device, profile=args.profile, ae_cache_dir=args.ae_cache_dir)
    metrics = run_nonlinear_pipeline(config, ProjectPaths(root=ROOT), output_tag=args.output_tag, ae_only=args.ae_only)
    print_summary(metrics)


if __name__ == "__main__":
    main()
