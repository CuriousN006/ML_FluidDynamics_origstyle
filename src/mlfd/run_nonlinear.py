from __future__ import annotations

import argparse
from argparse import Namespace

from .config import NonlinearConfig, ProjectPaths
from .nonlinear import run_nonlinear_pipeline


def _override(config: NonlinearConfig, **changes: object) -> NonlinearConfig:
    return config.__class__(**{**config.__dict__, **changes})


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def apply_cli_overrides(config: NonlinearConfig, args: Namespace) -> NonlinearConfig:
    if args.smoke:
        config = config.smoke()
    override_fields = {
        "layout": args.layout,
        "ae_architecture": args.ae_architecture,
        "ae_width_mult": args.ae_width_mult,
        "coordconv": args.coordconv,
        "coarse_loss_weight": args.coarse_loss_weight,
        "coarse_blur_kernel": args.coarse_blur_kernel,
        "coarse_blur_sigma": args.coarse_blur_sigma,
        "refine_blocks": args.refine_blocks,
        "refine_channels_mult": args.refine_channels_mult,
        "dynamics_model": args.dynamics_model,
        "ae_epochs": args.ae_epochs,
        "dyn_epochs": args.dyn_epochs,
        "latent_dim": args.latent_dim,
        "rollout_loss_weight": args.rollout_loss_weight,
        "gradient_loss_weight": args.gradient_loss_weight,
        "fft_loss_weight": args.fft_loss_weight,
        "dynamics_depth": args.dynamics_depth,
        "dynamics_hidden_dim": args.dynamics_hidden_dim,
        "train_rollout_stride": args.train_rollout_stride,
        "train_rollout_horizon": args.train_rollout_horizon,
        "validation_rollout_stride": args.validation_rollout_stride,
        "validation_rollout_horizon": args.validation_rollout_horizon,
        "ae_learning_rate": args.ae_learning_rate,
        "dyn_learning_rate": args.dyn_learning_rate,
        "ae_scheduler": args.ae_scheduler,
        "dyn_scheduler": args.dyn_scheduler,
        "ae_scheduler_factor": args.ae_scheduler_factor,
        "dyn_scheduler_factor": args.dyn_scheduler_factor,
        "ae_scheduler_patience": args.ae_scheduler_patience,
        "dyn_scheduler_patience": args.dyn_scheduler_patience,
        "ae_min_learning_rate": args.ae_min_learning_rate,
        "dyn_min_learning_rate": args.dyn_min_learning_rate,
        "latent_l1_weight": args.latent_l1_weight,
        "dyn_l2_weight": args.dyn_l2_weight,
        "device": args.device,
        "deterministic": args.deterministic,
        "save_checkpoint": args.save_checkpoint,
    }
    for key, value in override_fields.items():
        if value is not None:
            config = _override(config, **{key: value})
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the nonlinear fluid dynamics baseline.")
    parser.add_argument("--output-tag", default="baseline", help="Subdirectory name under output/nonlinear.")
    parser.add_argument("--smoke", action="store_true", help="Run a very short smoke configuration.")
    parser.add_argument("--ae-only", action="store_true", help="Train only the autoencoder and report AE-floor metrics.")
    parser.add_argument("--layout", choices=["landscape", "portrait"], default=None, help="Override frame layout.")
    parser.add_argument(
        "--ae-architecture",
        choices=["baseline", "residual", "residual_refine", "residual_multiscale"],
        default=None,
        help="Override the autoencoder architecture.",
    )
    parser.add_argument("--ae-width-mult", type=float, default=None, help="Override AE channel width multiplier.")
    parser.add_argument(
        "--coordconv",
        type=_parse_bool,
        default=None,
        help="Override encoder CoordConv positional channels. Use true or false.",
    )
    parser.add_argument("--coarse-loss-weight", type=float, default=None, help="Override coarse-to-fine auxiliary loss weight.")
    parser.add_argument("--coarse-blur-kernel", type=int, default=None, help="Override coarse target Gaussian kernel size.")
    parser.add_argument("--coarse-blur-sigma", type=float, default=None, help="Override coarse target Gaussian sigma.")
    parser.add_argument("--refine-blocks", type=int, default=None, help="Override the number of refinement residual blocks.")
    parser.add_argument(
        "--refine-channels-mult",
        type=float,
        default=None,
        help="Override refinement head channel multiplier.",
    )
    parser.add_argument(
        "--dynamics-model",
        choices=["mlp", "residual_linear"],
        default=None,
        help="Override the latent dynamics model type.",
    )
    parser.add_argument("--ae-epochs", type=int, default=None, help="Override autoencoder epochs.")
    parser.add_argument("--dyn-epochs", type=int, default=None, help="Override dynamics epochs.")
    parser.add_argument("--latent-dim", type=int, default=None, help="Override latent dimension.")
    parser.add_argument("--ae-learning-rate", type=float, default=None, help="Override AE learning rate.")
    parser.add_argument("--dyn-learning-rate", type=float, default=None, help="Override dynamics learning rate.")
    parser.add_argument(
        "--rollout-loss-weight",
        type=float,
        default=None,
        help="Override rollout loss weight in the latent dynamics model.",
    )
    parser.add_argument(
        "--gradient-loss-weight",
        type=float,
        default=None,
        help="Override the reconstruction gradient loss weight.",
    )
    parser.add_argument(
        "--fft-loss-weight",
        type=float,
        default=None,
        help="Override the reconstruction FFT magnitude loss weight.",
    )
    parser.add_argument(
        "--dynamics-depth",
        type=int,
        default=None,
        help="Override the number of hidden layers in the latent dynamics MLP.",
    )
    parser.add_argument(
        "--dynamics-hidden-dim",
        type=int,
        default=None,
        help="Override the hidden width of the latent dynamics MLP.",
    )
    parser.add_argument("--train-rollout-stride", type=int, default=None, help="Override training rollout stride.")
    parser.add_argument("--train-rollout-horizon", type=int, default=None, help="Override training rollout horizon.")
    parser.add_argument(
        "--validation-rollout-stride",
        type=int,
        default=None,
        help="Override validation rollout stride.",
    )
    parser.add_argument(
        "--validation-rollout-horizon",
        type=int,
        default=None,
        help="Override validation rollout horizon.",
    )
    parser.add_argument(
        "--ae-scheduler",
        choices=["none", "plateau"],
        default=None,
        help="Override the AE scheduler type.",
    )
    parser.add_argument(
        "--dyn-scheduler",
        choices=["none", "plateau"],
        default=None,
        help="Override the dynamics scheduler type.",
    )
    parser.add_argument("--ae-scheduler-factor", type=float, default=None, help="Override AE scheduler factor.")
    parser.add_argument("--dyn-scheduler-factor", type=float, default=None, help="Override dynamics scheduler factor.")
    parser.add_argument("--ae-scheduler-patience", type=int, default=None, help="Override AE scheduler patience.")
    parser.add_argument(
        "--dyn-scheduler-patience",
        type=int,
        default=None,
        help="Override dynamics scheduler patience.",
    )
    parser.add_argument("--ae-min-learning-rate", type=float, default=None, help="Override AE minimum learning rate.")
    parser.add_argument(
        "--dyn-min-learning-rate",
        type=float,
        default=None,
        help="Override dynamics minimum learning rate.",
    )
    parser.add_argument("--latent-l1-weight", type=float, default=None, help="Override AE latent L1 regularizer.")
    parser.add_argument("--dyn-l2-weight", type=float, default=None, help="Override dynamics L2 regularizer.")
    parser.add_argument(
        "--deterministic",
        type=_parse_bool,
        default=None,
        help="Override deterministic torch behavior. Use true or false.",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=_parse_bool,
        default=None,
        help="Persist autoencoder.pt for this run. Defaults to false.",
    )
    parser.add_argument("--device", default=None, help="Explicit torch device.")
    args = parser.parse_args()

    config = apply_cli_overrides(NonlinearConfig(), args)
    metrics = run_nonlinear_pipeline(config, ProjectPaths(), output_tag=args.output_tag, ae_only=args.ae_only)
    print(f"primary_score: {metrics['primary_score']:.6f}")
    if "ae_score" in metrics:
        print(f"ae_score: {metrics['ae_score']:.6f}")
    print(f"recon_rmse: {metrics['recon_rmse']:.6f}")
    if "rmse_t100" in metrics:
        print(f"rmse_t100: {metrics['rmse_t100']:.6f}")
    if "rmse_t150" in metrics:
        print(f"rmse_t150: {metrics['rmse_t150']:.6f}")
    print(f"peak_memory_gb: {metrics['peak_memory_gb']:.3f}")


if __name__ == "__main__":
    main()
