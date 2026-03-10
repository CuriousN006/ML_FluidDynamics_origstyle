from __future__ import annotations

import argparse

from .config import NonlinearConfig, ProjectPaths
from .nonlinear import run_nonlinear_pipeline


def _override(config: NonlinearConfig, **changes: object) -> NonlinearConfig:
    return config.__class__(**{**config.__dict__, **changes})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the nonlinear fluid dynamics baseline.")
    parser.add_argument("--output-tag", default="baseline", help="Subdirectory name under output/nonlinear.")
    parser.add_argument("--smoke", action="store_true", help="Run a very short smoke configuration.")
    parser.add_argument("--ae-epochs", type=int, default=None, help="Override autoencoder epochs.")
    parser.add_argument("--dyn-epochs", type=int, default=None, help="Override dynamics epochs.")
    parser.add_argument("--latent-dim", type=int, default=None, help="Override latent dimension.")
    parser.add_argument(
        "--rollout-loss-weight",
        type=float,
        default=None,
        help="Override rollout loss weight in the latent dynamics model.",
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
    parser.add_argument("--device", default=None, help="Explicit torch device.")
    args = parser.parse_args()

    config = NonlinearConfig()
    if args.smoke:
        config = config.smoke()
    if args.ae_epochs is not None:
        config = _override(config, ae_epochs=args.ae_epochs)
    if args.dyn_epochs is not None:
        config = _override(config, dyn_epochs=args.dyn_epochs)
    if args.latent_dim is not None:
        config = _override(config, latent_dim=args.latent_dim)
    if args.rollout_loss_weight is not None:
        config = _override(config, rollout_loss_weight=args.rollout_loss_weight)
    if args.dynamics_depth is not None:
        config = _override(config, dynamics_depth=args.dynamics_depth)
    if args.dynamics_hidden_dim is not None:
        config = _override(config, dynamics_hidden_dim=args.dynamics_hidden_dim)
    if args.device is not None:
        config = _override(config, device=args.device)
    metrics = run_nonlinear_pipeline(config, ProjectPaths(), output_tag=args.output_tag)
    print(f"primary_score: {metrics['primary_score']:.6f}")
    print(f"recon_rmse: {metrics['recon_rmse']:.6f}")
    print(f"rmse_t100: {metrics['rmse_t100']:.6f}")
    print(f"rmse_t150: {metrics['rmse_t150']:.6f}")
    print(f"peak_memory_gb: {metrics['peak_memory_gb']:.3f}")


if __name__ == "__main__":
    main()
