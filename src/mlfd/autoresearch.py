from __future__ import annotations

import argparse
import subprocess
import sys

from .config import AutoresearchConfig, ProjectPaths
from .experiments import build_record, init_results_file, load_records, record_experiment
from .utils import read_json


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _default_command(
    output_tag: str,
    smoke: bool,
    *,
    layout: str | None = None,
    ae_architecture: str | None = None,
    dynamics_model: str | None = None,
    ae_epochs: int | None = None,
    dyn_epochs: int | None = None,
    latent_dim: int | None = None,
    ae_learning_rate: float | None = None,
    dyn_learning_rate: float | None = None,
    rollout_loss_weight: float | None = None,
    gradient_loss_weight: float | None = None,
    dynamics_depth: int | None = None,
    dynamics_hidden_dim: int | None = None,
    train_rollout_stride: int | None = None,
    train_rollout_horizon: int | None = None,
    validation_rollout_stride: int | None = None,
    validation_rollout_horizon: int | None = None,
    ae_scheduler: str | None = None,
    dyn_scheduler: str | None = None,
    ae_scheduler_factor: float | None = None,
    dyn_scheduler_factor: float | None = None,
    ae_scheduler_patience: int | None = None,
    dyn_scheduler_patience: int | None = None,
    ae_min_learning_rate: float | None = None,
    dyn_min_learning_rate: float | None = None,
    latent_l1_weight: float | None = None,
    dyn_l2_weight: float | None = None,
    deterministic: bool | None = None,
    device: str | None = None,
) -> list[str]:
    command = [sys.executable, "-m", "mlfd.run_nonlinear", "--output-tag", output_tag]
    if smoke:
        command.append("--smoke")
    _append_optional(command, "--layout", layout)
    _append_optional(command, "--ae-architecture", ae_architecture)
    _append_optional(command, "--dynamics-model", dynamics_model)
    _append_optional(command, "--ae-epochs", ae_epochs)
    _append_optional(command, "--dyn-epochs", dyn_epochs)
    _append_optional(command, "--latent-dim", latent_dim)
    _append_optional(command, "--ae-learning-rate", ae_learning_rate)
    _append_optional(command, "--dyn-learning-rate", dyn_learning_rate)
    _append_optional(command, "--rollout-loss-weight", rollout_loss_weight)
    _append_optional(command, "--gradient-loss-weight", gradient_loss_weight)
    _append_optional(command, "--dynamics-depth", dynamics_depth)
    _append_optional(command, "--dynamics-hidden-dim", dynamics_hidden_dim)
    _append_optional(command, "--train-rollout-stride", train_rollout_stride)
    _append_optional(command, "--train-rollout-horizon", train_rollout_horizon)
    _append_optional(command, "--validation-rollout-stride", validation_rollout_stride)
    _append_optional(command, "--validation-rollout-horizon", validation_rollout_horizon)
    _append_optional(command, "--ae-scheduler", ae_scheduler)
    _append_optional(command, "--dyn-scheduler", dyn_scheduler)
    _append_optional(command, "--ae-scheduler-factor", ae_scheduler_factor)
    _append_optional(command, "--dyn-scheduler-factor", dyn_scheduler_factor)
    _append_optional(command, "--ae-scheduler-patience", ae_scheduler_patience)
    _append_optional(command, "--dyn-scheduler-patience", dyn_scheduler_patience)
    _append_optional(command, "--ae-min-learning-rate", ae_min_learning_rate)
    _append_optional(command, "--dyn-min-learning-rate", dyn_min_learning_rate)
    _append_optional(command, "--latent-l1-weight", latent_l1_weight)
    _append_optional(command, "--dyn-l2-weight", dyn_l2_weight)
    if deterministic is not None:
        _append_optional(command, "--deterministic", str(deterministic).lower())
    _append_optional(command, "--device", device)
    return command


def _run_and_collect(paths: ProjectPaths, command: list[str], output_tag: str, timeout_seconds: int) -> tuple[dict[str, float], str]:
    log_dir = paths.run_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{output_tag}.log"
    metrics_path = paths.nonlinear_dir / output_tag / "metrics.json"
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=paths.root,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics file not found: {metrics_path}")
    return read_json(metrics_path), str(log_path.relative_to(paths.root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch helpers for experiment logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-run", help="Initialize the run directories and ledger.")
    init_parser.add_argument("--run-tag", default=ProjectPaths().run_tag)

    run_parser = subparsers.add_parser("run-current", help="Run the current nonlinear code and record the result.")
    run_parser.add_argument("--description", default=AutoresearchConfig().baseline_description)
    run_parser.add_argument("--run-tag", default=ProjectPaths().run_tag)
    run_parser.add_argument("--smoke", action="store_true")
    run_parser.add_argument("--next-hypothesis", default="")
    run_parser.add_argument("--layout", choices=["landscape", "portrait"], default=None)
    run_parser.add_argument("--ae-architecture", choices=["baseline", "residual"], default=None)
    run_parser.add_argument("--dynamics-model", choices=["mlp", "residual_linear"], default=None)
    run_parser.add_argument("--ae-epochs", type=int, default=None)
    run_parser.add_argument("--dyn-epochs", type=int, default=None)
    run_parser.add_argument("--latent-dim", type=int, default=None)
    run_parser.add_argument("--ae-learning-rate", type=float, default=None)
    run_parser.add_argument("--dyn-learning-rate", type=float, default=None)
    run_parser.add_argument("--rollout-loss-weight", type=float, default=None)
    run_parser.add_argument("--gradient-loss-weight", type=float, default=None)
    run_parser.add_argument("--dynamics-depth", type=int, default=None)
    run_parser.add_argument("--dynamics-hidden-dim", type=int, default=None)
    run_parser.add_argument("--train-rollout-stride", type=int, default=None)
    run_parser.add_argument("--train-rollout-horizon", type=int, default=None)
    run_parser.add_argument("--validation-rollout-stride", type=int, default=None)
    run_parser.add_argument("--validation-rollout-horizon", type=int, default=None)
    run_parser.add_argument("--ae-scheduler", choices=["none", "plateau"], default=None)
    run_parser.add_argument("--dyn-scheduler", choices=["none", "plateau"], default=None)
    run_parser.add_argument("--ae-scheduler-factor", type=float, default=None)
    run_parser.add_argument("--dyn-scheduler-factor", type=float, default=None)
    run_parser.add_argument("--ae-scheduler-patience", type=int, default=None)
    run_parser.add_argument("--dyn-scheduler-patience", type=int, default=None)
    run_parser.add_argument("--ae-min-learning-rate", type=float, default=None)
    run_parser.add_argument("--dyn-min-learning-rate", type=float, default=None)
    run_parser.add_argument("--latent-l1-weight", type=float, default=None)
    run_parser.add_argument("--dyn-l2-weight", type=float, default=None)
    run_parser.add_argument("--deterministic", choices=["true", "false"], default=None)
    run_parser.add_argument("--device", default=None)

    args = parser.parse_args()
    paths = ProjectPaths(run_tag=args.run_tag)
    paths.ensure_directories()
    init_results_file(paths)

    if args.command == "init-run":
        print(f"Initialized run scaffold for {paths.run_tag}")
        print(f"Expected branch: {paths.branch_name}")
        return

    config = AutoresearchConfig(run_tag=paths.run_tag)
    experiment_id = len(load_records(paths)) + 1
    output_tag = f"exp-{experiment_id:04d}"
    command = _default_command(
        output_tag,
        args.smoke,
        layout=args.layout,
        ae_architecture=args.ae_architecture,
        dynamics_model=args.dynamics_model,
        ae_epochs=args.ae_epochs,
        dyn_epochs=args.dyn_epochs,
        latent_dim=args.latent_dim,
        ae_learning_rate=args.ae_learning_rate,
        dyn_learning_rate=args.dyn_learning_rate,
        rollout_loss_weight=args.rollout_loss_weight,
        gradient_loss_weight=args.gradient_loss_weight,
        dynamics_depth=args.dynamics_depth,
        dynamics_hidden_dim=args.dynamics_hidden_dim,
        train_rollout_stride=args.train_rollout_stride,
        train_rollout_horizon=args.train_rollout_horizon,
        validation_rollout_stride=args.validation_rollout_stride,
        validation_rollout_horizon=args.validation_rollout_horizon,
        ae_scheduler=args.ae_scheduler,
        dyn_scheduler=args.dyn_scheduler,
        ae_scheduler_factor=args.ae_scheduler_factor,
        dyn_scheduler_factor=args.dyn_scheduler_factor,
        ae_scheduler_patience=args.ae_scheduler_patience,
        dyn_scheduler_patience=args.dyn_scheduler_patience,
        ae_min_learning_rate=args.ae_min_learning_rate,
        dyn_min_learning_rate=args.dyn_min_learning_rate,
        latent_l1_weight=args.latent_l1_weight,
        dyn_l2_weight=args.dyn_l2_weight,
        deterministic=None if args.deterministic is None else args.deterministic == "true",
        device=args.device,
    )
    try:
        metrics, log_path = _run_and_collect(paths, command, output_tag, config.timeout_seconds)
        failure_reason = ""
        status = None
    except Exception as exc:  # noqa: BLE001
        metrics = {
            "primary_score": 0.0,
            "recon_rmse": 0.0,
            "rmse_t100": 0.0,
            "rmse_t150": 0.0,
            "peak_memory_gb": 0.0,
            "wall_seconds": float(config.timeout_seconds),
        }
        failure_reason = str(exc)
        status = "crash"
        log_path = str((paths.run_log_dir / f"{output_tag}.log").relative_to(paths.root))
    record = build_record(
        paths,
        config,
        metrics,
        description=args.description,
        status=status,
        failure_reason=failure_reason,
        next_hypothesis=args.next_hypothesis,
        metrics_path=str((paths.nonlinear_dir / output_tag / "metrics.json").relative_to(paths.root)),
        log_path=log_path,
    )
    record_experiment(paths, record)
    print(f"Recorded exp-{record.experiment_id:04d}")
    print(f"Status: {record.status}")
    print(f"Primary score: {record.primary_score:.6f}")


if __name__ == "__main__":
    main()
