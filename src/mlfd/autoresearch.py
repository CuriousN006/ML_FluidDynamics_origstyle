from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .config import AutoresearchConfig, ProjectPaths
from .experiments import best_record, build_record, init_results_file, load_records, record_experiment
from .idea_registry import check_idea_allowed, ensure_registry_files, load_registry
from .utils import read_json, run_git, utc_timestamp, write_json

SEARCH_FAMILIES = ("residual", "residual_refine", "residual_multiscale")


def _normalize_relpath(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _is_under_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    normalized = _normalize_relpath(path)
    return any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in prefixes)


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _default_command(
    output_tag: str,
    smoke: bool,
    *,
    ae_only: bool = False,
    layout: str | None = None,
    ae_architecture: str | None = None,
    ae_width_mult: float | None = None,
    coordconv: bool | None = None,
    coarse_loss_weight: float | None = None,
    coarse_blur_kernel: int | None = None,
    coarse_blur_sigma: float | None = None,
    refine_blocks: int | None = None,
    refine_channels_mult: float | None = None,
    dynamics_model: str | None = None,
    ae_epochs: int | None = None,
    dyn_epochs: int | None = None,
    latent_dim: int | None = None,
    ae_learning_rate: float | None = None,
    dyn_learning_rate: float | None = None,
    rollout_loss_weight: float | None = None,
    gradient_loss_weight: float | None = None,
    fft_loss_weight: float | None = None,
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
    if ae_only:
        command.append("--ae-only")
    _append_optional(command, "--layout", layout)
    _append_optional(command, "--ae-architecture", ae_architecture)
    _append_optional(command, "--ae-width-mult", ae_width_mult)
    if coordconv is not None:
        _append_optional(command, "--coordconv", str(coordconv).lower())
    _append_optional(command, "--coarse-loss-weight", coarse_loss_weight)
    _append_optional(command, "--coarse-blur-kernel", coarse_blur_kernel)
    _append_optional(command, "--coarse-blur-sigma", coarse_blur_sigma)
    _append_optional(command, "--refine-blocks", refine_blocks)
    _append_optional(command, "--refine-channels-mult", refine_channels_mult)
    _append_optional(command, "--dynamics-model", dynamics_model)
    _append_optional(command, "--ae-epochs", ae_epochs)
    _append_optional(command, "--dyn-epochs", dyn_epochs)
    _append_optional(command, "--latent-dim", latent_dim)
    _append_optional(command, "--ae-learning-rate", ae_learning_rate)
    _append_optional(command, "--dyn-learning-rate", dyn_learning_rate)
    _append_optional(command, "--rollout-loss-weight", rollout_loss_weight)
    _append_optional(command, "--gradient-loss-weight", gradient_loss_weight)
    _append_optional(command, "--fft-loss-weight", fft_loss_weight)
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


def _search_dir(paths: ProjectPaths, study_name: str) -> Path:
    directory = paths.run_log_dir / "ae_search" / study_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _campaign_dir(paths: ProjectPaths, campaign_name: str) -> Path:
    directory = paths.run_log_dir / "campaigns" / campaign_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _trial_output_tag(study_name: str, trial_number: int) -> str:
    return f"ae-search-{study_name}-trial-{trial_number:03d}"


def _write_study_summary(directory: Path, payload: dict[str, object]) -> None:
    write_json(directory / "study_summary.json", payload)


def _parse_family_cycle(raw: str) -> tuple[str, ...]:
    families = [item.strip() for item in raw.split(",") if item.strip()]
    if not families:
        raise ValueError("Family cycle cannot be empty.")
    invalid = [item for item in families if item not in SEARCH_FAMILIES]
    if invalid:
        raise ValueError(
            f"Unsupported family in cycle: {', '.join(invalid)}. "
            f"Expected one of: {', '.join(SEARCH_FAMILIES)}."
        )
    return tuple(families)


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _git_head_commit(root: Path) -> str:
    return run_git(root, ["rev-parse", "HEAD"]).stdout.strip()


def _git_diff_entries(root: Path) -> list[dict[str, str]]:
    result = run_git(root, ["diff", "--name-status", "HEAD", "--"])
    entries: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith(("R", "C")):
            raise RuntimeError(f"Renames/copies are not supported in candidate mode: {line}")
        if len(parts) < 2:
            continue
        entries.append({"status": status[:1], "path": _normalize_relpath(parts[1])})
    return entries


def _git_untracked_paths(root: Path) -> list[str]:
    result = run_git(root, ["ls-files", "--others", "--exclude-standard"])
    return [_normalize_relpath(line) for line in result.stdout.splitlines() if line.strip()]


def _collect_candidate_changes(paths: ProjectPaths) -> tuple[list[dict[str, str]], list[str]]:
    ignored_prefixes = tuple(f"{path.rstrip('/')}/" for path in paths.runtime_git_ignored_paths if path not in {"results.tsv"})
    ignored_paths = set(paths.runtime_git_ignored_paths)
    entries: list[dict[str, str]] = []
    out_of_scope: list[str] = []

    for item in _git_diff_entries(paths.root):
        relpath = item["path"]
        if relpath in ignored_paths or _is_under_prefix(relpath, ignored_prefixes):
            continue
        if _is_under_prefix(relpath, paths.candidate_code_prefixes):
            entries.append(item)
        else:
            out_of_scope.append(relpath)

    for relpath in _git_untracked_paths(paths.root):
        if relpath in ignored_paths or _is_under_prefix(relpath, ignored_prefixes):
            continue
        if _is_under_prefix(relpath, paths.candidate_code_prefixes):
            entries.append({"status": "A", "path": relpath})
        else:
            out_of_scope.append(relpath)

    deduped_entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in entries:
        key = (item["status"], item["path"])
        if key in seen:
            continue
        seen.add(key)
        deduped_entries.append(item)

    deduped_out_of_scope: list[str] = []
    seen_paths: set[str] = set()
    for relpath in out_of_scope:
        if relpath in seen_paths:
            continue
        seen_paths.add(relpath)
        deduped_out_of_scope.append(relpath)
    return deduped_entries, deduped_out_of_scope


def _git_commit_candidate(root: Path, candidate_paths: list[str], message: str) -> str:
    run_git(root, ["add", "--", *candidate_paths], capture_output=True)
    run_git(root, ["commit", "-m", message], capture_output=True)
    return _git_head_commit(root)


def _restore_discarded_candidate(root: Path, base_commit: str, candidate_entries: list[dict[str, str]]) -> None:
    run_git(root, ["reset", "--mixed", base_commit], capture_output=True)
    added_paths = [item["path"] for item in candidate_entries if item["status"] == "A"]
    restore_paths = [item["path"] for item in candidate_entries if item["status"] != "A"]
    if restore_paths:
        run_git(root, ["restore", "--source=HEAD", "--worktree", "--", *restore_paths], capture_output=True)
    if added_paths:
        run_git(root, ["clean", "-fd", "--", *added_paths], capture_output=True)


def _sample_search_params(trial: object, family: str) -> dict[str, object]:
    import optuna

    del optuna  # imported for type/runtime validation only
    if family == "residual_refine":
        return {
            "layout": "portrait",
            "ae_architecture": family,
            "ae_width_mult": 1.0,
            "coordconv": False,
            "latent_dim": int(trial.suggest_categorical("latent_dim", [24, 32])),
            "gradient_loss_weight": 0.1,
            "fft_loss_weight": 0.0,
            "latent_l1_weight": 1e-4,
            "coarse_loss_weight": float(trial.suggest_float("coarse_loss_weight", 0.15, 0.45)),
            "coarse_blur_kernel": 9,
            "coarse_blur_sigma": 2.0,
            "refine_blocks": int(trial.suggest_categorical("refine_blocks", [1, 2])),
            "refine_channels_mult": float(trial.suggest_categorical("refine_channels_mult", [1.0, 1.25])),
            "ae_learning_rate": float(trial.suggest_float("ae_learning_rate", 5e-4, 1.2e-3, log=True)),
        }
    params = {
        "layout": "portrait",
        "ae_architecture": family,
        "ae_width_mult": float(trial.suggest_float("ae_width_mult", 1.0, 2.0)),
        "coordconv": True if family == "residual_multiscale" else trial.suggest_categorical("coordconv", [False, True]),
        "latent_dim": int(trial.suggest_categorical("latent_dim", [24, 32, 48])),
        "gradient_loss_weight": float(trial.suggest_float("gradient_loss_weight", 0.05, 0.20)),
        "fft_loss_weight": float(trial.suggest_float("fft_loss_weight", 0.0, 0.10)),
        "latent_l1_weight": float(trial.suggest_float("latent_l1_weight", 1e-6, 5e-4, log=True)),
        "ae_learning_rate": float(trial.suggest_float("ae_learning_rate", 3e-4, 2e-3, log=True)),
    }
    return params


def _collect_top_trials(trials: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
    successful = [trial for trial in trials if trial["status"] == "ok"]
    successful.sort(key=lambda item: float(item["ae_score"]))
    return successful[:top_k]


def _enqueue_warm_start_trials(study: object, family: str) -> None:
    baseline_trial = {
        "ae_width_mult": 1.0,
        "latent_dim": 24,
        "gradient_loss_weight": 0.1,
        "fft_loss_weight": 0.0,
        "latent_l1_weight": 1e-4,
        "ae_learning_rate": 1e-3,
    }
    if family == "residual":
        study.enqueue_trial({**baseline_trial, "coordconv": False})
        study.enqueue_trial({**baseline_trial, "coordconv": True})
        return
    if family == "residual_refine":
        study.enqueue_trial(
            {
                **baseline_trial,
                "coordconv": False,
                "coarse_loss_weight": 0.25,
                "coarse_blur_kernel": 9,
                "coarse_blur_sigma": 2.0,
                "refine_blocks": 1,
                "refine_channels_mult": 1.0,
            }
        )
        return
    study.enqueue_trial({**baseline_trial, "coordconv": True})


def _execute_logged_run(
    paths: ProjectPaths,
    config: AutoresearchConfig,
    *,
    command: list[str],
    output_tag: str,
    description: str,
    next_hypothesis: str,
) -> object:
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
        description=description,
        status=status,
        failure_reason=failure_reason,
        next_hypothesis=next_hypothesis,
        metrics_path=str((paths.nonlinear_dir / output_tag / "metrics.json").relative_to(paths.root)),
        log_path=log_path,
    )
    record_experiment(paths, record)
    return record


def _run_search_ae(
    paths: ProjectPaths,
    config: AutoresearchConfig,
    *,
    study_name: str,
    family: str,
    trials: int,
    top_k: int,
    device: str | None,
) -> dict[str, object]:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised in runtime, not unit tests
        raise RuntimeError("Optuna is required for search-ae. Install it in the project environment.") from exc

    study_dir = _search_dir(paths, study_name)
    trial_payloads: list[dict[str, object]] = []
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    _enqueue_warm_start_trials(study, family)

    def objective(trial: optuna.trial.Trial) -> float:
        params = _sample_search_params(trial, family)
        output_tag = _trial_output_tag(study_name, trial.number + 1)
        command = _default_command(
            output_tag,
            False,
            ae_only=True,
            layout=str(params["layout"]),
            ae_architecture=str(params["ae_architecture"]),
            ae_width_mult=float(params["ae_width_mult"]),
            coordconv=bool(params["coordconv"]),
            coarse_loss_weight=float(params.get("coarse_loss_weight", 0.25)),
            coarse_blur_kernel=int(params.get("coarse_blur_kernel", 9)),
            coarse_blur_sigma=float(params.get("coarse_blur_sigma", 2.0)),
            refine_blocks=int(params.get("refine_blocks", 1)),
            refine_channels_mult=float(params.get("refine_channels_mult", 1.0)),
            latent_dim=int(params["latent_dim"]),
            ae_learning_rate=float(params["ae_learning_rate"]),
            gradient_loss_weight=float(params["gradient_loss_weight"]),
            fft_loss_weight=float(params["fft_loss_weight"]),
            latent_l1_weight=float(params["latent_l1_weight"]),
            device=device,
        )
        try:
            metrics, log_path = _run_and_collect(paths, command, output_tag, config.timeout_seconds)
            ae_score = float(metrics["ae_score"])
            payload = {
                "trial_number": trial.number + 1,
                "status": "ok",
                "ae_score": ae_score,
                "recon_rmse": float(metrics["recon_rmse"]),
                "ae_floor_rmse_t100": float(metrics["ae_floor_rmse_t100"]),
                "ae_floor_rmse_t150": float(metrics["ae_floor_rmse_t150"]),
                "metrics_path": str((paths.nonlinear_dir / output_tag / "metrics.json").relative_to(paths.root)),
                "log_path": log_path,
                "params": params,
            }
        except Exception as exc:  # noqa: BLE001
            ae_score = float("inf")
            payload = {
                "trial_number": trial.number + 1,
                "status": "crash",
                "ae_score": ae_score,
                "recon_rmse": 0.0,
                "ae_floor_rmse_t100": 0.0,
                "ae_floor_rmse_t150": 0.0,
                "metrics_path": str((paths.nonlinear_dir / output_tag / "metrics.json").relative_to(paths.root)),
                "log_path": str((paths.run_log_dir / f"{output_tag}.log").relative_to(paths.root)),
                "error": str(exc),
                "params": params,
            }
        trial_payloads.append(payload)
        _write_study_summary(
            study_dir,
            {
                "study_name": study_name,
                "family": family,
                "trials_requested": trials,
                "completed_trials": len(trial_payloads),
                "best_ae_score": min((float(item["ae_score"]) for item in trial_payloads if item["status"] == "ok"), default=None),
                "trials": trial_payloads,
            },
        )
        return ae_score

    study.optimize(objective, n_trials=trials)
    top_trials = _collect_top_trials(trial_payloads, top_k)
    reevaluated: list[dict[str, object]] = []
    for rank, trial_payload in enumerate(top_trials, start=1):
        params = dict(trial_payload["params"])
        experiment_id = len(load_records(paths)) + 1
        output_tag = f"exp-{experiment_id:04d}"
        command = _default_command(
            output_tag,
            False,
            layout=str(params["layout"]),
            ae_architecture=str(params["ae_architecture"]),
            ae_width_mult=float(params["ae_width_mult"]),
            coordconv=bool(params["coordconv"]),
            coarse_loss_weight=float(params.get("coarse_loss_weight", 0.25)),
            coarse_blur_kernel=int(params.get("coarse_blur_kernel", 9)),
            coarse_blur_sigma=float(params.get("coarse_blur_sigma", 2.0)),
            refine_blocks=int(params.get("refine_blocks", 1)),
            refine_channels_mult=float(params.get("refine_channels_mult", 1.0)),
            dynamics_model="residual_linear",
            latent_dim=int(params["latent_dim"]),
            ae_learning_rate=float(params["ae_learning_rate"]),
            gradient_loss_weight=float(params["gradient_loss_weight"]),
            fft_loss_weight=float(params["fft_loss_weight"]),
            latent_l1_weight=float(params["latent_l1_weight"]),
            device=device,
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
            description=(
                f"AE search top-{rank} reevaluation | family={params['ae_architecture']} | "
                f"latent={params['latent_dim']} | width={params['ae_width_mult']:.3f} | "
                f"coordconv={params['coordconv']} | refine_blocks={params.get('refine_blocks', 'n/a')}"
            ),
            status=status,
            failure_reason=failure_reason,
            next_hypothesis="Use the best reevaluated AE candidate as the new reference architecture.",
            metrics_path=str((paths.nonlinear_dir / output_tag / "metrics.json").relative_to(paths.root)),
            log_path=log_path,
        )
        record_experiment(paths, record)
        reevaluated.append(
            {
                "rank": rank,
                "output_tag": output_tag,
                "status": record.status,
                "primary_score": record.primary_score,
                "params": params,
            }
        )
    summary = {
        "study_name": study_name,
        "family": family,
        "trials_requested": trials,
        "completed_trials": len(trial_payloads),
        "best_ae_score": min((float(item["ae_score"]) for item in trial_payloads if item["status"] == "ok"), default=None),
        "trials": trial_payloads,
        "reevaluated_top_candidates": reevaluated,
        "best_reevaluated_score": min((float(item["primary_score"]) for item in reevaluated), default=None),
    }
    _write_study_summary(study_dir, summary)
    return summary


def _run_campaign(
    paths: ProjectPaths,
    config: AutoresearchConfig,
    *,
    campaign_name: str,
    family_cycle: tuple[str, ...],
    trials_per_round: int,
    top_k: int,
    device: str | None,
    max_rounds: int,
    max_hours: float,
    stop_file: Path,
    search_runner=_run_search_ae,
    time_source=time.monotonic,
) -> dict[str, object]:
    campaign_dir = _campaign_dir(paths, campaign_name)
    state_path = campaign_dir / "campaign_state.json"
    started_at = utc_timestamp()
    started_monotonic = time_source()
    rounds: list[dict[str, object]] = []
    status = "running"
    stop_reason = "running"
    round_number = 0

    while True:
        elapsed_seconds = max(0.0, time_source() - started_monotonic)
        if stop_file.exists():
            status = "stopped"
            stop_reason = "stop_file"
            break
        if max_rounds > 0 and round_number >= max_rounds:
            status = "completed"
            stop_reason = "max_rounds"
            break
        if max_hours > 0 and elapsed_seconds >= max_hours * 3600.0:
            status = "completed"
            stop_reason = "max_hours"
            break

        family = family_cycle[round_number % len(family_cycle)]
        study_name = f"{campaign_name}-round-{round_number + 1:03d}-{family}"
        records_before = load_records(paths)
        best_before = best_record(records_before)

        try:
            summary = search_runner(
                paths,
                config,
                study_name=study_name,
                family=family,
                trials=trials_per_round,
                top_k=top_k,
                device=device,
            )
            round_status = "ok"
            error = ""
        except Exception as exc:  # noqa: BLE001
            summary = {
                "study_name": study_name,
                "family": family,
                "completed_trials": 0,
                "reevaluated_top_candidates": [],
                "best_reevaluated_score": None,
            }
            round_status = "crash"
            error = str(exc)
            status = "crash"
            stop_reason = "crash"

        records_after = load_records(paths)
        best_after = best_record(records_after)
        rounds.append(
            {
                "round_number": round_number + 1,
                "family": family,
                "study_name": study_name,
                "status": round_status,
                "error": error,
                "experiments_before": len(records_before),
                "experiments_after": len(records_after),
                "best_score_before": None if best_before is None else best_before.primary_score,
                "best_score_after": None if best_after is None else best_after.primary_score,
                "best_reevaluated_score": summary.get("best_reevaluated_score"),
                "recorded_experiments": max(0, len(records_after) - len(records_before)),
            }
        )
        write_json(
            state_path,
            {
                "campaign_name": campaign_name,
                "run_tag": paths.run_tag,
                "started_at": started_at,
                "updated_at": utc_timestamp(),
                "status": status if status != "running" else "running",
                "family_cycle": list(family_cycle),
                "trials_per_round": trials_per_round,
                "top_k": top_k,
                "max_rounds": max_rounds,
                "max_hours": max_hours,
                "stop_file": _display_path(stop_file, paths.root),
                "stop_reason": stop_reason,
                "elapsed_seconds": elapsed_seconds,
                "rounds_completed": len(rounds),
                "rounds": rounds,
            },
        )
        round_number += 1
        if round_status != "ok":
            break

    final_state = read_json(state_path) if state_path.exists() else {
        "campaign_name": campaign_name,
        "run_tag": paths.run_tag,
        "started_at": started_at,
        "updated_at": utc_timestamp(),
        "status": status,
        "family_cycle": list(family_cycle),
        "trials_per_round": trials_per_round,
        "top_k": top_k,
        "max_rounds": max_rounds,
        "max_hours": max_hours,
        "stop_file": _display_path(stop_file, paths.root),
        "stop_reason": stop_reason,
        "elapsed_seconds": max(0.0, time_source() - started_monotonic),
        "rounds_completed": len(rounds),
        "rounds": rounds,
    }
    final_state["status"] = status
    final_state["stop_reason"] = stop_reason
    final_state["updated_at"] = utc_timestamp()
    final_state["elapsed_seconds"] = max(0.0, time_source() - started_monotonic)
    write_json(state_path, final_state)
    return final_state


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
    run_parser.add_argument(
        "--ae-architecture",
        choices=["baseline", "residual", "residual_refine", "residual_multiscale"],
        default=None,
    )
    run_parser.add_argument("--ae-width-mult", type=float, default=None)
    run_parser.add_argument("--coordconv", choices=["true", "false"], default=None)
    run_parser.add_argument("--coarse-loss-weight", type=float, default=None)
    run_parser.add_argument("--coarse-blur-kernel", type=int, default=None)
    run_parser.add_argument("--coarse-blur-sigma", type=float, default=None)
    run_parser.add_argument("--refine-blocks", type=int, default=None)
    run_parser.add_argument("--refine-channels-mult", type=float, default=None)
    run_parser.add_argument("--dynamics-model", choices=["mlp", "residual_linear"], default=None)
    run_parser.add_argument("--ae-epochs", type=int, default=None)
    run_parser.add_argument("--dyn-epochs", type=int, default=None)
    run_parser.add_argument("--latent-dim", type=int, default=None)
    run_parser.add_argument("--ae-learning-rate", type=float, default=None)
    run_parser.add_argument("--dyn-learning-rate", type=float, default=None)
    run_parser.add_argument("--rollout-loss-weight", type=float, default=None)
    run_parser.add_argument("--gradient-loss-weight", type=float, default=None)
    run_parser.add_argument("--fft-loss-weight", type=float, default=None)
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

    candidate_parser = subparsers.add_parser(
        "apply-candidate",
        help="Commit the current candidate code change, run it once, and automatically keep or discard it.",
    )
    candidate_parser.add_argument("--description", required=True)
    candidate_parser.add_argument("--idea-key", required=True)
    candidate_parser.add_argument("--run-tag", default=ProjectPaths().run_tag)
    candidate_parser.add_argument("--next-hypothesis", default="")
    candidate_parser.add_argument("--commit-message", default=None)
    candidate_parser.add_argument("--force-revisit-reason", default="")
    candidate_parser.add_argument("--smoke", action="store_true")
    candidate_parser.add_argument("--device", default=None)
    candidate_parser.add_argument(
        "--allow-clean",
        action="store_true",
        help="Allow a clean-tree baseline run without creating a candidate commit.",
    )

    search_parser = subparsers.add_parser("search-ae", help="Run an Optuna/TPE AE-only search and full-evaluate the top candidates.")
    search_parser.add_argument("--run-tag", default=ProjectPaths().run_tag)
    search_parser.add_argument("--study-name", default="residual-multiscale")
    search_parser.add_argument(
        "--family",
        choices=SEARCH_FAMILIES,
        default="residual_multiscale",
    )
    search_parser.add_argument("--trials", type=int, default=12)
    search_parser.add_argument("--top-k", type=int, default=3)
    search_parser.add_argument("--device", default=None)

    campaign_parser = subparsers.add_parser(
        "run-campaign",
        help="Run unattended AE-search rounds until max rounds are reached or a stop file appears.",
    )
    campaign_parser.add_argument("--run-tag", default=ProjectPaths().run_tag)
    campaign_parser.add_argument("--campaign-name", default="autonomous-ae")
    campaign_parser.add_argument("--families", default="residual_refine")
    campaign_parser.add_argument("--trials-per-round", type=int, default=8)
    campaign_parser.add_argument("--top-k", type=int, default=2)
    campaign_parser.add_argument("--max-rounds", type=int, default=0, help="0 means run until a stop file is created.")
    campaign_parser.add_argument("--max-hours", type=float, default=0.0, help="0 disables the wall-clock campaign limit.")
    campaign_parser.add_argument("--device", default=None)
    campaign_parser.add_argument("--stop-file", default=None, help="Optional explicit path for a campaign stop file.")

    args = parser.parse_args()
    paths = ProjectPaths(run_tag=args.run_tag)
    paths.ensure_directories()
    init_results_file(paths)
    ensure_registry_files(paths)

    if args.command == "init-run":
        print(f"Initialized run scaffold for {paths.run_tag}")
        print(f"Expected branch: {paths.branch_name}")
        return

    if args.command == "apply-candidate":
        config = AutoresearchConfig(run_tag=paths.run_tag)
        registry = load_registry(paths)
        check_idea_allowed(registry, args.idea_key, force_revisit_reason=args.force_revisit_reason)
        candidate_entries, out_of_scope = _collect_candidate_changes(paths)
        if out_of_scope:
            raise RuntimeError(
                "Candidate mode found modified files outside src/tests and outside runtime memory paths: "
                + ", ".join(out_of_scope)
            )
        command = _default_command(f"exp-{len(load_records(paths)) + 1:04d}", args.smoke, device=args.device)
        if not candidate_entries:
            if not args.allow_clean:
                raise RuntimeError("No candidate code changes found under src/ or tests/.")
            record = _execute_logged_run(
                paths,
                config,
                command=command,
                output_tag=f"exp-{len(load_records(paths)) + 1:04d}",
                description=f"[idea:{args.idea_key}] {args.description}",
                next_hypothesis=args.next_hypothesis,
            )
            print(f"Recorded exp-{record.experiment_id:04d}")
            print(f"Status: {record.status}")
            print(f"Primary score: {record.primary_score:.6f}")
            return

        base_commit = _git_head_commit(paths.root)
        candidate_paths = [item["path"] for item in candidate_entries]
        commit_message = args.commit_message or args.description
        candidate_commit = _git_commit_candidate(paths.root, candidate_paths, commit_message)
        output_tag = f"exp-{len(load_records(paths)) + 1:04d}"
        command = _default_command(output_tag, args.smoke, device=args.device)
        record = _execute_logged_run(
            paths,
            config,
            command=command,
            output_tag=output_tag,
            description=f"[idea:{args.idea_key}] {args.description}",
            next_hypothesis=(
                args.next_hypothesis
                if not args.force_revisit_reason.strip()
                else f"{args.next_hypothesis} | revisit_reason={args.force_revisit_reason}".strip(" |")
            ),
        )
        if record.status != "keep":
            _restore_discarded_candidate(paths.root, base_commit, candidate_entries)
            print(f"Discarded candidate {candidate_commit[:7]} and restored code to {base_commit[:7]}")
        else:
            print(f"Kept candidate commit {candidate_commit[:7]}")
        print(f"Recorded exp-{record.experiment_id:04d}")
        print(f"Status: {record.status}")
        print(f"Primary score: {record.primary_score:.6f}")
        return

    if args.command == "search-ae":
        config = AutoresearchConfig(run_tag=paths.run_tag)
        summary = _run_search_ae(
            paths,
            config,
            study_name=args.study_name,
            family=args.family,
            trials=args.trials,
            top_k=args.top_k,
            device=args.device,
        )
        print(f"Completed AE search: {args.study_name}")
        print(f"Completed trials: {summary['completed_trials']}")
        if summary["best_reevaluated_score"] is not None:
            print(f"Best reevaluated score: {summary['best_reevaluated_score']:.6f}")
        return

    if args.command == "run-campaign":
        config = AutoresearchConfig(run_tag=paths.run_tag)
        family_cycle = _parse_family_cycle(args.families)
        stop_file = Path(args.stop_file).resolve() if args.stop_file else (_campaign_dir(paths, args.campaign_name) / "STOP")
        state = _run_campaign(
            paths,
            config,
            campaign_name=args.campaign_name,
            family_cycle=family_cycle,
            trials_per_round=args.trials_per_round,
            top_k=args.top_k,
            device=args.device,
            max_rounds=args.max_rounds,
            max_hours=args.max_hours,
            stop_file=stop_file,
        )
        print(f"Campaign: {args.campaign_name}")
        print(f"Status: {state['status']}")
        print(f"Stop reason: {state['stop_reason']}")
        print(f"Rounds completed: {state['rounds_completed']}")
        print(f"Stop file: {state['stop_file']}")
        return

    config = AutoresearchConfig(run_tag=paths.run_tag)
    experiment_id = len(load_records(paths)) + 1
    output_tag = f"exp-{experiment_id:04d}"
    command = _default_command(
        output_tag,
        args.smoke,
        layout=args.layout,
        ae_architecture=args.ae_architecture,
        ae_width_mult=args.ae_width_mult,
        coordconv=None if args.coordconv is None else args.coordconv == "true",
        coarse_loss_weight=args.coarse_loss_weight,
        coarse_blur_kernel=args.coarse_blur_kernel,
        coarse_blur_sigma=args.coarse_blur_sigma,
        refine_blocks=args.refine_blocks,
        refine_channels_mult=args.refine_channels_mult,
        dynamics_model=args.dynamics_model,
        ae_epochs=args.ae_epochs,
        dyn_epochs=args.dyn_epochs,
        latent_dim=args.latent_dim,
        ae_learning_rate=args.ae_learning_rate,
        dyn_learning_rate=args.dyn_learning_rate,
        rollout_loss_weight=args.rollout_loss_weight,
        gradient_loss_weight=args.gradient_loss_weight,
        fft_loss_weight=args.fft_loss_weight,
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
    record = _execute_logged_run(
        paths,
        config,
        command=command,
        output_tag=output_tag,
        description=args.description,
        next_hypothesis=args.next_hypothesis,
    )
    print(f"Recorded exp-{record.experiment_id:04d}")
    print(f"Status: {record.status}")
    print(f"Primary score: {record.primary_score:.6f}")


if __name__ == "__main__":
    main()
