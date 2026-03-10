from __future__ import annotations

import csv
from dataclasses import dataclass, field

from .config import AutoresearchConfig, ProjectPaths
from .utils import short_git_commit, utc_timestamp


RESULT_FIELDS = [
    "commit",
    "primary_score",
    "recon_rmse",
    "rmse_t100",
    "rmse_t150",
    "memory_gb",
    "wall_seconds",
    "status",
    "description",
]


@dataclass
class ExperimentRecord:
    experiment_id: int
    commit: str
    primary_score: float
    recon_rmse: float
    rmse_t100: float
    rmse_t150: float
    memory_gb: float
    wall_seconds: float
    status: str
    description: str
    metrics_path: str
    log_path: str
    run_tag: str
    created_at: str
    failure_reason: str = ""
    next_hypothesis: str = ""
    config_summary: tuple[str, ...] = field(default_factory=tuple)


def init_results_file(paths: ProjectPaths) -> None:
    if paths.results_tsv.exists():
        return
    with paths.results_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()


def load_records(paths: ProjectPaths) -> list[ExperimentRecord]:
    if not paths.results_tsv.exists():
        return []
    records: list[ExperimentRecord] = []
    with paths.results_tsv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            records.append(
                ExperimentRecord(
                    experiment_id=idx,
                    commit=row["commit"],
                    primary_score=float(row["primary_score"]),
                    recon_rmse=float(row["recon_rmse"]),
                    rmse_t100=float(row["rmse_t100"]),
                    rmse_t150=float(row["rmse_t150"]),
                    memory_gb=float(row["memory_gb"]),
                    wall_seconds=float(row["wall_seconds"]),
                    status=row["status"],
                    description=row["description"],
                    metrics_path="",
                    log_path="",
                    run_tag=paths.run_tag,
                    created_at="",
                )
            )
    return records


def best_record(records: list[ExperimentRecord]) -> ExperimentRecord | None:
    kept = [record for record in records if record.status == "keep"]
    if kept:
        return min(kept, key=lambda record: record.primary_score)
    non_crash = [record for record in records if record.status != "crash"]
    if non_crash:
        return min(non_crash, key=lambda record: record.primary_score)
    return None


def decide_status(
    candidate_score: float,
    candidate_memory_gb: float,
    records: list[ExperimentRecord],
    config: AutoresearchConfig,
) -> str:
    if not records:
        return "keep"
    best = best_record(records)
    if best is None:
        return "keep"
    if candidate_score < best.primary_score * (1.0 - config.comparison_tolerance):
        return "keep"
    if abs(candidate_score - best.primary_score) <= best.primary_score * config.comparison_tolerance:
        if candidate_memory_gb <= best.memory_gb * (1.0 - config.vram_reduction_threshold):
            return "keep"
    return "discard"


def append_result(paths: ProjectPaths, record: ExperimentRecord) -> None:
    init_results_file(paths)
    with paths.results_tsv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writerow(
            {
                "commit": record.commit,
                "primary_score": f"{record.primary_score:.6f}",
                "recon_rmse": f"{record.recon_rmse:.6f}",
                "rmse_t100": f"{record.rmse_t100:.6f}",
                "rmse_t150": f"{record.rmse_t150:.6f}",
                "memory_gb": f"{record.memory_gb:.3f}",
                "wall_seconds": f"{record.wall_seconds:.1f}",
                "status": record.status,
                "description": record.description,
            }
        )


def write_experiment_markdown(paths: ProjectPaths, record: ExperimentRecord) -> None:
    path = paths.experiment_dir / f"exp-{record.experiment_id:04d}.md"
    if record.config_summary:
        changes_block = "\n".join(f"- {line}" for line in record.config_summary)
    else:
        changes_block = (
            "- This entry records the current code snapshot and its measured metrics.\n"
            "- Summarize the code or config changes for the next run beneath this bullet list."
        )
    content = f"""# Experiment {record.experiment_id:04d}

## Metadata

- Created at: {record.created_at}
- Run tag: {record.run_tag}
- Commit: {record.commit}
- Status: {record.status}

## Purpose

{record.description}

## Changes

{changes_block}

## Execution Conditions

- Metrics file: `{record.metrics_path}`
- Log file: `{record.log_path}`
- Wall time (s): {record.wall_seconds:.1f}
- Peak GPU memory (GB): {record.memory_gb:.3f}

## Key Metrics

- Primary score: {record.primary_score:.6f}
- Reconstruction RMSE: {record.recon_rmse:.6f}
- Rollout RMSE @ t=100: {record.rmse_t100:.6f}
- Rollout RMSE @ t=150: {record.rmse_t150:.6f}

## Keep/Discard Decision

- Decision: {record.status}
- Reason: {"Run crashed or timed out." if record.status == "crash" else "Compared against the current best experiment using the keep/discard policy."}

## Failure Analysis

{record.failure_reason or "No failure recorded."}

## Next Hypothesis

{record.next_hypothesis or "Choose the next hypothesis after comparing this run to the best current result."}
"""
    path.write_text(content, encoding="utf-8")


def refresh_state(paths: ProjectPaths) -> None:
    records = load_records(paths)
    best = best_record(records)
    recent = records[-5:]
    lines = ["# Research State", "", "## Current Best", ""]
    if best is None:
        lines.extend(
            [
                "- Experiment: not started",
                "- Commit: n/a",
                "- Primary score: n/a",
                "- Notes: Run the baseline nonlinear experiment.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Experiment: exp-{best.experiment_id:04d}",
                f"- Commit: {best.commit}",
                f"- Primary score: {best.primary_score:.6f}",
                f"- Notes: {best.description}",
            ]
        )
    lines.extend(["", "## Recent Experiments", ""])
    if recent:
        for record in recent:
            lines.append(
                f"- exp-{record.experiment_id:04d}: {record.status}, score={record.primary_score:.6f}, {record.description}"
            )
    else:
        lines.append("- None yet.")
    lines.extend(["", "## Avoid Repeating", "", "- Review discarded runs before retrying the same idea.", "", "## Next Priorities", ""])
    if best is None:
        lines.extend(
            [
                "1. Run the nonlinear baseline.",
                "2. Record the baseline as exp-0001.",
                "3. Compare follow-up changes against the baseline.",
            ]
        )
    else:
        lines.extend(
            [
                "1. Inspect the latest discarded run and extract the lesson.",
                "2. Try the next highest-value architecture or regularization change.",
                "3. Update the long-form report after a meaningful improvement.",
            ]
        )
    paths.state_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_run_index(paths: ProjectPaths) -> None:
    records = load_records(paths)
    best = best_record(records)
    lines = [f"# Run {paths.run_tag}", "", "## Timeline", ""]
    if records:
        for record in records:
            lines.append(
                f"- exp-{record.experiment_id:04d} | {record.status} | score={record.primary_score:.6f} | {record.description}"
            )
    else:
        lines.append("- Pending initialization.")
    lines.extend(["", "## Best Model Changes", ""])
    if best is None:
        lines.append("- None yet.")
    else:
        lines.append(f"- Current best: exp-{best.experiment_id:04d} ({best.primary_score:.6f})")
    lines.extend(["", "## Open Risks", ""])
    if any(record.status == "crash" for record in records):
        lines.append("- Recent crashes or timeouts need follow-up.")
    else:
        lines.append("- Continue searching for lower rollout error without increasing complexity too much.")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    (paths.run_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_record(
    paths: ProjectPaths,
    config: AutoresearchConfig,
    metrics: dict[str, float],
    description: str,
    status: str | None = None,
    failure_reason: str = "",
    next_hypothesis: str = "",
    metrics_path: str = "",
    log_path: str = "",
) -> ExperimentRecord:
    records = load_records(paths)
    decided_status = status or decide_status(metrics["primary_score"], metrics["peak_memory_gb"], records, config)
    return ExperimentRecord(
        experiment_id=len(records) + 1,
        commit=short_git_commit(paths.root),
        primary_score=float(metrics["primary_score"]),
        recon_rmse=float(metrics["recon_rmse"]),
        rmse_t100=float(metrics["rmse_t100"]),
        rmse_t150=float(metrics["rmse_t150"]),
        memory_gb=float(metrics["peak_memory_gb"]),
        wall_seconds=float(metrics["wall_seconds"]),
        status=decided_status,
        description=description,
        metrics_path=metrics_path,
        log_path=log_path,
        run_tag=paths.run_tag,
        created_at=utc_timestamp(),
        failure_reason=failure_reason,
        next_hypothesis=next_hypothesis,
        config_summary=_summarize_config(metrics),
    )


def record_experiment(paths: ProjectPaths, record: ExperimentRecord) -> None:
    append_result(paths, record)
    write_experiment_markdown(paths, record)
    refresh_state(paths)
    refresh_run_index(paths)


def _summarize_config(metrics: dict[str, float]) -> tuple[str, ...]:
    config = metrics.get("config", {})
    if not config:
        return ()
    summary = [
        f"field_name={config.get('field_name', 'n/a')}",
        f"layout={config.get('layout', 'n/a')}",
        f"ae_architecture={config.get('ae_architecture', 'n/a')}",
        f"ae_width_mult={config.get('ae_width_mult', 'n/a')}",
        f"coordconv={config.get('coordconv', 'n/a')}",
        f"dynamics_model={config.get('dynamics_model', 'n/a')}",
        f"latent_dim={config.get('latent_dim', 'n/a')}",
        f"ae_epochs={config.get('ae_epochs', 'n/a')}",
        f"dyn_epochs={config.get('dyn_epochs', 'n/a')}",
        f"ae_learning_rate={config.get('ae_learning_rate', 'n/a')}",
        f"dyn_learning_rate={config.get('dyn_learning_rate', 'n/a')}",
        f"ae_scheduler={config.get('ae_scheduler', 'n/a')}",
        f"dyn_scheduler={config.get('dyn_scheduler', 'n/a')}",
        f"latent_l1_weight={config.get('latent_l1_weight', 'n/a')}",
        f"dyn_l2_weight={config.get('dyn_l2_weight', 'n/a')}",
        f"gradient_loss_weight={config.get('gradient_loss_weight', 'n/a')}",
        f"fft_loss_weight={config.get('fft_loss_weight', 'n/a')}",
        f"rollout_loss_weight={config.get('rollout_loss_weight', 'n/a')}",
        f"train_rollout_stride={config.get('train_rollout_stride', 'n/a')}",
        f"train_rollout_horizon={config.get('train_rollout_horizon', 'n/a')}",
        f"validation_rollout_stride={config.get('validation_rollout_stride', 'n/a')}",
        f"validation_rollout_horizon={config.get('validation_rollout_horizon', 'n/a')}",
        f"deterministic={config.get('deterministic', 'n/a')}",
    ]
    if config.get("ae_scheduler") == "plateau":
        summary.extend(
            [
                f"ae_scheduler_factor={config.get('ae_scheduler_factor', 'n/a')}",
                f"ae_scheduler_patience={config.get('ae_scheduler_patience', 'n/a')}",
                f"ae_min_learning_rate={config.get('ae_min_learning_rate', 'n/a')}",
            ]
        )
    if config.get("dyn_scheduler") == "plateau":
        summary.extend(
            [
                f"dyn_scheduler_factor={config.get('dyn_scheduler_factor', 'n/a')}",
                f"dyn_scheduler_patience={config.get('dyn_scheduler_patience', 'n/a')}",
                f"dyn_min_learning_rate={config.get('dyn_min_learning_rate', 'n/a')}",
            ]
        )
    return tuple(summary)
