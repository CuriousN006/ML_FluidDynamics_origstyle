from __future__ import annotations

from pathlib import Path

import pytest

from mlfd.autoresearch import _parse_family_cycle, _run_campaign
from mlfd.config import AutoresearchConfig, ProjectPaths
from mlfd.experiments import build_record, init_results_file, load_records, record_experiment


def test_parse_family_cycle_validates_input() -> None:
    assert _parse_family_cycle("residual_refine, residual") == ("residual_refine", "residual")
    with pytest.raises(ValueError):
        _parse_family_cycle("")
    with pytest.raises(ValueError):
        _parse_family_cycle("unknown")


def test_run_campaign_stops_immediately_when_stop_file_exists(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path, run_tag="unit-test")
    paths.ensure_directories()
    init_results_file(paths)
    stop_file = tmp_path / "STOP"
    stop_file.write_text("", encoding="utf-8")

    called = 0

    def fake_search_runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called += 1
        return {}

    state = _run_campaign(
        paths,
        AutoresearchConfig(run_tag="unit-test"),
        campaign_name="demo",
        family_cycle=("residual_refine",),
        trials_per_round=1,
        top_k=1,
        device=None,
        max_rounds=0,
        max_hours=0.0,
        stop_file=stop_file,
        search_runner=fake_search_runner,
    )

    assert called == 0
    assert state["status"] == "stopped"
    assert state["stop_reason"] == "stop_file"
    assert state["rounds_completed"] == 0


def test_run_campaign_loops_until_max_rounds(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path, run_tag="unit-test")
    paths.ensure_directories()
    init_results_file(paths)

    def fake_search_runner(
        paths: ProjectPaths,
        config: AutoresearchConfig,
        *,
        study_name: str,
        family: str,
        trials: int,
        top_k: int,
        device: str | None,
    ) -> dict[str, object]:
        del family, trials, top_k, device
        metrics = {
            "primary_score": 0.50 - 0.05 * len(load_records(paths)),
            "recon_rmse": 0.10,
            "rmse_t100": 0.20,
            "rmse_t150": 0.30,
            "peak_memory_gb": 1.0,
            "wall_seconds": 12.0,
        }
        experiment_id = len(load_records(paths)) + 1
        output_tag = f"exp-{experiment_id:04d}"
        record = build_record(
            paths,
            config,
            metrics,
            description=f"stub from {study_name}",
            metrics_path=f"output/nonlinear/{output_tag}/metrics.json",
            log_path=f"output/autoresearch/{paths.run_tag}/{output_tag}.log",
        )
        record_experiment(paths, record)
        return {
            "study_name": study_name,
            "family": "residual_refine",
            "completed_trials": 1,
            "best_reevaluated_score": record.primary_score,
            "reevaluated_top_candidates": [{"output_tag": output_tag, "primary_score": record.primary_score}],
        }

    state = _run_campaign(
        paths,
        AutoresearchConfig(run_tag="unit-test"),
        campaign_name="demo",
        family_cycle=("residual_refine", "residual"),
        trials_per_round=1,
        top_k=1,
        device=None,
        max_rounds=2,
        max_hours=0.0,
        stop_file=tmp_path / "missing-stop-file",
        search_runner=fake_search_runner,
    )

    assert state["status"] == "completed"
    assert state["stop_reason"] == "max_rounds"
    assert state["rounds_completed"] == 2
    assert len(state["rounds"]) == 2
    assert load_records(paths)[-1].primary_score == pytest.approx(0.45)


def test_run_campaign_stops_when_max_hours_is_reached(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path, run_tag="unit-test")
    paths.ensure_directories()
    init_results_file(paths)

    called = 0
    clock_values = iter([0.0, 0.0, 1.1, 1.1, 1.1])

    def fake_clock() -> float:
        return next(clock_values)

    def fake_search_runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called += 1
        return {"best_reevaluated_score": None}

    state = _run_campaign(
        paths,
        AutoresearchConfig(run_tag="unit-test"),
        campaign_name="demo",
        family_cycle=("residual_refine",),
        trials_per_round=1,
        top_k=1,
        device=None,
        max_rounds=0,
        max_hours=1.0 / 3600.0,
        stop_file=tmp_path / "missing-stop-file",
        search_runner=fake_search_runner,
        time_source=fake_clock,
    )

    assert called == 1
    assert state["status"] == "completed"
    assert state["stop_reason"] == "max_hours"
    assert state["rounds_completed"] == 1
