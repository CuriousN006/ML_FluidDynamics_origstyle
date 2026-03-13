from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from mlfd.autoresearch import (
    _collect_candidate_changes,
    _git_commit_candidate,
    _git_head_commit,
    _parse_family_cycle,
    _restore_discarded_candidate,
    _run_campaign,
)
from mlfd.config import AutoresearchConfig, ProjectPaths
from mlfd.experiments import build_record, init_results_file, load_records, record_experiment
from mlfd.idea_registry import blocked_lines, check_idea_allowed, load_registry
from mlfd.utils import short_git_commit


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True, capture_output=True, text=True)


def test_parse_family_cycle_validates_input() -> None:
    assert _parse_family_cycle("residual_refine, residual") == ("residual_refine", "residual")
    with pytest.raises(ValueError):
        _parse_family_cycle("")
    with pytest.raises(ValueError):
        _parse_family_cycle("unknown")


def test_collect_candidate_changes_ignores_runtime_memory(tmp_path: Path) -> None:
    (tmp_path / "CYLINDER_ALL.mat").write_text("stub", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='tmp'\nversion='0.0.0'\n", encoding="utf-8")
    (tmp_path / "src" / "mlfd").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "research" / "experiments").mkdir(parents=True)
    (tmp_path / "output").mkdir()
    (tmp_path / "src" / "mlfd" / "models.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "results.tsv").write_text("commit\n", encoding="utf-8")
    _init_git_repo(tmp_path)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)

    (tmp_path / "src" / "mlfd" / "models.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "research" / "state.md").write_text("runtime note\n", encoding="utf-8")
    (tmp_path / "results.tsv").write_text("runtime ledger\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("oops\n", encoding="utf-8")

    paths = ProjectPaths(root=tmp_path, run_tag="unit-test")
    candidate_entries, out_of_scope = _collect_candidate_changes(paths)

    assert candidate_entries == [{"status": "M", "path": "src/mlfd/models.py"}]
    assert out_of_scope == ["README.md"]


def test_restore_discarded_candidate_keeps_runtime_logs(tmp_path: Path) -> None:
    (tmp_path / "CYLINDER_ALL.mat").write_text("stub", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='tmp'\nversion='0.0.0'\n", encoding="utf-8")
    (tmp_path / "src" / "mlfd").mkdir(parents=True)
    (tmp_path / "research").mkdir()
    (tmp_path / "src" / "mlfd" / "models.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "results.tsv").write_text("commit\n", encoding="utf-8")
    _init_git_repo(tmp_path)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)

    base_commit = _git_head_commit(tmp_path)
    (tmp_path / "src" / "mlfd" / "models.py").write_text("x = 2\n", encoding="utf-8")
    candidate_entries = [{"status": "M", "path": "src/mlfd/models.py"}]
    _git_commit_candidate(tmp_path, ["src/mlfd/models.py"], "candidate")
    (tmp_path / "results.tsv").write_text("new result\n", encoding="utf-8")

    _restore_discarded_candidate(tmp_path, base_commit, candidate_entries)

    assert (tmp_path / "src" / "mlfd" / "models.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (tmp_path / "results.tsv").read_text(encoding="utf-8") == "new result\n"
    assert _git_head_commit(tmp_path) == base_commit


def test_short_git_commit_ignores_runtime_paths(tmp_path: Path) -> None:
    (tmp_path / "CYLINDER_ALL.mat").write_text("stub", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='tmp'\nversion='0.0.0'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "results.tsv").write_text("commit\n", encoding="utf-8")
    _init_git_repo(tmp_path)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)

    commit = short_git_commit(tmp_path)
    (tmp_path / "results.tsv").write_text("dirty\n", encoding="utf-8")

    assert short_git_commit(tmp_path).endswith("-dirty")
    assert short_git_commit(tmp_path, ignored_paths=("results.tsv",)) == commit


def test_registry_blocks_known_idea(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path, run_tag="unit-test")
    paths.ensure_directories()
    registry = load_registry(paths)

    with pytest.raises(RuntimeError):
        check_idea_allowed(registry, "phase_refine_phase_residual_v1")


def test_registry_blocked_lines_include_phase_branch(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path, run_tag="unit-test")
    paths.ensure_directories()
    registry = load_registry(paths)
    lines = blocked_lines(registry)

    assert any("phase_refine_phase_residual_v1" in line for line in lines)


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
