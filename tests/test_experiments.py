from __future__ import annotations

from pathlib import Path

from mlfd.config import AutoresearchConfig, ProjectPaths
from mlfd.experiments import (
    _extract_idea_key_and_description,
    _format_experiment_label,
    ExperimentRecord,
    build_record,
    init_results_file,
    load_records,
    record_experiment,
    refresh_run_index,
    refresh_state,
    write_experiment_markdown,
)


def test_extract_idea_key_and_description_parses_prefixed_description() -> None:
    idea_key, description = _extract_idea_key_and_description(
        "[idea:residual_refine_decoder_conditioning_v1] add refine-path conditioning"
    )

    assert idea_key == "residual_refine_decoder_conditioning_v1"
    assert description == "add refine-path conditioning"


def test_extract_idea_key_and_description_leaves_plain_description_unchanged() -> None:
    idea_key, description = _extract_idea_key_and_description("baseline import check")

    assert idea_key is None
    assert description == "baseline import check"


def test_format_experiment_label_includes_idea_key_when_available() -> None:
    record = ExperimentRecord(
        experiment_id=61,
        commit="abc1234",
        primary_score=0.5,
        recon_rmse=0.1,
        rmse_t100=0.2,
        rmse_t150=0.3,
        memory_gb=1.0,
        wall_seconds=12.0,
        status="keep",
        description="[idea:residual_refine_decoder_conditioning_v1] add refine-path conditioning",
        metrics_path="metrics.json",
        log_path="run.log",
        run_tag="unit-test",
        created_at="2026-03-13T00:00:00Z",
    )

    assert (
        _format_experiment_label(record)
        == "exp-0061 | residual_refine_decoder_conditioning_v1 | add refine-path conditioning"
    )


def test_results_file_and_markdown(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path, run_tag="unit-test")
    paths.ensure_directories()
    init_results_file(paths)
    metrics = {
        "primary_score": 0.123,
        "recon_rmse": 0.11,
        "rmse_t100": 0.22,
        "rmse_t150": 0.33,
        "peak_memory_gb": 1.5,
        "wall_seconds": 12.3,
    }
    record = build_record(
        paths,
        AutoresearchConfig(run_tag="unit-test"),
        metrics,
        description="unit baseline",
        metrics_path="output/nonlinear/exp-0001/metrics.json",
        log_path="output/autoresearch/unit-test/exp-0001.log",
    )
    record_experiment(paths, record)
    records = load_records(paths)
    assert len(records) == 1
    assert records[0].status == "keep"
    assert (paths.experiment_dir / "exp-0001.md").exists()
    assert paths.state_md.exists()
    assert (paths.run_dir / "index.md").exists()


def test_experiment_markdown_shows_idea_key_and_clean_description(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path, run_tag="unit-test")
    paths.ensure_directories()
    record = ExperimentRecord(
        experiment_id=7,
        commit="abc1234",
        primary_score=0.123,
        recon_rmse=0.111,
        rmse_t100=0.222,
        rmse_t150=0.333,
        memory_gb=1.5,
        wall_seconds=12.3,
        status="keep",
        description="[idea:residual_refine_decoder_conditioning_v1] add refine-path conditioning",
        metrics_path="output/nonlinear/exp-0007/metrics.json",
        log_path="output/autoresearch/unit-test/exp-0007.log",
        run_tag="unit-test",
        created_at="2026-03-13T00:00:00Z",
    )

    write_experiment_markdown(paths, record)

    content = (paths.experiment_dir / "exp-0007.md").read_text(encoding="utf-8")
    assert "# Experiment 0007" in content
    assert "- Idea key: residual_refine_decoder_conditioning_v1" in content
    assert "## Purpose\n\nadd refine-path conditioning" in content
    assert "[idea:residual_refine_decoder_conditioning_v1]" not in content


def test_state_and_run_index_use_readable_experiment_labels(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path, run_tag="unit-test")
    paths.ensure_directories()
    init_results_file(paths)
    metrics = {
        "primary_score": 0.123,
        "recon_rmse": 0.11,
        "rmse_t100": 0.22,
        "rmse_t150": 0.33,
        "peak_memory_gb": 1.5,
        "wall_seconds": 12.3,
    }
    keyed_record = build_record(
        paths,
        AutoresearchConfig(run_tag="unit-test"),
        metrics,
        description="[idea:residual_refine_decoder_conditioning_v1] add refine-path conditioning",
        metrics_path="output/nonlinear/exp-0001/metrics.json",
        log_path="output/autoresearch/unit-test/exp-0001.log",
    )
    record_experiment(paths, keyed_record)

    plain_record = build_record(
        paths,
        AutoresearchConfig(run_tag="unit-test"),
        {
            **metrics,
            "primary_score": 0.124,
        },
        description="baseline import check",
        metrics_path="output/nonlinear/exp-0002/metrics.json",
        log_path="output/autoresearch/unit-test/exp-0002.log",
    )
    record_experiment(paths, plain_record)

    refresh_state(paths)
    refresh_run_index(paths)

    state_content = paths.state_md.read_text(encoding="utf-8")
    index_content = (paths.run_dir / "index.md").read_text(encoding="utf-8")

    assert "Generated short-term memory." in state_content
    assert "- Idea key: residual_refine_decoder_conditioning_v1" in state_content
    assert (
        "- exp-0001 | residual_refine_decoder_conditioning_v1 | add refine-path conditioning | keep | score=0.123000"
        in state_content
    )
    assert "- exp-0002 | baseline import check | discard | score=0.124000" in state_content
    assert "Generated campaign timeline." in index_content
    assert (
        "- exp-0001 | residual_refine_decoder_conditioning_v1 | add refine-path conditioning | keep | score=0.123000"
        in index_content
    )
    assert "- exp-0002 | baseline import check | discard | score=0.124000" in index_content
    assert (
        "- Current best: exp-0001 | residual_refine_decoder_conditioning_v1 | add refine-path conditioning (0.123000)"
        in index_content
    )
