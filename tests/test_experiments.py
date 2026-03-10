from __future__ import annotations

from pathlib import Path

from mlfd.config import AutoresearchConfig, ProjectPaths
from mlfd.experiments import build_record, init_results_file, load_records, record_experiment


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
