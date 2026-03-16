from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlfd.config import ProjectPaths
from mlfd.data import load_field_bundle


def _ensure_results_header(path: Path) -> None:
    if path.exists():
        return
    path.write_text("commit\tprimary_score\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")


def main() -> None:
    paths = ProjectPaths(root=ROOT)
    paths.ensure_directories()
    _ensure_results_header(ROOT / "results.tsv")
    bundle = load_field_bundle("VORTALL", paths, layout="portrait")
    print(f"Root: {ROOT}")
    print(f"Data file: {paths.data_file}")
    print(f"Snapshots: {bundle.num_snapshots}")
    print(f"Frame shape: {bundle.frames.shape[1]}x{bundle.frames.shape[2]}")
    print("Setup: ok")


if __name__ == "__main__":
    main()
