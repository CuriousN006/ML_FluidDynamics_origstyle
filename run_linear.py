from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlfd.config import LinearConfig, ProjectPaths
from mlfd.linear import run_linear_pipeline


def build_config(*, smoke: bool) -> LinearConfig:
    config = LinearConfig()
    return config.smoke() if smoke else config


def print_summary(metrics: dict[str, float | int]) -> None:
    print("---")
    print(f"best_rank:      {int(metrics['best_rank'])}")
    print(f"primary_score:  {float(metrics['primary_score']):.6f}")
    print(f"recon_rmse:     {float(metrics['recon_rmse']):.6f}")
    print(f"rmse_t100:      {float(metrics['rmse_t100']):.6f}")
    print(f"rmse_t150:      {float(metrics['rmse_t150']):.6f}")
    print(f"wall_seconds:   {float(metrics['wall_seconds']):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local DMD baseline under the active evaluation protocol.")
    parser.add_argument("--output-tag", default="baseline", help="Subdirectory name under output/linear.")
    parser.add_argument("--smoke", action="store_true", help="Run a smaller rank sweep for a quick verification.")
    args = parser.parse_args()

    metrics = run_linear_pipeline(build_config(smoke=args.smoke), ProjectPaths(root=ROOT), output_tag=args.output_tag)
    print_summary(metrics)


if __name__ == "__main__":
    main()
