from __future__ import annotations

import argparse

from .config import LinearConfig, ProjectPaths
from .linear import run_linear_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the linear fluid dynamics baseline.")
    parser.add_argument("--output-tag", default="baseline", help="Subdirectory name under output/linear.")
    args = parser.parse_args()
    metrics = run_linear_pipeline(LinearConfig(), ProjectPaths(), output_tag=args.output_tag)
    print(f"Linear metrics saved for {args.output_tag}")
    print(metrics["dmd"]["best_rank"])


if __name__ == "__main__":
    main()
