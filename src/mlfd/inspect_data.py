from __future__ import annotations

from .config import ProjectPaths
from .data import load_field_bundle
from .utils import write_json


def main() -> None:
    paths = ProjectPaths()
    paths.ensure_directories()
    summaries = {}
    for field_name in ("UALL", "VALL", "VORTALL"):
        bundle = load_field_bundle(field_name, paths)
        summaries[field_name] = bundle.summary()
    write_json(paths.output_dir / "data_summary.json", summaries)
    markdown = ["# Data Summary", ""]
    for field_name, summary in summaries.items():
        markdown.extend(
            [
                f"## {field_name}",
                "",
                f"- Matrix shape: `{summary['matrix_shape']}`",
                f"- Frames shape: `{summary['frames_shape']}`",
                f"- Range: `{summary['min']:.6f}` to `{summary['max']:.6f}`",
                f"- Mean/std: `{summary['mean']:.6f}` / `{summary['std']:.6f}`",
                "",
            ]
        )
    (paths.output_dir / "data_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Wrote {paths.output_dir / 'data_summary.json'}")


if __name__ == "__main__":
    main()
