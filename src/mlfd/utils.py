from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def run_git(
    root: Path,
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        capture_output=capture_output,
        text=text,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def short_git_commit(root: Path, ignored_paths: tuple[str, ...] = ()) -> str:
    try:
        commit_result = run_git(root, ["rev-parse", "--short", "HEAD"])
        status_args = ["status", "--short", "--untracked-files=all"]
        if ignored_paths:
            status_args.extend(["--", "."])
            status_args.extend(f":(exclude){path}" for path in ignored_paths)
        status_result = run_git(root, status_args)
        commit = commit_result.stdout.strip()
        if status_result.stdout.strip():
            return f"{commit}-dirty"
        return commit
    except (FileNotFoundError, subprocess.CalledProcessError):
        return f"nogit-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def git_branch(root: Path) -> str:
    try:
        result = run_git(root, ["branch", "--show-current"])
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
