from __future__ import annotations

import subprocess
from pathlib import Path

from mlfd.utils import short_git_commit


def _run_git(tmp_path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True)


def test_short_git_commit_marks_dirty_worktrees(tmp_path: Path) -> None:
    _run_git(tmp_path, "init")
    _run_git(tmp_path, "config", "user.name", "Codex Test")
    _run_git(tmp_path, "config", "user.email", "codex@example.com")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _run_git(tmp_path, "add", "tracked.txt")
    _run_git(tmp_path, "commit", "-m", "initial")

    clean_commit = short_git_commit(tmp_path)
    assert clean_commit
    assert not clean_commit.endswith("-dirty")

    tracked.write_text("dirty\n", encoding="utf-8")
    dirty_commit = short_git_commit(tmp_path)
    assert dirty_commit.endswith("-dirty")
