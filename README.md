# autoresearch

Minimal fluid-dynamics adaptation of the original `karpathy/autoresearch` idea,
kept intentionally simple for a local Windows + RTX workflow.

The idea: give an agent a small but real single-GPU research setup and let it
iterate autonomously. It edits `train.py`, runs one experiment, checks whether
`primary_score` improved, keeps or discards the change, and repeats.

## How it works

This workspace is intentionally treated as if only a few files really matter:

- **`CYLINDER_ALL.mat`** — fixed dataset in the repo root. Do not modify.
- **`prepare.py`** — fixed setup/data check script. Do not modify.
- **`train.py`** — the main file the agent edits. It defines the experiment
  defaults and launches one nonlinear run.
- **`program.md`** — baseline instructions for one agent. Point the agent here
  and let it run.
- **`results.tsv`** — untracked experiment ledger. Do not commit it.

## Project structure

```text
prepare.py      - fixed setup/data check script (do not modify)
train.py        - main experiment file (agent modifies this)
program.md      - agent instructions
pyproject.toml  - dependencies
src/mlfd/       - small runtime helper library behind train.py
```

The main metric is **`primary_score`** — lower is better.

## Quick start

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python prepare.py
.venv\Scripts\python train.py --smoke
```

If you already have the shared project environment, you can also reuse:

```powershell
$env:PYTHONPATH='D:\Projects\AutoResearch_ML_FD\ML_FluidDynamics_origstyle\src'
D:\Projects\ML_FluidDynamics_Project\ML_FluidDynamics\.venv\Scripts\python.exe prepare.py
```

## Running the agent

Spin up Codex/Claude in this folder and say something like:

```text
Read program.md and let's start a new fluid autoresearch run.
```

The `program.md` file is the main control surface.

## Design choices

- **Single main file to modify.** Default to `train.py`.
- **Keep/discard by git.** A run that wins advances the branch. A run that
  loses is reset away.
- **Manual, minimal logging.** `results.tsv` is the only required experiment
  ledger in the main loop.
- **Local Windows/RTX target.** This clone is meant to feel closer to
  `autoresearch-win-rtx` than to the richer fluid sandbox.

## Scope note

This folder is deliberately simpler than `ML_FluidDynamics_autonomous`. If you
want the richer idea-registry, generated research memory, recovery helpers, and
log-worktree workflow, use that other clone instead.
