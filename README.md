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
- **`run_linear.py`** — local `DMD` baseline runner under the same protocol.
- **`program.md`** — baseline instructions for one agent. Point the agent here
  and let it run.
- **`results.tsv`** — untracked experiment ledger. Do not commit it.
- **`proxy_results.tsv`** — untracked short-budget screening ledger. Do not
  commit it.

## Project structure

```text
prepare.py      - fixed setup/data check script (do not modify)
train.py        - main experiment file (agent modifies this)
run_linear.py   - local DMD baseline runner
program.md      - agent instructions
pyproject.toml  - dependencies
src/mlfd/       - small runtime helper library behind train.py
```

The main metric is **`primary_score`** — lower is better.

As of 2026-03-17, this worktree evaluates new runs under the fixed
`assignment_temporal_holdout_v1` protocol:

- snapshots `[0,120)` train, `[120,135)` val, `[135,151)` test
- transitions `[0,119)` train, `[119,134)` val, `[134,150)` test

This keeps an explicit held-out reconstruction test split of `16/151` snapshots
(about `10.6%`) while preserving a temporal validation window for model
selection. Under this protocol, `t=100` is a seen-region forecast and `t=150`
is a held-out future forecast.

Current active baseline:

- commit `d8c36d7`
- output `output/nonlinear/mar18-control-ae3600-dyn1800-01/`
- `primary_score=0.000255791203`

Current comparison status:

- `results.tsv` is the active full-run scoreboard
- `proxy_results.tsv` is the active short-budget screening ledger
- local `DMD` baseline output: `output/linear/mar17-assignment-dmd-baseline-01/`
- local `DMD` baseline: rank `15`, `primary_score=0.000466643474`
- local `DMD` baseline: `recon_nrmse=0.000163697573`, `nrmse_t100=0.000496426162`, `nrmse_t150=0.000569952222`
- current nonlinear champion `d8c36d7` is lower at `0.000255791203`

Archived pre-reset notes and ledgers live under `archive/`.

## Quick start

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python prepare.py
.venv\Scripts\python train.py --smoke
.venv\Scripts\python run_linear.py --output-tag dmd-baseline
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
- **Two-stage logging.** `proxy_results.tsv` tracks short proxy screening and
  `results.tsv` tracks long confirmation runs.
- **Aggressive promotion gate.** A proxy candidate must beat the proxy score of
  the latest full-validated champion by at least `20%` without regressing
  `t150` before it earns a full confirmation run.
- **Local Windows/RTX target.** This clone is meant to feel closer to
  `autoresearch-win-rtx` than to the richer fluid sandbox.

## Scope note

This folder is deliberately simpler than `ML_FluidDynamics_autonomous`. If you
want the richer idea-registry, generated research memory, recovery helpers, and
log-worktree workflow, use that other clone instead.
