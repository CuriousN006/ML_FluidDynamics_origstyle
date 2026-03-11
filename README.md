# ML Fluid Dynamics

Single-GPU fluid dynamics research workspace for the cylinder wake project in
[Final_project_DL2025-5.pdf](./Final_project_DL2025-5.pdf). The codebase is
organized around three layers:

1. A fixed evaluation harness that loads `CYLINDER_ALL.mat`.
2. Mutable research code for linear and nonlinear models.
3. Append-only experiment records in Markdown and TSV.

## Quick Start

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m mlfd.setup
.venv\Scripts\python -m mlfd.inspect_data
.venv\Scripts\python -m mlfd.run_linear
.venv\Scripts\python -m mlfd.run_nonlinear --smoke
```

## Entry Points

- `python -m mlfd.setup`: verify environment, CUDA, data file, and research dirs.
- `python -m mlfd.inspect_data`: write a data summary for `CYLINDER_ALL.mat`.
- `python -m mlfd.run_linear`: generate the linear baseline artifacts.
- `python -m mlfd.run_nonlinear`: train/evaluate the nonlinear baseline.
- `python -m mlfd.autoresearch init-run`: initialize a new research campaign.
- `python -m mlfd.autoresearch run-current`: execute and log the current code as an experiment.

## Research Memory

- `results.tsv` is the machine-readable experiment ledger.
- `research/experiments/*.md` keeps append-only experiment narratives.
- `research/state.md` is the short external memory for the next agent turn.
- `research/runs/<run-tag>/index.md` tracks campaign progress.
- `research/final_report.md` is the long-form report that should cite experiments and metrics.

## Current Worktree

- Branch: `codex/autoresearch/20260311-fluid-rtx3070-latent-sweep`
- Role: current nonlinear reference branch across the sibling worktrees.
- Current best: `exp-0051` with `primary_score=0.000565`.
- Preferred starting point for new AE-floor, decoder, or latent-space searches.

## Notes

- The runtime assumes `CYLINDER_ALL.mat` is present in the repo root.
- Some sibling worktrees currently reuse `D:\Projects\ML_FluidDynamics\.venv`; if a side worktree has no local `.venv`, run that interpreter with `PYTHONPATH=<worktree>\src`.
- Existing PNG dumps are treated as derived visualizations, not primary training data.
- Current winning configuration: `residual_refine`, `latent_dim=32`, `latent_l1_weight=1e-4`, `rollout_loss_weight=0.15`.
