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
- `python -m mlfd.autoresearch baseline-check`: run one clean baseline/debug check without candidate keep/discard automation.
- `python -m mlfd.autoresearch apply-candidate`: commit the current code change, run it once, and automatically keep or discard it.
- `python -m mlfd.autoresearch run-current`: execute and log the current code once without candidate keep/discard automation.
- `python -m mlfd.autoresearch run-campaign`: keep running unattended AE-search rounds until `--max-rounds`, `--max-hours`, or a stop file ends the campaign.

## Research Memory

- `results.tsv` is the machine-readable experiment ledger.
- `research/idea_registry.json` is the canonical machine-readable idea memory.
- `research/idea_registry.md` is the generated human-readable view used for onboarding.
- `research/experiments/*.md` keeps append-only experiment narratives.
- `research/state.md` is the short external memory for the next agent turn.
- `research/runs/<run-tag>/index.md` tracks campaign progress.
- `research/final_report.md` is the long-form report that should cite experiments and metrics.

## Current Worktree

- Branch: `codex/autoresearch/20260311-fluid-rtx3070-latent-sweep`
- Role: isolated autonomous sandbox cloned from the nonlinear reference branch.
- Imported reference: `exp-0051` with `primary_score=0.000565`.
- Preferred local workspace for original-style agentic autoresearch before promoting anything back to sibling worktrees.

## Notes

- The runtime assumes `CYLINDER_ALL.mat` is present in the repo root.
- This clone is intended as an isolated autonomous sandbox. Review local results here before importing anything back into the sibling worktrees.
- The primary workflow in this clone is original-style agentic autoresearch: the coding agent edits the nonlinear code, uses `apply-candidate --idea-key ...`, and repeats.
- `run-current` and `baseline-check` are for baseline validation or debugging, not for the main research loop.
- `apply-candidate` commits only `src/` and `tests/` changes, records the run, and restores the previous code automatically when the result is discarded or crashes.
- Idea keys should use `lower_snake_case`. Use a material version suffix such as `_v1`, `_v2` when the formulation itself changes.
- `run-campaign` is a bounded helper for unattended AE-screening, not the main research workflow.
- Some sibling worktrees currently reuse `D:\Projects\ML_FluidDynamics_Project\ML_FluidDynamics\.venv`; if a side worktree has no local `.venv`, run that interpreter with `PYTHONPATH=<worktree>\src`.
- Existing PNG dumps are treated as derived visualizations, not primary training data.
- Current winning configuration: `residual_refine`, `latent_dim=32`, `latent_l1_weight=1e-4`, `rollout_loss_weight=0.15`.
- This sandbox assumes local execution. Remote Azure GPU workflow is out of scope for this workspace.
